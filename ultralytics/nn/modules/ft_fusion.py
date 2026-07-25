# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Focused Taylor cross-modal fusion for paired RGB/thermal features."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelLayerNorm2d(nn.Module):
    """Normalize each spatial position across channels."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class _ReducedLocalQKV(nn.Module):
    """Learn a reduced Q/K/V basis and inject local context with static DWConv."""

    def __init__(self, channels, active_channels):
        super().__init__()
        projected_channels = active_channels * 3
        self.project_in = nn.Conv2d(channels, projected_channels, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(
            projected_channels,
            projected_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=projected_channels,
            bias=False,
        )

    def forward(self, x):
        return self.depthwise(self.project_in(x)).chunk(3, dim=1)


class FTCrossMerge(nn.Module):
    """Bidirectional partial-channel Focused Taylor fusion.

    The module accepts ``[rgb, ir]`` with two tensors shaped ``[B, C, H, W]``
    and returns one fused tensor shaped ``[B, C, H, W]``. Q/K/V are learned
    from all C input channels but projected to C/``reduction`` channels, so the
    attention does not rely on an arbitrary leading-channel slice.

    For each direction, the first-order Taylor numerator and denominator are
    evaluated as Q(K^T V), without materializing an N x N attention matrix.
    ReLU, power focusing, L2 normalization, and the V-sum constant term follow
    FT-Attention. Accumulation is performed in float32 to keep the square and
    denominator stable under AMP.

    ``residual_scale=0`` makes the initial output exactly ``rgb + ir``, matching
    the original baseline ADD merge while allowing the scalar gate to learn.
    """

    def __init__(
        self,
        channels,
        num_heads=2,
        reduction=4,
        focusing_factor=2,
        residual_scale=0.0,
        eps=1e-6,
    ):
        super().__init__()
        if channels <= 0 or num_heads <= 0 or reduction <= 0:
            raise ValueError("channels, num_heads and reduction must be positive")
        if channels % reduction != 0:
            raise ValueError("channels ({}) must be divisible by reduction ({})".format(channels, reduction))
        if focusing_factor <= 0:
            raise ValueError("focusing_factor must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        active_channels = channels // reduction
        if active_channels % num_heads != 0:
            raise ValueError(
                "active channels ({}) must be divisible by num_heads ({})".format(active_channels, num_heads)
            )

        self.channels = channels
        self.active_channels = active_channels
        self.num_heads = num_heads
        self.reduction = reduction
        self.focusing_factor = focusing_factor
        self.eps = eps
        self.norm = nn.ModuleList([_ChannelLayerNorm2d(channels), _ChannelLayerNorm2d(channels)])
        self.qkv = nn.ModuleList(
            [_ReducedLocalQKV(channels, active_channels), _ReducedLocalQKV(channels, active_channels)]
        )
        self.project_out = nn.Conv2d(active_channels * 2, channels, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale)))

    def _reshape_heads(self, x):
        b, c, h, w = x.shape
        return x.reshape(b, self.num_heads, c // self.num_heads, h * w)

    def _focused(self, x):
        # Square/focusing in float32 prevents fp16 overflow before normalization.
        x = F.relu(x.float()).add(self.eps).pow(self.focusing_factor)
        return F.normalize(x, p=2.0, dim=2, eps=self.eps)

    def _cross_attention(self, query, key, value):
        output_dtype = value.dtype
        q = self._focused(self._reshape_heads(query))
        k = self._focused(self._reshape_heads(key))
        v = self._reshape_heads(value).float()

        head_dim = q.shape[2]
        token_count = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)

        # K^T V in the paper, represented as [d, N] @ [N, d].
        kv = torch.matmul(k, v.transpose(-2, -1))
        numerator = torch.einsum("bhdn,bhde->bhen", q, kv) * scale
        numerator = numerator + v.sum(dim=-1, keepdim=True)

        denominator = torch.einsum("bhdn,bhd->bhn", q, k.sum(dim=-1)) * scale
        denominator = denominator.add(float(token_count)).clamp_min(self.eps)
        output = numerator / denominator.unsqueeze(2)
        return output.to(dtype=output_dtype)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("FTCrossMerge expects [rgb, ir]")
        rgb, ir = x
        if rgb.ndim != 4 or ir.ndim != 4:
            raise ValueError("FTCrossMerge expects two 4D feature tensors")
        if rgb.shape != ir.shape or rgb.shape[1] != self.channels:
            raise ValueError(
                "FTCrossMerge expects two [B, {}, H, W] tensors, got {} and {}".format(
                    self.channels, tuple(rgb.shape), tuple(ir.shape)
                )
            )

        q_rgb, k_rgb, v_rgb = self.qkv[0](self.norm[0](rgb))
        q_ir, k_ir, v_ir = self.qkv[1](self.norm[1](ir))
        rgb_from_ir = self._cross_attention(q_rgb, k_ir, v_ir)
        ir_from_rgb = self._cross_attention(q_ir, k_rgb, v_rgb)

        b, _, h, w = rgb.shape
        rgb_from_ir = rgb_from_ir.reshape(b, self.active_channels, h, w)
        ir_from_rgb = ir_from_rgb.reshape(b, self.active_channels, h, w)
        delta = self.project_out(torch.cat((rgb_from_ir, ir_from_rgb), dim=1))
        return rgb + ir + self.alpha * delta


__all__ = ("FTCrossMerge",)
