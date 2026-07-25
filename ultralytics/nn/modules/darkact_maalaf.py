# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Single-image MAA and light-adaptive RGB/thermal feature fusion.

The modules in this file adapt DarkAct's video-oriented MAA/LAF ideas to
paired RGB/thermal images for multi-scale object detection:

* :class:`MAA2D` replaces temporal differences with per-modality local
  contrast. RGB and thermal features are never directly subtracted.
* :class:`LAFMerge2D` learns global channel reliability and local spatial
  reliability, then optionally adds a partial-channel cross-modal value
  interaction. Its gates are zero-initialized so the initial output is exactly
  the original baseline merge ``RGB + IR``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelLayerNorm2d(nn.Module):
    """Normalize the channel vector independently at every spatial position."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class _StaticSaliencyBranch(nn.Module):
    """Generate one spatial saliency mask from modality-internal contrast."""

    def __init__(self, dilation=1, hidden_channels=8):
        super().__init__()
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.mask = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    @staticmethod
    def local_contrast(x):
        """D_m = |F_m - AvgPool_3x3(F_m)|, computed within one modality."""
        return (x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)).abs()

    def forward(self, x):
        contrast = self.local_contrast(x)
        descriptor = torch.cat((contrast.mean(dim=1, keepdim=True), contrast.amax(dim=1, keepdim=True)), dim=1)
        return torch.sigmoid(self.mask(descriptor))


class MAA2D(nn.Module):
    """Single-image, modality-independent static saliency attention.

    Input and output follow the repository's ``m.f == -3`` contract and are
    channel-concatenated tensors shaped ``[B, 2C, H, W]``. Each modality owns a
    separate saliency estimator and is enhanced as ``Y_m = F_m * (1 + beta*S_m)``.
    ``beta`` starts at zero, making this module an exact identity at
    initialization and preserving migrated baseline behavior.
    """

    def __init__(self, channels, dilation=1, hidden_channels=8):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.saliency = nn.ModuleList(
            [_StaticSaliencyBranch(dilation, hidden_channels), _StaticSaliencyBranch(dilation, hidden_channels)]
        )
        self.beta = nn.Parameter(torch.zeros(2, 1, 1, 1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels * 2:
            raise ValueError("MAA2D expects [B, {}, H, W], got {}".format(self.channels * 2, tuple(x.shape)))
        rgb, ir = x.chunk(2, dim=1)
        rgb = rgb * (1.0 + self.beta[0] * self.saliency[0](rgb))
        ir = ir * (1.0 + self.beta[1] * self.saliency[1](ir))
        return torch.cat((rgb, ir), dim=1)


class _PartialCrossModalValue(nn.Module):
    """DarkAct-inspired cross-modal value interaction on C/ratio channels."""

    def __init__(self, channels, num_heads, partial_ratio=4, dilation=1):
        super().__init__()
        if partial_ratio <= 0 or channels % partial_ratio != 0:
            raise ValueError("channels must be divisible by a positive partial_ratio")
        active_channels = channels // partial_ratio
        if num_heads <= 0 or active_channels % num_heads != 0:
            raise ValueError("active channels must be divisible by a positive num_heads")

        self.active_channels = active_channels
        self.num_heads = num_heads
        self.norm = nn.ModuleList([_ChannelLayerNorm2d(channels), _ChannelLayerNorm2d(channels)])
        self.qk = nn.ModuleList()
        self.value = nn.ModuleList()
        for _ in range(2):
            self.qk.append(
                nn.Sequential(
                    nn.Conv2d(channels, active_channels * 2, kernel_size=1, bias=False),
                    nn.Conv2d(
                        active_channels * 2,
                        active_channels * 2,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=active_channels * 2,
                        bias=False,
                    ),
                )
            )
            self.value.append(
                nn.Sequential(
                    nn.Conv2d(channels, active_channels, kernel_size=1, bias=False),
                    nn.Conv2d(
                        active_channels,
                        active_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=active_channels,
                        bias=False,
                    ),
                )
            )
        self.temperature = nn.Parameter(torch.ones(2, num_heads, 1, 1))
        self.project_out = nn.Conv2d(active_channels * 2, channels, kernel_size=1, bias=False)

    def _reshape_heads(self, x):
        b, c, h, w = x.shape
        return x.reshape(b, self.num_heads, c // self.num_heads, h * w)

    def _attend(self, query, key, opposite_value, direction):
        query = F.normalize(self._reshape_heads(query), dim=-1)
        key = F.normalize(self._reshape_heads(key), dim=-1)
        value = self._reshape_heads(opposite_value)
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.temperature[direction]
        attention = torch.softmax(attention, dim=-1)
        return torch.matmul(attention, value)

    def forward(self, rgb, ir):
        q_rgb, k_rgb = self.qk[0](self.norm[0](rgb)).chunk(2, dim=1)
        q_ir, k_ir = self.qk[1](self.norm[1](ir)).chunk(2, dim=1)
        v_rgb = self.value[0](rgb)
        v_ir = self.value[1](ir)

        rgb_from_ir = self._attend(q_rgb, k_rgb, v_ir, direction=0)
        ir_from_rgb = self._attend(q_ir, k_ir, v_rgb, direction=1)
        b, _, h, w = rgb.shape
        rgb_from_ir = rgb_from_ir.reshape(b, self.active_channels, h, w)
        ir_from_rgb = ir_from_rgb.reshape(b, self.active_channels, h, w)
        return self.project_out(torch.cat((rgb_from_ir, ir_from_rgb), dim=1))


class LAFMerge2D(nn.Module):
    """Light-adaptive RGB/thermal merge for one detection pyramid scale.

    The module combines:

    1. global channel reliability from average/max pooled RGB and IR features;
    2. local spatial reliability from modality-internal local contrast;
    3. optional C/``partial_ratio`` cross-modal value attention with dilated
       depth-wise local projections.

    It accepts ``[rgb, ir]`` and returns one fused ``[B, C, H, W]`` tensor. The
    global/local gate heads and cross residual scale are zero-initialized. Thus
    the initial softmax weights are both one after the factor of two and the
    exact initial output is ``rgb + ir``.
    """

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8):
        super().__init__()
        if channels <= 0 or gate_reduction <= 0:
            raise ValueError("channels and gate_reduction must be positive")
        self.channels = channels
        hidden_channels = max(channels // gate_reduction, 8)

        self.global_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * 2, kernel_size=1, bias=True),
        )
        local_hidden = max(min(hidden_channels, 32), 8)
        self.local_gate = nn.Sequential(
            nn.Conv2d(4, local_hidden, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(local_hidden, 2, kernel_size=1, bias=True),
        )

        self.cross = (
            _PartialCrossModalValue(channels, num_heads, partial_ratio, dilation) if num_heads > 0 else None
        )
        self.cross_scale = nn.Parameter(torch.zeros(1, 1, 1, 1))

        # Equal modal weights and no cross residual reproduce baseline ADD.
        nn.init.zeros_(self.global_gate[-1].weight)
        nn.init.zeros_(self.global_gate[-1].bias)
        nn.init.zeros_(self.local_gate[-1].weight)
        nn.init.zeros_(self.local_gate[-1].bias)

    @staticmethod
    def _contrast_descriptor(x):
        contrast = _StaticSaliencyBranch.local_contrast(x)
        return torch.cat((contrast.mean(dim=1, keepdim=True), contrast.amax(dim=1, keepdim=True)), dim=1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("LAFMerge2D expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(
                "LAFMerge2D expects two [B, {}, H, W] tensors, got {} and {}".format(
                    self.channels, tuple(rgb.shape), tuple(ir.shape)
                )
            )

        global_descriptor = torch.cat(
            (
                F.adaptive_avg_pool2d(rgb, 1),
                F.adaptive_max_pool2d(rgb, 1),
                F.adaptive_avg_pool2d(ir, 1),
                F.adaptive_max_pool2d(ir, 1),
            ),
            dim=1,
        )
        b, _, h, w = rgb.shape
        global_logits = self.global_gate(global_descriptor).reshape(b, 2, self.channels, 1, 1)

        local_descriptor = torch.cat((self._contrast_descriptor(rgb), self._contrast_descriptor(ir)), dim=1)
        local_logits = self.local_gate(local_descriptor).reshape(b, 2, 1, h, w)

        modal_weights = torch.softmax(global_logits + local_logits, dim=1) * 2.0
        fused = modal_weights[:, 0] * rgb + modal_weights[:, 1] * ir
        if self.cross is not None:
            fused = fused + self.cross_scale * self.cross(rgb, ir)
        return fused


__all__ = ("MAA2D", "LAFMerge2D")
