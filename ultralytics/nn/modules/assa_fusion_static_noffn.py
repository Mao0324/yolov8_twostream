# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Lightweight static-depthwise ASSA fusion without an FFN."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assa_fusion import ChannelLayerNorm2d


class StaticLocalQKVProjection(nn.Module):
    """Generate Q and shared K/V features with a static depth-wise 3x3 convolution."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("static depth-wise kernel size must be odd")
        self.project_in = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=channels * 2,
            bias=False,
        )

    def forward(self, x):
        return self.depthwise(self.project_in(x)).chunk(2, dim=1)


class ASSAFusionStaticNoFFN(nn.Module):
    """Bidirectional static-DW ASSA interaction without the gated FFN.

    The tensor contract matches :class:`ASSAFusion`: input and output are
    channel-concatenated RGB/IR tensors shaped ``[B, 2C, H, W]``. This class is
    intentionally separate so existing YAML files that use ``ASSAFusion`` keep
    the original dynamic-kernel plus FFN architecture unchanged.
    """

    def __init__(self, channels, num_heads=4, static_kernel=3, residual_scale=0.1):
        super().__init__()
        if channels <= 0 or num_heads <= 0:
            raise ValueError("channels and num_heads must be positive")
        if channels % num_heads != 0:
            raise ValueError("channels ({}) must be divisible by num_heads ({})".format(channels, num_heads))

        self.channels = channels
        self.num_heads = num_heads
        self.static_kernel = static_kernel
        self.norm_qkv = nn.ModuleList([ChannelLayerNorm2d(channels), ChannelLayerNorm2d(channels)])
        self.qkv = nn.ModuleList(
            [StaticLocalQKVProjection(channels, static_kernel), StaticLocalQKVProjection(channels, static_kernel)]
        )
        self.temperature = nn.Parameter(torch.ones(2, num_heads, 1, 1))
        self.project_out = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=1, bias=False) for _ in range(2)]
        )
        self.attn_scale = nn.Parameter(torch.full((2, 1, 1, 1), float(residual_scale)))

    def _reshape_heads(self, x):
        b, c, h, w = x.shape
        return x.reshape(b, self.num_heads, c // self.num_heads, h * w)

    def _cross_attention(self, query, key_value, direction):
        q = F.normalize(self._reshape_heads(query), dim=-1)
        k = F.normalize(self._reshape_heads(key_value), dim=-1)
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.temperature[direction]
        attention = F.relu(attention)
        return torch.matmul(attention, self._reshape_heads(key_value))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels * 2:
            raise ValueError(
                "ASSAFusionStaticNoFFN expects [B, {}, H, W], got {}".format(
                    self.channels * 2, tuple(x.shape)
                )
            )

        rgb, ir = x.chunk(2, dim=1)
        q_rgb, kv_rgb = self.qkv[0](self.norm_qkv[0](rgb))
        q_ir, kv_ir = self.qkv[1](self.norm_qkv[1](ir))

        rgb_from_ir = self._cross_attention(q_rgb, kv_ir, direction=0)
        ir_from_rgb = self._cross_attention(q_ir, kv_rgb, direction=1)
        b, _, h, w = rgb.shape
        rgb_from_ir = rgb_from_ir.reshape(b, self.channels, h, w)
        ir_from_rgb = ir_from_rgb.reshape(b, self.channels, h, w)

        rgb = rgb + self.attn_scale[0] * self.project_out[0](rgb_from_ir)
        ir = ir + self.attn_scale[1] * self.project_out[1](ir_from_rgb)
        return torch.cat((rgb, ir), dim=1)


__all__ = ("ASSAFusionStaticNoFFN",)
