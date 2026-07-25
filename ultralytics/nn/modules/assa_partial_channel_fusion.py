# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Partial-channel static-depthwise ASSA fusion for RGB/IR features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assa_fusion import ChannelLayerNorm2d
from .assa_fusion_static_noffn import StaticLocalQKVProjection


class PartialChannelASSAFusion(nn.Module):
    """Bidirectional ASSA on only C/partial_ratio channels, without an FFN.

    The first ``C / partial_ratio`` channels from each modality participate in
    the static-DW cross-modal attention. The original full C-channel tensors,
    including the bypassed channels, remain on the residual path. Attention
    outputs are projected from the reduced width back to C before residual
    addition. Input and output both use the ``[RGB, IR]`` concatenated contract
    shaped ``[B, 2C, H, W]``.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        static_kernel=3,
        residual_scale=0.1,
    ):
        super().__init__()
        if channels <= 0 or num_heads <= 0 or partial_ratio <= 0:
            raise ValueError("channels, num_heads and partial_ratio must be positive")
        if channels % partial_ratio != 0:
            raise ValueError("channels ({}) must be divisible by partial_ratio ({})".format(channels, partial_ratio))

        active_channels = channels // partial_ratio
        if active_channels % num_heads != 0:
            raise ValueError(
                "active channels ({}) must be divisible by num_heads ({})".format(active_channels, num_heads)
            )

        self.channels = channels
        self.active_channels = active_channels
        self.bypass_channels = channels - active_channels
        self.num_heads = num_heads
        self.partial_ratio = partial_ratio
        self.static_kernel = static_kernel

        self.norm_qkv = nn.ModuleList(
            [ChannelLayerNorm2d(active_channels), ChannelLayerNorm2d(active_channels)]
        )
        self.qkv = nn.ModuleList(
            [
                StaticLocalQKVProjection(active_channels, static_kernel),
                StaticLocalQKVProjection(active_channels, static_kernel),
            ]
        )
        self.temperature = nn.Parameter(torch.ones(2, num_heads, 1, 1))
        self.project_out = nn.ModuleList(
            [nn.Conv2d(active_channels, channels, kernel_size=1, bias=False) for _ in range(2)]
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
                "PartialChannelASSAFusion expects [B, {}, H, W], got {}".format(
                    self.channels * 2, tuple(x.shape)
                )
            )

        rgb, ir = x.chunk(2, dim=1)
        # Only the active C/ratio slice enters attention. The remaining channels
        # bypass every attention operation through the full-width residual path.
        rgb_active = rgb[:, : self.active_channels]
        ir_active = ir[:, : self.active_channels]

        q_rgb, kv_rgb = self.qkv[0](self.norm_qkv[0](rgb_active))
        q_ir, kv_ir = self.qkv[1](self.norm_qkv[1](ir_active))
        rgb_from_ir = self._cross_attention(q_rgb, kv_ir, direction=0)
        ir_from_rgb = self._cross_attention(q_ir, kv_rgb, direction=1)

        b, _, h, w = rgb_active.shape
        rgb_from_ir = rgb_from_ir.reshape(b, self.active_channels, h, w)
        ir_from_rgb = ir_from_rgb.reshape(b, self.active_channels, h, w)
        rgb = rgb + self.attn_scale[0] * self.project_out[0](rgb_from_ir)
        ir = ir + self.attn_scale[1] * self.project_out[1](ir_from_rgb)
        return torch.cat((rgb, ir), dim=1)


__all__ = ("PartialChannelASSAFusion",)
