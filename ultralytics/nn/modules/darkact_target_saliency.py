# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Target-saliency DarkAct context and paper LAF for aligned RGB/thermal features.

This module family is additive: the existing StaticMAA2D and
PaperLAFMergeFeedback2D implementations are not changed.  Each modality is
converted from a stage feature F to one spatial saliency logit S.  The LAF
query then uses S * Norm(F), avoiding any channel-wise semantic assumption
between an earlier MAA feature Y and a later stage feature F.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darkact_maalaf_v3 import PaperLAFMergeFeedback2D


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


class _StaticTargetSaliencyBranch(nn.Module):
    """Static multi-scale context that predicts one target-saliency logit."""

    def __init__(self, channels, partial_ratio=4, dilation=1, contrast_kernels=(3, 5), query_kernel=3):
        super().__init__()
        if partial_ratio <= 0 or channels % partial_ratio != 0:
            raise ValueError("channels must be divisible by a positive partial_ratio")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        if not contrast_kernels or any(k <= 0 or k % 2 == 0 for k in contrast_kernels):
            raise ValueError("contrast_kernels must contain positive odd integers")
        if query_kernel <= 0 or query_kernel % 2 == 0:
            raise ValueError("query_kernel must be a positive odd integer")

        active_channels = channels // partial_ratio
        self.active_channels = active_channels
        self.contrast_kernels = tuple(int(k) for k in contrast_kernels)
        self.query_kernel = int(query_kernel)
        self.norm = _ChannelLayerNorm2d(channels)

        # E^S: multi-scale local contrast on the exact stage feature F.
        self.saliency = nn.Sequential(
            nn.Conv2d(channels * len(self.contrast_kernels), active_channels, 1, bias=False),
            nn.Conv2d(
                active_channels,
                active_channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=active_channels,
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(active_channels, active_channels, 1, bias=False),
        )

        # E^Q: local average/max context, retained at the same HxW resolution.
        self.query = nn.Sequential(
            nn.Conv2d(channels * 2, active_channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(active_channels, active_channels, 1, bias=False),
        )
        self.query_mapping = nn.Conv2d(active_channels, active_channels, 1, bias=False)
        self.saliency_head = nn.Conv2d(active_channels, 1, 1, bias=True)

        # Start from a weak foreground prior rather than a saturated mask.
        nn.init.normal_(self.saliency_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.saliency_head.bias, math.log(0.1 / 0.9))

    def forward(self, x):
        normalized = self.norm(x)
        differences = [
            (normalized - F.avg_pool2d(normalized, k, stride=1, padding=k // 2)).abs()
            for k in self.contrast_kernels
        ]
        saliency = self.saliency(torch.cat(differences, dim=1))

        padding = self.query_kernel // 2
        descriptor = torch.cat(
            (
                F.avg_pool2d(normalized, self.query_kernel, stride=1, padding=padding),
                F.max_pool2d(normalized, self.query_kernel, stride=1, padding=padding),
            ),
            dim=1,
        )
        query = self.query_mapping(self.query(descriptor))

        b, _, h, w = saliency.shape
        attention = torch.matmul(query.flatten(2), saliency.flatten(2).transpose(-2, -1))
        attention = torch.softmax(attention / math.sqrt(self.active_channels), dim=-1)
        attended = torch.matmul(attention, saliency.flatten(2)).reshape(b, self.active_channels, h, w)
        return self.saliency_head(attended)


class StaticMAAContext2D(nn.Module):
    """Predict one target-saliency map per modality from aligned stage features.

    Input is ``[F_rgb, F_ir]`` and output is a tensor ``[B, 2, H, W]`` that
    packs the raw one-channel logits ``(S_rgb, S_ir)``.  The logits remain raw
    so the auxiliary OBB-mask criterion can use BCEWithLogitsLoss.
    """

    def __init__(self, channels, partial_ratio=4, dilation=1, contrast_kernels=(3, 5), query_kernel=3):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.branches = nn.ModuleList(
            [
                _StaticTargetSaliencyBranch(
                    channels, partial_ratio, dilation, contrast_kernels, query_kernel
                ),
                _StaticTargetSaliencyBranch(
                    channels, partial_ratio, dilation, contrast_kernels, query_kernel
                ),
            ]
        )
        # Enabled only by TargetSaliencyOBBModel after stride initialization.
        self.capture_saliency = False
        self._last_saliency_logits = None

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("StaticMAAContext2D expects [F_rgb, F_ir]")
        rgb, ir = x
        expected = (rgb.shape[0], self.channels, rgb.shape[2], rgb.shape[3])
        for name, tensor in (("F_rgb", rgb), ("F_ir", ir)):
            if tensor.ndim != 4 or tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        logits = torch.cat((self.branches[0](rgb), self.branches[1](ir)), dim=1)
        if self.capture_saliency:
            self._last_saliency_logits = logits
        return logits

    def pop_saliency_logits(self):
        """Return and clear the current forward's logits to avoid checkpoint graph retention."""
        logits = self._last_saliency_logits
        self._last_saliency_logits = None
        return logits


class TargetSaliencyPaperLAFMergeFeedback2D(PaperLAFMergeFeedback2D):
    """Paper LAF whose shared query is phi([S * Norm(F)]).

    Inputs are ``[saliency_logits, F_rgb, F_ir]``.  ``saliency_logits`` packs
    two one-channel maps, which are sigmoid-normalized and broadcast over the
    full C-channel stage features.  Keys, shared value, dynamic attention,
    zero-initialized output residual, and feedback behavior are inherited from
    the independent V3 implementation.
    """

    def __init__(self, channels, dilation=1, pool_kernel=3):
        super().__init__(channels, dilation, pool_kernel)
        self.feature_norm = nn.ModuleList(
            [_ChannelLayerNorm2d(channels), _ChannelLayerNorm2d(channels)]
        )

    def _validate_target_inputs(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise TypeError(
                "TargetSaliencyPaperLAFMergeFeedback2D expects [saliency_logits, F_rgb, F_ir]"
            )
        saliency_logits, rgb, ir = x
        expected_feature = (rgb.shape[0], self.channels, rgb.shape[2], rgb.shape[3])
        if rgb.ndim != 4 or tuple(rgb.shape) != expected_feature:
            raise ValueError(f"F_rgb must have shape {expected_feature}, got {tuple(rgb.shape)}")
        if ir.ndim != 4 or tuple(ir.shape) != expected_feature:
            raise ValueError(f"F_ir must have shape {expected_feature}, got {tuple(ir.shape)}")
        expected_saliency = (rgb.shape[0], 2, rgb.shape[2], rgb.shape[3])
        if saliency_logits.ndim != 4 or tuple(saliency_logits.shape) != expected_saliency:
            raise ValueError(
                f"saliency_logits must have shape {expected_saliency}, got {tuple(saliency_logits.shape)}"
            )
        return saliency_logits, rgb, ir

    def forward(self, x):
        saliency_logits, rgb, ir = self._validate_target_inputs(x)
        saliency_rgb, saliency_ir = saliency_logits.sigmoid().chunk(2, dim=1)
        normalized_rgb = self.feature_norm[0](rgb)
        normalized_ir = self.feature_norm[1](ir)

        # F_Q = phi([S_rgb * Norm(F_rgb); S_ir * Norm(F_ir)]).
        shared_query = self.query_mapping(
            torch.cat((saliency_rgb * normalized_rgb, saliency_ir * normalized_ir), dim=1)
        )
        rgb_key = self.key_mapping[0](rgb)
        ir_key = self.key_mapping[1](ir)
        shared_value = self._shared_value(rgb, ir)

        dynamic_rgb = self._channel_attention(shared_query, rgb_key, shared_value)
        dynamic_ir = self._channel_attention(shared_query, ir_key, shared_value)
        fused_laf = self.output_mlp[0](dynamic_rgb) + self.output_mlp[1](dynamic_ir)
        return rgb + fused_laf, ir + fused_laf, rgb + ir + fused_laf


__all__ = ("StaticMAAContext2D", "TargetSaliencyPaperLAFMergeFeedback2D")
