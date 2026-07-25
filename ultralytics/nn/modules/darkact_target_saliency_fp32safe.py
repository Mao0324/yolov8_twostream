# Ultralytics YOLO 🚀, AGPL-3.0 license
"""FP32-safe target-saliency context without changing its attention equation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darkact_target_saliency import StaticMAAContext2D, _StaticTargetSaliencyBranch


class _StaticTargetSaliencyBranchFP32Safe(_StaticTargetSaliencyBranch):
    """Evaluate only the attention reductions and Softmax in FP32 under AMP."""

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
        output_dtype = saliency.dtype
        # The P3 reduction spans N=80*80 positions. FP16 matmul can overflow
        # before the numerically stable Softmax is reached, so keep the exact
        # sqrt(active_channels) equation but evaluate this block in FP32.
        with torch.autocast(device_type=saliency.device.type, enabled=False):
            query_fp32 = query.float().flatten(2)
            saliency_fp32 = saliency.float().flatten(2)
            attention = torch.matmul(query_fp32, saliency_fp32.transpose(-2, -1))
            attention = torch.softmax(attention / math.sqrt(self.active_channels), dim=-1)
            attended = torch.matmul(attention, saliency_fp32).reshape(
                b, self.active_channels, h, w
            )
        return self.saliency_head(attended.to(dtype=output_dtype))


class StaticMAAContext2DFP32Safe(StaticMAAContext2D):
    """Drop-in StaticMAAContext2D variant with FP32-safe internal attention."""

    def __init__(
        self,
        channels,
        partial_ratio=4,
        dilation=1,
        contrast_kernels=(3, 5),
        query_kernel=3,
    ):
        # Construct directly instead of calling the parent initializer so the
        # additive variant has exactly one initialization pass per branch.
        nn.Module.__init__(self)
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.branches = nn.ModuleList(
            [
                _StaticTargetSaliencyBranchFP32Safe(
                    channels, partial_ratio, dilation, contrast_kernels, query_kernel
                ),
                _StaticTargetSaliencyBranchFP32Safe(
                    channels, partial_ratio, dilation, contrast_kernels, query_kernel
                ),
            ]
        )
        self.capture_saliency = False
        self._last_saliency_logits = None


__all__ = ("StaticMAAContext2DFP32Safe",)
