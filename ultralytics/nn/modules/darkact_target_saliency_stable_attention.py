# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Stable channel-attention variants for target-saliency ablations."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darkact_target_saliency import StaticMAAContext2D, _StaticTargetSaliencyBranch


class _StaticTargetSaliencyBranchStableBase(_StaticTargetSaliencyBranch):
    """Shared feature construction; subclasses change only channel-attention logits."""

    def _attention_logits(self, query_fp32, saliency_fp32):
        raise NotImplementedError

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
        with torch.autocast(device_type=saliency.device.type, enabled=False):
            query_fp32 = query.float().flatten(2)
            saliency_fp32 = saliency.float().flatten(2)
            logits = self._attention_logits(query_fp32, saliency_fp32)
            attention = torch.softmax(logits, dim=-1)
            # L2 normalization, when enabled, applies only to Q/K similarity;
            # V remains the original saliency feature to preserve its magnitude.
            attended = torch.matmul(attention, saliency_fp32).reshape(
                b, self.active_channels, h, w
            )
        return self.saliency_head(attended.to(dtype=output_dtype))


class _StaticTargetSaliencyBranchSqrtHW(_StaticTargetSaliencyBranchStableBase):
    """Scale the C_a x C_a channel-attention logits by sqrt(H*W)."""

    def _attention_logits(self, query_fp32, saliency_fp32):
        spatial_tokens = query_fp32.shape[-1]
        return torch.matmul(query_fp32, saliency_fp32.transpose(-2, -1)) / math.sqrt(
            spatial_tokens
        )


class _StaticTargetSaliencyBranchL2Temp(_StaticTargetSaliencyBranchStableBase):
    """Cosine channel similarity with one learnable temperature per modality/stage."""

    def __init__(
        self,
        channels,
        partial_ratio=4,
        dilation=1,
        contrast_kernels=(3, 5),
        query_kernel=3,
        temperature_init=0.2,
    ):
        super().__init__(
            channels, partial_ratio, dilation, contrast_kernels, query_kernel
        )
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive")
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))

    def _attention_logits(self, query_fp32, saliency_fp32):
        query_norm = F.normalize(query_fp32, p=2.0, dim=-1, eps=1e-6)
        saliency_norm = F.normalize(saliency_fp32, p=2.0, dim=-1, eps=1e-6)
        temperature = self.temperature.float().clamp(min=0.05, max=2.0)
        return torch.matmul(query_norm, saliency_norm.transpose(-2, -1)) / temperature


class _StaticMAAContext2DStableBase(StaticMAAContext2D):
    """Build two modality branches while retaining the parent's trainer contract."""

    branch_cls = None

    def __init__(
        self,
        channels,
        partial_ratio=4,
        dilation=1,
        contrast_kernels=(3, 5),
        query_kernel=3,
        *branch_args,
    ):
        nn.Module.__init__(self)
        if channels <= 0:
            raise ValueError("channels must be positive")
        if self.branch_cls is None:
            raise TypeError("branch_cls must be defined by a concrete context variant")
        self.channels = channels
        self.branches = nn.ModuleList(
            [
                self.branch_cls(
                    channels,
                    partial_ratio,
                    dilation,
                    contrast_kernels,
                    query_kernel,
                    *branch_args,
                ),
                self.branch_cls(
                    channels,
                    partial_ratio,
                    dilation,
                    contrast_kernels,
                    query_kernel,
                    *branch_args,
                ),
            ]
        )
        self.capture_saliency = False
        self._last_saliency_logits = None


class StaticMAAContext2DSqrtHW(_StaticMAAContext2DStableBase):
    """Target-saliency context using FP32 channel attention scaled by sqrt(HW)."""

    branch_cls = _StaticTargetSaliencyBranchSqrtHW


class StaticMAAContext2DL2Temp(_StaticMAAContext2DStableBase):
    """Target-saliency context using FP32 cosine channel attention and temperature."""

    branch_cls = _StaticTargetSaliencyBranchL2Temp


__all__ = ("StaticMAAContext2DSqrtHW", "StaticMAAContext2DL2Temp")
