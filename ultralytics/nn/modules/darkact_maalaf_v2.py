# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Paper-guided static MAA and backbone-feedback LAF for paired RGB/thermal images.

This V2 module family is intentionally separate from ``MAA2D`` and
``LAFMerge2D``. It keeps the original classes and their YAML semantics intact.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darkact_maalaf import LAFMerge2D


class _ChannelLayerNorm2dV2(nn.Module):
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


class _StaticPaperMAABranch(nn.Module):
    """Replace temporal saliency with static multi-scale structure saliency."""

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
        self.channels = channels
        self.active_channels = active_channels
        self.contrast_kernels = tuple(int(k) for k in contrast_kernels)
        self.query_kernel = int(query_kernel)
        self.norm = _ChannelLayerNorm2dV2(channels)

        # Static saliency E^S: keep C/ratio channels instead of collapsing to
        # the single spatial mask used by the V1 MAA2D implementation.
        self.saliency = nn.Sequential(
            nn.Conv2d(channels * len(self.contrast_kernels), active_channels, kernel_size=1, bias=False),
            nn.Conv2d(
                active_channels,
                active_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=active_channels,
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(active_channels, active_channels, kernel_size=1, bias=False),
        )

        # Spatial-tolerant query E^Q: local AP/MP retain HxW, the 1x1 MLP
        # reduces 2C to C/ratio, and query_mapping implements W_M^A.
        self.query_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, active_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(active_channels, active_channels, kernel_size=1, bias=False),
        )
        self.query_mapping = nn.Conv2d(active_channels, active_channels, kernel_size=1, bias=False)
        self.project_out = nn.Conv2d(active_channels, channels, kernel_size=1, bias=False)

    def _static_saliency(self, x):
        differences = [
            (x - F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)).abs()
            for k in self.contrast_kernels
        ]
        return self.saliency(torch.cat(differences, dim=1))

    def _spatial_tolerant_query(self, x):
        padding = self.query_kernel // 2
        descriptor = torch.cat(
            (
                F.avg_pool2d(x, kernel_size=self.query_kernel, stride=1, padding=padding),
                F.max_pool2d(x, kernel_size=self.query_kernel, stride=1, padding=padding),
            ),
            dim=1,
        )
        return self.query_mapping(self.query_mlp(descriptor))

    def forward_logits(self, x):
        """Return the unconstrained gate logits before sigmoid/tanh mapping."""
        normalized = self.norm(x)
        saliency = self._static_saliency(normalized)
        query = self._spatial_tolerant_query(x)

        b, _, h, w = saliency.shape
        query = query.flatten(2)
        saliency_flat = saliency.flatten(2)
        attention = torch.matmul(query, saliency_flat.transpose(-2, -1)) / math.sqrt(self.active_channels)
        attention = torch.softmax(attention, dim=-1)
        attended = torch.matmul(attention, saliency_flat).reshape(b, self.active_channels, h, w)
        return self.project_out(attended)

    def forward(self, x):
        # A sigmoid gate converts the paper-style channel-attention result into
        # a strictly non-negative enhancement mask for the selected 1A design.
        return torch.sigmoid(self.forward_logits(x))


class StaticMAA2D(nn.Module):
    """Static paper-style MAA with non-negative modality-specific enhancement.

    Input/output use the existing ``m.f == -3`` pair contract and have shape
    ``[B, 2C, H, W]``. For each modality, multi-scale local differences form
    E^S, AP/MP + MLP + W_M^A form E^Q, and channel attention produces a gate.
    ``beta = softplus(raw_beta)`` is always positive, so the final multiplier
    ``1 + beta * gate`` cannot suppress either modality by a factor below one.
    """

    def __init__(
        self,
        channels,
        partial_ratio=4,
        dilation=1,
        beta_init=0.01,
        contrast_kernels=(3, 5),
        query_kernel=3,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if beta_init <= 0:
            raise ValueError("beta_init must be positive for softplus parameterization")
        self.channels = channels
        self.branches = nn.ModuleList(
            [
                _StaticPaperMAABranch(channels, partial_ratio, dilation, contrast_kernels, query_kernel),
                _StaticPaperMAABranch(channels, partial_ratio, dilation, contrast_kernels, query_kernel),
            ]
        )
        raw_beta = math.log(math.expm1(float(beta_init)))
        self.raw_beta = nn.Parameter(torch.full((2, 1, 1, 1), raw_beta))

    @property
    def beta(self):
        return F.softplus(self.raw_beta)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels * 2:
            raise ValueError("StaticMAA2D expects [B, {}, H, W], got {}".format(self.channels * 2, tuple(x.shape)))
        rgb, ir = x.chunk(2, dim=1)
        outputs = []
        for index, feature in enumerate((rgb, ir)):
            gate = self.branches[index](feature)
            outputs.append(feature + self.beta[index] * feature * gate)
        return torch.cat(outputs, dim=1)


class ZeroCenteredStaticMAA2D(nn.Module):
    """Static MAA with a bounded, zero-centered bidirectional gate."""

    def __init__(
        self,
        channels,
        partial_ratio=4,
        dilation=1,
        beta_init=0.1,
        beta_max=0.5,
        contrast_kernels=(3, 5),
        query_kernel=3,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 0 < beta_init < beta_max <= 1:
            raise ValueError("expected 0 < beta_init < beta_max <= 1")
        self.channels = channels
        self.beta_max = float(beta_max)
        self.branches = nn.ModuleList(
            [
                _StaticPaperMAABranch(channels, partial_ratio, dilation, contrast_kernels, query_kernel),
                _StaticPaperMAABranch(channels, partial_ratio, dilation, contrast_kernels, query_kernel),
            ]
        )
        for branch in self.branches:
            nn.init.zeros_(branch.project_out.weight)

        beta_fraction = float(beta_init) / self.beta_max
        raw_beta = math.log(beta_fraction / (1.0 - beta_fraction))
        self.raw_beta = nn.Parameter(torch.full((2, 1, 1, 1), raw_beta))

    @property
    def beta(self):
        return self.beta_max * torch.sigmoid(self.raw_beta)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels * 2:
            raise ValueError(
                "ZeroCenteredStaticMAA2D expects [B, {}, H, W], got {}".format(
                    self.channels * 2, tuple(x.shape)
                )
            )
        rgb, ir = x.chunk(2, dim=1)
        outputs = []
        for index, feature in enumerate((rgb, ir)):
            gate = torch.tanh(self.branches[index].forward_logits(feature))
            outputs.append(feature * (1.0 + self.beta[index] * gate))
        return torch.cat(outputs, dim=1)


class LAFMergeFeedback2D(nn.Module):
    """LAF merge whose learned correction is fed back into both backbones.

    The neck receives the full LAF fused feature. The modality streams receive
    only ``fused - (rgb + ir)``. Since the wrapped V1 merge is initialized as
    exact ADD, V2 feedback is an exact identity at initialization and does not
    disturb migrated single-stream backbone weights.
    """

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8):
        super().__init__()
        self.channels = channels
        self.merge = LAFMerge2D(channels, num_heads, partial_ratio, dilation, gate_reduction)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("LAFMergeFeedback2D expects [rgb, ir]")
        rgb, ir = x
        fused = self.merge((rgb, ir))
        correction = fused - (rgb + ir)
        return rgb + correction, ir + correction, fused


__all__ = ("StaticMAA2D", "ZeroCenteredStaticMAA2D", "LAFMergeFeedback2D")
