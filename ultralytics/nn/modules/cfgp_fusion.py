# Ultralytics YOLO, AGPL-3.0 license
"""Lightweight CFGPNet-inspired fusion for paired RGB/thermal features."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CEAReliability(nn.Module):
    """Build one bounded spatial reliability map with grouped axis/global cues."""

    def __init__(self, channels, groups=8):
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups:
            raise ValueError("channels must be divisible by a positive groups value")
        self.channels = channels
        self.groups = groups
        group_channels = channels // groups
        self.group_channels = group_channels
        self.axis_conv = nn.Conv1d(
            group_channels,
            group_channels,
            kernel_size=3,
            padding=1,
            groups=group_channels,
            bias=False,
        )
        self.axis_norm = nn.GroupNorm(1, group_channels)
        self.local_norm = nn.GroupNorm(1, group_channels)
        self.global_conv = nn.Conv2d(
            group_channels,
            group_channels,
            kernel_size=3,
            padding=1,
            groups=group_channels,
            bias=False,
        )
        self.global_norm = nn.GroupNorm(1, group_channels)
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def _axis_gate(self, descriptor):
        return torch.sigmoid(self.axis_norm(self.axis_conv(descriptor)))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                "_CEAReliability expects [B, {}, H, W], got {}".format(self.channels, tuple(x.shape))
            )
        batch, _, height, width = x.shape
        grouped = x.reshape(batch * self.groups, self.group_channels, height, width)

        height_gate = self._axis_gate(grouped.mean(dim=3)).unsqueeze(3)
        width_gate = self._axis_gate(grouped.mean(dim=2)).unsqueeze(2)
        local = self.local_norm(grouped * height_gate * width_gate)
        global_context = self.global_norm(self.global_conv(grouped))

        local_query = F.softmax(F.adaptive_avg_pool2d(local, 1).flatten(1), dim=1)
        global_query = F.softmax(F.adaptive_avg_pool2d(global_context, 1).flatten(1), dim=1)
        local_h, global_h = local.mean(dim=3), global_context.mean(dim=3)
        local_w, global_w = local.mean(dim=2), global_context.mean(dim=2)
        scale = 1.0 / math.sqrt(self.group_channels)
        score_h = (
            (local_h * global_query.unsqueeze(-1)).sum(dim=1)
            + (global_h * local_query.unsqueeze(-1)).sum(dim=1)
        ) * scale
        score_w = (
            (local_w * global_query.unsqueeze(-1)).sum(dim=1)
            + (global_w * local_query.unsqueeze(-1)).sum(dim=1)
        ) * scale
        temperature = F.softplus(self.log_temperature).clamp_min(1e-3)
        axis_h = F.softmax(score_h / temperature, dim=-1) * height
        axis_w = F.softmax(score_w / temperature, dim=-1) * width
        # Multiplying back the axis lengths keeps the outer product centered
        # around one instead of collapsing to 1/(H*W) on high-resolution P3.
        reliability = torch.sigmoid(axis_h.unsqueeze(-1) * axis_w.unsqueeze(-2) - 1.0)
        return reliability.reshape(batch, self.groups, height, width).mean(dim=1, keepdim=True)


class CFGPCrossAttentionFusion(nn.Module):
    """Cross-CEA reliability exchange with optional lightweight ASAF selection.

    The input is ``[rgb, ir]`` and the output keeps the same channel count.
    ``selection='none'`` isolates CrossCEA-style spatial exchange. With
    ``selection='max'``, three inexpensive candidates (sum, max response and
    globally weighted modalities) are spatially calibrated before elementwise
    max selection. A zero-initialized residual gate makes the initial function
    exactly the baseline ``rgb + ir``.
    """

    VALID_SELECTIONS = {"none", "max"}

    def __init__(self, channels, groups=8, selection="max", reduction=8, residual_scale=0.0):
        super().__init__()
        if selection not in self.VALID_SELECTIONS:
            raise ValueError("selection must be one of {}, got {!r}".format(sorted(self.VALID_SELECTIONS), selection))
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        self.channels = channels
        self.selection = selection
        self.reliability = nn.ModuleList(
            [_CEAReliability(channels, groups), _CEAReliability(channels, groups)]
        )
        self.cross_gain = nn.Parameter(torch.ones(2, 1, 1, 1))
        if selection == "max":
            hidden = max(channels // reduction, 8)
            self.modality_gate = nn.Sequential(
                nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
            )
            self.candidate_gate = nn.Conv2d(5, 3, kernel_size=3, padding=1, bias=True)
            nn.init.zeros_(self.modality_gate[-1].weight)
            nn.init.zeros_(self.modality_gate[-1].bias)
            nn.init.zeros_(self.candidate_gate.weight)
            nn.init.zeros_(self.candidate_gate.bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    @staticmethod
    def _validate_pair(x, channels):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("CFGPCrossAttentionFusion expects [rgb, ir]")
        rgb, ir = x
        if rgb.ndim != 4 or ir.ndim != 4 or rgb.shape != ir.shape or rgb.shape[1] != channels:
            raise ValueError(
                "CFGPCrossAttentionFusion expects two [B, {}, H, W] tensors, got {} and {}".format(
                    channels, tuple(rgb.shape), tuple(ir.shape)
                )
            )
        return rgb, ir

    def _select(self, rgb, ir):
        pooled = torch.cat((F.adaptive_avg_pool2d(rgb, 1), F.adaptive_avg_pool2d(ir, 1)), dim=1)
        modality_weights = F.softmax(self.modality_gate(pooled), dim=1)
        weighted = 2.0 * (modality_weights[:, :1] * rgb + modality_weights[:, 1:] * ir)
        summed = rgb + ir
        strongest = 2.0 * torch.maximum(rgb, ir)
        descriptors = torch.cat(
            (
                rgb.mean(dim=1, keepdim=True),
                ir.mean(dim=1, keepdim=True),
                rgb.amax(dim=1, keepdim=True),
                ir.amax(dim=1, keepdim=True),
                (rgb - ir).abs().mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        gates = torch.sigmoid(self.candidate_gate(descriptors))
        candidates = torch.stack((summed, strongest, weighted), dim=1)
        return (candidates * gates.unsqueeze(2)).amax(dim=1)

    def forward(self, x):
        rgb, ir = self._validate_pair(x, self.channels)
        rgb_map = self.reliability[0](rgb)
        ir_map = self.reliability[1](ir)
        rgb_cross = rgb * (1.0 + self.cross_gain[0] * (2.0 * ir_map - 1.0))
        ir_cross = ir * (1.0 + self.cross_gain[1] * (2.0 * rgb_map - 1.0))
        proposal = rgb_cross + ir_cross if self.selection == "none" else self._select(rgb_cross, ir_cross)
        baseline = rgb + ir
        return baseline + self.residual_scale * (proposal - baseline)


__all__ = ("CFGPCrossAttentionFusion",)
