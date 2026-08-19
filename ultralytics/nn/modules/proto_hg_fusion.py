# Ultralytics YOLO, AGPL-3.0 license
"""Prototype hypergraph fusion adapted for paired RGB/thermal YOLO features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PrototypeEncoder(nn.Module):
    """Compress a feature map into a small semantic-prototype set."""

    def __init__(self, channels, prototype_channels, num_prototypes):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, prototype_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(prototype_channels),
            nn.SiLU(inplace=True),
        )
        self.assignment = nn.Conv2d(channels, num_prototypes, kernel_size=1, bias=True)

    def forward(self, x):
        batch, _, height, width = x.shape
        reduced = self.reduce(x).flatten(2)
        assignment = F.softmax(self.assignment(x).flatten(2).float(), dim=-1).to(dtype=reduced.dtype)
        prototypes = torch.bmm(assignment, reduced.transpose(1, 2))
        return prototypes, assignment, (height, width)


class ProtoHypergraphFusion(nn.Module):
    """Sparse prototype-level relation propagation and globally gated fusion.

    Compared with the paper, prototype features use a C/reduction bottleneck
    and are projected back as a low-rank dense residual. This keeps parameter
    and compute growth modest at P3-P5. ``relation='hard'`` uses binary top-k
    structure, while ``relation='soft'`` keeps confidence weights over the
    same neighborhood for a controlled ablation. The initial output is exactly
    ``rgb + ir`` because ``residual_scale`` starts at zero.
    """

    VALID_RELATIONS = {"hard", "soft"}

    def __init__(
        self,
        channels,
        num_prototypes=6,
        k_intra=3,
        k_cross=3,
        reduction=4,
        relation="hard",
        residual_scale=0.0,
        eps=1e-6,
    ):
        super().__init__()
        if channels <= 0 or num_prototypes <= 1 or reduction <= 0:
            raise ValueError("channels/reduction must be positive and num_prototypes must exceed one")
        if not 0 < k_intra <= num_prototypes or not 0 < k_cross <= num_prototypes:
            raise ValueError("k_intra and k_cross must be in [1, num_prototypes]")
        if relation not in self.VALID_RELATIONS:
            raise ValueError("relation must be one of {}, got {!r}".format(sorted(self.VALID_RELATIONS), relation))
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.k_intra = k_intra
        self.k_cross = k_cross
        self.relation = relation
        self.eps = eps
        prototype_channels = max(channels // reduction, 16)
        self.prototype_channels = prototype_channels
        self.encoder = nn.ModuleList(
            [_PrototypeEncoder(channels, prototype_channels, num_prototypes) for _ in range(2)]
        )
        self.project_back = nn.ModuleList(
            [nn.Conv2d(prototype_channels, channels, kernel_size=1, bias=False) for _ in range(2)]
        )
        gate_hidden = max(channels // 16, 8)
        self.modality_gate = nn.Sequential(
            nn.Conv2d(channels * 2, gate_hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 2, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.modality_gate[-1].weight)
        nn.init.zeros_(self.modality_gate[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    @staticmethod
    def _validate_pair(x, channels):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("ProtoHypergraphFusion expects [rgb, ir]")
        rgb, ir = x
        if rgb.ndim != 4 or ir.ndim != 4 or rgb.shape != ir.shape or rgb.shape[1] != channels:
            raise ValueError(
                "ProtoHypergraphFusion expects two [B, {}, H, W] tensors, got {} and {}".format(
                    channels, tuple(rgb.shape), tuple(ir.shape)
                )
            )
        return rgb, ir

    def _topk_relation(self, similarity, k, self_loops=False):
        values, indices = similarity.topk(k, dim=-1)
        if self.relation == "soft":
            edge_values = F.softmax(values, dim=-1).to(dtype=similarity.dtype)
        else:
            edge_values = torch.ones_like(values)
        relation = torch.zeros_like(similarity).scatter(-1, indices, edge_values)
        if self_loops:
            identity = torch.eye(similarity.shape[-1], device=similarity.device, dtype=similarity.dtype)
            relation = torch.maximum(relation, identity.unsqueeze(0))
        return relation

    def _joint_relation(self, rgb_prototypes, ir_prototypes):
        rgb_normalized = F.normalize(rgb_prototypes.float(), dim=-1, eps=self.eps)
        ir_normalized = F.normalize(ir_prototypes.float(), dim=-1, eps=self.eps)
        rgb_rgb = torch.bmm(rgb_normalized, rgb_normalized.transpose(1, 2))
        ir_ir = torch.bmm(ir_normalized, ir_normalized.transpose(1, 2))
        rgb_ir = torch.bmm(rgb_normalized, ir_normalized.transpose(1, 2))
        h_rr = self._topk_relation(rgb_rgb, self.k_intra, self_loops=True)
        h_tt = self._topk_relation(ir_ir, self.k_intra, self_loops=True)
        h_rt = self._topk_relation(rgb_ir, self.k_cross)
        top = torch.cat((h_rr, h_rt), dim=-1)
        bottom = torch.cat((h_rt.transpose(1, 2), h_tt), dim=-1)
        return torch.cat((top, bottom), dim=1)

    def _aggregate(self, source, relation):
        normalized = relation / relation.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.bmm(normalized.to(dtype=source.dtype), source)

    def _propagate(self, rgb_prototypes, ir_prototypes):
        vertices = torch.cat((rgb_prototypes, ir_prototypes), dim=1)
        relation = self._joint_relation(rgb_prototypes, ir_prototypes)
        hyperedges = self._aggregate(vertices, relation.transpose(1, 2))
        update = self._aggregate(hyperedges, relation)
        propagated = vertices + update
        return propagated.split(self.num_prototypes, dim=1)

    def _dense_update(self, propagated, original, assignment, spatial_shape, projection):
        delta = propagated - original
        dense = torch.bmm(delta.transpose(1, 2), assignment)
        dense = dense.reshape(dense.shape[0], self.prototype_channels, *spatial_shape)
        return projection(torch.tanh(dense))

    def forward(self, x):
        rgb, ir = self._validate_pair(x, self.channels)
        rgb_prototypes, rgb_assignment, spatial_shape = self.encoder[0](rgb)
        ir_prototypes, ir_assignment, _ = self.encoder[1](ir)
        rgb_propagated, ir_propagated = self._propagate(rgb_prototypes, ir_prototypes)
        rgb_modulated = rgb + self._dense_update(
            rgb_propagated, rgb_prototypes, rgb_assignment, spatial_shape, self.project_back[0]
        )
        ir_modulated = ir + self._dense_update(
            ir_propagated, ir_prototypes, ir_assignment, spatial_shape, self.project_back[1]
        )
        pooled = torch.cat(
            (F.adaptive_avg_pool2d(rgb_modulated, 1), F.adaptive_avg_pool2d(ir_modulated, 1)), dim=1
        )
        weights = F.softmax(self.modality_gate(pooled), dim=1)
        proposal = 2.0 * (weights[:, :1] * rgb_modulated + weights[:, 1:] * ir_modulated)
        baseline = rgb + ir
        return baseline + self.residual_scale * (proposal - baseline)


__all__ = ("ProtoHypergraphFusion",)
