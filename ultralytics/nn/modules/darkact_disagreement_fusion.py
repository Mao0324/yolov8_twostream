"""Lightweight disagreement-aware residual extension for DarkAct LAF fusion."""

import torch
import torch.nn as nn

from .darkact_maalaf_v2 import LAFMergeFeedback2D


class DisagreementLAFMergeFeedback2D(LAFMergeFeedback2D):
    """Add an explicit RGB/IR disagreement residual to the existing LAF merge.

    The original LAF weighted sum and partial-channel cross residual are kept
    unchanged.  A lightweight branch observes both modalities, their absolute
    difference, and their element-wise agreement.  ``gamma`` starts at zero,
    so a migrated parent checkpoint produces exactly the original LAF output
    before training.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        disagreement_ratio=4,
    ):
        super().__init__(channels, num_heads, partial_ratio, dilation, gate_reduction)
        if disagreement_ratio <= 0:
            raise ValueError("disagreement_ratio must be positive")
        hidden = max(channels // disagreement_ratio, 8)
        self.disagreement_project = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.disagreement_gamma = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("DisagreementLAFMergeFeedback2D expects [rgb, ir]")
        rgb, ir = x
        base_fused = self.merge((rgb, ir))
        interaction = torch.cat((rgb, ir, torch.abs(rgb - ir), rgb * ir), dim=1)
        delta = self.disagreement_project(interaction)
        fused = base_fused + self.disagreement_gamma * delta
        correction = fused - (rgb + ir)
        return rgb + correction, ir + correction, fused


class SemanticDisagreementLAFMergeFeedback2D(LAFMergeFeedback2D):
    """LAF fusion with a four-way semantic decomposition residual.

    P3/P4 features are decomposed into shared, RGB-specific, IR-specific, and
    disagreement components. A joint gate predicts four spatially varying
    weights. The semantic residual follows

        w_shared * F_shared
        - w_rgb * F_rgb_specific
        - w_ir * F_ir_specific
        - w_disagreement * F_disagreement.

    ``residual_gain`` is initialized to zero. Therefore a freshly migrated
    checkpoint is exactly equivalent to the parent LAF fusion, while the new
    branch can be learned without changing any training or augmentation
    settings.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        semantic_ratio=4,
    ):
        super().__init__(channels, num_heads, partial_ratio, dilation, gate_reduction)
        if semantic_ratio <= 0:
            raise ValueError("semantic_ratio must be positive")
        hidden = max(channels // semantic_ratio, 8)

        def projector(input_channels):
            return nn.Sequential(
                nn.Conv2d(input_channels, hidden, kernel_size=1, bias=False),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            )

        self.shared_project = projector(channels * 3)
        self.rgb_specific_project = projector(channels * 2)
        self.ir_specific_project = projector(channels * 2)
        self.disagreement_project = projector(channels)
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=1, bias=True),
        )
        self.residual_gain = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("SemanticDisagreementLAFMergeFeedback2D expects [rgb, ir]")
        rgb, ir = x
        base_fused = self.merge((rgb, ir))
        difference = rgb - ir
        absolute_difference = difference.abs()
        agreement = rgb * ir

        shared = self.shared_project(torch.cat((rgb, ir, agreement), dim=1))
        rgb_specific = self.rgb_specific_project(torch.cat((rgb, difference), dim=1))
        ir_specific = self.ir_specific_project(torch.cat((ir, -difference), dim=1))
        disagreement = self.disagreement_project(absolute_difference)

        weights = torch.softmax(
            self.weight_predictor(torch.cat((rgb, ir, absolute_difference, agreement), dim=1)),
            dim=1,
        )
        w_shared, w_rgb, w_ir, w_disagreement = weights.chunk(4, dim=1)
        semantic_delta = (
            w_shared * shared
            - w_rgb * rgb_specific
            - w_ir * ir_specific
            - w_disagreement * disagreement
        )
        fused = base_fused + self.residual_gain * semantic_delta
        correction = fused - (rgb + ir)
        return rgb + correction, ir + correction, fused


__all__ = ("DisagreementLAFMergeFeedback2D", "SemanticDisagreementLAFMergeFeedback2D")
