# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Additive lightweight refinement blocks for two-stream feature ablations."""

import torch
import torch.nn as nn

from .conv import Conv


class ZeroInitResidualRefine2D(nn.Module):
    """Refine one modality with a zero-initialized bottleneck residual.

    The residual branch is ``1x1 reduce -> 3x3 restore``. A learnable scalar
    ``gamma`` is initialized to zero, so the module starts as the exact identity
    ``Z = Y`` while keeping an independent refinement path for each stream.
    """

    def __init__(self, c1, c2, expansion=0.25):
        super().__init__()
        if c1 != c2:
            raise ValueError(f"ZeroInitResidualRefine2D requires c1 == c2, got {c1} and {c2}")
        if not 0 < expansion <= 1:
            raise ValueError("expansion must be in (0, 1]")

        hidden = max(1, int(c1 * expansion))
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden, c2, 3, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.cv2(self.cv1(x))


__all__ = ("ZeroInitResidualRefine2D",)
