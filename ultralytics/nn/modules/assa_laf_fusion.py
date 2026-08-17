# Ultralytics YOLO 🚀, AGPL-3.0 license
"""ASSA/DarkAct fusion variants with explicit, non-overlapping responsibilities."""

import torch
import torch.nn as nn

from .assa_partial_channel_fusion import PartialChannelASSAFusion
from .darkact_maalaf import LAFMerge2D
from .darkact_maalaf_v2 import LAFMergeFeedback2D


class ASSALAFMergeFeedback2D(nn.Module):
    """Align with partial ASSA, then fuse and feed back with LAF.

    ASSA exchanges information while the streams are separate. LAF then makes
    one fused lateral and feeds its correction into both backbones. The ASSA
    output projections are zero initialized, so inserting this class exactly
    preserves a migrated LAF model at initialization.
    """

    def __init__(self, channels, num_heads=4, partial_ratio=4, static_kernel=3,
                 assa_residual_scale=0.1, laf_dilation=1, gate_reduction=8):
        super().__init__()
        self.channels = channels
        self.align = PartialChannelASSAFusion(
            channels, num_heads, partial_ratio, static_kernel, assa_residual_scale
        )
        for projection in self.align.project_out:
            nn.init.zeros_(projection.weight)
        self.fuse = LAFMergeFeedback2D(
            channels, num_heads, partial_ratio, laf_dilation, gate_reduction
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("ASSALAFMergeFeedback2D expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(
                "expected equal [B, {}, H, W] streams, got {} and {}".format(
                    self.channels, tuple(rgb.shape), tuple(ir.shape)
                )
            )
        aligned_rgb, aligned_ir = self.align(torch.cat((rgb, ir), dim=1)).chunk(2, dim=1)
        return self.fuse((aligned_rgb, aligned_ir))


class ASSAReplacedLAFMerge2D(nn.Module):
    """LAF reliability gating whose original cross branch is replaced by ASSA.

    ``reliability`` is a LAF merge with ``num_heads=0``: it retains only the
    global channel gate and local spatial gate. ``assa`` supplies the sole
    cross-modal interaction. Its two directional residuals are added to the
    gated fused lateral, rather than rewriting the streams before LAF.

    Zero-initialized ASSA output projections make this exactly equivalent to a
    migrated LAF whose original ``cross_scale`` is zero.
    """

    def __init__(self, channels, num_heads=4, partial_ratio=4, static_kernel=3,
                 assa_residual_scale=0.1, laf_dilation=1, gate_reduction=8):
        super().__init__()
        self.channels = channels
        self.reliability = LAFMerge2D(
            channels, num_heads=0, partial_ratio=partial_ratio,
            dilation=laf_dilation, gate_reduction=gate_reduction,
        )
        # LAFMerge2D creates cross_scale even when num_heads=0, while its
        # forward uses that parameter only when ``cross`` exists. Leaving it
        # trainable makes multi-GPU DDP fail on the second iteration with an
        # unfinished reduction. Remove the disabled branch's scalar entirely.
        if self.reliability.cross is not None:
            raise RuntimeError("reliability-only LAF unexpectedly created a cross branch")
        self.reliability.register_parameter("cross_scale", None)
        self.assa = PartialChannelASSAFusion(
            channels, num_heads, partial_ratio, static_kernel, assa_residual_scale
        )
        for projection in self.assa.project_out:
            nn.init.zeros_(projection.weight)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("ASSAReplacedLAFMerge2D expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(
                "expected equal [B, {}, H, W] streams, got {} and {}".format(
                    self.channels, tuple(rgb.shape), tuple(ir.shape)
                )
            )
        gated_fused = self.reliability((rgb, ir))
        aligned_rgb, aligned_ir = self.assa(torch.cat((rgb, ir), dim=1)).chunk(2, dim=1)
        return gated_fused + (aligned_rgb - rgb) + (aligned_ir - ir)


class ASSAReplacedLAFMergeFeedback2D(nn.Module):
    """ASSA-replaced LAF merge with the unchanged DarkAct feedback contract."""

    def __init__(self, channels, num_heads=4, partial_ratio=4, static_kernel=3,
                 assa_residual_scale=0.1, laf_dilation=1, gate_reduction=8):
        super().__init__()
        self.channels = channels
        self.merge = ASSAReplacedLAFMerge2D(
            channels, num_heads, partial_ratio, static_kernel,
            assa_residual_scale, laf_dilation, gate_reduction,
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("ASSAReplacedLAFMergeFeedback2D expects [rgb, ir]")
        rgb, ir = x
        fused = self.merge((rgb, ir))
        correction = fused - (rgb + ir)
        return rgb + correction, ir + correction, fused


__all__ = (
    "ASSALAFMergeFeedback2D",
    "ASSAReplacedLAFMerge2D",
    "ASSAReplacedLAFMergeFeedback2D",
)
