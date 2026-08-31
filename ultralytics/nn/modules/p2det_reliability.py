# Ultralytics YOLO 🚀, AGPL-3.0 license
"""V16 detection-reliability prompts and prior-conditioned RGB/IR LAF."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darkact_maalaf import LAFMerge2D
from .darkact_maalaf_v2 import LAFMergeFeedback2D
from .darkact_disagreement_fusion import SemanticDisagreementLAFMergeFeedback2D


class _ChannelLayerNorm2d(nn.Module):
    """Per-position channel normalization used by each modality prompt head."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = x.var(1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class DetectionReliabilityPromptHead(nn.Module):
    """Predict one spatial detection-reliability logit from one modality."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 16, 8)
        self.norm = _ChannelLayerNorm2d(channels)
        self.body = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        # Equal initial reliability. The relative teacher, rather than this
        # initialization, determines which modality should become dominant.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        z = self.norm(x)
        contrast3 = (z - F.avg_pool2d(z, 3, stride=1, padding=1)).abs().mean(1, keepdim=True)
        contrast5 = (z - F.avg_pool2d(z, 5, stride=1, padding=2)).abs().mean(1, keepdim=True)
        descriptor = torch.cat((z.amax(1, keepdim=True), contrast3, contrast5), dim=1)
        return self.body(descriptor)


class DetectionObjectReliabilityPromptHead(nn.Module):
    """Bias-free dense logits that must derive object reliability from features."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 16, 8)
        self.norm = _ChannelLayerNorm2d(channels)
        self.body = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=False),
        )
        nn.init.zeros_(self.body[-1].weight)

    def forward(self, x):
        z = self.norm(x)
        contrast3 = (z - F.avg_pool2d(z, 3, stride=1, padding=1)).abs().mean(1, keepdim=True)
        contrast5 = (z - F.avg_pool2d(z, 5, stride=1, padding=2)).abs().mean(1, keepdim=True)
        descriptor = torch.cat((z.amax(1, keepdim=True), contrast3, contrast5), dim=1)
        return self.body(descriptor)


class DA016DualPromptAuxSemanticLAFMergeFeedback2D(SemanticDisagreementLAFMergeFeedback2D):
    """DA016 semantic fusion plus two auxiliary object-reliability Prompt heads.

    The Prompt is deliberately absent from the detection forward path.  The
    inherited DA016 semantic-disagreement merge and feedback are returned
    unchanged; Prompt supervision can only affect training through its
    auxiliary loss and shared input features.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        semantic_ratio=4,
        detach_prompt_features=False,
    ):
        super().__init__(
            channels,
            num_heads,
            partial_ratio,
            dilation,
            gate_reduction,
            semantic_ratio,
        )
        self.channels = channels
        self.rgb_prompt_head = DetectionObjectReliabilityPromptHead(channels)
        self.ir_prompt_head = DetectionObjectReliabilityPromptHead(channels)
        # B-Detach causal control: Prompt heads still learn, but their loss
        # cannot reshape the RGB/IR detector features.
        self.detach_prompt_features = bool(detach_prompt_features)
        self.capture_prompt = False
        self._last_aux_outputs = None

    def _prompt_maps(self, rgb, ir):
        if getattr(self, "detach_prompt_features", False):
            rgb, ir = rgb.detach(), ir.detach()
        prompt_logits = torch.cat((self.rgb_prompt_head(rgb), self.ir_prompt_head(ir)), dim=1)
        return prompt_logits, prompt_logits.softmax(dim=1)

    def _capture(self, prompt_logits, prompt_probabilities, **extra):
        if self.capture_prompt:
            self._last_aux_outputs = {
                "prompt_logits": prompt_logits,
                "prompt_probabilities": prompt_probabilities,
                **extra,
            }

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"{self.__class__.__name__} expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"invalid RGB/IR feature shapes: {tuple(rgb.shape)}, {tuple(ir.shape)}")
        # Auxiliary-only Prompts are a training regularizer. Standard
        # detection does not need their predictions, so preserve the exact
        # parent forward and avoid unnecessary inference compute. The trainer
        # explicitly enables capture during validation before model.forward().
        if not self.training and not self.capture_prompt:
            return SemanticDisagreementLAFMergeFeedback2D.forward(self, x)
        prompt_logits, prompt_probabilities = self._prompt_maps(rgb, ir)
        outputs = SemanticDisagreementLAFMergeFeedback2D.forward(self, x)
        self._capture(prompt_logits, prompt_probabilities)
        return outputs

    def pop_aux_outputs(self):
        outputs = self._last_aux_outputs
        self._last_aux_outputs = None
        return outputs


class DA016DualPromptResidualSemanticLAFMergeFeedback2D(
    DA016DualPromptAuxSemanticLAFMergeFeedback2D
):
    """DA016 semantic fusion plus a zero-init bounded Prompt residual.

    The original DA016 fused feature is always computed first.  The only new
    detection path is

        F_out = F_DA016 + alpha * (P_rgb - P_ir) * (F_rgb - F_ir) / 2

    where ``alpha = max_residual_gain * tanh(raw_residual_gain)``.  Thus alpha
    starts at exactly zero, stays bounded, and may learn either sign; there is
    intentionally no sign-preserving constraint in this ablation.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        semantic_ratio=4,
        max_residual_gain=0.25,
    ):
        super().__init__(
            channels,
            num_heads,
            partial_ratio,
            dilation,
            gate_reduction,
            semantic_ratio,
        )
        if float(max_residual_gain) <= 0:
            raise ValueError("max_residual_gain must be positive")
        self.max_residual_gain = float(max_residual_gain)
        self.raw_prompt_residual_gain = nn.Parameter(torch.zeros(1, 1, 1, 1))

    @property
    def prompt_residual_gain(self):
        return self.max_residual_gain * self.raw_prompt_residual_gain.tanh()

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"{self.__class__.__name__} expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"invalid RGB/IR feature shapes: {tuple(rgb.shape)}, {tuple(ir.shape)}")
        prompt_logits, prompt_probabilities = self._prompt_maps(rgb, ir)
        _, _, base_fused = SemanticDisagreementLAFMergeFeedback2D.forward(self, x)
        prompt_margin = (prompt_probabilities[:, :1] - prompt_probabilities[:, 1:2]).to(rgb.dtype)
        prompt_delta = 0.5 * prompt_margin * (rgb - ir)
        fused = base_fused + self.prompt_residual_gain.to(rgb.dtype) * prompt_delta
        correction = fused - (rgb + ir)
        self._capture(
            prompt_logits,
            prompt_probabilities,
            prompt_residual_gain=self.prompt_residual_gain,
        )
        return rgb + correction, ir + correction, fused


class DarkACTDualPromptAuxLAFMergeFeedback2D(LAFMergeFeedback2D):
    """Original DarkACT P5 LAF plus auxiliary RGB/IR reliability Prompts.

    This is the P5-specific counterpart of
    :class:`DA016DualPromptAuxSemanticLAFMergeFeedback2D`.  It deliberately
    inherits the original ``LAFMergeFeedback2D`` used by DA016 at P5 rather
    than replacing it with the P3/P4 semantic-disagreement fusion.  Prompt
    predictions are captured for auxiliary supervision but do not enter the
    detection forward path.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        detach_prompt_features=False,
    ):
        super().__init__(channels, num_heads, partial_ratio, dilation, gate_reduction)
        self.rgb_prompt_head = DetectionObjectReliabilityPromptHead(channels)
        self.ir_prompt_head = DetectionObjectReliabilityPromptHead(channels)
        self.detach_prompt_features = bool(detach_prompt_features)
        self.capture_prompt = False
        self._last_aux_outputs = None

    def _prompt_maps(self, rgb, ir):
        if getattr(self, "detach_prompt_features", False):
            rgb, ir = rgb.detach(), ir.detach()
        prompt_logits = torch.cat((self.rgb_prompt_head(rgb), self.ir_prompt_head(ir)), dim=1)
        return prompt_logits, prompt_logits.softmax(dim=1)

    def _capture(self, prompt_logits, prompt_probabilities, **extra):
        if self.capture_prompt:
            self._last_aux_outputs = {
                "prompt_logits": prompt_logits,
                "prompt_probabilities": prompt_probabilities,
                **extra,
            }

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"{self.__class__.__name__} expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"invalid RGB/IR feature shapes: {tuple(rgb.shape)}, {tuple(ir.shape)}")
        if not self.training and not self.capture_prompt:
            return LAFMergeFeedback2D.forward(self, x)
        prompt_logits, prompt_probabilities = self._prompt_maps(rgb, ir)
        outputs = LAFMergeFeedback2D.forward(self, x)
        self._capture(prompt_logits, prompt_probabilities)
        return outputs

    def pop_aux_outputs(self):
        outputs = self._last_aux_outputs
        self._last_aux_outputs = None
        return outputs


class DarkACTDualPromptResidualLAFMergeFeedback2D(
    DarkACTDualPromptAuxLAFMergeFeedback2D
):
    """Original DarkACT P5 LAF plus a zero-init bounded Prompt residual.

    The detection path is exactly

        F_P5 = F_original_P5_LAF
             + alpha_P5 * (P_rgb - P_ir) * (F_rgb - F_ir) / 2

    where ``alpha_P5 = max_residual_gain * tanh(raw_gain)``.  At construction
    alpha is exactly zero, so this wrapper is forward-equivalent to the
    original DA016 P5 LAF before training.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        max_residual_gain=0.25,
    ):
        super().__init__(channels, num_heads, partial_ratio, dilation, gate_reduction)
        if float(max_residual_gain) <= 0:
            raise ValueError("max_residual_gain must be positive")
        self.max_residual_gain = float(max_residual_gain)
        self.raw_prompt_residual_gain = nn.Parameter(torch.zeros(1, 1, 1, 1))

    @property
    def prompt_residual_gain(self):
        return self.max_residual_gain * self.raw_prompt_residual_gain.tanh()

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"{self.__class__.__name__} expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"invalid RGB/IR feature shapes: {tuple(rgb.shape)}, {tuple(ir.shape)}")
        prompt_logits, prompt_probabilities = self._prompt_maps(rgb, ir)
        _, _, base_fused = LAFMergeFeedback2D.forward(self, x)
        prompt_margin = (prompt_probabilities[:, :1] - prompt_probabilities[:, 1:2]).to(rgb.dtype)
        prompt_delta = 0.5 * prompt_margin * (rgb - ir)
        fused = base_fused + self.prompt_residual_gain.to(rgb.dtype) * prompt_delta
        correction = fused - (rgb + ir)
        self._capture(
            prompt_logits,
            prompt_probabilities,
            prompt_residual_gain=self.prompt_residual_gain,
        )
        return rgb + correction, ir + correction, fused


class PriorConditionedLAFMerge2D(LAFMerge2D):
    """DarkACT LAF whose RGB/IR gate explicitly consumes both prompt probabilities.

    All new prompt-to-gate projections and the inherited cross-modal residual
    are zero-initialized. Consequently a migrated DarkACT checkpoint initially
    computes exactly the original LAF result while retaining a direct trainable
    path from reliability prompts to modality weights.
    """

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8):
        super().__init__(channels, num_heads, partial_ratio, dilation, gate_reduction)
        hidden = max(channels // gate_reduction, 8)
        self.prompt_global_gate = nn.Sequential(
            nn.Conv2d(2, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1, bias=True),
        )
        local_hidden = max(min(hidden, 32), 8)
        self.prompt_local_gate = nn.Sequential(
            nn.Conv2d(2, local_hidden, 3, padding=dilation, dilation=dilation, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(local_hidden, 2, 1, bias=True),
        )
        nn.init.zeros_(self.prompt_global_gate[-1].weight)
        nn.init.zeros_(self.prompt_global_gate[-1].bias)
        nn.init.zeros_(self.prompt_local_gate[-1].weight)
        nn.init.zeros_(self.prompt_local_gate[-1].bias)
        self.capture_gate = False
        self._last_modal_weights = None

    def forward(self, x, prompt_probabilities):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("PriorConditionedLAFMerge2D expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"invalid RGB/IR feature shapes: {tuple(rgb.shape)}, {tuple(ir.shape)}")
        if prompt_probabilities.shape != (rgb.shape[0], 2, rgb.shape[2], rgb.shape[3]):
            raise ValueError(
                "prompt probabilities must be [B,2,H,W], got "
                f"{tuple(prompt_probabilities.shape)}"
            )

        descriptor = torch.cat(
            (
                F.adaptive_avg_pool2d(rgb, 1),
                F.adaptive_max_pool2d(rgb, 1),
                F.adaptive_avg_pool2d(ir, 1),
                F.adaptive_max_pool2d(ir, 1),
            ),
            dim=1,
        )
        b, _, h, w = rgb.shape
        feature_global = self.global_gate(descriptor).reshape(b, 2, self.channels, 1, 1)
        local_descriptor = torch.cat((self._contrast_descriptor(rgb), self._contrast_descriptor(ir)), dim=1)
        feature_local = self.local_gate(local_descriptor).reshape(b, 2, 1, h, w)

        prompt_global_descriptor = F.adaptive_avg_pool2d(prompt_probabilities, 1)
        prompt_global = self.prompt_global_gate(prompt_global_descriptor).reshape(
            b, 2, self.channels, 1, 1
        )
        prompt_local = self.prompt_local_gate(prompt_probabilities).reshape(b, 2, 1, h, w)

        # Shape [B, modality, channel, H, W]. The factor two preserves the
        # historical DarkACT convention: equal gates produce rgb + ir.
        modal_weights = torch.softmax(
            feature_global + feature_local + prompt_global + prompt_local, dim=1
        ) * 2.0
        fused = modal_weights[:, 0] * rgb + modal_weights[:, 1] * ir
        if self.cross is not None:
            fused = fused + self.cross_scale * self.cross(rgb, ir)
        if self.capture_gate:
            self._last_modal_weights = modal_weights
        return fused

    def pop_modal_weights(self):
        weights = self._last_modal_weights
        self._last_modal_weights = None
        return weights


class P2DualReliabilityPriorLAFMergeFeedback2D(nn.Module):
    """V16 P3/P4 dual-reliability Prompt, prompt-conditioned LAF, and feedback."""

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8):
        super().__init__()
        self.channels = channels
        self.merge = PriorConditionedLAFMerge2D(
            channels, num_heads, partial_ratio, dilation, gate_reduction
        )
        self.rgb_prompt_head = DetectionReliabilityPromptHead(channels)
        self.ir_prompt_head = DetectionReliabilityPromptHead(channels)
        self.capture_prompt = False
        self._last_aux_outputs = None

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"{self.__class__.__name__} expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"invalid RGB/IR feature shapes: {tuple(rgb.shape)}, {tuple(ir.shape)}")

        prompt_logits = torch.cat((self.rgb_prompt_head(rgb), self.ir_prompt_head(ir)), dim=1)
        prompt_probabilities = prompt_logits.softmax(dim=1)
        self.merge.capture_gate = self.capture_prompt
        fused = self.merge((rgb, ir), prompt_probabilities)

        if self.capture_prompt:
            self._last_aux_outputs = {
                "prompt_logits": prompt_logits,
                "prompt_probabilities": prompt_probabilities,
                "modal_weights": self.merge.pop_modal_weights(),
            }
        correction = fused - (rgb + ir)
        return rgb + correction, ir + correction, fused

    def pop_aux_outputs(self):
        outputs = self._last_aux_outputs
        self._last_aux_outputs = None
        return outputs


class MonotonicPriorConditionedLAFMerge2D(LAFMerge2D):
    """LAF with a positive, bias-free log-Prompt contribution to modal logits."""

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        initial_prompt_strength=1.0,
    ):
        super().__init__(channels, num_heads, partial_ratio, dilation, gate_reduction)
        if initial_prompt_strength <= 0:
            raise ValueError("initial_prompt_strength must be positive")
        inverse_softplus = torch.log(torch.expm1(torch.tensor(float(initial_prompt_strength))))
        self.raw_prompt_strength = nn.Parameter(inverse_softplus.reshape(1))
        self.capture_gate = False
        self._last_modal_weights = None

    @property
    def prompt_strength(self):
        return F.softplus(self.raw_prompt_strength)

    def forward(self, x, prompt_probabilities):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("MonotonicPriorConditionedLAFMerge2D expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"invalid RGB/IR feature shapes: {tuple(rgb.shape)}, {tuple(ir.shape)}")
        if prompt_probabilities.shape != (rgb.shape[0], 2, rgb.shape[2], rgb.shape[3]):
            raise ValueError(
                "prompt probabilities must be [B,2,H,W], got "
                f"{tuple(prompt_probabilities.shape)}"
            )

        descriptor = torch.cat(
            (
                F.adaptive_avg_pool2d(rgb, 1),
                F.adaptive_max_pool2d(rgb, 1),
                F.adaptive_avg_pool2d(ir, 1),
                F.adaptive_max_pool2d(ir, 1),
            ),
            dim=1,
        )
        batch_size, _, height, width = rgb.shape
        feature_global = self.global_gate(descriptor).reshape(batch_size, 2, self.channels, 1, 1)
        local_descriptor = torch.cat((self._contrast_descriptor(rgb), self._contrast_descriptor(ir)), dim=1)
        feature_local = self.local_gate(local_descriptor).reshape(batch_size, 2, 1, height, width)
        prompt_prior = (
            self.prompt_strength * prompt_probabilities.float().clamp_min(1e-6).log()
        ).to(dtype=rgb.dtype).unsqueeze(2)
        modal_weights = torch.softmax(feature_global + feature_local + prompt_prior, dim=1) * 2.0
        fused = modal_weights[:, 0] * rgb + modal_weights[:, 1] * ir
        if self.cross is not None:
            fused = fused + self.cross_scale * self.cross(rgb, ir)
        if self.capture_gate:
            self._last_modal_weights = modal_weights
        return fused

    def pop_modal_weights(self):
        weights = self._last_modal_weights
        self._last_modal_weights = None
        return weights


class P2DualObjectReliabilityMonotonicLAFMergeFeedback2D(nn.Module):
    """V17 bias-free object reliability Prompt with monotonic LAF conditioning."""

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8):
        super().__init__()
        self.channels = channels
        self.merge = MonotonicPriorConditionedLAFMerge2D(
            channels, num_heads, partial_ratio, dilation, gate_reduction
        )
        self.rgb_prompt_head = DetectionObjectReliabilityPromptHead(channels)
        self.ir_prompt_head = DetectionObjectReliabilityPromptHead(channels)
        self.capture_prompt = False
        self._last_aux_outputs = None

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"{self.__class__.__name__} expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"invalid RGB/IR feature shapes: {tuple(rgb.shape)}, {tuple(ir.shape)}")
        prompt_logits = torch.cat((self.rgb_prompt_head(rgb), self.ir_prompt_head(ir)), dim=1)
        prompt_probabilities = prompt_logits.softmax(dim=1)
        self.merge.capture_gate = self.capture_prompt
        fused = self.merge((rgb, ir), prompt_probabilities)
        if self.capture_prompt:
            self._last_aux_outputs = {
                "prompt_logits": prompt_logits,
                "prompt_probabilities": prompt_probabilities,
                "modal_weights": self.merge.pop_modal_weights(),
            }
        correction = fused - (rgb + ir)
        return rgb + correction, ir + correction, fused

    def pop_aux_outputs(self):
        outputs = self._last_aux_outputs
        self._last_aux_outputs = None
        return outputs


class SignPreservingPromptConditionedLAFMerge2D(LAFMerge2D):
    """Prompt-primary LAF whose feature residual cannot reverse modality direction.

    For Prompt log-odds ``d_p`` and arbitrary feature-gate margin ``d_f``:

        d_gate = d_p + rho * tanh(d_f) * abs(d_p),  0 <= rho < 1

    Therefore ``sign(d_gate) == sign(d_p)`` whenever the Prompt has a
    non-zero preference. The inherited DarkACT feature gate can adjust
    confidence and channel allocation, but it cannot turn an IR Prompt into
    an RGB gate (or vice versa).
    """

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8, rho=0.5):
        super().__init__(channels, num_heads, partial_ratio, dilation, gate_reduction)
        if not 0 <= float(rho) < 1:
            raise ValueError("sign-preserving feature residual rho must be in [0,1)")
        self.rho = float(rho)
        self.register_buffer("fixed_prompt_strength", torch.ones(1))
        self.capture_gate = False
        self._last_modal_weights = None

    @property
    def prompt_strength(self):
        return self.fixed_prompt_strength

    def forward(self, x, prompt_probabilities):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("SignPreservingPromptConditionedLAFMerge2D expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"invalid RGB/IR feature shapes: {tuple(rgb.shape)}, {tuple(ir.shape)}")
        if prompt_probabilities.shape != (rgb.shape[0], 2, rgb.shape[2], rgb.shape[3]):
            raise ValueError(
                "prompt probabilities must be [B,2,H,W], got "
                f"{tuple(prompt_probabilities.shape)}"
            )

        descriptor = torch.cat(
            (
                F.adaptive_avg_pool2d(rgb, 1),
                F.adaptive_max_pool2d(rgb, 1),
                F.adaptive_avg_pool2d(ir, 1),
                F.adaptive_max_pool2d(ir, 1),
            ),
            dim=1,
        )
        batch_size, _, height, width = rgb.shape
        feature_global = self.global_gate(descriptor).reshape(batch_size, 2, self.channels, 1, 1)
        local_descriptor = torch.cat((self._contrast_descriptor(rgb), self._contrast_descriptor(ir)), dim=1)
        feature_local = self.local_gate(local_descriptor).reshape(batch_size, 2, 1, height, width)
        feature_margin = (feature_global[:, 0] - feature_global[:, 1]) + (
            feature_local[:, 0] - feature_local[:, 1]
        )

        prompt = prompt_probabilities.float().clamp_min(1e-6)
        prompt_margin = (prompt[:, 0].log() - prompt[:, 1].log()).to(rgb.dtype).unsqueeze(1)
        gate_margin = prompt_margin + self.rho * feature_margin.tanh() * prompt_margin.abs()
        modal_logits = torch.stack((0.5 * gate_margin, -0.5 * gate_margin), dim=1)
        modal_weights = torch.softmax(modal_logits, dim=1) * 2.0
        fused = modal_weights[:, 0] * rgb + modal_weights[:, 1] * ir
        if self.cross is not None:
            fused = fused + self.cross_scale * self.cross(rgb, ir)
        if self.capture_gate:
            self._last_modal_weights = modal_weights
        return fused

    def pop_modal_weights(self):
        weights = self._last_modal_weights
        self._last_modal_weights = None
        return weights


class P2DualTrueObjectReliabilityLAFMergeFeedback2D(nn.Module):
    """V18 bias-free object Prompt with a sign-preserving modality gate."""

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8, rho=0.5):
        super().__init__()
        self.channels = channels
        self.merge = SignPreservingPromptConditionedLAFMerge2D(
            channels, num_heads, partial_ratio, dilation, gate_reduction, rho
        )
        self.rgb_prompt_head = DetectionObjectReliabilityPromptHead(channels)
        self.ir_prompt_head = DetectionObjectReliabilityPromptHead(channels)
        self.capture_prompt = False
        self._last_aux_outputs = None

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"{self.__class__.__name__} expects [rgb, ir]")
        rgb, ir = x
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"invalid RGB/IR feature shapes: {tuple(rgb.shape)}, {tuple(ir.shape)}")
        prompt_logits = torch.cat((self.rgb_prompt_head(rgb), self.ir_prompt_head(ir)), dim=1)
        prompt_probabilities = prompt_logits.softmax(dim=1)
        self.merge.capture_gate = self.capture_prompt
        fused = self.merge((rgb, ir), prompt_probabilities)
        if self.capture_prompt:
            self._last_aux_outputs = {
                "prompt_logits": prompt_logits,
                "prompt_probabilities": prompt_probabilities,
                "modal_weights": self.merge.pop_modal_weights(),
            }
        correction = fused - (rgb + ir)
        return rgb + correction, ir + correction, fused

    def pop_aux_outputs(self):
        outputs = self._last_aux_outputs
        self._last_aux_outputs = None
        return outputs


P2_RELIABILITY_PROMPT_MODULES = (P2DualReliabilityPriorLAFMergeFeedback2D,)
P2_OBJECT_RELIABILITY_PROMPT_MODULES = (
    P2DualObjectReliabilityMonotonicLAFMergeFeedback2D,
    P2DualTrueObjectReliabilityLAFMergeFeedback2D,
    DA016DualPromptAuxSemanticLAFMergeFeedback2D,
    DA016DualPromptResidualSemanticLAFMergeFeedback2D,
    DarkACTDualPromptAuxLAFMergeFeedback2D,
    DarkACTDualPromptResidualLAFMergeFeedback2D,
)
P2_TRUE_OBJECT_RELIABILITY_PROMPT_MODULES = (P2DualTrueObjectReliabilityLAFMergeFeedback2D,)
DA016_PROMPT_ABLATION_MODULES = (
    DA016DualPromptAuxSemanticLAFMergeFeedback2D,
    DA016DualPromptResidualSemanticLAFMergeFeedback2D,
    DarkACTDualPromptAuxLAFMergeFeedback2D,
    DarkACTDualPromptResidualLAFMergeFeedback2D,
)


__all__ = (
    "DetectionReliabilityPromptHead",
    "DetectionObjectReliabilityPromptHead",
    "DA016DualPromptAuxSemanticLAFMergeFeedback2D",
    "DA016DualPromptResidualSemanticLAFMergeFeedback2D",
    "DarkACTDualPromptAuxLAFMergeFeedback2D",
    "DarkACTDualPromptResidualLAFMergeFeedback2D",
    "PriorConditionedLAFMerge2D",
    "P2DualReliabilityPriorLAFMergeFeedback2D",
    "MonotonicPriorConditionedLAFMerge2D",
    "P2DualObjectReliabilityMonotonicLAFMergeFeedback2D",
    "SignPreservingPromptConditionedLAFMerge2D",
    "P2DualTrueObjectReliabilityLAFMergeFeedback2D",
    "P2_RELIABILITY_PROMPT_MODULES",
    "P2_OBJECT_RELIABILITY_PROMPT_MODULES",
    "P2_TRUE_OBJECT_RELIABILITY_PROMPT_MODULES",
    "DA016_PROMPT_ABLATION_MODULES",
)
