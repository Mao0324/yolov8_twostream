# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Detection-oriented dual prompts and dynamic expert recalibration for RGB/IR LAF fusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darkact_maalaf import LAFMerge2D


class _ChannelLayerNorm2d(nn.Module):
    """Normalize channels independently at every spatial position."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class _P2SpatialPromptHead(nn.Module):
    """Predict a one-channel spatial prompt without global HW-by-HW attention."""

    def __init__(self, channels, initial_probability=0.5):
        super().__init__()
        if channels <= 0 or not 0 < initial_probability < 1:
            raise ValueError("channels must be positive and initial_probability must be in (0, 1)")
        hidden = max(channels // 16, 8)
        self.norm = _ChannelLayerNorm2d(channels)
        self.body = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.constant_(self.body[-1].bias, math.log(initial_probability / (1.0 - initial_probability)))

    def forward(self, x):
        z = self.norm(x)
        contrast3 = (z - F.avg_pool2d(z, 3, stride=1, padding=1)).abs().mean(dim=1, keepdim=True)
        contrast5 = (z - F.avg_pool2d(z, 5, stride=1, padding=2)).abs().mean(dim=1, keepdim=True)
        descriptor = torch.cat(
            (z.mean(dim=1, keepdim=True), z.amax(dim=1, keepdim=True), contrast3, contrast5), dim=1
        )
        return self.body(descriptor)


class _P2PromptEmbedding(nn.Module):
    """Zero-initialized additive projection from a soft prompt to feature space."""

    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Conv2d(1, channels, 1, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, prompt):
        return self.projection(prompt)


class _P2ModalityExpert(nn.Module):
    """Prior-responsive bottleneck expert."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        if reduction <= 0:
            raise ValueError("expert reduction must be positive")
        hidden = max(channels // reduction, 8)
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        nn.init.zeros_(self.body[-1].weight)

    def forward(self, prior_mix):
        return self.body(prior_mix)


class _P2CBAMExpert(nn.Module):
    """Prompt-free channel/spatial attention expert operating only on fused features."""

    def __init__(self, channels, reduction=8, spatial_kernel=5):
        super().__init__()
        if reduction <= 0 or spatial_kernel not in (5, 7):
            raise ValueError("attention reduction must be positive and spatial_kernel must be 5 or 7")
        hidden = max(channels // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.projection = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, fused):
        channel_logits = self.channel_mlp(F.adaptive_avg_pool2d(fused, 1))
        channel_logits = channel_logits + self.channel_mlp(F.adaptive_max_pool2d(fused, 1))
        channel_attention = channel_logits.sigmoid()
        spatial_descriptor = torch.cat(
            (fused.mean(dim=1, keepdim=True), fused.amax(dim=1, keepdim=True)), dim=1
        )
        spatial_attention = self.spatial(spatial_descriptor).sigmoid()
        return self.projection(fused * channel_attention * spatial_attention)


class _P2ExpertGate(nn.Module):
    """Channel-wise softmax gate between modality and attention experts."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max((channels * 2 + 2) // reduction, 8)
        self.body = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        self.channels = channels

    def forward(self, fused, prior_mix, p_rgb, p_ir):
        descriptor = torch.cat(
            (
                F.adaptive_avg_pool2d(fused, 1),
                F.adaptive_avg_pool2d(prior_mix, 1),
                F.adaptive_avg_pool2d(p_rgb, 1),
                F.adaptive_avg_pool2d(p_ir, 1),
            ),
            dim=1,
        )
        logits = self.body(descriptor).reshape(fused.shape[0], 2, self.channels, 1, 1)
        return logits.softmax(dim=1)


class _P2PromptMergeFeedbackBase(nn.Module):
    """Shared prompt/LAF/feedback implementation with optional dual prompt and GDER."""

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        *,
        use_rgb_prompt=False,
        use_gder=False,
        expert_reduction=8,
    ):
        super().__init__()
        self.channels = channels
        self.use_rgb_prompt = bool(use_rgb_prompt)
        self.use_gder = bool(use_gder)

        # Preserve the exact historical parameter prefix model.<layer>.merge.*.
        self.merge = LAFMerge2D(channels, num_heads, partial_ratio, dilation, gate_reduction)
        self.ir_prompt_head = _P2SpatialPromptHead(channels, initial_probability=0.1)
        self.ir_prompt_embed = _P2PromptEmbedding(channels)
        if self.use_rgb_prompt:
            self.rgb_prompt_head = _P2SpatialPromptHead(channels, initial_probability=0.5)
            self.rgb_prompt_embed = _P2PromptEmbedding(channels)

        if self.use_gder:
            if not self.use_rgb_prompt:
                raise ValueError("GDER requires both RGB and IR prompts")
            self.modality_expert = _P2ModalityExpert(channels, expert_reduction)
            self.attention_expert = _P2CBAMExpert(channels, expert_reduction)
            self.expert_gate = _P2ExpertGate(channels, expert_reduction)

        self.capture_prompt = False
        self._last_prompt_logits = None
        self._last_gder_weights = None

    def _validate_inputs(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"{self.__class__.__name__} expects [rgb, ir]")
        rgb, ir = x
        expected = (rgb.shape[0], self.channels, rgb.shape[2], rgb.shape[3])
        if rgb.ndim != 4 or tuple(rgb.shape) != expected or tuple(ir.shape) != expected:
            raise ValueError(f"expected two tensors shaped {expected}, got {tuple(rgb.shape)} and {tuple(ir.shape)}")
        return rgb, ir

    def forward(self, x):
        rgb, ir = self._validate_inputs(x)
        ir_logits = self.ir_prompt_head(ir)
        p_ir = ir_logits.sigmoid()
        ir_prompted = ir + self.ir_prompt_embed(p_ir)

        if self.use_rgb_prompt:
            rgb_logits = self.rgb_prompt_head(rgb)
            p_rgb = rgb_logits.sigmoid()
            rgb_prompted = rgb + self.rgb_prompt_embed(p_rgb)
        else:
            rgb_logits = None
            p_rgb = None
            rgb_prompted = rgb

        fused_base = self.merge((rgb_prompted, ir_prompted))
        fused_refined = fused_base
        if self.use_gder:
            prior_mix = p_rgb * rgb_prompted + p_ir * ir_prompted
            expert_weights = self.expert_gate(fused_base, prior_mix, p_rgb, p_ir)
            modality_feature = self.modality_expert(prior_mix)
            attention_feature = self.attention_expert(fused_base)
            fused_refined = (
                fused_base
                + expert_weights[:, 0] * modality_feature
                + expert_weights[:, 1] * attention_feature
            )
            if self.capture_prompt:
                self._last_gder_weights = expert_weights.detach()

        if self.capture_prompt:
            self._last_prompt_logits = (rgb_logits, ir_logits)

        correction = fused_refined - (rgb + ir)
        return rgb + correction, ir + correction, fused_refined

    def pop_prompt_logits(self):
        """Return and clear prompt logits so an autograd graph is never retained across steps."""
        logits = self._last_prompt_logits
        self._last_prompt_logits = None
        return logits

    def pop_gder_weights(self):
        """Return and clear detached dynamic-expert weights for diagnostics."""
        weights = self._last_gder_weights
        self._last_gder_weights = None
        return weights


class P2IRPromptLAFMergeFeedback2D(_P2PromptMergeFeedbackBase):
    """IR target prompt plus the existing LAF merge and feedback."""

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8):
        super().__init__(channels, num_heads, partial_ratio, dilation, gate_reduction)


class P2DualPromptLAFMergeFeedback2D(_P2PromptMergeFeedbackBase):
    """RGB reliability and IR target prompts plus the existing LAF merge and feedback."""

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8):
        super().__init__(
            channels,
            num_heads,
            partial_ratio,
            dilation,
            gate_reduction,
            use_rgb_prompt=True,
        )


class P2DualPromptGDERMergeFeedback2D(_P2PromptMergeFeedbackBase):
    """Dual prompts, existing LAF, two specialized experts, dynamic gate, and feedback."""

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        expert_reduction=8,
    ):
        super().__init__(
            channels,
            num_heads,
            partial_ratio,
            dilation,
            gate_reduction,
            use_rgb_prompt=True,
            use_gder=True,
            expert_reduction=expert_reduction,
        )


class _P2RGBGlobalQualityHead(nn.Module):
    """Predict one image-level RGB quality logit from normalized channel statistics."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 16, 8)
        self.norm = _ChannelLayerNorm2d(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )

    def forward(self, x):
        z = self.norm(x)
        descriptor = torch.cat(
            (F.adaptive_avg_pool2d(z, 1), F.adaptive_max_pool2d(z, 1)), dim=1
        )
        return self.body(descriptor)


class _P2IdentityPreservingModalityExpert(nn.Module):
    """Keep RGB/IR identities separate until the learned bottleneck fusion."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        if reduction <= 0:
            raise ValueError("expert reduction must be positive")
        hidden = max(channels // reduction, 8)
        self.rgb_norm = _ChannelLayerNorm2d(channels)
        self.ir_norm = _ChannelLayerNorm2d(channels)
        self.rgb_projection = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.ir_projection = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1, bias=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, rgb, ir, p_rgb, p_ir):
        rgb_feature = self.rgb_projection(p_rgb * self.rgb_norm(rgb))
        ir_feature = self.ir_projection(p_ir * self.ir_norm(ir))
        return self.fusion(torch.cat((rgb_feature, ir_feature), dim=1))


class _P2FactorizedExpertGate(nn.Module):
    """V10 channel gate plus a zero-opened P4 spatial residual gate."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max((channels * 2 + 2) // reduction, 8)
        self.body = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        spatial_hidden = max(channels // 32, 8)
        self.spatial_body = nn.Sequential(
            nn.Conv2d(4, spatial_hidden, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                spatial_hidden,
                spatial_hidden,
                3,
                padding=1,
                groups=spatial_hidden,
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(spatial_hidden, 2, 1, bias=True),
        )
        self.gamma_spatial = nn.Parameter(torch.zeros(()))
        self.channels = channels
        self.capture_diagnostics = False
        self._last_diagnostics = None

    @staticmethod
    def _channel_descriptor(fused, prior_mix, p_rgb, p_ir):
        return torch.cat(
            (
                F.adaptive_avg_pool2d(fused, 1),
                F.adaptive_avg_pool2d(prior_mix, 1),
                F.adaptive_avg_pool2d(p_rgb, 1),
                F.adaptive_avg_pool2d(p_ir, 1),
            ),
            dim=1,
        )

    def forward(self, fused, prior_mix, p_rgb, p_ir):
        channel_logits = self.body(
            self._channel_descriptor(fused, prior_mix, p_rgb, p_ir)
        ).reshape(fused.shape[0], 2, self.channels, 1, 1)
        spatial_descriptor = torch.cat(
            (
                fused.mean(dim=1, keepdim=True),
                fused.amax(dim=1, keepdim=True),
                p_rgb,
                p_ir,
            ),
            dim=1,
        )
        spatial_logits = self.spatial_body(spatial_descriptor).unsqueeze(2)
        weights = (channel_logits + self.gamma_spatial * spatial_logits).softmax(dim=1)
        if self.capture_diagnostics:
            self._last_diagnostics = {
                "channel_logits_shape": tuple(channel_logits.shape),
                "spatial_logits_shape": tuple(spatial_logits.shape),
                "spatial_logits_std": spatial_logits.detach().float().std(unbiased=False),
                "gamma_spatial": self.gamma_spatial.detach().float(),
                "final_weight_std": weights.detach().float().std(unbiased=False),
            }
        return weights

    def pop_diagnostics(self):
        diagnostics = self._last_diagnostics
        self._last_diagnostics = None
        return diagnostics


class _P2SecondGenPromptMergeFeedbackBase(nn.Module):
    """V7-V10 prompt/LAF/GDER implementation isolated from historical V1-V6."""

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        *,
        use_rgb_spatial=False,
        use_rgb_global=False,
        use_gder=False,
        factorized_gate=False,
        expert_reduction=8,
    ):
        super().__init__()
        self.channels = channels
        self.use_rgb_spatial = bool(use_rgb_spatial)
        self.use_rgb_global = bool(use_rgb_global)
        self.use_gder = bool(use_gder)
        self.factorized_gate = bool(factorized_gate)

        # Keep the historical model.<layer>.merge.* prefix for maximum transfer.
        self.merge = LAFMerge2D(channels, num_heads, partial_ratio, dilation, gate_reduction)
        self.ir_prompt_head = _P2SpatialPromptHead(channels, initial_probability=0.1)
        self.ir_prompt_embed = _P2PromptEmbedding(channels)
        if self.use_rgb_spatial:
            self.rgb_prompt_head = _P2SpatialPromptHead(channels, initial_probability=0.5)
            self.rgb_prompt_embed = _P2PromptEmbedding(channels)
        if self.use_rgb_global:
            self.rgb_global_head = _P2RGBGlobalQualityHead(channels)
            self.rgb_global_embed = _P2PromptEmbedding(channels)

        if self.use_gder:
            if not self.use_rgb_spatial:
                raise ValueError("identity-preserving GDER requires RGB and IR spatial prompts")
            self.modality_expert = _P2IdentityPreservingModalityExpert(channels, expert_reduction)
            self.attention_expert = _P2CBAMExpert(channels, expert_reduction)
            gate_class = _P2FactorizedExpertGate if self.factorized_gate else _P2ExpertGate
            self.expert_gate = gate_class(channels, expert_reduction)

        self.capture_prompt = False
        self._last_aux_outputs = None
        self._last_gder_weights = None
        self._last_gder_diagnostics = None

    def _validate_inputs(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"{self.__class__.__name__} expects [rgb, ir]")
        rgb, ir = x
        if rgb.ndim != 4 or ir.ndim != 4:
            raise ValueError("RGB and IR inputs must both be four-dimensional")
        expected = (rgb.shape[0], self.channels, rgb.shape[2], rgb.shape[3])
        if tuple(rgb.shape) != expected or tuple(ir.shape) != expected:
            raise ValueError(f"expected two tensors shaped {expected}, got {tuple(rgb.shape)} and {tuple(ir.shape)}")
        return rgb, ir

    def forward(self, x):
        rgb, ir = self._validate_inputs(x)
        ir_spatial_logits = self.ir_prompt_head(ir)
        p_ir = ir_spatial_logits.sigmoid()
        ir_prompted = ir + self.ir_prompt_embed(p_ir)

        rgb_spatial_logits = None
        p_rgb = None
        rgb_prompted = rgb
        if self.use_rgb_spatial:
            rgb_spatial_logits = self.rgb_prompt_head(rgb)
            p_rgb = rgb_spatial_logits.sigmoid()
            rgb_prompted = rgb_prompted + self.rgb_prompt_embed(p_rgb)

        rgb_global_logits = None
        if self.use_rgb_global:
            rgb_global_logits = self.rgb_global_head(rgb)
            rgb_prompted = rgb_prompted + self.rgb_global_embed(rgb_global_logits.sigmoid())

        fused_base = self.merge((rgb_prompted, ir_prompted))
        fused_refined = fused_base
        if self.use_gder:
            # Preserve V6/V9 channel-gate inputs. The expert itself never receives
            # this summed prior and therefore retains modality identity.
            prior_mix = p_rgb * rgb_prompted + p_ir * ir_prompted
            if self.factorized_gate:
                self.expert_gate.capture_diagnostics = self.capture_prompt
            expert_weights = self.expert_gate(fused_base, prior_mix, p_rgb, p_ir)
            modality_feature = self.modality_expert(rgb_prompted, ir_prompted, p_rgb, p_ir)
            attention_feature = self.attention_expert(fused_base)
            fused_refined = (
                fused_base
                + expert_weights[:, 0] * modality_feature
                + expert_weights[:, 1] * attention_feature
            )
            if self.capture_prompt:
                self._last_gder_weights = expert_weights.detach()
                self._last_gder_diagnostics = {
                    "gate_shape": tuple(expert_weights.shape),
                    "modality_feature_shape": tuple(modality_feature.shape),
                    "attention_feature_shape": tuple(attention_feature.shape),
                    "factorized": self.factorized_gate,
                }
                if self.factorized_gate:
                    self._last_gder_diagnostics.update(self.expert_gate.pop_diagnostics() or {})

        if self.capture_prompt:
            self._last_aux_outputs = {
                "ir_spatial_logits": ir_spatial_logits,
                "rgb_spatial_logits": rgb_spatial_logits,
                "rgb_global_logit": rgb_global_logits,
            }

        correction = fused_refined - (rgb + ir)
        return rgb + correction, ir + correction, fused_refined

    def pop_aux_outputs(self):
        """Return and clear V7-V10 auxiliary outputs without retaining stale graphs."""
        outputs = self._last_aux_outputs
        self._last_aux_outputs = None
        return outputs

    def pop_gder_weights(self):
        weights = self._last_gder_weights
        self._last_gder_weights = None
        return weights

    def pop_gder_diagnostics(self):
        diagnostics = self._last_gder_diagnostics
        self._last_gder_diagnostics = None
        return diagnostics


class P2IRSpatialRGBGlobalLAFMergeFeedback2D(_P2SecondGenPromptMergeFeedbackBase):
    """V7 IR spatial prompt plus RGB global quality prompt, without RGB spatial prompt."""

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8):
        super().__init__(
            channels,
            num_heads,
            partial_ratio,
            dilation,
            gate_reduction,
            use_rgb_global=True,
        )


class P2DualPromptIdentityGDERMergeFeedback2D(_P2SecondGenPromptMergeFeedbackBase):
    """V8 dual spatial prompts plus identity-preserving channel-gated GDER."""

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        expert_reduction=8,
    ):
        super().__init__(
            channels,
            num_heads,
            partial_ratio,
            dilation,
            gate_reduction,
            use_rgb_spatial=True,
            use_gder=True,
            expert_reduction=expert_reduction,
        )


class P2DualSpatialLAFMergeFeedback2D(_P2SecondGenPromptMergeFeedbackBase):
    """V8 P3 dual spatial prompts and LAF without GDER or a global prompt."""

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8):
        super().__init__(
            channels,
            num_heads,
            partial_ratio,
            dilation,
            gate_reduction,
            use_rgb_spatial=True,
        )


class P2DualPromptRGBGlobalLAFMergeFeedback2D(_P2SecondGenPromptMergeFeedbackBase):
    """V9/V10 P3 dual spatial prompts plus an independent RGB global prompt."""

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8):
        super().__init__(
            channels,
            num_heads,
            partial_ratio,
            dilation,
            gate_reduction,
            use_rgb_spatial=True,
            use_rgb_global=True,
        )


class P2DualPromptRGBGlobalIdentityGDERMergeFeedback2D(_P2SecondGenPromptMergeFeedbackBase):
    """V9/V10 dual spatial/global prompts plus channel-only identity GDER."""

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        expert_reduction=8,
    ):
        super().__init__(
            channels,
            num_heads,
            partial_ratio,
            dilation,
            gate_reduction,
            use_rgb_spatial=True,
            use_rgb_global=True,
            use_gder=True,
            expert_reduction=expert_reduction,
        )


class P2DualPromptRGBGlobalIdentityGDERFactorizedMergeFeedback2D(
    _P2SecondGenPromptMergeFeedbackBase
):
    """V10 P4 dual prompts plus factorized channel/spatial identity GDER."""

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        expert_reduction=8,
    ):
        super().__init__(
            channels,
            num_heads,
            partial_ratio,
            dilation,
            gate_reduction,
            use_rgb_spatial=True,
            use_rgb_global=True,
            use_gder=True,
            factorized_gate=True,
            expert_reduction=expert_reduction,
        )


class _P2AsymmetricIdentityModalityExpert(nn.Module):
    """Raw-RGB and IR-target-prior branches that retain modality identity until fusion."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        if reduction <= 0:
            raise ValueError("expert reduction must be positive")
        hidden = max(channels // reduction, 8)
        self.rgb_norm = _ChannelLayerNorm2d(channels)
        self.ir_norm = _ChannelLayerNorm2d(channels)
        self.rgb_projection = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.ir_projection = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1, bias=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.capture_diagnostics = False
        self._last_diagnostics = None

    def forward(self, rgb, ir, p_ir):
        rgb_feature = self.rgb_projection(self.rgb_norm(rgb))
        ir_feature = self.ir_projection(p_ir * self.ir_norm(ir))
        concatenated = torch.cat((rgb_feature, ir_feature), dim=1)
        output = self.fusion(concatenated)
        if self.capture_diagnostics:
            self._last_diagnostics = {
                "rgb_branch_shape": tuple(rgb_feature.shape),
                "ir_branch_shape": tuple(ir_feature.shape),
                "concat_shape": tuple(concatenated.shape),
                "modality_feature_shape": tuple(output.shape),
            }
        return output

    def pop_diagnostics(self):
        diagnostics = self._last_diagnostics
        self._last_diagnostics = None
        return diagnostics


class _P2AsymmetricIRPromptMergeFeedbackBase(nn.Module):
    """V11-V13 IR-only prompt/LAF path with optional asymmetric identity GDER."""

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        *,
        use_gder=False,
        expert_reduction=8,
    ):
        super().__init__()
        self.channels = channels
        self.use_rgb_spatial = False
        self.use_rgb_global = False
        self.use_gder = bool(use_gder)
        self.factorized_gate = False
        self.asymmetric_gder = self.use_gder

        self.merge = LAFMerge2D(channels, num_heads, partial_ratio, dilation, gate_reduction)
        self.ir_prompt_head = _P2SpatialPromptHead(channels, initial_probability=0.1)
        self.ir_prompt_embed = _P2PromptEmbedding(channels)
        if self.use_gder:
            self.modality_expert = _P2AsymmetricIdentityModalityExpert(channels, expert_reduction)
            self.attention_expert = _P2CBAMExpert(channels, expert_reduction)
            self.expert_gate = _P2ExpertGate(channels, expert_reduction)

        self.capture_prompt = False
        self._last_aux_outputs = None
        self._last_gder_weights = None
        self._last_gder_diagnostics = None

    def _validate_inputs(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"{self.__class__.__name__} expects [rgb, ir]")
        rgb, ir = x
        expected = (rgb.shape[0], self.channels, rgb.shape[2], rgb.shape[3])
        if rgb.ndim != 4 or tuple(rgb.shape) != expected or tuple(ir.shape) != expected:
            raise ValueError(f"expected two tensors shaped {expected}, got {tuple(rgb.shape)} and {tuple(ir.shape)}")
        return rgb, ir

    def forward(self, x):
        rgb, ir = self._validate_inputs(x)
        ir_spatial_logits = self.ir_prompt_head(ir)
        p_ir = ir_spatial_logits.sigmoid()
        ir_prompted = ir + self.ir_prompt_embed(p_ir)

        fused_base = self.merge((rgb, ir_prompted))
        fused_refined = fused_base
        if self.use_gder:
            # This composite is used only by the unchanged-shape V8 channel
            # gate. The modality expert receives separate RGB/IR tensors.
            gate_prior_mix = rgb + p_ir * ir_prompted
            absent_rgb_prior_slot = torch.zeros_like(p_ir)
            expert_weights = self.expert_gate(
                fused_base, gate_prior_mix, absent_rgb_prior_slot, p_ir
            )
            self.modality_expert.capture_diagnostics = self.capture_prompt
            modality_feature = self.modality_expert(rgb, ir_prompted, p_ir)
            attention_feature = self.attention_expert(fused_base)
            fused_refined = (
                fused_base
                + expert_weights[:, 0] * modality_feature
                + expert_weights[:, 1] * attention_feature
            )
            if self.capture_prompt:
                self._last_gder_weights = expert_weights.detach()
                self._last_gder_diagnostics = {
                    "gate_shape": tuple(expert_weights.shape),
                    "attention_feature_shape": tuple(attention_feature.shape),
                    "factorized": False,
                    "asymmetric": True,
                    **(self.modality_expert.pop_diagnostics() or {}),
                }

        if self.capture_prompt:
            self._last_aux_outputs = {
                "ir_spatial_logits": ir_spatial_logits,
                "rgb_spatial_logits": None,
                "rgb_global_logit": None,
            }

        correction = fused_refined - (rgb + ir)
        return rgb + correction, ir + correction, fused_refined

    def pop_aux_outputs(self):
        outputs = self._last_aux_outputs
        self._last_aux_outputs = None
        return outputs

    def pop_gder_weights(self):
        weights = self._last_gder_weights
        self._last_gder_weights = None
        return weights

    def pop_gder_diagnostics(self):
        diagnostics = self._last_gder_diagnostics
        self._last_gder_diagnostics = None
        return diagnostics


class P2IRPromptLAFMergeFeedbackNoStaticMAA2D(_P2AsymmetricIRPromptMergeFeedbackBase):
    """V11/V13 raw RGB, IR spatial prompt, LAF, and feedback without GDER."""

    def __init__(self, channels, num_heads=4, partial_ratio=4, dilation=1, gate_reduction=8):
        super().__init__(channels, num_heads, partial_ratio, dilation, gate_reduction)


class P2IRPromptAsymIdentityGDERMergeFeedback2D(_P2AsymmetricIRPromptMergeFeedbackBase):
    """V12/V13 IR prompt plus raw-RGB/IR-prior asymmetric identity GDER."""

    def __init__(
        self,
        channels,
        num_heads=4,
        partial_ratio=4,
        dilation=1,
        gate_reduction=8,
        expert_reduction=8,
    ):
        super().__init__(
            channels,
            num_heads,
            partial_ratio,
            dilation,
            gate_reduction,
            use_gder=True,
            expert_reduction=expert_reduction,
        )


P2_PROMPT_MODULES = (
    P2IRPromptLAFMergeFeedback2D,
    P2DualPromptLAFMergeFeedback2D,
    P2DualPromptGDERMergeFeedback2D,
)

P2_SECOND_GEN_PROMPT_MODULES = (
    P2IRSpatialRGBGlobalLAFMergeFeedback2D,
    P2DualSpatialLAFMergeFeedback2D,
    P2DualPromptIdentityGDERMergeFeedback2D,
    P2DualPromptRGBGlobalLAFMergeFeedback2D,
    P2DualPromptRGBGlobalIdentityGDERMergeFeedback2D,
    P2DualPromptRGBGlobalIdentityGDERFactorizedMergeFeedback2D,
    P2IRPromptLAFMergeFeedbackNoStaticMAA2D,
    P2IRPromptAsymIdentityGDERMergeFeedback2D,
)

__all__ = (
    "P2IRPromptLAFMergeFeedback2D",
    "P2DualPromptLAFMergeFeedback2D",
    "P2DualPromptGDERMergeFeedback2D",
    "P2_PROMPT_MODULES",
    "P2IRSpatialRGBGlobalLAFMergeFeedback2D",
    "P2DualSpatialLAFMergeFeedback2D",
    "P2DualPromptIdentityGDERMergeFeedback2D",
    "P2DualPromptRGBGlobalLAFMergeFeedback2D",
    "P2DualPromptRGBGlobalIdentityGDERMergeFeedback2D",
    "P2DualPromptRGBGlobalIdentityGDERFactorizedMergeFeedback2D",
    "P2_SECOND_GEN_PROMPT_MODULES",
    "P2IRPromptLAFMergeFeedbackNoStaticMAA2D",
    "P2IRPromptAsymIdentityGDERMergeFeedback2D",
)
