# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Paper-style full-channel DarkAct LAF for paired RGB/thermal images.

This V3 family is intentionally additive. It reuses the existing
``StaticMAA2D`` outputs as explicit motion/saliency-conditioned features
``(Y_rgb, Y_ir)`` and does not change the behavior of any V1/V2 class.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvPReLUBN(nn.Sequential):
    """The lightweight mapping block shown in the DarkAct LAF diagram."""

    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.PReLU(out_channels),
            nn.BatchNorm2d(out_channels),
        )


class _ModalityKey(nn.Module):
    """Equation (5): full-C dilated 3x3 key mapping followed by Sigmoid."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.mapping = _ConvPReLUBN(channels, channels, kernel_size=3, dilation=dilation)

    def forward(self, x):
        return torch.sigmoid(self.mapping(x))


class _SeparateOutputMLP(_ConvPReLUBN):
    """One modality-specific output MLP with a zero-initialized residual."""

    def __init__(self, channels):
        super().__init__(channels, channels, kernel_size=1)
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


class PaperLAFMergeFeedback2D(nn.Module):
    """Full-channel implementation of DarkAct equations (4)-(8).

    Inputs are ``[(Y_rgb, Y_ir), F_rgb, F_ir]`` where ``Y`` is the explicit
    output of the preceding ``StaticMAA2D`` and ``F`` is the corresponding
    CNN stage output. Attention is performed along the complete channel axis,
    matching the paper's ``C x N`` formulation instead of using heads or a
    partial-channel approximation.

    The returned tuple is ``(F_rgb_next, F_ir_next, F_neck)``:

    * ``F_rgb_next = F_rgb + F_LAF``
    * ``F_ir_next = F_ir + F_LAF``
    * ``F_neck = F_rgb + F_ir + F_LAF``

    The two separate output MLPs are zero-initialized, so ``F_LAF`` is exactly
    zero initially. This preserves the migrated two-stream backbone and makes
    the neck output exactly equal to the original ``F_rgb + F_ir`` merge.
    """

    def __init__(self, channels, dilation=1, pool_kernel=3):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        if pool_kernel <= 0 or pool_kernel % 2 == 0:
            raise ValueError("pool_kernel must be a positive odd integer")

        self.channels = channels
        self.pool_kernel = pool_kernel
        self.scale = math.sqrt(channels)

        # Equation (4): phi([Y_rgb * F_rgb; Y_ir * F_ir]) -> shared F_Q.
        self.query_mapping = _ConvPReLUBN(channels * 2, channels, kernel_size=1)

        # Equation (5): independent full-channel keys for RGB and thermal.
        self.key_mapping = nn.ModuleList(
            [_ModalityKey(channels, dilation), _ModalityKey(channels, dilation)]
        )

        # Equation (6): local AP/MP retain HxW; concatenation has 4C channels.
        self.value_mapping = _ConvPReLUBN(channels * 4, channels, kernel_size=1)

        # Equation (8): independent modality MLPs followed by summation.
        self.output_mlp = nn.ModuleList(
            [_SeparateOutputMLP(channels), _SeparateOutputMLP(channels)]
        )

    def _validate_inputs(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise TypeError("PaperLAFMergeFeedback2D expects [(Y_rgb, Y_ir), F_rgb, F_ir]")
        maa_pair, rgb, ir = x
        if not isinstance(maa_pair, (list, tuple)) or len(maa_pair) != 2:
            raise TypeError("the first input must be the (Y_rgb, Y_ir) pair from StaticMAA2D")
        y_rgb, y_ir = maa_pair
        expected = (rgb.shape[0], self.channels, rgb.shape[2], rgb.shape[3])
        tensors = {"Y_rgb": y_rgb, "Y_ir": y_ir, "F_rgb": rgb, "F_ir": ir}
        for name, tensor in tensors.items():
            if tensor.ndim != 4 or tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
                )
        return y_rgb, y_ir, rgb, ir

    def _channel_attention(self, query, key, value):
        """Att(Q, K, V) on full CxN tensors using the paper's sqrt(C) scale.

        Full-channel P3 attention accumulates over N=6400 positions. Under AMP,
        doing these matmuls in FP16 can overflow before the numerically stable
        Softmax is reached. Keep the paper equation unchanged but evaluate its
        reductions and Softmax in FP32, then restore the feature dtype.
        """
        b, c, h, w = query.shape
        output_dtype = query.dtype
        with torch.autocast(device_type=query.device.type, enabled=False):
            query_fp32 = query.float().flatten(2)
            key_fp32 = key.float().flatten(2)
            value_fp32 = value.float().flatten(2)
            attention = torch.matmul(query_fp32, key_fp32.transpose(-2, -1)) / self.scale
            attention = torch.softmax(attention, dim=-1)
            output = torch.matmul(attention, value_fp32).reshape(b, c, h, w)
        return output.to(dtype=output_dtype)

    def _shared_value(self, rgb, ir):
        # Equation (6) begins with bidirectional full-C cross attention.
        cross_rgb = self._channel_attention(rgb, rgb, ir)
        cross_ir = self._channel_attention(ir, ir, rgb)
        cross = torch.cat((cross_rgb, cross_ir), dim=1)

        padding = self.pool_kernel // 2
        pooled = torch.cat(
            (
                F.avg_pool2d(cross, self.pool_kernel, stride=1, padding=padding),
                F.max_pool2d(cross, self.pool_kernel, stride=1, padding=padding),
            ),
            dim=1,
        )
        return self.value_mapping(pooled)

    def forward(self, x):
        y_rgb, y_ir, rgb, ir = self._validate_inputs(x)

        # Equation (4): MAA-conditioned shared light-adaptive query.
        shared_query = self.query_mapping(torch.cat((y_rgb * rgb, y_ir * ir), dim=1))
        rgb_key = self.key_mapping[0](rgb)
        ir_key = self.key_mapping[1](ir)
        shared_value = self._shared_value(rgb, ir)

        # Equations (7)-(8): two dynamic attentions, separate MLPs, then sum.
        dynamic_rgb = self._channel_attention(shared_query, rgb_key, shared_value)
        dynamic_ir = self._channel_attention(shared_query, ir_key, shared_value)
        fused_laf = self.output_mlp[0](dynamic_rgb) + self.output_mlp[1](dynamic_ir)

        return rgb + fused_laf, ir + fused_laf, rgb + ir + fused_laf


__all__ = ("PaperLAFMergeFeedback2D",)
