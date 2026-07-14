# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Adaptive sparse cross-modal attention for RGB-IR feature fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm2d(nn.Module):
    """Layer-normalize the channel vector at every spatial position."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class DynamicDepthwiseConv2d(nn.Module):
    """Pure-PyTorch pixel-wise dynamic depth-wise convolution.

    ASSANet generates one KxK depth-wise kernel for every channel and spatial
    location. Generating and applying one kernel offset at a time avoids the
    KxK feature expansion of ``unfold`` while remaining portable to CPU, CUDA
    and model export without ASSANet's CuPy-only custom operator.
    """

    def __init__(self, channels, kernel_size=3, bias=False):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("dynamic kernel size must be odd")
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.weight_generators = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=bias,
            )
            for _ in range(kernel_size * kernel_size)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError("expected {} channels, got {}".format(self.channels, c))

        padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        out = torch.zeros_like(x)
        index = 0
        for row in range(self.kernel_size):
            for column in range(self.kernel_size):
                shifted = padded[:, :, row : row + h, column : column + w]
                out = out + self.weight_generators[index](x) * shifted
                index += 1
        return out


class LocalQKVProjection(nn.Module):
    """ASSA local spatial-variant projection with shared K/V features."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.project_in = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.dynamic_conv = DynamicDepthwiseConv2d(channels * 2, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        # As in ASSANet, the second half is shared by K and V.
        return self.dynamic_conv(self.project_in(x)).chunk(2, dim=1)


class GatedFeedForward(nn.Module):
    """Gated depth-wise feed-forward network used after ASSA."""

    def __init__(self, channels, expansion=2.0):
        super().__init__()
        hidden = max(int(channels * expansion), 1)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden * 2, bias=False
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.depthwise(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(x1 * F.gelu(x2))


class ASSAFusion(nn.Module):
    """Bidirectional ASSA fusion for an RGB/IR feature pair.

    Input and output are channel-concatenated ``[RGB, IR]`` tensors of shape
    ``[B, 2C, H, W]``. Internally, RGB queries attend to IR keys/values and IR
    queries attend to RGB keys/values. Both results are residual updates, so
    each modality keeps its own feature stream after cross-modal interaction.

    This adapts ASSANet's transposed adaptive sparse self-attention from
    self-attention to symmetric cross-attention. ReLU is deliberately applied
    directly to the channel attention matrix, matching the paper's sparse
    selection and the official implementation (rather than softmax smoothing).
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        dynamic_kernel=3,
        ffn_expansion=2.0,
        residual_scale=0.1,
    ):
        super().__init__()
        if channels <= 0 or num_heads <= 0:
            raise ValueError("channels and num_heads must be positive")
        if channels % num_heads != 0:
            raise ValueError("channels ({}) must be divisible by num_heads ({})".format(channels, num_heads))

        self.channels = channels
        self.num_heads = num_heads
        self.norm_qkv = nn.ModuleList([ChannelLayerNorm2d(channels), ChannelLayerNorm2d(channels)])
        self.qkv = nn.ModuleList(
            [LocalQKVProjection(channels, dynamic_kernel), LocalQKVProjection(channels, dynamic_kernel)]
        )
        self.temperature = nn.Parameter(torch.ones(2, num_heads, 1, 1))
        self.project_out = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=1, bias=False) for _ in range(2)]
        )

        self.norm_ffn = nn.ModuleList([ChannelLayerNorm2d(channels), ChannelLayerNorm2d(channels)])
        self.ffn = nn.ModuleList(
            [GatedFeedForward(channels, ffn_expansion), GatedFeedForward(channels, ffn_expansion)]
        )
        self.attn_scale = nn.Parameter(torch.full((2, 1, 1, 1), float(residual_scale)))
        self.ffn_scale = nn.Parameter(torch.full((2, 1, 1, 1), float(residual_scale)))

    def _reshape_heads(self, x):
        b, c, h, w = x.shape
        return x.reshape(b, self.num_heads, c // self.num_heads, h * w)

    def _cross_attention(self, query, key_value, direction):
        q = F.normalize(self._reshape_heads(query), dim=-1)
        k = F.normalize(self._reshape_heads(key_value), dim=-1)

        # Transposed attention: [B, heads, C/head, C/head], independent of H*W.
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.temperature[direction]
        attention = F.relu(attention)
        return torch.matmul(attention, self._reshape_heads(key_value))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels * 2:
            raise ValueError(
                "ASSAFusion expects [B, {}, H, W], got {}".format(self.channels * 2, tuple(x.shape))
            )

        rgb, ir = x.chunk(2, dim=1)
        q_rgb, kv_rgb = self.qkv[0](self.norm_qkv[0](rgb))
        q_ir, kv_ir = self.qkv[1](self.norm_qkv[1](ir))

        rgb_from_ir = self._cross_attention(q_rgb, kv_ir, direction=0)
        ir_from_rgb = self._cross_attention(q_ir, kv_rgb, direction=1)
        b, _, h, w = rgb.shape
        rgb_from_ir = rgb_from_ir.reshape(b, self.channels, h, w)
        ir_from_rgb = ir_from_rgb.reshape(b, self.channels, h, w)

        rgb = rgb + self.attn_scale[0] * self.project_out[0](rgb_from_ir)
        ir = ir + self.attn_scale[1] * self.project_out[1](ir_from_rgb)
        rgb = rgb + self.ffn_scale[0] * self.ffn[0](self.norm_ffn[0](rgb))
        ir = ir + self.ffn_scale[1] * self.ffn[1](self.norm_ffn[1](ir))
        return torch.cat((rgb, ir), dim=1)


__all__ = ("ASSAFusion",)
