# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Shared teacher-map utilities for OBB-supervised spatial prompts."""

import torch
import torch.nn.functional as F


def _normalized_grid(height, width, device):
    """Return normalized feature-cell centers in y/x order."""
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    return torch.meshgrid(y, x, indexing="ij")


@torch.no_grad()
def rasterize_rotated_soft_centerness(reference, batch, sigma=0.5):
    """Rasterize normalized xywhr labels as max-composited rotated soft-centerness maps."""
    if not 0 < sigma <= 1:
        raise ValueError("sigma must be in (0, 1]")
    batch_size, _, height, width = reference.shape
    device = reference.device
    target = torch.zeros((batch_size, 1, height, width), device=device, dtype=torch.float32)
    boxes = batch["bboxes"].reshape(-1, 5).to(device=device, dtype=torch.float32)
    batch_idx = batch["batch_idx"].reshape(-1).to(device=device, dtype=torch.long)
    if boxes.numel() == 0:
        return target

    grid_y, grid_x = _normalized_grid(height, width, device)
    epsilon = torch.finfo(torch.float32).eps
    for image_index in range(batch_size):
        image_boxes = boxes[batch_idx == image_index]
        if not image_boxes.numel():
            continue
        image_target = torch.zeros((height, width), device=device, dtype=torch.float32)
        for box_chunk in image_boxes.split(64):
            cx, cy, box_w, box_h, angle = box_chunk.unbind(dim=1)
            dx = grid_x.unsqueeze(0) - cx[:, None, None]
            dy = grid_y.unsqueeze(0) - cy[:, None, None]
            cosine = angle.cos()[:, None, None]
            sine = angle.sin()[:, None, None]
            local_x = cosine * dx + sine * dy
            local_y = -sine * dx + cosine * dy
            nx = 2.0 * local_x / box_w.clamp_min(epsilon)[:, None, None]
            ny = 2.0 * local_y / box_h.clamp_min(epsilon)[:, None, None]
            inside = (nx.abs() <= 1.0) & (ny.abs() <= 1.0)
            centerness = torch.exp(-0.5 * ((nx / sigma).square() + (ny / sigma).square())) * inside
            image_target = torch.maximum(image_target, centerness.amax(dim=0))
        target[image_index, 0] = image_target
    return target


@torch.no_grad()
def build_rgb_reliability_teacher(images, size, exposure_sigma=0.25, smooth=True):
    """Build a deterministic spatial RGB exposure/gradient/contrast reliability teacher."""
    if images.ndim != 4 or images.shape[1] < 3:
        raise ValueError("images must be [B, >=3, H, W] with RGB in the first three channels")
    if exposure_sigma <= 0:
        raise ValueError("exposure_sigma must be positive")

    rgb = images[:, :3].detach().float().clamp(0.0, 1.0)
    luminance = (
        0.299 * rgb[:, 0:1]
        + 0.587 * rgb[:, 1:2]
        + 0.114 * rgb[:, 2:3]
    )
    exposure = torch.exp(-0.5 * ((luminance - 0.5) / exposure_sigma).square())

    sobel_x = luminance.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(luminance, sobel_x, padding=1)
    gy = F.conv2d(luminance, sobel_y, padding=1)
    gradient = torch.sqrt(gx.square() + gy.square() + 1e-12)
    gradient_mean = gradient.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    gradient_reliability = (1.0 - torch.exp(-gradient / gradient_mean)).clamp(0.0, 1.0)

    contrast = (luminance - F.avg_pool2d(luminance, 5, stride=1, padding=2)).abs()
    contrast_mean = contrast.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    contrast_reliability = (1.0 - torch.exp(-contrast / contrast_mean)).clamp(0.0, 1.0)

    target = (0.6 * exposure + 0.2 * gradient_reliability + 0.2 * contrast_reliability).clamp(0.0, 1.0)
    if smooth:
        target = F.avg_pool2d(target, 3, stride=1, padding=1)
    return F.interpolate(target, size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def build_rgb_global_quality_teacher(images, exposure_sigma=0.25):
    """Build the unsmoothed scalar RGB quality target used by P2Det V7/V9/V10."""
    if images.ndim != 4 or images.shape[1] < 3:
        raise ValueError("images must be [B, >=3, H, W] with RGB in the first three channels")
    if exposure_sigma <= 0:
        raise ValueError("exposure_sigma must be positive")

    rgb = images[:, :3].detach().float().clamp(0.0, 1.0)
    luminance = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    exposure = torch.exp(-0.5 * ((luminance - 0.5) / exposure_sigma).square())

    sobel_x = luminance.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(luminance, sobel_x, padding=1)
    gy = F.conv2d(luminance, sobel_y, padding=1)
    gradient = torch.sqrt(gx.square() + gy.square() + 1e-12)
    gradient_mean = gradient.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    gradient_reliability = (1.0 - torch.exp(-gradient / gradient_mean)).clamp(0.0, 1.0)

    contrast = (luminance - F.avg_pool2d(luminance, 5, stride=1, padding=2)).abs()
    contrast_mean = contrast.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    contrast_reliability = (1.0 - torch.exp(-contrast / contrast_mean)).clamp(0.0, 1.0)

    quality_map = (
        0.6 * exposure + 0.2 * gradient_reliability + 0.2 * contrast_reliability
    ).clamp(0.0, 1.0)
    return quality_map.mean(dim=(2, 3), keepdim=True)


__all__ = (
    "rasterize_rotated_soft_centerness",
    "build_rgb_reliability_teacher",
    "build_rgb_global_quality_teacher",
)
