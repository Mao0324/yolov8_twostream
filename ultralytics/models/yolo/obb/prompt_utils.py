# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Shared teacher-map utilities for OBB-supervised spatial prompts."""

import torch


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


__all__ = ("rasterize_rotated_soft_centerness",)
