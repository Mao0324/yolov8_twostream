# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Independent OBB trainer with target-saliency mask supervision."""

from copy import copy

import torch
import torch.nn.functional as F

from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.nn.modules import StaticMAAContext2D
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8OBBLoss


class TargetSaliencyOBBLoss(v8OBBLoss):
    """Add lightweight BCE+Dice supervision from normalized OBBs to S maps."""

    def __init__(self, model):
        super().__init__(model)
        self.context_modules = [m for m in model.modules() if isinstance(m, StaticMAAContext2D)]
        if not self.context_modules:
            raise ValueError("TargetSaliencyOBBLoss requires at least one StaticMAAContext2D")

        yaml = model.yaml
        self.saliency_gain = float(yaml.get("target_saliency_loss_gain", 0.05))
        self.mask_tolerance = float(yaml.get("target_saliency_mask_tolerance", 0.5))
        stage_weights = yaml.get("target_saliency_stage_weights", [1.0] * len(self.context_modules))
        if len(stage_weights) != len(self.context_modules):
            raise ValueError(
                "target_saliency_stage_weights must match the number of StaticMAAContext2D stages"
            )
        self.stage_weights = tuple(float(weight) for weight in stage_weights)
        if self.saliency_gain < 0 or self.mask_tolerance < 0 or sum(self.stage_weights) <= 0:
            raise ValueError("target saliency loss settings must be non-negative with a positive weight sum")

    @staticmethod
    def _normalized_grid(height, width, device):
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
        return torch.meshgrid(y, x, indexing="ij")

    def _rasterize_obb_masks(self, logits, batch):
        """Rasterize normalized xywhr labels directly at one feature-map resolution."""
        batch_size, _, height, width = logits.shape
        device = logits.device
        masks = torch.zeros((batch_size, 1, height, width), device=device, dtype=torch.float32)
        boxes = batch["bboxes"].view(-1, 5).to(device=device, dtype=torch.float32)
        batch_idx = batch["batch_idx"].view(-1).to(device=device, dtype=torch.long)
        if boxes.numel() == 0:
            return masks

        grid_y, grid_x = self._normalized_grid(height, width, device)
        expand_x = self.mask_tolerance / width
        expand_y = self.mask_tolerance / height
        with torch.no_grad():
            for image_index in range(batch_size):
                image_boxes = boxes[batch_idx == image_index]
                if not image_boxes.numel():
                    continue
                image_mask = torch.zeros((height, width), device=device, dtype=torch.bool)
                # Chunking bounds peak memory on crowded DroneVehicle frames.
                for box_chunk in image_boxes.split(64):
                    cx, cy, box_w, box_h, angle = box_chunk.unbind(dim=1)
                    dx = grid_x.unsqueeze(0) - cx[:, None, None]
                    dy = grid_y.unsqueeze(0) - cy[:, None, None]
                    cosine = angle.cos()[:, None, None]
                    sine = angle.sin()[:, None, None]
                    local_x = cosine * dx + sine * dy
                    local_y = -sine * dx + cosine * dy
                    inside = (local_x.abs() <= box_w[:, None, None] * 0.5 + expand_x) & (
                        local_y.abs() <= box_h[:, None, None] * 0.5 + expand_y
                    )
                    image_mask |= inside.any(dim=0)
                masks[image_index, 0] = image_mask
        return masks

    def _one_stage_loss(self, logits, batch):
        target = self._rasterize_obb_masks(logits, batch).expand(-1, 2, -1, -1)
        logits_fp32 = logits.float()

        # Balance sparse OBB foreground without allowing extreme small-object weights.
        foreground = target.mean().clamp_min(1e-6)
        positive_weight = ((1.0 - foreground) / foreground).clamp(min=1.0, max=20.0)
        elementwise_bce = F.binary_cross_entropy_with_logits(logits_fp32, target, reduction="none")
        bce_weight = torch.where(target > 0.5, positive_weight, torch.ones_like(target))
        bce = (elementwise_bce * bce_weight).mean()

        probability = logits_fp32.sigmoid()
        reduce_dims = (2, 3)
        intersection = (probability * target).sum(dim=reduce_dims)
        dice = 1.0 - (2.0 * intersection + 1.0) / (
            probability.sum(dim=reduce_dims) + target.sum(dim=reduce_dims) + 1.0
        )
        return bce + dice.mean()

    def _pop_saliency_logits(self):
        logits = [module.pop_saliency_logits() for module in self.context_modules]
        if any(stage_logits is None for stage_logits in logits):
            raise RuntimeError(
                "target saliency logits were not captured; use TargetSaliencyOBBModel for training and validation"
            )
        return logits

    def __call__(self, preds, batch):
        base_total, base_items = super().__call__(preds, batch)
        stage_logits = self._pop_saliency_logits()
        weighted_loss = sum(
            weight * self._one_stage_loss(logits, batch)
            for weight, logits in zip(self.stage_weights, stage_logits)
        ) / sum(self.stage_weights)
        saliency_loss = self.saliency_gain * weighted_loss
        batch_size = batch["img"].shape[0]
        total_loss = base_total + saliency_loss * batch_size
        loss_items = torch.cat((base_items, saliency_loss.detach().reshape(1)))
        return total_loss, loss_items


class TargetSaliencyOBBModel(OBBModel):
    """OBB model that captures StaticMAAContext2D logits after stride setup."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        for module in self.modules():
            if isinstance(module, StaticMAAContext2D):
                module.capture_saliency = True
                module._last_saliency_logits = None

    def init_criterion(self):
        return TargetSaliencyOBBLoss(self)


class TargetSaliencyOBBTrainer(OBBTrainer):
    """OBB trainer that reports the auxiliary target-saliency loss separately."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = TargetSaliencyOBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "saliency_loss"
        return yolo.obb.OBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


__all__ = ("TargetSaliencyOBBLoss", "TargetSaliencyOBBModel", "TargetSaliencyOBBTrainer")
