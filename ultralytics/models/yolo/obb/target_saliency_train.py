# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Independent OBB trainer with target-saliency mask supervision."""

from copy import copy
import weakref

import torch
import torch.nn.functional as F

from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.nn.modules import StaticMAAContext2D
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.torch_utils import de_parallel


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


class SoftCenternessTargetSaliencyOBBLoss(TargetSaliencyOBBLoss):
    """Supervise P3/P4 with rotated soft-centerness instead of hard OBB masks.

    P5 logits are still consumed by the fusion module and popped after every
    forward, but they do not receive an auxiliary saliency loss. This keeps the
    DA-010 architecture fixed and isolates the supervision-policy change.
    """

    def __init__(self, model):
        # Do not call TargetSaliencyOBBLoss.__init__: DA-014 deliberately has
        # two supervised stage weights for three context modules.
        v8OBBLoss.__init__(self, model)
        self._model_ref = weakref.ref(model)
        self.context_modules = [m for m in model.modules() if isinstance(m, StaticMAAContext2D)]
        if len(self.context_modules) < 2:
            raise ValueError("SoftCenternessTargetSaliencyOBBLoss requires at least P3/P4 context modules")

        yaml = model.yaml
        configured_stages = tuple(str(stage).upper() for stage in yaml.get(
            "target_saliency_supervised_stages", ["P3", "P4"]
        ))
        stage_to_index = {f"P{index + 3}": index for index in range(len(self.context_modules))}
        if not configured_stages or any(stage not in stage_to_index for stage in configured_stages):
            raise ValueError(
                f"invalid target_saliency_supervised_stages={configured_stages}; "
                f"available={tuple(stage_to_index)}"
            )
        self.supervised_stage_names = configured_stages
        self.supervised_stage_indices = tuple(stage_to_index[stage] for stage in configured_stages)

        stage_weights = yaml.get("target_saliency_stage_weights", [1.0, 0.5])
        if len(stage_weights) != len(self.supervised_stage_indices):
            raise ValueError("target_saliency_stage_weights must match supervised stages")
        self.stage_weights = tuple(float(weight) for weight in stage_weights)
        if any(weight < 0 for weight in self.stage_weights) or sum(self.stage_weights) <= 0:
            raise ValueError("target_saliency_stage_weights must be non-negative with a positive sum")

        self.saliency_gain = float(yaml.get("target_saliency_loss_gain", 0.025))
        self.gain_warmup_epochs = int(yaml.get("target_saliency_gain_warmup_epochs", 10))
        self.soft_sigma = float(yaml.get("target_saliency_soft_sigma", 0.5))
        if self.saliency_gain < 0 or self.gain_warmup_epochs < 0:
            raise ValueError("saliency gain and warmup epochs must be non-negative")
        if not 0 < self.soft_sigma <= 1:
            raise ValueError("target_saliency_soft_sigma must be in (0, 1]")

        # Per-epoch detached statistics are consumed by the trainer and written
        # as extra results.csv columns. They never participate in backprop.
        self._stat_sums = {}
        self._stat_counts = {}

    @property
    def current_gain(self):
        """Linearly warm the auxiliary gain from zero to the configured maximum."""
        model = self._model_ref()
        epoch = int(getattr(model, "target_saliency_epoch", self.gain_warmup_epochs))
        if self.gain_warmup_epochs == 0:
            return self.saliency_gain
        return self.saliency_gain * min(max(epoch / self.gain_warmup_epochs, 0.0), 1.0)

    def _rasterize_obb_masks(self, logits, batch):
        """Rasterize a rotated Gaussian-like centerness target inside each OBB.

        Coordinates are normalized by each rotated box half-width/half-height.
        The heatmap is one at the box center, decays smoothly toward its edges,
        is zero outside the OBB, and crowded boxes are merged with pointwise max.
        """
        batch_size, _, height, width = logits.shape
        device = logits.device
        masks = torch.zeros((batch_size, 1, height, width), device=device, dtype=torch.float32)
        boxes = batch["bboxes"].view(-1, 5).to(device=device, dtype=torch.float32)
        batch_idx = batch["batch_idx"].view(-1).to(device=device, dtype=torch.long)
        if boxes.numel() == 0:
            return masks

        grid_y, grid_x = self._normalized_grid(height, width, device)
        epsilon = torch.finfo(torch.float32).eps
        with torch.no_grad():
            for image_index in range(batch_size):
                image_boxes = boxes[batch_idx == image_index]
                if not image_boxes.numel():
                    continue
                image_heatmap = torch.zeros((height, width), device=device, dtype=torch.float32)
                for box_chunk in image_boxes.split(64):
                    cx, cy, box_w, box_h, angle = box_chunk.unbind(dim=1)
                    dx = grid_x.unsqueeze(0) - cx[:, None, None]
                    dy = grid_y.unsqueeze(0) - cy[:, None, None]
                    cosine = angle.cos()[:, None, None]
                    sine = angle.sin()[:, None, None]
                    local_x = cosine * dx + sine * dy
                    local_y = -sine * dx + cosine * dy
                    normalized_x = 2.0 * local_x / box_w.clamp_min(epsilon)[:, None, None]
                    normalized_y = 2.0 * local_y / box_h.clamp_min(epsilon)[:, None, None]
                    inside = (normalized_x.abs() <= 1.0) & (normalized_y.abs() <= 1.0)
                    squared_radius = (normalized_x / self.soft_sigma).square() + (
                        normalized_y / self.soft_sigma
                    ).square()
                    centerness = torch.exp(-0.5 * squared_radius) * inside
                    image_heatmap = torch.maximum(image_heatmap, centerness.amax(dim=0))
                masks[image_index, 0] = image_heatmap
        return masks

    def _one_stage_loss_from_target(self, logits, target):
        """Use continuous foreground weighting with BCE+Soft-Dice."""
        logits_fp32 = logits.float()
        foreground = target.mean().clamp_min(1e-6)
        positive_weight = ((1.0 - foreground) / foreground).clamp(min=1.0, max=20.0)
        elementwise_bce = F.binary_cross_entropy_with_logits(logits_fp32, target, reduction="none")
        # Soft centers receive the strongest sparse-foreground reweighting;
        # pixels near OBB edges transition continuously to background weight.
        bce_weight = 1.0 + (positive_weight - 1.0) * target
        bce = (elementwise_bce * bce_weight).mean()

        probability = logits_fp32.sigmoid()
        intersection = (probability * target).sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + 1.0) / (
            probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1.0
        )
        return bce + dice.mean()

    def _record_stat(self, name, value):
        value = float(value.detach().float().cpu())
        self._stat_sums[name] = self._stat_sums.get(name, 0.0) + value
        self._stat_counts[name] = self._stat_counts.get(name, 0) + 1

    def _accumulate_gate_stats(self, stage_name, logits, target):
        """Record whether a gate is spatially selective instead of constant."""
        with torch.no_grad():
            gate = logits.detach().float().sigmoid()
            centered = gate - 0.5
            self._record_stat(f"saliency/{stage_name}_gate_mean", gate.mean())
            self._record_stat(f"saliency/{stage_name}_gate_std", gate.std(unbiased=False))
            self._record_stat(f"saliency/{stage_name}_positive_ratio", (centered > 0).float().mean())
            self._record_stat(f"saliency/{stage_name}_negative_ratio", (centered < 0).float().mean())

            foreground_weight = target
            background_weight = 1.0 - target
            foreground_mean = (gate * foreground_weight).sum() / foreground_weight.sum().clamp_min(1e-6)
            background_mean = (gate * background_weight).sum() / background_weight.sum().clamp_min(1e-6)
            self._record_stat(f"saliency/{stage_name}_foreground_mean", foreground_mean)
            self._record_stat(f"saliency/{stage_name}_background_mean", background_mean)
            self._record_stat(f"saliency/{stage_name}_foreground_background_gap", foreground_mean - background_mean)

    def pop_epoch_gate_stats(self):
        """Return averaged rank-local epoch statistics and reset accumulators."""
        metrics = {
            name: self._stat_sums[name] / max(self._stat_counts.get(name, 0), 1)
            for name in sorted(self._stat_sums)
        }
        metrics["saliency/gain"] = self.current_gain
        self._stat_sums.clear()
        self._stat_counts.clear()
        return metrics

    def __call__(self, preds, batch):
        base_total, base_items = v8OBBLoss.__call__(self, preds, batch)
        all_stage_logits = self._pop_saliency_logits()  # also clears unsupervised P5

        weighted_losses = []
        for stage_name, stage_index, weight in zip(
            self.supervised_stage_names, self.supervised_stage_indices, self.stage_weights
        ):
            logits = all_stage_logits[stage_index]
            target = self._rasterize_obb_masks(logits, batch).expand(-1, 2, -1, -1)
            weighted_losses.append(weight * self._one_stage_loss_from_target(logits, target))
            self._accumulate_gate_stats(stage_name, logits, target)

        weighted_loss = sum(weighted_losses) / sum(self.stage_weights)
        saliency_loss = self.current_gain * weighted_loss
        batch_size = batch["img"].shape[0]
        total_loss = base_total + saliency_loss * batch_size
        loss_items = torch.cat((base_items, saliency_loss.detach().reshape(1)))
        return total_loss, loss_items


class SoftCenternessTargetSaliencyOBBModel(TargetSaliencyOBBModel):
    """Target-saliency OBB model using the DA-014 soft supervision policy."""

    def init_criterion(self):
        return SoftCenternessTargetSaliencyOBBLoss(self)


class SoftCenternessTargetSaliencyOBBTrainer(TargetSaliencyOBBTrainer):
    """DA-014 trainer with gain warmup and gate diagnostics in results.csv."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SoftCenternessTargetSaliencyOBBModel(
            cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        # The criterion reads this value without coupling the model to Trainer.
        de_parallel(self.model).target_saliency_epoch = self.epoch
        return batch

    def validate(self):
        # Keep validation auxiliary loss on the same warmup gain as training.
        de_parallel(self.model).target_saliency_epoch = self.epoch
        if self.ema:
            self.ema.ema.target_saliency_epoch = self.epoch
        return super().validate()

    def save_metrics(self, metrics):
        criterion = getattr(de_parallel(self.model), "criterion", None)
        if isinstance(criterion, SoftCenternessTargetSaliencyOBBLoss):
            metrics = {**metrics, **criterion.pop_epoch_gate_stats()}
        return super().save_metrics(metrics)


__all__ = (
    "TargetSaliencyOBBLoss",
    "TargetSaliencyOBBModel",
    "TargetSaliencyOBBTrainer",
    "SoftCenternessTargetSaliencyOBBLoss",
    "SoftCenternessTargetSaliencyOBBModel",
    "SoftCenternessTargetSaliencyOBBTrainer",
)
