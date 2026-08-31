# Ultralytics YOLO 🚀, AGPL-3.0 license
"""V17 object-balanced detection-reliability distillation for RGB/IR OBB."""

from copy import copy
import os
import weakref

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.obb.p2det_reliability_train import FrozenSingleModalityTeacherPair
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.nn.modules import (
    DA016_PROMPT_ABLATION_MODULES,
    P2_OBJECT_RELIABILITY_PROMPT_MODULES,
    P2_TRUE_OBJECT_RELIABILITY_PROMPT_MODULES,
)
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.torch_utils import de_parallel


def object_gaussian_reduce(field, batch, sigma=0.7, minimum_cells=2.0, chunk_size=64):
    """Reduce BCHW fields to one equally weighted vector per GT OBB."""
    if field.ndim != 4:
        raise ValueError(f"field must be BCHW, got {tuple(field.shape)}")
    if not 0 < sigma <= 1 or minimum_cells <= 0 or chunk_size <= 0:
        raise ValueError("sigma must be in (0,1], minimum_cells and chunk_size must be positive")
    batch_size, channels, height, width = field.shape
    boxes = batch["bboxes"].reshape(-1, 5).to(field.device, torch.float32)
    batch_idx = batch["batch_idx"].reshape(-1).to(field.device, torch.long)
    reduced = field.new_zeros((boxes.shape[0], channels), dtype=torch.float32)
    if not boxes.numel():
        return reduced
    if ((batch_idx < 0) | (batch_idx >= batch_size)).any():
        raise ValueError("GT batch indices are outside the field batch")

    field = field.float()
    grid_y = (torch.arange(height, device=field.device, dtype=torch.float32) + 0.5) / height
    grid_x = (torch.arange(width, device=field.device, dtype=torch.float32) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    epsilon = torch.finfo(torch.float32).eps
    for image_index in range(batch_size):
        image_objects = (batch_idx == image_index).nonzero(as_tuple=False).squeeze(1)
        image_field = field[image_index].flatten(1).transpose(0, 1)
        for object_chunk in image_objects.split(int(chunk_size)):
            chunk_boxes = boxes[object_chunk]
            cx, cy, box_w, box_h, angle = chunk_boxes.unbind(dim=1)
            box_w = box_w.clamp_min(minimum_cells / width)
            box_h = box_h.clamp_min(minimum_cells / height)
            dx = grid_x.unsqueeze(0) - cx[:, None, None]
            dy = grid_y.unsqueeze(0) - cy[:, None, None]
            cosine = angle.cos()[:, None, None]
            sine = angle.sin()[:, None, None]
            local_x = cosine * dx + sine * dy
            local_y = -sine * dx + cosine * dy
            nx = 2.0 * local_x / box_w[:, None, None].clamp_min(epsilon)
            ny = 2.0 * local_y / box_h[:, None, None].clamp_min(epsilon)
            inside = (nx.abs() <= 1.0) & (ny.abs() <= 1.0)
            weights = torch.exp(-0.5 * ((nx / sigma).square() + (ny / sigma).square())) * inside
            flat_weights = weights.flatten(1)
            chunk_values = (flat_weights @ image_field) / flat_weights.sum(1, keepdim=True).clamp_min(epsilon)
            reduced = reduced.index_copy(0, object_chunk, chunk_values)
    return reduced


def _weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(torch.finfo(values.dtype).eps)


def materialize_object_reliability_targets(batch):
    """Copy frozen-teacher inference tensors into ordinary tensors for autograd consumers."""
    source = batch["p2_reliability_object_targets"].reshape(-1, 2).to(dtype=torch.float32)
    # FrozenSingleModalityTeacherPair.attach_targets() intentionally runs under
    # torch.inference_mode(). A dtype-preserving .float() may return that same
    # inference tensor, which autograd is forbidden to save for Prompt backward.
    # clone() here executes outside inference_mode and creates a normal constant.
    return source.detach().clone()


def calibrated_balanced_mean(values, target, confidence, calibration_ratio=0.5, balanced_ratio=0.5):
    """Retain soft calibration while giving RGB-win and IR-win objects equal influence."""
    if values.ndim != 1 or target.shape != (values.shape[0], 2) or confidence.shape != values.shape:
        raise ValueError("invalid object loss/target/confidence shapes")
    calibrated = _weighted_mean(values, confidence)
    winners = target.argmax(1)
    weighted_values = values * confidence
    modality_sums = values.new_zeros(2).scatter_add(0, winners, weighted_values)
    modality_weights = values.new_zeros(2).scatter_add(0, winners, confidence)
    present = (modality_weights > 0).to(values.dtype)
    modality_means = modality_sums / modality_weights.clamp_min(torch.finfo(values.dtype).eps)
    balanced = (modality_means * present).sum() / present.sum().clamp_min(1.0)
    return calibration_ratio * calibrated + balanced_ratio * balanced, calibrated, balanced


def class_winner_balanced_mean(values, target, confidence, classes, nc):
    """Give each present (GT class, teacher-winning modality) group equal weight."""
    if values.ndim != 1 or target.shape != (values.shape[0], 2) or confidence.shape != values.shape:
        raise ValueError("invalid class-winner balanced loss shapes")
    classes = classes.reshape(-1).to(device=values.device, dtype=torch.long)
    if classes.shape[0] != values.shape[0]:
        raise ValueError("GT classes and reliability targets have different object counts")
    if ((classes < 0) | (classes >= int(nc))).any():
        raise ValueError(f"GT classes must be in [0,{int(nc) - 1}]")
    groups = classes * 2 + target.argmax(1)
    group_count = int(nc) * 2
    group_sums = values.new_zeros(group_count).scatter_add(0, groups, values * confidence)
    group_weights = values.new_zeros(group_count).scatter_add(0, groups, confidence)
    present = (group_weights > 0).to(values.dtype)
    group_means = group_sums / group_weights.clamp_min(torch.finfo(values.dtype).eps)
    return (group_means * present).sum() / present.sum().clamp_min(1.0)


def object_rank_loss(pooled_logits, target, max_objects=128, minimum_gap=0.1):
    """Pairwise rank reliability so a constant global prior cannot minimize the auxiliary objective."""
    if pooled_logits.shape != target.shape or pooled_logits.ndim != 2 or pooled_logits.shape[1] != 2:
        raise ValueError("pooled logits and targets must both be [objects,2]")
    count = pooled_logits.shape[0]
    if count < 2:
        return pooled_logits.sum() * 0.0
    target_score = (target[:, 0].clamp_min(1e-6).log() - target[:, 1].clamp_min(1e-6).log()).float()
    if count > max_objects:
        informative = (target[:, 0] - target[:, 1]).abs().topk(max_objects, sorted=False).indices
        target_score = target_score[informative]
        pooled_logits = pooled_logits[informative]
    predicted_score = pooled_logits[:, 0] - pooled_logits[:, 1]
    target_difference = target_score[:, None] - target_score[None, :]
    predicted_difference = predicted_score[:, None] - predicted_score[None, :]
    valid = torch.triu(target_difference.abs() >= minimum_gap, diagonal=1)
    direction = target_difference.sign()
    pair_loss = F.softplus(-direction * predicted_difference)
    valid = valid.to(pair_loss.dtype)
    return (pair_loss * valid).sum() / valid.sum().clamp_min(1.0)


def gate_direction_loss(gate_probabilities, target, confidence, minimum_margin=0.1, scale=4.0):
    """Weakly require confident teacher winners to have the same gate direction."""
    margin = (target[:, 0] - target[:, 1]).abs()
    selected = margin >= minimum_margin
    direction = (target[:, 0] - target[:, 1]).sign()
    gate_margin = gate_probabilities[:, 0] - gate_probabilities[:, 1]
    values = F.softplus(-scale * direction * gate_margin)
    return calibrated_balanced_mean(
        values,
        target,
        confidence * selected.to(confidence.dtype),
        calibration_ratio=0.5,
        balanced_ratio=0.5,
    )[0]


def jensen_shannon(*distributions):
    """Mean Jensen-Shannon divergence for two or more stage distributions."""

    if len(distributions) < 2:
        raise ValueError("Jensen-Shannon consistency requires at least two stages")
    midpoint = sum(distributions) / float(len(distributions))
    log_midpoint = midpoint.clamp_min(1e-8).log()
    divergences = [
        (distribution * (distribution.clamp_min(1e-8).log() - log_midpoint)).sum(1)
        for distribution in distributions
    ]
    return (sum(divergences) / float(len(divergences))).mean()


@torch.no_grad()
def pool_obb_reliability_maps(
    auxiliary_outputs,
    boxes_by_image,
    image_shape,
    stage_weights=(1.0, 0.5),
    sigma=0.7,
    minimum_cells=2.0,
    chunk_size=64,
):
    """Pool dense V17 Prompt/gate maps for detected OBBs in network-input pixels.

    Args:
        auxiliary_outputs: stage dictionaries returned by
            ``P2ObjectReliabilityOBBModel.predict_with_reliability_maps``.
        boxes_by_image: list of ``[N,5]`` tensors in ``xywhr`` format. Coordinates
            are network-input pixels and angles are radians.
        image_shape: network input ``(height, width)``.

    Returns:
        Two lists, one per image: Prompt reliability and realized gate weights,
        both shaped ``[N,2]`` in RGB/IR order.
    """
    if len(auxiliary_outputs) < 2 or len(auxiliary_outputs) != len(stage_weights):
        raise ValueError("inference pooling requires matching outputs/weights for at least two stages")
    batch_size = auxiliary_outputs[0]["prompt_logits"].shape[0]
    if len(boxes_by_image) != batch_size:
        raise ValueError(f"got {len(boxes_by_image)} box sets for a batch of {batch_size}")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    normalized_boxes, object_batches, counts = [], [], []
    for image_index, boxes in enumerate(boxes_by_image):
        boxes = boxes.reshape(-1, 5).to(auxiliary_outputs[0]["prompt_logits"].device).float().clone()
        counts.append(boxes.shape[0])
        if boxes.numel():
            boxes[:, :4] /= boxes.new_tensor((width, height, width, height))
            normalized_boxes.append(boxes)
            object_batches.append(torch.full((boxes.shape[0],), image_index, device=boxes.device))
    if not normalized_boxes:
        empty = [auxiliary_outputs[0]["prompt_logits"].new_zeros((0, 2)) for _ in counts]
        return empty, [item.clone() for item in empty]
    pooling_batch = {
        "bboxes": torch.cat(normalized_boxes),
        "batch_idx": torch.cat(object_batches),
    }
    prompt_logits, gate_probabilities = [], []
    for output in auxiliary_outputs:
        gate_map = output["modal_weights"].float().mean(dim=2) / 2.0
        pooled = object_gaussian_reduce(
            torch.cat((output["prompt_logits"].float(), gate_map), dim=1),
            pooling_batch,
            sigma,
            minimum_cells,
            chunk_size,
        )
        prompt, gate = pooled.split(2, dim=1)
        prompt_logits.append(prompt)
        gate_probabilities.append(gate / gate.sum(1, keepdim=True).clamp_min(1e-8))
    normalizer = float(sum(stage_weights))
    prompt = sum(weight * value for weight, value in zip(stage_weights, prompt_logits)) / normalizer
    prompt = prompt.softmax(dim=1)
    gate = sum(weight * value for weight, value in zip(stage_weights, gate_probabilities)) / normalizer
    return list(prompt.split(counts)), list(gate.split(counts))


class P2ObjectReliabilityOBBLoss(v8OBBLoss):
    """Base OBB loss plus equal-object, balanced, ranked reliability supervision."""

    def __init__(self, model):
        super().__init__(model)
        self._model_ref = weakref.ref(model)
        self.prompt_modules = [
            module for module in model.modules() if isinstance(module, P2_OBJECT_RELIABILITY_PROMPT_MODULES)
        ]
        yaml = model.yaml
        self.stage_names = tuple(str(x).upper() for x in yaml.get("p2_object_reliability_stages", ["P3", "P4"]))
        self.stage_weights = tuple(float(x) for x in yaml.get("p2_object_reliability_stage_weights", [1.0, 0.5]))
        self.gate_direction_gains = tuple(
            float(x) for x in yaml.get("p2_object_gate_direction_stage_gains", [0.2, 0.1])
        )
        allowed_stages = (("P3", "P4"), ("P3", "P4", "P5"))
        if self.stage_names not in allowed_stages:
            raise ValueError(f"object reliability stages must be P3/P4 or P3/P4/P5, got {self.stage_names}")
        if len(self.prompt_modules) != len(self.stage_names):
            raise ValueError(
                f"expected {len(self.stage_names)} Prompt modules for {self.stage_names}, "
                f"got {len(self.prompt_modules)}"
            )
        if len(self.stage_weights) != len(self.stage_names):
            raise ValueError("object reliability stage weights must match configured stages")
        if len(self.gate_direction_gains) != len(self.stage_names) or min(self.gate_direction_gains) < 0:
            raise ValueError("gate direction gains must match stages and be non-negative")
        if min(self.stage_weights) < 0 or sum(self.stage_weights) <= 0:
            raise ValueError("V17 stage weights must be non-negative with positive sum")
        self.stage_normalizer = float(yaml.get("p2_object_reliability_stage_normalizer", sum(self.stage_weights)))
        if self.stage_normalizer <= 0:
            raise ValueError("object reliability stage normalizer must be positive")
        self.gain = float(yaml.get("p2_object_reliability_gain", 0.05))
        self.warmup_epochs = int(yaml.get("p2_object_reliability_warmup_epochs", 10))
        self.sigma = float(yaml.get("p2_object_support_sigma", 0.7))
        self.minimum_cells = float(yaml.get("p2_object_minimum_cells", 2.0))
        self.chunk_size = int(yaml.get("p2_object_reduce_chunk_size", 64))
        self.calibration_ratio = float(yaml.get("p2_object_calibration_ratio", 0.5))
        self.balanced_ratio = float(yaml.get("p2_object_balanced_ratio", 0.5))
        self.hard_winner_gain = float(yaml.get("p2_object_hard_winner_gain", 0.0))
        self.soft_calibration_gain = float(yaml.get("p2_object_soft_calibration_gain", 1.0))
        self.hard_winner_min_margin = float(yaml.get("p2_object_hard_winner_min_margin", 0.1))
        self.rank_gain = float(yaml.get("p2_object_rank_gain", 0.1))
        self.rank_max_objects = int(yaml.get("p2_object_rank_max_objects", 128))
        self.rank_min_gap = float(yaml.get("p2_object_rank_min_gap", 0.1))
        self.confidence_floor = float(yaml.get("p2_object_confidence_floor", 0.25))
        self.gate_direction_min_margin = float(yaml.get("p2_object_gate_direction_min_margin", 0.1))
        self.gate_direction_scale = float(yaml.get("p2_object_gate_direction_scale", 4.0))
        self.consistency_gain = float(yaml.get("p2_object_cross_scale_consistency_gain", 0.05))
        self.class_winner_balance_ratio = float(yaml.get("p2_object_class_winner_balance_ratio", 0.0))
        if min(
            self.gain,
            self.warmup_epochs,
            self.rank_gain,
            self.consistency_gain,
            self.hard_winner_gain,
            self.soft_calibration_gain,
            self.hard_winner_min_margin,
        ) < 0:
            raise ValueError("V17 gains and warmup must be non-negative")
        if self.chunk_size <= 0 or self.rank_max_objects <= 1 or not 0 <= self.confidence_floor <= 1:
            raise ValueError("invalid V17 chunk/rank/confidence settings")
        if not 0 <= self.hard_winner_min_margin <= 1:
            raise ValueError("hard-winner minimum margin must be in [0,1]")
        if not 0 <= self.class_winner_balance_ratio <= 1:
            raise ValueError("class-winner balance ratio must be in [0,1]")
        if self.hard_winner_gain + self.soft_calibration_gain <= 0:
            raise ValueError("hard-winner and soft-calibration gains cannot both be zero")
        if min(self.calibration_ratio, self.balanced_ratio) < 0 or self.calibration_ratio + self.balanced_ratio <= 0:
            raise ValueError("V17 calibration/balanced ratios must have a positive sum")
        self._stat_sums, self._stat_counts = {}, {}
        self._teacher_winner_counts = None

    @property
    def warmup_factor(self):
        model = self._model_ref()
        epoch = int(getattr(model, "p2_object_reliability_epoch", self.warmup_epochs))
        return 1.0 if self.warmup_epochs == 0 else min(max((epoch + 1) / self.warmup_epochs, 0.0), 1.0)

    def _record_values(self, name, values):
        values = values.detach().float().reshape(-1)
        if not values.numel():
            return
        value_sum = values.sum()
        self._stat_sums[name] = self._stat_sums[name] + value_sum if name in self._stat_sums else value_sum
        self._stat_counts[name] = self._stat_counts.get(name, 0) + values.numel()

    @staticmethod
    def _safe_object_correlation(predicted, expected):
        """Return a one-element correlation tensor, or empty for fewer than two objects."""
        if predicted.numel() < 2:
            return predicted.new_empty(0)
        predicted = predicted.float() - predicted.float().mean()
        expected = expected.float() - expected.float().mean()
        denominator = predicted.square().sum().sqrt() * expected.square().sum().sqrt()
        return ((predicted * expected).sum() / denominator.clamp_min(1e-8)).reshape(1)

    def _record_object_metrics(self, stage, prompt, gate, target):
        target_winner = target.argmax(1)
        prompt_winner = prompt.argmax(1)
        self._record_values(f"p2/{stage}_object_prompt_accuracy", (prompt_winner == target_winner).float())
        if gate is not None:
            gate_winner = gate.argmax(1)
            self._record_values(f"p2/{stage}_object_gate_accuracy", (gate_winner == target_winner).float())
            self._record_values(f"p2/{stage}_gate_prompt_direction_agreement", (gate_winner == prompt_winner).float())
        self._record_values(f"p2/{stage}_object_prompt_mae", (prompt[:, 0] - target[:, 0]).abs())
        self._record_values(f"p2/{stage}_object_prompt_rgb", prompt[:, 0])
        self._record_values(f"p2/{stage}_object_target_rgb", target[:, 0])
        for modality, label in ((0, "rgb"), (1, "ir")):
            selected = target_winner == modality
            self._record_values(
                f"p2/{stage}_{label}_win_recall", (prompt_winner[selected] == modality).float()
            )
        if target.shape[0] > 1:
            centered_prompt = prompt[:, 0] - prompt[:, 0].mean()
            centered_target = target[:, 0] - target[:, 0].mean()
            denominator = centered_prompt.square().sum().sqrt() * centered_target.square().sum().sqrt()
            correlation = (centered_prompt * centered_target).sum() / denominator.clamp_min(1e-8)
            self._record_values(f"p2/{stage}_object_prompt_correlation", correlation.reshape(1))

    def pop_epoch_prompt_stats(self):
        result = {
            name: float(self._stat_sums[name].cpu()) / self._stat_counts[name]
            for name in sorted(self._stat_sums)
        }
        winner_counts = self._teacher_winner_counts
        total_winners = float(winner_counts.sum().cpu()) if winner_counts is not None else 0.0
        if total_winners:
            rgb_rate = float(winner_counts[0].cpu()) / total_winners
            result["p2/teacher_rgb_win_rate"] = rgb_rate
            result["p2/teacher_ir_win_rate"] = 1.0 - rgb_rate
            result["p2/teacher_majority_baseline"] = max(rgb_rate, 1.0 - rgb_rate)
        result["p2/object_reliability_warmup_factor"] = self.warmup_factor
        self._stat_sums.clear()
        self._stat_counts.clear()
        self._teacher_winner_counts = None
        return result

    def __call__(self, preds, batch):
        base_total, base_items = v8OBBLoss.__call__(self, preds, batch)
        if "p2_reliability_object_targets" not in batch:
            raise RuntimeError("V17 batch has no frozen-teacher reliability targets")
        outputs = [module.pop_aux_outputs() for module in self.prompt_modules]
        if any(output is None for output in outputs):
            raise RuntimeError("V17 Prompt/gate outputs were not captured")
        target = materialize_object_reliability_targets(batch)
        if not target.numel():
            zero = sum(output["prompt_logits"].sum() for output in outputs) * 0.0
            return base_total + zero, torch.cat((base_items, zero.detach().reshape(1)))
        margin = (target[:, 0] - target[:, 1]).abs()
        object_classes = batch["cls"].reshape(-1).to(device=target.device, dtype=torch.long)
        if object_classes.shape[0] != target.shape[0]:
            raise RuntimeError("GT class and reliability target counts do not match")
        confidence = self.confidence_floor + (1.0 - self.confidence_floor) * margin
        target_winner = target.argmax(1)
        # Boolean reductions are deterministic on CUDA and avoid the repeated
        # torch.bincount deterministic-algorithm warning in every DDP rank.
        winner_counts = torch.stack(((target_winner == 0).sum(), (target_winner == 1).sum())).detach().to(torch.float64)
        if self._teacher_winner_counts is None:
            self._teacher_winner_counts = winner_counts
        else:
            self._teacher_winner_counts = self._teacher_winner_counts + winner_counts

        prompt_losses, rank_losses, gate_losses, object_probabilities = [], [], [], []
        for stage, stage_weight, gate_gain, module, output in zip(
            self.stage_names,
            self.stage_weights,
            self.gate_direction_gains,
            self.prompt_modules,
            outputs,
        ):
            logits = output["prompt_logits"].float()
            modal_weights = output.get("modal_weights")
            if modal_weights is None and gate_gain > 0:
                raise RuntimeError(f"{stage} has gate supervision gain {gate_gain} but exposes no modal gate")
            gate_map = modal_weights.float().mean(dim=2) / 2.0 if modal_weights is not None else None
            # Every available field uses exactly the same OBB Gaussian support,
            # so rotated coordinates and weights are built once per stage.
            fields = (logits.log_softmax(dim=1), logits)
            if gate_map is not None:
                fields = (*fields, gate_map)
            pooled_fields = object_gaussian_reduce(
                torch.cat(fields, dim=1),
                batch,
                self.sigma,
                self.minimum_cells,
                self.chunk_size,
            )
            pooled_log_probs, pooled_logits = pooled_fields[:, :2], pooled_fields[:, 2:4]
            gate_probability = pooled_fields[:, 4:6] if gate_map is not None else None
            object_losses = -(target * pooled_log_probs).sum(1)
            soft_loss, calibrated, balanced = calibrated_balanced_mean(
                object_losses,
                target,
                confidence,
                calibration_ratio=self.calibration_ratio,
                balanced_ratio=self.balanced_ratio,
            )
            winner = target.argmax(1)
            hard_values = F.cross_entropy(pooled_logits, winner, reduction="none")
            hard_confidence = confidence * (margin >= self.hard_winner_min_margin).to(confidence.dtype)
            hard_target = F.one_hot(winner, num_classes=2).to(target.dtype)
            hard_loss = calibrated_balanced_mean(
                hard_values,
                hard_target,
                hard_confidence,
                calibration_ratio=0.0,
                balanced_ratio=1.0,
            )[0]
            base_prompt_loss = self.soft_calibration_gain * soft_loss + self.hard_winner_gain * hard_loss
            balance_ratio = self.class_winner_balance_ratio
            if balance_ratio > 0:
                class_winner_soft_loss = class_winner_balanced_mean(
                    object_losses,
                    target,
                    confidence,
                    object_classes,
                    self.nc,
                )
                class_winner_hard_loss = class_winner_balanced_mean(
                    hard_values,
                    hard_target,
                    hard_confidence,
                    object_classes,
                    self.nc,
                )
                class_winner_prompt_loss = (
                    self.soft_calibration_gain * class_winner_soft_loss
                    + self.hard_winner_gain * class_winner_hard_loss
                )
                prompt_loss = (1.0 - balance_ratio) * base_prompt_loss + balance_ratio * class_winner_prompt_loss
            else:
                class_winner_prompt_loss = None
                prompt_loss = base_prompt_loss
            rank_loss = object_rank_loss(
                pooled_logits, target, max_objects=self.rank_max_objects, minimum_gap=self.rank_min_gap
            )
            prompt_probability = pooled_logits.softmax(dim=1)
            if gate_probability is not None:
                gate_probability = gate_probability / gate_probability.sum(1, keepdim=True).clamp_min(1e-8)
                direction_loss = gate_direction_loss(
                    gate_probability,
                    target,
                    confidence,
                    minimum_margin=self.gate_direction_min_margin,
                    scale=self.gate_direction_scale,
                )
            else:
                direction_loss = pooled_logits.sum() * 0.0
            prompt_losses.append(stage_weight * prompt_loss)
            rank_losses.append(stage_weight * rank_loss)
            gate_losses.append(stage_weight * gate_gain * direction_loss)
            object_probabilities.append(prompt_probability)

            self._record_values(f"p2/{stage}_object_calibrated_loss", calibrated.reshape(1))
            self._record_values(f"p2/{stage}_object_balanced_loss", balanced.reshape(1))
            self._record_values(f"p2/{stage}_object_hard_winner_loss", hard_loss.reshape(1))
            if class_winner_prompt_loss is not None:
                self._record_values(
                    f"p2/{stage}_object_class_winner_balanced_loss",
                    class_winner_prompt_loss.reshape(1),
                )
            self._record_values(f"p2/{stage}_object_rank_loss", rank_loss.reshape(1))
            self._record_values(f"p2/{stage}_gate_direction_loss", direction_loss.reshape(1))
            merge = getattr(module, "merge", None)
            if merge is not None and hasattr(merge, "prompt_strength"):
                self._record_values(
                    f"p2/{stage}_prompt_strength", merge.prompt_strength.detach().reshape(1)
                )
            if hasattr(module, "prompt_residual_gain"):
                self._record_values(
                    f"p2/{stage}_prompt_residual_gain",
                    module.prompt_residual_gain.detach().reshape(1),
                )
            self._record_object_metrics(
                stage,
                prompt_probability.detach(),
                gate_probability.detach() if gate_probability is not None else None,
                target.detach(),
            )
            if self.class_winner_balance_ratio > 0:
                for class_index in range(self.nc):
                    selected = object_classes == class_index
                    self._record_values(
                        f"p2/{stage}_class{class_index}_prompt_accuracy",
                        (prompt_probability[selected].argmax(1) == target_winner[selected]).float(),
                    )
                    self._record_values(
                        f"p2/{stage}_class{class_index}_prompt_correlation",
                        self._safe_object_correlation(
                            prompt_probability[selected, 0],
                            target[selected, 0],
                        ),
                    )

        normalizer = self.stage_normalizer
        prompt_raw = sum(prompt_losses) / normalizer
        rank_raw = sum(rank_losses) / normalizer
        gate_raw = sum(gate_losses) / normalizer
        consistency = jensen_shannon(*object_probabilities)
        raw_auxiliary = prompt_raw + self.rank_gain * rank_raw + gate_raw + self.consistency_gain * consistency
        auxiliary = self.gain * self.warmup_factor * raw_auxiliary
        self._record_values("p2/object_cross_scale_consistency", consistency.reshape(1))
        self._record_values("p2/object_prompt_raw_loss", prompt_raw.reshape(1))
        self._record_values("p2/object_gate_raw_loss", gate_raw.reshape(1))
        teacher_quality = batch.get("p2_teacher_object_quality")
        if teacher_quality is not None and teacher_quality.numel():
            self._record_values("p2/teacher_rgb_quality", teacher_quality[:, 0])
            self._record_values("p2/teacher_ir_quality", teacher_quality[:, 1])
            self._record_values("p2/teacher_relative_margin", margin)
        total = base_total + auxiliary * batch["img"].shape[0]
        return total, torch.cat((base_items, auxiliary.detach().reshape(1)))


class P2ObjectReliabilityOBBModel(OBBModel):
    """OBB model capturing multi-scale object Prompt and modal gate tensors for loss."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.prompt_modules = [
            module for module in self.modules() if isinstance(module, P2_OBJECT_RELIABILITY_PROMPT_MODULES)
        ]
        self.reliability_stage_names = tuple(
            str(value).upper() for value in self.yaml.get("p2_object_reliability_stages", ["P3", "P4"])
        )
        if len(self.prompt_modules) != len(self.reliability_stage_names):
            raise ValueError(
                f"P2ObjectReliabilityOBBModel expects one Prompt module per stage "
                f"{self.reliability_stage_names}, got {len(self.prompt_modules)}"
            )
        self._set_prompt_capture(False)

    def _set_prompt_capture(self, enabled):
        for module in self.prompt_modules:
            module.capture_prompt = bool(enabled)
            merge = getattr(module, "merge", None)
            if merge is not None:
                merge.capture_gate = bool(enabled)
            if not enabled:
                module._last_aux_outputs = None
                if merge is not None:
                    merge._last_modal_weights = None

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        if preds is None:
            self._set_prompt_capture(True)
            try:
                preds = self.forward(batch["img"])
            finally:
                for module in self.prompt_modules:
                    module.capture_prompt = False
                    merge = getattr(module, "merge", None)
                    if merge is not None:
                        merge.capture_gate = False
        return self.criterion(preds, batch)

    def init_criterion(self):
        return P2ObjectReliabilityOBBLoss(self)

    @torch.no_grad()
    def predict_with_reliability_maps(self, images, **predict_kwargs):
        """Return detector output plus configured stage maps for per-detection pooling."""
        was_training = self.training
        self.eval()
        self._set_prompt_capture(True)
        try:
            predictions = self.predict(images, **predict_kwargs)
            auxiliary_outputs = [module.pop_aux_outputs() for module in self.prompt_modules]
            if any(output is None for output in auxiliary_outputs):
                raise RuntimeError("inference did not capture all configured reliability maps")
            auxiliary_outputs = [
                {name: value.detach() for name, value in output.items()} for output in auxiliary_outputs
            ]
        finally:
            self._set_prompt_capture(False)
            self.train(was_training)
        return predictions, auxiliary_outputs


class P2TrueObjectReliabilityOBBModel(P2ObjectReliabilityOBBModel):
    """V18 model requiring hard-winner Prompt and sign-preserving P3/P4 gates."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if len(self.prompt_modules) != 2 or not all(
            isinstance(module, P2_TRUE_OBJECT_RELIABILITY_PROMPT_MODULES)
            for module in self.prompt_modules
        ):
            raise ValueError("P2TrueObjectReliabilityOBBModel requires two V18 sign-preserving modules")
        if float(self.yaml.get("p2_object_hard_winner_gain", 0.0)) <= 0:
            raise ValueError("V18 requires a positive hard-winner Prompt loss gain")


class DA016PromptAblationOBBModel(P2ObjectReliabilityOBBModel):
    """DA016-parent model with independently auxiliary/residual P3/P4/P5 Prompts."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if not all(isinstance(module, DA016_PROMPT_ABLATION_MODULES) for module in self.prompt_modules):
            raise ValueError("DA016 Prompt ablation contains an unsupported Prompt wrapper")
        if any(float(value) != 0.0 for value in self.yaml.get("p2_object_gate_direction_stage_gains", [])):
            raise ValueError("DA016 Prompt ablations must not use gate-direction supervision")


class P2ObjectReliabilityOBBValidator(OBBValidator):
    reliability_teacher_pair = None

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if self.reliability_teacher_pair is None:
            raise RuntimeError("V17 validator teacher pair was not initialized")
        return self.reliability_teacher_pair.attach_targets(batch)


class P2ObjectReliabilityOBBTrainer(OBBTrainer):
    """DDP-safe trainer for V17 true object-level reliability distillation."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = P2ObjectReliabilityOBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if not weights:
            raise RuntimeError(
                "V17 formal training requires an embedded-architecture .pt checkpoint; "
                "build it with tools/make_p2det_v17_checkpoint.py"
            )
        model.load(weights)
        if hasattr(weights, "p2det_migration"):
            model.p2det_migration = dict(weights.p2det_migration)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "object_reliability_loss"
        return P2ObjectReliabilityOBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        rgb_teacher = os.getenv("P2DET_V16_RGB_TEACHER")
        ir_teacher = os.getenv("P2DET_V16_IR_TEACHER")
        if not rgb_teacher or not ir_teacher:
            raise RuntimeError("Set distinct P2DET_V16_RGB_TEACHER and P2DET_V16_IR_TEACHER checkpoints")
        model = de_parallel(self.model)
        teacher_pair = FrozenSingleModalityTeacherPair(
            rgb_teacher,
            ir_teacher,
            self.device,
            self.data["names"],
            temperature=float(model.yaml.get("p2_teacher_temperature", 0.2)),
            topk=int(model.yaml.get("p2_teacher_match_topk", 128)),
        )
        self.reliability_teacher_pair = teacher_pair
        if self.validator is not None:
            self.validator.reliability_teacher_pair = teacher_pair

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        model = de_parallel(self.model)
        model.p2_object_reliability_epoch = self.epoch
        return self.reliability_teacher_pair.attach_targets(batch)

    def validate(self):
        training_model = de_parallel(self.model)
        training_model.p2_object_reliability_epoch = self.epoch
        validation_model = self.ema.ema if self.ema else training_model
        validation_model.p2_object_reliability_epoch = self.epoch
        validation_model._set_prompt_capture(True)
        try:
            return super().validate()
        finally:
            validation_model._set_prompt_capture(False)

    def save_metrics(self, metrics):
        criterion = getattr(de_parallel(self.model), "criterion", None)
        if isinstance(criterion, P2ObjectReliabilityOBBLoss):
            metrics = {**metrics, **criterion.pop_epoch_prompt_stats()}
        return super().save_metrics(metrics)


class P2TrueObjectReliabilityOBBTrainer(P2ObjectReliabilityOBBTrainer):
    """DDP-safe trainer that reconstructs the embedded V18 model class."""

    def check_resume(self, overrides):
        """Keep performance-only V18 overrides when resuming a checkpoint."""
        super().check_resume(overrides)
        if self.resume:
            for name in ("workers", "plots", "deterministic"):
                if name in overrides:
                    setattr(self.args, name, overrides[name])

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = P2TrueObjectReliabilityOBBModel(
            cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1
        )
        if not weights:
            raise RuntimeError(
                "V18 formal training requires an embedded-architecture .pt checkpoint; "
                "build it with tools/make_p2det_v18_checkpoint.py"
            )
        model.load(weights)
        if hasattr(weights, "p2det_migration"):
            model.p2det_migration = dict(weights.p2det_migration)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "true_object_reliability_loss"
        return P2ObjectReliabilityOBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


class DA016PromptAblationOBBTrainer(P2ObjectReliabilityOBBTrainer):
    """DDP-safe trainer for registered DA016-parent Prompt ablations."""

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            for name in ("workers", "plots", "deterministic"):
                if name in overrides:
                    setattr(self.args, name, overrides[name])

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DA016PromptAblationOBBModel(
            cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1
        )
        if not weights:
            raise RuntimeError(
                "DA016 Prompt ablations require an embedded-architecture .pt checkpoint; "
                "build it with tools/make_da016_prompt_ablation_checkpoints.py"
            )
        model.load(weights)
        if hasattr(weights, "da016_prompt_migration"):
            model.da016_prompt_migration = dict(weights.da016_prompt_migration)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "object_reliability_loss"
        return P2ObjectReliabilityOBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


__all__ = (
    "P2ObjectReliabilityOBBLoss",
    "P2ObjectReliabilityOBBModel",
    "P2ObjectReliabilityOBBTrainer",
    "P2ObjectReliabilityOBBValidator",
    "P2TrueObjectReliabilityOBBModel",
    "P2TrueObjectReliabilityOBBTrainer",
    "DA016PromptAblationOBBModel",
    "DA016PromptAblationOBBTrainer",
    "calibrated_balanced_mean",
    "class_winner_balanced_mean",
    "gate_direction_loss",
    "materialize_object_reliability_targets",
    "object_gaussian_reduce",
    "object_rank_loss",
    "pool_obb_reliability_maps",
)
