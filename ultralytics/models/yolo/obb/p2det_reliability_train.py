# Ultralytics YOLO 🚀, AGPL-3.0 license
"""V16 OBB training with frozen single-modality detection-reliability teachers."""

from copy import copy
import os
from pathlib import Path
import weakref

import torch
import torch.nn.functional as F

from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.nn.modules import P2_RELIABILITY_PROMPT_MODULES
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.metrics import probiou
from ultralytics.utils.torch_utils import de_parallel


def _standard_single_stream_forward(model, images):
    """Run an ordinary YOLO graph without this fork's six-channel splitter."""
    outputs = []
    x = images
    for module in model.model:
        if module.f != -1:
            x = outputs[module.f] if isinstance(module.f, int) else [x if j == -1 else outputs[j] for j in module.f]
        x = module(x)
        outputs.append(x if module.i in model.save else None)
    return x


@torch.no_grad()
def match_detection_quality(prediction, batch, nc, topk=128, object_chunk_size=256):
    """Match GT OBBs to a teacher with chunked cross-image vectorization.

    The result is identical to matching one image at a time: every GT still
    selects its own class-specific top-k anchors and returns the maximum of
    class confidence times ProbIoU. Chunking only bounds the temporary
    ``[objects, anchors]`` score tensor and removes one Python/CUDA launch loop
    per image, which matters for DroneVehicle batches containing ~1000 boxes.
    """
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    if prediction.ndim != 3 or prediction.shape[1] < 4 + nc + 1:
        raise ValueError(f"unexpected OBB teacher prediction shape {tuple(prediction.shape)}")
    boxes = torch.cat((prediction[:, :4], prediction[:, 4 + nc : 5 + nc]), dim=1).permute(0, 2, 1)
    scores = prediction[:, 4 : 4 + nc].permute(0, 2, 1)

    gt_boxes = batch["bboxes"].reshape(-1, 5).to(device=prediction.device, dtype=torch.float32).clone()
    gt_classes = batch["cls"].reshape(-1).to(device=prediction.device, dtype=torch.long)
    gt_batch = batch["batch_idx"].reshape(-1).to(device=prediction.device, dtype=torch.long)
    if gt_boxes.numel() == 0:
        return prediction.new_zeros((0,), dtype=torch.float32)
    height, width = batch["img"].shape[-2:]
    gt_boxes[:, :4] *= gt_boxes.new_tensor((width, height, width, height))
    quality = prediction.new_zeros((gt_boxes.shape[0],), dtype=torch.float32)

    if ((gt_classes < 0) | (gt_classes >= nc)).any():
        raise ValueError(f"GT classes must be in [0,{nc - 1}]")
    if object_chunk_size <= 0:
        raise ValueError("teacher match object chunk size must be positive")
    object_indices = torch.arange(gt_boxes.shape[0], device=prediction.device)
    for object_chunk in object_indices.split(int(object_chunk_size)):
        chunk_batch = gt_batch[object_chunk]
        chunk_classes = gt_classes[object_chunk]
        # Advanced indexing selects one image and one class for every GT,
        # producing [objects, anchors] without iterating over batch images.
        class_scores = scores[chunk_batch, :, chunk_classes].float()
        count = min(int(topk), class_scores.shape[1])
        confidence, anchor_indices = class_scores.topk(count, dim=1, sorted=False)
        candidate_boxes = boxes[chunk_batch[:, None], anchor_indices].float()
        expanded_gt = gt_boxes[object_chunk, None, :].expand_as(candidate_boxes)
        iou = probiou(expanded_gt.reshape(-1, 5), candidate_boxes.reshape(-1, 5)).reshape_as(confidence)
        quality[object_chunk] = (confidence * iou.clamp(0.0, 1.0)).amax(dim=1)
    return quality


class FrozenSingleModalityTeacherPair:
    """Two distinct frozen 3-channel OBB teachers producing relative reliability."""

    def __init__(self, rgb_checkpoint, ir_checkpoint, device, names, temperature=0.2, topk=128):
        from ultralytics import YOLO

        self.temperature = float(temperature)
        self.topk = int(topk)
        if self.temperature <= 0 or self.topk <= 0:
            raise ValueError("teacher temperature and topk must be positive")
        rgb_path = Path(rgb_checkpoint).expanduser().resolve()
        ir_path = Path(ir_checkpoint).expanduser().resolve()
        if rgb_path == ir_path:
            raise ValueError("RGB-only and IR-only teachers must be distinct checkpoints")
        for label, path in (("RGB-only", rgb_path), ("IR-only", ir_path)):
            if path.suffix.lower() != ".pt" or not path.is_file():
                raise FileNotFoundError(f"{label} teacher checkpoint not found: {path}")

        self.models = []
        self.device_type = torch.device(device).type
        self.teacher_dtype = torch.float16 if self.device_type == "cuda" else torch.float32
        expected_names = self._normalize_names(names)
        for label, path in (("RGB-only", rgb_path), ("IR-only", ir_path)):
            teacher = YOLO(str(path), task="obb").model.float().to(device).eval()
            if self.teacher_dtype == torch.float16:
                teacher.half()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self._validate_single_stream_teacher(teacher, label)
            teacher_names = self._normalize_names(teacher.names)
            if teacher_names != expected_names:
                raise ValueError(
                    f"{label} teacher class order {teacher_names} does not match dataset {expected_names}"
                )
            self.models.append(teacher)
        self.nc = len(expected_names)
        LOGGER.info(
            "P2Det V16 frozen teachers: RGB=%s, IR=%s, precision=%s, temperature=%g, topk=%d",
            rgb_path,
            ir_path,
            "FP16 autocast" if self.teacher_dtype == torch.float16 else "FP32",
            self.temperature,
            self.topk,
        )

    @staticmethod
    def _normalize_names(names):
        if isinstance(names, dict):
            return tuple(str(names[index]) for index in sorted(names))
        return tuple(str(name) for name in names)

    @staticmethod
    def _validate_single_stream_teacher(model, label):
        first_conv = next((module for module in model.modules() if isinstance(module, torch.nn.Conv2d)), None)
        if first_conv is None or first_conv.in_channels != 3:
            raise ValueError(f"{label} teacher must have a 3-channel input convolution")
        invalid_sources = []
        for module in model.model:
            sources = (module.f,) if isinstance(module.f, int) else tuple(module.f)
            if any(source in {-3, -4, -5, -6} for source in sources):
                invalid_sources.append((module.i, module.f))
        if invalid_sources:
            raise ValueError(f"{label} teacher is a two-stream graph, found routes {invalid_sources[:4]}")

    @torch.inference_mode()
    def attach_targets(self, batch):
        images = batch["img"]
        if images.ndim != 4 or images.shape[1] != 6:
            raise ValueError(f"V16 expects paired six-channel images, got {tuple(images.shape)}")
        modality_images = (
            images[:, :3].to(dtype=self.teacher_dtype),
            images[:, 3:6].to(dtype=self.teacher_dtype),
        )
        qualities = []
        for teacher, modality_image in zip(self.models, modality_images):
            with torch.cuda.amp.autocast(
                enabled=self.device_type == "cuda",
                dtype=torch.float16,
            ):
                prediction = _standard_single_stream_forward(teacher, modality_image)
            qualities.append(match_detection_quality(prediction, batch, self.nc, self.topk))
        paired_quality = torch.stack(qualities, dim=1)
        relative = torch.softmax(paired_quality / self.temperature, dim=1)
        batch["p2_reliability_object_targets"] = relative
        batch["p2_teacher_object_quality"] = paired_quality
        return batch


@torch.no_grad()
def rasterize_relative_reliability(
    reference,
    batch,
    sigma=0.7,
    minimum_cells=2.0,
    chunk_size=64,
):
    """Vectorize per-object relative teacher probabilities inside GT support."""
    if not 0 < sigma <= 1 or minimum_cells <= 0 or chunk_size <= 0:
        raise ValueError("sigma must be in (0,1], minimum_cells and chunk_size must be positive")
    batch_size, channels, height, width = reference.shape
    if channels != 2:
        raise ValueError("relative reliability logits must have two modality channels")
    boxes = batch["bboxes"].reshape(-1, 5).to(reference.device, torch.float32)
    batch_idx = batch["batch_idx"].reshape(-1).to(reference.device, torch.long)
    object_targets = batch["p2_reliability_object_targets"].reshape(-1, 2).to(reference.device, torch.float32)
    if boxes.shape[0] != object_targets.shape[0]:
        raise ValueError("teacher object target count does not match GT box count")
    target_sum = reference.new_zeros((batch_size, 2, height, width), dtype=torch.float32)
    weight_sum = reference.new_zeros((batch_size, 1, height, width), dtype=torch.float32)
    if boxes.numel() == 0:
        return target_sum, weight_sum
    if ((batch_idx < 0) | (batch_idx >= batch_size)).any():
        raise ValueError("GT batch indices are outside the reference batch")

    grid_y = (torch.arange(height, device=reference.device, dtype=torch.float32) + 0.5) / height
    grid_x = (torch.arange(width, device=reference.device, dtype=torch.float32) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    epsilon = torch.finfo(torch.float32).eps
    # Chunking bounds temporary [objects,H,W] tensors while reducing thousands
    # of per-object CUDA kernel launches to a few vectorized launches per image.
    for image_index in range(batch_size):
        image_objects = (batch_idx == image_index).nonzero(as_tuple=False).squeeze(1)
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
            nx = 2.0 * local_x / box_w.clamp_min(epsilon)[:, None, None]
            ny = 2.0 * local_y / box_h.clamp_min(epsilon)[:, None, None]
            inside = (nx.abs() <= 1.0) & (ny.abs() <= 1.0)
            weights = torch.exp(-0.5 * ((nx / sigma).square() + (ny / sigma).square())) * inside
            flat_weights = weights.flatten(1)
            target_sum[image_index] += (
                object_targets[object_chunk].transpose(0, 1) @ flat_weights
            ).reshape(2, height, width)
            weight_sum[image_index, 0] += weights.sum(dim=0)
    target = target_sum / weight_sum.clamp_min(epsilon)
    return target, weight_sum.clamp(max=1.0)


class P2ReliabilityOBBLoss(v8OBBLoss):
    """Unchanged OBB loss plus teacher-relative soft cross entropy at P3/P4."""

    def __init__(self, model):
        super().__init__(model)
        self._model_ref = weakref.ref(model)
        self.prompt_modules = [
            module for module in model.modules() if isinstance(module, P2_RELIABILITY_PROMPT_MODULES)
        ]
        if len(self.prompt_modules) != 2:
            raise ValueError(f"V16 requires exactly P3/P4 reliability modules, got {len(self.prompt_modules)}")
        yaml = model.yaml
        self.stage_names = tuple(str(x).upper() for x in yaml.get("p2_reliability_supervised_stages", ["P3", "P4"]))
        self.stage_weights = tuple(float(x) for x in yaml.get("p2_reliability_stage_weights", [1.0, 0.5]))
        if self.stage_names != ("P3", "P4") or len(self.stage_weights) != 2:
            raise ValueError("V16 reliability supervision is intentionally fixed to P3/P4")
        if any(weight < 0 for weight in self.stage_weights) or sum(self.stage_weights) <= 0:
            raise ValueError("reliability stage weights must be non-negative with positive sum")
        self.gain = float(yaml.get("p2_reliability_gain", 0.05))
        self.warmup_epochs = int(yaml.get("p2_reliability_warmup_epochs", 10))
        self.sigma = float(yaml.get("p2_reliability_support_sigma", 0.7))
        self.minimum_cells = float(yaml.get("p2_reliability_minimum_cells", 2.0))
        self.raster_chunk_size = int(yaml.get("p2_reliability_raster_chunk_size", 64))
        if min(self.gain, self.warmup_epochs) < 0 or self.raster_chunk_size <= 0:
            raise ValueError("reliability gain/warmup must be non-negative and raster chunk size must be positive")
        self._stat_sums, self._stat_counts = {}, {}

    @property
    def warmup_factor(self):
        model = self._model_ref()
        epoch = int(getattr(model, "p2_reliability_epoch", self.warmup_epochs))
        return 1.0 if self.warmup_epochs == 0 else min(max((epoch + 1) / self.warmup_epochs, 0.0), 1.0)

    def _record(self, name, value):
        self._stat_sums[name] = self._stat_sums.get(name, 0.0) + float(value.detach().float().cpu())
        self._stat_counts[name] = self._stat_counts.get(name, 0) + 1

    def pop_epoch_prompt_stats(self):
        result = {name: self._stat_sums[name] / self._stat_counts[name] for name in sorted(self._stat_sums)}
        result["p2/reliability_warmup_factor"] = self.warmup_factor
        self._stat_sums.clear()
        self._stat_counts.clear()
        return result

    def __call__(self, preds, batch):
        base_total, base_items = v8OBBLoss.__call__(self, preds, batch)
        if "p2_reliability_object_targets" not in batch:
            raise RuntimeError("V16 batch has no frozen-teacher reliability targets")
        outputs = [module.pop_aux_outputs() for module in self.prompt_modules]
        if any(output is None for output in outputs):
            raise RuntimeError("V16 prompt/gate outputs were not captured")

        stage_losses = []
        for stage, weight, output in zip(self.stage_names, self.stage_weights, outputs):
            logits = output["prompt_logits"].float()
            target, support = rasterize_relative_reliability(
                logits,
                batch,
                sigma=self.sigma,
                minimum_cells=self.minimum_cells,
                chunk_size=self.raster_chunk_size,
            )
            pixel_loss = -(target * logits.log_softmax(dim=1)).sum(dim=1, keepdim=True)
            stage_loss = (pixel_loss * support).sum() / support.sum().clamp_min(1.0)
            stage_losses.append(weight * stage_loss)

            probabilities = output["prompt_probabilities"].detach().float()
            modal_weights = output["modal_weights"].detach().float()
            self._record(f"p2/{stage}_rgb_prompt_mean", probabilities[:, 0].mean())
            self._record(f"p2/{stage}_ir_prompt_mean", probabilities[:, 1].mean())
            self._record(f"p2/{stage}_prompt_std", probabilities.std(unbiased=False))
            self._record(
                f"p2/{stage}_rgb_prompt_fg",
                (probabilities[:, 0:1] * support).sum() / support.sum().clamp_min(1.0),
            )
            self._record(f"p2/{stage}_rgb_gate_mean", modal_weights[:, 0].mean())
            self._record(f"p2/{stage}_ir_gate_mean", modal_weights[:, 1].mean())
            self._record(f"p2/{stage}_gate_std", modal_weights.std(unbiased=False))
            self._record(f"p2/{stage}_target_rgb", (target[:, 0:1] * support).sum() / support.sum().clamp_min(1.0))
            expanded_support = support.expand_as(probabilities)
            self._record(
                f"p2/{stage}_prompt_teacher_mae",
                ((probabilities - target).abs() * expanded_support).sum()
                / expanded_support.sum().clamp_min(1.0),
            )
            prompt_agreement = (probabilities.argmax(1, keepdim=True) == target.argmax(1, keepdim=True)).float()
            self._record(
                f"p2/{stage}_prompt_teacher_agreement",
                (prompt_agreement * support).sum() / support.sum().clamp_min(1.0),
            )
            spatial_gate = modal_weights.mean(dim=2) / 2.0
            gate_agreement = (spatial_gate.argmax(1, keepdim=True) == target.argmax(1, keepdim=True)).float()
            self._record(
                f"p2/{stage}_gate_teacher_agreement",
                (gate_agreement * support).sum() / support.sum().clamp_min(1.0),
            )

        teacher_quality = batch.get("p2_teacher_object_quality")
        if teacher_quality is not None and teacher_quality.numel():
            self._record("p2/teacher_rgb_quality", teacher_quality[:, 0].mean())
            self._record("p2/teacher_ir_quality", teacher_quality[:, 1].mean())
            relative = batch["p2_reliability_object_targets"].float()
            entropy = -(relative * relative.clamp_min(1e-8).log()).sum(1).mean()
            self._record("p2/teacher_relative_entropy", entropy)
            self._record("p2/teacher_relative_margin", (relative[:, 0] - relative[:, 1]).abs().mean())

        raw_auxiliary = sum(stage_losses) / sum(self.stage_weights)
        auxiliary = self.gain * self.warmup_factor * raw_auxiliary
        total = base_total + auxiliary * batch["img"].shape[0]
        return total, torch.cat((base_items, auxiliary.detach().reshape(1)))


class P2ReliabilityOBBModel(OBBModel):
    """OBB model that captures V16 prompt and modality-gate tensors for loss only."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.prompt_modules = [
            module for module in self.modules() if isinstance(module, P2_RELIABILITY_PROMPT_MODULES)
        ]
        if len(self.prompt_modules) != 2:
            raise ValueError(f"P2ReliabilityOBBModel expects exactly two V16 modules, got {len(self.prompt_modules)}")
        self._set_prompt_capture(False)

    def _set_prompt_capture(self, enabled):
        for module in self.prompt_modules:
            module.capture_prompt = bool(enabled)
            module.merge.capture_gate = bool(enabled)
            if not enabled:
                module._last_aux_outputs = None
                module.merge._last_modal_weights = None

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
                    module.merge.capture_gate = False
        return self.criterion(preds, batch)

    def init_criterion(self):
        return P2ReliabilityOBBLoss(self)


class P2ReliabilityOBBValidator(OBBValidator):
    """Validation adapter that generates the same frozen-teacher targets."""

    reliability_teacher_pair = None

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if self.reliability_teacher_pair is None:
            raise RuntimeError("V16 validator teacher pair was not initialized")
        return self.reliability_teacher_pair.attach_targets(batch)


class P2ReliabilityOBBTrainer(OBBTrainer):
    """DDP-safe trainer for V16 relative reliability distillation."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = P2ReliabilityOBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if not weights:
            raise RuntimeError(
                "V16 formal training requires an embedded-architecture .pt checkpoint; "
                "build it with tools/make_p2det_v16_checkpoint.py"
            )
        model.load(weights)
        if hasattr(weights, "p2det_migration"):
            model.p2det_migration = dict(weights.p2det_migration)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "reliability_prompt_loss"
        return P2ReliabilityOBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        rgb_teacher = os.getenv("P2DET_V16_RGB_TEACHER")
        ir_teacher = os.getenv("P2DET_V16_IR_TEACHER")
        if not rgb_teacher or not ir_teacher:
            raise RuntimeError(
                "Set P2DET_V16_RGB_TEACHER and P2DET_V16_IR_TEACHER to distinct frozen "
                "3-channel OBB checkpoints"
            )
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
        model.p2_reliability_epoch = self.epoch
        return self.reliability_teacher_pair.attach_targets(batch)

    def validate(self):
        training_model = de_parallel(self.model)
        training_model.p2_reliability_epoch = self.epoch
        validation_model = self.ema.ema if self.ema else training_model
        validation_model.p2_reliability_epoch = self.epoch
        validation_model._set_prompt_capture(True)
        try:
            return super().validate()
        finally:
            validation_model._set_prompt_capture(False)

    def save_metrics(self, metrics):
        criterion = getattr(de_parallel(self.model), "criterion", None)
        if isinstance(criterion, P2ReliabilityOBBLoss):
            metrics = {**metrics, **criterion.pop_epoch_prompt_stats()}
        return super().save_metrics(metrics)


__all__ = (
    "FrozenSingleModalityTeacherPair",
    "P2ReliabilityOBBLoss",
    "P2ReliabilityOBBModel",
    "P2ReliabilityOBBTrainer",
    "P2ReliabilityOBBValidator",
    "match_detection_quality",
    "rasterize_relative_reliability",
)
