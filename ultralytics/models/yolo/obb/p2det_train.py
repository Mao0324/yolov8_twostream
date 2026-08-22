# Ultralytics YOLO 🚀, AGPL-3.0 license
"""OBB model, loss, and trainer for detection-oriented P2 prompts and GDER."""

from copy import copy
import weakref

import torch
import torch.nn.functional as F

from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.nn.modules import (
    P2_PROMPT_MODULES,
    P2_SECOND_GEN_PROMPT_MODULES,
    P2DualPromptGDERMergeFeedback2D,
)
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.torch_utils import de_parallel

from .prompt_utils import (
    build_rgb_global_quality_teacher,
    build_rgb_reliability_teacher,
    rasterize_rotated_soft_centerness,
)


class P2PromptOBBLoss(v8OBBLoss):
    """Add IR soft-centerness and RGB reliability prompt losses to unchanged OBB losses."""

    def __init__(self, model):
        super().__init__(model)
        self._model_ref = weakref.ref(model)
        self.prompt_modules = [module for module in model.modules() if isinstance(module, P2_PROMPT_MODULES)]
        if not self.prompt_modules:
            raise ValueError("P2PromptOBBLoss requires at least one P2 prompt merge module")

        yaml = model.yaml
        available_stages = tuple(f"P{index + 3}" for index in range(len(self.prompt_modules)))
        configured_stages = tuple(
            str(stage).upper() for stage in yaml.get("p2_prompt_supervised_stages", ["P3", "P4"])
        )
        stage_to_index = {stage: index for index, stage in enumerate(available_stages)}
        if not configured_stages or any(stage not in stage_to_index for stage in configured_stages):
            raise ValueError(
                f"invalid p2_prompt_supervised_stages={configured_stages}; available={available_stages}"
            )
        self.supervised_stage_names = configured_stages
        self.supervised_stage_indices = tuple(stage_to_index[stage] for stage in configured_stages)

        stage_weights = yaml.get("p2_prompt_stage_weights", [1.0, 0.5])
        if len(stage_weights) != len(self.supervised_stage_indices):
            raise ValueError("p2_prompt_stage_weights must match p2_prompt_supervised_stages")
        self.stage_weights = tuple(float(weight) for weight in stage_weights)
        if any(weight < 0 for weight in self.stage_weights) or sum(self.stage_weights) <= 0:
            raise ValueError("P2 prompt stage weights must be non-negative with a positive sum")

        self.ir_gain = float(yaml.get("p2_ir_prompt_gain", 0.025))
        self.rgb_gain = float(yaml.get("p2_rgb_prompt_gain", 0.01))
        self.warmup_epochs = int(yaml.get("p2_prompt_warmup_epochs", 10))
        self.ir_soft_sigma = float(yaml.get("p2_ir_soft_sigma", 0.5))
        self.rgb_exposure_sigma = float(yaml.get("p2_rgb_exposure_sigma", 0.25))
        if min(self.ir_gain, self.rgb_gain, self.warmup_epochs) < 0:
            raise ValueError("P2 prompt gains and warmup must be non-negative")
        if not 0 < self.ir_soft_sigma <= 1 or self.rgb_exposure_sigma <= 0:
            raise ValueError("invalid P2 prompt teacher sigma")

        self._stat_sums = {}
        self._stat_counts = {}

    @property
    def warmup_factor(self):
        model = self._model_ref()
        epoch = int(getattr(model, "p2_prompt_epoch", self.warmup_epochs))
        if self.warmup_epochs == 0:
            return 1.0
        return min(max(epoch / self.warmup_epochs, 0.0), 1.0)

    @staticmethod
    def _ir_loss(logits, target):
        logits = logits.float()
        target = target.float()
        foreground = target.mean().clamp_min(1e-6)
        positive_weight = ((1.0 - foreground) / foreground).clamp(min=1.0, max=20.0)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * (1.0 + (positive_weight - 1.0) * target)).mean()
        probability = logits.sigmoid()
        intersection = (probability * target).sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + 1.0) / (
            probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1.0
        )
        return bce + dice.mean()

    @staticmethod
    def _rgb_loss(logits, target):
        return F.mse_loss(logits.float().sigmoid(), target.float())

    def _pop_prompt_outputs(self):
        outputs = [module.pop_prompt_logits() for module in self.prompt_modules]
        if any(output is None for output in outputs):
            raise RuntimeError("P2 prompt logits were not captured by P2PromptOBBModel")
        gder_weights = [module.pop_gder_weights() for module in self.prompt_modules]
        return outputs, gder_weights

    def _record_stat(self, name, value):
        value = float(value.detach().float().cpu())
        self._stat_sums[name] = self._stat_sums.get(name, 0.0) + value
        self._stat_counts[name] = self._stat_counts.get(name, 0) + 1

    def _record_prompt_stats(self, stage_name, rgb_logits, ir_logits, ir_target, rgb_target):
        with torch.no_grad():
            ir_prompt = ir_logits.detach().float().sigmoid()
            foreground_weight = ir_target
            background_weight = 1.0 - ir_target
            foreground_mean = (ir_prompt * foreground_weight).sum() / foreground_weight.sum().clamp_min(1e-6)
            background_mean = (ir_prompt * background_weight).sum() / background_weight.sum().clamp_min(1e-6)
            self._record_stat(f"p2/{stage_name}_ir_prompt_mean", ir_prompt.mean())
            self._record_stat(f"p2/{stage_name}_ir_prompt_std", ir_prompt.std(unbiased=False))
            self._record_stat(f"p2/{stage_name}_ir_fg_mean", foreground_mean)
            self._record_stat(f"p2/{stage_name}_ir_bg_mean", background_mean)
            self._record_stat(f"p2/{stage_name}_ir_fg_bg_gap", foreground_mean - background_mean)
            if rgb_logits is not None and rgb_target is not None:
                rgb_prompt = rgb_logits.detach().float().sigmoid()
                self._record_stat(f"p2/{stage_name}_rgb_prompt_mean", rgb_prompt.mean())
                self._record_stat(f"p2/{stage_name}_rgb_prompt_std", rgb_prompt.std(unbiased=False))
                self._record_stat(f"p2/{stage_name}_rgb_prompt_mse", F.mse_loss(rgb_prompt, rgb_target))

    def _record_gder_stats(self, stage_name, weights):
        if weights is None:
            return
        self._record_stat(f"p2/{stage_name}_gder_modality_weight", weights[:, 0].mean())
        self._record_stat(f"p2/{stage_name}_gder_attention_weight", weights[:, 1].mean())

    def pop_epoch_prompt_stats(self):
        metrics = {
            name: self._stat_sums[name] / max(self._stat_counts.get(name, 0), 1)
            for name in sorted(self._stat_sums)
        }
        metrics["p2/prompt_warmup_factor"] = self.warmup_factor
        self._stat_sums.clear()
        self._stat_counts.clear()
        return metrics

    def __call__(self, preds, batch):
        base_total, base_items = v8OBBLoss.__call__(self, preds, batch)
        prompt_outputs, gder_weights = self._pop_prompt_outputs()  # clears unsupervised P5 as well

        ir_losses = []
        rgb_losses = []
        for stage_name, stage_index, weight in zip(
            self.supervised_stage_names, self.supervised_stage_indices, self.stage_weights
        ):
            rgb_logits, ir_logits = prompt_outputs[stage_index]
            ir_target = rasterize_rotated_soft_centerness(ir_logits, batch, sigma=self.ir_soft_sigma)
            rgb_target = None
            ir_losses.append(weight * self._ir_loss(ir_logits, ir_target))
            if rgb_logits is not None:
                rgb_target = build_rgb_reliability_teacher(
                    batch["img"], rgb_logits.shape[-2:], exposure_sigma=self.rgb_exposure_sigma
                )
                rgb_losses.append(weight * self._rgb_loss(rgb_logits, rgb_target))
            self._record_prompt_stats(stage_name, rgb_logits, ir_logits, ir_target, rgb_target)

        for stage_index, weights in enumerate(gder_weights):
            self._record_gder_stats(f"P{stage_index + 3}", weights)

        normalization = sum(self.stage_weights)
        raw_ir_loss = sum(ir_losses) / normalization
        raw_rgb_loss = sum(rgb_losses) / normalization if rgb_losses else raw_ir_loss.new_zeros(())
        factor = self.warmup_factor
        ir_prompt_loss = self.ir_gain * factor * raw_ir_loss
        rgb_prompt_loss = self.rgb_gain * factor * raw_rgb_loss
        batch_size = batch["img"].shape[0]
        total_loss = base_total + (ir_prompt_loss + rgb_prompt_loss) * batch_size
        loss_items = torch.cat(
            (base_items, ir_prompt_loss.detach().reshape(1), rgb_prompt_loss.detach().reshape(1))
        )
        return total_loss, loss_items


class P2PromptOBBModel(OBBModel):
    """OBB model that captures prompt logits only while computing train/validation loss."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.prompt_modules = [module for module in self.modules() if isinstance(module, P2_PROMPT_MODULES)]
        if not self.prompt_modules:
            raise ValueError("P2PromptOBBModel requires at least one P2 prompt merge module")
        self._set_prompt_capture(False)

    def _set_prompt_capture(self, enabled):
        for module in self.prompt_modules:
            module.capture_prompt = bool(enabled)
            if not enabled:
                module._last_prompt_logits = None
                module._last_gder_weights = None

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        if preds is None:
            self._set_prompt_capture(True)
            try:
                preds = self.forward(batch["img"])
            finally:
                # Saved logits remain available; ordinary predict/export never captures them.
                for module in self.prompt_modules:
                    module.capture_prompt = False
        return self.criterion(preds, batch)

    def init_criterion(self):
        return P2PromptOBBLoss(self)


class P2PromptOBBTrainer(OBBTrainer):
    """OBB trainer with prompt losses, warmup state, and detached diagnostics."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = P2PromptOBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
            if hasattr(weights, "p2det_migration"):
                model.p2det_migration = dict(weights.p2det_migration)
        else:
            # The parent process hands an in-memory model to the Trainer, but
            # Ultralytics DDP serializes only args.model into its temporary
            # launcher. Each child therefore reconstructs the model from YAML
            # with weights=None. Apply the exact same canonical -> baseline ->
            # P2Det transfer in every child before DDP broadcasts parameters.
            # Resume runs arrive above with checkpoint weights and are never
            # overwritten by this fresh-training initialization path.
            from tools.p2det_weight_transfer import migrate_p2det_from_canonical

            report = migrate_p2det_from_canonical(model)
            LOGGER.info(
                "P2Det fresh-model transfer "
                f"rank={RANK}: canonical->baseline max_abs_diff="
                f"{report['canonical_baseline_max_abs_diff']:.9g}, "
                "reconstructed baseline max_abs_diff="
                f"{report['reconstructed_baseline_max_abs_diff']:.9g}, "
                "baseline->P2Det max_abs_diff="
                f"{report['baseline_p2det_max_abs_diff']:.9g}, "
                f"copied={report['baseline_p2det_tensors_copied']}"
            )
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "ir_prompt_loss", "rgb_prompt_loss"
        return yolo.obb.OBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        de_parallel(self.model).p2_prompt_epoch = self.epoch
        return batch

    def validate(self):
        training_model = de_parallel(self.model)
        training_model.p2_prompt_epoch = self.epoch
        validation_model = self.ema.ema if self.ema else training_model
        validation_model.p2_prompt_epoch = self.epoch
        # BaseValidator performs inference first and then calls
        # model.loss(batch, preds). Since preds is already supplied, loss()
        # does not run its own capture-enabled forward. Keep capture enabled on
        # the exact model (EMA when present) used by validation for the full
        # validation loop, then clear it so predict/export remain unchanged.
        validation_model._set_prompt_capture(True)
        try:
            return super().validate()
        finally:
            validation_model._set_prompt_capture(False)

    def save_metrics(self, metrics):
        criterion = getattr(de_parallel(self.model), "criterion", None)
        if isinstance(criterion, P2PromptOBBLoss):
            metrics = {**metrics, **criterion.pop_epoch_prompt_stats()}
        return super().save_metrics(metrics)


class P2SecondGenOBBLoss(v8OBBLoss):
    """Fixed six-field V7+ loss without changing the historical V1-V6 loss."""

    def __init__(self, model):
        super().__init__(model)
        self._model_ref = weakref.ref(model)
        self.prompt_modules = [
            module for module in model.modules() if isinstance(module, P2_SECOND_GEN_PROMPT_MODULES)
        ]
        if not self.prompt_modules:
            raise ValueError("P2SecondGenOBBLoss requires at least one second-generation prompt module")

        yaml = model.yaml
        self.available_stages = tuple(f"P{index + 3}" for index in range(len(self.prompt_modules)))
        self.stage_to_index = {stage: index for index, stage in enumerate(self.available_stages)}
        self.spatial_stages, self.spatial_indices, self.spatial_weights = self._stage_config(
            yaml,
            "p2_prompt_supervised_stages",
            "p2_prompt_stage_weights",
            ["P3", "P4"],
            [1.0, 0.5],
        )
        global_default_stages = [
            stage
            for stage, module in zip(self.available_stages, self.prompt_modules)
            if module.use_rgb_global
        ]
        global_default_weights = [0.5 ** index for index in range(len(global_default_stages))]
        self.global_stages, self.global_indices, self.global_weights = self._stage_config(
            yaml,
            "p2_global_supervised_stages",
            "p2_global_stage_weights",
            global_default_stages,
            global_default_weights,
            allow_empty=True,
        )

        self.ir_gain = float(yaml.get("p2_ir_prompt_gain", 0.025))
        self.rgb_spatial_gain = float(yaml.get("p2_rgb_spatial_gain", yaml.get("p2_rgb_prompt_gain", 0.01)))
        self.rgb_global_gain = float(yaml.get("p2_rgb_global_gain", 0.005))
        self.warmup_epochs = int(yaml.get("p2_prompt_warmup_epochs", 10))
        self.ir_soft_sigma = float(yaml.get("p2_ir_soft_sigma", 0.5))
        self.rgb_exposure_sigma = float(yaml.get("p2_rgb_exposure_sigma", 0.25))
        if min(self.ir_gain, self.rgb_spatial_gain, self.rgb_global_gain, self.warmup_epochs) < 0:
            raise ValueError("P2 second-generation prompt gains and warmup must be non-negative")
        if not 0 < self.ir_soft_sigma <= 1 or self.rgb_exposure_sigma <= 0:
            raise ValueError("invalid P2 second-generation teacher sigma")
        self._stat_sums = {}
        self._stat_counts = {}

    def _stage_config(
        self,
        yaml,
        stage_key,
        weight_key,
        default_stages,
        default_weights,
        allow_empty=False,
    ):
        stages = tuple(str(stage).upper() for stage in yaml.get(stage_key, default_stages))
        if (not stages and not allow_empty) or any(stage not in self.stage_to_index for stage in stages):
            raise ValueError(f"invalid {stage_key}={stages}; available={self.available_stages}")
        weights = tuple(float(weight) for weight in yaml.get(weight_key, default_weights))
        if len(weights) != len(stages):
            raise ValueError(f"{weight_key} must match {stage_key}")
        if any(weight < 0 for weight in weights) or (weights and sum(weights) <= 0):
            raise ValueError(f"{weight_key} must be non-negative with a positive sum")
        return stages, tuple(self.stage_to_index[stage] for stage in stages), weights

    @property
    def warmup_factor(self):
        model = self._model_ref()
        epoch = int(getattr(model, "p2_prompt_epoch", self.warmup_epochs))
        if self.warmup_epochs == 0:
            return 1.0
        return min(max(epoch / self.warmup_epochs, 0.0), 1.0)

    def _pop_aux_outputs(self):
        outputs = [module.pop_aux_outputs() for module in self.prompt_modules]
        if any(output is None for output in outputs):
            raise RuntimeError("P2 second-generation auxiliary outputs were not captured")
        weights = [module.pop_gder_weights() for module in self.prompt_modules]
        diagnostics = [module.pop_gder_diagnostics() for module in self.prompt_modules]
        return outputs, weights, diagnostics

    def _record_stat(self, name, value):
        value = float(value.detach().float().cpu())
        self._stat_sums[name] = self._stat_sums.get(name, 0.0) + value
        self._stat_counts[name] = self._stat_counts.get(name, 0) + 1

    def pop_epoch_prompt_stats(self):
        metrics = {
            name: self._stat_sums[name] / max(self._stat_counts.get(name, 0), 1)
            for name in sorted(self._stat_sums)
        }
        metrics["p2/prompt_warmup_factor"] = self.warmup_factor
        self._stat_sums.clear()
        self._stat_counts.clear()
        return metrics

    @staticmethod
    def _weighted_average(losses, weights, reference):
        if not losses:
            return reference.new_zeros(())
        return sum(losses) / sum(weights)

    def __call__(self, preds, batch):
        base_total, base_items = v8OBBLoss.__call__(self, preds, batch)
        outputs, gder_weights, diagnostics = self._pop_aux_outputs()
        reference = base_total

        ir_losses = []
        rgb_spatial_losses = []
        rgb_spatial_active_weights = []
        for stage, index, weight in zip(self.spatial_stages, self.spatial_indices, self.spatial_weights):
            output = outputs[index]
            ir_logits = output["ir_spatial_logits"]
            ir_target = rasterize_rotated_soft_centerness(ir_logits, batch, sigma=self.ir_soft_sigma)
            ir_losses.append(weight * P2PromptOBBLoss._ir_loss(ir_logits, ir_target))
            ir_prompt = ir_logits.detach().float().sigmoid()
            foreground_mean = (ir_prompt * ir_target).sum() / ir_target.sum().clamp_min(1e-6)
            background = 1.0 - ir_target
            background_mean = (ir_prompt * background).sum() / background.sum().clamp_min(1e-6)
            self._record_stat(f"p2/{stage}_ir_prompt_mean", ir_prompt.mean())
            self._record_stat(f"p2/{stage}_ir_prompt_std", ir_prompt.std(unbiased=False))
            self._record_stat(f"p2/{stage}_ir_fg_mean", foreground_mean)
            self._record_stat(f"p2/{stage}_ir_bg_mean", background_mean)
            self._record_stat(f"p2/{stage}_ir_fg_bg_gap", foreground_mean - background_mean)

            rgb_logits = output["rgb_spatial_logits"]
            if rgb_logits is not None:
                rgb_target = build_rgb_reliability_teacher(
                    batch["img"], rgb_logits.shape[-2:], exposure_sigma=self.rgb_exposure_sigma
                )
                rgb_spatial_losses.append(weight * P2PromptOBBLoss._rgb_loss(rgb_logits, rgb_target))
                rgb_spatial_active_weights.append(weight)
                self._record_stat(f"p2/{stage}_rgb_spatial_mean", rgb_logits.sigmoid().mean())
                self._record_stat(
                    f"p2/{stage}_rgb_spatial_std",
                    rgb_logits.sigmoid().std(unbiased=False),
                )
                self._record_stat(
                    f"p2/{stage}_rgb_spatial_mse",
                    F.mse_loss(rgb_logits.float().sigmoid(), rgb_target.float()),
                )

        global_target = None
        rgb_global_losses = []
        for stage, index, weight in zip(self.global_stages, self.global_indices, self.global_weights):
            logits = outputs[index]["rgb_global_logit"]
            if logits is None:
                raise RuntimeError(f"{stage} is configured for RGB global loss but has no global head")
            if global_target is None:
                global_target = build_rgb_global_quality_teacher(
                    batch["img"], exposure_sigma=self.rgb_exposure_sigma
                )
                self._record_stat("p2/rgb_global_teacher_mean", global_target.mean())
                self._record_stat("p2/rgb_global_teacher_std", global_target.std(unbiased=False))
                self._record_stat("p2/rgb_global_teacher_min", global_target.min())
                self._record_stat("p2/rgb_global_teacher_max", global_target.max())
            rgb_global_losses.append(weight * P2PromptOBBLoss._rgb_loss(logits, global_target))
            self._record_stat(f"p2/{stage}_rgb_global_mean", logits.sigmoid().mean())
            self._record_stat(
                f"p2/{stage}_rgb_global_std", logits.sigmoid().std(unbiased=False)
            )
            self._record_stat(
                f"p2/{stage}_rgb_global_mse",
                F.mse_loss(logits.float().sigmoid(), global_target.float()),
            )

        for stage_index, weights in enumerate(gder_weights):
            if weights is not None:
                self._record_stat(f"p2/P{stage_index + 3}_gder_modality_weight", weights[:, 0].mean())
                self._record_stat(f"p2/P{stage_index + 3}_gder_attention_weight", weights[:, 1].mean())
                self._record_stat(
                    f"p2/P{stage_index + 3}_gder_modality_weight_std",
                    weights[:, 0].std(unbiased=False),
                )
                if getattr(self.prompt_modules[stage_index], "asymmetric_gder", False):
                    self._record_stat(
                        f"p2/P{stage_index + 3}_gder_gate_std",
                        weights.std(unbiased=False),
                    )
        for stage_index, diagnostic in enumerate(diagnostics):
            if not diagnostic:
                continue
            for key in ("spatial_logits_std", "gamma_spatial", "final_weight_std"):
                if key in diagnostic:
                    self._record_stat(f"p2/P{stage_index + 3}_{key}", diagnostic[key])

        factor = self.warmup_factor
        raw_ir = self._weighted_average(ir_losses, self.spatial_weights, reference)
        raw_rgb_spatial = self._weighted_average(
            rgb_spatial_losses, rgb_spatial_active_weights, reference
        )
        raw_rgb_global = self._weighted_average(rgb_global_losses, self.global_weights, reference)
        ir_prompt_loss = self.ir_gain * factor * raw_ir
        rgb_spatial_loss = self.rgb_spatial_gain * factor * raw_rgb_spatial
        rgb_global_loss = self.rgb_global_gain * factor * raw_rgb_global
        auxiliary = ir_prompt_loss + rgb_spatial_loss + rgb_global_loss
        total_loss = base_total + auxiliary * batch["img"].shape[0]
        loss_items = torch.cat(
            (
                base_items,
                ir_prompt_loss.detach().reshape(1),
                rgb_spatial_loss.detach().reshape(1),
                rgb_global_loss.detach().reshape(1),
            )
        )
        return total_loss, loss_items


class P2SecondGenOBBModel(OBBModel):
    """Dedicated second-generation model with dictionary-based auxiliary capture."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.prompt_modules = [
            module for module in self.modules() if isinstance(module, P2_SECOND_GEN_PROMPT_MODULES)
        ]
        if not self.prompt_modules:
            raise ValueError("P2SecondGenOBBModel requires a second-generation prompt module")
        self._set_prompt_capture(False)

    def _set_prompt_capture(self, enabled):
        for module in self.prompt_modules:
            module.capture_prompt = bool(enabled)
            if module.factorized_gate:
                module.expert_gate.capture_diagnostics = bool(enabled)
            if not enabled:
                module._last_aux_outputs = None
                module._last_gder_weights = None
                module._last_gder_diagnostics = None

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
                    if module.factorized_gate:
                        module.expert_gate.capture_diagnostics = False
        return self.criterion(preds, batch)

    def init_criterion(self):
        return P2SecondGenOBBLoss(self)


class P2SecondGenOBBTrainer(OBBTrainer):
    """DDP-safe trainer for second-generation P2Det with a stable six-field loss."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = P2SecondGenOBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
            if hasattr(weights, "p2det_migration"):
                model.p2det_migration = dict(weights.p2det_migration)
        else:
            from tools.p2det_weight_transfer import migrate_p2det_from_canonical

            report = migrate_p2det_from_canonical(model)
            LOGGER.info(
                "P2Det second-generation fresh-model transfer "
                f"rank={RANK}: canonical->baseline max_abs_diff="
                f"{report['canonical_baseline_max_abs_diff']:.9g}, "
                "reconstructed baseline max_abs_diff="
                f"{report['reconstructed_baseline_max_abs_diff']:.9g}, "
                "baseline->P2Det max_abs_diff="
                f"{report['baseline_p2det_max_abs_diff']:.9g}, "
                f"copied={report['baseline_p2det_tensors_copied']}"
            )
        return model

    def get_validator(self):
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "ir_prompt_loss",
            "rgb_spatial_loss",
            "rgb_global_loss",
        )
        return yolo.obb.OBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        de_parallel(self.model).p2_prompt_epoch = self.epoch
        return batch

    def validate(self):
        training_model = de_parallel(self.model)
        training_model.p2_prompt_epoch = self.epoch
        validation_model = self.ema.ema if self.ema else training_model
        validation_model.p2_prompt_epoch = self.epoch
        validation_model._set_prompt_capture(True)
        try:
            return super().validate()
        finally:
            validation_model._set_prompt_capture(False)

    def save_metrics(self, metrics):
        criterion = getattr(de_parallel(self.model), "criterion", None)
        if isinstance(criterion, P2SecondGenOBBLoss):
            metrics = {**metrics, **criterion.pop_epoch_prompt_stats()}
        return super().save_metrics(metrics)


__all__ = (
    "P2PromptOBBLoss",
    "P2PromptOBBModel",
    "P2PromptOBBTrainer",
    "P2SecondGenOBBLoss",
    "P2SecondGenOBBModel",
    "P2SecondGenOBBTrainer",
)
