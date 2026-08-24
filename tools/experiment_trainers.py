"""Importable Detect/OBB wrappers with provenance for single-GPU and DDP runs."""

from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils import RANK

from tools.training_monitor import RemoteMonitorTrainerMixin

try:
    from ultralytics.models.yolo.obb.target_saliency_train import TargetSaliencyOBBTrainer
except ModuleNotFoundError as exc:
    TargetSaliencyOBBTrainer = None
    _TARGET_SALIENCY_IMPORT_ERROR = exc
else:
    _TARGET_SALIENCY_IMPORT_ERROR = None


SNAPSHOT_STAGE_ENV = "YOLO_EXPERIMENT_SNAPSHOT_STAGE"
RESERVATION_ENV = "YOLO_EXPERIMENT_RESERVATION"


def _patch_trusted_torch_load() -> None:
    """Keep trusted local checkpoints loadable in spawned DDP children on newer PyTorch."""

    original = torch.load
    if getattr(original, "_experiment_trusted_patch", False):
        return
    supports_weights_only = "weights_only" in inspect.signature(original).parameters

    def trusted_load(*args, **kwargs):
        if supports_weights_only:
            kwargs.setdefault("weights_only", False)
        else:
            kwargs.pop("weights_only", None)
        return original(*args, **kwargs)

    trusted_load._experiment_trusted_patch = True
    torch.load = trusted_load


_patch_trusted_torch_load()


def _materialize_snapshot(save_dir: Path) -> None:
    stage_value = os.environ.get(SNAPSHOT_STAGE_ENV)
    if not stage_value:
        raise RuntimeError(f"{SNAPSHOT_STAGE_ENV} is required by the registered experiment trainer")
    stage = Path(stage_value)
    if not stage.is_dir():
        raise FileNotFoundError(f"experiment snapshot stage not found: {stage}")

    destination = Path(save_dir) / "provenance"
    shutil.copytree(stage, destination, dirs_exist_ok=True)

    reservation_value = os.environ.get(RESERVATION_ENV)
    if reservation_value:
        Path(reservation_value).unlink(missing_ok=True)


class _SnapshotMixin:
    """Write provenance at the last safe point before model/dataloader setup."""

    def _do_train(self, world_size=1):
        # In DDP only rank 0 writes; other ranks wait when super()._do_train() initializes the process group.
        if RANK in {-1, 0}:
            _materialize_snapshot(self.save_dir)
        return super()._do_train(world_size)


class ExperimentOBBTrainer(RemoteMonitorTrainerMixin, _SnapshotMixin, OBBTrainer):
    """Default OBB trainer with provenance and rank-0 remote monitoring."""


class ExperimentDetectionTrainer(RemoteMonitorTrainerMixin, _SnapshotMixin, DetectionTrainer):
    """Default HBB trainer with provenance and rank-0 remote monitoring."""


TRAINER_WRAPPERS = {
    "ultralytics.models.yolo.detect.train:DetectionTrainer": ExperimentDetectionTrainer,
    "ultralytics.models.yolo.obb.train:OBBTrainer": ExperimentOBBTrainer,
}

if TargetSaliencyOBBTrainer is not None:

    class ExperimentTargetSaliencyOBBTrainer(
        RemoteMonitorTrainerMixin,
        _SnapshotMixin,
        TargetSaliencyOBBTrainer,
    ):
        """Target-saliency OBB trainer with provenance and rank-0 remote monitoring."""

    TRAINER_WRAPPERS[
        "ultralytics.models.yolo.obb.target_saliency_train:TargetSaliencyOBBTrainer"
    ] = ExperimentTargetSaliencyOBBTrainer


def get_trainer_wrapper(trainer_class: str):
    """Return a static importable wrapper; never silently fall back for custom losses."""

    if (
        trainer_class == "ultralytics.models.yolo.obb.target_saliency_train:TargetSaliencyOBBTrainer"
        and _TARGET_SALIENCY_IMPORT_ERROR is not None
    ):
        raise ValueError(
            f"trainer_class {trainer_class!r} is unavailable: {_TARGET_SALIENCY_IMPORT_ERROR}"
        ) from _TARGET_SALIENCY_IMPORT_ERROR
    try:
        return TRAINER_WRAPPERS[trainer_class]
    except KeyError as exc:
        available = ", ".join(sorted(TRAINER_WRAPPERS))
        raise ValueError(
            f"trainer_class {trainer_class!r} has no provenance wrapper; registered: {available}"
        ) from exc


__all__ = (
    "ExperimentDetectionTrainer",
    "ExperimentOBBTrainer",
    "get_trainer_wrapper",
)
