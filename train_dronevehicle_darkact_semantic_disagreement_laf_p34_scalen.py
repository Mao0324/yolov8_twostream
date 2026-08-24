#!/usr/bin/env python3
"""Train scale-n DA-016 with DDP-safe monitor and queue integration."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"
os.environ.setdefault("YOLO_MONITOR_URL", "https://monitor.maocong.me")
os.environ["YOLO_MONITOR_USE_PROXY"] = "false"

ROOT = Path(__file__).resolve().parent
MONITOR_DIR = ROOT / "monitor"
_required_python_paths = (str(ROOT), str(MONITOR_DIR))
for _path in reversed(_required_python_paths):
    if _path not in sys.path:
        sys.path.insert(0, _path)
_existing_python_paths = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
_existing_python_paths = [entry for entry in _existing_python_paths if entry not in _required_python_paths]
os.environ["PYTHONPATH"] = os.pathsep.join((*_required_python_paths, *_existing_python_paths))

import torch

from ultralytics import YOLO
import ultralytics.nn.tasks  # noqa: F401  # Register custom DarkAct modules.
from monitored_obb_trainer import MonitoredOBBTrainer
from tools.pretrained_rerun_tracker import repo_path
from tools.queue_runtime import resolve_queue_runtime
from yolo_monitor import YoloExperimentMonitor


CHECKPOINT = str(
    repo_path("pre-pth/yolov8n-obb_twostream_darkact_da016_semantic_disagreement_laf_p34_scalen.pt")
)
DATA = str(repo_path("data/dronevehicle.yaml"))
PROJECT = str(repo_path("runs/DroneVehicle_OBB_FusionTransfer"))
EXPERIMENT_NAME = "DarkAct_DA016N_SemanticDisagreementLAF_P34_H2-4-8_R4-NoStaticMAA_v1_scalen"

_torch_load = torch.load
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


def create_monitor(experiment_name: str) -> YoloExperimentMonitor:
    """Create the standard rank-0 monitor without importing unrelated trainers."""

    if not os.getenv("YOLO_MONITOR_URL") or not os.getenv("YOLO_MONITOR_TOKEN"):
        raise RuntimeError("Set YOLO_MONITOR_URL and YOLO_MONITOR_TOKEN before training")
    return YoloExperimentMonitor(experiment_name=experiment_name)


def main():
    """Launch a manual or Agent queue-managed monitored training run."""

    queue = resolve_queue_runtime(CHECKPOINT, default_device="0,1", default_batch=64)
    monitor = create_monitor(EXPERIMENT_NAME)
    model = YOLO(queue.checkpoint, task="obb")
    return monitor.run(
        model.train,
        trainer=MonitoredOBBTrainer,
        data=DATA,
        batch=queue.batch,
        epochs=100,
        imgsz=640,
        workers=8,
        device=queue.device,
        project=PROJECT,
        name=EXPERIMENT_NAME,
        exist_ok=False,
        task="obb",
        resume=queue.resume or False,
    )


if __name__ == "__main__":
    results = main()
