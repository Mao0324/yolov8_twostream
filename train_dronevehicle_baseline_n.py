#!/usr/bin/env python3
"""Train the YOLOv8n RGB/IR two-stream baseline with monitor support."""

from __future__ import annotations

import inspect
import os

# Keep optional experiment integrations from creating duplicate remote runs.
os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

import torch

from ultralytics import YOLO
import ultralytics.nn.tasks  # noqa: F401  # Register custom two-stream modules for checkpoint loading.
from tools.pretrained_rerun_tracker import repo_path
from tools.queue_runtime import resolve_queue_runtime
from tools.training_monitor import MonitoredOBBTrainer, create_monitor


CHECKPOINT = str(repo_path("pre-pth/yolov8n-obb_twostream_baseline.pt"))
DATA = str(repo_path("data/dronevehicle.yaml"))
PROJECT = str(repo_path("runs_baseline_n"))
EXPERIMENT_NAME = "YOLOv8n_TwoStream_Baseline_v1"

_torch_load = torch.load
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    """Load repository-owned serialized Ultralytics models across PyTorch versions."""

    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


def main():
    """Launch a manual or queue-managed monitored training run."""

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
