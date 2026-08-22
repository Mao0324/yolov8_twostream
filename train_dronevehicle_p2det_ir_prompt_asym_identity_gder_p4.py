#!/usr/bin/env python3
"""Train P2D-013: P2D-012 with P5 GDER removed and P5 IR Prompt retained."""

# 对应模型 YAML：yaml/yolov8s-P2Det-IRPrompt-AsymIdentityGDER-P4-NoStaticMAA-PostC2f-v13.yaml
from __future__ import annotations

import inspect
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

import torch

from ultralytics import YOLO
from tools.pretrained_rerun_tracker import repo_path
from tools.queue_runtime import resolve_queue_runtime
from tools.training_monitor import MonitoredP2SecondGenOBBTrainer, create_monitor


CHECKPOINT = str(repo_path("pre-pth/yolov8s-obb.pt"))
MODEL_YAML = str(repo_path("yaml/yolov8s-P2Det-IRPrompt-AsymIdentityGDER-P4-NoStaticMAA-PostC2f-v13.yaml"))
EXPERIMENT_NAME = "P2D-013_IRPrompt-AsymIdentityGDER-P4-NoStaticMAA_PostC2f_v13"

_torch_load = torch.load
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


def main():
    queue = resolve_queue_runtime(CHECKPOINT, default_device="0,1")
    model = YOLO(queue.resume if queue.resume else MODEL_YAML, task="obb")
    monitor = create_monitor(EXPERIMENT_NAME)
    return monitor.run(
        model.train,
        trainer=MonitoredP2SecondGenOBBTrainer,
        data=str(repo_path("data/dronevehicle.yaml")),
        batch=queue.batch,
        epochs=100,
        imgsz=640,
        workers=8,
        device=queue.device,
        project="DroneVehicle_OBB_FusionTransfer",
        name=EXPERIMENT_NAME,
        exist_ok=False,
        task="obb",
        resume=queue.resume or False,
    )


if __name__ == "__main__":
    main()
