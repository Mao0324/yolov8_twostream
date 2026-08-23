#!/usr/bin/env python3
"""Train the scale-n P2Det V13 checkpoint with queue/monitor integration."""

from __future__ import annotations

import inspect
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

import torch

from ultralytics import YOLO
import ultralytics.nn.tasks  # noqa: F401  # Register custom modules before loading the checkpoint.
from tools.pretrained_rerun_tracker import repo_path
from tools.queue_runtime import resolve_queue_runtime
from tools.training_monitor import MonitoredP2SecondGenOBBTrainer, create_monitor


CHECKPOINT = str(
    repo_path(
        "pre-pth/yolov8n-obb_twostream_p2det_irprompt_"
        "asymidentitygder_p4_no_staticmaa_postc2f_v13_scalen.pt"
    )
)
DATA = str(repo_path("data/dronevehicle.yaml"))
PROJECT = str(repo_path("DroneVehicle_OBB_FusionTransfer"))
EXPERIMENT_NAME = "P2D-013N_IRPrompt-AsymIdentityGDER-P4-NoStaticMAA_PostC2f_v13_scalen"

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
    """Launch a manual or Agent queue-managed monitored training run."""

    queue = resolve_queue_runtime(CHECKPOINT, default_device="0,1", default_batch=64)
    monitor = create_monitor(EXPERIMENT_NAME)
    model = YOLO(queue.checkpoint, task="obb")
    return monitor.run(
        model.train,
        trainer=MonitoredP2SecondGenOBBTrainer,
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
