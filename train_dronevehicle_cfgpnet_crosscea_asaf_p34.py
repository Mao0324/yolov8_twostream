#!/usr/bin/env python3
"""Train CFGPNet CF-002 (CrossCEA + ASAF on P3/P4)."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

import torch

from ultralytics import YOLO
from tools.queue_runtime import resolve_queue_runtime
from tools.training_monitor import MonitoredOBBTrainer, create_monitor


ROOT = Path(__file__).resolve().parent
CHECKPOINT = str(ROOT / "pre-pth/yolov8s-obb_twostream_cfgp_crosscea_asaf_p34.pt")
EXPERIMENT_NAME = "CFGP_CF002_CrossCEA_ASAF_P34_v1"

_torch_load = torch.load
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load

QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="0,1")
monitor = create_monitor(EXPERIMENT_NAME)
model = YOLO(QUEUE.checkpoint, task="obb")

results = monitor.run(
    model.train,
    trainer=MonitoredOBBTrainer,
    data=str(ROOT / "data/dronevehicle.yaml"),
    batch=QUEUE.batch,
    epochs=100,
    imgsz=640,
    workers=8,
    device=QUEUE.device,
    project=str(ROOT / "runs/DroneVehicle_OBB_CFGP"),
    name=EXPERIMENT_NAME,
    exist_ok=False,
    task="obb",
    resume=QUEUE.resume or False,
)
