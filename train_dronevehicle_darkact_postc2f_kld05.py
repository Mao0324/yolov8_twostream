"""Train the best-mAP50 DA-004 architecture with 0.5 ProbIoU + 0.5 paper KLD."""

import inspect
import os
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

import torch

from ultralytics import YOLO
from tools.queue_runtime import resolve_queue_runtime
from tools.training_monitor import create_monitor
from monitored_kld_obb_trainer import MonitoredKLDProbIoUOBBTrainer


_torch_load = torch.load
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load

ROOT = Path(__file__).resolve().parent
CHECKPOINT = str(ROOT / "pre-pth/yolov8s-obb_twostream_darkact_postc2f_kld05_v1.pt")
EXPERIMENT_NAME = "DarkAct_PostC2fStaticMAA_LAFFeedback_P345_KLDProbIoU05_v1"
QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="0,1")
monitor = create_monitor(EXPERIMENT_NAME)
model = YOLO(QUEUE.checkpoint, task="obb")

results = monitor.run(
    model.train,
    trainer=MonitoredKLDProbIoUOBBTrainer,
    data=str(ROOT / "data/dronevehicle.yaml"),
    batch=QUEUE.batch,
    epochs=100,
    imgsz=640,
    workers=8,
    device=QUEUE.device,
    project="DroneVehicle_OBB_FusionTransfer",
    name=EXPERIMENT_NAME,
    exist_ok=False,
    task="obb",
    resume=QUEUE.resume or False,
)
