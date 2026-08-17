"""Shared runtime helper for the two ASSA-replaced LAF experiment entrypoints."""
import inspect
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"
import torch

from ultralytics import YOLO
from tools.pretrained_rerun_tracker import checkpoint_train, repo_path
from tools.queue_runtime import resolve_queue_runtime
from tools.training_monitor import MonitoredOBBTrainer, create_monitor

_torch_load = torch.load
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


def run(checkpoint, experiment_name):
    checkpoint = str(repo_path(checkpoint))
    queue = resolve_queue_runtime(checkpoint, default_device="0,7")
    monitor = create_monitor(experiment_name)
    model = YOLO(queue.checkpoint, task="obb")
    return monitor.run(
        checkpoint_train, model, queue.checkpoint,
        trainer=MonitoredOBBTrainer,
        data=str(repo_path("data/dronevehicle.yaml")), batch=queue.batch,
        epochs=100, imgsz=640, workers=8, device=queue.device,
        project="DroneVehicle_OBB_FusionTransfer", name=experiment_name,
        exist_ok=False, task="obb", resume=queue.resume or False,
    )
