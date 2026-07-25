# 训练（DroneVehicle ASSAFusion P3/P4/P5，checkpoint-direct DDP 重跑）
# 对应模型 YAML：yaml/yolov8s-ASSAFusion.yaml
import inspect
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"
import torch

from ultralytics import YOLO
import ultralytics.nn.tasks  # noqa: F401
from tools.pretrained_rerun_tracker import repo_path, tracked_train
from tools.queue_runtime import resolve_queue_runtime
from tools.training_monitor import MonitoredOBBTrainer, create_monitor

_torch_load = torch.load
_torch_load_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _torch_load_trusted_checkpoint(*args, **kwargs):
    if _torch_load_supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_trusted_checkpoint

CHECKPOINT = str(repo_path("pre-pth/yolov8s-obb_twostream.pt"))
EXPERIMENT_NAME = "ASSANet_ASSAFusion_P345_H2-4-8_DynK3-FFN2_v1"
QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="0,7")
monitor = create_monitor(EXPERIMENT_NAME)
model = YOLO(QUEUE.checkpoint, task="obb")

results = monitor.run(
    tracked_train,
    model,
    "PT-R002",
    QUEUE.checkpoint,
    trainer=MonitoredOBBTrainer,
    data=str(repo_path("data/dronevehicle.yaml")),
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
