# 训练 DA-014：旋转 soft-centerness、P3/P4-only、gain warmup 与 gate 诊断。
# 对应 YAML：
# yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P34-SoftCenterness-Warmup-v5.yaml
import inspect
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"
import torch

from ultralytics import YOLO
from tools.pretrained_rerun_tracker import repo_path, tracked_train
from tools.queue_runtime import resolve_queue_runtime
from tools.training_monitor import (
    MonitoredSoftCenternessTargetSaliencyOBBTrainer,
    create_monitor,
)

_torch_load = torch.load
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load

CHECKPOINT = str(
    repo_path("pre-pth/yolov8s-obb_twostream_darkact_target_saliency_soft_centerness_p34_warmup_v5.pt")
)
EXPERIMENT_NAME = (
    "DarkAct_TargetSaliencyPaperLAF_P34SoftCenterness-W1-0p5-"
    "Gain0p025-Warmup10-GateStats_v5"
)
QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="0,7")
monitor = create_monitor(EXPERIMENT_NAME)
model = YOLO(QUEUE.checkpoint, task="obb")

results = monitor.run(
    tracked_train,
    model,
    "PT-R020",
    QUEUE.checkpoint,
    trainer=MonitoredSoftCenternessTargetSaliencyOBBTrainer,
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
