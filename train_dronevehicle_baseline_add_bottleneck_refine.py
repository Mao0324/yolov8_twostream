# 训练（DroneVehicle baseline ADD + P3/P4/P5 BottleneckRefine）
# 对应模型 YAML：yaml/yolov8s-baseline-ADD-P345-BottleneckRefine.yaml
import inspect

import torch
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"
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

# 直接从迁移 checkpoint 启动，checkpoint 内嵌该 BottleneckRefine YAML。
CHECKPOINT = str(repo_path("pre-pth/yolov8s-obb_twostream_baseline_add_p345_bottleneck_refine.pt"))
EXPERIMENT_NAME = "YOLOv8_BottleneckRefine_P345_HNA_E0p5-K1-3_v1"
QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="0,7")
monitor = create_monitor(EXPERIMENT_NAME)
model = YOLO(QUEUE.checkpoint, task="obb")

# 3) 使用 6、7 卡训练。
# 命名字段：YOLOv8（来源）_BottleneckRefine（模块）_P345（尺度）_
# HNA（无注意力头）_E0p5-K1-3（e=0.5、卷积核 1×1/3×3）_v1（版本）。
results = monitor.run(
    tracked_train,
    model,
    "PT-R006",
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
