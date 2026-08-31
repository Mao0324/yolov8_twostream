# 训练（DroneVehicle）
from pathlib import Path

import torch

from ultralytics import YOLO
import ultralytics.nn.tasks  # noqa: F401

_torch_load = torch.load


def _torch_load_trusted_checkpoint(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_trusted_checkpoint

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_baseline.pt"

# 多卡 DDP 必须直接从内嵌目标结构的 .pt 启动。
model = YOLO(str(CHECKPOINT), task="obb")

from tools.experiment_layout import organize_train_args

train_args = organize_train_args("train", {
    "data": str(ROOT / "data/dronevehicle.yaml"),
    "model": str(CHECKPOINT),
    "batch": 64,
    "epochs": 100,
    "imgsz": 640,
    "workers": 8,
    "device": "3,5",
    "project": "runs_baseline",
    "name": "train",
    "task": "obb",
})
results = model.train(**train_args)
