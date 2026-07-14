# 训练（DroneVehicle）
import torch

from ultralytics import YOLO
import ultralytics.nn.tasks  # noqa: F401

_torch_load = torch.load


def _torch_load_trusted_checkpoint(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_trusted_checkpoint

# 1) 模型结构
model = YOLO('/media/biiteam/新加卷/biiteam/MCONG/Yolov8_TwoStream/yaml/baseline.yaml')

# 2) 预训练权重（如不存在可注释掉）
model.load('/media/biiteam/新加卷/biiteam/MCONG/Yolov8_TwoStream/pre-pth/yolov8s-obb_twostream_baseline.pt')

# 3) 训练
results = model.train(
    data='/media/biiteam/新加卷/biiteam/MCONG/Yolov8_TwoStream/data/dronevehicle.yaml',
    batch=64,
    epochs=100,
    imgsz=640,
    workers=8,
    device='3,5',
    project="runs_baseline",
    task='obb'
)
