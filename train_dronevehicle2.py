# 训练（DroneVehicle Partial-Channel ASSA P3+P4）
# 对应模型 YAML：yaml/yolov8s-PartialChannelASSAFusion-P34-R4-StaticDW-NoFFN.yaml
import inspect

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

# 直接从迁移 checkpoint 启动，确保双卡 DDP 子进程加载权重。
CHECKPOINT = str(repo_path("pre-pth/yolov8s-obb_twostream_partialchannel_assafusion_p34_r4_staticdw_noffn.pt"))
EXPERIMENT_NAME = "ASSANet_PartialChannelASSAFusion_P34_H2-4_R4-StaticDW-NoFFN_v1"
QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="1,2")
monitor = create_monitor(EXPERIMENT_NAME)
model = YOLO(QUEUE.checkpoint, task='obb')

# 训练正常完成后自动更新重跑记录表。
results = monitor.run(
    tracked_train,
    model,
    "PT-R004",
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
    task='obb',
    resume=QUEUE.resume or False,
)
