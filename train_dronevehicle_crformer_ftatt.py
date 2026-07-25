# 训练（DroneVehicle CR-former FTCrossMerge P3）
# 对应模型 YAML：yaml/yolov8s-CRFormer-FTCrossMerge-P3-R4.yaml
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

# 直接从迁移 checkpoint 启动，确保双卡 DDP 加载权重。
CHECKPOINT = str(repo_path("pre-pth/yolov8s-obb_twostream_crformer_ftcrossmerge_p3_r4.pt"))
EXPERIMENT_NAME = "CRFormer_FTCrossMerge_P3_H2_R4-P2-StaticDW-NoFFN_v1"
QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="6,7")
monitor = create_monitor(EXPERIMENT_NAME)
model = YOLO(QUEUE.checkpoint, task="obb")

# 3) 实验名遵循《命名.txt》：Paper_Module_Stages_Heads_Structure_Version。
results = monitor.run(
    tracked_train,
    model,
    "PT-R005",
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
