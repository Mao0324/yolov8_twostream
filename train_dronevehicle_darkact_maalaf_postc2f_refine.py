# 训练（DroneVehicle DarkAct post-C2f StaticMAA2D + zero-init refine + LAF feedback）
# 对应模型 YAML：yaml/yolov8s-DarkAct-MAA2D-Refine-LAFMerge-P345-R4-PostC2f-v1.yaml
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

# 直接从该 Refine 架构的迁移 checkpoint 启动。
CHECKPOINT = str(repo_path("pre-pth/yolov8s-obb_twostream_darkact_staticmaa_refine_laffeedback_postc2f_p345_e025_v1.pt"))
EXPERIMENT_NAME = "DarkAct_StaticMAA2DRefineLAFMergeFeedback_P345_H2-4-8_PostC2f-RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v1"
QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="4,5")
monitor = create_monitor(EXPERIMENT_NAME)
model = YOLO(QUEUE.checkpoint, task="obb")

# project/name 遵循命名.txt：
# {Paper}_{Module}_{Stages}_{Heads}_{Structure}_{Version}
results = monitor.run(
    tracked_train,
    model,
    "PT-R011",
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
