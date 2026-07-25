# 训练（DroneVehicle DarkAct StaticMAA2D + full-C paper LAF feedback V3）
# 对应模型 YAML：yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4-v3.yaml
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

# 直接从 V3 迁移 checkpoint 启动，确保双卡 DDP 子进程加载权重。
CHECKPOINT = str(repo_path("pre-pth/yolov8s-obb_twostream_darkact_staticmaa_paperlaf_fullc_p345_v3.pt"))
EXPERIMENT_NAME = "DarkAct_PaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-StaticMAA_v3"
QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="1,3")
monitor = create_monitor(EXPERIMENT_NAME)
model = YOLO(QUEUE.checkpoint, task="obb")

# project/name 严格遵循命名.txt：
#    {Paper}_{Module}_{Stages}_{Heads}_{Structure}_{Version}
results = monitor.run(
    tracked_train,
    model,
    "PT-R009",
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
