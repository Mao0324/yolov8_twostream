# 训练（DroneVehicle DarkAct 目标显著性监督 + full-C PaperLAF feedback）
# 对应模型 YAML：yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-v1.yaml
import inspect
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"
import torch

from ultralytics import YOLO
from tools.pretrained_rerun_tracker import repo_path, tracked_train
from tools.queue_runtime import resolve_queue_runtime
from tools.training_monitor import MonitoredTargetSaliencyOBBTrainer, create_monitor

_torch_load = torch.load
_torch_load_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _torch_load_trusted_checkpoint(*args, **kwargs):
    if _torch_load_supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_trusted_checkpoint

# 1) 迁移 checkpoint 内已嵌入新 YAML；直接从 pt 启动，确保 DDP 子进程也加载迁移权重。
#    其结构为 Post-C2f/SPPF F -> StaticMAAContext2D -> 单通道目标显著性 S。
CHECKPOINT = str(repo_path("pre-pth/yolov8s-obb_twostream_darkact_target_saliency_paperlaf_p345_v1.pt"))
EXPERIMENT_NAME = "DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS_v1"
QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="4,5")
monitor = create_monitor(EXPERIMENT_NAME)
model = YOLO(QUEUE.checkpoint, task="obb")

# 2) 专用 Trainer 将 OBB 标注在 P3/P4/P5 尺度栅格化，单独记录 saliency_loss。
#    project/name 严格遵循命名.txt：
#    {Paper}_{Module}_{Stages}_{Heads}_{Structure}_{Version}
#‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️
#❗模型从第 7 轮开始真实发生了数值发散。全是nan，不用这个训练，转去 yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-FP32Safe-v2.yaml
#‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️

results = monitor.run(
    tracked_train,
    model,
    "PT-R014",
    QUEUE.checkpoint,
    trainer=MonitoredTargetSaliencyOBBTrainer,
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
#‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️
#❗模型从第 7 轮开始真实发生了数值发散。全是nan，不用这个训练，转去 yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-FP32Safe-v2.yaml
#‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️
