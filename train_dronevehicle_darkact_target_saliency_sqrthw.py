# 训练（DroneVehicle 目标显著性 PaperLAF，FP32 attention + sqrt(HW) 缩放）
# 对应模型 YAML：yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-SqrtHW-v3.yaml
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
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load

CHECKPOINT = str(repo_path("pre-pth/yolov8s-obb_twostream_darkact_target_saliency_paperlaf_p345_sqrthw_v3.pt"))
EXPERIMENT_NAME = "DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn-SqrtHW_v3"
QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="0,7")
monitor = create_monitor(EXPERIMENT_NAME)
model = YOLO(QUEUE.checkpoint, task="obb")

results = monitor.run(
    tracked_train,
    model,
    "PT-R016",
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
