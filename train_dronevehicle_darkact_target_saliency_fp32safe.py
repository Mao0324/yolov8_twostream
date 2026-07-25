# 训练（DroneVehicle 目标显著性 PaperLAF，仅修复 attention FP16 溢出）
# 对应模型 YAML：yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-FP32Safe-v2.yaml
import inspect
import os
import sys
from pathlib import Path

if not os.getenv("YOLO_MONITOR_URL"):
    os.environ["YOLO_MONITOR_URL"] = "https://monitor.maocong.me"

os.environ["YOLO_MONITOR_USE_PROXY"] = "false"

# Ultralytics DDP 子进程会重新 import Trainer；同时写入 PYTHONPATH
# 才能让 monitor/ 中的可导入 Trainer 在子进程继续可见。
ROOT = Path(__file__).resolve().parent
MONITOR_DIR = ROOT / "monitor"
required_paths = (str(ROOT), str(MONITOR_DIR))
for path in reversed(required_paths):
    if path not in sys.path:
        sys.path.insert(0, path)
pythonpath_entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
pythonpath_entries = [entry for entry in pythonpath_entries if entry not in required_paths]
os.environ["PYTHONPATH"] = os.pathsep.join((*required_paths, *pythonpath_entries))

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"
import torch

from ultralytics import YOLO
from monitored_target_saliency_trainer import MonitoredTargetSaliencyOBBTrainer
from tools.pretrained_rerun_tracker import repo_path, tracked_train
from tools.queue_runtime import resolve_queue_runtime
from yolo_monitor import YoloExperimentMonitor

_torch_load = torch.load
_torch_load_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _torch_load_trusted_checkpoint(*args, **kwargs):
    if _torch_load_supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_trusted_checkpoint

# checkpoint 内嵌 FP32Safe V2 YAML，DDP 子进程会直接重载它。
CHECKPOINT = str(repo_path("pre-pth/yolov8s-obb_twostream_darkact_target_saliency_paperlaf_p345_fp32safe_v2.pt"))
EXPERIMENT_NAME = (
    "DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_"
    "FullC-DilK3-PoolK3-OBBMaskS-FP32Attn_v2"
)
QUEUE = resolve_queue_runtime(CHECKPOINT, default_device="4,5")

if not os.getenv("YOLO_MONITOR_URL") or not os.getenv("YOLO_MONITOR_TOKEN"):
    raise RuntimeError("Set YOLO_MONITOR_URL and YOLO_MONITOR_TOKEN before training")

model = YOLO(QUEUE.checkpoint, task="obb")
monitor = YoloExperimentMonitor(experiment_name=EXPERIMENT_NAME)

# 父进程先创建远程实验并将同一 run_id 传给 DDP 子进程；
# 子进程中只有 rank 0 会上报进度。
results = monitor.run(
    tracked_train,
    model,
    "PT-R015",
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
