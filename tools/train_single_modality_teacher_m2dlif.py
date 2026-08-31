"""Shared DDP-safe entry logic for RGB-only and IR-only M2D-LIF OBB teachers."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from tools.dronevehicle_m2dlif import install_trusted_torch_load, prepare_temporary_dataset


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb.pt"
LABEL_ROOT = Path(
    "/media/biiteam/新加卷1/biiteam/MCONG/datasets/"
    "M2D-LIFlabels/DroneVehicle_train_val_labels/labels"
)
PROJECT = ROOT / "runs/DroneVehicle_OBB_SingleModalityTeachers"
EXPERIMENT_NAMES = {
    "rgb": "P2D-Teacher_RGBOnly_M2DLIFLabels_v1",
    "ir": "P2D-Teacher_IROnly_M2DLIFLabels_v1",
}


def resolved(path):
    return Path(path).expanduser().resolve()


def validate_ddp_checkpoint(model, checkpoint):
    """Require automatic DDP to serialize the actual pretrained .pt path."""
    expected = resolved(checkpoint)
    if expected.suffix.lower() != ".pt" or not expected.is_file():
        raise FileNotFoundError(f"teacher initialization checkpoint not found: {expected}")
    override, ckpt_path = model.overrides.get("model"), model.ckpt_path
    if not override or resolved(override) != expected:
        raise RuntimeError(f"model.overrides['model']={override!r}, expected {str(expected)!r}")
    if not ckpt_path or resolved(ckpt_path) != expected:
        raise RuntimeError(f"model.ckpt_path={ckpt_path!r}, expected {str(expected)!r}")


def train_single_modality_teacher(modality):
    """Train one 3-channel teacher while preserving queue and monitor behavior."""
    modality = str(modality).lower()
    if modality not in EXPERIMENT_NAMES:
        raise ValueError(f"unsupported teacher modality {modality!r}")
    os.environ["P2DET_SINGLE_MODALITY_TEACHER"] = modality
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_MODE"] = "DISABLED"

    from tools.queue_runtime import resolve_queue_runtime

    queue = resolve_queue_runtime(str(CHECKPOINT), default_device="auto", default_batch=64)
    active_checkpoint = resolved(queue.checkpoint)
    if not active_checkpoint.is_file():
        raise FileNotFoundError(f"active checkpoint not found: {active_checkpoint}")
    if not LABEL_ROOT.is_dir():
        raise FileNotFoundError(f"M2D-LIF DroneVehicle labels not found: {LABEL_ROOT}")

    with tempfile.TemporaryDirectory(prefix=f"dronevehicle_m2dlif_{modality}_teacher_") as temporary_dir:
        data_yaml = prepare_temporary_dataset(Path(temporary_dir), label_root=LABEL_ROOT)

        import torch
        from tools.training_monitor import MonitoredSingleModalityOBBTrainer, create_monitor
        from ultralytics import YOLO

        install_trusted_torch_load(torch)
        model = YOLO(queue.checkpoint, task="obb")
        validate_ddp_checkpoint(model, queue.checkpoint)
        experiment_name = EXPERIMENT_NAMES[modality]
        monitor = create_monitor(experiment_name)
        result = monitor.run(
            model.train,
            trainer=MonitoredSingleModalityOBBTrainer,
            data=str(data_yaml),
            batch=queue.batch,
            epochs=100,
            imgsz=640,
            workers=8,
            device=queue.device,
            project=str(PROJECT),
            name=experiment_name,
            exist_ok=False,
            task="obb",
            resume=queue.resume or False,
        )

    print(f"Temporary M2D-LIF files for the {modality.upper()} teacher removed.")
    return result


__all__ = (
    "CHECKPOINT",
    "EXPERIMENT_NAMES",
    "LABEL_ROOT",
    "PROJECT",
    "train_single_modality_teacher",
    "validate_ddp_checkpoint",
)
