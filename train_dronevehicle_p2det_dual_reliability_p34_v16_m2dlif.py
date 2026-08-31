#!/usr/bin/env python3
"""Train P2D-016 with M2D-LIF DroneVehicle train/val annotations.

M2D-LIF class IDs are remapped to this repository's class order in a temporary
directory. Source images and annotations are never modified. The two frozen
single-modality teachers and the student must use the same remapped class order.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tools.dronevehicle_m2dlif import install_trusted_torch_load, prepare_temporary_dataset


os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_p2det_dualreliability_p34_priorlaf_v16.pt"
LABEL_ROOT = Path(
    "/media/biiteam/新加卷1/biiteam/MCONG/datasets/"
    "M2D-LIFlabels/DroneVehicle_train_val_labels/labels"
)
PROJECT = ROOT / "runs/DroneVehicle_OBB_FusionTransfer"
EXPERIMENT_NAME = "P2D-016_DualReliabilityPrompt-P34-PriorLAF-NoGDER_v16_M2DLIFLabels_v1"
TEACHER_EXPERIMENTS = {
    "P2DET_V16_RGB_TEACHER": "TCH-001__rgb-only",
    "P2DET_V16_IR_TEACHER": "TCH-002__ir-only",
}


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def validate_ddp_checkpoint(model, checkpoint: str | Path) -> None:
    """Require the model metadata serialized by automatic DDP to name the .pt."""

    expected = resolved(checkpoint)
    if expected.suffix.lower() != ".pt" or not expected.is_file():
        raise FileNotFoundError(f"DDP initialization checkpoint not found: {expected}")
    override, ckpt_path = model.overrides.get("model"), model.ckpt_path
    if not override or resolved(override) != expected:
        raise RuntimeError(f"model.overrides['model']={override!r}, expected {str(expected)!r}")
    if not ckpt_path or resolved(ckpt_path) != expected:
        raise RuntimeError(f"model.ckpt_path={ckpt_path!r}, expected {str(expected)!r}")


def discover_teacher(environment_name: str) -> Path:
    """Resolve an explicit teacher or the newest registered M2D-LIF attempt."""

    explicit = os.getenv(environment_name, "").strip()
    if explicit:
        return resolved(explicit)
    from tools.experiment_layout import latest_run

    run = latest_run(TEACHER_EXPERIMENTS[environment_name], train_labels="m2dlif-v1", require="weights/best.pt")
    return run / "weights/best.pt"


def require_teachers() -> dict[str, Path]:
    """Discover two distinct single-modality teachers and export them to DDP children."""

    resolved_paths = {
        "P2DET_V16_RGB_TEACHER": discover_teacher("P2DET_V16_RGB_TEACHER"),
        "P2DET_V16_IR_TEACHER": discover_teacher("P2DET_V16_IR_TEACHER"),
    }
    for name, path in resolved_paths.items():
        if path.suffix.lower() != ".pt" or not path.is_file():
            raise FileNotFoundError(
                f"{name} not found: {path}. Train both single-modality teachers first."
            )
        os.environ[name] = str(path)
    if len(set(resolved_paths.values())) != 2:
        raise RuntimeError("RGB-only and IR-only teachers must be distinct checkpoints")
    print(f"RGB-only teacher: {resolved_paths['P2DET_V16_RGB_TEACHER']}")
    print(f"IR-only teacher:  {resolved_paths['P2DET_V16_IR_TEACHER']}")
    return resolved_paths


def main():
    from tools.queue_runtime import resolve_queue_runtime

    require_teachers()
    if not LABEL_ROOT.is_dir():
        raise FileNotFoundError(f"M2D-LIF DroneVehicle labels not found: {LABEL_ROOT}")
    queue = resolve_queue_runtime(str(CHECKPOINT), default_device="auto", default_batch=64)
    active_checkpoint = resolved(queue.checkpoint)
    if not active_checkpoint.is_file():
        hint = (
            "\nBuild it first with: python -B tools/make_p2det_v16_checkpoint.py"
            if not queue.resume
            else ""
        )
        raise FileNotFoundError(f"active checkpoint not found: {active_checkpoint}{hint}")

    # Keep remapped labels, manifests, and the temporary YAML alive until all
    # automatic DDP children have exited.
    with tempfile.TemporaryDirectory(prefix="dronevehicle_m2dlif_p2det_v16_") as temporary_dir:
        data_yaml = prepare_temporary_dataset(Path(temporary_dir), label_root=LABEL_ROOT)

        import torch
        # Configure repository PYTHONPATH before Ultralytics creates its DDP script.
        from tools.training_monitor import MonitoredP2ReliabilityOBBTrainer, create_monitor
        from ultralytics import YOLO
        import ultralytics.nn.tasks  # noqa: F401

        install_trusted_torch_load(torch)
        model = YOLO(queue.checkpoint, task="obb")
        validate_ddp_checkpoint(model, queue.checkpoint)
        monitor = create_monitor(EXPERIMENT_NAME)
        result = monitor.run(
            model.train,
            trainer=MonitoredP2ReliabilityOBBTrainer,
            data=str(data_yaml),
            batch=queue.batch,
            epochs=100,
            imgsz=640,
            workers=8,
            device=queue.device,
            project=str(PROJECT),
            name=EXPERIMENT_NAME,
            exist_ok=False,
            task="obb",
            resume=queue.resume or False,
        )

    print("Temporary M2D-LIF labels, manifests, YAML, and dataset caches removed.")
    return result


if __name__ == "__main__":
    main()
