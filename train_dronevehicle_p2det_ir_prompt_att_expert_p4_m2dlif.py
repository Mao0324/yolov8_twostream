#!/usr/bin/env python3
"""Train P2Det V15 with M2D-LIF DroneVehicle train/val annotations.

M2D-LIF class IDs are remapped to this repository's class order in a temporary
directory. Source images and annotations are never modified. Queue-managed runs
use the Monitor Agent's checkpoint, batch, device, and resume settings.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tools.dronevehicle_m2dlif import install_trusted_torch_load, prepare_temporary_dataset


os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

ROOT = Path(__file__).resolve().parent
CHECKPOINT = (
    ROOT
    / "pre-pth/yolov8s-obb_twostream_p2det_irprompt_attexpert_p4_no_staticmaa_postc2f_v15.pt"
)
LABEL_ROOT = Path(
    "/media/biiteam/新加卷1/biiteam/MCONG/datasets/"
    "M2D-LIFlabels/DroneVehicle_train_val_labels/labels"
)
PROJECT = ROOT / "runs/DroneVehicle_OBB_FusionTransfer"
EXPERIMENT_NAME = "P2D-015_IRPrompt-AttExpert-P4-NoStaticMAA_PostC2f_v15_M2DLIFLabels_v1"


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def validate_ddp_checkpoint(model, checkpoint: str) -> None:
    """Require the model metadata serialized by automatic DDP to name the .pt."""

    expected = resolved(checkpoint)
    if expected.suffix.lower() != ".pt" or not expected.is_file():
        raise FileNotFoundError(f"DDP initialization checkpoint not found: {expected}")

    override = model.overrides.get("model")
    ckpt_path = model.ckpt_path
    if not override or resolved(override) != expected:
        raise RuntimeError(f"model.overrides['model']={override!r}, expected {str(expected)!r}")
    if not ckpt_path or resolved(ckpt_path) != expected:
        raise RuntimeError(f"model.ckpt_path={ckpt_path!r}, expected {str(expected)!r}")


def main():
    from tools.queue_runtime import resolve_queue_runtime

    queue = resolve_queue_runtime(str(CHECKPOINT), default_device="0,1", default_batch=64)
    active_checkpoint = resolved(queue.checkpoint)
    if not active_checkpoint.is_file():
        build_hint = (
            "\nBuild it first with: python -B tools/make_p2det_v15_checkpoint.py"
            if not queue.resume
            else ""
        )
        raise FileNotFoundError(f"active checkpoint not found: {active_checkpoint}{build_hint}")

    # Keep the temporary dataset alive until all DDP children have exited.
    with tempfile.TemporaryDirectory(prefix="dronevehicle_m2dlif_p2det_v15_") as temporary_dir:
        data_yaml = prepare_temporary_dataset(Path(temporary_dir), label_root=LABEL_ROOT)

        import torch
        # This import configures the repository PYTHONPATH before automatic DDP
        # starts and exposes the Monitor-compatible P2Det trainer to each child.
        from tools.training_monitor import MonitoredP2SecondGenOBBTrainer, create_monitor
        from ultralytics import YOLO
        import ultralytics.nn.tasks  # noqa: F401  # Register local two-stream modules.

        install_trusted_torch_load(torch)
        model = YOLO(queue.checkpoint, task="obb")
        validate_ddp_checkpoint(model, queue.checkpoint)
        monitor = create_monitor(EXPERIMENT_NAME)

        result = monitor.run(
            model.train,
            trainer=MonitoredP2SecondGenOBBTrainer,
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
