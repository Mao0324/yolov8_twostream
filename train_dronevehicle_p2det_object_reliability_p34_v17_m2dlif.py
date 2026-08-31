#!/usr/bin/env python3
"""Train P2D-017 true object-level RGB/IR reliability on M2D-LIF labels."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tools.dronevehicle_m2dlif import install_trusted_torch_load, prepare_temporary_dataset
from train_dronevehicle_p2det_dual_reliability_p34_v16_m2dlif import (
    LABEL_ROOT,
    PROJECT,
    require_teachers,
    resolved,
    validate_ddp_checkpoint,
)


os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_p2det_objectreliability_p34_monotonicpriorlaf_v17.pt"
EXPERIMENT_NAME = "P2D-017_ObjectReliabilityPrompt-P34-MonotonicPriorLAF-NoGDER_v17_M2DLIFLabels_v2"


def main():
    from tools.queue_runtime import resolve_queue_runtime

    require_teachers()
    if not LABEL_ROOT.is_dir():
        raise FileNotFoundError(f"M2D-LIF DroneVehicle labels not found: {LABEL_ROOT}")
    queue = resolve_queue_runtime(str(CHECKPOINT), default_device="auto", default_batch=64)
    active_checkpoint = resolved(queue.checkpoint)
    if not active_checkpoint.is_file():
        hint = "\nBuild it first with: python -B tools/make_p2det_v17_checkpoint.py" if not queue.resume else ""
        raise FileNotFoundError(f"active checkpoint not found: {active_checkpoint}{hint}")

    # Keep remapped labels and YAML alive until all automatic DDP children exit.
    with tempfile.TemporaryDirectory(prefix="dronevehicle_m2dlif_p2det_v17_") as temporary_dir:
        data_yaml = prepare_temporary_dataset(Path(temporary_dir), label_root=LABEL_ROOT)

        import torch
        # Configure repository PYTHONPATH before Ultralytics creates its DDP script.
        from tools.training_monitor import MonitoredP2ObjectReliabilityOBBTrainer, create_monitor
        from ultralytics import YOLO
        import ultralytics.nn.tasks  # noqa: F401

        install_trusted_torch_load(torch)
        # DDP children must deserialize this embedded V17 .pt. Never use
        # YOLO(YAML)+load(checkpoint), which would random-initialize children.
        model = YOLO(queue.checkpoint, task="obb")
        validate_ddp_checkpoint(model, queue.checkpoint)
        monitor = create_monitor(EXPERIMENT_NAME)
        result = monitor.run(
            model.train,
            trainer=MonitoredP2ObjectReliabilityOBBTrainer,
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
