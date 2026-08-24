#!/usr/bin/env python3
"""Train three-class FLIR DarkAct SemanticDisagreementLAF P3/P4 scale-n HBB."""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("COMET_MODE", "DISABLED")

import torch

from ultralytics import YOLO
import ultralytics.nn.tasks  # noqa: F401  # Register custom two-stream modules.
from tools.prepare_flir_align_hbb import DEFAULT_SOURCE, prepare_dataset, validate_prepared_dataset
from tools.queue_runtime import resolve_queue_runtime
from tools.training_monitor import MonitoredDetectionTrainer, create_monitor


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/flir_align.yaml"
CHECKPOINT = ROOT / "pre-pth/yolov8n-hbb_twostream_darkact_semantic_disagreement_laf_p34_flir_3class_scalen.pt"
PROJECT = ROOT / "runs/FLIR_Align_HBB_FusionTransfer"
EXPERIMENT_NAME = "FLIR_DarkAct_SemanticDisagreementLAF_P34_R4_NoStaticMAA_scalen_3Class_HBB"

_torch_load = torch.load
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE, help="original flir_align root")
    parser.add_argument("--weights", type=Path, default=CHECKPOINT, help="migrated HBB checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--name", default=EXPERIMENT_NAME)
    parser.add_argument("--resume", type=Path, help="resume from a previous last.pt checkpoint")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true", help="use an already prepared data/flir_align_hbb")
    return parser.parse_args()


def main():
    """Launch a manual or Agent queue-managed monitored FLIR HBB run."""

    args = _parse_args()
    if not args.skip_prepare:
        manifest = prepare_dataset(args.source_root, ROOT / "data/flir_align_hbb")
        print(
            "[DATA] prepared FLIR pairs: "
            + ", ".join(
                f"{split}={stats['images']} images/{stats['boxes']} boxes"
                for split, stats in manifest["splits"].items()
            )
        )
    else:
        validate_prepared_dataset(ROOT / "data/flir_align_hbb")

    requested_checkpoint = (args.resume or args.weights).expanduser().resolve()
    queue = resolve_queue_runtime(
        str(requested_checkpoint),
        default_device=args.device,
        default_batch=args.batch,
    )
    queue_checkpoint = Path(queue.checkpoint).expanduser().resolve()
    if not queue_checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {queue_checkpoint}\n"
            "Run: python tools/make_twostream_hbb_weights_flir.py"
        )
    # A manual --resume path takes the same resume route as the Queue Agent's
    # YOLO_QUEUE_RESUME_CHECKPOINT value.
    resume_checkpoint = queue.resume or (str(args.resume.expanduser().resolve()) if args.resume else "")
    monitor = create_monitor(args.name)
    model = YOLO(str(queue_checkpoint), task="detect")
    return monitor.run(
        model.train,
        trainer=MonitoredDetectionTrainer,
        data=str(DATA),
        batch=queue.batch,
        epochs=args.epochs,
        imgsz=args.imgsz,
        workers=args.workers,
        device=queue.device,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        task="detect",
        resume=resume_checkpoint or False,
    )


if __name__ == "__main__":
    results = main()
