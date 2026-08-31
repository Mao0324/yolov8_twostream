"""Train yolov8s-ASSAFusion P3/P4/P5 with DroneVehicle and Monitor."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

import torch

from tools.queue_runtime import resolve_queue_runtime
# This import prepares repository PYTHONPATH for Ultralytics' DDP temporary
# script before any multi-GPU trainer is constructed.
from tools.training_monitor import MonitoredOBBTrainer, create_monitor
from ultralytics import YOLO
import ultralytics.nn.tasks  # noqa: F401  # Register local two-stream modules.


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream.pt"
EXPERIMENT_NAME = "ASSANet_ASSAFusion_P345_H2-4-8_DynK3-FFN2_v3"

_torch_load = torch.load
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


def _assert_checkpoint_direct(model: YOLO, checkpoint: str) -> None:
    """Fail before DDP if Ultralytics did not retain the real .pt path."""
    expected = Path(checkpoint).expanduser().resolve()
    override = Path(str(model.overrides.get("model", ""))).expanduser().resolve()
    ckpt_path = Path(str(model.ckpt_path or "")).expanduser().resolve()
    if expected.suffix != ".pt" or override != expected or ckpt_path != expected:
        raise RuntimeError(
            "DDP requires checkpoint-direct startup: "
            f"expected={expected}, overrides.model={override}, ckpt_path={ckpt_path}"
        )


def main() -> None:
    # Manual default uses an idle pair on this host. Monitor Agent jobs still
    # use their CUDA_VISIBLE_DEVICES allocation through resolve_queue_runtime().
    queue = resolve_queue_runtime(str(CHECKPOINT), default_device="3,4")
    active_checkpoint = Path(queue.checkpoint).expanduser().resolve()
    if not active_checkpoint.is_file():
        raise FileNotFoundError(
            f"resolved checkpoint not found: {active_checkpoint}\n"
            "Create it first with: python tools/make_assafusion_p345_checkpoint.py"
        )

    monitor = create_monitor(EXPERIMENT_NAME)

    # Always construct from the resolved .pt. YOLO(yaml) + load(pt) loses the
    # parent-side loaded weights when Ultralytics serializes a DDP child script.
    model = YOLO(queue.checkpoint, task="obb")
    _assert_checkpoint_direct(model, queue.checkpoint)

    monitor.run(
        model.train,
        trainer=MonitoredOBBTrainer,
        data=str(ROOT / "data/dronevehicle.yaml"),
        batch=queue.batch,
        epochs=100,
        imgsz=640,
        workers=8,
        device=queue.device,
        project=str(ROOT / "DroneVehicle_OBB_FusionTransfer"),
        name=EXPERIMENT_NAME,
        exist_ok=False,
        task="obb",
        resume=queue.resume or False,
    )


if __name__ == "__main__":
    main()
