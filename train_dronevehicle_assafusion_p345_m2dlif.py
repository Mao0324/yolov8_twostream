#!/usr/bin/env python3
"""Train ASSAFusion P3/P4/P5 with the M2D-LIF DroneVehicle annotations.

The M2D-LIF train/val labels are read from::

    /media/biiteam/新加卷1/biiteam/MCONG/datasets/
    M2D-LIFlabels/DroneVehicle_train_val_labels/labels/{train,val}

RGB/IR images continue to come from ``DroneVehicle_twostream_3``.  A temporary
paired dataset is built for training because M2D-LIF uses a different class
order.  Source images and annotations are never modified.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from tools.dronevehicle_m2dlif import (
    install_trusted_torch_load,
    prepare_temporary_dataset,
)


os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream.pt"
PROJECT = ROOT / "DroneVehicle_OBB_FusionTransfer"
EXPERIMENT_NAME = "ASSANet_ASSAFusion_P345_H2-4-8_DynK3-FFN2_M2DLIF_v1"
MANUAL_GPU_COUNT = 2
MANUAL_MIN_FREE_MIB = 8192
MANUAL_MIN_FREE_RATIO = 0.80
MANUAL_MAX_UTILIZATION = 10


def _query_gpu_status(timeout: int = 5) -> list[tuple[int, int, int, int]]:
    """Return physical ``(index, total_mib, free_mib, utilization)`` rows."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("automatic GPU selection requires nvidia-smi, but it was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"nvidia-smi did not respond within {timeout} seconds") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"nvidia-smi failed during automatic GPU selection: {detail}")

    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            rows.append(tuple(int(field) for field in fields))
        except ValueError:
            continue
    if not rows:
        raise RuntimeError("nvidia-smi returned no parseable GPU status rows")
    return rows


def _select_idle_manual_gpus() -> str:
    """Select two idle GPUs for a manual run and return Ultralytics device IDs."""

    rows = _query_gpu_status()
    visible_text = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    logical_index = None
    if visible_text:
        visible = [value.strip() for value in visible_text.split(",") if value.strip()]
        if not visible or any(not value.isdigit() for value in visible):
            raise RuntimeError(
                "manual automatic GPU selection requires numeric CUDA_VISIBLE_DEVICES entries, "
                f"got {visible_text!r}"
            )
        logical_index = {int(physical): logical for logical, physical in enumerate(visible)}
        rows = [row for row in rows if row[0] in logical_index]

    idle = [
        row
        for row in rows
        if row[2] >= MANUAL_MIN_FREE_MIB
        and row[2] / row[1] >= MANUAL_MIN_FREE_RATIO
        and row[3] <= MANUAL_MAX_UTILIZATION
    ]
    idle.sort(key=lambda row: (-row[2], row[3], row[0]))

    snapshot = ", ".join(
        f"GPU {index}: {free_mib}/{total_mib} MiB free, util {utilization}%"
        for index, total_mib, free_mib, utilization in sorted(rows)
    )
    if len(idle) < MANUAL_GPU_COUNT:
        raise RuntimeError(
            f"found only {len(idle)} idle GPU(s), but {MANUAL_GPU_COUNT} are required; {snapshot}. "
            f"Idle requires >= {MANUAL_MIN_FREE_MIB} MiB and >= "
            f"{MANUAL_MIN_FREE_RATIO:.0%} memory free with utilization <= "
            f"{MANUAL_MAX_UTILIZATION}%."
        )

    selected = idle[:MANUAL_GPU_COUNT]
    physical_ids = [row[0] for row in selected]
    device_ids = physical_ids if logical_index is None else [logical_index[index] for index in physical_ids]
    print(
        f"Auto-selected idle physical GPUs {','.join(map(str, physical_ids))} "
        f"(Ultralytics device={','.join(map(str, device_ids))}). Current snapshot: {snapshot}"
    )
    return ",".join(map(str, device_ids))


def _assert_checkpoint_direct(model, checkpoint: str) -> None:
    """Fail before DDP if Ultralytics did not retain the resolved .pt path."""

    expected = Path(checkpoint).expanduser().resolve()
    override = Path(str(model.overrides.get("model", ""))).expanduser().resolve()
    ckpt_path = Path(str(model.ckpt_path or "")).expanduser().resolve()
    if expected.suffix != ".pt" or override != expected or ckpt_path != expected:
        raise RuntimeError(
            "DDP requires checkpoint-direct startup: "
            f"expected={expected}, overrides.model={override}, ckpt_path={ckpt_path}"
        )


def main() -> None:
    from tools.queue_runtime import resolve_queue_runtime

    # Monitor Agent jobs keep their CUDA_VISIBLE_DEVICES allocation. Only a
    # genuinely manual run queries the host and chooses two idle GPUs. Resolve
    # this before preparing thousands of temporary labels so a busy host fails
    # fast without unnecessary data work.
    default_device = "0,1" if os.getenv("YOLO_QUEUE_JOB_ID") else _select_idle_manual_gpus()
    queue = resolve_queue_runtime(str(CHECKPOINT), default_device=default_device, default_batch=64)
    active_checkpoint = Path(queue.checkpoint).expanduser().resolve()
    if not active_checkpoint.is_file():
        raise FileNotFoundError(
            f"resolved checkpoint not found: {active_checkpoint}\n"
            "Create it first with: python tools/make_assafusion_p345_checkpoint.py"
        )

    # Keep the temporary labels, manifests and YAML alive until all DDP workers
    # have completed. TemporaryDirectory also cleans them up after an error.
    with tempfile.TemporaryDirectory(prefix="dronevehicle_m2dlif_assafusion_p345_") as temporary_dir:
        data_yaml = prepare_temporary_dataset(Path(temporary_dir))

        import torch
        # This import prepares repository PYTHONPATH for Ultralytics' DDP
        # temporary script before a multi-GPU trainer is constructed.
        from tools.training_monitor import MonitoredOBBTrainer, create_monitor
        from ultralytics import YOLO
        import ultralytics.nn.tasks  # noqa: F401  # Register local two-stream modules.

        install_trusted_torch_load(torch)
        monitor = create_monitor(EXPERIMENT_NAME)

        # Construct directly from the resolved .pt so automatic DDP children
        # load the same pretrained checkpoint instead of random YAML weights.
        model = YOLO(queue.checkpoint, task="obb")
        _assert_checkpoint_direct(model, queue.checkpoint)

        monitor.run(
            model.train,
            trainer=MonitoredOBBTrainer,
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


if __name__ == "__main__":
    main()
