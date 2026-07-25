"""Resolve training settings supplied by the experiment queue Agent."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class QueueRuntime:
    """Effective settings for either Agent-managed or manual training."""

    checkpoint: str
    batch: int
    resume: str
    device: str


def resolve_queue_runtime(
    checkpoint: str,
    *,
    default_device: str,
    default_batch: int = 64,
) -> QueueRuntime:
    """Apply queue overrides while preserving the script's manual defaults."""
    batch_text = (os.getenv("YOLO_QUEUE_BATCH") or "").strip()
    try:
        batch = int(batch_text) if batch_text else default_batch
    except ValueError as exc:
        raise ValueError(f"YOLO_QUEUE_BATCH must be an integer, got {batch_text!r}") from exc
    if batch <= 0:
        raise ValueError(f"YOLO_QUEUE_BATCH must be positive, got {batch}")

    resume = os.getenv("YOLO_QUEUE_RESUME_CHECKPOINT", "").strip()
    active_checkpoint = resume or checkpoint

    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if os.getenv("YOLO_QUEUE_JOB_ID") and visible_devices:
        physical_devices = [value.strip() for value in visible_devices.split(",") if value.strip()]
        if not physical_devices:
            raise ValueError("CUDA_VISIBLE_DEVICES contains no GPU IDs")
        device = ",".join(str(index) for index in range(len(physical_devices)))
    else:
        device = default_device

    return QueueRuntime(
        checkpoint=active_checkpoint,
        batch=batch,
        resume=resume,
        device=device,
    )
