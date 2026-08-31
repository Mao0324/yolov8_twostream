"""Resolve training settings supplied by the experiment queue Agent."""

from dataclasses import dataclass
import os
import subprocess


@dataclass(frozen=True)
class QueueRuntime:
    """Effective settings for either Agent-managed or manual training."""

    checkpoint: str
    batch: int
    resume: str
    device: str


@dataclass(frozen=True)
class GPUStatus:
    """One physical GPU snapshot returned by nvidia-smi."""

    index: int
    utilization: int
    memory_used_mib: int
    memory_free_mib: int
    memory_total_mib: int


def query_gpu_status(timeout=5):
    """Query physical GPU utilization and memory without importing torch."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("automatic GPU selection requires nvidia-smi") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("nvidia-smi timed out during automatic GPU selection") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"nvidia-smi failed during automatic GPU selection: {detail}")

    statuses = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            statuses.append(GPUStatus(*(int(field) for field in fields)))
        except ValueError:
            continue
    if not statuses:
        raise RuntimeError("nvidia-smi returned no parseable GPU rows")
    return statuses


def select_idle_gpus(count=2):
    """Select truly idle physical GPUs and return an Ultralytics device string."""
    count = int(os.getenv("YOLO_AUTO_GPU_COUNT", count))
    min_free = int(os.getenv("YOLO_AUTO_GPU_MIN_FREE_MIB", "12000"))
    max_used = int(os.getenv("YOLO_AUTO_GPU_MAX_USED_MIB", "2048"))
    max_utilization = int(os.getenv("YOLO_AUTO_GPU_MAX_UTILIZATION", "10"))
    if min(count, min_free, max_used) <= 0 or max_utilization < 0:
        raise ValueError("automatic GPU selection thresholds are invalid")

    statuses = query_gpu_status()
    eligible = [
        gpu
        for gpu in statuses
        if gpu.memory_free_mib >= min_free
        and gpu.memory_used_mib <= max_used
        and gpu.utilization <= max_utilization
    ]
    eligible.sort(key=lambda gpu: (-gpu.memory_free_mib, gpu.utilization, gpu.index))
    snapshot = "; ".join(
        f"GPU {gpu.index}: util={gpu.utilization}%, used={gpu.memory_used_mib} MiB, "
        f"free={gpu.memory_free_mib} MiB/{gpu.memory_total_mib} MiB"
        for gpu in sorted(statuses, key=lambda item: item.index)
    )
    if len(eligible) < count:
        raise RuntimeError(
            f"Need {count} idle GPUs but found {len(eligible)} with free>={min_free} MiB, "
            f"used<={max_used} MiB, utilization<={max_utilization}%. {snapshot}"
        )
    selected = eligible[:count]
    device = ",".join(str(gpu.index) for gpu in selected)
    print(f"Auto-selected idle physical GPUs: {device}. {snapshot}")
    return device


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
    elif str(default_device).strip().lower() == "auto":
        device = select_idle_gpus()
    else:
        device = default_device

    return QueueRuntime(
        checkpoint=active_checkpoint,
        batch=batch,
        resume=resume,
        device=device,
    )


__all__ = (
    "GPUStatus",
    "QueueRuntime",
    "query_gpu_status",
    "resolve_queue_runtime",
    "select_idle_gpus",
)
