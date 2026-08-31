"""Shared bootstrap for DDP-safe monitoring in repository training scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MONITOR_DIR = ROOT / "monitor"


def _bootstrap_monitor_imports() -> None:
    """Make repository and monitor modules importable in parent/DDP processes."""
    os.environ.setdefault("YOLO_MONITOR_URL", "https://monitor.maocong.me")
    os.environ["YOLO_MONITOR_USE_PROXY"] = "false"

    required_paths = (str(ROOT), str(MONITOR_DIR))
    for path in reversed(required_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    existing_paths = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    existing_paths = [entry for entry in existing_paths if entry not in required_paths]
    # Ultralytics writes its DDP launcher under ~/.config/Ultralytics/DDP, so
    # the child cannot rely on the training script's repository working path.
    os.environ["PYTHONPATH"] = os.pathsep.join((*required_paths, *existing_paths))


_bootstrap_monitor_imports()

from monitored_detection_trainer import MonitoredDetectionTrainer  # noqa: E402
from monitored_obb_trainer import MonitoredOBBTrainer  # noqa: E402
from monitored_single_modality_obb_trainer import MonitoredSingleModalityOBBTrainer  # noqa: E402
from monitored_p2det_trainer import (  # noqa: E402
    MonitoredDA016PromptAblationOBBTrainer,
    MonitoredP2ObjectReliabilityOBBTrainer,
    MonitoredP2PromptOBBTrainer,
    MonitoredP2ReliabilityOBBTrainer,
    MonitoredP2SecondGenOBBTrainer,
    MonitoredP2TrueObjectReliabilityOBBTrainer,
)
from yolo_monitor import RemoteMonitorTrainerMixin, YoloExperimentMonitor  # noqa: E402


_TARGET_SALIENCY_EXPORTS = {
    "MonitoredSoftCenternessTargetSaliencyOBBTrainer",
    "MonitoredTargetSaliencyOBBTrainer",
}


def __getattr__(name: str):
    """Load optional target-saliency OBB trainers only when requested."""

    if name not in _TARGET_SALIENCY_EXPORTS:
        raise AttributeError(name)
    from monitored_target_saliency_trainer import (
        MonitoredSoftCenternessTargetSaliencyOBBTrainer,
        MonitoredTargetSaliencyOBBTrainer,
    )

    exports = {
        "MonitoredSoftCenternessTargetSaliencyOBBTrainer": MonitoredSoftCenternessTargetSaliencyOBBTrainer,
        "MonitoredTargetSaliencyOBBTrainer": MonitoredTargetSaliencyOBBTrainer,
    }
    globals().update(exports)
    return exports[name]


def create_monitor(experiment_name: str) -> YoloExperimentMonitor:
    """Validate required configuration and create a monitor for one run."""
    if not os.getenv("YOLO_MONITOR_URL") or not os.getenv("YOLO_MONITOR_TOKEN"):
        raise RuntimeError("Set YOLO_MONITOR_URL and YOLO_MONITOR_TOKEN before training")
    return YoloExperimentMonitor(experiment_name=experiment_name)


__all__ = (
    "MonitoredDetectionTrainer",
    "MonitoredOBBTrainer",
    "MonitoredSingleModalityOBBTrainer",
    "MonitoredP2PromptOBBTrainer",
    "MonitoredP2ObjectReliabilityOBBTrainer",
    "MonitoredP2TrueObjectReliabilityOBBTrainer",
    "MonitoredP2ReliabilityOBBTrainer",
    "MonitoredP2SecondGenOBBTrainer",
    "MonitoredDA016PromptAblationOBBTrainer",
    "MonitoredSoftCenternessTargetSaliencyOBBTrainer",
    "MonitoredTargetSaliencyOBBTrainer",
    "RemoteMonitorTrainerMixin",
    "YoloExperimentMonitor",
    "create_monitor",
)
