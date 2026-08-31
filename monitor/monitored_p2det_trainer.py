"""Importable DDP-safe monitored trainer for P2Det prompt/GDER experiments."""

from ultralytics.models.yolo.obb.p2det_reliability_train import P2ReliabilityOBBTrainer
from ultralytics.models.yolo.obb.p2det_object_reliability_train import (
    DA016PromptAblationOBBTrainer,
    P2ObjectReliabilityOBBTrainer,
    P2TrueObjectReliabilityOBBTrainer,
)
from ultralytics.models.yolo.obb.p2det_train import P2PromptOBBTrainer, P2SecondGenOBBTrainer

from yolo_monitor import RemoteMonitorTrainerMixin


class MonitoredP2PromptOBBTrainer(RemoteMonitorTrainerMixin, P2PromptOBBTrainer):
    """P2PromptOBBTrainer plus rank-0 remote monitoring callbacks."""

    pass


class MonitoredP2SecondGenOBBTrainer(RemoteMonitorTrainerMixin, P2SecondGenOBBTrainer):
    """P2SecondGenOBBTrainer plus rank-0 remote monitoring callbacks."""

    pass


class MonitoredP2ReliabilityOBBTrainer(RemoteMonitorTrainerMixin, P2ReliabilityOBBTrainer):
    """V16 reliability trainer plus rank-0 remote monitoring callbacks."""

    pass


class MonitoredP2ObjectReliabilityOBBTrainer(
    RemoteMonitorTrainerMixin, P2ObjectReliabilityOBBTrainer
):
    """V17 object-reliability trainer plus rank-0 remote monitoring callbacks."""

    pass


class MonitoredP2TrueObjectReliabilityOBBTrainer(
    RemoteMonitorTrainerMixin, P2TrueObjectReliabilityOBBTrainer
):
    """V18 true object-reliability trainer plus remote monitoring."""

    pass


class MonitoredDA016PromptAblationOBBTrainer(
    RemoteMonitorTrainerMixin, DA016PromptAblationOBBTrainer
):
    """Strict DA016-parent Prompt ablation trainer plus remote monitoring."""

    pass
