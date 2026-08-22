"""Importable DDP-safe monitored trainer for P2Det prompt/GDER experiments."""

from ultralytics.models.yolo.obb.p2det_train import P2PromptOBBTrainer, P2SecondGenOBBTrainer

from yolo_monitor import RemoteMonitorTrainerMixin


class MonitoredP2PromptOBBTrainer(RemoteMonitorTrainerMixin, P2PromptOBBTrainer):
    """P2PromptOBBTrainer plus rank-0 remote monitoring callbacks."""

    pass


class MonitoredP2SecondGenOBBTrainer(RemoteMonitorTrainerMixin, P2SecondGenOBBTrainer):
    """P2SecondGenOBBTrainer plus rank-0 remote monitoring callbacks."""

    pass
