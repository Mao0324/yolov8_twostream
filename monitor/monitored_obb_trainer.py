"""Importable DDP-safe OBB Trainer with remote experiment monitoring."""

from ultralytics.models.yolo.obb.train import OBBTrainer

from yolo_monitor import RemoteMonitorTrainerMixin


class MonitoredOBBTrainer(RemoteMonitorTrainerMixin, OBBTrainer):
    """Standard OBBTrainer plus rank-0 remote monitoring callbacks."""

    pass
