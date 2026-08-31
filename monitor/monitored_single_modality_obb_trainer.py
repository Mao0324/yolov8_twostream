"""Importable DDP-safe monitored trainer for single-modality OBB teachers."""

from ultralytics.models.yolo.obb.single_modality_train import SingleModalityOBBTrainer

from yolo_monitor import RemoteMonitorTrainerMixin


class MonitoredSingleModalityOBBTrainer(RemoteMonitorTrainerMixin, SingleModalityOBBTrainer):
    """SingleModalityOBBTrainer plus rank-0 remote monitoring callbacks."""

    pass
