"""Importable DDP-safe HBB Trainer with remote experiment monitoring."""

from ultralytics.models.yolo.detect.train import DetectionTrainer

from yolo_monitor import RemoteMonitorTrainerMixin


class MonitoredDetectionTrainer(RemoteMonitorTrainerMixin, DetectionTrainer):
    """DetectionTrainer plus rank-0 remote monitoring callbacks."""

    pass


__all__ = ("MonitoredDetectionTrainer",)
