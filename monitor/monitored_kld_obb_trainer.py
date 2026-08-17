"""Importable DDP-safe KLD/ProbIoU OBB trainer with remote monitoring."""

from ultralytics.models.yolo.obb.kld_train import KLDProbIoUOBBTrainer

from yolo_monitor import RemoteMonitorTrainerMixin


class MonitoredKLDProbIoUOBBTrainer(RemoteMonitorTrainerMixin, KLDProbIoUOBBTrainer):
    """Mixed KLD trainer plus rank-0 remote monitoring callbacks."""

    pass


__all__ = ("MonitoredKLDProbIoUOBBTrainer",)
