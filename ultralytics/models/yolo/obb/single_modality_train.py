# Ultralytics YOLO 🚀, AGPL-3.0 license
"""True 3-channel RGB-only/IR-only OBB teacher training for this two-stream fork."""

from copy import copy
import os

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import RANK


def resolve_teacher_modality():
    """Read and validate the modality inherited by automatic DDP children."""
    modality = os.getenv("P2DET_SINGLE_MODALITY_TEACHER", "").strip().lower()
    if modality not in {"rgb", "ir"}:
        raise RuntimeError("P2DET_SINGLE_MODALITY_TEACHER must be 'rgb' or 'ir'")
    return modality


def select_modality(images, modality):
    """Select one three-channel modality from a paired six-channel batch."""
    if images.ndim != 4:
        raise ValueError(f"expected BCHW images, got {tuple(images.shape)}")
    if images.shape[1] == 3:
        return images
    if images.shape[1] != 6:
        raise ValueError(f"expected paired six-channel images, got {tuple(images.shape)}")
    return images[:, :3] if modality == "rgb" else images[:, 3:6]


class SingleModalityOBBModel(OBBModel):
    """Ordinary YOLOv8 OBB graph, bypassing the repository's two-stream router."""

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        # DetectionModel currently probes stride with a six-channel dummy. Use
        # one half only for that probe; all train/val batches arrive as 3ch.
        if x.ndim != 4 or x.shape[1] not in {3, 6}:
            raise ValueError(f"SingleModalityOBBModel expects 3 channels, got {tuple(x.shape)}")
        if x.shape[1] == 6:
            x = x[:, :3]
        outputs, embeddings, profile_times = [], [], []
        for module in self.model:
            if module.f != -1:
                x = (
                    outputs[module.f]
                    if isinstance(module.f, int)
                    else [x if source == -1 else outputs[source] for source in module.f]
                )
            if profile:
                self._profile_one_layer(module, x, profile_times)
            x = module(x)
            outputs.append(x if module.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization

                feature_visualization(x, module.type, module.i, save_dir=visualize)
            if embed and module.i in embed:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if module.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x


class SingleModalityOBBValidator(OBBValidator):
    """OBB validator that exposes only the selected teacher modality."""

    def __init__(self, *args, modality=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = modality or resolve_teacher_modality()

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        batch["img"] = select_modality(batch["img"], self.modality)
        return batch


class SingleModalityOBBTrainer(OBBTrainer):
    """Train a true RGB-only or IR-only OBB model from paired data."""

    def __init__(self, *args, **kwargs):
        self.modality = resolve_teacher_modality()
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SingleModalityOBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return SingleModalityOBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            modality=self.modality,
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        batch["img"] = select_modality(batch["img"], self.modality)
        return batch


__all__ = (
    "SingleModalityOBBModel",
    "SingleModalityOBBTrainer",
    "SingleModalityOBBValidator",
    "resolve_teacher_modality",
    "select_modality",
)
