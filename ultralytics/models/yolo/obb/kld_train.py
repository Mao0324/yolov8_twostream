"""OBB training with a paper-faithful FP32 KLD/ProbIoU mixed regression loss."""

from copy import copy

import torch
from torch import nn

from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import RotatedBboxLoss, v8OBBLoss
from ultralytics.utils.metrics import probiou


class KLDivergenceRotationLoss(nn.Module):
    """Bounded D_KL(N_pred || N_target) rotation loss from AMCA-YOLOv11.

    Boxes are ``(x, y, w, h, theta)`` and use the paper covariance scale
    ``diag(w**2 / 4, h**2 / 4)``. All numerically sensitive operations run in
    FP32 even when the surrounding training step uses AMP.
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = float(eps)

    @staticmethod
    def _covariance(boxes, eps):
        w2 = boxes[..., 2:3].clamp_min(eps).square() * 0.25
        h2 = boxes[..., 3:4].clamp_min(eps).square() * 0.25
        angle = boxes[..., 4:5]
        cos, sin = angle.cos(), angle.sin()
        cos2, sin2 = cos.square(), sin.square()
        a = w2 * cos2 + h2 * sin2
        b = w2 * sin2 + h2 * cos2
        c = (w2 - h2) * cos * sin
        return a, b, c

    def forward(self, pred_bboxes, target_bboxes):
        device_type = pred_bboxes.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            pred = pred_bboxes.float()
            target = target_bboxes.float()
            ap, bp, cp = self._covariance(pred, self.eps)
            at, bt, ct = self._covariance(target, self.eps)

            det_p = (ap * bp - cp.square()).clamp_min(self.eps)
            det_t = (at * bt - ct.square()).clamp_min(self.eps)
            dx = target[..., 0:1] - pred[..., 0:1]
            dy = target[..., 1:2] - pred[..., 1:2]

            # tr(Sigma_t^-1 Sigma_p) and (mu_t-mu_p)^T Sigma_t^-1 (mu_t-mu_p).
            trace = (bt * ap + at * bp - 2.0 * ct * cp) / det_t
            mahalanobis = (bt * dx.square() + at * dy.square() - 2.0 * ct * dx * dy) / det_t
            log_det_ratio = torch.log(det_t) - torch.log(det_p)
            divergence = (0.5 * (trace + mahalanobis + log_det_ratio - 2.0)).clamp_min(0.0)
            return 1.0 - torch.rsqrt(1.0 + divergence)


class KLDProbIoURotatedBboxLoss(RotatedBboxLoss):
    """Mix positive-sample ProbIoU and KLD losses without changing assignment."""

    def __init__(self, reg_max, use_dfl=False, kld_mix=0.5):
        super().__init__(reg_max, use_dfl)
        if not 0.0 <= kld_mix <= 1.0:
            raise ValueError(f"kld_mix must be in [0, 1], got {kld_mix}")
        self.kld_mix = float(kld_mix)
        self.kld = KLDivergenceRotationLoss()

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_fg, target_fg = pred_bboxes[fg_mask], target_bboxes[fg_mask]
        probiou_loss = 1.0 - probiou(pred_fg, target_fg)
        kld_loss = self.kld(pred_fg, target_fg)
        mixed_loss = (1.0 - self.kld_mix) * probiou_loss.float() + self.kld_mix * kld_loss
        loss_box = (mixed_loss * weight.float()).sum() / target_scores_sum.float()

        if self.use_dfl:
            from ultralytics.utils.tal import bbox2dist
            from ultralytics.utils.ops import xywh2xyxy

            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.zeros((), device=pred_dist.device)
        return loss_box, loss_dfl


class KLDProbIoUOBBLoss(v8OBBLoss):
    """Standard YOLOv8 OBB objective with only its positive box loss replaced."""

    def __init__(self, model):
        super().__init__(model)
        mix = float(model.yaml.get("kld_mix", 0.5))
        self.bbox_loss = KLDProbIoURotatedBboxLoss(
            self.reg_max - 1, use_dfl=self.use_dfl, kld_mix=mix
        ).to(self.device)


class KLDProbIoUOBBModel(OBBModel):
    """OBB model selecting the isolated mixed KLD criterion."""

    def init_criterion(self):
        return KLDProbIoUOBBLoss(self)


class KLDProbIoUOBBTrainer(OBBTrainer):
    """Trainer that builds KLDProbIoUOBBModel while retaining the standard assigner."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = KLDProbIoUOBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.obb.OBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


__all__ = (
    "KLDivergenceRotationLoss",
    "KLDProbIoURotatedBboxLoss",
    "KLDProbIoUOBBLoss",
    "KLDProbIoUOBBModel",
    "KLDProbIoUOBBTrainer",
)
