#!/usr/bin/env python3
"""CPU acceptance checks for P2Det V18 without starting a training run."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.dist import generate_ddp_file

from tools.dronevehicle_m2dlif import install_trusted_torch_load
from ultralytics.models.yolo.obb.p2det_object_reliability_train import (
    calibrated_balanced_mean,
    object_gaussian_reduce,
    object_rank_loss,
    pool_obb_reliability_maps,
)


CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_p2det_trueobjectreliability_p34_signpreservinglaf_v18.pt"


class _DDPSerializationProbe:
    pass


def _check_sign_preservation(module, index):
    if hasattr(module.merge, "raw_prompt_strength") or float(module.merge.prompt_strength) != 1.0:
        raise RuntimeError(f"layer {index} does not use fixed unit Prompt strength")
    if not 0 <= module.merge.rho < 1:
        raise RuntimeError(f"layer {index} rho does not guarantee sign preservation")

    # Deliberately make the inherited feature gate extreme. The gate must still
    # follow every non-neutral Prompt, down to a very small margin.
    with torch.no_grad():
        for final in (module.merge.global_gate[-1], module.merge.local_gate[-1]):
            final.weight.normal_(mean=0.0, std=100.0)
            final.bias.uniform_(-100.0, 100.0)
    rgb = torch.randn(2, module.channels, 4, 4)
    ir = torch.randn_like(rgb)
    for rgb_probability in (0.5001, 0.55, 0.8, 0.4999, 0.45, 0.2):
        prompt = torch.tensor((rgb_probability, 1.0 - rgb_probability)).view(1, 2, 1, 1)
        prompt = prompt.expand(rgb.shape[0], 2, rgb.shape[2], rgb.shape[3])
        module.merge.capture_gate = True
        module.merge((rgb, ir), prompt)
        weights = module.merge.pop_modal_weights()
        gate_difference = weights[:, 0] - weights[:, 1]
        expected_sign = 1 if rgb_probability > 0.5 else -1
        if not (gate_difference * expected_sign > 0).all():
            raise RuntimeError(f"layer {index} feature gate reversed Prompt={rgb_probability}")

    neutral = torch.full((rgb.shape[0], 2, 4, 4), 0.5)
    module.merge.capture_gate = True
    module.merge((rgb, ir), neutral)
    weights = module.merge.pop_modal_weights()
    if not torch.equal(weights[:, 0], weights[:, 1]):
        raise RuntimeError(f"layer {index} neutral Prompt is not an exactly neutral gate")


def main():
    install_trusted_torch_load(torch)
    wrapped = YOLO(str(CHECKPOINT), task="obb")
    model = wrapped.model.float().eval()
    expected = {
        10: "P2DualTrueObjectReliabilityLAFMergeFeedback2D",
        15: "P2DualTrueObjectReliabilityLAFMergeFeedback2D",
        22: "LAFMergeFeedback2D",
        35: "OBB",
    }
    for index, expected_type in expected.items():
        actual = type(model.model[index]).__name__
        if actual != expected_type:
            raise RuntimeError(f"checkpoint layer {index}: expected {expected_type}, got {actual}")
    yaml = model.yaml
    required = {
        "p2_teacher_match_topk": 128,
        "p2_object_hard_winner_gain": 1.0,
        "p2_object_soft_calibration_gain": 0.25,
        "p2_object_cross_scale_consistency_gain": 0.0,
    }
    for name, expected_value in required.items():
        if yaml.get(name) != expected_value:
            raise RuntimeError(f"V18 setting {name}={yaml.get(name)!r}; expected {expected_value!r}")
    migration = model.p2det_migration
    if migration.get("true_object_reliability_version") != 18:
        raise RuntimeError("checkpoint has no V18 migration metadata")
    if migration.get("darkact_v18_max_abs_diff") != 0.0:
        raise RuntimeError("DarkACT tensors were not copied exactly")

    for index in (10, 15):
        module = model.model[index]
        if module.rgb_prompt_head.body[-1].bias is not None or module.ir_prompt_head.body[-1].bias is not None:
            raise RuntimeError(f"layer {index} Prompt output must remain bias-free")
        _check_sign_preservation(module, index)

    # Hard winner balancing gives RGB-win and IR-win objects opposite local
    # gradients even when the dataset is IR-majority. Soft calibration remains
    # deliberately weaker and does not define which modality must win.
    logits = torch.zeros(4, 2, requires_grad=True)
    target = torch.tensor([[0.80, 0.20], [0.20, 0.80], [0.10, 0.90], [0.30, 0.70]])
    winner = target.argmax(1)
    hard_values = F.cross_entropy(logits, winner, reduction="none")
    hard_target = F.one_hot(winner, num_classes=2).float()
    hard_loss = calibrated_balanced_mean(
        hard_values,
        hard_target,
        torch.ones(4),
        calibration_ratio=0.0,
        balanced_ratio=1.0,
    )[0]
    rank_loss = object_rank_loss(logits, target, max_objects=128, minimum_gap=0.1)
    (hard_loss + 0.2 * rank_loss).backward()
    score_gradient = logits.grad[:, 0] - logits.grad[:, 1]
    if not (score_gradient[0] < 0 and (score_gradient[1:] > 0).all()):
        raise RuntimeError("hard winner objective does not separate RGB/IR objects")

    # The optimized combined P3/P4 object reduction remains differentiable.
    field = torch.randn(2, 6, 12, 12, requires_grad=True)
    reduction_batch = {
        "bboxes": torch.tensor(
            [[0.25, 0.30, 0.20, 0.10, 0.2], [0.70, 0.65, 0.15, 0.30, -0.4], [0.50, 0.50, 0.2, 0.2, 0.0]]
        ),
        "batch_idx": torch.tensor([0, 0, 1]),
    }
    object_gaussian_reduce(field, reduction_batch).square().mean().backward()
    if field.grad is None or not torch.isfinite(field.grad).all() or float(field.grad.abs().sum()) == 0:
        raise RuntimeError("object Gaussian reduction has no finite gradient")

    # The custom checkpoint exposes per-detection Prompt and realized-gate maps.
    sample = torch.rand(2, 6, 64, 64)
    _, auxiliary = model.predict_with_reliability_maps(sample)
    boxes = [torch.tensor([[20.0, 20.0, 12.0, 8.0, 0.2]]), torch.tensor([[35.0, 30.0, 10.0, 15.0, -0.3]])]
    prompt, gate = pool_obb_reliability_maps(auxiliary, boxes, sample.shape[-2:])
    for prompt_values, gate_values in zip(prompt, gate):
        if prompt_values.shape != (1, 2) or gate_values.shape != (1, 2):
            raise RuntimeError("per-detection reliability output has the wrong shape")
        if not torch.allclose(prompt_values.sum(1), torch.ones(1), atol=1e-5):
            raise RuntimeError("pooled Prompt is not normalized")
        if not torch.allclose(gate_values.sum(1), torch.ones(1), atol=1e-5):
            raise RuntimeError("pooled gate is not normalized")
        if (prompt_values[:, 0] - prompt_values[:, 1]) * (gate_values[:, 0] - gate_values[:, 1]) < 0:
            raise RuntimeError("pooled realized gate disagrees with the pooled Prompt")

    # Exercise the real detector criterion and V18 auxiliary criterion together.
    # A normal trainer replaces model.args before constructing the criterion;
    # supply the same loss gains explicitly in this standalone checkpoint test.
    model.train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.p2_object_reliability_epoch = 4
    full_batch = {
        "img": torch.rand(2, 6, 64, 64),
        "batch_idx": torch.tensor([0, 0, 1]),
        "cls": torch.tensor([[0.0], [2.0], [4.0]]),
        "bboxes": reduction_batch["bboxes"],
        "p2_reliability_object_targets": target[:3],
        "p2_teacher_object_quality": target[:3],
    }
    total_loss, loss_items = model.loss(full_batch)
    total_loss.backward()
    prompt_gradients = []
    for index in (10, 15):
        module = model.model[index]
        for head in (module.rgb_prompt_head, module.ir_prompt_head):
            gradient = head.body[-1].weight.grad
            prompt_gradients.append(float(gradient.abs().sum()) if gradient is not None else 0.0)
    if (
        not torch.isfinite(total_loss)
        or not torch.isfinite(loss_items).all()
        or min(prompt_gradients) <= 0
    ):
        raise RuntimeError("full V18 detection plus reliability loss/backward failed")

    expected_path = CHECKPOINT.resolve()
    if Path(wrapped.overrides["model"]).resolve() != expected_path or Path(wrapped.ckpt_path).resolve() != expected_path:
        raise RuntimeError("YOLO metadata does not point to the embedded V18 .pt")
    probe = _DDPSerializationProbe()
    probe.args = SimpleNamespace(model=str(expected_path), device="0,1", batch=64)
    ddp_file = Path(generate_ddp_file(probe))
    try:
        ddp_text = ddp_file.read_text(encoding="utf-8")
        if repr(str(expected_path)) not in ddp_text or "device': '0,1'" not in ddp_text:
            raise RuntimeError("DDP temporary overrides do not serialize the V18 .pt and two GPUs")
    finally:
        ddp_file.unlink(missing_ok=True)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print("P2Det V18 acceptance passed")
    print(f"parameters={parameters:,}")
    print("hard_balanced_winner_supervision=True")
    print("feature_gate_cannot_reverse_prompt=True")
    print("fixed_prompt_strength=True")
    print("cross_scale_consistency=False")
    print("topk=128")
    print("per_detection_inference_pooling=True")
    print("full_detection_and_reliability_backward=True")
    print("ddp_checkpoint_and_temp_metadata=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
