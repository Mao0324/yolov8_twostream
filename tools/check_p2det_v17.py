#!/usr/bin/env python3
"""Fast CPU acceptance checks for P2Det V17 without starting training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO
from ultralytics.utils.dist import generate_ddp_file

from tools.dronevehicle_m2dlif import install_trusted_torch_load
from ultralytics.models.yolo.obb.p2det_object_reliability_train import (
    calibrated_balanced_mean,
    gate_direction_loss,
    materialize_object_reliability_targets,
    object_gaussian_reduce,
    object_rank_loss,
    pool_obb_reliability_maps,
)


CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_p2det_objectreliability_p34_monotonicpriorlaf_v17.pt"


class _DDPSerializationProbe:
    pass


def main():
    install_trusted_torch_load(torch)
    wrapped = YOLO(str(CHECKPOINT), task="obb")
    model = wrapped.model.float().eval()
    expected = {
        10: "P2DualObjectReliabilityMonotonicLAFMergeFeedback2D",
        15: "P2DualObjectReliabilityMonotonicLAFMergeFeedback2D",
        22: "LAFMergeFeedback2D",
    }
    for index, expected_type in expected.items():
        if type(model.model[index]).__name__ != expected_type:
            raise RuntimeError(f"checkpoint layer {index} is not {expected_type}")
    if model.yaml.get("p2_teacher_match_topk") != 128:
        raise RuntimeError("V17 changed the required teacher topk=128")
    if model.p2det_migration["neutral_forward_max_abs_diff"] != 0.0:
        raise RuntimeError("V17 checkpoint does not preserve the DarkACT neutral forward")
    for index in (10, 15):
        module = model.model[index]
        if module.rgb_prompt_head.body[-1].bias is not None or module.ir_prompt_head.body[-1].bias is not None:
            raise RuntimeError(f"layer {index} Prompt output is not bias-free")
        if float(module.merge.prompt_strength) <= 0:
            raise RuntimeError(f"layer {index} Prompt strength is not positive")
        for layer in (module.merge.global_gate[-1], module.merge.local_gate[-1]):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        features = [torch.ones(1, module.channels, 4, 4), torch.ones(1, module.channels, 4, 4)]
        realized = []
        for prior in ((0.8, 0.2), (0.2, 0.8)):
            probability = torch.tensor(prior).view(1, 2, 1, 1).expand(1, 2, 4, 4)
            module.merge.capture_gate = True
            module.merge(features, probability)
            realized.append(module.merge.pop_modal_weights().mean((0, 2, 3, 4)) / 2.0)
        if not (realized[0][0] > realized[0][1] and realized[1][0] < realized[1][1]):
            raise RuntimeError(f"layer {index} Prompt-to-gate mapping is not monotonic")

    # Equal-object reduction must be differentiable.
    field = torch.randn(2, 2, 12, 12, requires_grad=True)
    reduction_batch = {
        "bboxes": torch.tensor(
            [[0.25, 0.30, 0.20, 0.10, 0.2], [0.70, 0.65, 0.15, 0.30, -0.4], [0.50, 0.50, 0.2, 0.2, 0.0]]
        ),
        "batch_idx": torch.tensor([0, 0, 1]),
    }
    reduced = object_gaussian_reduce(field, reduction_batch)
    reduced.square().mean().backward()
    if field.grad is None or not torch.isfinite(field.grad).all() or float(field.grad.abs().sum()) == 0:
        raise RuntimeError("object Gaussian reduction has no finite gradient")

    # Fusing fields into one reduction must preserve the former three-call
    # values exactly up to normal floating-point tolerance.
    fields = [torch.randn(2, 2, 12, 12) for _ in range(3)]
    separate = torch.cat([object_gaussian_reduce(value, reduction_batch) for value in fields], dim=1)
    combined = object_gaussian_reduce(torch.cat(fields, dim=1), reduction_batch)
    if not torch.allclose(separate, combined, atol=1e-7, rtol=1e-6):
        raise RuntimeError("combined object reduction changed pooled values")

    # A mixed RGB/IR-winner batch must push different objects in opposite directions.
    logits = torch.zeros(4, 2, requires_grad=True)
    target = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.3, 0.7]])
    confidence = torch.ones(4)
    object_ce = -(target * logits.log_softmax(1)).sum(1)
    prompt_loss = calibrated_balanced_mean(object_ce, target, confidence)[0]
    rank_loss = object_rank_loss(logits, target, max_objects=128)
    direction_loss = gate_direction_loss(logits.softmax(1), target, confidence)
    (prompt_loss + 0.1 * rank_loss + 0.1 * direction_loss).backward()
    score_gradient = logits.grad[:, 0] - logits.grad[:, 1]
    if not (score_gradient[0] < 0 and (score_gradient[1:] > 0).all()):
        raise RuntimeError("V17 object losses do not separate RGB/IR teacher winners")

    # Real teachers produce inference tensors. The loss boundary must convert
    # them to normal constants before autograd tries to save them for backward.
    with torch.inference_mode():
        inference_target = torch.softmax(torch.randn(4, 2), dim=1)
    safe_target = materialize_object_reliability_targets(
        {"p2_reliability_object_targets": inference_target}
    )
    if torch.is_inference(safe_target):
        raise RuntimeError("teacher reliability target remained an inference tensor")
    autograd_logits = torch.zeros(4, 2, requires_grad=True)
    (-(safe_target * autograd_logits.log_softmax(1)).sum(1).mean()).backward()
    if autograd_logits.grad is None or not torch.isfinite(autograd_logits.grad).all():
        raise RuntimeError("materialized teacher target cannot supervise Prompt autograd")

    # The saved custom model must expose maps that can be pooled on inference OBBs.
    sample = torch.rand(2, 6, 64, 64)
    _, auxiliary = model.predict_with_reliability_maps(sample)
    boxes = [torch.tensor([[20.0, 20.0, 12.0, 8.0, 0.2]]), torch.tensor([[35.0, 30.0, 10.0, 15.0, -0.3]])]
    prompt, gate = pool_obb_reliability_maps(auxiliary, boxes, sample.shape[-2:])
    for values in (*prompt, *gate):
        if values.shape != (1, 2) or not torch.allclose(values.sum(1), torch.ones(1), atol=1e-5):
            raise RuntimeError("per-detection inference reliability is not normalized")

    expected_path = CHECKPOINT.resolve()
    if Path(wrapped.overrides["model"]).resolve() != expected_path or Path(wrapped.ckpt_path).resolve() != expected_path:
        raise RuntimeError("YOLO checkpoint metadata does not point to the V17 .pt")
    probe = _DDPSerializationProbe()
    probe.args = SimpleNamespace(model=str(expected_path), device="0,1", batch=64)
    ddp_file = Path(generate_ddp_file(probe))
    try:
        ddp_text = ddp_file.read_text(encoding="utf-8")
        if repr(str(expected_path)) not in ddp_text or "device': '0,1'" not in ddp_text:
            raise RuntimeError("DDP temporary overrides do not serialize the V17 .pt and two GPUs")
    finally:
        ddp_file.unlink(missing_ok=True)

    print("P2Det V17 acceptance passed")
    print("bias_free_object_prompt=True")
    print("monotonic_prompt_gate=True")
    print("equal_object_gradient=True")
    print("combined_reduction_equivalent=True")
    print("mixed_winner_gradients=True")
    print("teacher_inference_tensor_materialized=True")
    print("per_detection_inference_pooling=True")
    print("topk=128")
    print("ddp_checkpoint_and_temp_metadata=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
