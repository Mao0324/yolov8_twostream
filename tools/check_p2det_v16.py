#!/usr/bin/env python3
"""Fast CPU acceptance checks for P2Det V16 (no dataset or training run)."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from ultralytics import YOLO

from tools.dronevehicle_m2dlif import install_trusted_torch_load
from ultralytics.models.yolo.obb.p2det_reliability_train import rasterize_relative_reliability
from ultralytics.nn.modules import LAFMergeFeedback2D, P2DualReliabilityPriorLAFMergeFeedback2D


CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_p2det_dualreliability_p34_priorlaf_v16.pt"


def main():
    install_trusted_torch_load(torch)
    torch.manual_seed(16)
    rgb = torch.randn(2, 64, 16, 16, requires_grad=True)
    ir = torch.randn(2, 64, 16, 16, requires_grad=True)
    darkact = LAFMergeFeedback2D(64, 2, 4, 1, 8).eval()
    v16 = P2DualReliabilityPriorLAFMergeFeedback2D(64, 2, 4, 1, 8).eval()
    v16.merge.load_state_dict(darkact.merge.state_dict(), strict=False)
    v16.capture_prompt = True
    darkact_output = darkact((rgb, ir))[2]
    v16_output = v16((rgb, ir))[2]
    auxiliary = v16.pop_aux_outputs()
    if not torch.equal(darkact_output, v16_output):
        raise RuntimeError("neutral V16 module does not exactly reproduce DarkACT")
    if not torch.allclose(auxiliary["prompt_probabilities"].sum(1), torch.ones(2, 16, 16)):
        raise RuntimeError("prompt probabilities are not normalized over modality")
    if not torch.allclose(auxiliary["modal_weights"].sum(1), torch.full((2, 64, 16, 16), 2.0)):
        raise RuntimeError("LAF modality weights do not preserve the factor-two convention")

    reference = torch.zeros(1, 2, 16, 16)
    common = {"bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.15, 0.2]]), "batch_idx": torch.tensor([0])}
    target_a, support_a = rasterize_relative_reliability(
        reference, {**common, "p2_reliability_object_targets": torch.tensor([[0.8, 0.2]])}
    )
    target_b, support_b = rasterize_relative_reliability(
        reference, {**common, "p2_reliability_object_targets": torch.tensor([[0.2, 0.8]])}
    )
    if not torch.equal(support_a, support_b):
        raise RuntimeError("GT support changed when only teacher reliability was swapped")
    if not torch.allclose(target_a[:, 0], target_b[:, 1]) or not torch.allclose(target_a[:, 1], target_b[:, 0]):
        raise RuntimeError("relative reliability channels did not swap under the counterfactual")

    # Both direct distillation and detection-through-gate paths must be live.
    v16.train()
    v16.capture_prompt = True
    fused = v16((rgb, ir))[2]
    auxiliary = v16.pop_aux_outputs()
    labels = torch.tensor([0, 1])[:, None, None].expand(2, 16, 16)
    loss = fused.square().mean() + F.cross_entropy(auxiliary["prompt_logits"], labels)
    loss.backward()
    required_gradients = (
        v16.rgb_prompt_head.body[-1].weight.grad,
        v16.ir_prompt_head.body[-1].weight.grad,
        v16.merge.prompt_local_gate[-1].weight.grad,
        v16.merge.prompt_global_gate[-1].weight.grad,
    )
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in required_gradients):
        raise RuntimeError("a V16 prompt/gate path has no finite gradient")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"build V16 checkpoint first: {CHECKPOINT}")
    wrapped = YOLO(str(CHECKPOINT), task="obb")
    model = wrapped.model
    expected = {10: "P2DualReliabilityPriorLAFMergeFeedback2D", 15: "P2DualReliabilityPriorLAFMergeFeedback2D", 22: "LAFMergeFeedback2D"}
    for index, expected_type in expected.items():
        if type(model.model[index]).__name__ != expected_type:
            raise RuntimeError(f"checkpoint layer {index} is not {expected_type}")
    if model.p2det_migration["neutral_forward_max_abs_diff"] != 0.0:
        raise RuntimeError("checkpoint migration did not preserve the DarkACT forward")
    if Path(wrapped.overrides["model"]).resolve() != CHECKPOINT.resolve() or Path(wrapped.ckpt_path).resolve() != CHECKPOINT.resolve():
        raise RuntimeError("DDP checkpoint metadata does not point to the V16 .pt")

    print("P2Det V16 acceptance passed")
    print("neutral_darkact_max_abs_diff=0")
    print("teacher_swap_support_invariant=True")
    print("prompt_and_prior_gate_gradients=True")
    print("ddp_checkpoint_metadata=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
