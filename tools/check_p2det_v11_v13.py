#!/usr/bin/env python3
"""CPU-only acceptance checks for P2Det V11-V13; never launches formal training."""

from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.obb.p2det_train import P2SecondGenOBBModel
from ultralytics.nn.modules import (
    P2_SECOND_GEN_PROMPT_MODULES,
    P2IRPromptAsymIdentityGDERMergeFeedback2D,
    P2IRPromptLAFMergeFeedback2D,
    P2IRPromptLAFMergeFeedbackNoStaticMAA2D,
)
from ultralytics.nn.modules.p2det_prompt_gder import (
    _P2AsymmetricIdentityModalityExpert,
    _P2ExpertGate,
    _P2IdentityPreservingModalityExpert,
)
from ultralytics.utils.torch_utils import get_flops

from tools.p2det_weight_transfer import apply_p2det_baseline_transfer, load_p2det_transfer_reference


VARIANTS = {
    "v11": ROOT / "yaml/yolov8s-P2Det-IRPrompt-P34-NoStaticMAA-NoGDER-PostC2f-v11.yaml",
    "v12": ROOT / "yaml/yolov8s-P2Det-IRPrompt-AsymIdentityGDER-P45-NoStaticMAA-PostC2f-v12.yaml",
    "v13": ROOT / "yaml/yolov8s-P2Det-IRPrompt-AsymIdentityGDER-P4-NoStaticMAA-PostC2f-v13.yaml",
}
PROFILE_VARIANTS = {
    "v1": ROOT / "yaml/yolov8s-P2Det-IRPrompt-P34-PostC2f-v1.yaml",
    "v6": ROOT / "yaml/yolov8s-P2Det-DualPrompt-GDER-P45-NoStaticMAA-v6.yaml",
    "v8": ROOT / "yaml/yolov8s-P2Det-DualPrompt-IdentityGDER-P45-NoStaticMAA-v8.yaml",
    **VARIANTS,
}


def flatten_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from flatten_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_tensors(item)


def require_finite(name, value):
    tensors = list(flatten_tensors(value))
    if not tensors or not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise AssertionError(f"{name} is missing tensors or contains NaN/Inf")


def grad_sum(parameter, name):
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise AssertionError(f"missing or non-finite gradient: {name}")
    return float(gradient.detach().abs().sum())


def check_structure(version, path, imgsz):
    wrapper = YOLO(str(path), task="obb")
    model = wrapper.model.eval()
    modules = [module for module in model.modules() if isinstance(module, P2_SECOND_GEN_PROMPT_MODULES)]
    for module in modules:
        module.capture_prompt = True

    fused_shapes = []
    gate_shapes = []
    expert_shapes = []
    attention_shapes = []
    handles = []
    for module in modules:
        handles.append(
            module.register_forward_hook(
                lambda _m, _a, output: fused_shapes.append(tuple(output[2].shape))
            )
        )
        if module.use_gder:
            if type(module.expert_gate) is not _P2ExpertGate:
                raise AssertionError(f"{version} does not directly reuse the V8 channel gate")
            handles.append(
                module.expert_gate.register_forward_hook(
                    lambda _m, _a, output: gate_shapes.append(
                        (tuple(output.shape), float((output.sum(dim=1) - 1.0).abs().max()))
                    )
                )
            )
            handles.append(
                module.modality_expert.register_forward_hook(
                    lambda _m, _a, output: expert_shapes.append(tuple(output.shape))
                )
            )
            handles.append(
                module.attention_expert.register_forward_hook(
                    lambda _m, _a, output: attention_shapes.append(tuple(output.shape))
                )
            )

    x = torch.randn(1, 6, imgsz, imgsz)
    try:
        with torch.no_grad():
            output = model(x)
    finally:
        for handle in handles:
            handle.remove()
    require_finite(f"{version} output", output)

    aux = [module.pop_aux_outputs() for module in modules]
    diagnostics = [module.pop_gder_diagnostics() for module in modules]
    for module in modules:
        module.pop_gder_weights()
        module.capture_prompt = False
        if hasattr(module, "rgb_prompt_head") or hasattr(module, "rgb_prompt_embed"):
            raise AssertionError(f"{version} unexpectedly contains RGB spatial prompt parameters")
        if hasattr(module, "rgb_global_head") or hasattr(module, "rgb_global_embed"):
            raise AssertionError(f"{version} unexpectedly contains RGB global prompt parameters")

    for stage, values in enumerate(aux, start=3):
        if values is None:
            raise AssertionError(f"{version} P{stage} did not capture auxiliary outputs")
        if values["rgb_spatial_logits"] is not None or values["rgb_global_logit"] is not None:
            raise AssertionError(f"{version} P{stage} produced a forbidden RGB prompt")
        ir_logits = values["ir_spatial_logits"]
        require_finite(f"{version} P{stage} IR prompt", (ir_logits, ir_logits.sigmoid()))
        expected = (1, 1, imgsz // (2 ** stage), imgsz // (2 ** stage))
        if tuple(ir_logits.shape) != expected:
            raise AssertionError(f"{version} P{stage} IR shape {tuple(ir_logits.shape)} != {expected}")

    if any(error > 1e-6 for _, error in gate_shapes):
        raise AssertionError(f"{version} gate weights do not sum to one: {gate_shapes}")
    if version == "v11" and any(module.use_gder for module in modules):
        raise AssertionError("V11 contains GDER")
    if version == "v12":
        if [module.use_gder for module in modules] != [False, True, True]:
            raise AssertionError("V12 must place GDER at P4/P5")
        if not isinstance(modules[2], P2IRPromptAsymIdentityGDERMergeFeedback2D):
            raise AssertionError("V12 P5 must retain its IR Prompt inside asymmetric GDER")
    if version == "v13":
        if [module.use_gder for module in modules] != [False, True, False]:
            raise AssertionError("V13 must place GDER only at P4")
        p5 = modules[2]
        if not isinstance(p5, P2IRPromptLAFMergeFeedbackNoStaticMAA2D):
            raise AssertionError("V13 P5 must retain IR Prompt/LAF")
        if any(hasattr(p5, name) for name in ("modality_expert", "attention_expert", "expert_gate")):
            raise AssertionError("V13 P5 contains residual GDER parameters")

    expected_gate_shapes = {
        "v11": [],
        "v12": [(1, 2, 256, 1, 1), (1, 2, 512, 1, 1)],
        "v13": [(1, 2, 256, 1, 1)],
    }[version]
    if [shape for shape, _ in gate_shapes] != expected_gate_shapes:
        raise AssertionError(f"{version} gate shapes mismatch: {gate_shapes}")
    for diagnostic in diagnostics:
        if diagnostic:
            for key in ("rgb_branch_shape", "ir_branch_shape", "concat_shape", "modality_feature_shape"):
                if key not in diagnostic:
                    raise AssertionError(f"{version} missing asymmetric expert diagnostic {key}")

    summary = {
        "prompts": [tuple(values["ir_spatial_logits"].shape) for values in aux],
        "fused": fused_shapes,
        "experts": expert_shapes,
        "attention": attention_shapes,
        "gates": gate_shapes,
        "diagnostics": diagnostics,
        "layer25": type(model.model[25]).__name__,
    }
    print(f"{version} structure/forward: OK {summary}")
    del wrapper, model, modules, output, aux, diagnostics
    gc.collect()
    return summary


def check_parent_reuse():
    """Prove V11 reuses V1 Prompt/LAF exactly and V12 retains V8 parameter shapes."""
    torch.manual_seed(17)
    v1 = P2IRPromptLAFMergeFeedback2D(64, 2, 4, 1, 8).eval()
    v11 = P2IRPromptLAFMergeFeedbackNoStaticMAA2D(64, 2, 4, 1, 8).eval()
    v11.load_state_dict(v1.state_dict(), strict=True)
    inputs = (torch.randn(1, 64, 12, 12), torch.randn(1, 64, 12, 12))
    with torch.no_grad():
        old_outputs = v1(inputs)
        new_outputs = v11(inputs)
    maximum = max(
        float((old_value - new_value).abs().max())
        for old_value, new_value in zip(old_outputs, new_outputs)
    )
    if maximum != 0.0:
        raise AssertionError(f"V11 Prompt/LAF wrapper differs from V1: {maximum}")

    v8_expert = _P2IdentityPreservingModalityExpert(64, 8)
    v12_expert = _P2AsymmetricIdentityModalityExpert(64, 8)
    v8_gate = _P2ExpertGate(64, 8)
    v12_gate = _P2ExpertGate(64, 8)
    expert_shapes = {key: tuple(value.shape) for key, value in v8_expert.state_dict().items()}
    asym_expert_shapes = {key: tuple(value.shape) for key, value in v12_expert.state_dict().items()}
    gate_shapes = {key: tuple(value.shape) for key, value in v8_gate.state_dict().items()}
    asym_gate_shapes = {key: tuple(value.shape) for key, value in v12_gate.state_dict().items()}
    if expert_shapes != asym_expert_shapes or gate_shapes != asym_gate_shapes:
        raise AssertionError("V12 asymmetric expert/gate parameter shapes diverge from V8")

    v12_p5 = P2IRPromptAsymIdentityGDERMergeFeedback2D(64, 2, 4, 1, 8, 8).eval()
    v13_p5 = P2IRPromptLAFMergeFeedbackNoStaticMAA2D(64, 2, 4, 1, 8).eval()
    v13_state = v13_p5.state_dict()
    v12_state = v12_p5.state_dict()
    v13_p5.load_state_dict({key: v12_state[key] for key in v13_state}, strict=True)
    with torch.no_grad():
        p5_with_gder = v12_p5(inputs)
        p5_without_gder = v13_p5(inputs)
    p5_initial_maximum = max(
        float((left - right).abs().max())
        for left, right in zip(p5_with_gder, p5_without_gder)
    )
    if p5_initial_maximum != 0.0:
        raise AssertionError(f"V13 changes P5 base Prompt/LAF flow: {p5_initial_maximum}")
    print(
        "parent reuse: "
        f"V11-vs-V1 max_abs_diff={maximum:.9g}, "
        f"V8/V12 expert tensors={len(expert_shapes)}, gate tensors={len(gate_shapes)}, "
        f"V12/V13 P5 initial max_abs_diff={p5_initial_maximum:.9g}"
    )
    return maximum


def load_real_training_batch(imgsz):
    cfg = get_cfg(
        overrides={
            "imgsz": imgsz,
            "task": "obb",
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
        }
    )
    data = check_det_dataset(str(ROOT / "data/dronevehicle.yaml"))
    dataset = build_yolo_dataset(
        cfg,
        data["train"],
        data["train_ir"],
        1,
        data,
        mode="train",
        rect=False,
        stride=32,
    )
    batch = dataset.collate_fn([dataset[0]])
    batch["img"] = batch["img"].float() / 255.0
    return cfg, batch


def check_real_batch_backward(version, path, cfg, batch):
    torch.manual_seed(111)
    model = P2SecondGenOBBModel(str(path), verbose=False).train()
    model.args = cfg
    model.p2_prompt_epoch = 10
    total, items = model(batch)
    require_finite(f"{version} loss", (total, items))
    if items.numel() != 6 or items[4].item() != 0.0 or items[5].item() != 0.0:
        raise AssertionError(f"{version} must have six loss fields with both RGB losses exactly zero: {items}")
    total.backward()

    gradients = {}
    for stage, module in enumerate(model.prompt_modules, start=3):
        gradients[f"P{stage}.ir_head"] = grad_sum(
            module.ir_prompt_head.body[-1].weight, f"{version}.P{stage}.ir_head"
        )
        gradients[f"P{stage}.ir_embed"] = grad_sum(
            module.ir_prompt_embed.projection.weight, f"{version}.P{stage}.ir_embed"
        )
        gradients[f"P{stage}.laf"] = grad_sum(
            next(module.merge.parameters()), f"{version}.P{stage}.laf"
        )
        if module.use_gder:
            gradients[f"P{stage}.rgb_reduce"] = grad_sum(
                module.modality_expert.rgb_projection[0].weight, f"{version}.P{stage}.rgb_reduce"
            )
            gradients[f"P{stage}.ir_reduce"] = grad_sum(
                module.modality_expert.ir_projection[0].weight, f"{version}.P{stage}.ir_reduce"
            )
            gradients[f"P{stage}.identity_fusion"] = grad_sum(
                module.modality_expert.fusion[0].weight, f"{version}.P{stage}.identity_fusion"
            )
            gradients[f"P{stage}.final_projection"] = grad_sum(
                module.modality_expert.fusion[-1].weight, f"{version}.P{stage}.final_projection"
            )
            gradients[f"P{stage}.attention"] = grad_sum(
                module.attention_expert.projection.weight, f"{version}.P{stage}.attention"
            )
            gradients[f"P{stage}.channel_gate"] = grad_sum(
                module.expert_gate.body[-1].weight, f"{version}.P{stage}.channel_gate"
            )
    gradients["obb_head"] = grad_sum(
        next(model.model[-1].parameters()), f"{version}.obb_head"
    )
    if version == "v13" and any(
        key.startswith("P5.") and key.split(".", 1)[1] in {
            "rgb_reduce", "ir_reduce", "identity_fusion", "final_projection", "attention", "channel_gate"
        }
        for key in gradients
    ):
        raise AssertionError("V13 backward found P5 GDER gradients")
    print(f"{version} real training batch: losses={items.tolist()} grad_abs_sums={gradients}")
    del model, total, items
    gc.collect()
    return gradients


def check_zero_init_opening(steps=3):
    torch.manual_seed(222)
    module = P2IRPromptAsymIdentityGDERMergeFeedback2D(32, 4, 4, 1, 8, 8).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
    rgb = torch.randn(2, 32, 10, 10)
    ir = torch.randn(2, 32, 10, 10)
    target = torch.randn(2, 32, 10, 10)
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        fused = module((rgb, ir))[2]
        loss = (fused - target).square().mean()
        loss.backward()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "final_projection": grad_sum(module.modality_expert.fusion[-1].weight, "final_projection"),
            "rgb_reduce": grad_sum(module.modality_expert.rgb_projection[0].weight, "rgb_reduce"),
            "ir_reduce": grad_sum(module.modality_expert.ir_projection[0].weight, "ir_reduce"),
            "identity_fusion": grad_sum(module.modality_expert.fusion[0].weight, "identity_fusion"),
            "attention": grad_sum(module.attention_expert.projection.weight, "attention"),
            "channel_gate": grad_sum(module.expert_gate.body[-1].weight, "channel_gate"),
        }
        optimizer.step()
        history.append(record)
    if history[0]["final_projection"] <= 0:
        raise AssertionError("asymmetric expert final projection lacks first-step gradient")
    for key in ("rgb_reduce", "ir_reduce", "identity_fusion", "channel_gate"):
        if history[-1][key] <= 0:
            raise AssertionError(f"{key} did not open within three steps")
    print(f"asymmetric zero-init three-step opening: OK {history}")
    return history


def check_transfers():
    reference = load_p2det_transfer_reference()
    print(
        "canonical -> baseline: "
        f"verified={reference.canonical_baseline_tensors_verified} "
        f"max_abs_diff={reference.canonical_baseline_max_abs_diff:.9g}; "
        f"reconstructed={reference.reconstructed_baseline_tensors_verified} "
        f"max_abs_diff={reference.reconstructed_baseline_max_abs_diff:.9g}"
    )
    reports = {}
    for version, path in VARIANTS.items():
        model = P2SecondGenOBBModel(str(path), verbose=False)
        report = apply_p2det_baseline_transfer(model, reference)
        reports[version] = {
            key: report[key]
            for key in (
                "baseline_p2det_tensors_copied",
                "baseline_p2det_max_abs_diff",
                "matched_parameter_count",
                "total_parameter_count",
                "transfer_ratio",
                "missing_target_tensor_count",
                "unexpected_source_tensor_count",
            )
        }
        print(f"{version} transfer: {reports[version]}")
        del model
        gc.collect()
    return reports


def profile_models():
    profiles = {}
    for version, path in PROFILE_VARIANTS.items():
        model = YOLO(str(path), task="obb").model.eval()
        profiles[version] = {
            "params": sum(parameter.numel() for parameter in model.parameters()),
            "gflops_640": get_flops(model, imgsz=640),
        }
        print(f"{version} profile: {profiles[version]}")
        del model
        gc.collect()
    return profiles


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-real-batch", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.imgsz <= 0 or args.imgsz % 32:
        raise ValueError("--imgsz must be a positive multiple of 32")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    for version, path in VARIANTS.items():
        check_structure(version, path, args.imgsz)
    check_parent_reuse()
    check_zero_init_opening(3)
    if not args.skip_real_batch:
        cfg, batch = load_real_training_batch(args.imgsz)
        for version, path in VARIANTS.items():
            check_real_batch_backward(version, path, cfg, batch)
    if not args.skip_transfer:
        check_transfers()
    if not args.skip_profile:
        profile_models()
    print("P2Det V11-V13 acceptance checks: PASS")


if __name__ == "__main__":
    main()
