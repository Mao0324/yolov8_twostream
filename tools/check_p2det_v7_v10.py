#!/usr/bin/env python3
"""CPU-safe acceptance checks for P2Det V7-V10; never starts a training run."""

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
from ultralytics.models.yolo.obb.prompt_utils import build_rgb_global_quality_teacher
from ultralytics.nn.modules import (
    P2_SECOND_GEN_PROMPT_MODULES,
    P2DualPromptRGBGlobalIdentityGDERFactorizedMergeFeedback2D,
)
from ultralytics.nn.modules.p2det_prompt_gder import _P2ExpertGate, _P2FactorizedExpertGate
from ultralytics.utils.torch_utils import get_flops

from tools.p2det_weight_transfer import apply_p2det_baseline_transfer, load_p2det_transfer_reference


NEW_VARIANTS = {
    "v7": ROOT / "yaml/yolov8s-P2Det-IRSpatial-RGBGlobal-P34-PostC2f-v7.yaml",
    "v8": ROOT / "yaml/yolov8s-P2Det-DualPrompt-IdentityGDER-P45-NoStaticMAA-v8.yaml",
    "v9": ROOT / "yaml/yolov8s-P2Det-DualPrompt-RGBGlobal-IdentityGDER-P45-NoStaticMAA-v9.yaml",
    "v10": ROOT / "yaml/yolov8s-P2Det-DualPrompt-RGBGlobal-IdentityGDER-FactorizedP4-NoStaticMAA-v10.yaml",
}
PROFILE_VARIANTS = {
    "v1": ROOT / "yaml/yolov8s-P2Det-IRPrompt-P34-PostC2f-v1.yaml",
    "v6": ROOT / "yaml/yolov8s-P2Det-DualPrompt-GDER-P45-NoStaticMAA-v6.yaml",
    **NEW_VARIANTS,
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


def finite(name, value):
    tensors = list(flatten_tensors(value))
    if not tensors or not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise AssertionError(f"{name} is missing tensors or contains NaN/Inf")


def check_structure(version, path, imgsz):
    wrapper = YOLO(str(path), task="obb")
    model = wrapper.model.eval()
    modules = [module for module in model.modules() if isinstance(module, P2_SECOND_GEN_PROMPT_MODULES)]
    for module in modules:
        module.capture_prompt = True

    expert_shapes = []
    attention_shapes = []
    gate_shapes = []
    spatial_logit_shapes = []
    handles = []
    for module in modules:
        if module.use_gder:
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
            handles.append(
                module.expert_gate.register_forward_hook(
                    lambda _m, _a, output: gate_shapes.append(
                        (tuple(output.shape), float((output.sum(dim=1) - 1.0).abs().max()))
                    )
                )
            )
            if module.factorized_gate:
                handles.append(
                    module.expert_gate.spatial_body.register_forward_hook(
                        lambda _m, _a, output: spatial_logit_shapes.append(tuple(output.shape))
                    )
                )

    x = torch.randn(1, 6, imgsz, imgsz)
    try:
        with torch.no_grad():
            output = model(x)
    finally:
        for handle in handles:
            handle.remove()
    finite(f"{version} output", output)

    aux = [module.pop_aux_outputs() for module in modules]
    for module in modules:
        module.pop_gder_weights()
        module.pop_gder_diagnostics()
        module.capture_prompt = False
    for stage, values in enumerate(aux, start=3):
        if values is None:
            raise AssertionError(f"{version} P{stage} did not expose auxiliary outputs")
        finite(f"{version} P{stage} IR spatial", values["ir_spatial_logits"])
        if values["rgb_spatial_logits"] is not None:
            finite(f"{version} P{stage} RGB spatial", values["rgb_spatial_logits"])
        if values["rgb_global_logit"] is not None:
            finite(f"{version} P{stage} RGB global", values["rgb_global_logit"])
            if tuple(values["rgb_global_logit"].shape) != (1, 1, 1, 1):
                raise AssertionError(f"{version} P{stage} invalid global shape")
    if any(error > 1e-6 for _, error in gate_shapes):
        raise AssertionError(f"{version} expert weights do not sum to one: {gate_shapes}")
    if version == "v10":
        expected = [(1, 2, imgsz // 16, imgsz // 16)]
        if spatial_logit_shapes != expected:
            raise AssertionError(f"V10 spatial logits mismatch: {spatial_logit_shapes} != {expected}")
        p4 = modules[1]
        if p4.expert_gate.gamma_spatial.item() != 0.0:
            raise AssertionError("V10 gamma_spatial is not initialized to zero")

    result = {
        "aux": [
            {key: None if value is None else tuple(value.shape) for key, value in values.items()}
            for values in aux
        ],
        "expert_shapes": expert_shapes,
        "attention_shapes": attention_shapes,
        "gate_shapes": gate_shapes,
        "spatial_logits": spatial_logit_shapes,
    }
    print(f"{version} structure/forward: OK {result}")
    del wrapper, model, modules, output, aux
    gc.collect()
    return result


def load_real_batches(imgsz, teacher_samples):
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

    count = min(teacher_samples, len(dataset))
    teacher_batch = dataset.collate_fn([dataset[index] for index in range(count)])
    teacher_images = teacher_batch["img"].float() / 255.0
    teacher = build_rgb_global_quality_teacher(teacher_images)
    stats = {
        "samples": count,
        "mean": float(teacher.mean()),
        "std": float(teacher.std(unbiased=False)),
        "min": float(teacher.min()),
        "max": float(teacher.max()),
    }
    print(f"RGB global teacher: {stats}")
    return cfg, batch, stats


def grad_sum(parameter, name):
    grad = parameter.grad
    if grad is None or not torch.isfinite(grad).all():
        raise AssertionError(f"missing or non-finite gradient: {name}")
    return float(grad.detach().abs().sum())


def check_real_batch_backward(version, path, cfg, batch):
    torch.manual_seed(101)
    model = P2SecondGenOBBModel(str(path), verbose=False).train()
    model.args = cfg
    model.p2_prompt_epoch = 10
    total, items = model(batch)
    if items.numel() != 6:
        raise AssertionError(f"{version} loss schema has {items.numel()} fields, expected 6")
    finite(f"{version} loss", (total, items))
    total.backward()

    gradients = {}
    for stage_index, module in enumerate(model.prompt_modules, start=3):
        prefix = f"P{stage_index}"
        gradients[f"{prefix}.ir_head"] = grad_sum(
            module.ir_prompt_head.body[-1].weight, f"{version}.{prefix}.ir_head"
        )
        gradients[f"{prefix}.ir_embed"] = grad_sum(
            module.ir_prompt_embed.projection.weight, f"{version}.{prefix}.ir_embed"
        )
        if module.use_rgb_spatial:
            gradients[f"{prefix}.rgb_spatial_head"] = grad_sum(
                module.rgb_prompt_head.body[-1].weight, f"{version}.{prefix}.rgb_spatial_head"
            )
            gradients[f"{prefix}.rgb_spatial_embed"] = grad_sum(
                module.rgb_prompt_embed.projection.weight, f"{version}.{prefix}.rgb_spatial_embed"
            )
        if module.use_rgb_global:
            gradients[f"{prefix}.rgb_global_head"] = grad_sum(
                module.rgb_global_head.body[-1].weight, f"{version}.{prefix}.rgb_global_head"
            )
            gradients[f"{prefix}.rgb_global_embed"] = grad_sum(
                module.rgb_global_embed.projection.weight, f"{version}.{prefix}.rgb_global_embed"
            )
        if module.use_gder:
            gradients[f"{prefix}.identity_rgb_reduce"] = grad_sum(
                module.modality_expert.rgb_projection[0].weight,
                f"{version}.{prefix}.identity_rgb_reduce",
            )
            gradients[f"{prefix}.identity_ir_reduce"] = grad_sum(
                module.modality_expert.ir_projection[0].weight,
                f"{version}.{prefix}.identity_ir_reduce",
            )
            gradients[f"{prefix}.identity_fusion_final"] = grad_sum(
                module.modality_expert.fusion[-1].weight,
                f"{version}.{prefix}.identity_fusion_final",
            )
            gradients[f"{prefix}.attention_final"] = grad_sum(
                module.attention_expert.projection.weight,
                f"{version}.{prefix}.attention_final",
            )
            gradients[f"{prefix}.gate_final"] = grad_sum(
                module.expert_gate.body[-1].weight,
                f"{version}.{prefix}.gate_final",
            )
            if module.factorized_gate:
                gradients[f"{prefix}.spatial_gate"] = grad_sum(
                    module.expert_gate.spatial_body[-1].weight,
                    f"{version}.{prefix}.spatial_gate",
                )
                gradients[f"{prefix}.gamma_spatial"] = grad_sum(
                    module.expert_gate.gamma_spatial,
                    f"{version}.{prefix}.gamma_spatial",
                )

    print(f"{version} real-batch loss/backward: items={items.tolist()} grad_abs_sums={gradients}")
    del model, total, items
    gc.collect()
    return gradients


def check_zero_init_opening(steps=3):
    torch.manual_seed(202)
    module = P2DualPromptRGBGlobalIdentityGDERFactorizedMergeFeedback2D(
        32, 4, 4, 1, 8, 8
    ).train()
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
            "expert_final_grad": grad_sum(module.modality_expert.fusion[-1].weight, "expert_final"),
            "rgb_reduce_grad": grad_sum(module.modality_expert.rgb_projection[0].weight, "rgb_reduce"),
            "ir_reduce_grad": grad_sum(module.modality_expert.ir_projection[0].weight, "ir_reduce"),
            "fusion_pre_grad": grad_sum(module.modality_expert.fusion[0].weight, "fusion_pre"),
            "attention_final_grad": grad_sum(module.attention_expert.projection.weight, "attention_final"),
            "gate_grad": grad_sum(module.expert_gate.body[-1].weight, "gate"),
            "spatial_gate_grad": grad_sum(module.expert_gate.spatial_body[-1].weight, "spatial_gate"),
            "gamma_grad": grad_sum(module.expert_gate.gamma_spatial, "gamma"),
            "gamma_value_before_step": float(module.expert_gate.gamma_spatial.detach()),
        }
        optimizer.step()
        record["gamma_value_after_step"] = float(module.expert_gate.gamma_spatial.detach())
        history.append(record)

    if history[0]["expert_final_grad"] <= 0 or history[0]["attention_final_grad"] <= 0:
        raise AssertionError("zero-initialized final expert projections did not receive first-step gradients")
    if history[-1]["rgb_reduce_grad"] <= 0 or history[-1]["ir_reduce_grad"] <= 0:
        raise AssertionError("identity expert upstream gradients did not open")
    if history[-1]["gate_grad"] <= 0 or history[-1]["spatial_gate_grad"] <= 0:
        raise AssertionError("factorized gate gradients did not open within three steps")
    if module.expert_gate.gamma_spatial.detach().abs().item() == 0:
        raise AssertionError("gamma_spatial did not leave zero")
    print(f"zero-init gradient opening: OK {history}")
    return history


def check_v10_v9_gate_equivalence():
    torch.manual_seed(303)
    channel_gate = _P2ExpertGate(32, 8).eval()
    factorized_gate = _P2FactorizedExpertGate(32, 8).eval()
    factorized_gate.body.load_state_dict(channel_gate.body.state_dict())
    fused = torch.randn(2, 32, 9, 11)
    prior = torch.randn_like(fused)
    p_rgb = torch.rand(2, 1, 9, 11)
    p_ir = torch.rand(2, 1, 9, 11)
    with torch.no_grad():
        v9 = channel_gate(fused, prior, p_rgb, p_ir)
        v10 = factorized_gate(fused, prior, p_rgb, p_ir)
    maximum = float((v9.expand_as(v10) - v10).abs().max())
    if maximum >= 1e-6:
        raise AssertionError(f"V10 gamma=0 gate differs from V9: {maximum}")
    print(f"V10 gamma=0 vs V9 channel gate: max_abs_diff={maximum:.9g}")
    return maximum


def profile_models():
    results = {}
    for version, path in PROFILE_VARIANTS.items():
        model = YOLO(str(path), task="obb").model.eval()
        results[version] = {
            "params": sum(parameter.numel() for parameter in model.parameters()),
            "gflops_640": get_flops(model, imgsz=640),
        }
        print(f"{version} profile: {results[version]}")
        del model
        gc.collect()
    return results


def check_transfers():
    reference = load_p2det_transfer_reference()
    print(
        "canonical -> baseline: "
        f"verified={reference.canonical_baseline_tensors_verified} "
        f"max_abs_diff={reference.canonical_baseline_max_abs_diff:.9g}; "
        f"reconstructed={reference.reconstructed_baseline_tensors_verified} "
        f"max_abs_diff={reference.reconstructed_baseline_max_abs_diff:.9g}"
    )
    results = {}
    for version, path in NEW_VARIANTS.items():
        model = P2SecondGenOBBModel(str(path), verbose=False)
        report = apply_p2det_baseline_transfer(model, reference)
        results[version] = {
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
        print(f"{version} transfer: {results[version]}")
        del model
        gc.collect()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--teacher-samples", type=int, default=16)
    parser.add_argument("--skip-real-batch", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.imgsz <= 0 or args.imgsz % 32:
        raise ValueError("--imgsz must be a positive multiple of 32")
    if args.threads <= 0 or args.teacher_samples <= 0:
        raise ValueError("--threads and --teacher-samples must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    for version, path in NEW_VARIANTS.items():
        check_structure(version, path, args.imgsz)
    check_v10_v9_gate_equivalence()
    check_zero_init_opening(3)

    if not args.skip_real_batch:
        cfg, batch, _teacher_stats = load_real_batches(args.imgsz, args.teacher_samples)
        for version, path in NEW_VARIANTS.items():
            check_real_batch_backward(version, path, cfg, batch)
    if not args.skip_transfer:
        check_transfers()
    if not args.skip_profile:
        profile_models()
    print("P2Det V7-V10 acceptance checks: PASS")


if __name__ == "__main__":
    main()
