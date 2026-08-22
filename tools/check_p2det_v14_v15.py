#!/usr/bin/env python3
"""Acceptance checks for P2Det V14/V15 expert ablations; never starts formal training."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import random
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
    P2IRPromptAsymIdentityGDERMergeFeedback2D,
    P2IRPromptAsymModExpertMergeFeedback2D,
    P2IRPromptAttExpertMergeFeedback2D,
)
from ultralytics.nn.modules.p2det_prompt_gder import (
    _P2AsymmetricIdentityModalityExpert,
    _P2CBAMExpert,
    _P2ExpertGate,
)
from ultralytics.utils.torch_utils import get_flops

from tools.p2det_weight_transfer import apply_p2det_baseline_transfer, load_p2det_transfer_reference


VARIANTS = {
    "v13": ROOT / "yaml/yolov8s-P2Det-IRPrompt-AsymIdentityGDER-P4-NoStaticMAA-PostC2f-v13.yaml",
    "v14": ROOT / "yaml/yolov8s-P2Det-IRPrompt-AsymModExpert-P4-NoStaticMAA-PostC2f-v14.yaml",
    "v15": ROOT / "yaml/yolov8s-P2Det-IRPrompt-AttExpert-P4-NoStaticMAA-PostC2f-v15.yaml",
}
PROFILE_VARIANTS = {
    "v11": ROOT / "yaml/yolov8s-P2Det-IRPrompt-P34-NoStaticMAA-NoGDER-PostC2f-v11.yaml",
    "v12": ROOT / "yaml/yolov8s-P2Det-IRPrompt-AsymIdentityGDER-P45-NoStaticMAA-PostC2f-v12.yaml",
    **VARIANTS,
}
EXPECTED_P4_TYPES = {
    "v13": P2IRPromptAsymIdentityGDERMergeFeedback2D,
    "v14": P2IRPromptAsymModExpertMergeFeedback2D,
    "v15": P2IRPromptAttExpertMergeFeedback2D,
}
EXPECTED_COMPONENTS = {
    "v13": (True, True, True),
    "v14": (True, False, False),
    "v15": (False, True, False),
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
    if not tensors or any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise AssertionError(f"{name} contains no tensor or contains NaN/Inf")


def grad_sum(parameter, name):
    if parameter.grad is None:
        raise AssertionError(f"{name} gradient is missing")
    if not torch.isfinite(parameter.grad).all():
        raise AssertionError(f"{name} gradient contains NaN/Inf")
    return float(parameter.grad.detach().abs().sum())


def component_presence(module):
    module_types = tuple(type(child) for child in module.modules())
    return (
        _P2AsymmetricIdentityModalityExpert in module_types,
        _P2CBAMExpert in module_types,
        _P2ExpertGate in module_types,
    )


def check_structure_and_forward(version, path, imgsz):
    torch.manual_seed(100)
    wrapper = YOLO(str(path), task="obb")
    model = wrapper.model.float().eval()
    prompt_modules = [module for module in model.modules() if hasattr(module, "ir_prompt_head")]
    if len(prompt_modules) != 3:
        raise AssertionError(f"{version} expected P3/P4/P5 prompt modules, got {len(prompt_modules)}")
    p4 = prompt_modules[1]
    if type(p4) is not EXPECTED_P4_TYPES[version]:
        raise AssertionError(f"{version} P4 is {type(p4).__name__}")

    components = component_presence(p4)
    if components != EXPECTED_COMPONENTS[version]:
        raise AssertionError(f"{version} P4 component tree {components} != {EXPECTED_COMPONENTS[version]}")
    parameter_names = tuple(name for name, _ in p4.named_parameters())
    if version == "v14" and any("attention_expert" in name or "expert_gate" in name for name in parameter_names):
        raise AssertionError("V14 retains attention/gate parameters")
    if version == "v15" and any("modality_expert" in name or "expert_gate" in name for name in parameter_names):
        raise AssertionError("V15 retains modality/gate parameters")

    captures = {}
    handles = [
        p4.merge.register_forward_hook(lambda _m, _a, output: captures.update(fused_base=output.detach())),
        p4.register_forward_hook(
            lambda _m, _a, output: captures.update(fused_refined=output[2].detach())
        ),
    ]
    if hasattr(p4, "modality_expert"):
        handles.append(
            p4.modality_expert.register_forward_hook(
                lambda _m, _a, output: captures.update(modality_feature=output.detach())
            )
        )
    if hasattr(p4, "attention_expert"):
        handles.append(
            p4.attention_expert.register_forward_hook(
                lambda _m, _a, output: captures.update(attention_feature=output.detach())
            )
        )
    if hasattr(p4, "expert_gate"):
        handles.append(
            p4.expert_gate.register_forward_hook(
                lambda _m, _a, output: captures.update(expert_gate=output.detach())
            )
        )

    x = torch.randn(1, 6, imgsz, imgsz)
    try:
        with torch.no_grad():
            output = model(x)
    finally:
        for handle in handles:
            handle.remove()
    require_finite(f"{version} detection output", output)
    require_finite(f"{version} P4 captures", captures)

    expected_shape = (1, 256, imgsz // 16, imgsz // 16)
    for key in ("fused_base", "fused_refined", "modality_feature", "attention_feature"):
        if key in captures and tuple(captures[key].shape) != expected_shape:
            raise AssertionError(f"{version} {key} shape {tuple(captures[key].shape)} != {expected_shape}")
    neutral_diff = float((captures["fused_refined"] - captures["fused_base"]).abs().max())
    if neutral_diff >= 1e-6:
        raise AssertionError(f"{version} neutral refinement is not zero: {neutral_diff}")
    for key in ("modality_feature", "attention_feature"):
        if key in captures and float(captures[key].abs().max()) >= 1e-6:
            raise AssertionError(f"{version} neutral {key} is not zero")
    if "expert_gate" in captures:
        gate_sum_error = float((captures["expert_gate"].sum(dim=1) - 1.0).abs().max())
        if gate_sum_error >= 1e-6:
            raise AssertionError(f"{version} gate does not sum to one: {gate_sum_error}")

    summary = {
        "p4_type": type(p4).__name__,
        "components_mod_att_gate": components,
        "p4_shape": expected_shape,
        "neutral_max_abs_diff": neutral_diff,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    print(f"{version} build/structure/forward: OK {summary}")
    del wrapper, model, prompt_modules, p4, output, captures
    gc.collect()
    return summary


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
        raise AssertionError(f"{version} six-field loss/RGB zeros violated: {items}")
    total.backward()

    p4 = model.prompt_modules[1]
    gradients = {
        "ir_prompt_head": grad_sum(p4.ir_prompt_head.body[-1].weight, f"{version}.ir_prompt_head"),
        "ir_prompt_embed": grad_sum(p4.ir_prompt_embed.projection.weight, f"{version}.ir_prompt_embed"),
        "laf": grad_sum(next(p4.merge.parameters()), f"{version}.laf"),
        "obb_head": grad_sum(next(model.model[-1].parameters()), f"{version}.obb_head"),
    }
    if version == "v14":
        gradients["expert_final_projection"] = grad_sum(
            p4.modality_expert.fusion[-1].weight, "v14.expert_final_projection"
        )
    elif version == "v15":
        gradients["expert_final_projection"] = grad_sum(
            p4.attention_expert.projection.weight, "v15.expert_final_projection"
        )
    if any(value <= 0 for value in gradients.values()):
        raise AssertionError(f"{version} required first-step gradient is zero: {gradients}")
    print(f"{version} real-batch loss/backward: OK losses={items.tolist()} grads={gradients}")
    del model, total, items, p4
    gc.collect()
    return gradients


def check_three_step_opening(version, steps=3):
    torch.manual_seed(222)
    module = (
        P2IRPromptAsymModExpertMergeFeedback2D(32, 4, 4, 1, 8, 8)
        if version == "v14"
        else P2IRPromptAttExpertMergeFeedback2D(32, 4, 4, 1, 8, 8)
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
        require_finite(f"{version} step {step} forward/loss", (fused, loss))
        loss.backward()
        if version == "v14":
            record = {
                "step": step,
                "final_projection": grad_sum(module.modality_expert.fusion[-1].weight, "v14.final"),
                "rgb_reduce": grad_sum(module.modality_expert.rgb_projection[0].weight, "v14.rgb_reduce"),
                "ir_reduce": grad_sum(module.modality_expert.ir_projection[0].weight, "v14.ir_reduce"),
                "identity_fusion": grad_sum(module.modality_expert.fusion[0].weight, "v14.fusion"),
            }
        else:
            record = {
                "step": step,
                "final_projection": grad_sum(module.attention_expert.projection.weight, "v15.final"),
                "channel_upstream": grad_sum(module.attention_expert.channel_mlp[0].weight, "v15.channel"),
                "spatial_upstream": grad_sum(module.attention_expert.spatial.weight, "v15.spatial"),
            }
        optimizer.step()
        history.append(record)
    if history[0]["final_projection"] <= 0:
        raise AssertionError(f"{version} final projection has no first-step gradient")
    upstream = ("rgb_reduce", "ir_reduce", "identity_fusion") if version == "v14" else (
        "channel_upstream",
        "spatial_upstream",
    )
    if any(history[-1][key] <= 0 for key in upstream):
        raise AssertionError(f"{version} upstream did not open in three steps: {history}")
    print(f"{version} three-step zero-init opening: OK {history}")
    return history


def check_transfers():
    reference = load_p2det_transfer_reference()
    if reference.canonical_baseline_max_abs_diff != 0.0 or reference.reconstructed_baseline_max_abs_diff != 0.0:
        raise AssertionError("canonical/baseline transfer base is not exact")
    reports = {}
    for version in ("v14", "v15"):
        model = P2SecondGenOBBModel(str(VARIANTS[version]), verbose=False)
        report = apply_p2det_baseline_transfer(model, reference)
        reports[version] = {
            "matched_keys": report["matched_target_tensor_count"],
            "missing_keys": report["missing_target_tensor_count"],
            "unexpected_keys": report["unexpected_source_tensor_count"],
            "matched_parameter_count": report["matched_parameter_count"],
            "total_parameter_count": report["total_parameter_count"],
            "transfer_ratio": report["transfer_ratio"],
            "max_abs_diff": report["baseline_p2det_max_abs_diff"],
        }
        if reports[version]["max_abs_diff"] != 0.0:
            raise AssertionError(f"{version} baseline transfer is not exact")
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
    v13 = profiles["v13"]
    for version in ("v14", "v15"):
        delta = {
            "params_removed": v13["params"] - profiles[version]["params"],
            "gflops_removed": v13["gflops_640"] - profiles[version]["gflops_640"],
        }
        if delta["params_removed"] <= 0 or delta["gflops_removed"] <= 0:
            raise AssertionError(f"{version} is not lighter than V13: {delta}")
        print(f"v13 -> {version}: {delta}")
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
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    for version, path in VARIANTS.items():
        check_structure_and_forward(version, path, args.imgsz)
    for version in ("v14", "v15"):
        check_three_step_opening(version)
    if not args.skip_real_batch:
        cfg, batch = load_real_training_batch(args.imgsz)
        for version in ("v14", "v15"):
            check_real_batch_backward(version, VARIANTS[version], cfg, batch)
    if not args.skip_transfer:
        check_transfers()
    if not args.skip_profile:
        profile_models()
    print("P2Det V14-V15 acceptance checks: PASS")


if __name__ == "__main__":
    main()
