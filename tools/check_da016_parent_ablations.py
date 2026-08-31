#!/usr/bin/env python3
"""Audit registered strict DA016-parent checkpoints and DDP metadata."""

from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.dist import generate_ddp_file

from tools.dronevehicle_m2dlif import install_trusted_torch_load
from tools.training_monitor import (
    MonitoredDA016PromptAblationOBBTrainer,
    MonitoredOBBTrainer,
)
from train_dronevehicle_p2det_dual_reliability_p34_v16_m2dlif import validate_ddp_checkpoint
from ultralytics.nn.modules import (
    DA016DualPromptAuxSemanticLAFMergeFeedback2D,
    DA016DualPromptResidualSemanticLAFMergeFeedback2D,
    DarkACTDualPromptAuxLAFMergeFeedback2D,
    DarkACTDualPromptResidualLAFMergeFeedback2D,
    LAFMergeFeedback2D,
    SemanticDisagreementLAFMergeFeedback2D,
)
from ultralytics.models.yolo.obb.p2det_object_reliability_train import class_winner_balanced_mean


CHECKPOINTS = {
    "A": ROOT / "pre-pth/yolov8s-obb_twostream_darkact_da016_semantic_disagreement_laf_p34.pt",
    "B": ROOT / "pre-pth/yolov8s-obb_twostream_da016_b_dualprompt_auxonly_p34.pt",
    "C": ROOT / "pre-pth/yolov8s-obb_twostream_da016_c_dualprompt_zero_init_bounded_residual_p34.pt",
    "D": ROOT / "pre-pth/yolov8s-obb_twostream_da016_d_dualprompt_p3_residual_p4_auxonly.pt",
    "E": ROOT / "pre-pth/yolov8s-obb_twostream_da016_e_c_plus_p5_dualprompt_auxonly.pt",
    "F": ROOT / "pre-pth/yolov8s-obb_twostream_da016_f_e_plus_p5_dualprompt_residual.pt",
    "B-Detach": ROOT / "pre-pth/yolov8s-obb_twostream_da016_b_detach_prompt_gradient_control_p34.pt",
    "G": ROOT / "pre-pth/yolov8s-obb_twostream_da016_g_b_plus_p5_dualprompt_auxonly.pt",
    "H25": ROOT / "pre-pth/yolov8s-obb_twostream_da016_h25_b_class_winner_balanced_aux_p34.pt",
    "H50": ROOT / "pre-pth/yolov8s-obb_twostream_da016_h50_b_class_winner_balanced_aux_p34.pt",
    "H75": ROOT / "pre-pth/yolov8s-obb_twostream_da016_h75_b_class_winner_balanced_aux_p34.pt",
    "H": ROOT / "pre-pth/yolov8s-obb_twostream_da016_h_b_class_winner_balanced_aux_p34.pt",
    "J": ROOT / "pre-pth/yolov8s-obb_twostream_da016_j_dualprompt_p4_residual_p3_auxonly.pt",
}


def prediction_tensor(model, sample):
    prediction = model(sample)
    return (prediction[0] if isinstance(prediction, (tuple, list)) else prediction).float()


def check_ddp_serialization(label, model, checkpoint):
    validate_ddp_checkpoint(model, checkpoint)
    trainer_class = MonitoredOBBTrainer if label == "A" else MonitoredDA016PromptAblationOBBTrainer
    shell = object.__new__(trainer_class)
    shell.args = get_cfg(
        DEFAULT_CFG,
        overrides={
            "model": str(checkpoint.resolve()),
            "task": "obb",
            "data": str((ROOT / "data/dronevehicle.yaml").resolve()),
            "device": "0,1",
            "batch": 64,
            "epochs": 100,
            "project": str((ROOT / "runs/DroneVehicle_OBB/_ddp-audits/da016/seed=000").resolve()),
            "name": f"attempt={label}",
        },
    )
    temporary = Path(generate_ddp_file(shell))
    try:
        content = temporary.read_text(encoding="utf-8")
        expected = f"'model': '{checkpoint.resolve()}'"
        if expected not in content:
            raise RuntimeError(f"{label} DDP temp file does not serialize the checkpoint: {temporary}")
    finally:
        temporary.unlink(missing_ok=True)


def check_p5_wrapper_formula(e_model, f_model):
    """Exercise P5 with nonzero Prompt maps/gain so zero init cannot hide a bad formula."""
    e_module, f_module = e_model.model[22], f_model.model[22]
    if not issubclass(type(f_module), type(e_module)):
        raise RuntimeError("F P5 wrapper does not inherit E P5 wrapper")
    torch.manual_seed(225016)
    rgb = torch.randn(2, f_module.channels, 8, 8)
    ir = torch.randn_like(rgb)
    with torch.no_grad():
        # Initial last-layer weights are zero. Make the two Prompt maps differ
        # only in this in-memory audit model, then use a nonzero bounded alpha.
        f_module.rgb_prompt_head.body[-1].weight.fill_(0.01)
        f_module.ir_prompt_head.body[-1].weight.fill_(-0.01)
        f_module.raw_prompt_residual_gain.fill_(0.4)
        logits, probabilities = f_module._prompt_maps(rgb, ir)
        base_outputs = LAFMergeFeedback2D.forward(f_module, [rgb, ir])
        actual_outputs = f_module([rgb, ir])
        margin = probabilities[:, :1] - probabilities[:, 1:2]
        expected_fused = base_outputs[2] + f_module.prompt_residual_gain * margin * (rgb - ir) / 2
        difference = float((actual_outputs[2] - expected_fused).abs().max())
    if difference > 1e-6:
        raise RuntimeError(f"F P5 nonzero-alpha formula differs by {difference}")
    return difference


def check_auxiliary_inference_skip(model, sample):
    """Aux-only Prompts must be skipped in normal eval and available on explicit capture."""
    calls = []
    hooks = [
        module.rgb_prompt_head.register_forward_hook(lambda *_: calls.append(1))
        for module in model.prompt_modules
    ]
    try:
        model.eval()
        model._set_prompt_capture(False)
        with torch.inference_mode():
            prediction_tensor(model, sample)
        if calls:
            raise RuntimeError("auxiliary-only Prompt ran during standard inference")
        model._set_prompt_capture(True)
        with torch.inference_mode():
            prediction_tensor(model, sample)
        if len(calls) != len(model.prompt_modules):
            raise RuntimeError("explicit reliability capture did not run every Prompt")
        if any(module.pop_aux_outputs() is None for module in model.prompt_modules):
            raise RuntimeError("explicit reliability capture produced no Prompt outputs")
    finally:
        for hook in hooks:
            hook.remove()
        model._set_prompt_capture(False)


def check_detach_gradient(normal_model, detached_model):
    """Prompt loss reaches B features but is cut at B-Detach features."""
    torch.manual_seed(101516)
    gradient_norms = []
    for model in (normal_model, detached_model):
        module = model.model[10].float().train()
        with torch.no_grad():
            module.rgb_prompt_head.body[-1].weight.fill_(0.01)
            module.ir_prompt_head.body[-1].weight.fill_(-0.01)
        rgb = torch.randn(2, module.channels, 12, 12, requires_grad=True)
        ir = torch.randn_like(rgb, requires_grad=True)
        logits, _ = module._prompt_maps(rgb, ir)
        logits.square().mean().backward()
        total = sum(
            float(gradient.abs().sum()) if gradient is not None else 0.0
            for gradient in (rgb.grad, ir.grad)
        )
        gradient_norms.append(total)
    if gradient_norms[0] <= 0 or gradient_norms[1] != 0:
        raise RuntimeError(f"B/B-Detach feature gradients are invalid: {gradient_norms}")
    return gradient_norms


def check_class_winner_balance():
    values = torch.tensor([1.0, 3.0, 10.0])
    target = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    confidence = torch.ones(3)
    classes = torch.tensor([0, 0, 1])
    result = class_winner_balanced_mean(values, target, confidence, classes, nc=5)
    # Present groups are (class 0, RGB): mean 2 and (class 1, IR): mean 10.
    if float(result) != 6.0:
        raise RuntimeError(f"class-winner group balance expected 6, got {float(result)}")


def main():
    install_trusted_torch_load(torch)
    for label, checkpoint in CHECKPOINTS.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint}")
    wrappers = {label: YOLO(str(path), task="obb") for label, path in CHECKPOINTS.items()}
    models = {label: wrapper.model.float().eval() for label, wrapper in wrappers.items()}

    # Stage-level inheritance is the definition of the strict ablation.
    expected_types = {
        "A": {
            10: SemanticDisagreementLAFMergeFeedback2D,
            15: SemanticDisagreementLAFMergeFeedback2D,
            22: LAFMergeFeedback2D,
        },
        "B": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            22: LAFMergeFeedback2D,
        },
        "C": {
            10: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            15: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            22: LAFMergeFeedback2D,
        },
        "D": {
            10: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            22: LAFMergeFeedback2D,
        },
        "E": {
            10: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            15: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            22: DarkACTDualPromptAuxLAFMergeFeedback2D,
        },
        "F": {
            10: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            15: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            22: DarkACTDualPromptResidualLAFMergeFeedback2D,
        },
        "B-Detach": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            22: LAFMergeFeedback2D,
        },
        "G": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            22: DarkACTDualPromptAuxLAFMergeFeedback2D,
        },
        "H25": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            22: LAFMergeFeedback2D,
        },
        "H50": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            22: LAFMergeFeedback2D,
        },
        "H75": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            22: LAFMergeFeedback2D,
        },
        "H": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            22: LAFMergeFeedback2D,
        },
        "J": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            22: LAFMergeFeedback2D,
        },
    }
    for label, expected_stages in expected_types.items():
        model = models[label]
        if len(model.model) != 36:
            raise RuntimeError(f"{label} has {len(model.model)} layers, expected 36")
        for index in (10, 15, 22):
            expected_type = expected_stages[index]
            if type(model.model[index]) is not expected_type:
                raise RuntimeError(
                    f"{label} layer {index}: expected {expected_type.__name__}, "
                    f"got {type(model.model[index]).__name__}"
                )
        check_ddp_serialization(label, wrappers[label], CHECKPOINTS[label])

    source_state = models["A"].state_dict()
    prompt_labels = tuple(label for label in CHECKPOINTS if label != "A")
    for label in prompt_labels:
        target_state = models[label].state_dict()
        difference = max(
            float((target_state[name].float() - value.float()).abs().max())
            for name, value in source_state.items()
        )
        if difference != 0.0:
            raise RuntimeError(f"{label} shared DA016 tensors differ: {difference}")
        if int(models[label].yaml.get("p2_teacher_match_topk", -1)) != 128:
            raise RuntimeError(f"{label} teacher topk is not 128")
        if any(float(value) != 0.0 for value in models[label].yaml["p2_object_gate_direction_stage_gains"]):
            raise RuntimeError(f"{label} unexpectedly enables gate-direction supervision")

    b_state = models["B"].state_dict()
    prompt_keys = [name for name in b_state if "prompt_head" in name]
    for label in ("C", "D", "E", "F", "B-Detach", "G", "H25", "H50", "H75", "H", "J"):
        target_state = models[label].state_dict()
        if not prompt_keys or any(not torch.equal(b_state[name], target_state[name]) for name in prompt_keys):
            raise RuntimeError(f"B/{label} Prompt heads are not identically initialized")
    for index in (10, 15):
        if float(models["C"].model[index].prompt_residual_gain) != 0.0:
            raise RuntimeError(f"C layer {index} residual gain is not zero")
    if float(models["D"].model[10].prompt_residual_gain) != 0.0:
        raise RuntimeError("D P3 residual gain is not zero")
    if hasattr(models["D"].model[15], "raw_prompt_residual_gain"):
        raise RuntimeError("D P4 must inherit B auxiliary-only behavior without a residual")
    if hasattr(models["J"].model[10], "raw_prompt_residual_gain"):
        raise RuntimeError("J P3 must inherit B auxiliary-only behavior without a residual")
    if float(models["J"].model[15].prompt_residual_gain) != 0.0:
        raise RuntimeError("J P4 residual gain is not zero")
    for label in ("E", "F"):
        for index in (10, 15):
            if float(models[label].model[index].prompt_residual_gain) != 0.0:
                raise RuntimeError(f"{label} layer {index} residual gain is not zero")
    if hasattr(models["E"].model[22], "raw_prompt_residual_gain"):
        raise RuntimeError("E P5 must be auxiliary-only without a residual")
    if float(models["F"].model[22].prompt_residual_gain) != 0.0:
        raise RuntimeError("F P5 residual gain is not zero")
    e_p5_state = models["E"].model[22].state_dict()
    f_p5_state = models["F"].model[22].state_dict()
    p5_prompt_keys = [name for name in e_p5_state if "prompt_head" in name]
    if not p5_prompt_keys or any(not torch.equal(e_p5_state[name], f_p5_state[name]) for name in p5_prompt_keys):
        raise RuntimeError("E/F P5 Prompt heads are not identically initialized")
    g_p5_state = models["G"].model[22].state_dict()
    if any(not torch.equal(e_p5_state[name], g_p5_state[name]) for name in p5_prompt_keys):
        raise RuntimeError("E/F/G P5 Prompt heads are not identically initialized")
    if not all(models["B-Detach"].model[index].detach_prompt_features for index in (10, 15)):
        raise RuntimeError("B-Detach did not detach both Prompt inputs")
    for label in ("B", "G", "H25", "H50", "H75", "H"):
        if any(getattr(models[label].model[index], "detach_prompt_features", False) for index in (10, 15)):
            raise RuntimeError(f"{label} unexpectedly detaches Prompt inputs")

    # The H strength sweep must start from exactly the same model tensors.
    h_state = models["H"].state_dict()
    for label in ("H25", "H50", "H75"):
        candidate = models[label].state_dict()
        if candidate.keys() != h_state.keys() or any(
            not torch.equal(candidate[name], value) for name, value in h_state.items()
        ):
            raise RuntimeError(f"{label}/H initialization tensors are not exactly equal")

    # The three-stage loss must preserve C's P3/P4 denominator and add P5 as
    # an extra 0.25-weighted auxiliary term.
    for label in ("E", "F", "G"):
        criterion = models[label].init_criterion()
        if criterion.stage_names != ("P3", "P4", "P5"):
            raise RuntimeError(f"{label} loss does not supervise P3/P4/P5")
        if criterion.stage_weights != (1.0, 0.5, 0.25) or criterion.stage_normalizer != 1.5:
            raise RuntimeError(f"{label} stage weights/normalizer changed")
    expected_balance_ratios = {
        "B": 0.0,
        "B-Detach": 0.0,
        "H25": 0.25,
        "H50": 0.50,
        "H75": 0.75,
        "H": 1.0,
        "J": 0.0,
    }
    for label, expected_ratio in expected_balance_ratios.items():
        criterion = models[label].init_criterion()
        if criterion.stage_names != ("P3", "P4") or criterion.stage_weights != (1.0, 0.5):
            raise RuntimeError(f"{label} does not preserve B's P3/P4 stage setup")
        if criterion.class_winner_balance_ratio != expected_ratio:
            raise RuntimeError(f"{label} class-winner balance ratio is incorrect")
    check_class_winner_balance()

    torch.manual_seed(16016)
    sample = torch.randn(1, 6, 128, 128)
    with torch.inference_mode():
        predictions = {label: prediction_tensor(model, sample) for label, model in models.items()}
    for label in prompt_labels:
        difference = float((predictions[label] - predictions["A"]).abs().max())
        if difference != 0.0:
            raise RuntimeError(f"{label} initial prediction differs from A: {difference}")
    for label in ("B", "B-Detach", "G", "H25", "H50", "H75", "H"):
        check_auxiliary_inference_skip(models[label], sample)
    detach_gradient_norms = check_detach_gradient(models["B"], models["B-Detach"])
    p5_formula_difference = check_p5_wrapper_formula(models["E"], models["F"])

    counts = {label: sum(parameter.numel() for parameter in model.parameters()) for label, model in models.items()}
    print("DA016 strict parent audit: PASS")
    for label in CHECKPOINTS:
        print(f"{label}: checkpoint={CHECKPOINTS[label]}")
        print(f"{label}: params={counts[label]}")
        print(f"{label}: overrides_model={wrappers[label].overrides['model']}")
        print(f"{label}: ckpt_path={wrappers[label].ckpt_path}")
        print(f"{label}: ddp_temp_model=PASS")
    print(f"B_minus_A_params={counts['B'] - counts['A']}")
    print(f"C_minus_A_params={counts['C'] - counts['A']}")
    print(f"D_minus_A_params={counts['D'] - counts['A']}")
    print(f"E_minus_A_params={counts['E'] - counts['A']}")
    print(f"F_minus_A_params={counts['F'] - counts['A']}")
    print(f"B-Detach_minus_A_params={counts['B-Detach'] - counts['A']}")
    print(f"G_minus_A_params={counts['G'] - counts['A']}")
    print(f"H25_minus_A_params={counts['H25'] - counts['A']}")
    print(f"H50_minus_A_params={counts['H50'] - counts['A']}")
    print(f"H75_minus_A_params={counts['H75'] - counts['A']}")
    print(f"H_minus_A_params={counts['H'] - counts['A']}")
    print(f"J_minus_A_params={counts['J'] - counts['A']}")
    print("B_initial_forward_max_abs_diff=0.0")
    print("C_initial_forward_max_abs_diff=0.0")
    print("D_initial_forward_max_abs_diff=0.0")
    print("E_initial_forward_max_abs_diff=0.0")
    print("F_initial_forward_max_abs_diff=0.0")
    print("B-Detach_initial_forward_max_abs_diff=0.0")
    print("G_initial_forward_max_abs_diff=0.0")
    print("H25_initial_forward_max_abs_diff=0.0")
    print("H50_initial_forward_max_abs_diff=0.0")
    print("H75_initial_forward_max_abs_diff=0.0")
    print("H_initial_forward_max_abs_diff=0.0")
    print("J_initial_forward_max_abs_diff=0.0")
    print("B_C_D_E_F_J_P34_prompt_initialization_equal=True")
    print("E_F_P5_prompt_initialization_equal=True")
    print(f"F_P5_nonzero_alpha_formula_max_abs_diff={p5_formula_difference}")
    print(f"B_Prompt_feature_gradient_sum={detach_gradient_norms[0]}")
    print(f"B-Detach_Prompt_feature_gradient_sum={detach_gradient_norms[1]}")
    print("B_BDetach_G_H25_H50_H75_H_standard_inference_skips_auxiliary_Prompt=True")
    print("H_class_winner_balanced_mean_test=PASS")
    print("D_inheritance=P3_from_C_residual,P4_from_B_auxiliary_only,P5_from_A_DarkACT_LAF")
    print("E_inheritance=C_plus_P5_auxiliary_Prompt_wrapping_original_DarkACT_LAF")
    print("F_inheritance=E_plus_P5_zero_init_bounded_Prompt_residual")
    print("G_inheritance=B_plus_P5_auxiliary_Prompt_no_detection_path")
    print("B-Detach_inheritance=B_with_auxiliary_feature_gradient_cut")
    print("H_inheritance=B_plus_class_by_teacher_winner_balanced_auxiliary_loss")
    print("H25_H50_H75_inheritance=B_plus_partial_class_by_teacher_winner_balanced_auxiliary_loss")
    print("H25_H50_H75_H_initial_state_dict_equal=True")
    print("J_inheritance=P3_from_B_auxiliary_only,P4_from_C_residual,P5_from_A_DarkACT_LAF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
