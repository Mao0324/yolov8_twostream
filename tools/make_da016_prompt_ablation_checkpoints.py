#!/usr/bin/env python3
"""Build strict DA016-parent initialization checkpoints for registered ablations."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.obb.p2det_object_reliability_train import (
    DA016PromptAblationOBBModel,
)
from ultralytics.nn.modules import (
    DA016DualPromptAuxSemanticLAFMergeFeedback2D,
    DA016DualPromptResidualSemanticLAFMergeFeedback2D,
    DarkACTDualPromptAuxLAFMergeFeedback2D,
    DarkACTDualPromptResidualLAFMergeFeedback2D,
)

from tools.dronevehicle_m2dlif import install_trusted_torch_load


SOURCE = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_da016_semantic_disagreement_laf_p34.pt"
BUILD_SEED = 16016
VARIANTS = {
    "b": {
        "yaml": ROOT / "yaml/yolov8s-DA016-B-DualPromptAuxOnly-P34.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_b_dualprompt_auxonly_p34.pt",
        "stage_modules": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
        },
        "name": "DA016-B-DualPromptAuxOnly-P34",
        "inherits": "A + P3/P4 auxiliary Prompt supervision",
        "parent": "A",
        "prompt_detection_stages": (),
        "train_script": "train_dronevehicle_da016_parent_ablation_b_prompt_aux_m2dlif.py",
    },
    "b_detach": {
        "yaml": ROOT / "yaml/yolov8s-DA016-BDetach-PromptGradientControl-P34.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_b_detach_prompt_gradient_control_p34.pt",
        "stage_modules": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
        },
        "name": "DA016-BDetach-PromptGradientControl-P34",
        "inherits": "B with Prompt inputs detached; auxiliary gradient cannot update detector features",
        "parent": "B",
        "prompt_detection_stages": (),
        "detach_prompt_features": True,
        "train_script": "train_dronevehicle_da016_b_detach_causal_control_m2dlif.py",
    },
    "c": {
        "yaml": ROOT / "yaml/yolov8s-DA016-C-DualPromptZeroInitBoundedResidual-P34.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_c_dualprompt_zero_init_bounded_residual_p34.pt",
        "stage_modules": {
            10: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            15: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
        },
        "name": "DA016-C-DualPromptZeroInitBoundedResidual-P34",
        "inherits": "B + zero-init bounded Prompt residual at P3/P4",
        "parent": "B",
        "prompt_detection_stages": ("P3", "P4"),
        "train_script": "train_dronevehicle_da016_parent_ablation_c_prompt_residual_m2dlif.py",
    },
    "d": {
        "yaml": ROOT / "yaml/yolov8s-DA016-D-DualPromptP3Residual-P4AuxOnly.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_d_dualprompt_p3_residual_p4_auxonly.pt",
        "stage_modules": {
            # P3 inherits C's residual detection path; P4 inherits B exactly.
            10: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
        },
        "name": "DA016-D-DualPromptP3Residual-P4AuxOnly",
        "inherits": "B + C's zero-init bounded Prompt residual at P3 only; P4 remains B",
        "parent": "B",
        "prompt_detection_stages": ("P3",),
        "train_script": "train_dronevehicle_da016_parent_ablation_d_p3_prompt_residual_m2dlif.py",
    },
    "j": {
        "yaml": ROOT / "yaml/yolov8s-DA016-J-DualPromptP4Residual-P3AuxOnly.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_j_dualprompt_p4_residual_p3_auxonly.pt",
        "stage_modules": {
            # P3 inherits B exactly; P4 inherits C's residual detection path.
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
        },
        "name": "DA016-J-DualPromptP4Residual-P3AuxOnly",
        "inherits": "B + C's zero-init bounded Prompt residual at P4 only; P3 remains B",
        "parent": "B",
        "prompt_detection_stages": ("P4",),
        "train_script": "train_dronevehicle_da016_parent_ablation_j_p4_prompt_residual_m2dlif.py",
    },
    "e": {
        "yaml": ROOT / "yaml/yolov8s-DA016-E-CPlusP5DualPromptAuxOnly.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_e_c_plus_p5_dualprompt_auxonly.pt",
        "stage_modules": {
            # P3/P4 inherit C exactly. P5 wraps the original DarkACT LAF only
            # to expose auxiliary Prompt maps; its detection forward is exact.
            10: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            15: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            22: DarkACTDualPromptAuxLAFMergeFeedback2D,
        },
        "name": "DA016-E-CPlusP5DualPromptAuxOnly",
        "inherits": "C + P5 auxiliary Prompt supervision; original P5 DarkACT LAF detection path",
        "parent": "C",
        "prompt_detection_stages": ("P3", "P4"),
        "train_script": "train_dronevehicle_da016_parent_ablation_e_p5_prompt_aux_m2dlif.py",
    },
    "f": {
        "yaml": ROOT / "yaml/yolov8s-DA016-F-CPlusP5DualPromptResidual.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_f_e_plus_p5_dualprompt_residual.pt",
        "stage_modules": {
            # P3/P4 inherit C/E exactly. P5 inherits E and adds one scalar,
            # zero-initialized bounded Prompt residual gain.
            10: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            15: DA016DualPromptResidualSemanticLAFMergeFeedback2D,
            22: DarkACTDualPromptResidualLAFMergeFeedback2D,
        },
        "name": "DA016-F-CPlusP5DualPromptResidual",
        "inherits": "E + zero-init bounded Prompt residual at P5",
        "parent": "E",
        "prompt_detection_stages": ("P3", "P4", "P5"),
        "train_script": "train_dronevehicle_da016_parent_ablation_f_p5_prompt_residual_m2dlif.py",
    },
    "g": {
        "yaml": ROOT / "yaml/yolov8s-DA016-G-BPlusP5DualPromptAuxOnly.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_g_b_plus_p5_dualprompt_auxonly.pt",
        "stage_modules": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            22: DarkACTDualPromptAuxLAFMergeFeedback2D,
        },
        "name": "DA016-G-BPlusP5DualPromptAuxOnly",
        "inherits": "B + P5 auxiliary Prompt supervision; no Prompt enters detection",
        "parent": "B",
        "prompt_detection_stages": (),
        "train_script": "train_dronevehicle_da016_g_b_plus_p5_prompt_aux_m2dlif.py",
    },
    "h25": {
        "yaml": ROOT / "yaml/yolov8s-DA016-H25-BPlusClassWinnerBalancedAux-P34.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_h25_b_class_winner_balanced_aux_p34.pt",
        "stage_modules": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
        },
        "name": "DA016-H25-BPlusClassWinnerBalancedAux-P34",
        "inherits": "B + 25% interpolation toward class-by-teacher-winner balanced Prompt loss",
        "parent": "B",
        "prompt_detection_stages": (),
        "class_winner_balance_ratio": 0.25,
        "train_script": "train_dronevehicle_da016_h25_class_winner_balanced_aux_m2dlif.py",
    },
    "h50": {
        "yaml": ROOT / "yaml/yolov8s-DA016-H50-BPlusClassWinnerBalancedAux-P34.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_h50_b_class_winner_balanced_aux_p34.pt",
        "stage_modules": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
        },
        "name": "DA016-H50-BPlusClassWinnerBalancedAux-P34",
        "inherits": "B + 50% interpolation toward class-by-teacher-winner balanced Prompt loss",
        "parent": "B",
        "prompt_detection_stages": (),
        "class_winner_balance_ratio": 0.50,
        "train_script": "train_dronevehicle_da016_h50_class_winner_balanced_aux_m2dlif.py",
    },
    "h75": {
        "yaml": ROOT / "yaml/yolov8s-DA016-H75-BPlusClassWinnerBalancedAux-P34.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_h75_b_class_winner_balanced_aux_p34.pt",
        "stage_modules": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
        },
        "name": "DA016-H75-BPlusClassWinnerBalancedAux-P34",
        "inherits": "B + 75% interpolation toward class-by-teacher-winner balanced Prompt loss",
        "parent": "B",
        "prompt_detection_stages": (),
        "class_winner_balance_ratio": 0.75,
        "train_script": "train_dronevehicle_da016_h75_class_winner_balanced_aux_m2dlif.py",
    },
    "h": {
        "yaml": ROOT / "yaml/yolov8s-DA016-H-BPlusClassWinnerBalancedAux-P34.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_da016_h_b_class_winner_balanced_aux_p34.pt",
        "stage_modules": {
            10: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
            15: DA016DualPromptAuxSemanticLAFMergeFeedback2D,
        },
        "name": "DA016-H-BPlusClassWinnerBalancedAux-P34",
        "inherits": "B + equal aggregation over present (GT class, teacher winner) groups",
        "parent": "B",
        "prompt_detection_stages": (),
        "class_winner_balance_ratio": 1.0,
        "train_script": "train_dronevehicle_da016_h_b_class_winner_balanced_aux_m2dlif.py",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_tensor(model, sample):
    prediction = model(sample)
    return (prediction[0] if isinstance(prediction, (tuple, list)) else prediction).float()


def build_variant(key, source, sample):
    spec = VARIANTS[key]
    target_yaml, output = spec["yaml"], spec["output"]
    if not target_yaml.is_file():
        raise FileNotFoundError(f"target YAML not found: {target_yaml}")

    # The same seed gives matching stages identical Prompt-head initialization.
    # Every residual gain is an additional exactly-zero scalar at initialization.
    torch.manual_seed(BUILD_SEED)
    target = DA016PromptAblationOBBModel(
        str(target_yaml), ch=3, nc=None, verbose=True
    ).float().eval()
    if len(target.model) != 36:
        raise RuntimeError(f"{key.upper()} target has {len(target.model)} layers; expected 36")
    for index in sorted(spec["stage_modules"]):
        module = target.model[index]
        expected_module = spec["stage_modules"][index]
        if type(module) is not expected_module:
            raise RuntimeError(
                f"{key.upper()} layer {index}: expected {expected_module.__name__}, "
                f"got {type(module).__name__}"
            )
        if module.rgb_prompt_head.body[-1].bias is not None or module.ir_prompt_head.body[-1].bias is not None:
            raise RuntimeError(f"{key.upper()} layer {index}: Prompt output must be bias-free")
        if not torch.count_nonzero(module.rgb_prompt_head.body[-1].weight).eq(0):
            raise RuntimeError(f"{key.upper()} layer {index}: RGB Prompt output is not zero-initialized")
        if not torch.count_nonzero(module.ir_prompt_head.body[-1].weight).eq(0):
            raise RuntimeError(f"{key.upper()} layer {index}: IR Prompt output is not zero-initialized")
        expects_residual = issubclass(
            expected_module,
            (
                DA016DualPromptResidualSemanticLAFMergeFeedback2D,
                DarkACTDualPromptResidualLAFMergeFeedback2D,
            ),
        )
        if expects_residual and float(module.prompt_residual_gain) != 0.0:
            raise RuntimeError(
                f"{key.upper()} layer {index}: Prompt residual gain is not exactly zero"
            )
        if not expects_residual and hasattr(module, "raw_prompt_residual_gain"):
            raise RuntimeError(
                f"{key.upper()} layer {index}: auxiliary-only stage unexpectedly has a residual"
            )
        expected_detach = bool(spec.get("detach_prompt_features", False))
        if bool(getattr(module, "detach_prompt_features", False)) != expected_detach:
            raise RuntimeError(
                f"{key.upper()} layer {index}: detach_prompt_features does not equal {expected_detach}"
            )

    expected_balance = float(spec.get("class_winner_balance_ratio", 0.0))
    configured_balance = float(target.yaml.get("p2_object_class_winner_balance_ratio", 0.0))
    if configured_balance != expected_balance:
        raise RuntimeError(
            f"{key.upper()}: class-winner balance ratio {configured_balance} != {expected_balance}"
        )

    source_state, target_state = source.state_dict(), target.state_dict()
    copied = []
    with torch.no_grad():
        for name, value in source_state.items():
            if name not in target_state or target_state[name].shape != value.shape:
                raise RuntimeError(f"DA016 tensor cannot map into {key.upper()}: {name} {tuple(value.shape)}")
            target_state[name].copy_(value.to(dtype=target_state[name].dtype))
            copied.append(name)
    target.load_state_dict(target_state, strict=True)
    max_transfer_difference = max(
        float((target.state_dict()[name].float() - source_state[name].float()).abs().max())
        for name in copied
    )
    if max_transfer_difference != 0.0:
        raise RuntimeError(f"{key.upper()} DA016 transfer is not exact: {max_transfer_difference}")

    with torch.inference_mode():
        source_prediction = prediction_tensor(source, sample)
        target_prediction = prediction_tensor(target, sample)
    forward_difference = float((source_prediction - target_prediction).abs().max())
    if forward_difference != 0.0:
        raise RuntimeError(
            f"{key.upper()} initialization is not forward-equivalent to DA016: {forward_difference}"
        )

    migration = {
        "ablation": spec["name"],
        "inheritance": spec["inherits"],
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "target_yaml": str(target_yaml),
        "target_yaml_sha256": sha256(target_yaml),
        "build_seed": BUILD_SEED,
        "copied_tensor_count": len(copied),
        "da016_transfer_max_abs_diff": max_transfer_difference,
        "da016_forward_max_abs_diff": forward_difference,
        "semantic_disagreement_path_preserved": True,
        "prompt_enters_detection_path": bool(spec["prompt_detection_stages"]),
        "p3_prompt_enters_detection_path": "P3" in spec["prompt_detection_stages"],
        "p4_prompt_enters_detection_path": "P4" in spec["prompt_detection_stages"],
        "p5_prompt_enters_detection_path": "P5" in spec["prompt_detection_stages"],
        "detach_prompt_features": bool(spec.get("detach_prompt_features", False)),
        "class_winner_balance_ratio": expected_balance,
        "gate_direction_supervision": False,
    }
    target.args = {"task": "obb", "model": str(target_yaml)}
    target.da016_prompt_migration = migration
    checkpoint = {
        "date": datetime.now().isoformat(),
        "version": ultralytics.__version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(target).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {"task": "obb", "model": str(target_yaml)},
        "migration": migration,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    print(f"variant={key.upper()}")
    print(f"saved={output}")
    print(f"parameters={sum(parameter.numel() for parameter in target.parameters())}")
    print(f"copied_tensor_count={len(copied)}")
    print(f"da016_forward_max_abs_diff={forward_difference}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    args = parser.parse_args()
    install_trusted_torch_load(torch)
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"DA016 initialization checkpoint not found: {SOURCE}\n"
            "Build it first with: python -B tools/make_darkact_da016_checkpoint.py"
        )
    source = YOLO(str(SOURCE), task="obb").model.float().eval()
    torch.manual_seed(916016)
    sample = torch.randn(1, 6, 128, 128)
    keys = tuple(VARIANTS) if args.variant == "all" else (args.variant,)
    for key in keys:
        build_variant(key, source, sample)
    from tools.update_da016_experiment_map import update_experiment_map

    print(f"experiment_map={update_experiment_map(VARIANTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
