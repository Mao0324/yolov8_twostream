#!/usr/bin/env python3
"""Read-only counterfactual diagnostics for P2Det V16 reliability prompts/gates."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import types

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def object_average(field, batch, sigma=0.7, minimum_cells=2.0):
    """Gaussian-weight a BCHW field separately inside every GT OBB."""
    _, channels, height, width = field.shape
    boxes = batch["bboxes"].reshape(-1, 5).to(field.device, torch.float32)
    batch_idx = batch["batch_idx"].reshape(-1).to(field.device, torch.long)
    if not boxes.numel():
        return field.new_zeros((0, channels), dtype=torch.float32)
    grid_y = (torch.arange(height, device=field.device, dtype=torch.float32) + 0.5) / height
    grid_x = (torch.arange(width, device=field.device, dtype=torch.float32) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    values = []
    epsilon = torch.finfo(torch.float32).eps
    for box, image_index in zip(boxes, batch_idx):
        cx, cy, box_w, box_h, angle = box
        box_w = box_w.clamp_min(minimum_cells / width)
        box_h = box_h.clamp_min(minimum_cells / height)
        dx, dy = grid_x - cx, grid_y - cy
        cosine, sine = angle.cos(), angle.sin()
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        nx = 2.0 * local_x / box_w.clamp_min(epsilon)
        ny = 2.0 * local_y / box_h.clamp_min(epsilon)
        inside = (nx.abs() <= 1.0) & (ny.abs() <= 1.0)
        weights = torch.exp(-0.5 * ((nx / sigma).square() + (ny / sigma).square())) * inside
        values.append(
            (field[image_index].float() * weights).flatten(1).sum(1) / weights.sum().clamp_min(epsilon)
        )
    return torch.stack(values)


def gate_components(merge, rgb, ir, prompt):
    """Reproduce the four additive V16 gate-logit components."""
    descriptor = torch.cat(
        (
            F.adaptive_avg_pool2d(rgb, 1),
            F.adaptive_max_pool2d(rgb, 1),
            F.adaptive_avg_pool2d(ir, 1),
            F.adaptive_max_pool2d(ir, 1),
        ),
        dim=1,
    )
    batch_size, _, height, width = rgb.shape
    feature_global = merge.global_gate(descriptor).reshape(batch_size, 2, merge.channels, 1, 1)
    local_descriptor = torch.cat((merge._contrast_descriptor(rgb), merge._contrast_descriptor(ir)), dim=1)
    feature_local = merge.local_gate(local_descriptor).reshape(batch_size, 2, 1, height, width)
    prompt_global = merge.prompt_global_gate(F.adaptive_avg_pool2d(prompt, 1)).reshape(
        batch_size, 2, merge.channels, 1, 1
    )
    prompt_local = merge.prompt_local_gate(prompt).reshape(batch_size, 2, 1, height, width)
    return feature_global, feature_local, prompt_global, prompt_local


def weights_from_components(components):
    return torch.softmax(sum(components), dim=1) * 2.0


def modality_field(weights):
    """Convert B2CHW scaled LAF weights to B2HW probabilities."""
    return weights.float().mean(dim=2) / 2.0


def component_margin_field(component, height, width):
    """Return IR-minus-RGB mean-channel logit margin as B1HW."""
    component = component.expand(-1, -1, -1, height, width)
    component = component.mean(dim=2)
    return (component[:, 1:2] - component[:, 0:1]).float()


def fused_with_weights(merge, rgb, ir, weights):
    fused = weights[:, 0] * rgb + weights[:, 1] * ir
    if merge.cross is not None:
        fused = fused + merge.cross_scale * merge.cross(rgb, ir)
    return fused


@contextmanager
def p4_intervention(module, mode):
    """Temporarily change only how the trained P4 merge consumes its Prompt."""
    merge = module.merge
    original = merge.forward

    def forward(this, x, prompt):
        rgb, ir = x
        if mode == "neutral":
            prompt = torch.full_like(prompt, 0.5)
            return original(x, prompt)
        if mode == "swapped":
            return original(x, prompt.flip(1))
        components = gate_components(this, rgb, ir, prompt)
        if mode == "feature_only":
            weights = weights_from_components(components[:2])
        elif mode == "prompt_direct":
            weights = prompt.unsqueeze(2).expand(-1, -1, rgb.shape[1], -1, -1) * 2.0
        elif mode == "equal":
            weights = torch.ones(
                rgb.shape[0], 2, rgb.shape[1], rgb.shape[2], rgb.shape[3],
                device=rgb.device, dtype=rgb.dtype,
            )
        else:
            raise ValueError(mode)
        return fused_with_weights(this, rgb, ir, weights)

    merge.forward = types.MethodType(forward, merge)
    try:
        yield
    finally:
        merge.forward = original


def binary_metrics(prediction, target):
    prediction = prediction.float()
    target = target.float()
    predicted_class = prediction.argmax(1)
    target_class = target.argmax(1)
    accuracy = (predicted_class == target_class).float().mean()
    recalls = []
    for modality in (0, 1):
        mask = target_class == modality
        recalls.append((predicted_class[mask] == modality).float().mean() if mask.any() else accuracy.new_nan())
    balanced = torch.stack(recalls).nanmean()
    mae = (prediction[:, 0] - target[:, 0]).abs().mean()
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "rgb_win_recall": float(recalls[0]),
        "ir_win_recall": float(recalls[1]),
        "mae": float(mae),
        "rgb_win_rate": float((predicted_class == 0).float().mean()),
    }


def pearson(x, y):
    x, y = x.float(), y.float()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.square().sum().sqrt() * y.square().sum().sqrt()).clamp_min(1e-12))


def summarize_stage(records, target):
    tensors = {key: torch.cat(value) for key, value in records.items()}
    target_class = target.argmax(1)
    rgb_teacher_win_rate = float((target_class == 0).float().mean())
    majority_accuracy = max(rgb_teacher_win_rate, 1.0 - rgb_teacher_win_rate)
    global_prior = target.mean(0, keepdim=True).expand_as(target)
    result = {
        "objects": int(target.shape[0]),
        "teacher_rgb_win_rate": rgb_teacher_win_rate,
        "teacher_ir_win_rate": 1.0 - rgb_teacher_win_rate,
        "majority_accuracy": majority_accuracy,
        "teacher_rgb_mean": float(target[:, 0].mean()),
        "teacher_rgb_std": float(target[:, 0].std(unbiased=False)),
        "global_prior_mae": float((global_prior[:, 0] - target[:, 0]).abs().mean()),
    }
    for name in ("prompt", "gate", "feature_only_gate", "neutral_gate", "swapped_gate", "prompt_only_gate"):
        result[name] = binary_metrics(tensors[name], target)
        result[name]["rgb_correlation"] = pearson(tensors[name][:, 0], target[:, 0])
        result[name]["rgb_mean"] = float(tensors[name][:, 0].mean())
        result[name]["rgb_std"] = float(tensors[name][:, 0].std(unbiased=False))
    result["gate_prompt_accuracy"] = float(
        (tensors["gate"].argmax(1) == tensors["prompt"].argmax(1)).float().mean()
    )
    result["counterfactual"] = {
        "normal_minus_feature_only_mae": float((tensors["gate"] - tensors["feature_only_gate"]).abs().mean()),
        "normal_minus_neutral_mae": float((tensors["gate"] - tensors["neutral_gate"]).abs().mean()),
        "normal_minus_swapped_mae": float((tensors["gate"] - tensors["swapped_gate"]).abs().mean()),
        "swap_monotonic_fraction": float(
            (
                (tensors["swapped_prompt"][:, 1] - tensors["prompt"][:, 1])
                * (tensors["swapped_gate"][:, 1] - tensors["gate"][:, 1])
                > 0
            ).float().mean()
        ),
        "swap_prompt_gate_delta_correlation": pearson(
            tensors["swapped_prompt"][:, 1] - tensors["prompt"][:, 1],
            tensors["swapped_gate"][:, 1] - tensors["gate"][:, 1],
        ),
    }
    result["mean_ir_minus_rgb_logit"] = {
        name: float(tensors[name].mean())
        for name in ("feature_global_margin", "feature_local_margin", "prompt_global_margin", "prompt_local_margin")
    }
    result["feature_and_contribution"] = {
        "rgb_raw_energy_fraction": float(tensors["raw_energy_fraction"][:, 0].mean()),
        "rgb_weighted_energy_fraction": float(tensors["weighted_energy_fraction"][:, 0].mean()),
        "rgb_gate_probability": float(tensors["gate"][:, 0].mean()),
    }
    return result


def latest(pattern):
    candidates = [path for path in ROOT.glob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(pattern)
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=24)
    parser.add_argument("--loss-images", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.images, args.loss_images, args.batch, args.imgsz) <= 0:
        raise ValueError("all numeric arguments must be positive")

    from tools.dronevehicle_m2dlif import install_trusted_torch_load, prepare_temporary_dataset

    install_trusted_torch_load(torch)
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.models.yolo.obb.p2det_reliability_train import FrozenSingleModalityTeacherPair
    from ultralytics.nn.modules import P2_RELIABILITY_PROMPT_MODULES
    from ultralytics.utils import yaml_load
    from ultralytics.utils.loss import v8OBBLoss

    device = torch.device(args.device)
    checkpoint = latest(
        "runs/DroneVehicle_OBB_FusionTransfer/"
        "P2D-016_DualReliabilityPrompt-P34-PriorLAF-NoGDER_v16_M2DLIFLabels_v*/weights/best.pt"
    )
    rgb_teacher = latest(
        "runs/DroneVehicle_OBB_SingleModalityTeachers/P2D-Teacher_RGBOnly_M2DLIFLabels_v*/weights/best.pt"
    )
    ir_teacher = latest(
        "runs/DroneVehicle_OBB_SingleModalityTeachers/P2D-Teacher_IROnly_M2DLIFLabels_v*/weights/best.pt"
    )
    model = YOLO(str(checkpoint), task="obb").model.float().to(device).eval()
    if isinstance(model.args, dict):
        model.args = get_cfg(overrides=model.args)
    prompt_modules = [module for module in model.modules() if isinstance(module, P2_RELIABILITY_PROMPT_MODULES)]
    if len(prompt_modules) != 2:
        raise RuntimeError(f"expected P3/P4 Prompt modules, got {len(prompt_modules)}")
    for module in prompt_modules:
        module.capture_prompt = True
        module.merge.capture_gate = True

    captured_inputs = {}
    handles = []
    for name, module in zip(("P3", "P4"), prompt_modules):
        def capture(_, inputs, stage=name):
            rgb, ir = inputs[0]
            captured_inputs[stage] = (rgb.detach(), ir.detach())
        handles.append(module.register_forward_pre_hook(capture))

    teacher_pair = FrozenSingleModalityTeacherPair(
        rgb_teacher, ir_teacher, device, model.names,
        temperature=float(model.yaml.get("p2_teacher_temperature", 0.2)),
        topk=int(model.yaml.get("p2_teacher_match_topk", 128)),
    )
    base_criterion = v8OBBLoss(model)
    stage_records = {"P3": {}, "P4": {}}
    targets = []
    loss_records = {name: [] for name in ("normal", "neutral", "swapped", "feature_only", "prompt_direct", "equal")}

    def append(stage, name, value):
        stage_records[stage].setdefault(name, []).append(value.detach().float().cpu())

    with tempfile.TemporaryDirectory(prefix="p2det_v16_diagnostic_") as temporary:
        data_yaml = prepare_temporary_dataset(Path(temporary))
        data = yaml_load(data_yaml)
        cfg = get_cfg(overrides={
            "imgsz": args.imgsz, "task": "obb", "rect": False, "cache": False,
            "single_cls": False, "classes": None, "fraction": 1.0,
        })
        dataset = build_yolo_dataset(
            cfg, data["val"], data["val_ir"], args.batch, data,
            mode="val", rect=False, stride=32,
        )
        count = min(args.images, len(dataset))
        indices = torch.linspace(0, len(dataset) - 1, steps=count).round().long().unique().tolist()
        processed = 0
        with torch.inference_mode():
            for offset in range(0, len(indices), args.batch):
                selected = indices[offset : offset + args.batch]
                batch = dataset.collate_fn([dataset[index] for index in selected])
                for key, value in tuple(batch.items()):
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)
                batch["img"] = batch["img"].float() / 255.0
                teacher_pair.attach_targets(batch)
                targets.append(batch["p2_reliability_object_targets"].detach().float().cpu())

                for module in prompt_modules:
                    module.capture_prompt = True
                    module.merge.capture_gate = True
                prediction = model(batch["img"])
                auxiliaries = [module.pop_aux_outputs() for module in prompt_modules]

                for stage, module, auxiliary in zip(("P3", "P4"), prompt_modules, auxiliaries):
                    rgb, ir = captured_inputs[stage]
                    prompt = auxiliary["prompt_probabilities"].float()
                    components = gate_components(module.merge, rgb, ir, prompt)
                    normal_weights = weights_from_components(components)
                    if not torch.allclose(normal_weights, auxiliary["modal_weights"].float(), atol=2e-5, rtol=2e-5):
                        raise RuntimeError(f"{stage} recomputed gate differs from captured gate")
                    feature_weights = weights_from_components(components[:2])
                    neutral_prompt = torch.full_like(prompt, 0.5)
                    neutral_components = gate_components(module.merge, rgb, ir, neutral_prompt)
                    neutral_weights = weights_from_components(neutral_components)
                    swapped_prompt = prompt.flip(1)
                    swapped_components = gate_components(module.merge, rgb, ir, swapped_prompt)
                    swapped_weights = weights_from_components(swapped_components)
                    prompt_weights = weights_from_components(components[2:])

                    fields = {
                        "prompt": prompt,
                        "swapped_prompt": swapped_prompt,
                        "gate": modality_field(normal_weights),
                        "feature_only_gate": modality_field(feature_weights),
                        "neutral_gate": modality_field(neutral_weights),
                        "swapped_gate": modality_field(swapped_weights),
                        "prompt_only_gate": modality_field(prompt_weights),
                    }
                    for name, field in fields.items():
                        append(stage, name, object_average(field, batch))

                    height, width = prompt.shape[-2:]
                    for name, component in zip(
                        ("feature_global_margin", "feature_local_margin", "prompt_global_margin", "prompt_local_margin"),
                        components,
                    ):
                        append(stage, name, object_average(component_margin_field(component, height, width), batch))

                    raw_energy = torch.stack(
                        (rgb.float().square().mean(1).sqrt(), ir.float().square().mean(1).sqrt()), dim=1
                    )
                    weighted_energy = torch.stack(
                        (
                            (normal_weights[:, 0].float() * rgb.float()).square().mean(1).sqrt(),
                            (normal_weights[:, 1].float() * ir.float()).square().mean(1).sqrt(),
                        ),
                        dim=1,
                    )
                    raw_energy = raw_energy / raw_energy.sum(1, keepdim=True).clamp_min(1e-12)
                    weighted_energy = weighted_energy / weighted_energy.sum(1, keepdim=True).clamp_min(1e-12)
                    append(stage, "raw_energy_fraction", object_average(raw_energy, batch))
                    append(stage, "weighted_energy_fraction", object_average(weighted_energy, batch))

                if processed < args.loss_images:
                    _, normal_items = base_criterion(prediction, batch)
                    loss_records["normal"].append(normal_items.detach().float().cpu())
                    for module in prompt_modules:
                        module.capture_prompt = False
                        module.merge.capture_gate = False
                    for mode in ("neutral", "swapped", "feature_only", "prompt_direct", "equal"):
                        with p4_intervention(prompt_modules[1], mode):
                            variant_prediction = model(batch["img"])
                        _, items = base_criterion(variant_prediction, batch)
                        loss_records[mode].append(items.detach().float().cpu())
                processed += len(selected)

    for handle in handles:
        handle.remove()
    target = torch.cat(targets)
    report = {
        "checkpoint": str(checkpoint),
        "rgb_teacher": str(rgb_teacher),
        "ir_teacher": str(ir_teacher),
        "images": len(indices),
        "objects": int(target.shape[0]),
        "stages": {
            stage: summarize_stage(stage_records[stage], target) for stage in ("P3", "P4")
        },
        "p4_detection_loss_counterfactual": {},
    }
    for name, values in loss_records.items():
        mean_items = torch.stack(values).mean(0)
        report["p4_detection_loss_counterfactual"][name] = {
            "box": float(mean_items[0]), "cls": float(mean_items[1]), "dfl": float(mean_items[2]),
            "sum": float(mean_items.sum()),
        }
    normal_sum = report["p4_detection_loss_counterfactual"]["normal"]["sum"]
    for values in report["p4_detection_loss_counterfactual"].values():
        values["sum_change_percent"] = (values["sum"] / normal_sum - 1.0) * 100.0
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
