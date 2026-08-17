#!/usr/bin/env python3
"""Create DA-012-equivalent checkpoints for both ASSA-replaced LAF variants."""
from __future__ import annotations

import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO

SOURCE = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_laf_refine_p34_no_staticmaa_v1.pt"
VARIANTS = (
    {
        "name": "P34",
        "layers": (10, 15),
        "yaml": ROOT / "yaml/yolov8s-ASSAReplaceLAFCross-P34-RefineP34-v1.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_assareplace_lafcross_p34_refinep34_v1.pt",
    },
    {
        "name": "P3",
        "layers": (10,),
        "yaml": ROOT / "yaml/yolov8s-ASSAReplaceLAFCross-P3-RefineP34-v1.yaml",
        "output": ROOT / "pre-pth/yolov8s-obb_twostream_assareplace_lafcross_p3_refinep34_v1.pt",
    },
)


def _translated_key(source_key, replaced_layers):
    for layer in replaced_layers:
        prefix = f"model.{layer}.merge."
        if not source_key.startswith(prefix):
            continue
        suffix = source_key[len(prefix):]
        if suffix.startswith(("global_gate.", "local_gate.")):
            return prefix + "reliability." + suffix
        if suffix == "cross_scale" or suffix.startswith("cross."):
            return None
    return source_key


def _tensor_outputs(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_tensor_outputs(item))
        return tensors
    return []


def _assert_ddp_parameter_coverage(model):
    """Fail migration if a trainable parameter is absent from the train graph."""
    audit_model = deepcopy(model).train()
    audit_model.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    outputs = _tensor_outputs(audit_model(torch.randn(2, 6, 64, 64)))
    differentiable = [tensor for tensor in outputs if tensor.requires_grad]
    if not differentiable:
        raise RuntimeError("DDP audit found no differentiable model outputs")
    sum(tensor.float().mean() for tensor in differentiable).backward()
    unused = [
        name for name, parameter in audit_model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if unused:
        raise RuntimeError(f"DDP audit found unused trainable parameters: {unused[:20]}")


def _migrate(source, variant):
    target = YOLO(str(variant["yaml"]), task="obb").model.float()
    source_state, target_state = source.state_dict(), target.state_dict()
    copied, intentionally_removed = [], []
    for source_key, value in source_state.items():
        target_key = _translated_key(source_key, variant["layers"])
        if target_key is None:
            intentionally_removed.append(source_key)
            continue
        if target_key not in target_state or target_state[target_key].shape != value.shape:
            raise RuntimeError(f"cannot migrate {source_key} -> {target_key}")
        target_state[target_key].copy_(value)
        copied.append((source_key, target_key))
    target.load_state_dict(target_state, strict=True)

    for layer in variant["layers"]:
        source_scale = source.model[layer].merge.cross_scale
        if torch.count_nonzero(source_scale).item() != 0:
            raise RuntimeError(f"source layer {layer} LAF cross_scale is not zero")
        for projection in target.model[layer].merge.assa.project_out:
            if torch.count_nonzero(projection.weight).item() != 0:
                raise RuntimeError(f"target layer {layer} ASSA projection is not zero")
        replaced_names = dict(target.model[layer].named_parameters())
        if "merge.reliability.cross_scale" in replaced_names:
            raise RuntimeError(f"target layer {layer} retains unused LAF cross_scale")

    source.eval()
    target.eval()
    torch.manual_seed(0)
    sample = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        source_outputs = _tensor_outputs(source(sample))
        target_outputs = _tensor_outputs(target(sample))
    if len(source_outputs) != len(target_outputs):
        raise RuntimeError("source and target output structures differ")
    max_abs_diff = max((a - b).abs().max().item() for a, b in zip(source_outputs, target_outputs))
    if max_abs_diff > 1e-6:
        raise RuntimeError(f"functional migration mismatch: max_abs_diff={max_abs_diff}")
    _assert_ddp_parameter_coverage(target)

    target.args = {"task": "obb", "model": str(variant["yaml"])}
    checkpoint = {
        "date": datetime.now().isoformat(), "version": ultralytics.__version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com", "epoch": -1,
        "best_fitness": None, "model": deepcopy(target).half(), "ema": None,
        "updates": None, "optimizer": None,
        "train_args": {"task": "obb", "model": str(variant["yaml"])},
        "migration": {
            "source": str(SOURCE), "target_yaml": str(variant["yaml"]),
            "copied_tensors": len(copied),
            "intentionally_removed_laf_cross_tensors": len(intentionally_removed),
            "assa_replaced_layers": list(variant["layers"]),
            "max_abs_diff_from_da012": max_abs_diff,
            "ddp_unused_trainable_parameters": 0,
        },
    }
    output = variant["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    print(
        f"[{variant['name']}] saved {output.name}: {output.stat().st_size / 2**20:.2f} MiB, "
        f"copied={len(copied)}, removed_cross={len(intentionally_removed)}, diff={max_abs_diff}"
    )


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source = YOLO(str(SOURCE), task="obb").model.float()
    for variant in VARIANTS:
        _migrate(source, variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
