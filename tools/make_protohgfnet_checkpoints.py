#!/usr/bin/env python3
"""Create baseline-equivalent checkpoints for the ProtoHGFNet experiment family."""

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
import yaml
from ultralytics import YOLO


SOURCE = ROOT / "pre-pth/yolov8s-obb_twostream_baseline.pt"
VARIANTS = (
    (
        "PHG-001",
        ROOT / "yaml/protohgf_hard_p345.yaml",
        ROOT / "pre-pth/yolov8s-obb_twostream_protohgf_hard_p345.pt",
        (23, 24, 25),
    ),
    (
        "PHG-002",
        ROOT / "yaml/protohgf_hard_p34.yaml",
        ROOT / "pre-pth/yolov8s-obb_twostream_protohgf_hard_p34.pt",
        (23, 24),
    ),
    (
        "PHG-003",
        ROOT / "yaml/protohgf_soft_p345.yaml",
        ROOT / "pre-pth/yolov8s-obb_twostream_protohgf_soft_p345.pt",
        (23, 24, 25),
    ),
)


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
    audit = deepcopy(model).train()
    audit.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    outputs = [tensor for tensor in _tensor_outputs(audit(torch.randn(2, 6, 64, 64))) if tensor.requires_grad]
    if not outputs:
        raise RuntimeError("DDP audit found no differentiable outputs")
    sum(tensor.float().mean() for tensor in outputs).backward()
    unused = [name for name, parameter in audit.named_parameters() if parameter.requires_grad and parameter.grad is None]
    if unused:
        raise RuntimeError("DDP audit found unused parameters: {}".format(unused[:20]))


def _migrate(source, experiment_id, yaml_path, output, fusion_layers):
    target = YOLO(str(yaml_path), task="obb").model.float()
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied = []
    for key, value in source_state.items():
        if key not in target_state or target_state[key].shape != value.shape:
            raise RuntimeError("cannot migrate baseline tensor {}".format(key))
        target_state[key].copy_(value)
        copied.append(key)
    target.load_state_dict(target_state, strict=True)

    for layer in fusion_layers:
        module = target.model[layer]
        if module.__class__.__name__ != "ProtoHypergraphFusion":
            raise RuntimeError("layer {} is not ProtoHypergraphFusion".format(layer))
        if module.residual_scale.detach().item() != 0.0:
            raise RuntimeError("layer {} residual scale is not zero".format(layer))

    source.eval()
    target.eval()
    torch.manual_seed(0)
    sample = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        source_outputs = _tensor_outputs(source(sample))
        target_outputs = _tensor_outputs(target(sample))
    if len(source_outputs) != len(target_outputs):
        raise RuntimeError("source and target output structures differ")
    max_abs_diff = max((left - right).abs().max().item() for left, right in zip(source_outputs, target_outputs))
    if max_abs_diff > 1e-6:
        raise RuntimeError("functional baseline mismatch: max_abs_diff={}".format(max_abs_diff))
    _assert_ddp_parameter_coverage(target)

    target.yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    target.args = {"task": "obb", "model": str(yaml_path)}
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
        "train_args": {"task": "obb", "model": str(yaml_path)},
        "migration": {
            "experiment_id": experiment_id,
            "source": str(SOURCE),
            "target_yaml": str(yaml_path),
            "copied_tensors": len(copied),
            "fusion_layers": list(fusion_layers),
            "max_abs_diff_from_baseline": max_abs_diff,
            "ddp_unused_trainable_parameters": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    print(
        "[{}] saved {} ({:.2f} MiB), copied={}, diff={}".format(
            experiment_id, output.name, output.stat().st_size / 2**20, len(copied), max_abs_diff
        )
    )


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    torch.manual_seed(0)
    source = YOLO(str(SOURCE), task="obb").model.float()
    for variant in VARIANTS:
        _migrate(source, *variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
