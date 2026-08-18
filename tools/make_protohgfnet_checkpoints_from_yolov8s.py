#!/usr/bin/env python3
"""Migrate canonical YOLOv8s-OBB weights into every ProtoHGFNet variant."""

from __future__ import annotations

import hashlib
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

from make_twostream_obb_weights import (
    EXPECTED_SOURCE_LAYERS,
    SINGLE_TO_IR,
    SINGLE_TO_RGB_SHARED,
    _assert_copied_values,
    _copy_mapped_weights,
    _validate_architecture,
)


SOURCE = ROOT / "pre-pth/yolov8s-obb.pt"
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
EXPECTED_TARGET_LAYERS = {
    0: "Conv",
    4: "Conv",
    6: "C2f_Faster",
    8: "RIFusion",
    9: "C2f",
    11: "C2f_Faster",
    13: "RIFusion",
    14: "C2f",
    16: "C2f_Faster",
    18: "RIFusion",
    19: "C2f",
    21: "C2f_Faster",
    23: "ProtoHypergraphFusion",
    24: "ProtoHypergraphFusion",
    25: "ProtoHypergraphFusion",
    38: "OBB",
}


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
        raise RuntimeError("DDP audit found unused trainable parameters: {}".format(unused[:20]))


def _migrate(source_model, experiment_id, yaml_path, output, fusion_layers):
    target_model = YOLO(str(yaml_path), task="obb").model.float()
    expected_layers = dict(EXPECTED_TARGET_LAYERS)
    if 25 not in fusion_layers:
        expected_layers[25] = "ADD"
    _validate_architecture(target_model, expected_layers, "target", expected_layers=39)
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    rgb_copied, rgb_missing, rgb_mismatch = _copy_mapped_weights(
        source_state, target_state, SINGLE_TO_RGB_SHARED, "single->rgb/shared"
    )
    ir_copied, ir_missing, ir_mismatch = _copy_mapped_weights(
        source_state, target_state, SINGLE_TO_IR, "single->ir"
    )
    if rgb_missing or not rgb_copied or not ir_copied:
        raise RuntimeError(
            "incomplete migration: rgb_missing={}, rgb_copied={}, ir_copied={}".format(
                rgb_missing, len(rgb_copied), len(ir_copied)
            )
        )
    target_model.load_state_dict(target_state, strict=True)
    loaded_state = target_model.state_dict()
    _assert_copied_values(source_state, loaded_state, rgb_copied)
    _assert_copied_values(source_state, loaded_state, ir_copied)
    for layer in fusion_layers:
        module = target_model.model[layer]
        if type(module).__name__ != "ProtoHypergraphFusion":
            raise RuntimeError("layer {} is not ProtoHypergraphFusion".format(layer))
        if module.residual_scale.detach().item() != 0.0:
            raise RuntimeError("layer {} residual_scale is not zero".format(layer))
    _assert_ddp_parameter_coverage(target_model)
    target_model.yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    target_model.args = {"task": "obb", "model": str(yaml_path)}
    checkpoint = {
        "date": datetime.now().isoformat(),
        "version": ultralytics.__version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(target_model).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {"task": "obb", "model": str(yaml_path)},
        "migration": {
            "experiment_id": experiment_id,
            "source": str(SOURCE.resolve()),
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "target_yaml": str(yaml_path.resolve()),
            "rgb_shared_copied": len(rgb_copied),
            "ir_copied": len(ir_copied),
            "rgb_shared_shape_mismatch": rgb_mismatch,
            "ir_shape_mismatch": ir_mismatch,
            "fusion_layers": list(fusion_layers),
            "ddp_unused_trainable_parameters": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    print("[{}] saved {} ({:.2f} MiB), source={}".format(
        experiment_id, output.name, output.stat().st_size / 2**20, SOURCE
    ))


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    torch.manual_seed(0)
    source_model = YOLO(str(SOURCE), task="obb").model.float()
    _validate_architecture(source_model, EXPECTED_SOURCE_LAYERS, "source", expected_layers=23)
    for variant in VARIANTS:
        _migrate(source_model, *variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
