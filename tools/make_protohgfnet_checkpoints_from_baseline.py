#!/usr/bin/env python3
"""Create ProtoHGFNet checkpoints with the exact baseline initialization.

The canonical single-stream checkpoint remains the provenance source. The
existing two-stream baseline checkpoint is used as the authoritative common
state so every fusion variant starts from the same tensors as baseline.
"""

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

from make_twostream_obb_weights import EXPECTED_SOURCE_LAYERS, _validate_architecture


CANONICAL_SOURCE = ROOT / "pre-pth/yolov8s-obb.pt"
BASELINE_CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_baseline.pt"
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
EXPECTED_BASELINE_LAYERS = {
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
    23: "ADD",
    24: "ADD",
    25: "ADD",
    38: "OBB",
}
EXPECTED_VARIANT_LAYERS = {
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


def _copy_baseline_state(baseline_state, target_state):
    missing = []
    mismatched = []
    copied = 0
    for key, value in baseline_state.items():
        if key not in target_state:
            missing.append(key)
            continue
        if target_state[key].shape != value.shape:
            mismatched.append((key, tuple(value.shape), tuple(target_state[key].shape)))
            continue
        target_state[key].copy_(value.detach().to(dtype=target_state[key].dtype))
        copied += 1
    if missing or mismatched:
        raise RuntimeError(
            "baseline/variant common-state mismatch: missing={} mismatched={} first_missing={} first_mismatch={}".format(
                len(missing), len(mismatched), missing[:3], mismatched[:3]
            )
        )
    return copied


def _assert_common_state_equal(baseline_state, target_state):
    unequal = []
    for key, value in baseline_state.items():
        if not torch.equal(value.cpu(), target_state[key].cpu()):
            unequal.append(key)
    if unequal:
        raise RuntimeError("baseline common-state verification failed: {}".format(unequal[:20]))
    return len(baseline_state)


def _migrate(baseline_state, baseline_metadata, experiment_id, yaml_path, output, fusion_layers):
    torch.manual_seed(0)
    target_model = YOLO(str(yaml_path), task="obb").model.float()
    expected_layers = dict(EXPECTED_VARIANT_LAYERS)
    if 25 not in fusion_layers:
        expected_layers[25] = "ADD"
    _validate_architecture(target_model, expected_layers, "target", expected_layers=39)
    target_state = target_model.state_dict()
    copied = _copy_baseline_state(baseline_state, target_state)
    target_model.load_state_dict(target_state, strict=True)
    verified = _assert_common_state_equal(baseline_state, target_model.state_dict())
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
        "optimizer": None,
        "train_args": {"task": "obb", "model": str(yaml_path)},
        "migration": {
            "experiment_id": experiment_id,
            "source": str(BASELINE_CHECKPOINT.resolve()),
            "canonical_source": str(CANONICAL_SOURCE.resolve()),
            "canonical_source_sha256": hashlib.sha256(CANONICAL_SOURCE.read_bytes()).hexdigest(),
            "baseline_reference": str(BASELINE_CHECKPOINT.resolve()),
            "baseline_reference_migration": baseline_metadata,
            "target_yaml": str(yaml_path.resolve()),
            "baseline_common_tensors_copied": copied,
            "baseline_common_tensors_verified": verified,
            "fusion_layers": list(fusion_layers),
            "ddp_unused_trainable_parameters": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    print("[{}] saved {} ({:.2f} MiB), baseline_common={}, source={}".format(
        experiment_id, output.name, output.stat().st_size / 2**20, verified, BASELINE_CHECKPOINT
    ))


def main():
    if not CANONICAL_SOURCE.is_file():
        raise FileNotFoundError(CANONICAL_SOURCE)
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASELINE_CHECKPOINT)
    canonical = YOLO(str(CANONICAL_SOURCE), task="obb").model.float()
    _validate_architecture(canonical, EXPECTED_SOURCE_LAYERS, "canonical source", expected_layers=23)
    baseline_checkpoint = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=False)
    baseline_model = baseline_checkpoint["model"].float()
    _validate_architecture(baseline_model, EXPECTED_BASELINE_LAYERS, "baseline reference", expected_layers=39)
    baseline_state = baseline_model.state_dict()
    baseline_metadata = baseline_checkpoint.get("migration", {})
    for variant in VARIANTS:
        _migrate(baseline_state, baseline_metadata, *variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
