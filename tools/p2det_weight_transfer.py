#!/usr/bin/env python3
"""Canonical YOLOv8s-OBB to P2Det transfer with exact two-stream baseline verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO

from tools.make_twostream_obb_weights import (
    EXPECTED_SOURCE_LAYERS,
    SINGLE_TO_IR,
    SINGLE_TO_RGB_SHARED,
    _validate_architecture,
)


CANONICAL_SOURCE = ROOT / "pre-pth/yolov8s-obb.pt"
BASELINE_REFERENCE = ROOT / "pre-pth/yolov8s-obb_twostream_baseline.pt"

# Logical common layers from yaml/baseline.yaml to every index-preserving P2Det YAML.
BASELINE_TO_P2DET = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    9: 8,
    11: 9,
    10: 12,
    12: 13,
    14: 14,
    16: 15,
    15: 18,
    17: 19,
    19: 20,
    20: 21,
    21: 22,
    22: 23,
    28: 28,
    31: 31,
    32: 32,
    34: 34,
    35: 35,
    37: 37,
    38: 38,
}

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


@dataclass
class P2DetTransferReference:
    """Loaded canonical/baseline states and their exact provenance audit."""

    canonical_source: Path
    baseline_reference: Path
    baseline_state: Mapping[str, torch.Tensor]
    canonical_source_sha256: str
    canonical_baseline_tensors_verified: int
    canonical_baseline_max_abs_diff: float
    reconstructed_baseline_tensors_verified: int
    reconstructed_baseline_max_abs_diff: float


def _replace_layer_index(key, source_index, target_index):
    prefix = f"model.{source_index}."
    if not key.startswith(prefix):
        return None
    return f"model.{target_index}." + key[len(prefix) :]


def _mapped_pairs(source, target, layer_map):
    """Yield compatible source/target state keys under a logical layer map."""
    for source_key, source_value in source.items():
        for source_index, target_index in layer_map.items():
            target_key = _replace_layer_index(source_key, source_index, target_index)
            if target_key is None:
                continue
            if target_key in target and source_value.shape == target[target_key].shape:
                yield source_key, target_key
            break


def _max_abs_diff(left, right, key_pairs):
    maximum = 0.0
    count = 0
    for left_key, right_key in key_pairs:
        left_value = left[left_key].detach().cpu()
        right_value = right[right_key].detach().cpu().to(dtype=left_value.dtype)
        count += 1
        if torch.equal(left_value, right_value):
            continue
        difference = float((left_value.float() - right_value.float()).abs().max())
        maximum = max(maximum, difference)
    return count, maximum


def load_p2det_transfer_reference(
    canonical_source=CANONICAL_SOURCE,
    baseline_reference=BASELINE_REFERENCE,
):
    """Load both checkpoints and require canonical->baseline mapped tensors to be exactly equal."""
    canonical_source = Path(canonical_source).expanduser().resolve()
    baseline_reference = Path(baseline_reference).expanduser().resolve()
    if not canonical_source.is_file():
        raise FileNotFoundError(f"canonical source checkpoint not found: {canonical_source}")
    if not baseline_reference.is_file():
        raise FileNotFoundError(f"two-stream baseline reference not found: {baseline_reference}")

    canonical_model = YOLO(str(canonical_source), task="obb").model.float()
    _validate_architecture(canonical_model, EXPECTED_SOURCE_LAYERS, "canonical source", expected_layers=23)
    baseline_checkpoint = torch.load(baseline_reference, map_location="cpu")
    baseline_model = (baseline_checkpoint.get("ema") or baseline_checkpoint.get("model")).float()
    _validate_architecture(baseline_model, EXPECTED_BASELINE_LAYERS, "baseline reference", expected_layers=39)

    canonical_state = canonical_model.state_dict()
    baseline_state = baseline_model.state_dict()
    pairs = list(_mapped_pairs(canonical_state, baseline_state, SINGLE_TO_RGB_SHARED))
    pairs.extend(_mapped_pairs(canonical_state, baseline_state, SINGLE_TO_IR))
    verified, maximum = _max_abs_diff(canonical_state, baseline_state, pairs)
    if not verified or maximum != 0.0:
        raise RuntimeError(
            "canonical yolov8s-obb.pt does not exactly match the two-stream baseline mapping: "
            f"verified={verified}, max_abs_diff={maximum}"
        )

    # Reconstruct the transfer base by overlaying all canonical-mappable
    # tensors onto the authoritative two-stream state. Non-mappable
    # C2f_Faster tensors remain the baseline values. The complete resulting
    # state must still be bit-exact to the reference checkpoint.
    reconstructed_state = {key: value.detach().clone() for key, value in baseline_state.items()}
    with torch.no_grad():
        for canonical_key, baseline_key in pairs:
            reconstructed_state[baseline_key].copy_(
                canonical_state[canonical_key].detach().to(dtype=reconstructed_state[baseline_key].dtype)
            )
    all_baseline_pairs = [(key, key) for key in baseline_state]
    reconstructed_verified, reconstructed_maximum = _max_abs_diff(
        reconstructed_state, baseline_state, all_baseline_pairs
    )
    if reconstructed_maximum != 0.0:
        raise RuntimeError(
            "canonical-overlaid two-stream state differs from baseline reference: "
            f"verified={reconstructed_verified}, max_abs_diff={reconstructed_maximum}"
        )

    return P2DetTransferReference(
        canonical_source=canonical_source,
        baseline_reference=baseline_reference,
        baseline_state=reconstructed_state,
        canonical_source_sha256=hashlib.sha256(canonical_source.read_bytes()).hexdigest(),
        canonical_baseline_tensors_verified=verified,
        canonical_baseline_max_abs_diff=maximum,
        reconstructed_baseline_tensors_verified=reconstructed_verified,
        reconstructed_baseline_max_abs_diff=reconstructed_maximum,
    )


def apply_p2det_baseline_transfer(model, reference):
    """Copy baseline-common tensors into P2Det and require exact equality after loading."""
    if len(model.model) != 39:
        raise RuntimeError(f"P2Det target has {len(model.model)} layers; expected 39")
    target_state = model.state_dict()
    pairs = list(_mapped_pairs(reference.baseline_state, target_state, BASELINE_TO_P2DET))
    if not pairs:
        raise RuntimeError("baseline->P2Det mapping copied no tensors")
    with torch.no_grad():
        for baseline_key, target_key in pairs:
            target_state[target_key].copy_(
                reference.baseline_state[baseline_key].detach().to(dtype=target_state[target_key].dtype)
            )
    model.load_state_dict(target_state, strict=True)
    verified, maximum = _max_abs_diff(reference.baseline_state, model.state_dict(), pairs)
    if maximum != 0.0:
        raise RuntimeError(
            f"P2Det common state differs from two-stream baseline: verified={verified}, max_abs_diff={maximum}"
        )

    target_parameters = dict(model.named_parameters())
    mapped_target_keys = {target_key for _, target_key in pairs}
    mapped_source_keys = {source_key for source_key, _ in pairs}
    missing_target_keys = sorted(set(target_state) - mapped_target_keys)
    unexpected_source_keys = sorted(set(reference.baseline_state) - mapped_source_keys)
    matched_parameters = sum(
        parameter.numel() for key, parameter in target_parameters.items() if key in mapped_target_keys
    )
    total_parameters = sum(parameter.numel() for parameter in target_parameters.values())
    report = {
        "source": str(reference.canonical_source),
        "source_sha256": reference.canonical_source_sha256,
        "baseline_reference": str(reference.baseline_reference),
        "canonical_baseline_tensors_verified": reference.canonical_baseline_tensors_verified,
        "canonical_baseline_max_abs_diff": reference.canonical_baseline_max_abs_diff,
        "reconstructed_baseline_tensors_verified": reference.reconstructed_baseline_tensors_verified,
        "reconstructed_baseline_max_abs_diff": reference.reconstructed_baseline_max_abs_diff,
        "baseline_p2det_tensors_copied": len(pairs),
        "baseline_p2det_tensors_verified": verified,
        "baseline_p2det_max_abs_diff": maximum,
        "matched_parameter_count": matched_parameters,
        "total_parameter_count": total_parameters,
        "transfer_ratio": matched_parameters / total_parameters,
        "matched_target_keys": sorted(mapped_target_keys),
        "missing_target_keys": missing_target_keys,
        "unexpected_source_keys": unexpected_source_keys,
        "matched_target_tensor_count": len(mapped_target_keys),
        "missing_target_tensor_count": len(missing_target_keys),
        "unexpected_source_tensor_count": len(unexpected_source_keys),
    }
    model.p2det_migration = report
    return report


def migrate_p2det_from_canonical(
    model,
    canonical_source=CANONICAL_SOURCE,
    baseline_reference=BASELINE_REFERENCE,
):
    """Load/verify canonical provenance and apply the exact baseline-common P2Det transfer."""
    reference = load_p2det_transfer_reference(canonical_source, baseline_reference)
    return apply_p2det_baseline_transfer(model, reference)


__all__ = (
    "CANONICAL_SOURCE",
    "BASELINE_REFERENCE",
    "BASELINE_TO_P2DET",
    "P2DetTransferReference",
    "load_p2det_transfer_reference",
    "apply_p2det_baseline_transfer",
    "migrate_p2det_from_canonical",
)
