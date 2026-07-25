#!/usr/bin/env python3
"""Migrate YOLOv8s-OBB weights to the ADD P3/P4/P5 BottleneckRefine RGB/IR model.

The migration keeps the three newly added Bottleneck refinement blocks at
their initialized values, copies the single-stream backbone into the RGB
branch, copies compatible weights into the IR branch, and copies the shared
neck/OBB head. Shape-mismatched classification outputs are intentionally
skipped when source and target class counts differ.

Usage:
    python tools/make_twostream_obb_weights.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO


DEFAULT_SOURCE = ROOT / "pre-pth/yolov8s-obb.pt"
DEFAULT_TARGET_YAML = ROOT / "yaml/yolov8s-baseline-ADD-P345-BottleneckRefine.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_baseline_add_p345_bottleneck_refine.pt"

# Source YOLOv8s-OBB layer -> target RGB/shared layer. These indices are tied
# to the ADD P3/P4/P5 BottleneckRefine YAML and validated before checkpoint save.
SINGLE_TO_RGB_SHARED = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 9,
    5: 10,
    6: 14,
    7: 15,
    8: 19,
    9: 20,
    12: 31,
    15: 34,
    16: 35,
    18: 37,
    19: 38,
    21: 40,
    22: 41,
}

# Source backbone layer -> target IR layer. C2f -> C2f_Faster transfers are
# key-and-shape checked, so only genuinely compatible tensors are copied.
SINGLE_TO_IR = {
    0: 4,
    1: 5,
    2: 6,
    3: 7,
    4: 11,
    5: 12,
    6: 16,
    7: 17,
    8: 21,
    9: 22,
}

EXPECTED_SOURCE_LAYERS = {
    0: "Conv",
    1: "Conv",
    2: "C2f",
    3: "Conv",
    4: "C2f",
    5: "Conv",
    6: "C2f",
    7: "Conv",
    8: "C2f",
    9: "SPPF",
    22: "OBB",
}

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
    23: "ADD",
    24: "Bottleneck",
    25: "ADD",
    26: "Bottleneck",
    27: "ADD",
    28: "Bottleneck",
    41: "OBB",
}

StateDict = MutableMapping[str, torch.Tensor]


def _layer_type(model, index: int) -> str:
    return type(model.model[index]).__name__


def _validate_architecture(model, expected: Mapping[int, str], label: str, expected_layers: int) -> None:
    if len(model.model) != expected_layers:
        raise RuntimeError(f"{label} model has {len(model.model)} layers; expected {expected_layers}")
    errors = []
    for index, expected_type in expected.items():
        actual_type = _layer_type(model, index)
        if actual_type != expected_type:
            errors.append(f"layer {index}: expected {expected_type}, got {actual_type}")
    if errors:
        raise RuntimeError(f"{label} architecture does not match the migration map: " + "; ".join(errors))


def _replace_layer_index(key: str, source_index: int, target_index: int) -> Optional[str]:
    prefix = f"model.{source_index}."
    if not key.startswith(prefix):
        return None
    return f"model.{target_index}." + key[len(prefix) :]


def _copy_mapped_weights(
    source: Mapping[str, torch.Tensor],
    target: StateDict,
    layer_map: Mapping[int, int],
    tag: str,
) -> Tuple[Dict[str, str], int, int]:
    copied = {}
    missing = []
    mismatched = []
    per_layer = defaultdict(int)

    for source_key, source_value in source.items():
        target_key = None
        source_layer = None
        target_layer = None
        for source_index, target_index in layer_map.items():
            target_key = _replace_layer_index(source_key, source_index, target_index)
            if target_key is not None:
                source_layer, target_layer = source_index, target_index
                break
        if target_key is None:
            continue
        if target_key not in target:
            missing.append((source_key, target_key))
            continue
        if source_value.shape != target[target_key].shape:
            mismatched.append((source_key, target_key, tuple(source_value.shape), tuple(target[target_key].shape)))
            continue

        target[target_key] = source_value.detach().to(dtype=target[target_key].dtype).clone()
        copied[target_key] = source_key
        per_layer[(source_layer, target_layer)] += 1

    print(f"[{tag}] copied={len(copied)}, missing={len(missing)}, shape_mismatch={len(mismatched)}")
    print(f"[{tag}] copied_by_layer=" + ", ".join(f"{src}->{dst}:{count}" for (src, dst), count in per_layer.items()))
    for source_key, target_key, source_shape, target_shape in mismatched:
        print(f"[{tag}] skip shape {source_key} {source_shape} -> {target_key} {target_shape}")
    return copied, len(missing), len(mismatched)


def _assert_copied_values(
    source: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor], copied: Mapping[str, str]
) -> None:
    for target_key, source_key in copied.items():
        expected = source[source_key].to(dtype=target[target_key].dtype)
        if not torch.equal(target[target_key].cpu(), expected.cpu()):
            raise RuntimeError(f"copied tensor verification failed: {source_key} -> {target_key}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Single-stream YOLOv8s-OBB checkpoint")
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML, help="Two-stream target YAML")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output two-stream checkpoint")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_path = args.source.expanduser().resolve()
    target_yaml = args.target_yaml.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source_path}")
    if not target_yaml.is_file():
        raise FileNotFoundError(f"target YAML not found: {target_yaml}")
    if output_path == source_path:
        raise ValueError("output path must be different from source path")

    print(f"[INFO] source: {source_path}")
    source_yolo = YOLO(str(source_path), task="obb")
    source_model = source_yolo.model.float()
    _validate_architecture(source_model, EXPECTED_SOURCE_LAYERS, "source", expected_layers=23)

    print(f"[INFO] target: {target_yaml}")
    target_yolo = YOLO(str(target_yaml), task="obb")
    target_model = target_yolo.model.float()
    _validate_architecture(target_model, EXPECTED_TARGET_LAYERS, "target", expected_layers=42)

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    rgb_copied, rgb_missing, rgb_mismatch = _copy_mapped_weights(
        source_state, target_state, SINGLE_TO_RGB_SHARED, "single->rgb/shared"
    )
    ir_copied, ir_missing, ir_mismatch = _copy_mapped_weights(
        source_state, target_state, SINGLE_TO_IR, "single->ir"
    )

    if rgb_missing:
        raise RuntimeError(f"RGB/shared migration encountered {rgb_missing} missing target keys")
    if not rgb_copied or not ir_copied:
        raise RuntimeError("migration copied no weights into one or more target branches")

    target_model.load_state_dict(target_state, strict=True)
    loaded_state = target_model.state_dict()
    _assert_copied_values(source_state, loaded_state, rgb_copied)
    _assert_copied_values(source_state, loaded_state, ir_copied)

    target_model.args = {"task": "obb", "model": str(target_yaml)}
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
        "train_args": {"task": "obb", "model": str(target_yaml)},
        "migration": {
            "source": str(source_path),
            "target_yaml": str(target_yaml),
            "rgb_shared_copied": len(rgb_copied),
            "rgb_shared_shape_mismatch": rgb_mismatch,
            "ir_copied": len(ir_copied),
            "ir_missing": ir_missing,
            "ir_shape_mismatch": ir_mismatch,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(checkpoint, str(temporary_path))
    temporary_path.replace(output_path)
    print(f"[DONE] saved: {output_path}")
    print(f"[DONE] size: {output_path.stat().st_size / (1024 ** 2):.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
