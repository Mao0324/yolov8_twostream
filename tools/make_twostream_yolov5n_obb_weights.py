#!/usr/bin/env python3
"""Migrate official yolov5nu Detect weights into the two-stream YOLOv5n OBB baseline."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import OBBModel


DEFAULT_SOURCE = ROOT / "pre-pth/yolov5nu.pt"
DEFAULT_TARGET_YAML = ROOT / "yaml/baseline-v5n.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov5nu_twostream_baseline_v5n_obb.pt"
OFFICIAL_SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5nu.pt"
MIGRATION_SEED = 0

# Official YOLOv5u Detect layer -> target RGB/shared layer.
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
    10: 26,
    13: 29,
    14: 30,
    17: 33,
    18: 34,
    20: 36,
    21: 37,
    23: 39,
    24: 40,
}

# The same official backbone is copied into the IR branch.
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
    2: "C3",
    4: "C3",
    6: "C3",
    8: "C3",
    9: "SPPF",
    10: "Conv",
    13: "C3",
    17: "C3",
    20: "C3",
    23: "C3",
    24: "Detect",
}

EXPECTED_TARGET_LAYERS = {
    0: "Conv",
    2: "C3",
    4: "Conv",
    6: "C3",
    8: "RIFusion",
    9: "C3",
    11: "C3",
    13: "RIFusion",
    14: "C3",
    16: "C3",
    18: "RIFusion",
    19: "C3",
    20: "SPPF",
    21: "C3",
    22: "SPPF",
    23: "ADD",
    24: "ADD",
    25: "ADD",
    29: "C3",
    33: "C3",
    36: "C3",
    39: "C3",
    40: "OBB",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_layers(model, expected: Mapping[int, str], expected_count: int, label: str) -> None:
    if len(model.model) != expected_count:
        raise RuntimeError(f"{label} has {len(model.model)} layers; expected {expected_count}")
    errors = []
    for index, expected_type in expected.items():
        actual_type = type(model.model[index]).__name__
        if actual_type != expected_type:
            errors.append(f"layer {index}: expected {expected_type}, got {actual_type}")
    if errors:
        raise RuntimeError(f"{label} architecture mismatch: " + "; ".join(errors))


def _mapped_target_key(source_key: str, layer_map: Mapping[int, int]):
    source_layer = int(source_key.split(".")[1])
    if source_layer not in layer_map:
        return None
    return source_key.replace(
        f"model.{source_layer}.",
        f"model.{layer_map[source_layer]}.",
        1,
    )


def _copy_mapped(source_state, target_state, layer_map, label: str):
    copied = {}
    missing = []
    mismatched = []
    per_layer = defaultdict(int)
    for source_key, source_value in source_state.items():
        target_key = _mapped_target_key(source_key, layer_map)
        if target_key is None:
            continue
        if target_key not in target_state:
            missing.append((source_key, target_key))
            continue
        if source_value.shape != target_state[target_key].shape:
            mismatched.append(
                (source_key, target_key, tuple(source_value.shape), tuple(target_state[target_key].shape))
            )
            continue
        target_state[target_key].copy_(source_value.to(dtype=target_state[target_key].dtype))
        copied[target_key] = source_key
        per_layer[(int(source_key.split(".")[1]), int(target_key.split(".")[1]))] += 1
    print(f"[{label}] copied={len(copied)}, missing={len(missing)}, shape_mismatch={len(mismatched)}")
    print(
        f"[{label}] copied_by_layer="
        + ", ".join(f"{source}->{target}:{count}" for (source, target), count in per_layer.items())
    )
    return copied, missing, mismatched


def _assert_exact(source_state, target_state, copied) -> float:
    maximum = 0.0
    for target_key, source_key in copied.items():
        source_value = source_state[source_key].detach().cpu().float()
        target_value = target_state[target_key].detach().cpu().float()
        if not torch.equal(source_value, target_value):
            maximum = max(maximum, float((source_value - target_value).abs().max()))
    return maximum


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    target_yaml = args.target_yaml.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"official source checkpoint not found: {source}\nDownload it from {OFFICIAL_SOURCE_URL}"
        )
    if not target_yaml.is_file():
        raise FileNotFoundError(f"target YAML not found: {target_yaml}")
    if output == source:
        raise ValueError("output must not overwrite the official source checkpoint")

    source_model = YOLO(str(source), task="detect").model.float()
    _validate_layers(source_model, EXPECTED_SOURCE_LAYERS, 25, "source")
    if source_model.model[-1].nc != 80:
        raise RuntimeError(f"official source Detect head must have 80 classes, got {source_model.model[-1].nc}")

    config = yaml.safe_load(target_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("scale") != "n" or config.get("nc") != 5:
        raise RuntimeError("target YAML must be a mapping with scale:n and nc:5")
    torch.manual_seed(MIGRATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MIGRATION_SEED)
    target_model = OBBModel(cfg=config, ch=3, nc=config["nc"], verbose=False).float()
    _validate_layers(target_model, EXPECTED_TARGET_LAYERS, 41, "target")
    if tuple(target_model.model[0].conv.weight.shape) != (16, 3, 6, 6):
        raise RuntimeError("target did not build at YOLOv5n width")

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    rgb_copied, rgb_missing, rgb_mismatched = _copy_mapped(
        source_state, target_state, SINGLE_TO_RGB_SHARED, "source->rgb/shared"
    )
    ir_copied, ir_missing, ir_mismatched = _copy_mapped(
        source_state, target_state, SINGLE_TO_IR, "source->ir"
    )
    if rgb_missing or ir_missing:
        raise RuntimeError(f"migration has missing keys: rgb={len(rgb_missing)}, ir={len(ir_missing)}")
    if not rgb_copied or not ir_copied:
        raise RuntimeError("migration copied no tensors into one or more branches")
    if not rgb_mismatched or any(not item[0].startswith("model.24.cv3.") for item in rgb_mismatched):
        raise RuntimeError("shape mismatches are not confined to the 80-class Detect classification branch")
    if ir_mismatched:
        raise RuntimeError(f"IR backbone has {len(ir_mismatched)} unexpected shape mismatches")

    target_model.load_state_dict(target_state, strict=True)
    all_copied = {**rgb_copied, **ir_copied}
    migration_maxdiff = _assert_exact(source_state, target_model.state_dict(), all_copied)
    if migration_maxdiff != 0.0:
        raise RuntimeError(f"in-memory migrated tensors are not exact: max_abs_diff={migration_maxdiff}")

    target_parameters = dict(target_model.named_parameters())
    copied_parameter_count = sum(
        target_parameters[key].numel() for key in all_copied if key in target_parameters
    )
    migration = {
        "source": str(source),
        "source_url": OFFICIAL_SOURCE_URL,
        "source_sha256": _sha256(source),
        "target_yaml": str(target_yaml),
        "target_yaml_sha256": _sha256(target_yaml),
        "migration_seed": MIGRATION_SEED,
        "target_scale": "n",
        "source_classes": 80,
        "target_classes": 5,
        "rgb_shared_tensors_copied": len(rgb_copied),
        "ir_tensors_copied": len(ir_copied),
        "total_tensors_copied": len(all_copied),
        "classification_shape_mismatches": len(rgb_mismatched),
        "copied_parameter_count": copied_parameter_count,
        "target_parameter_count": sum(parameter.numel() for parameter in target_model.parameters()),
        "migration_max_abs_diff": migration_maxdiff,
    }
    target_model.args = {"task": "obb", "model": str(target_yaml)}
    target_model.yolov5u_migration = migration
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
        "migration": migration,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    saved_checkpoint = torch.load(temporary, map_location="cpu", weights_only=False)
    saved_model = deepcopy(saved_checkpoint.get("ema") or saved_checkpoint["model"]).float()
    serialized_maxdiff = _assert_exact(source_state, saved_model.state_dict(), all_copied)
    if serialized_maxdiff != 0.0:
        raise RuntimeError(f"serialized migrated tensors are not exact: max_abs_diff={serialized_maxdiff}")
    saved_checkpoint["migration"]["serialized_max_abs_diff"] = serialized_maxdiff
    saved_checkpoint["model"].yolov5u_migration = saved_checkpoint["migration"]
    torch.save(saved_checkpoint, temporary)
    temporary.replace(output)

    print(f"saved={output}")
    print(f"size_mib={output.stat().st_size / 2**20:.2f}")
    print(f"tensors_copied={len(all_copied)}")
    print(f"parameters_copied={copied_parameter_count}")
    print(f"classification_shape_mismatches={len(rgb_mismatched)}")
    print(f"serialized_max_abs_diff={serialized_maxdiff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
