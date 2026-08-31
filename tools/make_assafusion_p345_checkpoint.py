#!/usr/bin/env python3
"""Migrate the requested YOLOv8s-OBB weights into ASSAFusion P3/P4/P5."""

from __future__ import annotations

import argparse
import hashlib
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from tools.make_twostream_obb_weights import (  # noqa: E402
    EXPECTED_SOURCE_LAYERS,
    _assert_copied_values,
    _copy_mapped_weights,
    _validate_architecture,
)


DEFAULT_SOURCE = Path(
    "/media/biiteam/新加卷1/biiteam/MCONG/Yolov8_TwoStream/pre-pth/yolov8s-obb.pt"
)
DEFAULT_TARGET_YAML = ROOT / "yaml/yolov8s-ASSAFusion.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream.pt"
MIGRATION_SEED = 0

SINGLE_TO_RGB_SHARED = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 9,
    5: 11,
    6: 12,
    7: 19,
    8: 20,
    9: 21,
    12: 28,
    15: 31,
    16: 32,
    18: 34,
    19: 35,
    21: 37,
    22: 38,
}

# C2f_Faster receives all tensors whose names and shapes are compatible; its
# remaining tensors keep their deterministic target initialization.
SINGLE_TO_IR = {
    0: 4,
    1: 5,
    2: 6,
    3: 7,
    4: 8,
    5: 13,
    6: 14,
    7: 16,
    8: 17,
    9: 18,
}

EXPECTED_TARGET_LAYERS = {
    0: "Conv",
    4: "Conv",
    6: "C2f_Faster",
    8: "C2f_Faster",
    9: "C2f",
    10: "ASSAFusion",
    11: "Conv",
    12: "C2f",
    13: "Conv",
    14: "C2f_Faster",
    15: "ASSAFusion",
    16: "Conv",
    17: "C2f_Faster",
    18: "SPPF",
    19: "Conv",
    20: "C2f",
    21: "SPPF",
    22: "ASSAFusion",
    23: "ADD",
    24: "ADD",
    25: "ADD",
    38: "OBB",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
        raise ValueError("output path must differ from source checkpoint")

    torch.manual_seed(MIGRATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MIGRATION_SEED)

    print(f"[INFO] source: {source_path}")
    source_model = YOLO(str(source_path), task="obb").model.float()
    _validate_architecture(source_model, EXPECTED_SOURCE_LAYERS, "source", expected_layers=23)

    print(f"[INFO] target: {target_yaml}")
    target_model = YOLO(str(target_yaml), task="obb").model.float()
    _validate_architecture(target_model, EXPECTED_TARGET_LAYERS, "target", expected_layers=39)

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
    migration = {
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "target_yaml": str(target_yaml),
        "target_yaml_sha256": hashlib.sha256(target_yaml.read_bytes()).hexdigest(),
        "migration_seed": MIGRATION_SEED,
        "rgb_shared_copied": len(rgb_copied),
        "rgb_shared_missing": rgb_missing,
        "rgb_shared_shape_mismatch": rgb_mismatch,
        "ir_copied": len(ir_copied),
        "ir_missing": ir_missing,
        "ir_shape_mismatch": ir_mismatch,
        "copied_tensor_count": len(rgb_copied) + len(ir_copied),
    }
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(output_path)
    print(f"[DONE] copied tensors: {migration['copied_tensor_count']}")
    print(f"[DONE] saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
