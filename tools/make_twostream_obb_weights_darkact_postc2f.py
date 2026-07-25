#!/usr/bin/env python3
"""Migrate YOLOv8s-OBB weights to the post-C2f StaticMAA/LAF model."""

from __future__ import annotations

import argparse
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

from make_twostream_obb_weights_darkact_v2 import (
    DEFAULT_SOURCE,
    EXPECTED_SOURCE_LAYERS,
    _assert_copied_values,
    _copy_mapped_weights,
    _validate_architecture,
)


DEFAULT_TARGET_YAML = ROOT / "yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4-PostC2f-v1.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_staticmaa_laffeedback_postc2f_p345_v1.pt"

SINGLE_TO_RGB_SHARED = {
    0: 0, 1: 1, 2: 2, 3: 3,
    4: 8, 5: 12, 6: 14, 7: 18, 8: 20, 9: 21,
    12: 28, 15: 31, 16: 32, 18: 34, 19: 35, 21: 37, 22: 38,
}
SINGLE_TO_IR = {
    0: 4, 1: 5, 2: 6, 3: 7,
    4: 9, 5: 13, 6: 15, 7: 19, 8: 22, 9: 23,
}
EXPECTED_TARGET_LAYERS = {
    0: "Conv", 4: "Conv", 6: "C2f_Faster",
    8: "C2f", 9: "C2f_Faster", 10: "StaticMAA2D", 11: "LAFMergeFeedback2D",
    12: "Conv", 14: "C2f", 15: "C2f_Faster", 16: "StaticMAA2D", 17: "LAFMergeFeedback2D",
    18: "Conv", 20: "C2f", 21: "SPPF", 22: "C2f_Faster", 23: "SPPF",
    24: "StaticMAA2D", 25: "LAFMergeFeedback2D", 38: "OBB",
}


def _run_migration(source_path, target_yaml, output_path, rgb_map, ir_map, expected_target, expected_layers):
    source_path = source_path.expanduser().resolve()
    target_yaml = target_yaml.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source_path}")
    if not target_yaml.is_file():
        raise FileNotFoundError(f"target YAML not found: {target_yaml}")
    if output_path == source_path:
        raise ValueError("output path must be different from source path")

    print(f"[INFO] source: {source_path}")
    source_model = YOLO(str(source_path), task="obb").model.float()
    _validate_architecture(source_model, EXPECTED_SOURCE_LAYERS, "source", 23)
    print(f"[INFO] target: {target_yaml}")
    target_model = YOLO(str(target_yaml), task="obb").model.float()
    _validate_architecture(target_model, expected_target, "target", expected_layers)

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    rgb_copied, rgb_missing, rgb_mismatch = _copy_mapped_weights(
        source_state, target_state, rgb_map, "single->rgb/shared"
    )
    ir_copied, ir_missing, ir_mismatch = _copy_mapped_weights(
        source_state, target_state, ir_map, "single->ir"
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


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = _parse_args()
    _run_migration(
        args.source, args.target_yaml, args.output,
        SINGLE_TO_RGB_SHARED, SINGLE_TO_IR, EXPECTED_TARGET_LAYERS, 39,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
