#!/usr/bin/env python3
"""Migrate YOLOv8s-OBB weights to the full-C paper-style DarkAct LAF V3 model.

StaticMAA2D and PaperLAFMergeFeedback2D retain their initialized parameters.
Compatible single-stream weights are copied into the two backbones and shared
neck/head using the V3 YAML's verified layer map.
"""

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
    SINGLE_TO_IR,
    SINGLE_TO_RGB_SHARED,
    _assert_copied_values,
    _copy_mapped_weights,
    _validate_architecture,
)


DEFAULT_TARGET_YAML = ROOT / "yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4-v3.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_staticmaa_paperlaf_fullc_p345_v3.pt"

EXPECTED_TARGET_LAYERS = {
    0: "Conv",
    4: "Conv",
    6: "C2f_Faster",
    8: "StaticMAA2D",
    9: "C2f",
    10: "C2f_Faster",
    11: "PaperLAFMergeFeedback2D",
    12: "Conv",
    14: "StaticMAA2D",
    15: "C2f",
    16: "C2f_Faster",
    17: "PaperLAFMergeFeedback2D",
    18: "Conv",
    20: "StaticMAA2D",
    21: "C2f",
    22: "SPPF",
    23: "C2f_Faster",
    24: "SPPF",
    25: "PaperLAFMergeFeedback2D",
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
        raise ValueError("output path must be different from source path")

    print(f"[INFO] source: {source_path}")
    source_model = YOLO(str(source_path), task="obb").model.float()
    _validate_architecture(source_model, EXPECTED_SOURCE_LAYERS, "source", 23)

    print(f"[INFO] target: {target_yaml}")
    target_model = YOLO(str(target_yaml), task="obb").model.float()
    _validate_architecture(target_model, EXPECTED_TARGET_LAYERS, "target", 39)

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
