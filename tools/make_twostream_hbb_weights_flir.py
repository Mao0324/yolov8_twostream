#!/usr/bin/env python3
"""Migrate YOLOv8n HBB weights to the FLIR DarkAct two-stream Detect model."""

from __future__ import annotations

import argparse
import inspect
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

from tools.make_twostream_obb_weights_darkact_v2 import (
    _assert_copied_values,
    _copy_mapped_weights,
    _validate_architecture,
)


DEFAULT_SOURCE = ROOT / "yolov8n.pt"
DEFAULT_TARGET_YAML = (
    ROOT / "fliryaml/yolov8n-DarkAct-SemanticDisagreementLAF-P34-R4-NoStaticMAA-FLIR-HBB_scalen.yaml"
)
DEFAULT_OUTPUT = (
    ROOT / "pre-pth/yolov8n-hbb_twostream_darkact_semantic_disagreement_laf_p34_flir_3class_scalen.pt"
)

# Standard YOLOv8n HBB single-stream layer -> FLIR RGB/shared layer.
SINGLE_TO_RGB_SHARED = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 8,
    5: 11,
    6: 13,
    7: 16,
    8: 18,
    9: 19,
    12: 25,
    15: 28,
    16: 29,
    18: 31,
    19: 32,
    21: 34,
    22: 35,
}

# Copy compatible backbone tensors into the thermal branch. C2f ->
# C2f_Faster tensors with different internal names remain initialized.
SINGLE_TO_IR = {
    0: 4,
    1: 5,
    2: 6,
    3: 7,
    4: 9,
    5: 12,
    6: 14,
    7: 17,
    8: 20,
    9: 21,
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
    22: "Detect",
}
EXPECTED_TARGET_LAYERS = {
    0: "Conv",
    4: "Conv",
    6: "C2f_Faster",
    8: "C2f",
    9: "C2f_Faster",
    10: "SemanticDisagreementLAFMergeFeedback2D",
    11: "Conv",
    13: "C2f",
    14: "C2f_Faster",
    15: "SemanticDisagreementLAFMergeFeedback2D",
    16: "Conv",
    18: "C2f",
    19: "SPPF",
    20: "C2f_Faster",
    21: "SPPF",
    22: "LAFMergeFeedback2D",
    35: "Detect",
}

_torch_load = torch.load
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


def migrate(source_path: Path, target_yaml: Path, output_path: Path) -> dict[str, int]:
    source_path = source_path.expanduser().resolve(strict=True)
    target_yaml = target_yaml.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve()
    if output_path == source_path:
        raise ValueError("output path must be different from source path")

    print(f"[INFO] source: {source_path}")
    source_model = YOLO(str(source_path), task="detect").model.float()
    _validate_architecture(source_model, EXPECTED_SOURCE_LAYERS, "source", 23)
    print(f"[INFO] target: {target_yaml}")
    target_model = YOLO(str(target_yaml), task="detect").model.float()
    _validate_architecture(target_model, EXPECTED_TARGET_LAYERS, "target", 36)

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    rgb_copied, rgb_missing, rgb_mismatch = _copy_mapped_weights(
        source_state, target_state, SINGLE_TO_RGB_SHARED, "single->rgb/shared"
    )
    ir_copied, ir_missing, ir_mismatch = _copy_mapped_weights(
        source_state, target_state, SINGLE_TO_IR, "single->thermal"
    )
    if rgb_missing:
        raise RuntimeError(f"RGB/shared migration encountered {rgb_missing} missing target keys")
    if not rgb_copied or not ir_copied:
        raise RuntimeError("migration copied no weights into one or more target branches")

    target_model.load_state_dict(target_state, strict=True)
    loaded_state = target_model.state_dict()
    _assert_copied_values(source_state, loaded_state, rgb_copied)
    _assert_copied_values(source_state, loaded_state, ir_copied)
    target_model.args = {"task": "detect", "model": str(target_yaml)}

    stats = {
        "rgb_shared_copied": len(rgb_copied),
        "rgb_shared_shape_mismatch": rgb_mismatch,
        "thermal_copied": len(ir_copied),
        "thermal_missing": ir_missing,
        "thermal_shape_mismatch": ir_mismatch,
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
        "train_args": {"task": "detect", "model": str(target_yaml)},
        "migration": {"source": str(source_path), "target_yaml": str(target_yaml), **stats},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(checkpoint, str(temporary_path))
    temporary_path.replace(output_path)
    print(f"[DONE] saved: {output_path}")
    print(f"[DONE] size: {output_path.stat().st_size / (1024 ** 2):.2f} MiB")
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    migrate(args.source, args.target_yaml, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
