#!/usr/bin/env python3
"""Migrate YOLOv8s-OBB weights to the target-saliency PaperLAF model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_twostream_obb_weights_darkact_postc2f import (  # noqa: E402
    SINGLE_TO_IR,
    SINGLE_TO_RGB_SHARED,
    _run_migration,
)
from make_twostream_obb_weights_darkact_v2 import DEFAULT_SOURCE  # noqa: E402


DEFAULT_TARGET_YAML = ROOT / "yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-v1.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_target_saliency_paperlaf_p345_v1.pt"

EXPECTED_TARGET_LAYERS = {
    0: "Conv",
    4: "Conv",
    6: "C2f_Faster",
    8: "C2f",
    9: "C2f_Faster",
    10: "StaticMAAContext2D",
    11: "TargetSaliencyPaperLAFMergeFeedback2D",
    12: "Conv",
    14: "C2f",
    15: "C2f_Faster",
    16: "StaticMAAContext2D",
    17: "TargetSaliencyPaperLAFMergeFeedback2D",
    18: "Conv",
    20: "C2f",
    21: "SPPF",
    22: "C2f_Faster",
    23: "SPPF",
    24: "StaticMAAContext2D",
    25: "TargetSaliencyPaperLAFMergeFeedback2D",
    38: "OBB",
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = _parse_args()
    _run_migration(
        args.source,
        args.target_yaml,
        args.output,
        SINGLE_TO_RGB_SHARED,
        SINGLE_TO_IR,
        EXPECTED_TARGET_LAYERS,
        39,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
