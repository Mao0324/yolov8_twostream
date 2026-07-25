#!/usr/bin/env python3
"""Migrate YOLOv8s-OBB weights to post-C2f StaticMAA/refine/LAF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_twostream_obb_weights_darkact_postc2f import DEFAULT_SOURCE, _run_migration


DEFAULT_TARGET_YAML = ROOT / "yaml/yolov8s-DarkAct-MAA2D-Refine-LAFMerge-P345-R4-PostC2f-v1.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_staticmaa_refine_laffeedback_postc2f_p345_e025_v1.pt"

SINGLE_TO_RGB_SHARED = {
    0: 0, 1: 1, 2: 2, 3: 3,
    4: 8, 5: 14, 6: 16, 7: 22, 8: 24, 9: 25,
    12: 34, 15: 37, 16: 38, 18: 40, 19: 41, 21: 43, 22: 44,
}
SINGLE_TO_IR = {
    0: 4, 1: 5, 2: 6, 3: 7,
    4: 9, 5: 15, 6: 17, 7: 23, 8: 26, 9: 27,
}
EXPECTED_TARGET_LAYERS = {
    0: "Conv", 4: "Conv", 6: "C2f_Faster",
    8: "C2f", 9: "C2f_Faster", 10: "StaticMAA2D",
    11: "ZeroInitResidualRefine2D", 12: "ZeroInitResidualRefine2D", 13: "LAFMergeFeedback2D",
    14: "Conv", 16: "C2f", 17: "C2f_Faster", 18: "StaticMAA2D",
    19: "ZeroInitResidualRefine2D", 20: "ZeroInitResidualRefine2D", 21: "LAFMergeFeedback2D",
    22: "Conv", 24: "C2f", 25: "SPPF", 26: "C2f_Faster", 27: "SPPF",
    28: "StaticMAA2D", 29: "ZeroInitResidualRefine2D", 30: "ZeroInitResidualRefine2D",
    31: "LAFMergeFeedback2D", 44: "OBB",
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
        args.source, args.target_yaml, args.output,
        SINGLE_TO_RGB_SHARED, SINGLE_TO_IR, EXPECTED_TARGET_LAYERS, 45,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
