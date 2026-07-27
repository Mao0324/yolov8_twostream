#!/usr/bin/env python3
"""Migrate YOLOv8s-OBB weights to DA-012 (LAF-only + P3/P4 fused Refine)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_twostream_obb_weights_darkact_postc2f import DEFAULT_SOURCE, _run_migration  # noqa: E402


DEFAULT_TARGET_YAML = ROOT / "yaml/yolov8s-DarkAct-LAFMergeFeedback2D-Refine-P34-R4-NoStaticMAA-v1.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_laf_refine_p34_no_staticmaa_v1.pt"

# DA-007 的 backbone 索引不变；两个 Refine 插入 head 起点后，neck/OBB 索引整体后移 2。
SINGLE_TO_RGB_SHARED = {
    0: 0, 1: 1, 2: 2, 3: 3,
    4: 8, 5: 11, 6: 13, 7: 16, 8: 18, 9: 19,
    12: 27, 15: 30, 16: 31, 18: 33, 19: 34, 21: 36, 22: 37,
}
SINGLE_TO_IR = {
    0: 4, 1: 5, 2: 6, 3: 7,
    4: 9, 5: 12, 6: 14, 7: 17, 8: 20, 9: 21,
}
EXPECTED_TARGET_LAYERS = {
    0: "Conv", 4: "Conv", 6: "C2f_Faster",
    8: "C2f", 9: "C2f_Faster", 10: "LAFMergeFeedback2D",
    11: "Conv", 13: "C2f", 14: "C2f_Faster", 15: "LAFMergeFeedback2D",
    16: "Conv", 18: "C2f", 19: "SPPF", 20: "C2f_Faster", 21: "SPPF",
    22: "LAFMergeFeedback2D",
    23: "ZeroInitResidualRefine2D", 24: "ZeroInitResidualRefine2D",
    37: "OBB",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _run_migration(
        args.source,
        args.target_yaml,
        args.output,
        SINGLE_TO_RGB_SHARED,
        SINGLE_TO_IR,
        EXPECTED_TARGET_LAYERS,
        38,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
