#!/usr/bin/env python3
"""Independently migrate single-stream YOLOv8s-OBB weights to post-LAF refine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_twostream_obb_weights_darkact_postc2f import _run_migration  # noqa: E402


DEFAULT_SOURCE = ROOT / "pre-pth/yolov8s-obb.pt"
DEFAULT_TARGET_YAML = ROOT / "yaml/yolov8s-DarkAct-MAA2D-LAFMerge-Refine-P345-R4-v1.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_staticmaa_laffeedback_refine_p345_e025_v1.pt"

# YOLOv8s-OBB 单流层 -> 新模型 RGB/共享层。新增 Refine 26/27/28 独立初始化。
SINGLE_TO_RGB_SHARED = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 9,
    5: 12,
    6: 15,
    7: 18,
    8: 21,
    9: 22,
    12: 31,
    15: 34,
    16: 35,
    18: 37,
    19: 38,
    21: 40,
    22: 41,
}

# 同一单流 backbone 权重独立迁移到 IR 分支；仅复制键名和 shape 兼容的张量。
SINGLE_TO_IR = {
    0: 4,
    1: 5,
    2: 6,
    3: 7,
    4: 10,
    5: 13,
    6: 16,
    7: 19,
    8: 23,
    9: 24,
}

EXPECTED_TARGET_LAYERS = {
    0: "Conv",
    4: "Conv",
    6: "C2f_Faster",
    8: "StaticMAA2D",
    9: "C2f",
    10: "C2f_Faster",
    11: "LAFMergeFeedback2D",
    12: "Conv",
    14: "StaticMAA2D",
    15: "C2f",
    16: "C2f_Faster",
    17: "LAFMergeFeedback2D",
    18: "Conv",
    20: "StaticMAA2D",
    21: "C2f",
    22: "SPPF",
    23: "C2f_Faster",
    24: "SPPF",
    25: "LAFMergeFeedback2D",
    26: "ZeroInitResidualRefine2D",
    27: "ZeroInitResidualRefine2D",
    28: "ZeroInitResidualRefine2D",
    41: "OBB",
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Single-stream YOLOv8s-OBB checkpoint")
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML, help="Post-LAF refine YAML")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output two-stream checkpoint")
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
        42,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
