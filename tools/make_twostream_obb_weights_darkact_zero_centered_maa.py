#!/usr/bin/env python3
"""Migrate YOLOv8s-OBB weights to DA-013 (zero-centered bidirectional MAA)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_twostream_obb_weights_darkact_postc2f import _run_migration  # noqa: E402
from make_twostream_obb_weights_darkact_v2 import (  # noqa: E402
    DEFAULT_SOURCE,
    SINGLE_TO_IR,
    SINGLE_TO_RGB_SHARED,
)


DEFAULT_TARGET_YAML = ROOT / "yaml/yolov8s-DarkAct-ZeroCenteredMAA2D-LAFMerge-P345-R4-v1.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_zerocenteredmaa_laffeedback_p345_v1.pt"

# 除 MAA 类名外，层位置和权重迁移关系与 DA-002 完全相同。
EXPECTED_TARGET_LAYERS = {
    0: "Conv", 4: "Conv", 6: "C2f_Faster",
    8: "ZeroCenteredStaticMAA2D", 9: "C2f", 10: "C2f_Faster", 11: "LAFMergeFeedback2D",
    12: "Conv", 14: "ZeroCenteredStaticMAA2D", 15: "C2f", 16: "C2f_Faster",
    17: "LAFMergeFeedback2D", 18: "Conv",
    20: "ZeroCenteredStaticMAA2D", 21: "C2f", 22: "SPPF", 23: "C2f_Faster",
    24: "SPPF", 25: "LAFMergeFeedback2D", 38: "OBB",
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
        39,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
