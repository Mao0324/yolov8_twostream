#!/usr/bin/env python3
"""Migrate yolov8s-obb.pt into the DA-015 two-stream architecture."""

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


DEFAULT_SOURCE = ROOT / "pre-pth/yolov8s-obb.pt"
DEFAULT_YAML = ROOT / "yaml/yolov8s-DarkAct-PostC2f-DisagreementLAF-P34-R4-v1.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_da015_disagreement_laf_p34.pt"

EXPECTED_TARGET_LAYERS = {
    0: "Conv",
    4: "Conv",
    6: "C2f_Faster",
    8: "C2f",
    9: "C2f_Faster",
    10: "StaticMAA2D",
    11: "DisagreementLAFMergeFeedback2D",
    12: "Conv",
    14: "C2f",
    15: "C2f_Faster",
    16: "StaticMAA2D",
    17: "DisagreementLAFMergeFeedback2D",
    18: "Conv",
    20: "C2f",
    21: "SPPF",
    22: "C2f_Faster",
    23: "SPPF",
    24: "StaticMAA2D",
    25: "LAFMergeFeedback2D",
    38: "OBB",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_YAML)
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
