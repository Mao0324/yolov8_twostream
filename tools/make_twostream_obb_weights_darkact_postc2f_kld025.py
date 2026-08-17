#!/usr/bin/env python3
"""Migrate YOLOv8s-OBB weights to the DA-004 KLD lambda=0.25 ablation."""

from pathlib import Path

from make_twostream_obb_weights_darkact_postc2f import (
    EXPECTED_TARGET_LAYERS,
    SINGLE_TO_IR,
    SINGLE_TO_RGB_SHARED,
    _run_migration,
)
from make_twostream_obb_weights_darkact_v2 import DEFAULT_SOURCE


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_YAML = ROOT / "yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4-PostC2f-KLD025-v1.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_postc2f_kld025_v1.pt"


def main():
    _run_migration(
        DEFAULT_SOURCE,
        DEFAULT_TARGET_YAML,
        DEFAULT_OUTPUT,
        SINGLE_TO_RGB_SHARED,
        SINGLE_TO_IR,
        EXPECTED_TARGET_LAYERS,
        39,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
