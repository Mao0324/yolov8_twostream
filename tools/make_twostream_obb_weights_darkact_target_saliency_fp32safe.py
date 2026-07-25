#!/usr/bin/env python3
"""Directly migrate canonical YOLOv8s-OBB weights to FP32Safe target saliency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.darkact_target_saliency_variant_migration import migrate_variant  # noqa: E402

DEFAULT_TARGET_YAML = ROOT / "yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-FP32Safe-v2.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_target_saliency_paperlaf_p345_fp32safe_v2.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    migrate_variant(args.target_yaml, args.output, "StaticMAAContext2DFP32Safe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
