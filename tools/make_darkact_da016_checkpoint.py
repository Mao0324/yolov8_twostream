#!/usr/bin/env python3
"""Migrate yolov8s-obb.pt into the DA-016 semantic-disagreement architecture."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_twostream_obb_weights_darkact_no_staticmaa import (  # noqa: E402
    SINGLE_TO_IR,
    SINGLE_TO_RGB_SHARED,
)
from make_twostream_obb_weights_darkact_postc2f import _run_migration  # noqa: E402


DEFAULT_SOURCE = ROOT / "pre-pth/yolov8s-obb.pt"
DEFAULT_YAML = ROOT / "yaml/yolov8s-DarkAct-SemanticDisagreementLAF-P34-R4-NoStaticMAA-v1.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_da016_semantic_disagreement_laf_p34.pt"
MIGRATION_SEED = 0

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
    35: "OBB",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.manual_seed(MIGRATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MIGRATION_SEED)
    _run_migration(
        args.source,
        args.target_yaml,
        args.output,
        SINGLE_TO_RGB_SHARED,
        SINGLE_TO_IR,
        EXPECTED_TARGET_LAYERS,
        36,
    )
    source_sha256 = hashlib.sha256(args.source.expanduser().resolve().read_bytes()).hexdigest()
    checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
    migration = checkpoint["migration"]
    migration["source_sha256"] = source_sha256
    migration["migration_seed"] = MIGRATION_SEED
    migration["copied_tensor_count"] = migration["rgb_shared_copied"] + migration["ir_copied"]
    temporary_path = args.output.with_suffix(args.output.suffix + ".metadata.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
