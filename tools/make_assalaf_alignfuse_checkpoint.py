#!/usr/bin/env python3
"""Migrate the DA-012 checkpoint into ASSA-LAF Align->Fuse->Refine v1.

All learned DarkAct backbone, LAF, neck, refine, and OBB parameters are copied.
The new P3/P4 ASSA projections remain zero initialized, making the target
network functionally identical to the source at the migration boundary.
"""
from __future__ import annotations

import argparse
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

DEFAULT_SOURCE = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_laf_refine_p34_no_staticmaa_v1.pt"
DEFAULT_YAML = ROOT / "yaml/yolov8s-ASSALAF-AlignFuseFeedback-Refine-P34-v1.yaml"
DEFAULT_OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_assalaf_alignfuse_refine_p34_v1.pt"


def _target_key(source_key: str) -> str:
    for layer in (10, 15):
        prefix = f"model.{layer}."
        if source_key.startswith(prefix):
            return prefix + "fuse." + source_key[len(prefix):]
    return source_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = YOLO(str(args.source.resolve()), task="obb").model.float()
    target = YOLO(str(args.target_yaml.resolve()), task="obb").model.float()
    source_state, target_state = source.state_dict(), target.state_dict()
    copied = []
    for source_key, value in source_state.items():
        target_key = _target_key(source_key)
        if target_key in target_state and target_state[target_key].shape == value.shape:
            target_state[target_key].copy_(value)
            copied.append((source_key, target_key))

    expected_source = set(source_state)
    copied_source = {key for key, _ in copied}
    missing = sorted(expected_source - copied_source)
    if missing:
        raise RuntimeError(f"failed to migrate {len(missing)} source tensors; first: {missing[:5]}")
    target.load_state_dict(target_state, strict=True)

    for layer in (10, 15):
        for projection in target.model[layer].align.project_out:
            if torch.count_nonzero(projection.weight).item() != 0:
                raise RuntimeError(f"layer {layer} ASSA projection is not zero initialized")
    target.args = {"task": "obb", "model": str(args.target_yaml.resolve())}

    checkpoint = {
        "date": datetime.now().isoformat(), "version": ultralytics.__version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com", "epoch": -1,
        "best_fitness": None, "model": deepcopy(target).half(), "ema": None,
        "updates": None, "optimizer": None,
        "train_args": {"task": "obb", "model": str(args.target_yaml.resolve())},
        "migration": {
            "source": str(args.source.resolve()), "target_yaml": str(args.target_yaml.resolve()),
            "copied_tensors": len(copied), "new_assa_layers": [10, 15],
            "functional_start": "DA-012 exact; ASSA output projections are zero",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    print(f"saved {args.output} ({args.output.stat().st_size / 2**20:.2f} MiB), copied {len(copied)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
