#!/usr/bin/env python3
"""Build the embedded-architecture P2Det V16 checkpoint from DarkACT LAF."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO

from tools.dronevehicle_m2dlif import install_trusted_torch_load


SOURCE = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_laffeedback2d_p345_r4_no_staticmaa_v1.pt"
TARGET_YAML = ROOT / "yaml/yolov8s-P2Det-DualReliabilityPrompt-P34-PriorLAF-NoGDER-v16.yaml"
OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_p2det_dualreliability_p34_priorlaf_v16.pt"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_architecture(model):
    expected = {
        10: "P2DualReliabilityPriorLAFMergeFeedback2D",
        15: "P2DualReliabilityPriorLAFMergeFeedback2D",
        22: "LAFMergeFeedback2D",
        35: "OBB",
    }
    if len(model.model) != 36:
        raise RuntimeError(f"V16 target has {len(model.model)} layers; expected 36")
    errors = []
    for index, expected_type in expected.items():
        actual = type(model.model[index]).__name__
        if actual != expected_type:
            errors.append(f"layer {index}: expected {expected_type}, got {actual}")
    if tuple(model.model[0].conv.weight.shape) != (32, 3, 3, 3):
        errors.append(f"target is not scale-s: {tuple(model.model[0].conv.weight.shape)}")
    if errors:
        raise RuntimeError("; ".join(errors))


def main():
    install_trusted_torch_load(torch)
    for label, path in (("DarkACT source", SOURCE), ("V16 YAML", TARGET_YAML)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    torch.manual_seed(16)
    source = YOLO(str(SOURCE), task="obb").model.float().eval()
    target = YOLO(str(TARGET_YAML), task="obb").model.float().eval()
    validate_architecture(target)
    if len(source.model) != len(target.model):
        raise RuntimeError(f"source/target layer counts differ: {len(source.model)} vs {len(target.model)}")

    source_state, target_state = source.state_dict(), target.state_dict()
    copied = []
    with torch.no_grad():
        for key, value in source_state.items():
            if key not in target_state or target_state[key].shape != value.shape:
                raise RuntimeError(f"DarkACT tensor cannot map into V16: {key} {tuple(value.shape)}")
            target_state[key].copy_(value.to(dtype=target_state[key].dtype))
            copied.append(key)
    target.load_state_dict(target_state, strict=True)
    maximum = max(
        float((target.state_dict()[key].float() - source_state[key].float()).abs().max()) for key in copied
    )
    if maximum != 0.0:
        raise RuntimeError(f"DarkACT transfer is not exact: max_abs_diff={maximum}")

    # Prompt-conditioned gate projections are zero, so the complete detector
    # must reproduce its DarkACT carrier before learning begins.
    sample = torch.randn(1, 6, 128, 128)
    with torch.inference_mode():
        source_prediction = source(sample)[0]
        target_prediction = target(sample)[0]
    forward_difference = float((source_prediction.float() - target_prediction.float()).abs().max())
    if forward_difference != 0.0:
        raise RuntimeError(f"V16 neutral forward differs from DarkACT: max_abs_diff={forward_difference}")

    migration = {
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "target_yaml": str(TARGET_YAML),
        "target_yaml_sha256": sha256(TARGET_YAML),
        "copied_tensor_count": len(copied),
        "darkact_v16_max_abs_diff": maximum,
        "neutral_forward_max_abs_diff": forward_difference,
        "target_scale": "s",
    }
    target.args = {"task": "obb", "model": str(TARGET_YAML)}
    target.p2det_migration = migration
    checkpoint = {
        "date": datetime.now().isoformat(),
        "version": ultralytics.__version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(target).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {"task": "obb", "model": str(TARGET_YAML)},
        "migration": migration,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(OUTPUT)
    print(f"saved={OUTPUT}")
    print(f"size_mib={OUTPUT.stat().st_size / 2**20:.2f}")
    print(f"copied_tensor_count={len(copied)}")
    print(f"darkact_v16_max_abs_diff={maximum}")
    print(f"neutral_forward_max_abs_diff={forward_difference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
