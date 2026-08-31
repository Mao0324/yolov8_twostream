#!/usr/bin/env python3
"""Build an embedded P2Det V18 checkpoint from the DarkACT carrier."""

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
from ultralytics.models.yolo.obb.p2det_object_reliability_train import P2TrueObjectReliabilityOBBModel

from tools.dronevehicle_m2dlif import install_trusted_torch_load


SOURCE = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_laffeedback2d_p345_r4_no_staticmaa_v1.pt"
TARGET_YAML = ROOT / "yaml/yolov8s-P2Det-TrueObjectReliabilityPrompt-P34-SignPreservingLAF-NoGDER-v18.yaml"
OUTPUT = ROOT / "pre-pth/yolov8s-obb_twostream_p2det_trueobjectreliability_p34_signpreservinglaf_v18.pt"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_architecture(model):
    expected = {
        10: "P2DualTrueObjectReliabilityLAFMergeFeedback2D",
        15: "P2DualTrueObjectReliabilityLAFMergeFeedback2D",
        22: "LAFMergeFeedback2D",
        35: "OBB",
    }
    if len(model.model) != 36:
        raise RuntimeError(f"V18 target has {len(model.model)} layers; expected 36")
    for index, expected_type in expected.items():
        actual = type(model.model[index]).__name__
        if actual != expected_type:
            raise RuntimeError(f"layer {index}: expected {expected_type}, got {actual}")
    for index in (10, 15):
        module = model.model[index]
        if module.rgb_prompt_head.body[-1].bias is not None or module.ir_prompt_head.body[-1].bias is not None:
            raise RuntimeError(f"layer {index}: V18 Prompt output must be bias-free")
        if hasattr(module.merge, "raw_prompt_strength") or float(module.merge.prompt_strength) != 1.0:
            raise RuntimeError(f"layer {index}: V18 Prompt strength must be fixed at one")
        if not 0 <= module.merge.rho < 1:
            raise RuntimeError(f"layer {index}: invalid sign-preserving rho")


def main():
    install_trusted_torch_load(torch)
    for label, path in (("DarkACT source", SOURCE), ("V18 YAML", TARGET_YAML)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    torch.manual_seed(18)
    source = YOLO(str(SOURCE), task="obb").model.float().eval()
    target = P2TrueObjectReliabilityOBBModel(str(TARGET_YAML), ch=3, nc=None, verbose=True).float().eval()
    validate_architecture(target)
    source_state, target_state = source.state_dict(), target.state_dict()
    copied = []
    with torch.no_grad():
        for key, value in source_state.items():
            if key not in target_state or target_state[key].shape != value.shape:
                raise RuntimeError(f"DarkACT tensor cannot map into V18: {key} {tuple(value.shape)}")
            target_state[key].copy_(value.to(dtype=target_state[key].dtype))
            copied.append(key)
    target.load_state_dict(target_state, strict=True)
    transfer_difference = max(
        float((target.state_dict()[key].float() - source_state[key].float()).abs().max()) for key in copied
    )
    if transfer_difference != 0.0:
        raise RuntimeError(f"DarkACT transfer is not exact: max_abs_diff={transfer_difference}")

    sample = torch.randn(1, 6, 128, 128)
    with torch.inference_mode():
        source_prediction = source(sample)[0].float()
        target_prediction = target(sample)[0].float()
    if not torch.isfinite(target_prediction).all():
        raise RuntimeError("V18 initialization produced non-finite predictions")
    carrier_change = float((source_prediction - target_prediction).abs().max())

    migration = {
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "target_yaml": str(TARGET_YAML),
        "target_yaml_sha256": sha256(TARGET_YAML),
        "copied_tensor_count": len(copied),
        "darkact_v18_max_abs_diff": transfer_difference,
        "expected_prompt_primary_forward_change": carrier_change,
        "target_scale": "s",
        "true_object_reliability_version": 18,
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
    print(f"darkact_v18_max_abs_diff={transfer_difference}")
    print(f"expected_prompt_primary_forward_change={carrier_change}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
