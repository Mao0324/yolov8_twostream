#!/usr/bin/env python3
"""Build the P2Det V15 scale-s checkpoint with exact baseline-common weights."""

from __future__ import annotations

import hashlib
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

from tools.dronevehicle_m2dlif import install_trusted_torch_load
from tools.p2det_weight_transfer import migrate_p2det_from_canonical


SOURCE = ROOT / "pre-pth/yolov8s-obb.pt"
BASELINE = ROOT / "pre-pth/yolov8s-obb_twostream_baseline.pt"
TARGET_YAML = ROOT / "yaml/yolov8s-P2Det-IRPrompt-AttExpert-P4-NoStaticMAA-PostC2f-v15.yaml"
OUTPUT = (
    ROOT
    / "pre-pth/yolov8s-obb_twostream_p2det_irprompt_attexpert_p4_no_staticmaa_postc2f_v15.pt"
)
MIGRATION_SEED = 0

EXPECTED_TARGET_LAYERS = {
    0: "Conv",
    4: "Conv",
    6: "C2f_Faster",
    8: "C2f",
    9: "C2f_Faster",
    10: "RIFusion",
    11: "P2IRPromptLAFMergeFeedbackNoStaticMAA2D",
    17: "P2IRPromptAttExpertMergeFeedback2D",
    25: "P2IRPromptLAFMergeFeedbackNoStaticMAA2D",
    38: "OBB",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_target(model) -> None:
    if len(model.model) != 39:
        raise RuntimeError(f"target model has {len(model.model)} layers; expected 39")
    errors = []
    for index, expected_type in EXPECTED_TARGET_LAYERS.items():
        actual_type = type(model.model[index]).__name__
        if actual_type != expected_type:
            errors.append(f"layer {index}: expected {expected_type}, got {actual_type}")
    first_conv_shape = tuple(model.model[0].conv.weight.shape)
    if first_conv_shape != (32, 3, 3, 3):
        errors.append(f"layer 0 is not scale-s width: {first_conv_shape}")
    if errors:
        raise RuntimeError("target architecture mismatch: " + "; ".join(errors))


def main() -> int:
    install_trusted_torch_load(torch)
    for label, path in (
        ("source", SOURCE),
        ("baseline", BASELINE),
        ("target YAML", TARGET_YAML),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    torch.manual_seed(MIGRATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MIGRATION_SEED)

    model = YOLO(str(TARGET_YAML), task="obb").model.float()
    validate_target(model)
    report = migrate_p2det_from_canonical(
        model,
        canonical_source=SOURCE,
        baseline_reference=BASELINE,
    )
    exactness_keys = (
        "canonical_baseline_max_abs_diff",
        "reconstructed_baseline_max_abs_diff",
        "baseline_p2det_max_abs_diff",
    )
    failures = {key: report[key] for key in exactness_keys if report[key] != 0.0}
    if failures:
        raise RuntimeError(f"migration is not bit-exact: {failures}")
    if report["unexpected_source_tensor_count"] != 0:
        raise RuntimeError(
            "not every baseline tensor was mapped into P2Det: "
            f"unexpected={report['unexpected_source_tensor_count']}"
        )

    model.args = {"task": "obb", "model": str(TARGET_YAML)}
    migration = {
        **report,
        "target_yaml": str(TARGET_YAML),
        "target_yaml_sha256": sha256(TARGET_YAML),
        "baseline_sha256": sha256(BASELINE),
        "migration_seed": MIGRATION_SEED,
        "target_scale": "s",
    }
    model.p2det_migration = migration
    checkpoint = {
        "date": datetime.now().isoformat(),
        "version": ultralytics.__version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(model).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {"task": "obb", "model": str(TARGET_YAML)},
        "migration": migration,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(OUTPUT)
    print(f"saved={OUTPUT}")
    print(f"size_mib={OUTPUT.stat().st_size / 2**20:.2f}")
    print(f"baseline_tensors={report['baseline_p2det_tensors_verified']}")
    print(f"baseline_parameters={report['matched_parameter_count']}")
    for key in exactness_keys:
        print(f"{key}={report[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
