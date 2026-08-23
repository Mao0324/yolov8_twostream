#!/usr/bin/env python3
"""Build the scale-n P2Det V13 checkpoint with an exact baseline-common state."""

from __future__ import annotations

import argparse
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
import yaml
from ultralytics.nn.tasks import OBBModel

from tools.p2det_weight_transfer import migrate_p2det_from_canonical


DEFAULT_SOURCE = ROOT / "pre-pth/yolov8n-obb.pt"
DEFAULT_BASELINE = ROOT / "pre-pth/yolov8n-obb_twostream_baseline.pt"
DEFAULT_TARGET_YAML = (
    ROOT / "yaml/yolov8s-P2Det-IRPrompt-AsymIdentityGDER-P4-NoStaticMAA-PostC2f-v13_scalen.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "pre-pth/yolov8n-obb_twostream_p2det_irprompt_asymidentitygder_p4_no_staticmaa_postc2f_v13_scalen.pt"
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
    17: "P2IRPromptAsymIdentityGDERMergeFeedback2D",
    25: "P2IRPromptLAFMergeFeedbackNoStaticMAA2D",
    38: "OBB",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_scale_n_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"target YAML root must be a mapping: {path}")
    if config.get("scale") != "n":
        raise RuntimeError(f"target YAML must explicitly declare scale: n, got {config.get('scale')!r}")
    if config.get("nc") != 5:
        raise RuntimeError(f"target YAML must declare nc: 5, got {config.get('nc')!r}")
    return config


def _validate_target(model: OBBModel) -> None:
    if len(model.model) != 39:
        raise RuntimeError(f"target model has {len(model.model)} layers; expected 39")
    errors = []
    for index, expected_type in EXPECTED_TARGET_LAYERS.items():
        actual_type = type(model.model[index]).__name__
        if actual_type != expected_type:
            errors.append(f"layer {index}: expected {expected_type}, got {actual_type}")
    first_conv_shape = tuple(model.model[0].conv.weight.shape)
    if first_conv_shape != (16, 3, 3, 3):
        errors.append(f"layer 0 is not scale-n width: {first_conv_shape}")
    if errors:
        raise RuntimeError("target architecture mismatch: " + "; ".join(errors))


def main() -> int:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    baseline = args.baseline.expanduser().resolve()
    target_yaml = args.target_yaml.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for label, path in (("source", source), ("baseline", baseline), ("target YAML", target_yaml)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if output in {source, baseline}:
        raise ValueError("output must not overwrite a source checkpoint")

    torch.manual_seed(MIGRATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MIGRATION_SEED)

    # yaml_model_load() infers "s" from this historical filename and would
    # overwrite its explicit scale: n. Building from the parsed mapping keeps
    # the requested scale without changing repository-wide loader behavior.
    config = _load_scale_n_config(target_yaml)
    model = OBBModel(cfg=config, ch=3, nc=config["nc"], verbose=False).float()
    _validate_target(model)
    report = migrate_p2det_from_canonical(
        model,
        canonical_source=source,
        baseline_reference=baseline,
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

    model.args = {"task": "obb", "model": str(target_yaml)}
    migration = {
        **report,
        "target_yaml": str(target_yaml),
        "target_yaml_sha256": _sha256(target_yaml),
        "baseline_sha256": _sha256(baseline),
        "migration_seed": MIGRATION_SEED,
        "target_scale": "n",
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
        "train_args": {"task": "obb", "model": str(target_yaml)},
        "migration": migration,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    print(f"saved={output}")
    print(f"size_mib={output.stat().st_size / 2**20:.2f}")
    print(f"baseline_tensors={report['baseline_p2det_tensors_verified']}")
    print(f"baseline_parameters={report['matched_parameter_count']}")
    for key in exactness_keys:
        print(f"{key}={report[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
