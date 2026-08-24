#!/usr/bin/env python3
"""Build the scale-s FLIR DarkAct two-stream HBB checkpoint from yolov8s.pt."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from tools.make_twostream_hbb_weights_flir import migrate


DEFAULT_SOURCE = ROOT / "pre-pth/yolov8s.pt"
DEFAULT_TARGET_YAML = (
    ROOT / "fliryaml/yolov8s-DarkAct-SemanticDisagreementLAF-P34-R4-NoStaticMAA-FLIR-HBB_scales.yaml"
)
DEFAULT_OUTPUT = (
    ROOT / "pre-pth/yolov8s-hbb_twostream_darkact_semantic_disagreement_laf_p34_flir_3class_scales.pt"
)
MIGRATION_SEED = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_target_yaml(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"target YAML root must be a mapping: {path}")
    if config.get("scale") != "s":
        raise RuntimeError(f"target YAML must declare scale: s, got {config.get('scale')!r}")
    if config.get("nc") != 3:
        raise RuntimeError(f"target YAML must declare nc: 3, got {config.get('nc')!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = args.source.expanduser().resolve(strict=True)
    target_yaml = args.target_yaml.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output == source:
        raise ValueError("output must not overwrite the source checkpoint")

    _validate_target_yaml(target_yaml)
    torch.manual_seed(MIGRATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MIGRATION_SEED)

    stats = migrate(source, target_yaml, output)
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint["model"]
    first_conv_shape = tuple(model.model[0].conv.weight.shape)
    if first_conv_shape != (32, 3, 3, 3):
        raise RuntimeError(
            f"migrated target did not build at scale s: first Conv shape={first_conv_shape}"
        )

    migration = checkpoint.setdefault("migration", {})
    migration.update(
        {
            "source_sha256": _sha256(source),
            "target_yaml_sha256": _sha256(target_yaml),
            "migration_seed": MIGRATION_SEED,
            "target_scale": "s",
            "target_classes": 3,
            "copied_tensor_count": stats["rgb_shared_copied"] + stats["thermal_copied"],
        }
    )
    model.darkact_migration = migration

    temporary = output.with_suffix(output.suffix + ".metadata.tmp")
    torch.save(checkpoint, temporary)
    reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
    reloaded_model = reloaded.get("ema") or reloaded["model"]
    if tuple(reloaded_model.model[0].conv.weight.shape) != first_conv_shape:
        raise RuntimeError("serialized scale-s checkpoint failed structural verification")
    temporary.replace(output)

    print(f"[DONE] source_sha256: {migration['source_sha256']}")
    print(f"[DONE] target_yaml_sha256: {migration['target_yaml_sha256']}")
    print(f"[DONE] copied_tensor_count: {migration['copied_tensor_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
