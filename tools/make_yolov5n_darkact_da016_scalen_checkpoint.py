#!/usr/bin/env python3
"""Build scale-n YOLOv5 DA-016 with a bit-exact baseline common state."""

from __future__ import annotations

import argparse
import hashlib
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import OBBModel

from tools.make_twostream_yolov5n_obb_weights import SINGLE_TO_IR, SINGLE_TO_RGB_SHARED


DEFAULT_SOURCE = ROOT / "pre-pth/yolov5nu.pt"
DEFAULT_BASELINE = ROOT / "pre-pth/yolov5nu_twostream_baseline_v5n_obb.pt"
DEFAULT_TARGET_YAML = (
    ROOT / "yaml/yolov5s-DarkAct-SemanticDisagreementLAF-P34-R4-NoStaticMAA-v1_scalen.yaml"
)
DEFAULT_OUTPUT = (
    ROOT / "pre-pth/yolov5nu_twostream_darkact_da016_semantic_disagreement_laf_p34_scalen.pt"
)
MIGRATION_SEED = 0

# All stateful layers from baseline-v5n.yaml have an exact target counterpart.
# Its RIFusion/ADD layers have no state and are replaced by DarkAct fusion.
BASELINE_TO_TARGET = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    9: 8,
    11: 9,
    10: 11,
    12: 12,
    14: 13,
    16: 14,
    15: 16,
    17: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    26: 23,
    29: 26,
    30: 27,
    33: 30,
    34: 31,
    36: 33,
    37: 34,
    39: 36,
    40: 37,
}

EXPECTED_BASELINE_LAYERS = {
    0: "Conv",
    2: "C3",
    4: "Conv",
    6: "C3",
    8: "RIFusion",
    9: "C3",
    11: "C3",
    13: "RIFusion",
    14: "C3",
    16: "C3",
    18: "RIFusion",
    19: "C3",
    20: "SPPF",
    21: "C3",
    22: "SPPF",
    23: "ADD",
    24: "ADD",
    25: "ADD",
    29: "C3",
    33: "C3",
    36: "C3",
    39: "C3",
    40: "OBB",
}

EXPECTED_TARGET_LAYERS = {
    0: "Conv",
    2: "C3",
    4: "Conv",
    6: "C3",
    8: "C3",
    9: "C3",
    10: "SemanticDisagreementLAFMergeFeedback2D",
    13: "C3",
    14: "C3",
    15: "SemanticDisagreementLAFMergeFeedback2D",
    18: "C3",
    19: "SPPF",
    20: "C3",
    21: "SPPF",
    22: "LAFMergeFeedback2D",
    26: "C3",
    30: "C3",
    33: "C3",
    36: "C3",
    37: "OBB",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_layers(model, expected: Mapping[int, str], expected_count: int, label: str) -> None:
    if len(model.model) != expected_count:
        raise RuntimeError(f"{label} has {len(model.model)} layers; expected {expected_count}")
    errors = []
    for index, expected_type in expected.items():
        actual_type = type(model.model[index]).__name__
        if actual_type != expected_type:
            errors.append(f"layer {index}: expected {expected_type}, got {actual_type}")
    if errors:
        raise RuntimeError(f"{label} architecture mismatch: " + "; ".join(errors))


def _replace_layer(key: str, source_layer: int, target_layer: int) -> str:
    return key.replace(f"model.{source_layer}.", f"model.{target_layer}.", 1)


def _compatible_pairs(source, target, layer_map):
    pairs = []
    for source_key, source_value in source.items():
        source_layer = int(source_key.split(".")[1])
        if source_layer not in layer_map:
            continue
        target_key = _replace_layer(source_key, source_layer, layer_map[source_layer])
        if target_key in target and source_value.shape == target[target_key].shape:
            pairs.append((source_key, target_key))
    return pairs


def _max_abs_diff(left, right, pairs) -> float:
    maximum = 0.0
    for left_key, right_key in pairs:
        left_value = left[left_key].detach().cpu().float()
        right_value = right[right_key].detach().cpu().float()
        if not torch.equal(left_value, right_value):
            maximum = max(maximum, float((left_value - right_value).abs().max()))
    return maximum


def _copy_complete_baseline(baseline_state, target_model):
    target_state = target_model.state_dict()
    pairs = []
    for baseline_key, baseline_value in baseline_state.items():
        baseline_layer = int(baseline_key.split(".")[1])
        if baseline_layer not in BASELINE_TO_TARGET:
            raise RuntimeError(f"baseline tensor has no target mapping: {baseline_key}")
        target_key = _replace_layer(baseline_key, baseline_layer, BASELINE_TO_TARGET[baseline_layer])
        if target_key not in target_state:
            raise RuntimeError(f"target tensor is missing: {baseline_key} -> {target_key}")
        if baseline_value.shape != target_state[target_key].shape:
            raise RuntimeError(
                f"shape mismatch: {baseline_key} {tuple(baseline_value.shape)} -> "
                f"{target_key} {tuple(target_state[target_key].shape)}"
            )
        target_state[target_key].copy_(baseline_value.to(dtype=target_state[target_key].dtype))
        pairs.append((baseline_key, target_key))
    if len({target_key for _, target_key in pairs}) != len(pairs):
        raise RuntimeError("multiple baseline tensors map to one target tensor")
    target_model.load_state_dict(target_state, strict=True)
    return pairs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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

    source_model = YOLO(str(source), task="detect").model.float()
    baseline_checkpoint = torch.load(baseline, map_location="cpu", weights_only=False)
    baseline_model = (baseline_checkpoint.get("ema") or baseline_checkpoint["model"]).float()
    _validate_layers(baseline_model, EXPECTED_BASELINE_LAYERS, 41, "baseline")
    source_state = source_model.state_dict()
    baseline_state = baseline_model.state_dict()

    source_baseline_pairs = _compatible_pairs(source_state, baseline_state, SINGLE_TO_RGB_SHARED)
    source_baseline_pairs.extend(_compatible_pairs(source_state, baseline_state, SINGLE_TO_IR))
    source_baseline_maxdiff = _max_abs_diff(source_state, baseline_state, source_baseline_pairs)
    if len(source_baseline_pairs) != 589 or source_baseline_maxdiff != 0.0:
        raise RuntimeError(
            "source does not exactly match the baseline provenance: "
            f"verified={len(source_baseline_pairs)}/589, max_abs_diff={source_baseline_maxdiff}"
        )

    config = yaml.safe_load(target_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("scale") != "n" or config.get("nc") != 5:
        raise RuntimeError("target YAML must be a mapping with scale:n and nc:5")
    torch.manual_seed(MIGRATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MIGRATION_SEED)
    # Parse the mapping directly so the historical yolov5s filename cannot
    # override its explicit scale:n declaration.
    target_model = OBBModel(cfg=config, ch=3, nc=config["nc"], verbose=False).float()
    _validate_layers(target_model, EXPECTED_TARGET_LAYERS, 38, "target")
    if tuple(target_model.model[0].conv.weight.shape) != (16, 3, 6, 6):
        raise RuntimeError("target did not build at YOLOv5n width")

    baseline_target_pairs = _copy_complete_baseline(baseline_state, target_model)
    baseline_target_maxdiff = _max_abs_diff(
        baseline_state, target_model.state_dict(), baseline_target_pairs
    )
    if len(baseline_target_pairs) != len(baseline_state) or baseline_target_maxdiff != 0.0:
        raise RuntimeError(
            "target is not bit-exact to the complete baseline state: "
            f"verified={len(baseline_target_pairs)}/{len(baseline_state)}, "
            f"max_abs_diff={baseline_target_maxdiff}"
        )

    migration = {
        "source": str(source),
        "source_sha256": _sha256(source),
        "baseline_reference": str(baseline),
        "baseline_sha256": _sha256(baseline),
        "target_yaml": str(target_yaml),
        "target_yaml_sha256": _sha256(target_yaml),
        "migration_seed": MIGRATION_SEED,
        "target_scale": "n",
        "source_baseline_tensors_verified": len(source_baseline_pairs),
        "source_baseline_max_abs_diff": source_baseline_maxdiff,
        "baseline_target_tensors_verified": len(baseline_target_pairs),
        "baseline_target_max_abs_diff": baseline_target_maxdiff,
        "baseline_parameter_count": sum(parameter.numel() for parameter in baseline_model.parameters()),
        "target_parameter_count": sum(parameter.numel() for parameter in target_model.parameters()),
    }
    target_model.args = {"task": "obb", "model": str(target_yaml)}
    target_model.darkact_migration = migration
    checkpoint = {
        "date": datetime.now().isoformat(),
        "version": ultralytics.__version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(target_model).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {"task": "obb", "model": str(target_yaml)},
        "migration": migration,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    saved_checkpoint = torch.load(temporary, map_location="cpu", weights_only=False)
    saved_model = deepcopy(saved_checkpoint.get("ema") or saved_checkpoint["model"]).float()
    saved_pairs = _compatible_pairs(baseline_state, saved_model.state_dict(), BASELINE_TO_TARGET)
    serialized_maxdiff = _max_abs_diff(baseline_state, saved_model.state_dict(), saved_pairs)
    if len(saved_pairs) != len(baseline_state) or serialized_maxdiff != 0.0:
        raise RuntimeError(
            "serialized target differs from baseline: "
            f"verified={len(saved_pairs)}/{len(baseline_state)}, max_abs_diff={serialized_maxdiff}"
        )
    saved_checkpoint["migration"]["serialized_baseline_target_max_abs_diff"] = serialized_maxdiff
    saved_checkpoint["model"].darkact_migration = saved_checkpoint["migration"]
    torch.save(saved_checkpoint, temporary)
    temporary.replace(output)

    print(f"saved={output}")
    print(f"size_mib={output.stat().st_size / 2**20:.2f}")
    print(f"source_baseline_tensors={len(source_baseline_pairs)}")
    print(f"source_baseline_max_abs_diff={source_baseline_maxdiff}")
    print(f"baseline_target_tensors={len(saved_pairs)}")
    print(f"baseline_target_max_abs_diff={serialized_maxdiff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
