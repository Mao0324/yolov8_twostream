"""Direct, deterministic migration from the canonical YOLOv8s-OBB checkpoint."""

from __future__ import annotations

import hashlib
import inspect
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

import ultralytics
from ultralytics import YOLO
from tools.make_twostream_obb_weights_darkact_postc2f import (
    SINGLE_TO_IR,
    SINGLE_TO_RGB_SHARED,
)
from tools.make_twostream_obb_weights_darkact_v2 import (
    EXPECTED_SOURCE_LAYERS,
    _assert_copied_values,
    _copy_mapped_weights,
    _validate_architecture,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SOURCE = (ROOT / "pre-pth/yolov8s-obb.pt").resolve()
MIGRATION_SEED = 0

EXPECTED_TARGET_BASE = {
    0: "Conv",
    4: "Conv",
    6: "C2f_Faster",
    8: "C2f",
    9: "C2f_Faster",
    11: "TargetSaliencyPaperLAFMergeFeedback2D",
    12: "Conv",
    14: "C2f",
    15: "C2f_Faster",
    17: "TargetSaliencyPaperLAFMergeFeedback2D",
    18: "Conv",
    20: "C2f",
    21: "SPPF",
    22: "C2f_Faster",
    23: "SPPF",
    25: "TargetSaliencyPaperLAFMergeFeedback2D",
    38: "OBB",
}


def _patch_trusted_torch_load():
    original = torch.load
    if getattr(original, "_target_variant_trusted_patch", False):
        return
    supports_weights_only = "weights_only" in inspect.signature(original).parameters

    def trusted_load(*args, **kwargs):
        if supports_weights_only:
            kwargs.setdefault("weights_only", False)
        else:
            kwargs.pop("weights_only", None)
        return original(*args, **kwargs)

    trusted_load._target_variant_trusted_patch = True
    torch.load = trusted_load


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migrate_variant(
    target_yaml: Path,
    output_path: Path,
    target_context_type: str,
    expected_temperature_tensors: int = 0,
):
    """Build one variant directly from the canonical source using a fixed seed."""
    source_path = CANONICAL_SOURCE
    target_yaml = target_yaml.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"canonical source checkpoint not found: {source_path}")
    if not target_yaml.is_file():
        raise FileNotFoundError(f"target YAML not found: {target_yaml}")
    if output_path == source_path:
        raise ValueError("output path must be different from the canonical source")

    _patch_trusted_torch_load()
    print(f"[INFO] canonical source: {source_path}")
    source_model = YOLO(str(source_path), task="obb").model.float()
    _validate_architecture(source_model, EXPECTED_SOURCE_LAYERS, "source", 23)

    # All target-only modules must start from the same deterministic random
    # state across architecture variants. Source checkpoint loading happens
    # before this boundary and therefore cannot perturb the target seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MIGRATION_SEED)
        target_model = YOLO(str(target_yaml), task="obb").model.float()

    expected_target = {
        **EXPECTED_TARGET_BASE,
        10: target_context_type,
        16: target_context_type,
        24: target_context_type,
    }
    _validate_architecture(target_model, expected_target, "target", 39)

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    temperature_keys = sorted(key for key in target_state if key.endswith(".temperature"))
    if len(temperature_keys) != expected_temperature_tensors:
        raise RuntimeError(
            f"expected {expected_temperature_tensors} temperature tensors, got {temperature_keys}"
        )
    for key in temperature_keys:
        if target_state[key].numel() != 1 or not torch.allclose(
            target_state[key].float(), torch.tensor(0.2)
        ):
            raise RuntimeError(f"unexpected temperature initialization for {key}: {target_state[key]}")

    rgb_copied, rgb_missing, rgb_mismatch = _copy_mapped_weights(
        source_state, target_state, SINGLE_TO_RGB_SHARED, "single->rgb/shared"
    )
    ir_copied, ir_missing, ir_mismatch = _copy_mapped_weights(
        source_state, target_state, SINGLE_TO_IR, "single->ir"
    )
    if rgb_missing:
        raise RuntimeError(f"RGB/shared migration encountered {rgb_missing} missing target keys")
    if not rgb_copied or not ir_copied:
        raise RuntimeError("migration copied no weights into one or more target branches")

    target_model.load_state_dict(target_state, strict=True)
    loaded_state = target_model.state_dict()
    _assert_copied_values(source_state, loaded_state, rgb_copied)
    _assert_copied_values(source_state, loaded_state, ir_copied)
    target_model.args = {"task": "obb", "model": str(target_yaml)}

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
        "migration": {
            "source": str(source_path),
            "source_sha256": _sha256(source_path),
            "target_yaml": str(target_yaml),
            "method": "direct deterministic single-stream to two-stream mapping",
            "migration_seed": MIGRATION_SEED,
            "rgb_shared_copied": len(rgb_copied),
            "rgb_shared_shape_mismatch": rgb_mismatch,
            "ir_copied": len(ir_copied),
            "ir_missing": ir_missing,
            "ir_shape_mismatch": ir_mismatch,
            "initialized_temperature_tensors": temperature_keys,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(checkpoint, str(temporary_path))
    temporary_path.replace(output_path)
    print(f"[DONE] source: {source_path}")
    print(f"[DONE] seed: {MIGRATION_SEED}")
    print(f"[DONE] RGB/shared copied: {len(rgb_copied)}")
    print(f"[DONE] IR copied: {len(ir_copied)}")
    print(f"[DONE] initialized temperatures: {len(temperature_keys)}")
    print(f"[DONE] saved: {output_path}")
    return output_path
