#!/usr/bin/env python3
"""Evaluate one registered experiment's best checkpoint on the test split."""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tools.experiment_registry import (
    RegistryError,
    artifact_snapshot,
    get_manifest,
    load_registry,
    repo_path,
    resolve_training,
    validate_registry,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--device", required=True, help="GPU ID(s), e.g. 0 or 0,1")
    parser.add_argument("--run-dir", type=Path, help="override the latest registered run")
    parser.add_argument("--checkpoint", type=Path, help="override RUN_DIR/weights/best.pt")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _patch_trusted_torch_load():
    original = torch.load
    if getattr(original, "_experiment_trusted_patch", False):
        return
    supports_weights_only = "weights_only" in inspect.signature(original).parameters

    def trusted_load(*args, **kwargs):
        if supports_weights_only:
            kwargs.setdefault("weights_only", False)
        else:
            kwargs.pop("weights_only", None)
        return original(*args, **kwargs)

    trusted_load._experiment_trusted_patch = True
    torch.load = trusted_load


def main():
    args = _parse_args()
    manifests = load_registry()
    errors, warnings = validate_registry(manifests)
    if errors:
        raise RegistryError("registry validation failed:\n- " + "\n- ".join(errors))
    for warning in warnings:
        print("WARNING: {}".format(warning), file=sys.stderr)

    manifest = get_manifest(args.experiment_id, manifests)
    resolved = resolve_training(manifest)
    snapshot = artifact_snapshot(manifest)
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else repo_path(snapshot.get("run_dir"))
    if run_dir is None or not run_dir.is_dir():
        raise RegistryError("no completed/active run found; pass --run-dir explicitly")
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else run_dir / "weights/best.pt"
    if not checkpoint.is_file():
        raise RegistryError("test checkpoint not found: {}".format(checkpoint))

    output_dir = run_dir / "test_result"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_log = output_dir / "test.txt"
    _patch_trusted_torch_load()
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    handler = logging.FileHandler(test_log, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    try:
        metrics = YOLO(str(checkpoint), task="obb").val(
            data=resolved["args"]["data"],
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=str(run_dir),
            name="test_result",
            exist_ok=True,
            task="obb",
        )
    finally:
        LOGGER.removeHandler(handler)
        handler.close()

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    with test_log.open("a", encoding="utf-8") as handle:
        # Registry parser consumes the last two numeric fields on an "all" row.
        handle.write("all 0 0 0 0 {:.6f} {:.6f}\n".format(map50, map50_95))
    print("{} test mAP50={:.6f}, mAP50-95={:.6f}".format(args.experiment_id, map50, map50_95))
    print("saved {}".format(test_log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
