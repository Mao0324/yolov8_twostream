#!/usr/bin/env python3
"""Mechanically update architecture manifests to the seeded run layout."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.migrate_experiment_layout import run_specs


PLANNED_ROOTS = {
    "ASSA-001": "runs/DroneVehicle_OBB/train-labels=official-v1/assanet/mainline/ASSA-001__assafusion-p345",
    "ASSA-002": "runs/DroneVehicle_OBB/train-labels=official-v1/assanet/mainline/ASSA-002__staticdw-p4",
    "ASSA-003": "runs/DroneVehicle_OBB/train-labels=official-v1/assanet/mainline/ASSA-003__partial-channel-p34",
    "BL-002": "runs/DroneVehicle_OBB/train-labels=official-v1/baseline/mainline/BL-002__bottleneck-refine-p345",
    "DA-008": "runs/DroneVehicle_OBB/train-labels=official-v1/darkact/mainline/DA-008__target-saliency",
}


def manifest_roots() -> dict[str, str]:
    roots = dict(PLANNED_ROOTS)
    for spec in run_specs():
        experiment_id = spec.experiment.split("__", 1)[0]
        if experiment_id in roots:
            continue
        target = Path(spec.target)
        roots[experiment_id] = target.parents[1].as_posix()
    return roots


def main() -> int:
    roots = manifest_roots()
    written = 0
    for path in sorted((ROOT / "experiments").glob("*/manifests/*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        experiment_id = str(payload["id"])
        if experiment_id not in roots:
            raise RuntimeError(f"no seeded output root for {experiment_id}")
        output = payload["training"]["output"]
        output["project"] = roots[experiment_id]
        output["name"] = "attempt=01"
        output["layout"] = "seeded-v1"
        payload["legacy"] = {
            "run_dir": None,
            "relocated_runs": "experiments/run_registry.yaml",
        }
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        written += 1
        print(f"UPDATED {path.relative_to(ROOT)} -> {roots[experiment_id]}")
    print(f"OK: {written} manifests updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
