#!/usr/bin/env python3
"""Reproduce the cross-branch baseline-ceiling diagnosis from committed artifacts."""

from __future__ import annotations

import csv
import io
import json
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BRANCHES = (
    "main",
    "ASSAFusion",
    "CRFormer",
    "DarkACT",
    "exp/cfgpnet-lite",
    "exp/protohgfnet-lite",
)
CLASSES = ("car", "truck", "bus", "van", "freight_car")
CONFIG_KEYS = (
    "epochs",
    "batch",
    "imgsz",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "cos_lr",
    "close_mosaic",
    "seed",
    "deterministic",
    "amp",
)
BASELINE_PATH = "runs_baseline/train2/test_result/test.txt"


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, errors="replace"
    )


def git_show(branch: str, path: str) -> str | None:
    try:
        return git("show", f"{branch}:{path}")
    except subprocess.CalledProcessError:
        return None


def parse_test_table(text: str) -> dict[str, Any]:
    pattern = re.compile(
        r"^\s*(all|car|truck|bus|van|freight_car)\s+"
        r"(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        re.MULTILINE,
    )
    rows: dict[str, dict[str, Any]] = {}
    for match in pattern.finditer(text):
        label, images, instances, precision, recall, map50, map50_95 = match.groups()
        rows[label] = {
            "images": int(images),
            "instances": int(instances),
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(map50),
            "map50_95": float(map50_95),
        }
    if "all" not in rows:
        raise ValueError("No final all-class metrics table found")

    metadata: dict[str, str] = {}
    for key in ("weights", "data", "split", "imgsz", "batch", "device"):
        matches = re.findall(rf"^{key}:\s*(.+)$", text, re.MULTILINE)
        if matches:
            metadata[key] = matches[-1].strip()
    return {"rows": rows, "metadata": metadata}


def run_dir_for(test_path: str) -> str:
    return str(PurePosixPath(test_path).parent.parent)


def parse_args(branch: str, test_path: str) -> dict[str, Any]:
    text = git_show(branch, f"{run_dir_for(test_path)}/args.yaml")
    if not text:
        return {}
    return yaml.safe_load(text) or {}


def parse_curve(branch: str, test_path: str) -> dict[str, Any]:
    text = git_show(branch, f"{run_dir_for(test_path)}/results.csv")
    if not text:
        return {}
    rows = []
    for raw in csv.DictReader(io.StringIO(text), skipinitialspace=True):
        try:
            row = {key.strip(): float(value.strip()) for key, value in raw.items()}
        except (TypeError, ValueError, AttributeError):
            continue
        row["fitness"] = (
            0.1 * row["metrics/mAP50(B)"]
            + 0.9 * row["metrics/mAP50-95(B)"]
        )
        rows.append(row)
    if not rows:
        return {}
    best = max(rows, key=lambda row: row["fitness"])
    max50 = max(rows, key=lambda row: row["metrics/mAP50(B)"])
    max50_95 = max(rows, key=lambda row: row["metrics/mAP50-95(B)"])
    last = rows[-1]
    return {
        "epochs_completed": len(rows),
        "best_fitness_epoch": int(best["epoch"]),
        "best_val_map50": best["metrics/mAP50(B)"],
        "best_val_map50_95": best["metrics/mAP50-95(B)"],
        "max_val_map50": max50["metrics/mAP50(B)"],
        "max_val_map50_epoch": int(max50["epoch"]),
        "max_val_map50_95": max50_95["metrics/mAP50-95(B)"],
        "max_val_map50_95_epoch": int(max50_95["epoch"]),
        "last_val_map50": last["metrics/mAP50(B)"],
        "last_val_map50_95": last["metrics/mAP50-95(B)"],
    }


def family_for(path: str) -> str:
    if path.startswith("runs_baseline/"):
        return "Baseline"
    name = PurePosixPath(run_dir_for(path)).name
    if "BottleneckRefine" in name:
        return "BottleneckRefine"
    if name.startswith("ASSA"):
        return "ASSAFusion"
    if name.startswith("CRFormer"):
        return "CRFormer"
    if name.startswith("DarkAct"):
        return "DarkACT"
    if "CFGP" in name:
        return "CFGPNet"
    if "ProtoHGF" in name:
        return "ProtoHGFNet"
    return "Other"


def discover() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_blobs: set[str] = set()
    for branch in BRANCHES:
        for path in git("ls-tree", "-r", "--name-only", branch).splitlines():
            if not re.search(r"test_result[^/]*/test\.txt$", path):
                continue
            blob = git("rev-parse", f"{branch}:{path}").strip()
            if blob in seen_blobs:
                continue
            seen_blobs.add(blob)
            text = git_show(branch, path)
            if not text:
                continue
            try:
                parsed = parse_test_table(text)
            except ValueError:
                continue
            args = parse_args(branch, path)
            curve = parse_curve(branch, path)
            all_metrics = parsed["rows"]["all"]
            record: dict[str, Any] = {
                "branch": branch,
                "path": path,
                "blob": blob,
                "run_name": PurePosixPath(run_dir_for(path)).name,
                "family": family_for(path),
                "weights": parsed["metadata"].get("weights", ""),
                "test_data": parsed["metadata"].get("data", ""),
                "test_split": parsed["metadata"].get("split", ""),
                "test_imgsz": parsed["metadata"].get("imgsz", ""),
                "test_batch": parsed["metadata"].get("batch", ""),
                **all_metrics,
                "train_model": str(args.get("model", "")),
                "pretrained_flag": args.get("pretrained"),
                "checkpoint_direct": str(args.get("model", "")).endswith(".pt"),
                **{f"cfg_{key}": args.get(key) for key in CONFIG_KEYS},
                **curve,
            }
            for class_name in CLASSES:
                class_metrics = parsed["rows"].get(class_name, {})
                record[f"{class_name}_map50"] = class_metrics.get("map50")
                record[f"{class_name}_map50_95"] = class_metrics.get("map50_95")
            records.append(record)
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: float, digits: int = 6) -> float:
    return round(value, digits)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = discover()
    comparable = [
        row
        for row in records
        if row["map50"] > 0.83
        and row["checkpoint_direct"]
        and row["pretrained_flag"] is True
    ]
    comparable.sort(key=lambda row: (-row["map50"], -row["map50_95"], row["path"]))
    baseline = next(row for row in comparable if row["path"] == BASELINE_PATH)
    variants = [row for row in comparable if row["path"] != BASELINE_PATH]

    for row in comparable:
        row["delta_map50_vs_baseline"] = row["map50"] - baseline["map50"]
        row["delta_map50_95_vs_baseline"] = row["map50_95"] - baseline["map50_95"]

    best = comparable[0]
    class_deltas = []
    for class_name in CLASSES:
        class_deltas.append(
            {
                "class": class_name,
                "baseline_map50": baseline[f"{class_name}_map50"],
                "best_map50": best[f"{class_name}_map50"],
                "delta_map50": best[f"{class_name}_map50"]
                - baseline[f"{class_name}_map50"],
                "baseline_map50_95": baseline[f"{class_name}_map50_95"],
                "best_map50_95": best[f"{class_name}_map50_95"],
                "delta_map50_95": best[f"{class_name}_map50_95"]
                - baseline[f"{class_name}_map50_95"],
            }
        )

    signatures = Counter(
        tuple(row.get(f"cfg_{key}") for key in CONFIG_KEYS) for row in comparable
    )
    test_metric_tuples = Counter(
        (row["images"], row["instances"], row["test_split"], row["test_imgsz"])
        for row in comparable
    )
    branches_without_new_results = []
    darkact_paths = {
        row["path"] for row in records if row["branch"] == "DarkACT"
    }
    for branch in ("exp/cfgpnet-lite", "exp/protohgfnet-lite"):
        own_paths = {row["path"] for row in records if row["branch"] == branch}
        if not (own_paths - darkact_paths):
            branches_without_new_results.append(branch)

    summary = {
        "source_branches": list(BRANCHES),
        "unique_committed_test_artifacts": len(records),
        "comparable_threshold": "mAP50 > 0.83, args.pretrained=true, args.model is .pt",
        "comparable_runs": len(comparable),
        "variant_runs": len(variants),
        "branches_without_additional_committed_test_results": branches_without_new_results,
        "test_population_signatures": [
            {
                "images": key[0],
                "instances": key[1],
                "split": key[2],
                "imgsz": key[3],
                "count": count,
            }
            for key, count in test_metric_tuples.items()
        ],
        "training_config_signature_count": len(signatures),
        "training_config_keys": list(CONFIG_KEYS),
        "baseline": {
            key: baseline.get(key)
            for key in (
                "branch",
                "path",
                "map50",
                "map50_95",
                "precision",
                "recall",
                "best_fitness_epoch",
                "best_val_map50",
                "best_val_map50_95",
                "last_val_map50",
                "last_val_map50_95",
            )
        },
        "best_observed": {
            "branch": best["branch"],
            "path": best["path"],
            "family": best["family"],
            "map50": best["map50"],
            "map50_95": best["map50_95"],
            "delta_map50_vs_baseline": rounded(
                best["delta_map50_vs_baseline"]
            ),
            "delta_map50_95_vs_baseline": rounded(
                best["delta_map50_95_vs_baseline"]
            ),
            "best_fitness_epoch": best.get("best_fitness_epoch"),
            "best_val_map50": best.get("best_val_map50"),
            "best_val_map50_95": best.get("best_val_map50_95"),
        },
        "variant_distribution": {
            "map50_min": min(row["map50"] for row in variants),
            "map50_mean": rounded(statistics.mean(row["map50"] for row in variants)),
            "map50_median": rounded(statistics.median(row["map50"] for row in variants)),
            "map50_max": max(row["map50"] for row in variants),
            "mean_delta_map50_vs_baseline": rounded(
                statistics.mean(row["delta_map50_vs_baseline"] for row in variants)
            ),
            "map50_95_min": min(row["map50_95"] for row in variants),
            "map50_95_mean": rounded(
                statistics.mean(row["map50_95"] for row in variants)
            ),
            "map50_95_median": rounded(
                statistics.median(row["map50_95"] for row in variants)
            ),
            "map50_95_max": max(row["map50_95"] for row in variants),
            "mean_delta_map50_95_vs_baseline": rounded(
                statistics.mean(
                    row["delta_map50_95_vs_baseline"] for row in variants
                )
            ),
        },
        "theoretical_macro_map50_headroom": rounded(1.0 - baseline["map50"]),
        "class_deltas_for_best_observed": class_deltas,
        "validation_limits": {
            "unique_seeds": sorted(
                {row["cfg_seed"] for row in comparable if row["cfg_seed"] is not None}
            ),
            "per_image_predictions_committed": False,
            "paired_bootstrap_possible_from_committed_artifacts": False,
            "test_set_used_for_many_model_comparisons": True,
        },
    }

    write_csv(OUT / "all_unique_test_results.csv", records)
    write_csv(OUT / "comparable_pretrained_results.csv", comparable)
    write_csv(OUT / "best_model_class_deltas.csv", class_deltas)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
