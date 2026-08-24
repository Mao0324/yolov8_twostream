#!/usr/bin/env python3
"""Audit RGB/IR pairing and estimate residual geometric displacement."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


ROOT = Path("/media/biiteam/新加卷1/biiteam/MCONG/datasets/DroneVehicle_twostream_3")
OUT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
SAMPLE_SIZES = {"train": 200, "val": 50, "test": 150}
CLASS_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "van", 4: "freight_car"}


def quantile(values: Iterable[float], q: float) -> float | None:
    items = np.asarray(list(values), dtype=np.float64)
    if not len(items):
        return None
    return float(np.quantile(items, q))


def rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_index(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    }


def label_index(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in directory.glob("*.txt")}


def sample_evenly(stems: list[str], size: int) -> list[str]:
    if len(stems) <= size:
        return stems
    indices = np.linspace(0, len(stems) - 1, num=size, dtype=int)
    return [stems[index] for index in indices]


def gradient_feature(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    image = cv2.GaussianBlur(image, (0, 0), 1.2)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.GaussianBlur(magnitude, (0, 0), 1.0)
    mean, std = cv2.meanStdDev(magnitude)
    return (magnitude - float(mean[0, 0])) / max(float(std[0, 0]), 1e-6)


def estimate_translation(rgb_path: Path, ir_path: Path) -> dict[str, Any]:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None:
        return {"read_ok": False, "ecc_ok": False}
    if rgb.shape != ir.shape:
        return {
            "read_ok": True,
            "ecc_ok": False,
            "rgb_height": rgb.shape[0],
            "rgb_width": rgb.shape[1],
            "ir_height": ir.shape[0],
            "ir_width": ir.shape[1],
        }

    rgb_feature = gradient_feature(rgb)
    ir_feature = gradient_feature(ir)
    height, width = rgb.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    phase_shift, phase_response = cv2.phaseCorrelate(rgb_feature, ir_feature, window)
    phase_dx, phase_dy = map(float, phase_shift)
    if math.hypot(phase_dx, phase_dy) > 20:
        phase_dx = phase_dy = 0.0

    warp = np.eye(2, 3, dtype=np.float32)
    warp[0, 2] = phase_dx
    warp[1, 2] = phase_dy
    try:
        ecc, warp = cv2.findTransformECC(
            rgb_feature,
            ir_feature,
            warp,
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
            None,
            5,
        )
        dx, dy = float(warp[0, 2]), float(warp[1, 2])
        return {
            "read_ok": True,
            "ecc_ok": True,
            "rgb_height": height,
            "rgb_width": width,
            "ir_height": height,
            "ir_width": width,
            "phase_dx_px": phase_dx,
            "phase_dy_px": phase_dy,
            "phase_response": float(phase_response),
            "ecc_dx_px": dx,
            "ecc_dy_px": dy,
            "ecc_shift_px": math.hypot(dx, dy),
            "ecc_correlation": float(ecc),
        }
    except cv2.error:
        return {
            "read_ok": True,
            "ecc_ok": False,
            "rgb_height": height,
            "rgb_width": width,
            "ir_height": height,
            "ir_width": width,
            "phase_dx_px": phase_dx,
            "phase_dy_px": phase_dy,
            "phase_response": float(phase_response),
        }


def estimate_affine(rgb_path: Path, ir_path: Path) -> dict[str, Any]:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None or rgb.shape != ir.shape:
        return {"affine_ok": False}
    rgb_feature = gradient_feature(rgb)
    ir_feature = gradient_feature(ir)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        ecc, warp = cv2.findTransformECC(
            rgb_feature,
            ir_feature,
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 160, 1e-6),
            None,
            5,
        )
    except cv2.error:
        return {"affine_ok": False}

    matrix = warp[:, :2].astype(np.float64)
    translation = warp[:, 2].astype(np.float64)
    height, width = rgb.shape
    points = np.array(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1], [(width - 1) / 2, (height - 1) / 2]],
        dtype=np.float64,
    )
    transformed = points @ matrix.T + translation
    displacement = np.linalg.norm(transformed - points, axis=1)
    scale_x = float(np.linalg.norm(matrix[:, 0]))
    scale_y = float(np.linalg.norm(matrix[:, 1]))
    rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    return {
        "affine_ok": True,
        "affine_correlation": float(ecc),
        "affine_center_shift_px": float(displacement[-1]),
        "affine_max_corner_shift_px": float(displacement[:4].max()),
        "affine_scale_x": scale_x,
        "affine_scale_y": scale_y,
        "affine_rotation_deg": rotation,
    }


def count_labels(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def profile_boxes(label_paths: Iterable[Path]) -> tuple[list[dict[str, Any]], int, Counter[int]]:
    measurements: list[dict[str, Any]] = []
    invalid = 0
    vertex_counts: Counter[int] = Counter()
    for path in label_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                coordinate_count = len(parts) - 1
                if len(parts) < 7 or coordinate_count % 2:
                    invalid += 1
                    continue
                try:
                    class_id = int(parts[0])
                    coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
                except (ValueError, OverflowError):
                    invalid += 1
                    continue
                if not np.isfinite(coords).all():
                    invalid += 1
                    continue
                coords[:, 0] *= 640
                coords[:, 1] *= 512
                (_, _), (width, height), _ = cv2.minAreaRect(coords)
                if width <= 0 or height <= 0:
                    invalid += 1
                    continue
                vertex_counts[len(coords)] += 1
                measurements.append(
                    {
                        "class": CLASS_NAMES.get(class_id, str(class_id)),
                        "short_side_px": min(float(width), float(height)),
                        "long_side_px": max(float(width), float(height)),
                    }
                )
    return measurements, invalid, vertex_counts


def load_alignment_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in source.items():
                if value == "":
                    continue
                if value in {"True", "False"}:
                    row[key] = value == "True"
                    continue
                if key in {"split", "stem", "sequence_bucket"}:
                    row[key] = value
                    continue
                try:
                    number = float(value)
                    row[key] = int(number) if number.is_integer() else number
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def summarize_boxes(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["all"] = measurements
    for row in measurements:
        groups[row["class"]].append(row)
    rows = []
    for name in ("all", "car", "truck", "bus", "van", "freight_car"):
        group = groups.get(name, [])
        rows.append(
            {
                "class": name,
                "instances": len(group),
                "short_side_p10_px": rounded(quantile((row["short_side_px"] for row in group), 0.10), 2),
                "short_side_p50_px": rounded(quantile((row["short_side_px"] for row in group), 0.50), 2),
                "short_side_p90_px": rounded(quantile((row["short_side_px"] for row in group), 0.90), 2),
                "long_side_p50_px": rounded(quantile((row["long_side_px"] for row in group), 0.50), 2),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-alignment",
        action="store_true",
        help="Reuse alignment_samples.csv and recompute pairing/label summaries only.",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pairing_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    all_label_paths: list[Path] = []

    for split in SPLITS:
        print(f"[pairing] {split}", flush=True)
        rgb = image_index(ROOT / "images" / split)
        ir = image_index(ROOT / "image" / split)
        labels = label_index(ROOT / "labels" / split)
        rgb_stems, ir_stems, label_stems = set(rgb), set(ir), set(labels)
        common = sorted(rgb_stems & ir_stems & label_stems)
        all_label_paths.extend(labels[stem] for stem in common)

        dimension_matches = 0
        unreadable = 0
        dimension_counts: dict[str, int] = defaultdict(int)
        for stem in common:
            try:
                with Image.open(rgb[stem]) as rgb_image, Image.open(ir[stem]) as ir_image:
                    rgb_size, ir_size = rgb_image.size, ir_image.size
                dimension_counts[f"rgb_{rgb_size[0]}x{rgb_size[1]}"] += 1
                dimension_counts[f"ir_{ir_size[0]}x{ir_size[1]}"] += 1
                if rgb_size == ir_size:
                    dimension_matches += 1
            except OSError:
                unreadable += 1

        pairing_rows.append(
            {
                "split": split,
                "rgb_files": len(rgb),
                "ir_files": len(ir),
                "label_files": len(labels),
                "exact_triplets": len(common),
                "triplet_coverage_pct": round(100 * len(common) / max(len(rgb_stems | ir_stems | label_stems), 1), 4),
                "rgb_without_ir": len(rgb_stems - ir_stems),
                "ir_without_rgb": len(ir_stems - rgb_stems),
                "images_without_label": len((rgb_stems & ir_stems) - label_stems),
                "labels_without_pair": len(label_stems - (rgb_stems & ir_stems)),
                "dimension_matches": dimension_matches,
                "dimension_match_pct": round(100 * dimension_matches / max(len(common), 1), 4),
                "unreadable_pairs": unreadable,
                "dimension_profile": "; ".join(f"{key}:{value}" for key, value in sorted(dimension_counts.items())),
            }
        )

        if not args.reuse_alignment:
            sampled = sample_evenly(common, SAMPLE_SIZES[split])
            for index, stem in enumerate(sampled):
                row = {
                    "split": split,
                    "stem": stem,
                    "sequence_bucket": f"{split}:{int(stem) // 1000:02d}k" if stem.isdigit() else f"{split}:other",
                    "objects": count_labels(labels[stem]),
                    **estimate_translation(rgb[stem], ir[stem]),
                }
                if index % 10 == 0:
                    row.update(estimate_affine(rgb[stem], ir[stem]))
                alignment_rows.append(row)
            print(f"[alignment] {split}: {len(sampled)} sampled", flush=True)

    if args.reuse_alignment:
        alignment_path = OUT / "alignment_samples.csv"
        if not alignment_path.exists():
            parser.error(f"cannot reuse missing file: {alignment_path}")
        alignment_rows = load_alignment_rows(alignment_path)
        print(f"[alignment] reused {len(alignment_rows)} rows", flush=True)

    box_measurements, invalid_labels, vertex_counts = profile_boxes(all_label_paths)
    box_summary = summarize_boxes(box_measurements)

    trusted = [
        row
        for row in alignment_rows
        if row.get("ecc_ok")
        and row.get("ecc_correlation", 0) >= 0.50
        and row.get("ecc_shift_px", 999) <= 10
    ]
    affine_trusted = [
        row
        for row in alignment_rows
        if row.get("affine_ok")
        and row.get("affine_correlation", 0) >= 0.50
        and 0.98 <= row.get("affine_scale_x", 0) <= 1.02
        and 0.98 <= row.get("affine_scale_y", 0) <= 1.02
        and abs(row.get("affine_rotation_deg", 99)) <= 1.0
    ]

    bucket_rows = []
    for bucket in sorted({row["sequence_bucket"] for row in trusted}):
        group = [row for row in trusted if row["sequence_bucket"] == bucket]
        bucket_rows.append(
            {
                "sequence_bucket": bucket,
                "samples": len(group),
                "median_shift_px": rounded(quantile((row["ecc_shift_px"] for row in group), 0.5), 3),
                "p90_shift_px": rounded(quantile((row["ecc_shift_px"] for row in group), 0.9), 3),
                "median_dx_px": rounded(quantile((row["ecc_dx_px"] for row in group), 0.5), 3),
                "median_dy_px": rounded(quantile((row["ecc_dy_px"] for row in group), 0.5), 3),
                "median_correlation": rounded(quantile((row["ecc_correlation"] for row in group), 0.5), 3),
            }
        )

    split_rows = []
    for split in SPLITS:
        sampled_group = [row for row in alignment_rows if row["split"] == split]
        group = [row for row in trusted if row["split"] == split]
        split_rows.append(
            {
                "split": split,
                "samples": len(sampled_group),
                "trusted_samples": len(group),
                "trusted_rate_pct": rounded(100 * len(group) / max(len(sampled_group), 1), 2),
                "median_shift_px": rounded(quantile((row["ecc_shift_px"] for row in group), 0.5), 3),
                "p90_shift_px": rounded(quantile((row["ecc_shift_px"] for row in group), 0.9), 3),
                "within_2px_pct": rounded(100 * sum(row["ecc_shift_px"] <= 2 for row in group) / max(len(group), 1), 2),
            }
        )

    shift_bins = (
        ("≤1 px", 0, 1),
        ("1–2 px", 1, 2),
        ("2–3 px", 2, 3),
        ("3–5 px", 3, 5),
        ("5–10 px", 5, 10),
    )
    shift_distribution = []
    for label, lower, upper in shift_bins:
        count = sum(lower < row["ecc_shift_px"] <= upper for row in trusted)
        if lower == 0:
            count = sum(row["ecc_shift_px"] <= upper for row in trusted)
        shift_distribution.append(
            {
                "range": label,
                "samples": count,
                "share_pct": rounded(100 * count / max(len(trusted), 1), 2),
            }
        )

    median_short_side = box_summary[0]["short_side_p50_px"]
    summary = {
        "dataset_root": str(ROOT),
        "pairing": pairing_rows,
        "total_exact_triplets": sum(row["exact_triplets"] for row in pairing_rows),
        "all_pairs_complete": all(row["triplet_coverage_pct"] == 100.0 for row in pairing_rows),
        "all_dimensions_match": all(row["dimension_match_pct"] == 100.0 for row in pairing_rows),
        "alignment_method": "Gradient-magnitude phase correlation followed by translation-only ECC; deterministic evenly spaced sample.",
        "alignment_sample_count": len(alignment_rows),
        "trusted_alignment_count": len(trusted),
        "trusted_alignment_rate_pct": rounded(100 * len(trusted) / max(len(alignment_rows), 1), 2),
        "median_dx_px": rounded(quantile((row["ecc_dx_px"] for row in trusted), 0.5), 3),
        "median_dy_px": rounded(quantile((row["ecc_dy_px"] for row in trusted), 0.5), 3),
        "median_shift_px": rounded(quantile((row["ecc_shift_px"] for row in trusted), 0.5), 3),
        "p75_shift_px": rounded(quantile((row["ecc_shift_px"] for row in trusted), 0.75), 3),
        "p90_shift_px": rounded(quantile((row["ecc_shift_px"] for row in trusted), 0.90), 3),
        "p95_shift_px": rounded(quantile((row["ecc_shift_px"] for row in trusted), 0.95), 3),
        "share_within_1px_pct": rounded(100 * sum(row["ecc_shift_px"] <= 1 for row in trusted) / max(len(trusted), 1), 2),
        "share_within_2px_pct": rounded(100 * sum(row["ecc_shift_px"] <= 2 for row in trusted) / max(len(trusted), 1), 2),
        "share_within_3px_pct": rounded(100 * sum(row["ecc_shift_px"] <= 3 for row in trusted) / max(len(trusted), 1), 2),
        "median_ecc_correlation": rounded(quantile((row["ecc_correlation"] for row in trusted), 0.5), 3),
        "affine_sample_count": sum("affine_ok" in row for row in alignment_rows),
        "trusted_affine_count": len(affine_trusted),
        "affine_median_center_shift_px": rounded(quantile((row["affine_center_shift_px"] for row in affine_trusted), 0.5), 3),
        "affine_p90_max_corner_shift_px": rounded(quantile((row["affine_max_corner_shift_px"] for row in affine_trusted), 0.9), 3),
        "affine_median_rotation_deg": rounded(quantile((row["affine_rotation_deg"] for row in affine_trusted), 0.5), 4),
        "affine_median_scale_x": rounded(quantile((row["affine_scale_x"] for row in affine_trusted), 0.5), 5),
        "affine_median_scale_y": rounded(quantile((row["affine_scale_y"] for row in affine_trusted), 0.5), 5),
        "box_instances": len(box_measurements),
        "invalid_label_rows": invalid_labels,
        "label_vertex_count_distribution": {str(key): value for key, value in sorted(vertex_counts.items())},
        "box_size_summary": box_summary,
        "split_alignment_summary": split_rows,
        "trusted_shift_distribution": shift_distribution,
        "median_shift_as_pct_of_median_short_side": rounded(100 * quantile((row["ecc_shift_px"] for row in trusted), 0.5) / median_short_side, 2),
        "limitations": [
            "Cross-modal gradient registration estimates scene-level geometry; it is not an independent annotation of each vehicle center.",
            "Thermal blooming, shadows, vegetation, and moving objects can reduce ECC correlation without implying a camera calibration failure.",
            "The dataset supplies one shared OBB label file per RGB/IR pair, so label equality is an assumption of the prepared dataset rather than proof of pixel-perfect object alignment.",
        ],
    }

    write_csv(OUT / "pairing_summary.csv", pairing_rows)
    write_csv(OUT / "alignment_samples.csv", alignment_rows)
    write_csv(OUT / "sequence_alignment_summary.csv", bucket_rows)
    write_csv(OUT / "split_alignment_summary.csv", split_rows)
    write_csv(OUT / "shift_distribution.csv", shift_distribution)
    write_csv(OUT / "box_size_summary.csv", box_summary)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
