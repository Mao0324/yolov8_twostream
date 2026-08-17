#!/usr/bin/env python3
"""Bucket YOLO OBB ground truths by geometry, overlap proxy, and local density.

DroneVehicle has no explicit occlusion flag. ``overlap_proxy`` therefore uses
the maximum polygon IoU with another GT in the same image. ``density`` counts
other GT centers within a normalized image-radius. These are transparent
proxies, not ground-truth occlusion annotations or detector accuracy metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = Path(
    "/media/biiteam/新加卷1/biiteam/MCONG/datasets/DroneVehicle_twostream_3/labels/test"
)
DEFAULT_OUTPUT = ROOT / "analysis_reports/dronevehicle_obb_geometry_buckets"
CLASS_NAMES = ("car", "truck", "bus", "van", "freight_car")


def _bucket(value, bounds, labels):
    for boundary, label in zip(bounds, labels):
        if value < boundary:
            return label
    return labels[-1]


def _polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) * 0.5


def _geometry(points):
    rect = cv2.minAreaRect(points.astype(np.float32))
    box = cv2.boxPoints(rect)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    width, height = float(lengths[0]), float(lengths[1])
    long_edge = edges[int(np.argmax(lengths))]
    angle = math.degrees(math.atan2(float(long_edge[1]), float(long_edge[0]))) % 180.0
    angle = min(angle, 180.0 - angle)  # orientation folded to [0, 90]
    return {
        "area": _polygon_area(points),
        "aspect_ratio": max(width, height) / max(min(width, height), 1e-9),
        "angle_deg": angle,
        "center": points.mean(axis=0),
    }


def _max_pair_iou(polygons):
    count = len(polygons)
    result = np.zeros(count, dtype=np.float32)
    areas = np.asarray([_polygon_area(p) for p in polygons], dtype=np.float32)
    for i in range(count):
        for j in range(i + 1, count):
            inter, _ = cv2.intersectConvexConvex(polygons[i].astype(np.float32), polygons[j].astype(np.float32))
            union = float(areas[i] + areas[j] - inter)
            iou = float(inter) / union if union > 0.0 else 0.0
            result[i] = max(result[i], iou)
            result[j] = max(result[j], iou)
    return result


def _load_label(path):
    objects = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 7 or len(fields) % 2 != 1:
            raise ValueError(
                f"{path}:{line_number}: expected class plus at least three xy polygon points, got {len(fields)} fields"
            )
        class_id = int(fields[0])
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"{path}:{line_number}: unsupported class id {class_id}")
        points = np.asarray([float(value) for value in fields[1:]], dtype=np.float32).reshape(-1, 2)
        objects.append((class_id, points))
    return objects


def analyze(labels_dir, density_radius=0.10):
    counts = Counter()
    total_objects = 0
    files = sorted(Path(labels_dir).glob("*.txt"))
    for path in files:
        objects = _load_label(path)
        if not objects:
            continue
        polygons = [points for _, points in objects]
        geometry = [_geometry(points) for points in polygons]
        centers = np.stack([item["center"] for item in geometry])
        distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        neighbors = ((distances < density_radius) & (distances > 0.0)).sum(axis=1)
        overlaps = _max_pair_iou(polygons)

        for (class_id, _), item, neighbor_count, overlap in zip(objects, geometry, neighbors, overlaps):
            dimensions = {
                "area": _bucket(item["area"], (0.001, 0.005), ("small", "medium", "large")),
                "aspect_ratio": _bucket(item["aspect_ratio"], (2.0, 4.0), ("compact_<2", "elongated_2-4", "very_elongated_>=4")),
                "angle": _bucket(item["angle_deg"], (15.0, 30.0, 60.0), ("0-15", "15-30", "30-60", "60-90")),
                "overlap_proxy": _bucket(float(overlap), (0.01, 0.10), ("isolated_<0.01", "touching_0.01-0.10", "overlap_>=0.10")),
                "density": _bucket(int(neighbor_count), (1, 4), ("sparse_0", "moderate_1-3", "dense_>=4")),
            }
            class_name = CLASS_NAMES[class_id]
            for dimension, bucket in dimensions.items():
                counts[(class_name, dimension, bucket)] += 1
            total_objects += 1
    return files, counts, total_objects


def write_outputs(output_dir, labels_dir, files, counts, total_objects, density_radius):
    output_dir.mkdir(parents=True, exist_ok=True)
    class_totals = Counter()
    for (class_name, dimension, _), count in counts.items():
        if dimension == "area":
            class_totals[class_name] += count
    rows = []
    for (class_name, dimension, bucket), count in sorted(counts.items()):
        rows.append(
            {
                "class": class_name,
                "dimension": dimension,
                "bucket": bucket,
                "count": count,
                "class_share": count / class_totals[class_name],
            }
        )

    csv_path = output_dir / "bucket_counts.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("class", "dimension", "bucket", "count", "class_share"))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "labels_dir": str(Path(labels_dir).resolve()),
        "label_files": len(files),
        "objects": total_objects,
        "classes": list(CLASS_NAMES),
        "thresholds": {
            "area_normalized": [0.001, 0.005],
            "aspect_ratio": [2.0, 4.0],
            "orientation_degrees_folded": [15.0, 30.0, 60.0],
            "overlap_proxy_max_gt_polygon_iou": [0.01, 0.10],
            "density_neighbor_radius_normalized": density_radius,
            "density_neighbor_count": [1, 4],
        },
        "caveat": "overlap_proxy and density are derived GT proxies; DroneVehicle has no explicit occlusion annotation",
        "rows": rows,
    }
    json_path = output_dir / "bucket_counts.json"
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return csv_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--density-radius", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.labels.is_dir():
        raise FileNotFoundError(f"label directory not found: {args.labels}")
    if not 0.0 < args.density_radius <= 1.0:
        raise ValueError("density-radius must be in (0, 1]")
    files, counts, total_objects = analyze(args.labels, args.density_radius)
    csv_path, json_path = write_outputs(
        args.output, args.labels, files, counts, total_objects, args.density_radius
    )
    print(f"analyzed {len(files)} label files and {total_objects} objects")
    print(f"saved {csv_path}")
    print(f"saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
