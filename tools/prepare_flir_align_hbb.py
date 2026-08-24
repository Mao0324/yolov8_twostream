#!/usr/bin/env python3
"""Prepare aligned FLIR COCO annotations for this fork's two-stream HBB loader."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/media/biiteam/新加卷1/biiteam/MCONG/datasets/flir_align")
DEFAULT_OUTPUT = ROOT / "data/flir_align_hbb"
EXPECTED_CATEGORIES = {0: "car", 1: "person", 2: "bicycle"}
# FLIR's *_new.json annotations retain every paired image but intentionally
# remove the extremely sparse dog category used by the original annotations.
SPLITS = {"train": "train_new.json", "test": "test_new.json"}


def _ensure_directory_link(link: Path, target: Path) -> None:
    """Create a directory symlink without replacing unrelated user data."""

    target = target.resolve(strict=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve(strict=False) != target:
            raise RuntimeError(f"existing symlink points elsewhere: {link} -> {link.resolve(strict=False)}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing non-symlink path: {link}")
    link.symlink_to(target, target_is_directory=True)


def _load_annotations(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    categories = {int(item["id"]): str(item["name"]) for item in data.get("categories", [])}
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(f"unexpected categories in {path}: {categories}; expected {EXPECTED_CATEGORIES}")
    return data


def _yolo_line(annotation: dict[str, Any], width: int, height: int) -> str | None:
    if annotation.get("ignore", 0) or annotation.get("iscrowd", 0):
        return None
    category_id = int(annotation["category_id"])
    if category_id not in EXPECTED_CATEGORIES:
        raise ValueError(f"unknown category_id={category_id} in annotation {annotation.get('id')}")
    x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + box_width, 0.0), float(width))
    y2 = min(max(y + box_height, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    center_x = (x1 + x2) / (2.0 * width)
    center_y = (y1 + y2) / (2.0 * height)
    normalized_width = (x2 - x1) / width
    normalized_height = (y2 - y1) / height
    return (
        f"{category_id} {center_x:.8f} {center_y:.8f} "
        f"{normalized_width:.8f} {normalized_height:.8f}"
    )


def _prepare_split(source: Path, output: Path, split: str, annotation_name: str) -> dict[str, int]:
    visible = source / "visible" / split
    thermal = source / "thermal" / split
    annotation_path = source / "coco_annotations" / annotation_name
    for required in (visible, thermal, annotation_path):
        if not required.exists():
            raise FileNotFoundError(f"required FLIR path not found: {required}")

    _ensure_directory_link(output / "images" / split, visible)
    _ensure_directory_link(output / "image" / split, thermal)
    label_dir = output / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)

    data = _load_annotations(annotation_path)
    annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        annotations_by_image[str(annotation["image_id"])].append(annotation)

    box_count = 0
    skipped_count = 0
    duplicate_count = 0
    seen_names: set[str] = set()
    ordered_names: list[str] = []
    for image in data.get("images", []):
        image_id = str(image["id"])
        file_name = Path(str(image["file_name"])).name
        if file_name in seen_names:
            raise ValueError(f"duplicate file_name in {annotation_path}: {file_name}")
        seen_names.add(file_name)
        ordered_names.append(file_name)
        if not (visible / file_name).is_file():
            raise FileNotFoundError(f"visible image listed by JSON is missing: {visible / file_name}")
        if not (thermal / file_name).is_file():
            raise FileNotFoundError(f"thermal pair listed by JSON is missing: {thermal / file_name}")
        width, height = int(image["width"]), int(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image size for {file_name}: {width}x{height}")

        lines = []
        unique_lines: set[str] = set()
        for annotation in annotations_by_image.get(image_id, []):
            line = _yolo_line(annotation, width, height)
            if line is None:
                skipped_count += 1
            elif line in unique_lines:
                duplicate_count += 1
            else:
                unique_lines.add(line)
                lines.append(line)
                box_count += 1
        label_path = label_dir / f"{Path(file_name).stem}.txt"
        temporary_path = label_path.with_suffix(".txt.tmp")
        temporary_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        temporary_path.replace(label_path)

    visible_names = {path.name for path in visible.iterdir() if path.is_file()}
    thermal_names = {path.name for path in thermal.iterdir() if path.is_file()}
    if visible_names != seen_names:
        raise ValueError(
            f"{split} visible/JSON mismatch: JSON={len(seen_names)}, directory={len(visible_names)}"
        )
    if thermal_names != seen_names:
        raise ValueError(
            f"{split} thermal/JSON mismatch: JSON={len(seen_names)}, directory={len(thermal_names)}"
        )

    # Directory paths in a dataset YAML are resolved through symlinks by this
    # fork, which loses the required images/image naming convention. Explicit
    # image-list files preserve the alias paths used by img2label_paths() and
    # BaseDataset.load_image().
    list_stem = "train" if split == "train" else "val"
    for list_path, modality_dir in (
        (output / list_stem, "images"),
        (output / f"{list_stem}_ir", "image"),
    ):
        lines = [str(output / modality_dir / split / file_name) for file_name in sorted(ordered_names)]
        temporary_path = list_path.with_suffix(list_path.suffix + ".tmp")
        temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary_path.replace(list_path)
    return {
        "images": len(seen_names),
        "boxes": box_count,
        "skipped": skipped_count,
        "duplicates_removed": duplicate_count,
    }


def prepare_dataset(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Build the paired three-class layout and deterministic YOLO HBB labels."""

    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_stats = {
        split: _prepare_split(source, output, split, annotation_name)
        for split, annotation_name in SPLITS.items()
    }
    # Label content changed from four to three classes. Remove Ultralytics'
    # generated caches so a subsequent --skip-prepare run cannot reuse the old
    # dog-inclusive label scan.
    for cache_path in (output / "labels").glob("*.cache"):
        cache_path.unlink()
    manifest = {
        "format": "YOLO HBB xywh normalized",
        "source": str(source),
        "categories": EXPECTED_CATEGORIES,
        "annotation_files": SPLITS,
        "excluded_categories": ["dog"],
        "splits": split_stats,
    }
    manifest_path = output / "manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest


def validate_prepared_dataset(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Reject stale prepared data created with the legacy four-class protocol."""

    output = output.expanduser().resolve()
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepared FLIR manifest not found: {manifest_path}")
    manifest = _load_manifest(manifest_path)
    categories = {int(key): str(value) for key, value in manifest.get("categories", {}).items()}
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(
            f"stale FLIR prepared data in {output}: categories={categories}; "
            f"expected {EXPECTED_CATEGORIES}. Re-run tools/prepare_flir_align_hbb.py."
        )
    if manifest.get("annotation_files") != SPLITS:
        raise ValueError(
            f"stale FLIR annotation protocol in {manifest_path}: "
            f"{manifest.get('annotation_files')}; expected {SPLITS}"
        )
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"manifest root must be an object: {path}")
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="FLIR aligned dataset root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="prepared two-stream dataset root")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = prepare_dataset(args.source, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
