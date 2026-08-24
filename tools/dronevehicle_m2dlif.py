"""Temporary DroneVehicle train/val adapter for the M2D-LIF annotations."""

from __future__ import annotations

import inspect
import json
from pathlib import Path


M2DLIF_LABEL_ROOT = Path(
    "/media/biiteam/新加卷1/biiteam/MCONG/datasets/"
    "M2D-LIFlabels/DroneVehicle_train_val_labels/labels"
)
SOURCE_DATA_ROOT = Path(
    "/media/biiteam/新加卷1/biiteam/MCONG/datasets/DroneVehicle_twostream_3"
)

# M2D-LIF: 0 car, 1 truck, 2 freight_car, 3 bus, 4 van
# This repository: 0 car, 1 truck, 2 bus, 3 van, 4 freight_car
M2DLIF_TO_PROJECT_CLASS = {0: 0, 1: 1, 2: 4, 3: 2, 4: 3}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def files_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    """Index regular files by stem and reject ambiguous duplicate names."""

    if not directory.is_dir():
        raise FileNotFoundError(f"directory not found: {directory}")
    indexed = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in indexed:
            raise ValueError(f"duplicate stem in {directory}: {indexed[path.stem].name}, {path.name}")
        indexed[path.stem] = path
    return indexed


def prepare_split(
    stage: Path,
    split: str,
    *,
    label_root: Path = M2DLIF_LABEL_ROOT,
    source_data_root: Path = SOURCE_DATA_ROOT,
) -> tuple[Path, Path, int, int]:
    """Create one temporary paired split and remap the M2D-LIF class IDs."""

    rgb_dir = (source_data_root / "images" / split).resolve()
    ir_dir = (source_data_root / "image" / split).resolve()
    label_dir = (label_root / split).resolve()

    rgb_files = files_by_stem(rgb_dir, IMAGE_SUFFIXES)
    ir_files = files_by_stem(ir_dir, IMAGE_SUFFIXES)
    label_files = files_by_stem(label_dir, {".txt"})
    if not rgb_files:
        raise ValueError(f"no RGB images found for split={split}: {rgb_dir}")

    rgb_stems = set(rgb_files)
    problems = []
    for kind, stems in (("IR", set(ir_files)), ("label", set(label_files))):
        missing = sorted(rgb_stems - stems)
        extra = sorted(stems - rgb_stems)
        if missing:
            problems.append(f"{kind} missing {len(missing)} (first: {missing[:5]})")
        if extra:
            problems.append(f"{kind} extra {len(extra)} (first: {extra[:5]})")
    if problems:
        raise ValueError(f"split={split} does not match: " + "; ".join(problems))

    filename_mismatches = [
        stem for stem, rgb_path in rgb_files.items() if ir_files[stem].name != rgb_path.name
    ]
    if filename_mismatches:
        raise ValueError(
            f"split={split} RGB/IR filenames differ for {len(filename_mismatches)} pairs "
            f"(first: {filename_mismatches[:5]})"
        )

    (stage / "images" / split).symlink_to(rgb_dir, target_is_directory=True)
    (stage / "image" / split).symlink_to(ir_dir, target_is_directory=True)
    temporary_labels = stage / "labels" / split
    temporary_labels.mkdir(parents=True)

    box_count = 0
    image_names = [rgb_files[stem].name for stem in sorted(rgb_files)]
    for image_name in image_names:
        source = label_files[Path(image_name).stem]
        remapped_lines = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 9:
                raise ValueError(
                    f"expected OBB class+8 coordinates at {source}:{line_number}, "
                    f"got {len(fields)} fields"
                )
            try:
                source_class = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"non-numeric label at {source}:{line_number}") from exc
            if source_class not in M2DLIF_TO_PROJECT_CLASS:
                raise ValueError(f"unknown M2D-LIF class {source_class} at {source}:{line_number}")
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                raise ValueError(f"non-normalized coordinate at {source}:{line_number}")
            fields[0] = str(M2DLIF_TO_PROJECT_CLASS[source_class])
            remapped_lines.append(" ".join(fields))
            box_count += 1
        (temporary_labels / source.name).write_text(
            "\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8"
        )

    rgb_manifest = stage / split
    ir_manifest = stage / f"{split}_ir"
    rgb_manifest.write_text(
        "".join(f"{stage / 'images' / split / name}\n" for name in image_names), encoding="utf-8"
    )
    ir_manifest.write_text(
        "".join(f"{stage / 'image' / split / name}\n" for name in image_names), encoding="utf-8"
    )
    return rgb_manifest, ir_manifest, len(image_names), box_count


def prepare_temporary_dataset(
    stage: Path,
    *,
    label_root: Path = M2DLIF_LABEL_ROOT,
    source_data_root: Path = SOURCE_DATA_ROOT,
) -> Path:
    """Build a temporary paired-data YAML without modifying source labels or images."""

    stage = Path(stage)
    (stage / "images").mkdir()
    (stage / "image").mkdir()
    (stage / "labels").mkdir()

    manifests = {}
    for split in ("train", "val"):
        rgb_manifest, ir_manifest, image_count, box_count = prepare_split(
            stage,
            split,
            label_root=Path(label_root),
            source_data_root=Path(source_data_root),
        )
        manifests[split] = rgb_manifest
        manifests[f"{split}_ir"] = ir_manifest
        print(f"[M2D-LIF] {split}: {image_count} RGB/IR pairs, {box_count} remapped boxes")

    yaml_path = stage / "dronevehicle_m2dlif_train.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {json.dumps(str(stage), ensure_ascii=False)}",
                f"train: {json.dumps(str(manifests['train']), ensure_ascii=False)}",
                f"val: {json.dumps(str(manifests['val']), ensure_ascii=False)}",
                f"train_ir: {json.dumps(str(manifests['train_ir']), ensure_ascii=False)}",
                f"val_ir: {json.dumps(str(manifests['val_ir']), ensure_ascii=False)}",
                "names:",
                "  0: car",
                "  1: truck",
                "  2: bus",
                "  3: van",
                "  4: freight_car",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def install_trusted_torch_load(torch_module) -> None:
    """Allow loading trusted local checkpoints across supported PyTorch versions."""

    torch_load = torch_module.load
    supports_weights_only = "weights_only" in inspect.signature(torch_load).parameters

    def trusted_torch_load(*args, **kwargs):
        if supports_weights_only:
            kwargs.setdefault("weights_only", False)
        else:
            kwargs.pop("weights_only", None)
        return torch_load(*args, **kwargs)

    torch_module.load = trusted_torch_load


__all__ = (
    "M2DLIF_LABEL_ROOT",
    "M2DLIF_TO_PROJECT_CLASS",
    "SOURCE_DATA_ROOT",
    "files_by_stem",
    "install_trusted_torch_load",
    "prepare_split",
    "prepare_temporary_dataset",
)
