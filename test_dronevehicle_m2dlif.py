#!/usr/bin/env python3
"""
用途：
    使用 M2D-LIF 发布的 DroneVehicle 测试标注，对本仓库训练得到的双流 OBB 模型进行测试。
    M2D-LIF 将 DroneVehicle 测试集放在 ``labels/val`` 中；本脚本仍明确以 ``split=test``
    运行。测试前会将 M2D-LIF 类别 ID 临时映射到本仓库的类别顺序：
    0->0(car)、1->1(truck)、2->4(freight_car)、3->2(bus)、4->3(van)。

默认输入：
    模型权重：通过命令行 ``--weights /path/to/best.pt`` 指定。
    测试标注：
        /media/biiteam/新加卷1/biiteam/MCONG/datasets/
        M2D-LIFlabels/DroneVehicle_test_labels/labels/val
    RGB 测试图像：
        /media/biiteam/新加卷1/biiteam/MCONG/datasets/
        DroneVehicle_twostream_3/images/test
    IR 测试图像：
        /media/biiteam/新加卷1/biiteam/MCONG/datasets/
        DroneVehicle_twostream_3/image/test

输出：
    默认不保留文件：重映射标签、数据清单、YAML、标签缓存和验证结果均写入
    ``/tmp/dronevehicle_m2dlif_*``，进程正常结束或异常退出后自动清理，指标打印到终端。
    若传入 ``--project /输出目录``，则永久结果写入
    ``<project>/<name>/``；其中 ``name`` 默认是 ``test_m2dlif``，并额外保存 ``test.txt``。

示例：
    python -B test_dronevehicle_m2dlif.py --weights /path/to/best.pt
    python -B test_dronevehicle_m2dlif.py --weights /path/to/best.pt \
        --project /path/to/output --name test_m2dlif
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = Path(
    "/media/biiteam/新加卷1/biiteam/MCONG/datasets/"
    "M2D-LIFlabels/DroneVehicle_test_labels/labels/val"
)
DEFAULT_RGB = Path(
    "/media/biiteam/新加卷1/biiteam/MCONG/datasets/"
    "DroneVehicle_twostream_3/images/test"
)
DEFAULT_IR = Path(
    "/media/biiteam/新加卷1/biiteam/MCONG/datasets/"
    "DroneVehicle_twostream_3/image/test"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

# M2D-LIF: 0 car, 1 truck, 2 freight_car, 3 bus, 4 van
# This repo: 0 car, 1 truck, 2 bus, 3 van, 4 freight_car
M2DLIF_TO_PROJECT_CLASS = {0: 0, 1: 1, 2: 4, 3: 2, 4: 3}


def query_gpu_free_memory(timeout=5):
    """Return ``[(physical_index, free_mib), ...]`` using nvidia-smi."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("--device auto requires nvidia-smi, but it was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"nvidia-smi did not respond within {timeout} seconds") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"nvidia-smi failed while selecting a test GPU: {detail}")

    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            gpus.append((int(fields[0]), int(fields[1])))
        except ValueError:
            continue
    if not gpus:
        raise RuntimeError("nvidia-smi returned no parseable GPU memory rows")
    return gpus


def resolve_test_device(requested, min_free_mib):
    """Resolve ``auto`` to the GPU with the most free memory."""

    requested = str(requested).strip().lower()
    if requested not in {"", "auto"}:
        return requested

    gpus = sorted(query_gpu_free_memory(), key=lambda item: (-item[1], item[0]))
    selected_index, selected_free = gpus[0]
    snapshot = ", ".join(f"GPU {index}: {free} MiB free" for index, free in sorted(gpus))
    if selected_free < min_free_mib:
        raise RuntimeError(
            f"No GPU has at least {min_free_mib} MiB free. Current snapshot: {snapshot}. "
            "Lower --batch/--min-free-mib or explicitly use --device cpu."
        )
    print(f"Auto-selected GPU {selected_index} ({selected_free} MiB free). {snapshot}")
    return str(selected_index)


def files_by_stem(directory, suffixes):
    """Index regular files by stem and reject ambiguous duplicate stems."""

    indexed = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in indexed:
            raise ValueError(f"duplicate stem in {directory}: {indexed[path.stem].name}, {path.name}")
        indexed[path.stem] = path
    return indexed


def validate_inputs(weights, labels, rgb, ir):
    """Validate that every RGB/IR test pair has exactly one M2D-LIF label."""

    for kind, path in (("weights", weights), ("labels", labels), ("RGB images", rgb), ("IR images", ir)):
        expected = path.is_file() if kind == "weights" else path.is_dir()
        if not expected:
            raise FileNotFoundError(f"{kind} not found: {path}")

    rgb_files = files_by_stem(rgb, IMAGE_SUFFIXES)
    ir_files = files_by_stem(ir, IMAGE_SUFFIXES)
    label_files = files_by_stem(labels, {".txt"})
    if not rgb_files:
        raise ValueError(f"no test images found in {rgb}")

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
        raise ValueError("test data do not match: " + "; ".join(problems))

    # This fork derives the IR path by replacing /images/ with /image/, so paired
    # files must also share the same extension.
    extension_mismatches = [
        stem for stem, path in rgb_files.items() if ir_files[stem].name != path.name
    ]
    if extension_mismatches:
        raise ValueError(
            "RGB/IR filenames differ for "
            f"{len(extension_mismatches)} pairs (first: {extension_mismatches[:5]})"
        )
    return [rgb_files[stem].name for stem in sorted(rgb_files)]


def write_temporary_dataset(stage, labels, rgb, ir, image_names):
    """Create an automatically cleaned dataset view with remapped label IDs."""

    (stage / "images").mkdir()
    (stage / "image").mkdir()
    (stage / "labels/test").mkdir(parents=True)
    (stage / "images/test").symlink_to(rgb, target_is_directory=True)
    (stage / "image/test").symlink_to(ir, target_is_directory=True)

    # Do not modify the downloaded annotations. Only the small text labels are
    # rewritten inside /tmp and TemporaryDirectory removes them after the run.
    for image_name in image_names:
        source = labels / Path(image_name).with_suffix(".txt").name
        remapped_lines = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 9:
                raise ValueError(f"expected OBB class+8 coordinates at {source}:{line_number}, got {len(fields)}")
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
        (stage / "labels/test" / source.name).write_text(
            "\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8"
        )

    rgb_manifest = stage / "test"
    ir_manifest = stage / "test_ir"
    rgb_manifest.write_text(
        "".join(f"{stage / 'images/test' / name}\n" for name in image_names), encoding="utf-8"
    )
    ir_manifest.write_text(
        "".join(f"{stage / 'image/test' / name}\n" for name in image_names), encoding="utf-8"
    )

    # train/val keys are required by this fork's YAML checker but are never read
    # because validation below explicitly uses split=test.
    quoted_rgb = json.dumps(str(rgb_manifest), ensure_ascii=False)
    quoted_ir = json.dumps(str(ir_manifest), ensure_ascii=False)
    yaml_path = stage / "m2dlif_test.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {json.dumps(str(stage), ensure_ascii=False)}",
                f"train: {quoted_rgb}",
                f"val: {quoted_rgb}",
                f"test: {quoted_rgb}",
                f"train_ir: {quoted_ir}",
                f"val_ir: {quoted_ir}",
                f"test_ir: {quoted_ir}",
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


def save_test_report(validator, args):
    """Save one compact persistent report when --project is supplied."""

    metrics = validator.metrics
    speed = metrics.speed
    inference_ms = speed.get("inference", 0.0)
    lines = [
        f"weights: {args.weights}",
        f"labels: {args.labels}",
        f"rgb: {args.rgb}",
        f"ir: {args.ir}",
        "class_map: M2D-LIF 0->0, 1->1, 2->4, 3->2, 4->3",
        "split: test",
        f"imgsz: {args.imgsz}",
        f"batch: {args.batch}",
        f"device: {args.device}",
        "",
        f"{'Class':>22}{'Images':>11}{'Instances':>11}{'Box(P':>11}{'R':>11}{'mAP50':>11}{'mAP50-95)':>11}",
    ]
    row_format = "{:>22}{:>11d}{:>11d}" + "{:>11.3g}" * 4
    lines.append(
        row_format.format("all", validator.seen, int(validator.nt_per_class.sum()), *metrics.mean_results())
    )
    for result_index, class_index in enumerate(metrics.ap_class_index):
        class_index = int(class_index)
        lines.append(
            row_format.format(
                validator.names[class_index],
                validator.seen,
                int(validator.nt_per_class[class_index]),
                *metrics.class_result(result_index),
            )
        )
    lines.extend(
        [
            "",
            "Speed: "
            f"{speed.get('preprocess', 0.0):.1f}ms preprocess, "
            f"{inference_ms:.1f}ms inference, "
            f"{speed.get('loss', 0.0):.1f}ms loss, "
            f"{speed.get('postprocess', 0.0):.1f}ms postprocess per image",
            f"forward FPS: {1000.0 / inference_ms:.2f}" if inference_ms > 0 else "forward FPS: N/A",
        ]
    )
    report_path = Path(validator.save_dir) / "test.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Test report saved to {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="trained model weights, e.g. runs/.../weights/best.pt")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="M2D-LIF test label directory")
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB, help="existing DroneVehicle RGB test directory")
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR, help="existing DroneVehicle IR test directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="CUDA device such as 0, cpu, or auto")
    parser.add_argument("--min-free-mib", type=int, default=8192)
    parser.add_argument(
        "--project",
        default=None,
        help="persist results here; by default all Ultralytics output is temporary and removed",
    )
    parser.add_argument("--name", default="test_m2dlif")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--plots", action="store_true", help="save validation plots (requires --project)")
    parser.add_argument("--check-only", action="store_true", help="check image/label pairing without loading a model")
    return parser.parse_args()


def main():
    args = parse_args()
    args.weights = Path(args.weights).expanduser().resolve()
    args.labels = args.labels.expanduser().resolve()
    args.rgb = args.rgb.expanduser().resolve()
    args.ir = args.ir.expanduser().resolve()
    if args.min_free_mib <= 0:
        raise ValueError(f"--min-free-mib must be positive, got {args.min_free_mib}")
    if args.plots and not args.project:
        raise ValueError("--plots requires --project so the generated plots are not immediately removed")

    image_names = validate_inputs(args.weights, args.labels, args.rgb, args.ir)
    print(f"Validated {len(image_names)} RGB/IR/label test pairs from {args.labels}")
    if args.check_only:
        return

    args.device = resolve_test_device(args.device, args.min_free_mib)
    persistent = args.project is not None
    if persistent:
        args.project = str(Path(args.project).expanduser().resolve())

    # All adapter files, label caches, and (by default) validation outputs live
    # under this temporary directory and are removed even when validation fails.
    with tempfile.TemporaryDirectory(prefix="dronevehicle_m2dlif_") as temporary_dir:
        stage = Path(temporary_dir)
        data_yaml = write_temporary_dataset(stage, args.labels, args.rgb, args.ir, image_names)
        output_project = args.project if persistent else str(stage / "output")

        # GPU selection happens before importing PyTorch/Ultralytics.
        from ultralytics import YOLO
        import ultralytics.nn.tasks  # noqa: F401

        model = YOLO(str(args.weights))
        if persistent:
            model.add_callback("on_val_end", lambda validator: save_test_report(validator, args))
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=output_project,
            name=args.name,
            conf=args.conf,
            iou=args.iou,
            task="obb",
            plots=args.plots,
            save_json=False,
            save_txt=False,
            exist_ok=True,
        )
        print(metrics)

    if not persistent:
        print("Temporary dataset view, cache, and validation output removed.")


if __name__ == "__main__":
    main()
