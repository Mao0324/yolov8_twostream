#!/usr/bin/env python3
"""Evaluate a three-class two-stream model on the aligned FLIR HBB test split."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data/flir_align.yaml"
PREPARED_DATA = ROOT / "data/flir_align_hbb"


def query_gpu_free_memory(timeout=5):
    """Return ``[(physical_index, free_mib), ...]`` without importing CUDA/PyTorch."""

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


def resolve_test_device(requested, min_free_mib=4096):
    """Resolve ``auto`` to the physical GPU with the most free memory."""

    requested = str(requested).strip().lower()
    if requested not in {"", "auto"}:
        return requested

    gpus = sorted(query_gpu_free_memory(), key=lambda item: (-item[1], item[0]))
    selected_index, selected_free = gpus[0]
    snapshot = ", ".join(f"GPU {index}: {free_mib} MiB free" for index, free_mib in sorted(gpus))
    if selected_free < min_free_mib:
        raise RuntimeError(
            "No GPU has enough free memory for safe FLIR HBB validation: "
            f"required >= {min_free_mib} MiB; {snapshot}. "
            "Wait for a training job to finish, lower --batch together with --min-free-mib, "
            "or explicitly use --device cpu."
        )
    print(
        f"Auto-selected CUDA device {selected_index} with {selected_free} MiB free "
        f"(minimum {min_free_mib} MiB). Current snapshot: {snapshot}"
    )
    return str(selected_index)


def infer_project(weights, requested_project):
    """Use the training run as project when weights are ``<run>/weights/*.pt``."""

    if requested_project:
        return str(Path(requested_project).expanduser().resolve())
    weights = Path(weights).expanduser().resolve()
    if weights.parent.name == "weights":
        return str(weights.parent.parent)
    return str(ROOT / "runs/FLIR_HBB_Evaluation")


def save_test_report(validator, cli_args):
    """Save final HBB metrics to ``test.txt`` in the validation directory."""

    metrics = validator.metrics
    speed = metrics.speed
    inference_ms = speed.get("inference", 0.0)
    lines = [
        f"weights: {Path(cli_args.weights).expanduser().resolve()}",
        f"data: {Path(cli_args.data).expanduser().resolve()}",
        f"split: {cli_args.split}",
        "task: detect",
        "annotation_type: HBB",
        f"imgsz: {cli_args.imgsz}",
        f"batch: {cli_args.batch}",
        f"device: {cli_args.device}",
        "",
        f"{'Class':>22}{'Images':>11}{'Instances':>11}{'Box(P':>11}{'R':>11}{'mAP50':>11}{'mAP50-95)':>11}",
    ]

    row_format = "{:>22}{:>11d}{:>11d}" + "{:>11.3g}" * 4
    lines.append(
        row_format.format(
            "all",
            validator.seen,
            int(validator.nt_per_class.sum()),
            *metrics.mean_results(),
        )
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
            f"前向传播 FPS: {1000.0 / inference_ms:.2f}" if inference_ms > 0 else "前向传播 FPS: N/A",
            f"Results saved to {validator.save_dir}",
        ]
    )

    report_path = Path(validator.save_dir) / "test.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Test metrics saved to {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="trained FLIR HBB weights, normally <run>/weights/best.pt",
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="FLIR dataset YAML")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="CUDA device such as 0; auto selects the GPU with most free memory",
    )
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="output project; defaults to the training run containing weights/",
    )
    parser.add_argument("--name", type=str, default="test_result")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="prepare the paired FLIR HBB layout before validation",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="original flir_align root used with --prepare-data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data}")
    if args.min_free_mib <= 0:
        raise ValueError(f"--min-free-mib must be positive, got {args.min_free_mib}")

    if args.prepare_data:
        from tools.prepare_flir_align_hbb import DEFAULT_SOURCE, prepare_dataset

        manifest = prepare_dataset(args.source_root or DEFAULT_SOURCE, PREPARED_DATA)
        print(
            "[DATA] prepared FLIR pairs: "
            + ", ".join(
                f"{split}={stats['images']} images/{stats['boxes']} boxes"
                for split, stats in manifest["splits"].items()
            )
        )
    else:
        from tools.prepare_flir_align_hbb import validate_prepared_dataset

        validate_prepared_dataset(PREPARED_DATA)

    args.weights = str(weights)
    args.data = str(data)
    args.project = infer_project(weights, args.project)
    args.device = resolve_test_device(args.device, args.min_free_mib)

    # Select the physical GPU before importing PyTorch/Ultralytics.
    from ultralytics import YOLO
    import ultralytics.nn.tasks  # noqa: F401  # Register custom two-stream modules.

    model = YOLO(args.weights, task="detect")
    detect_head = model.model.model[-1]
    model_nc = int(getattr(detect_head, "nc", len(model.names)))
    if model_nc != 3:
        raise ValueError(
            f"FLIR evaluation now uses 3 classes (car/person/bicycle), but {weights} has nc={model_nc}. "
            "Retrain with the three-class checkpoint/config instead of evaluating legacy four-class weights."
        )
    model.add_callback("on_val_end", lambda validator: save_test_report(validator, args))
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save_json=args.save_json,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        exist_ok=args.exist_ok,
        task="detect",
    )
    print(metrics)
    return metrics


if __name__ == "__main__":
    main()
