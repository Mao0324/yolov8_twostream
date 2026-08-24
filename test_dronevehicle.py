from pathlib import Path
import argparse
import subprocess
#python test_dronevehicle.py --weights /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs_assa_ir_to_rgb2/train/weights/best.pt --project /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs_assa_ir_to_rgb2/train  --name test_result


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


def resolve_test_device(requested, min_free_mib=8192):
    """Resolve ``auto`` to the physical GPU with the most currently free memory."""
    requested = str(requested).strip().lower()
    if requested not in {"", "auto"}:
        return requested

    gpus = sorted(query_gpu_free_memory(), key=lambda item: (-item[1], item[0]))
    selected_index, selected_free = gpus[0]
    snapshot = ", ".join(f"GPU {index}: {free_mib} MiB free" for index, free_mib in sorted(gpus))
    if selected_free < min_free_mib:
        raise RuntimeError(
            "No GPU has enough free memory for safe DroneVehicle validation: "
            f"required >= {min_free_mib} MiB; {snapshot}. "
            "Wait for a training job to finish, lower --batch together with --min-free-mib, "
            "or explicitly use --device cpu."
        )
    print(
        f"Auto-selected CUDA device {selected_index} with {selected_free} MiB free "
        f"(minimum {min_free_mib} MiB). Current snapshot: {snapshot}"
    )
    return str(selected_index)


def save_test_report(validator, cli_args):
    """Save the final OBB metrics table to test.txt in the validation directory."""
    metrics = validator.metrics
    speed = metrics.speed
    inference_ms = speed.get("inference", 0.0)

    lines = [
        f"weights: {Path(cli_args.weights).resolve()}",
        f"data: {Path(cli_args.data).resolve()}",
        "split: test",
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
    repo_root = Path(__file__).resolve().parent
    default_data = repo_root / "data" / "dronevehicle.yaml"

    parser = argparse.ArgumentParser(description="Test DroneVehicle on test split.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights, e.g. runs/.../weights/best.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(default_data),
        help="Dataset yaml path (default: data/dronevehicle.yaml)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size")
    parser.add_argument("--workers", type=int, default=0, help="Validation dataloader workers")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="CUDA device, e.g. 0; default auto selects the GPU with most free memory",
    )
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=8192,
        help="Minimum free GPU memory required by --device auto (default: 8192 MiB)",
    )
    parser.add_argument("--project", type=str, default=None, help="Output project directory")
    parser.add_argument("--name", type=str, default="test_dronevehicle", help="Run name")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_free_mib <= 0:
        raise ValueError(f"--min-free-mib must be positive, got {args.min_free_mib}")
    args.device = resolve_test_device(args.device, args.min_free_mib)

    # Device selection must finish before importing Ultralytics/PyTorch so
    # select_device() can safely set CUDA_VISIBLE_DEVICES to the chosen card.
    from ultralytics import YOLO
    import ultralytics.nn.tasks  # noqa: F401

    model = YOLO(args.weights)
    model.add_callback("on_val_end", lambda validator: save_test_report(validator, args))

    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        conf=args.conf,
        iou=args.iou,
        task="obb",
    )
    print(metrics)


if __name__ == "__main__":
    main()
