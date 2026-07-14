from pathlib import Path
import argparse
#python test_dronevehicle.py --weights /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs_assa_ir_to_rgb2/train/weights/best.pt --project /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs_assa_ir_to_rgb2/train  --name test_result


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
    parser.add_argument("--device", type=str, default="0", help="CUDA device, e.g. 0 or 0,1")
    parser.add_argument("--project", type=str, default=None, help="Output project directory")
    parser.add_argument("--name", type=str, default="test_dronevehicle", help="Run name")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    return parser.parse_args()


def main():
    args = parse_args()
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
