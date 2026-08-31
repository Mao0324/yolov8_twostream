#!/usr/bin/env python3
"""Read-only V13 P4 expert/gate diagnostics on a bounded DroneVehicle split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules import P2IRPromptAsymIdentityGDERMergeFeedback2D


DEFAULT_WEIGHTS = (
    ROOT
    / "runs/DroneVehicle_OBB/train-labels=official-v1/p2det/mainline"
    / "P2D-013__asymidentity-p4/seed=000/attempt=01"
    / "weights"
    / "best.pt"
)


class RunningMoments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0

    def update(self, value):
        value = value.detach().double().reshape(-1).cpu()
        self.count += value.numel()
        self.total += float(value.sum())
        self.total_square += float(value.square().sum())

    def result(self):
        if not self.count:
            raise RuntimeError("diagnostic statistic received no values")
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {"mean": mean, "std": variance**0.5, "count": self.count}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=ROOT / "data/dronevehicle.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_device(value):
    if value == "cpu":
        return torch.device("cpu")
    if value.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def build_loader(data_path, split, imgsz, batch_size):
    data = check_det_dataset(str(data_path))
    ir_key = f"{split}_ir"
    if not data.get(split) or not data.get(ir_key):
        raise KeyError(f"dataset must define both {split} and {ir_key}")
    cfg = get_cfg(
        overrides={
            "imgsz": imgsz,
            "task": "obb",
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
        }
    )
    dataset = build_yolo_dataset(
        cfg,
        data[split],
        data[ir_key],
        batch_size,
        data,
        mode="val",
        rect=False,
        stride=32,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
    )


def main():
    args = parse_args()
    if args.batches <= 0 or args.batch <= 0:
        raise ValueError("--batches and --batch must be positive")
    if args.imgsz <= 0 or args.imgsz % 32:
        raise ValueError("--imgsz must be a positive multiple of 32")
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)

    device = resolve_device(args.device)
    model = YOLO(str(weights), task="obb").model.float().to(device).eval()
    p4_modules = [
        module for module in model.modules() if isinstance(module, P2IRPromptAsymIdentityGDERMergeFeedback2D)
    ]
    if len(p4_modules) != 1:
        raise RuntimeError(f"expected exactly one V13 asymmetric GDER module, found {len(p4_modules)}")
    p4 = p4_modules[0]

    capture = {}
    hooks = [
        p4.merge.register_forward_hook(lambda _m, _a, output: capture.update(fused_base=output.detach())),
        p4.modality_expert.register_forward_hook(
            lambda _m, _a, output: capture.update(modality_feature=output.detach())
        ),
        p4.attention_expert.register_forward_hook(
            lambda _m, _a, output: capture.update(attention_feature=output.detach())
        ),
        p4.expert_gate.register_forward_hook(
            lambda _m, _a, output: capture.update(expert_weights=output.detach())
        ),
    ]

    moments = {
        "w_mod": RunningMoments(),
        "w_att": RunningMoments(),
        "e_mod_relative_norm": RunningMoments(),
        "e_att_relative_norm": RunningMoments(),
    }
    gate_batches = []
    samples = 0
    batches = 0
    loader = build_loader(args.data, args.split, args.imgsz, args.batch)
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if batch_index >= args.batches:
                    break
                capture.clear()
                images = batch["img"].float().div_(255.0).to(device)
                _ = model(images)
                required = {"fused_base", "modality_feature", "attention_feature", "expert_weights"}
                if set(capture) != required:
                    raise RuntimeError(f"incomplete hook capture: {sorted(capture)}")
                if any(not torch.isfinite(value).all() for value in capture.values()):
                    raise RuntimeError("V13 expert capture contains NaN/Inf")

                base = capture["fused_base"].float().flatten(1).norm(dim=1).clamp_min(1e-12)
                mod_ratio = capture["modality_feature"].float().flatten(1).norm(dim=1) / base
                att_ratio = capture["attention_feature"].float().flatten(1).norm(dim=1) / base
                weights_batch = capture["expert_weights"].float().squeeze(-1).squeeze(-1)
                moments["w_mod"].update(weights_batch[:, 0])
                moments["w_att"].update(weights_batch[:, 1])
                moments["e_mod_relative_norm"].update(mod_ratio)
                moments["e_att_relative_norm"].update(att_ratio)
                gate_batches.append(weights_batch.cpu())
                samples += images.shape[0]
                batches += 1
    finally:
        for hook in hooks:
            hook.remove()

    if not gate_batches:
        raise RuntimeError("no diagnostic batches were processed")
    all_weights = torch.cat(gate_batches, dim=0)
    per_channel_variance_mean = float(all_weights.var(dim=0, unbiased=False).mean())
    report = {
        "weights": str(weights),
        "data": str(args.data.expanduser().resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batches": batches,
        "samples": samples,
        "w_mod": moments["w_mod"].result(),
        "w_att": moments["w_att"].result(),
        "e_mod_relative_norm": moments["e_mod_relative_norm"].result(),
        "e_att_relative_norm": moments["e_att_relative_norm"].result(),
        "mean_per_channel_gate_variance": per_channel_variance_mean,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
