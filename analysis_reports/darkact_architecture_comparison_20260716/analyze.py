"""Reproduce the DarkAct experiment comparison and learned-weight probes.

The probe uses 16 evenly spaced DroneVehicle test pairs, applies the same
RGB/IR BGR-to-RGB conversion and 640 letterbox used by this fork, and measures
the actual MAA multipliers and LAF correction on the saved EMA in best.pt.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules import (
    LAFMerge2D,
    LAFMergeFeedback2D,
    MAA2D,
    PaperLAFMergeFeedback2D,
    StaticMAA2D,
)
from ultralytics.utils.torch_utils import get_flops


OUT = Path(__file__).with_name("analysis.json")
TEST_RGB = Path("/media/biiteam/新加卷/biiteam/MCONG/datasets/DroneVehicle_twostream_3/images/test")
RUNS = {
    "baseline": ROOT / "runs_baseline/train",
    "v1": ROOT / "DroneVehicle_OBB_FusionTransfer/DarkAct_MAA2DLAFMerge_P345_H2-4-8_StaticSaliency-R4-DW3D1-2-2_v1",
    "v2": ROOT / "DroneVehicle_OBB_FusionTransfer/DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_MSK3-5-R4-PosBeta-DW3D1-2-2_v2",
    "v3": ROOT / "DroneVehicle_OBB_FusionTransfer/DarkAct_PaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-StaticMAA_v3",
}
DISPLAY = {
    "baseline": "Baseline ADD",
    "v1": "V1 MAA2D + LAFMerge2D",
    "v2": "V2 StaticMAA2D + LAF feedback",
    "v3": "V3 paper-style full-C LAF feedback",
}


def read_results(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    metric = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: row[metric])
    tail = rows[-10:]
    return {
        "epochs": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_map50": best["metrics/mAP50(B)"],
        "best_map5095": best[metric],
        "last10_map5095_mean": float(np.mean([row[metric] for row in tail])),
        "last10_map5095_std": float(np.std([row[metric] for row in tail])),
        "final_train_box_loss": rows[-1]["train/box_loss"],
        "final_train_cls_loss": rows[-1]["train/cls_loss"],
        "final_train_dfl_loss": rows[-1]["train/dfl_loss"],
    }


def read_test(path: Path) -> dict:
    text = path.read_text(errors="replace")
    rows = {}
    pattern = re.compile(
        r"^\s*(all|car|truck|bus|van|freight_car)\s+\d+\s+\d+\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        rows[match.group(1)] = {
            "precision": float(match.group(2)),
            "recall": float(match.group(3)),
            "map50": float(match.group(4)),
            "map5095": float(match.group(5)),
        }

    # The baseline log preserved exact aggregate values and exact class maps.
    exact = re.search(r"results_dict:\s*(\{[^\n]+\})", text)
    if exact:
        values = ast.literal_eval(exact.group(1))
        rows["all"].update(
            precision=float(values["metrics/precision(B)"]),
            recall=float(values["metrics/recall(B)"]),
            map50=float(values["metrics/mAP50(B)"]),
            map5095=float(values["metrics/mAP50-95(B)"]),
        )
    maps = re.search(r"maps:\s*array\(\[([^\]]+)\]", text)
    if maps:
        for name, value in zip(("car", "truck", "bus", "van", "freight_car"), maps.group(1).split(",")):
            rows[name]["map5095"] = float(value)

    latency = re.search(r"Speed:.*?([0-9.]+)ms inference", text)
    fps = re.search(r"前向传播 FPS:\s*([0-9.]+)", text)
    return {
        "rows": rows,
        "inference_ms": float(latency.group(1)),
        "fps": float(fps.group(1)),
        "precision_note": "exact" if exact else "rounded_to_3_decimals",
    }


def letterbox_pair(path: Path, size: int = 640) -> torch.Tensor:
    rgb = cv2.imread(str(path))
    ir = cv2.imread(str(path).replace("/images/", "/image/"))
    if rgb is None or ir is None:
        raise FileNotFoundError(path)
    h, w = rgb.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = round(h * ratio), round(w * ratio)
    if (nh, nw) != (h, w):
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        ir = cv2.resize(ir, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    image = np.full((size, size, 6), 114, dtype=np.uint8)
    image[top : top + nh, left : left + nw, :3] = rgb
    image[top : top + nh, left : left + nw, 3:] = ir
    rgb_chw = image[:, :, :3][:, :, ::-1].transpose(2, 0, 1)
    ir_chw = image[:, :, 3:][:, :, ::-1].transpose(2, 0, 1)
    pair = np.ascontiguousarray(np.concatenate((rgb_chw, ir_chw), axis=0))
    return torch.from_numpy(pair).float().div_(255.0)


def rms_ratio(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    num = delta.flatten(1).square().mean(1).sqrt()
    den = base.flatten(1).square().mean(1).sqrt().clamp_min(1e-12)
    return num / den


def probe_model(model: torch.nn.Module, sample_paths: list[Path], batch_size: int = 4) -> list[dict]:
    observed: dict[tuple[int, str], dict[str, list[float] | float]] = {}
    handles = []
    context = {"brightness": None}

    def bucket(layer: int, kind: str) -> dict:
        return observed.setdefault((layer, kind), {"layer": layer, "kind": kind})

    for layer, module in enumerate(model.model):
        if isinstance(module, (MAA2D, StaticMAA2D)):

            def maa_hook(mod, inputs, output, layer=layer):
                rgb, ir = inputs[0].chunk(2, dim=1)
                if isinstance(mod, MAA2D):
                    gates = [mod.saliency[i](x) for i, x in enumerate((rgb, ir))]
                    betas = mod.beta.flatten()
                else:
                    gates = [mod.branches[i](x) for i, x in enumerate((rgb, ir))]
                    betas = mod.beta.flatten()
                gains = [(1 + betas[i] * gates[i]).flatten(1).mean(1) for i in range(2)]
                row = bucket(layer, "MAA")
                row.setdefault("gain_rgb", []).extend(gains[0].detach().cpu().tolist())
                row.setdefault("gain_ir", []).extend(gains[1].detach().cpu().tolist())
                row.setdefault("gate_rgb_spatial_std", []).extend(
                    gates[0].flatten(1).std(dim=1, unbiased=False).detach().cpu().tolist()
                )
                row.setdefault("gate_ir_spatial_std", []).extend(
                    gates[1].flatten(1).std(dim=1, unbiased=False).detach().cpu().tolist()
                )
                row["beta_rgb"] = float(betas[0])
                row["beta_ir"] = float(betas[1])

            handles.append(module.register_forward_hook(maa_hook))

        if isinstance(module, (LAFMerge2D, LAFMergeFeedback2D)):

            def laf_hook(mod, inputs, output, layer=layer):
                merge = mod if isinstance(mod, LAFMerge2D) else mod.merge
                rgb, ir = inputs[0]
                descriptor = torch.cat(
                    (
                        F.adaptive_avg_pool2d(rgb, 1),
                        F.adaptive_max_pool2d(rgb, 1),
                        F.adaptive_avg_pool2d(ir, 1),
                        F.adaptive_max_pool2d(ir, 1),
                    ),
                    dim=1,
                )
                b, _, h, w = rgb.shape
                global_logits = merge.global_gate(descriptor).reshape(b, 2, merge.channels, 1, 1)
                local_descriptor = torch.cat(
                    (merge._contrast_descriptor(rgb), merge._contrast_descriptor(ir)), dim=1
                )
                local_logits = merge.local_gate(local_descriptor).reshape(b, 2, 1, h, w)
                weights = torch.softmax(global_logits + local_logits, dim=1) * 2
                fused = output if isinstance(mod, LAFMerge2D) else output[2]
                base = rgb + ir
                gated = weights[:, 0] * rgb + weights[:, 1] * ir
                cross_term = merge.cross_scale * merge.cross(rgb, ir)
                row = bucket(layer, "LAF")
                rgb_weight = weights[:, 0].flatten(1).mean(1)
                row.setdefault("rgb_weight", []).extend(rgb_weight.detach().cpu().tolist())
                row.setdefault("ir_weight", []).extend((2 - rgb_weight).detach().cpu().tolist())
                row.setdefault("corr_rms_ratio", []).extend(rms_ratio(fused - base, base).detach().cpu().tolist())
                row.setdefault("gate_corr_rms_ratio", []).extend(
                    rms_ratio(gated - base, base).detach().cpu().tolist()
                )
                row.setdefault("cross_corr_rms_ratio", []).extend(
                    rms_ratio(cross_term, base).detach().cpu().tolist()
                )
                row.setdefault("brightness", []).extend(context["brightness"])
                row["cross_scale"] = float(merge.cross_scale)

            handles.append(module.register_forward_hook(laf_hook))

        if isinstance(module, PaperLAFMergeFeedback2D):

            def paper_hook(mod, inputs, output, layer=layer):
                _, rgb, ir = inputs[0]
                base = rgb + ir
                correction = output[2] - base
                row = bucket(layer, "PaperLAF")
                row.setdefault("corr_rms_ratio", []).extend(rms_ratio(correction, base).detach().cpu().tolist())
                cosine = F.cosine_similarity(correction.flatten(1), base.flatten(1), dim=1)
                row.setdefault("corr_cos_add", []).extend(cosine.detach().cpu().tolist())
                gamma = torch.cat([branch[-1].weight.detach().flatten() for branch in mod.output_mlp]).float()
                row["output_bn_gamma_rms"] = float(gamma.square().mean().sqrt())

            handles.append(module.register_forward_hook(paper_hook))

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sample_paths), batch_size):
            batch = torch.stack([letterbox_pair(path) for path in sample_paths[start : start + batch_size]])
            context["brightness"] = batch[:, :3].mean(dim=(1, 2, 3)).tolist()
            model(batch)
    for handle in handles:
        handle.remove()

    rows = []
    for key in sorted(observed):
        source = observed[key]
        row = {"layer": source["layer"], "kind": source["kind"]}
        for name, value in source.items():
            if name in {"layer", "kind", "brightness"}:
                continue
            if isinstance(value, list):
                row[name + "_mean"] = float(np.mean(value))
                row[name + "_std"] = float(np.std(value))
            else:
                row[name] = value
        if "rgb_weight" in source:
            brightness = np.asarray(source["brightness"])
            rgb_weight = np.asarray(source["rgb_weight"])
            row["brightness_rgb_weight_corr"] = float(np.corrcoef(brightness, rgb_weight)[0, 1])
        rows.append(row)
    return rows


def main() -> None:
    files = sorted(TEST_RGB.glob("*"))
    sample_indices = np.linspace(0, len(files) - 1, 16, dtype=int)
    sample_paths = [files[i] for i in sample_indices]
    output = {
        "generated_at": "2026-07-16T12:00:00+08:00",
        "sample_files": [path.name for path in sample_paths],
        "runs": {},
    }

    for run_id, run_dir in RUNS.items():
        checkpoint = torch.load(run_dir / "weights/best.pt", map_location="cpu")
        model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
        output["runs"][run_id] = {
            "display": DISPLAY[run_id],
            "path": str(run_dir.relative_to(ROOT)),
            "validation": read_results(run_dir / "results.csv"),
            "test": read_test(run_dir / "test_result/test.txt"),
            "params": sum(parameter.numel() for parameter in model.parameters()),
            "gflops_640": float(get_flops(model, imgsz=640)),
            "probe": [] if run_id == "baseline" else probe_model(model, sample_paths),
        }

    baseline = output["runs"]["baseline"]
    for run_id in ("v1", "v2", "v3"):
        run = output["runs"][run_id]
        run["delta_vs_baseline"] = {
            "test_map5095_pp": 100 * (run["test"]["rows"]["all"]["map5095"] - baseline["test"]["rows"]["all"]["map5095"]),
            "best_val_map5095_pp": 100 * (run["validation"]["best_map5095"] - baseline["validation"]["best_map5095"]),
            "params_pct": 100 * (run["params"] / baseline["params"] - 1),
            "gflops_pct": 100 * (run["gflops_640"] / baseline["gflops_640"] - 1),
            "fps_pct": 100 * (run["test"]["fps"] / baseline["test"]["fps"] - 1),
        }
        for class_name in ("car", "truck", "bus", "van", "freight_car"):
            run["test"]["rows"][class_name]["delta_map5095_pp"] = 100 * (
                run["test"]["rows"][class_name]["map5095"]
                - baseline["test"]["rows"][class_name]["map5095"]
            )

    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
