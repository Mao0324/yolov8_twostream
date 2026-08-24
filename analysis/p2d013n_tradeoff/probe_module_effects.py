#!/usr/bin/env python3
"""Probe learned IRPrompt/GDER behavior on a fixed test-image sample."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import ultralytics.nn.tasks  # noqa: F401


WEIGHTS = (
    ROOT
    / "DroneVehicle_OBB_FusionTransfer"
    / "P2D-013N_IRPrompt-AsymIdentityGDER-P4-NoStaticMAA_PostC2f_v13_scalen"
    / "weights"
    / "best.pt"
)
DATASET = Path("/media/biiteam/新加卷1/biiteam/MCONG/datasets/DroneVehicle_twostream_3")
SAMPLE_IDS = tuple(f"{index:05d}" for index in range(1, 9))


def trusted_load(*args, **kwargs):
    """Load repository-owned checkpoints across PyTorch versions."""
    if "weights_only" in inspect.signature(_TORCH_LOAD).parameters:
        kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


_TORCH_LOAD = torch.load
torch.load = trusted_load


def rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().mean().sqrt())


def load_batch() -> torch.Tensor:
    letterbox = LetterBox((640, 640), auto=False, scaleup=False, stride=32)
    images = []
    for sample_id in SAMPLE_IDS:
        rgb_path = DATASET / "images" / "test" / f"{sample_id}.jpg"
        ir_path = DATASET / "image" / "test" / f"{sample_id}.jpg"
        rgb = cv2.imread(str(rgb_path))
        ir = cv2.imread(str(ir_path))
        if rgb is None or ir is None:
            raise FileNotFoundError(f"Missing RGB/IR pair for {sample_id}")
        paired = np.dstack((rgb, ir))
        paired = letterbox(image=paired)
        rgb_chw = np.ascontiguousarray(paired[..., :3].transpose(2, 0, 1)[::-1])
        ir_chw = np.ascontiguousarray(paired[..., 3:].transpose(2, 0, 1)[::-1])
        images.append(np.concatenate((rgb_chw, ir_chw), axis=0))
    return torch.from_numpy(np.stack(images)).float() / 255.0


def main() -> None:
    model = YOLO(str(WEIGHTS), task="obb").model.float().eval()
    rows = []
    handles = []

    def make_hook(stage: str):
        def hook(module, inputs, _output):
            rgb, ir = inputs[0]
            with torch.no_grad():
                prompt = module.ir_prompt_head(ir).sigmoid()
                embedded = module.ir_prompt_embed(prompt)
                prompted_ir = ir + embedded
                fused = module.merge((rgb, prompted_ir))
                row = {
                    "stage": stage,
                    "prompt_mean": float(prompt.mean()),
                    "prompt_std": float(prompt.std(unbiased=False)),
                    "ir_rms": rms(ir),
                    "prompt_embedding_rms": rms(embedded),
                    "prompt_embedding_to_ir_pct": 100.0 * rms(embedded) / max(rms(ir), 1e-12),
                    "fused_base_rms": rms(fused),
                    "has_gder": bool(module.use_gder),
                }
                if module.use_gder:
                    gate_mix = rgb + prompt * prompted_ir
                    zeros = torch.zeros_like(prompt)
                    weights = module.expert_gate(fused, gate_mix, zeros, prompt)
                    modality = module.modality_expert(rgb, prompted_ir, prompt)
                    attention = module.attention_expert(fused)
                    weighted_modality = weights[:, 0] * modality
                    weighted_attention = weights[:, 1] * attention
                    row.update(
                        {
                            "gder_modality_weight_mean": float(weights[:, 0].mean()),
                            "gder_attention_weight_mean": float(weights[:, 1].mean()),
                            "gder_gate_std": float(weights.std(unbiased=False)),
                            "modality_expert_rms": rms(modality),
                            "attention_expert_rms": rms(attention),
                            "weighted_modality_to_fused_pct": 100.0
                            * rms(weighted_modality)
                            / max(rms(fused), 1e-12),
                            "weighted_attention_to_fused_pct": 100.0
                            * rms(weighted_attention)
                            / max(rms(fused), 1e-12),
                        }
                    )
                rows.append(row)

        return hook

    for index, module in enumerate(model.model):
        if type(module).__name__.startswith("P2IRPrompt"):
            stage = {11: "P3", 17: "P4", 25: "P5"}.get(index, f"layer_{index}")
            handles.append(module.register_forward_hook(make_hook(stage)))

    batch = load_batch()
    with torch.no_grad():
        model(batch)
    for handle in handles:
        handle.remove()

    parameter_rows = []
    selected = (
        "ir_prompt_embed.projection.weight",
        "modality_expert.fusion.3.weight",
        "attention_expert.projection.weight",
        "expert_gate.body.2.weight",
        "expert_gate.body.2.bias",
    )
    for name, parameter in model.named_parameters():
        if any(token in name for token in selected):
            value = parameter.detach().float()
            parameter_rows.append(
                {
                    "parameter": name,
                    "shape": list(value.shape),
                    "l2": float(value.norm()),
                    "abs_mean": float(value.abs().mean()),
                    "abs_max": float(value.abs().max()),
                }
            )

    output = {
        "weights": str(WEIGHTS.relative_to(ROOT)),
        "sample_ids": list(SAMPLE_IDS),
        "sample_size": len(SAMPLE_IDS),
        "module_probe": rows,
        "parameter_probe": parameter_rows,
    }
    output_path = Path(__file__).with_name("module_probe.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
