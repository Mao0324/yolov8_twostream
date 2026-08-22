#!/usr/bin/env python3
"""CPU-safe structural, numerical, transfer, and loss checks for P2Det variants."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.obb.p2det_train import P2PromptOBBModel
from ultralytics.nn.modules import (
    LAFMergeFeedback2D,
    P2_PROMPT_MODULES,
    P2DualPromptGDERMergeFeedback2D,
    P2DualPromptLAFMergeFeedback2D,
    P2IRPromptLAFMergeFeedback2D,
)
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils.torch_utils import get_flops

from tools.p2det_weight_transfer import apply_p2det_baseline_transfer, load_p2det_transfer_reference


BASE_YAML = ROOT / "yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4-PostC2f-v1.yaml"
VARIANTS = (
    ROOT / "yaml/yolov8s-P2Det-IRPrompt-P34-PostC2f-v1.yaml",
    ROOT / "yaml/yolov8s-P2Det-DualPrompt-P34-PostC2f-v2.yaml",
    ROOT / "yaml/yolov8s-P2Det-DualPrompt-GDER-P4-PostC2f-v3.yaml",
    ROOT / "yaml/yolov8s-P2Det-DualPrompt-P345-GDER-P45-PostC2f-v4.yaml",
    ROOT / "yaml/yolov8s-P2Det-DualPrompt-GDER-P345-PostC2f-v5.yaml",
)
OPTIONAL = ROOT / "yaml/yolov8s-P2Det-DualPrompt-GDER-P45-NoStaticMAA-v6.yaml"
def _flatten_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def check_neutral_initialization():
    torch.manual_seed(7)
    old = LAFMergeFeedback2D(64, 2, 4, 1, 8).eval()
    inputs = (torch.randn(1, 64, 12, 12), torch.randn(1, 64, 12, 12))
    maximum = 0.0
    for module_type in (
        P2IRPromptLAFMergeFeedback2D,
        P2DualPromptLAFMergeFeedback2D,
        P2DualPromptGDERMergeFeedback2D,
    ):
        new = module_type(64, 2, 4, 1, 8).eval()
        new.merge.load_state_dict(old.merge.state_dict())
        with torch.no_grad():
            old_outputs = old(inputs)
            new_outputs = new(inputs)
        difference = max(float((old_value - new_value).abs().max()) for old_value, new_value in zip(old_outputs, new_outputs))
        maximum = max(maximum, difference)
        if difference >= 1e-5:
            raise AssertionError(f"neutral initialization failed for {module_type.__name__}: {difference}")
    print(f"neutral initialization max_abs_diff={maximum:.9g}")
    return maximum


def check_structure(path, imgsz):
    model = P2PromptOBBModel(path, ch=3, verbose=False).eval()
    prompt_modules = [module for module in model.modules() if isinstance(module, P2_PROMPT_MODULES)]
    prompt_shapes = []
    head_shapes = []
    gder_sums = []
    handles = []

    for module in prompt_modules:
        handles.append(module.ir_prompt_head.register_forward_hook(lambda _, __, output: prompt_shapes.append(tuple(output.shape))))
        if getattr(module, "use_rgb_prompt", False):
            handles.append(module.rgb_prompt_head.register_forward_hook(lambda _, __, output: prompt_shapes.append(tuple(output.shape))))
        if isinstance(module, P2DualPromptGDERMergeFeedback2D):
            handles.append(
                module.expert_gate.register_forward_hook(
                    lambda _, __, output: gder_sums.append(float((output.sum(dim=1) - 1.0).abs().max()))
                )
            )

    def capture_head_inputs(_, args):
        head_shapes.extend(tuple(tensor.shape) for tensor in args[0])

    handles.append(model.model[-1].register_forward_pre_hook(capture_head_inputs))
    try:
        with torch.no_grad():
            output = model.predict(torch.randn(1, 6, imgsz, imgsz))
    finally:
        for handle in handles:
            handle.remove()

    tensors = _flatten_tensors(output)
    if not tensors or not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise AssertionError(f"non-finite or missing output tensors for {path.name}")
    expected_head = [(1, 128, imgsz // 8, imgsz // 8), (1, 256, imgsz // 16, imgsz // 16), (1, 512, imgsz // 32, imgsz // 32)]
    if head_shapes != expected_head:
        raise AssertionError(f"unexpected OBB head inputs for {path.name}: {head_shapes}")
    if any(shape[0] != 1 or shape[1] != 1 for shape in prompt_shapes):
        raise AssertionError(f"invalid prompt shape for {path.name}: {prompt_shapes}")
    if any(error > 1e-6 for error in gder_sums):
        raise AssertionError(f"GDER weights do not sum to one for {path.name}: {gder_sums}")
    if any(module._last_prompt_logits is not None for module in prompt_modules):
        raise AssertionError(f"ordinary inference retained prompt logits for {path.name}")

    params = sum(parameter.numel() for parameter in model.parameters())
    flops = get_flops(model, imgsz=640)
    print(
        f"{path.name}: forward=ok finite=ok head={head_shapes} prompts={prompt_shapes} "
        f"gder_sum_error={max(gder_sums, default=0.0):.3g} params={params} GFLOPs@640={flops:.4f}"
    )
    return model, params, flops


def check_loss_backward(path):
    model = P2PromptOBBModel(path, ch=3, verbose=False).train()
    model.args = get_cfg()
    model.p2_prompt_epoch = 1
    batch = {
        "img": torch.rand(1, 6, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.15, 0.2]]),
    }
    total, items = model(batch)
    if items.numel() != 5 or not torch.isfinite(total) or not torch.isfinite(items).all():
        raise AssertionError(f"invalid P2 losses for {path.name}: total={total}, items={items}")
    total.backward()

    prompt_modules = [module for module in model.modules() if isinstance(module, P2_PROMPT_MODULES)]
    for module in prompt_modules:
        if module.ir_prompt_head.body[-1].weight.grad is None or module.ir_prompt_embed.projection.weight.grad is None:
            raise AssertionError(f"missing IR prompt gradient in {path.name}")
        if module.use_rgb_prompt and (
            module.rgb_prompt_head.body[-1].weight.grad is None
            or module.rgb_prompt_embed.projection.weight.grad is None
        ):
            raise AssertionError(f"missing RGB prompt gradient in {path.name}")
        if module.use_gder and (
            module.modality_expert.body[-1].weight.grad is None
            or module.attention_expert.projection.weight.grad is None
            or module.expert_gate.body[-1].weight.grad is None
        ):
            raise AssertionError(f"missing GDER gradient in {path.name}")

    empty = {**batch, "batch_idx": torch.empty(0), "cls": torch.empty(0, 1), "bboxes": torch.empty(0, 5)}
    empty_total, empty_items = model(empty)
    if not torch.isfinite(empty_total) or not torch.isfinite(empty_items).all():
        raise AssertionError(f"empty-target batch is non-finite for {path.name}")
    print(f"{path.name}: loss/backward=ok losses={[float(value) for value in items]} empty_target=ok")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=320, help="CPU structural-forward size; must be divisible by 32")
    parser.add_argument("--include-v6", action="store_true")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads used by structural checks")
    parser.add_argument("--skip-loss", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.imgsz <= 0 or args.imgsz % 32:
        raise ValueError("--imgsz must be a positive multiple of 32")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    check_neutral_initialization()

    base = OBBModel(BASE_YAML, ch=3, verbose=False).eval()
    base_params = sum(parameter.numel() for parameter in base.parameters())
    base_flops = get_flops(base, imgsz=640)
    print(f"baseline: params={base_params} GFLOPs@640={base_flops:.4f}")

    paths = list(VARIANTS) + ([OPTIONAL] if args.include_v6 else [])
    models = {}
    for path in paths:
        model, params, flops = check_structure(path, args.imgsz)
        models[path] = model
        print(f"  delta: params={params - base_params:+d} GFLOPs={flops - base_flops:+.4f}")

    if not args.skip_loss:
        check_loss_backward(VARIANTS[0])
        check_loss_backward(VARIANTS[3])

    if not args.skip_transfer:
        reference = load_p2det_transfer_reference()
        print(
            "canonical -> baseline: "
            f"verified={reference.canonical_baseline_tensors_verified} "
            f"max_abs_diff={reference.canonical_baseline_max_abs_diff:.9g}; "
            "reconstructed full baseline: "
            f"verified={reference.reconstructed_baseline_tensors_verified} "
            f"max_abs_diff={reference.reconstructed_baseline_max_abs_diff:.9g}"
        )
        for path, model in models.items():
            report = apply_p2det_baseline_transfer(model, reference)
            print(
                f"{path.name}: transfer tensors={report['baseline_p2det_tensors_copied']} "
                f"max_abs_diff={report['baseline_p2det_max_abs_diff']:.9g} "
                f"matched_parameters={report['matched_parameter_count']} "
                f"total_parameters={report['total_parameter_count']} "
                f"ratio={report['transfer_ratio']:.6%}"
            )


if __name__ == "__main__":
    main()
