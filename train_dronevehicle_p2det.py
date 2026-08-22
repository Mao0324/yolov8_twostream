#!/usr/bin/env python3
"""Train P2Det from canonical YOLOv8s-OBB with exact two-stream baseline common state."""

import argparse
import inspect
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

import torch

from ultralytics import YOLO
from tools.pretrained_rerun_tracker import repo_path
from tools.p2det_weight_transfer import (
    BASELINE_REFERENCE,
    CANONICAL_SOURCE,
    migrate_p2det_from_canonical,
)
from tools.queue_runtime import resolve_queue_runtime
from tools.training_monitor import (
    MonitoredP2PromptOBBTrainer,
    MonitoredP2SecondGenOBBTrainer,
    create_monitor,
)


VARIANTS = {
    "ir_prompt": (
        "yaml/yolov8s-P2Det-IRPrompt-P34-PostC2f-v1.yaml",
        "P2D-001_IRPrompt-P34_PostC2f_v1",
    ),
    "dual_prompt": (
        "yaml/yolov8s-P2Det-DualPrompt-P34-PostC2f-v2.yaml",
        "P2D-002_DualPrompt-P34_PostC2f_v2",
    ),
    "gder_p4": (
        "yaml/yolov8s-P2Det-DualPrompt-GDER-P4-PostC2f-v3.yaml",
        "P2D-003_DualPrompt-GDER-P4_PostC2f_v3",
    ),
    "gder_p45": (
        "yaml/yolov8s-P2Det-DualPrompt-P345-GDER-P45-PostC2f-v4.yaml",
        "P2D-004_DualPrompt-P345-GDER-P45_PostC2f_v4",
    ),
    "gder_p345": (
        "yaml/yolov8s-P2Det-DualPrompt-GDER-P345-PostC2f-v5.yaml",
        "P2D-005_DualPrompt-GDER-P345_PostC2f_v5",
    ),
    "no_static_maa": (
        "yaml/yolov8s-P2Det-DualPrompt-GDER-P45-NoStaticMAA-v6.yaml",
        "P2D-006_DualPrompt-GDER-P45-NoStaticMAA_v6",
    ),
    "global_only": (
        "yaml/yolov8s-P2Det-IRSpatial-RGBGlobal-P34-PostC2f-v7.yaml",
        "P2D-007_IRSpatial-RGBGlobal-P34_PostC2f_v7",
    ),
    "identity_gder": (
        "yaml/yolov8s-P2Det-DualPrompt-IdentityGDER-P45-NoStaticMAA-v8.yaml",
        "P2D-008_DualPrompt-IdentityGDER-P45-NoStaticMAA_v8",
    ),
    "global_identity_gder": (
        "yaml/yolov8s-P2Det-DualPrompt-RGBGlobal-IdentityGDER-P45-NoStaticMAA-v9.yaml",
        "P2D-009_DualPrompt-RGBGlobal-IdentityGDER-P45-NoStaticMAA_v9",
    ),
    "factorized_p4": (
        "yaml/yolov8s-P2Det-DualPrompt-RGBGlobal-IdentityGDER-FactorizedP4-NoStaticMAA-v10.yaml",
        "P2D-010_DualPrompt-RGBGlobal-IdentityGDER-FactorizedP4-NoStaticMAA_v10",
    ),
    "ir_only_no_static": (
        "yaml/yolov8s-P2Det-IRPrompt-P34-NoStaticMAA-NoGDER-PostC2f-v11.yaml",
        "P2D-011_IRPrompt-P34-NoStaticMAA-NoGDER_PostC2f_v11",
    ),
    "asym_identity_gder_p45": (
        "yaml/yolov8s-P2Det-IRPrompt-AsymIdentityGDER-P45-NoStaticMAA-PostC2f-v12.yaml",
        "P2D-012_IRPrompt-AsymIdentityGDER-P45-NoStaticMAA_PostC2f_v12",
    ),
    "asym_identity_gder_p4": (
        "yaml/yolov8s-P2Det-IRPrompt-AsymIdentityGDER-P4-NoStaticMAA-PostC2f-v13.yaml",
        "P2D-013_IRPrompt-AsymIdentityGDER-P4-NoStaticMAA_PostC2f_v13",
    ),
}

# Keep every historical semantic key and add explicit version aliases.
VARIANTS.update(
    {
        "v1": VARIANTS["ir_prompt"],
        "v2": VARIANTS["dual_prompt"],
        "v3": VARIANTS["gder_p4"],
        "v4": VARIANTS["gder_p45"],
        "v5": VARIANTS["gder_p345"],
        "v6": VARIANTS["no_static_maa"],
        "v7": VARIANTS["global_only"],
        "v8": VARIANTS["identity_gder"],
        "v9": VARIANTS["global_identity_gder"],
        "v10": VARIANTS["factorized_p4"],
        "v11": VARIANTS["ir_only_no_static"],
        "v12": VARIANTS["asym_identity_gder_p45"],
        "v13": VARIANTS["asym_identity_gder_p4"],
    }
)
SECOND_GEN_VARIANTS = {
    "global_only",
    "identity_gder",
    "global_identity_gder",
    "factorized_p4",
    "ir_only_no_static",
    "asym_identity_gder_p45",
    "asym_identity_gder_p4",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
}

_torch_load = torch.load
_supports_weights_only = "weights_only" in inspect.signature(_torch_load).parameters


def _trusted_torch_load(*args, **kwargs):
    if _supports_weights_only:
        kwargs.setdefault("weights_only", False)
    else:
        kwargs.pop("weights_only", None)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


def report_checkpoint_transfer(report):
    """Print the mandatory canonical/baseline/P2Det exactness audit."""
    print(f"canonical source: {report['source']}")
    print(f"baseline reference: {report['baseline_reference']}")
    print(f"canonical -> baseline tensors verified: {report['canonical_baseline_tensors_verified']}")
    print(f"canonical -> baseline max_abs_diff: {report['canonical_baseline_max_abs_diff']:.9g}")
    print(f"reconstructed baseline tensors verified: {report['reconstructed_baseline_tensors_verified']}")
    print(f"reconstructed baseline max_abs_diff: {report['reconstructed_baseline_max_abs_diff']:.9g}")
    print(f"baseline -> P2Det tensors copied: {report['baseline_p2det_tensors_copied']}")
    print(f"baseline -> P2Det tensors verified: {report['baseline_p2det_tensors_verified']}")
    print(f"baseline -> P2Det max_abs_diff: {report['baseline_p2det_max_abs_diff']:.9g}")
    print(f"matched parameter count: {report['matched_parameter_count']}")
    print(f"total parameter count: {report['total_parameter_count']}")
    print(f"transfer ratio: {report['transfer_ratio']:.6%}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="gder_p45")
    parser.add_argument(
        "--check-transfer-only",
        action="store_true",
        help="build the selected YAML, run exact transfer checks, and exit without training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    yaml_path, experiment_name = VARIANTS[args.variant]
    trainer_class = (
        MonitoredP2SecondGenOBBTrainer
        if args.variant in SECOND_GEN_VARIANTS
        else MonitoredP2PromptOBBTrainer
    )
    if args.check_transfer_only:
        torch.manual_seed(0)
        model = YOLO(str(repo_path(yaml_path)), task="obb")
        report_checkpoint_transfer(
            migrate_p2det_from_canonical(
                model.model,
                canonical_source=CANONICAL_SOURCE,
                baseline_reference=BASELINE_REFERENCE,
            )
        )
        return None

    queue = resolve_queue_runtime(str(CANONICAL_SOURCE), default_device="4,5")

    if queue.resume:
        model = YOLO(queue.resume, task="obb")
    else:
        torch.manual_seed(0)
        model = YOLO(str(repo_path(yaml_path)), task="obb")
        transfer_report = migrate_p2det_from_canonical(
            model.model,
            canonical_source=CANONICAL_SOURCE,
            baseline_reference=BASELINE_REFERENCE,
        )
        report_checkpoint_transfer(transfer_report)
        # Model.train() uses this truthy checkpoint marker to pass the migrated
        # in-memory model into P2PromptOBBTrainer.get_model().
        model.ckpt = {"model": model.model, "p2det_migration": transfer_report}

    monitor = create_monitor(experiment_name)
    return monitor.run(
        model.train,
        trainer=trainer_class,
        data=str(repo_path("data/dronevehicle.yaml")),
        batch=queue.batch,
        epochs=100,
        imgsz=640,
        workers=8,
        device=queue.device,
        project="DroneVehicle_OBB_FusionTransfer",
        name=experiment_name,
        exist_ok=False,
        task="obb",
        resume=queue.resume or False,
    )


if __name__ == "__main__":
    main()
