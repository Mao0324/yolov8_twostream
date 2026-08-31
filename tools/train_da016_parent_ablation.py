"""Shared Monitor/DDP-safe runner for registered DA016-parent ablations."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from tools.dronevehicle_m2dlif import (
    M2DLIF_LABEL_ROOT,
    install_trusted_torch_load,
    prepare_temporary_dataset,
)
from train_dronevehicle_p2det_dual_reliability_p34_v16_m2dlif import (
    PROJECT,
    require_teachers,
    resolved,
    validate_ddp_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def run_da016_parent_ablation(*, checkpoint, experiment_name, prompt_supervision):
    """Train one ablation while preserving queue overrides and DDP checkpoint identity."""

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_MODE"] = "DISABLED"
    if prompt_supervision:
        require_teachers()
    if not M2DLIF_LABEL_ROOT.is_dir():
        raise FileNotFoundError(f"M2D-LIF DroneVehicle labels not found: {M2DLIF_LABEL_ROOT}")

    from tools.queue_runtime import resolve_queue_runtime

    queue = resolve_queue_runtime(str(checkpoint), default_device="auto", default_batch=64)
    active_checkpoint = resolved(queue.checkpoint)
    if not active_checkpoint.is_file():
        if queue.resume:
            hint = ""
        elif prompt_supervision:
            hint = "\nBuild the registered Prompt experiment with: python -B tools/make_da016_prompt_ablation_checkpoints.py"
        else:
            hint = "\nBuild A first with: python -B tools/make_darkact_da016_checkpoint.py"
        raise FileNotFoundError(f"active checkpoint not found: {active_checkpoint}{hint}")

    prefix = "dronevehicle_m2dlif_da016_prompt_" if prompt_supervision else "dronevehicle_m2dlif_da016_a_"
    # The temporary labels/YAML must outlive all automatic DDP children.
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary_dir:
        data_yaml = prepare_temporary_dataset(Path(temporary_dir), label_root=M2DLIF_LABEL_ROOT)

        import torch
        # This import establishes repository PYTHONPATH before Ultralytics
        # serializes and launches its temporary DDP script.
        from tools.training_monitor import (
            MonitoredDA016PromptAblationOBBTrainer,
            MonitoredOBBTrainer,
            create_monitor,
        )
        from ultralytics import YOLO
        import ultralytics.nn.tasks  # noqa: F401

        install_trusted_torch_load(torch)
        # Required for multi-GPU correctness: the .pt path, not a YAML plus an
        # in-memory load(), is what every DDP child reconstructs.
        model = YOLO(queue.checkpoint, task="obb")
        validate_ddp_checkpoint(model, queue.checkpoint)
        trainer = MonitoredDA016PromptAblationOBBTrainer if prompt_supervision else MonitoredOBBTrainer
        monitor = create_monitor(experiment_name)
        result = monitor.run(
            model.train,
            trainer=trainer,
            data=str(data_yaml),
            batch=queue.batch,
            epochs=100,
            imgsz=640,
            workers=int(os.getenv("P2DET_DATALOADER_WORKERS", "4")),
            device=queue.device,
            project=str(PROJECT),
            name=experiment_name,
            exist_ok=False,
            task="obb",
            seed=0,
            deterministic=False,
            plots=False,
            resume=queue.resume or False,
        )

    print("Temporary M2D-LIF labels, manifests, YAML, and dataset caches removed.")
    return result


__all__ = ("run_da016_parent_ablation",)
