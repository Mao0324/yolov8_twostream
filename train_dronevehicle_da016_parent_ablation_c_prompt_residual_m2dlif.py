#!/usr/bin/env python3
"""C: DA016 plus dual Prompt and a zero-init bounded residual modulation."""

from pathlib import Path

from tools.train_da016_parent_ablation import run_da016_parent_ablation


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_da016_c_dualprompt_zero_init_bounded_residual_p34.pt"
EXPERIMENT_NAME = "DA016-C_DualReliabilityPrompt-ZeroInitBoundedResidual-P34_M2DLIFLabels_v1"


def main():
    return run_da016_parent_ablation(
        checkpoint=CHECKPOINT,
        experiment_name=EXPERIMENT_NAME,
        prompt_supervision=True,
    )


if __name__ == "__main__":
    main()
