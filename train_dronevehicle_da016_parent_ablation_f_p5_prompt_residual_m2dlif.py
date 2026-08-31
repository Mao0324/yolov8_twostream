#!/usr/bin/env python3
"""F: E plus one zero-init bounded P5 Prompt residual.

P3/P4 and all Prompt supervision inherit E unchanged. P5 retains the original
DarkACT LAF as its base and adds only the bounded reliability residual.
"""

from pathlib import Path

from tools.train_da016_parent_ablation import run_da016_parent_ablation


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_da016_f_e_plus_p5_dualprompt_residual.pt"
EXPERIMENT_NAME = "DA016-F_EPlusP5ZeroInitBoundedResidual_M2DLIFLabels_v1"


def main():
    return run_da016_parent_ablation(
        checkpoint=CHECKPOINT,
        experiment_name=EXPERIMENT_NAME,
        prompt_supervision=True,
    )


if __name__ == "__main__":
    main()
