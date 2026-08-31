#!/usr/bin/env python3
"""E: C at P3/P4 plus auxiliary-only dual reliability Prompts at P5.

P5 inherits the original DarkACT LAF forward path. Its new Prompt heads receive
the same frozen-teacher object supervision as P3/P4 but do not affect detection.
"""

from pathlib import Path

from tools.train_da016_parent_ablation import run_da016_parent_ablation


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_da016_e_c_plus_p5_dualprompt_auxonly.pt"
EXPERIMENT_NAME = "DA016-E_DualReliabilityPrompt-P34Residual-P5AuxOnly_M2DLIFLabels_v1"


def main():
    return run_da016_parent_ablation(
        checkpoint=CHECKPOINT,
        experiment_name=EXPERIMENT_NAME,
        prompt_supervision=True,
    )


if __name__ == "__main__":
    main()
