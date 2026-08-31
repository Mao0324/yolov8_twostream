#!/usr/bin/env python3
"""G: B plus an auxiliary-only P5 reliability Prompt.

P3/P4 remain exactly B. P5 retains the original DarkACT LAF detection path.
No Prompt enters fusion at any scale.
"""

from pathlib import Path

from tools.train_da016_parent_ablation import run_da016_parent_ablation


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_da016_g_b_plus_p5_dualprompt_auxonly.pt"
EXPERIMENT_NAME = "DA016-G_BPlusP5DualReliabilityPromptAuxOnly_M2DLIFLabels_v1"


def main():
    return run_da016_parent_ablation(
        checkpoint=CHECKPOINT,
        experiment_name=EXPERIMENT_NAME,
        prompt_supervision=True,
    )


if __name__ == "__main__":
    main()
