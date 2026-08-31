#!/usr/bin/env python3
"""B-Detach: causal control for B's auxiliary feature regularization.

Prompt heads and teacher loss remain B, but Prompt inputs are detached so the
auxiliary gradient cannot update the RGB/IR detector features.
"""

from pathlib import Path

from tools.train_da016_parent_ablation import run_da016_parent_ablation


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_da016_b_detach_prompt_gradient_control_p34.pt"
EXPERIMENT_NAME = "DA016-BDetach_PromptGradientCausalControl-P34_M2DLIFLabels_v1"


def main():
    return run_da016_parent_ablation(
        checkpoint=CHECKPOINT,
        experiment_name=EXPERIMENT_NAME,
        prompt_supervision=True,
    )


if __name__ == "__main__":
    main()
