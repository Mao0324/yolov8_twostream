#!/usr/bin/env python3
"""H: B with class-by-teacher-winner balanced Prompt auxiliary supervision.

Architecture and detection path remain exactly B. Only the per-object Prompt
loss aggregation changes; no Prompt enters any fusion gate.
"""

from pathlib import Path

from tools.train_da016_parent_ablation import run_da016_parent_ablation


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_da016_h_b_class_winner_balanced_aux_p34.pt"
EXPERIMENT_NAME = "DA016-H_BPlusClassWinnerBalancedPromptAux-P34_M2DLIFLabels_v1"


def main():
    return run_da016_parent_ablation(
        checkpoint=CHECKPOINT,
        experiment_name=EXPERIMENT_NAME,
        prompt_supervision=True,
    )


if __name__ == "__main__":
    main()
