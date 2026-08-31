#!/usr/bin/env python3
"""J: B's dual-Prompt auxiliary supervision plus C's residual at P4 only.

Architecture inheritance:
    A -> B: add P3/P4 dual reliability Prompt auxiliary supervision.
    B -> J: add C's zero-init bounded Prompt residual only at P4.
    P3 remains exactly B (Prompt auxiliary-only); P5 remains original DarkACT LAF.

The embedded-architecture initialization checkpoint is required so automatic
multi-GPU DDP children reconstruct J and load the DA016 parent weights.
"""

from pathlib import Path

from tools.train_da016_parent_ablation import run_da016_parent_ablation


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_da016_j_dualprompt_p4_residual_p3_auxonly.pt"
EXPERIMENT_NAME = "DA016-J_DualReliabilityPrompt-P4ZeroInitBoundedResidual-P3AuxOnly_M2DLIFLabels_v1"


def main():
    return run_da016_parent_ablation(
        checkpoint=CHECKPOINT,
        experiment_name=EXPERIMENT_NAME,
        prompt_supervision=True,
    )


if __name__ == "__main__":
    main()
