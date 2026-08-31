#!/usr/bin/env python3
"""D: B's dual-Prompt auxiliary supervision plus C's residual at P3 only.

Architecture inheritance:
    A -> B: add P3/P4 dual reliability Prompt auxiliary supervision.
    B -> D: add C's zero-init bounded Prompt residual only at P3.
    P4 remains exactly B (Prompt auxiliary-only); P5 remains original DarkACT LAF.

The embedded-architecture initialization checkpoint is required so automatic
multi-GPU DDP children reconstruct D rather than falling back to a YAML model.
"""

from pathlib import Path

from tools.train_da016_parent_ablation import run_da016_parent_ablation


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_da016_d_dualprompt_p3_residual_p4_auxonly.pt"
EXPERIMENT_NAME = "DA016-D_DualReliabilityPrompt-P3ZeroInitBoundedResidual-P4AuxOnly_M2DLIFLabels_v1"


def main():
    return run_da016_parent_ablation(
        checkpoint=CHECKPOINT,
        experiment_name=EXPERIMENT_NAME,
        prompt_supervision=True,
    )


if __name__ == "__main__":
    main()
