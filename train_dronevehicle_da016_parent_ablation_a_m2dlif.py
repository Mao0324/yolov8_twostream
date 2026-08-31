#!/usr/bin/env python3
"""A: train the unmodified DA016 semantic-disagreement detector on M2D-LIF labels."""

from pathlib import Path

from tools.train_da016_parent_ablation import run_da016_parent_ablation


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_darkact_da016_semantic_disagreement_laf_p34.pt"
EXPERIMENT_NAME = "DA016-A_OriginalSemanticDisagreementLAF-P34_M2DLIFLabels_v1"


def main():
    return run_da016_parent_ablation(
        checkpoint=CHECKPOINT,
        experiment_name=EXPERIMENT_NAME,
        prompt_supervision=False,
    )


if __name__ == "__main__":
    main()
