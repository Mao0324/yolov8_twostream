#!/usr/bin/env python3
"""H50：在 B 与 H 的 Prompt loss 聚合之间做 50% 强度插值。

我们正在做的严格消融：
    L_prompt = 0.50 * L_B + 0.50 * L_class_winner

除了 ``p2_object_class_winner_balance_ratio=0.50``，模型结构、P3/P4
Prompt 辅助监督、teacher、topk=128、stage weights 和总辅助增益 0.10
全部保持 H/B 不变。H50 用于判断完全等权的 H 是否过强，同时尽量保留
truck、van、freight_car 的收益。
"""

from pathlib import Path

from tools.train_da016_parent_ablation import run_da016_parent_ablation


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_da016_h50_b_class_winner_balanced_aux_p34.pt"
EXPERIMENT_NAME = "DA016-H50_BPlusClassWinnerBalancedPromptAux-r050-P34_M2DLIFLabels_v1"


def main():
    # 使用共享入口以保留 Monitor 自动选卡、M2D-LIF 标注和多卡 DDP 的 .pt 规则。
    return run_da016_parent_ablation(
        checkpoint=CHECKPOINT,
        experiment_name=EXPERIMENT_NAME,
        prompt_supervision=True,
    )


if __name__ == "__main__":
    main()
