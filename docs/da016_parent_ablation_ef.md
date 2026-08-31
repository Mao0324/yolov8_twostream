# DA016-parent 消融 E/F：P5 Prompt

## 继承关系

```text
A：原始 DA016
└── B：A + P3/P4 双 Prompt 辅助监督
    └── C：B + P3/P4 zero-init 有界 Prompt residual
        └── E：C + P5 双 Prompt 辅助监督（不进入 P5 融合）
            └── F：E + P5 zero-init 有界 Prompt residual
```

P5 使用自己的包装器，保持 P3/P4 的语义分歧融合与 P5 原始 DarkACT LAF 的差异：

```text
LAFMergeFeedback2D
└── DarkACTDualPromptAuxLAFMergeFeedback2D
    └── DarkACTDualPromptResidualLAFMergeFeedback2D
```

- E 的 P5 包装器只计算并暴露 `P_rgb/P_ir`，检测前向直接调用原始
  `LAFMergeFeedback2D.forward()`。
- F 完整继承 E，只新增一个标量 `raw_prompt_residual_gain`，并在原始 P5
  融合结果后添加 residual。

## F 的 P5 检测路径

```text
F_P5 = F_original_P5_LAF
     + alpha_P5 * (P_rgb - P_ir) * (F_rgb - F_ir) / 2

alpha_P5 = 0.25 * tanh(raw_alpha_P5)
raw_alpha_P5 = 0（初始化）
```

这里 `P_rgb/P_ir` 是两个 Prompt logit 做 modality softmax 后的概率图。因此
`P_rgb-P_ir` 位于 `[-1, 1]`，而 `alpha_P5` 位于 `(-0.25, 0.25)`。初始
`alpha_P5=0`，所以 F 与 E、原始 DA016 的初始检测前向完全等价。

## 辅助监督与严格消融

E/F 均使用 P3/P4/P5 三尺度 Prompt loss，stage weights 为
`[1.0, 0.5, 0.25]`。normalizer 固定为 `1.5`，保持 C 原有 P3/P4 项的绝对
权重，P5 是额外的辅助项，不会因为加入 P5 而把 P3/P4 整体缩小。

训练数据仍由 `tools/dronevehicle_m2dlif.py` 临时替换为
`/media/biiteam/新加卷1/biiteam/MCONG/datasets/M2D-LIFlabels/` 的标注。
两个冻结单模态 teacher、`temperature=0.2`、`topk=128` 和其余可靠性监督
设置均不改变。

## 文件与命令

- E YAML：`yaml/yolov8s-DA016-E-CPlusP5DualPromptAuxOnly.yaml`
- F YAML：`yaml/yolov8s-DA016-F-CPlusP5DualPromptResidual.yaml`
- E 训练：`train_dronevehicle_da016_parent_ablation_e_p5_prompt_aux_m2dlif.py`
- F 训练：`train_dronevehicle_da016_parent_ablation_f_p5_prompt_residual_m2dlif.py`

```bash
python -B tools/make_da016_prompt_ablation_checkpoints.py --variant e
python -B tools/make_da016_prompt_ablation_checkpoints.py --variant f
python -B tools/check_da016_parent_ablations.py

python train_dronevehicle_da016_parent_ablation_e_p5_prompt_aux_m2dlif.py
python train_dronevehicle_da016_parent_ablation_f_p5_prompt_residual_m2dlif.py
```

训练脚本使用共享的 `resolve_queue_runtime()`，兼容 Monitor 的 checkpoint、
device、batch 和 resume 覆盖；双卡 DDP 始终从嵌入目标结构的 `.pt` 启动。
