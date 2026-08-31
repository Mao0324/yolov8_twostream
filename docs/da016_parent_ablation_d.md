# DA016-parent 消融 D：继承关系

D 是 B 与 C 结果之后的单变量消融，不引入新的融合专家或新的 Prompt 定义。

```text
A：原始 DA016
└── B：A + P3/P4 双 Prompt 辅助监督
    ├── C：B + P3/P4 zero-init 有界 Prompt 残差
    └── D：B + 仅 P3 zero-init 有界 Prompt 残差
             └── P4 保持 B：Prompt 只参与辅助损失
```

## 逐层继承

| 层级 | A | B | C | D 的来源 |
|---|---|---|---|---|
| P3 | DA016 semantic-disagreement | A + Prompt 辅助 | B + Prompt 残差 | **继承 C 的 P3** |
| P4 | DA016 semantic-disagreement | A + Prompt 辅助 | B + Prompt 残差 | **继承 B 的 P4** |
| P5 | DarkACT LAF | 不变 | 不变 | **继承 A/B/C，不变** |

D 的检测特征为：

```text
P3: F_out = F_DA016 + alpha_P3 * (P_rgb - P_ir) * (F_rgb - F_ir) / 2
P4: F_out = F_DA016
P5: 原始 DarkACT LAF

alpha_P3 = 0.25 * tanh(raw_alpha_P3), raw_alpha_P3 初始化为 0
```

P3、P4 的 Prompt 仍由与 B/C 相同的两个冻结单模态 teacher 监督；`topk=128`、temperature、损失权重、warmup 和数据配置均不改变。P4 虽然预测并学习 Prompt，但 Prompt 不进入 P4 检测特征。

## 参数与初始化约束

- D 相对 B 只多一个 P3 标量 `raw_prompt_residual_gain`。
- D 相对 C 少一个 P4 标量 `raw_prompt_residual_gain`。
- P3 残差零初始化，因此 D 的初始检测输出必须与 A/B/C 完全一致。
- 初始化 checkpoint 直接嵌入 D 的目标结构；双卡 DDP 必须由该 `.pt` 启动。

## 文件

- 模型 YAML：`yaml/yolov8s-DA016-D-DualPromptP3Residual-P4AuxOnly.yaml`
- 初始化权重：`pre-pth/yolov8s-obb_twostream_da016_d_dualprompt_p3_residual_p4_auxonly.pt`
- 训练入口：`train_dronevehicle_da016_parent_ablation_d_p3_prompt_residual_m2dlif.py`
- 初始化构建：`python -B tools/make_da016_prompt_ablation_checkpoints.py --variant d`
- 静态审计：`python -B tools/check_da016_parent_ablations.py`
