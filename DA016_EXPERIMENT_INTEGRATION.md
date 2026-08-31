# DA016 实验集成关系（自动生成）

> 本文件由 `tools/update_da016_experiment_map.py` 根据
> `tools/make_da016_prompt_ablation_checkpoints.py::VARIANTS` 自动生成。
> 不要手工编辑本文件；以后新增实验时登记到 `VARIANTS`，构建 checkpoint 后会自动更新。

## 实验树

```text
A：原始 DA016
└── B：A + P3/P4 auxiliary Prompt supervision
    ├── B-Detach：B with Prompt inputs detached; auxiliary gradient cannot update detector features
    ├── C：B + zero-init bounded Prompt residual at P3/P4
    │   └── E：C + P5 auxiliary Prompt supervision; original P5 DarkACT LAF detection path
    │       └── F：E + zero-init bounded Prompt residual at P5
    ├── D：B + C's zero-init bounded Prompt residual at P3 only; P4 remains B
    ├── J：B + C's zero-init bounded Prompt residual at P4 only; P3 remains B
    ├── G：B + P5 auxiliary Prompt supervision; no Prompt enters detection
    ├── H25：B + 25% interpolation toward class-by-teacher-winner balanced Prompt loss
    ├── H50：B + 50% interpolation toward class-by-teacher-winner balanced Prompt loss
    ├── H75：B + 75% interpolation toward class-by-teacher-winner balanced Prompt loss
    └── H：B + equal aggregation over present (GT class, teacher winner) groups
```

## 严格变量表

| ID | 直接父实验 | 唯一新增机制 | Prompt 监督尺度 | Prompt 进入检测尺度 | Detach | Class×Winner 平衡 |
|---|---|---|---|---|---:|---:|
| A | — | 原始 DA016 | 无 | 无 | 否 | 0 |
| B | A | A + P3/P4 auxiliary Prompt supervision | P3/P4 | 无 | 否 | 0 |
| B-Detach | B | B with Prompt inputs detached; auxiliary gradient cannot update detector features | P3/P4 | 无 | 是 | 0 |
| C | B | B + zero-init bounded Prompt residual at P3/P4 | P3/P4 | P3/P4 | 否 | 0 |
| D | B | B + C's zero-init bounded Prompt residual at P3 only; P4 remains B | P3/P4 | P3 | 否 | 0 |
| J | B | B + C's zero-init bounded Prompt residual at P4 only; P3 remains B | P3/P4 | P4 | 否 | 0 |
| E | C | C + P5 auxiliary Prompt supervision; original P5 DarkACT LAF detection path | P3/P4/P5 | P3/P4 | 否 | 0 |
| F | E | E + zero-init bounded Prompt residual at P5 | P3/P4/P5 | P3/P4/P5 | 否 | 0 |
| G | B | B + P5 auxiliary Prompt supervision; no Prompt enters detection | P3/P4/P5 | 无 | 否 | 0 |
| H25 | B | B + 25% interpolation toward class-by-teacher-winner balanced Prompt loss | P3/P4 | 无 | 否 | 0.25 |
| H50 | B | B + 50% interpolation toward class-by-teacher-winner balanced Prompt loss | P3/P4 | 无 | 否 | 0.5 |
| H75 | B | B + 75% interpolation toward class-by-teacher-winner balanced Prompt loss | P3/P4 | 无 | 否 | 0.75 |
| H | B | B + equal aggregation over present (GT class, teacher winner) groups | P3/P4 | 无 | 否 | 1 |

## 文件索引

| ID | YAML | 初始化 checkpoint | 训练入口 | checkpoint 状态 |
|---|---|---|---|---|
| B | `yaml/yolov8s-DA016-B-DualPromptAuxOnly-P34.yaml` | `pre-pth/yolov8s-obb_twostream_da016_b_dualprompt_auxonly_p34.pt` | `train_dronevehicle_da016_parent_ablation_b_prompt_aux_m2dlif.py` | 已生成 |
| B-Detach | `yaml/yolov8s-DA016-BDetach-PromptGradientControl-P34.yaml` | `pre-pth/yolov8s-obb_twostream_da016_b_detach_prompt_gradient_control_p34.pt` | `train_dronevehicle_da016_b_detach_causal_control_m2dlif.py` | 已生成 |
| C | `yaml/yolov8s-DA016-C-DualPromptZeroInitBoundedResidual-P34.yaml` | `pre-pth/yolov8s-obb_twostream_da016_c_dualprompt_zero_init_bounded_residual_p34.pt` | `train_dronevehicle_da016_parent_ablation_c_prompt_residual_m2dlif.py` | 已生成 |
| D | `yaml/yolov8s-DA016-D-DualPromptP3Residual-P4AuxOnly.yaml` | `pre-pth/yolov8s-obb_twostream_da016_d_dualprompt_p3_residual_p4_auxonly.pt` | `train_dronevehicle_da016_parent_ablation_d_p3_prompt_residual_m2dlif.py` | 已生成 |
| J | `yaml/yolov8s-DA016-J-DualPromptP4Residual-P3AuxOnly.yaml` | `pre-pth/yolov8s-obb_twostream_da016_j_dualprompt_p4_residual_p3_auxonly.pt` | `train_dronevehicle_da016_parent_ablation_j_p4_prompt_residual_m2dlif.py` | 已生成 |
| E | `yaml/yolov8s-DA016-E-CPlusP5DualPromptAuxOnly.yaml` | `pre-pth/yolov8s-obb_twostream_da016_e_c_plus_p5_dualprompt_auxonly.pt` | `train_dronevehicle_da016_parent_ablation_e_p5_prompt_aux_m2dlif.py` | 已生成 |
| F | `yaml/yolov8s-DA016-F-CPlusP5DualPromptResidual.yaml` | `pre-pth/yolov8s-obb_twostream_da016_f_e_plus_p5_dualprompt_residual.pt` | `train_dronevehicle_da016_parent_ablation_f_p5_prompt_residual_m2dlif.py` | 已生成 |
| G | `yaml/yolov8s-DA016-G-BPlusP5DualPromptAuxOnly.yaml` | `pre-pth/yolov8s-obb_twostream_da016_g_b_plus_p5_dualprompt_auxonly.pt` | `train_dronevehicle_da016_g_b_plus_p5_prompt_aux_m2dlif.py` | 已生成 |
| H25 | `yaml/yolov8s-DA016-H25-BPlusClassWinnerBalancedAux-P34.yaml` | `pre-pth/yolov8s-obb_twostream_da016_h25_b_class_winner_balanced_aux_p34.pt` | `train_dronevehicle_da016_h25_class_winner_balanced_aux_m2dlif.py` | 已生成 |
| H50 | `yaml/yolov8s-DA016-H50-BPlusClassWinnerBalancedAux-P34.yaml` | `pre-pth/yolov8s-obb_twostream_da016_h50_b_class_winner_balanced_aux_p34.pt` | `train_dronevehicle_da016_h50_class_winner_balanced_aux_m2dlif.py` | 已生成 |
| H75 | `yaml/yolov8s-DA016-H75-BPlusClassWinnerBalancedAux-P34.yaml` | `pre-pth/yolov8s-obb_twostream_da016_h75_b_class_winner_balanced_aux_p34.pt` | `train_dronevehicle_da016_h75_class_winner_balanced_aux_m2dlif.py` | 已生成 |
| H | `yaml/yolov8s-DA016-H-BPlusClassWinnerBalancedAux-P34.yaml` | `pre-pth/yolov8s-obb_twostream_da016_h_b_class_winner_balanced_aux_p34.pt` | `train_dronevehicle_da016_h_b_class_winner_balanced_aux_m2dlif.py` | 已生成 |

## B 分支的因果解释

- B：Prompt 不进入融合，但 Prompt loss 会反传到 RGB/IR detector features，因此属于训练期多任务正则化。
- B-Detach：Prompt head 仍受 teacher 监督，但输入 feature 被 detach；用于判断 B 的检测收益是否来自辅助梯度。
- G：只在 B 上增加 P5 auxiliary Prompt。比较 `G-B` 得到纯 P5 辅助监督收益。
- H25/H50/H75/H：检测结构与 B 完全相同，只把 Prompt object loss 按 25%/50%/75%/100%
  插值到 `(GT class, teacher-winning modality)` 组等权聚合，检查完全平衡是否过强。
- D/J/C 构成 P3/P4 residual 的二因素消融：D 仅 P3，J 仅 P4，C 同时 P3/P4。
- C/E/F 属于 Prompt residual 分支，不用于替代上述 B 分支因果对照。

二维交互应使用：

```text
P5 auxiliary 的纯收益           = G - B
有 P3/P4 residual 时的 P5 收益 = E - C
交互效应                        = (E-C) - (G-B)
```

P3/P4 residual 二因素应使用：

```text
P3 residual 单独收益 = D - B
P4 residual 单独收益 = J - B
P3×P4 交互效应       = C - D - J + B
```

## H 的平衡强度定义

`p2_object_class_winner_balance_ratio=r` 不是额外 loss 倍率，而是两种聚合方式的插值：

```text
L_prompt = (1-r) * L_B + r * L_class×winner
```

其中 B/H25/H50/H75/H 分别为 `r=0/0.25/0.50/0.75/1.00`。H 为完全等权每个
存在的 class×winner 组；四个平衡实验均保持 `p2_object_reliability_gain=0.10` 等其他参数不变。

## 所有实验共同约束

- 数据标注：`/media/biiteam/新加卷1/biiteam/MCONG/datasets/M2D-LIFlabels/`。
- 两个冻结的 RGB-only/IR-only teacher，FP16/autocast，`temperature=0.2`，`topk=128`。
- Prompt gate-direction supervision保持关闭；B/G/H25/H50/H75/H/B-Detach 的 Prompt 均不进入检测融合。
- Monitor 训练使用 `resolve_queue_runtime()`；多卡 DDP 必须从嵌入结构的 `.pt` 启动。
- auxiliary-only Prompt 在普通推理时自动跳过；训练和训练期 validation 仍完整计算 Prompt。

## 更新方法

新增实验时在 `VARIANTS` 中登记完整父节点、机制、YAML、checkpoint 和训练入口，然后执行：

```bash
python -B tools/update_da016_experiment_map.py
```

运行 `tools/make_da016_prompt_ablation_checkpoints.py` 构建任一实验后也会自动执行同一更新。
