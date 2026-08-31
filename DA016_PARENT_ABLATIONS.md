# DA016-parent 严格消融 A--F

> A--H、B-Detach 及后续实验的最新集成关系以自动生成的
> `DA016_EXPERIMENT_INTEGRATION.md` 为准。

所有实验都使用同一个 DA016 初始化 checkpoint、M2D-LIF train/val 标注、`seed=0`、
100 epochs、imgsz 640、默认总 batch 64。多卡训练始终通过嵌入结构的 `.pt` 启动。

| 组 | P3/P4 检测主路径 | Prompt 辅助监督 | Prompt 进入检测路径 | 参数量 |
|---|---|---:|---:|---:|
| A | 原始 `SemanticDisagreementLAFMergeFeedback2D` | 否 | 否 | 16,032,977 |
| B | 完整保留 A | 是 | 否 | 16,036,289 |
| C | 完整保留 A，再加 bounded residual | 是 | 是 | 16,036,291 |
| D | P3 同 C，P4 同 B，P5 同 A | 是 | 仅 P3 | 16,036,290 |
| E | P3/P4 同 C；P5 原始 LAF + Prompt aux | 是 | P3/P4 | 16,040,707 |
| F | 完整继承 E；P5 再加 bounded residual | 是 | P3/P4/P5 | 16,040,708 |

C 的唯一新检测路径为：

```text
F_out = F_DA016 + alpha * (P_rgb - P_ir) * (F_rgb - F_ir) / 2
alpha = 0.25 * tanh(raw_alpha)
raw_alpha 初始化为 0
```

`alpha` 可学习正负方向，没有符号保持约束；其绝对值始终小于 0.25。

## 构建 B--F 初始化 checkpoint

```bash
python -B tools/make_da016_prompt_ablation_checkpoints.py
```

## 审计初始化和 DDP 序列化

```bash
python -B tools/check_da016_parent_ablations.py
```

该审计要求：共享 DA016 tensor 完全相等、B--F 初始前向与 A 的最大差值为 0、
对应尺度 Prompt 初始化相同，并检查自动 DDP 临时脚本中的 `model` 是对应 `.pt`。

## 启动训练

```bash
python train_dronevehicle_da016_parent_ablation_a_m2dlif.py
python train_dronevehicle_da016_parent_ablation_b_prompt_aux_m2dlif.py
python train_dronevehicle_da016_parent_ablation_c_prompt_residual_m2dlif.py
python train_dronevehicle_da016_parent_ablation_d_p3_prompt_residual_m2dlif.py
python train_dronevehicle_da016_parent_ablation_e_p5_prompt_aux_m2dlif.py
python train_dronevehicle_da016_parent_ablation_f_p5_prompt_residual_m2dlif.py
```

手动运行时会自动选择两张空闲卡。Monitor Agent 设置 `CUDA_VISIBLE_DEVICES`、
`YOLO_QUEUE_BATCH` 或 `YOLO_QUEUE_RESUME_CHECKPOINT` 时，脚本使用
`resolve_queue_runtime()` 的分配结果。可用 `P2DET_DATALOADER_WORKERS` 调整每个
DDP rank 的 worker 数，默认 4。

B--F 自动发现最新的 RGB-only 和 IR-only teacher，也可显式指定：

```bash
export P2DET_V16_RGB_TEACHER=path/to/rgb_best.pt
export P2DET_V16_IR_TEACHER=path/to/ir_best.pt
```
