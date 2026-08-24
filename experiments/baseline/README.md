# Baseline 实验谱系

> 本文件由 Manifest 自动生成；模型结构以各实验链接的 YAML 为准。

生成时间：2026-08-23T16:11:14+08:00

## 架构树

```text
BL-001 [tested] RGB C2f + IR C2f_Faster -> P3/P4/P5 ADD
└── BL-002 [planned] RGB/IR -> P3/P4/P5 ADD -> BottleneckRefine
```

## 架构与产物

| ID | 数据集 | Task/Scale | 状态 | 父实验 | 唯一改动 | P3 | P4 | P5 | Epoch | Test mAP50-95 | 尝试次数 | YAML | 当前运行 |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|---|
| BL-001 | dronevehicle | OBB | tested | — | 首个登记的 RGB/IR 双流 OBB 基线实验。 | RGB C2f + IR C2f_Faster -> ADD | RGB C2f + IR C2f_Faster -> ADD | RGB SPPF + IR SPPF -> ADD | 100/100 | 0.670 | 1 | [YAML](../../yaml/baseline.yaml) | [Run](../../runs_baseline/train) |
| BL-002 | dronevehicle | OBB | planned | BL-001 | 在 P3、P4、P5 的 ADD 输出后分别增加一个 e=0.5、卷积核 1x1/3x3 的 Bottleneck 精炼块。 | RGB C2f + IR C2f_Faster -> ADD -> Bottleneck(e=0.5, k=1/3) | RGB C2f + IR C2f_Faster -> ADD -> Bottleneck(e=0.5, k=1/3) | RGB SPPF + IR SPPF -> ADD -> Bottleneck(e=0.5, k=1/3) | 0/100 | — | 0 | [YAML](../../yaml/yolov8s-baseline-ADD-P345-BottleneckRefine.yaml) | — |

## 实验卡片

### BL-001 · YOLOv8s RGB/IR 双流 ADD 基线

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.670`。
- 数据集：`dronevehicle`，任务/尺度：`OBB`。
- 架构：`RGB C2f + IR C2f_Faster -> P3/P4/P5 ADD`。
- 假设：以非对称 RGB C2f 与 IR C2f_Faster 双流骨干及逐尺度 ADD 融合作为后续融合模块的统一对照基线。
- 相对变化：首个登记的 RGB/IR 双流 OBB 基线实验。
- 文件：[Manifest](manifests/BL-001.yaml) · [YAML](../../yaml/baseline.yaml) · [Train](../../train_dronevehicle_baseline.py) · —。

### BL-002 · ADD 后 P3/P4/P5 BottleneckRefine 基线

- 状态：`planned`，进度 `0/100`，Test mAP50-95 `—`。
- 数据集：`dronevehicle`，任务/尺度：`OBB`。
- 架构：`RGB/IR -> P3/P4/P5 ADD -> BottleneckRefine`。
- 假设：在逐尺度 ADD 融合后增加轻量残差 Bottleneck，可在进入 FPN/PAN 前改善同尺度局部特征表达。
- 相对变化：在 P3、P4、P5 的 ADD 输出后分别增加一个 e=0.5、卷积核 1x1/3x3 的 Bottleneck 精炼块。
- 文件：[Manifest](manifests/BL-002.yaml) · [YAML](../../yaml/yolov8s-baseline-ADD-P345-BottleneckRefine.yaml) · [Train](../../train_dronevehicle_baseline_add_bottleneck_refine.py) · [迁移脚本](../../tools/make_twostream_obb_weights.py)。
