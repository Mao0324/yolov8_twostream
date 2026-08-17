# ASSANet 实验谱系

> 本文件由 Manifest 自动生成；模型结构以各实验链接的 YAML 为准。

生成时间：2026-08-17T13:57:11+08:00

## 架构树

```text
ASSA-001 [tested] P3/P4/P5 ASSAFusion(DynK3, FFN2) -> ADD
├── ASSA-002 [tested] P4 ASSAFusion(StaticDW K3, NoFFN); P3/P5 identity
└── ASSA-003 [tested] P3/P4 PartialChannelASSAFusion(R4, StaticDW K3, NoFFN)
```

## 架构与产物

| ID | 状态 | 父实验 | 唯一改动 | P3 | P4 | P5 | Epoch | Test mAP50-95 | 尝试次数 | YAML | 当前运行 |
|---|---|---|---|---|---|---|---:|---:|---:|---|---|
| ASSA-001 | tested | BL-001 | 将三个尺度进入共享 neck 前的直接模态融合替换为带 DynK3 和 FFN2 的 ASSAFusion 交互后 ADD。 | ASSAFusion(heads=2, DynK3, FFN2) -> ADD | ASSAFusion(heads=4, DynK3, FFN2) -> ADD | ASSAFusion(heads=8, DynK3, FFN2) -> ADD | 100/100 | 0.679 | 1 | [YAML](../../yaml/yolov8s-ASSAFusion.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/ASSANet_ASSAFusion_P345_H2-4-8_DynK3-FFN2_v1) |
| ASSA-002 | tested | ASSA-001 | 仅保留 P4、4 heads 的跨模态注意力，将动态局部投影改为 StaticDW K3，并删除 FFN；P3/P5 恢复恒等双流传递。 | RIFusion identity -> ADD | ASSAFusionStaticNoFFN(heads=4, StaticDW K3) -> ADD | RIFusion identity -> ADD | 100/100 | 0.672 | 1 | [YAML](../../yaml/yolov8s-ASSAFusion-P4-StaticDW-NoFFN.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/ASSANet_ASSAFusion_P4_H4_StaticDW-NoFFN_v1) |
| ASSA-003 | tested | ASSA-001 | P3/P4 改为 C/4 部分通道、StaticDW K3、无 FFN 的 ASSAFusion，移除 P5 注意力。 | PartialChannelASSAFusion(heads=2, R4, StaticDW K3, NoFFN) -> ADD | PartialChannelASSAFusion(heads=4, R4, StaticDW K3, NoFFN) -> ADD | RIFusion identity -> ADD | 100/100 | 0.676 | 1 | [YAML](../../yaml/yolov8s-PartialChannelASSAFusion-P34-R4-StaticDW-NoFFN.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/ASSANet_PartialChannelASSAFusion_P34_H2-4_R4-StaticDW-NoFFN_v1) |

## 实验卡片

### ASSA-001 · P3/P4/P5 动态 ASSAFusion

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.679`。
- 架构：`P3/P4/P5 ASSAFusion(DynK3, FFN2) -> ADD`。
- 假设：在 P3、P4、P5 使用双向跨模态稀疏通道注意力、动态局部投影和门控 FFN，可优于直接 ADD 的双流基线。
- 相对变化：将三个尺度进入共享 neck 前的直接模态融合替换为带 DynK3 和 FFN2 的 ASSAFusion 交互后 ADD。
- 文件：[Manifest](manifests/ASSA-001.yaml) · [YAML](../../yaml/yolov8s-ASSAFusion.yaml) · [旧 Train](../../train_dronevehicle_assafusion_p345.py) · —。

### ASSA-002 · 仅 P4 静态 DW 无 FFN 的 ASSAFusion

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.672`。
- 架构：`P4 ASSAFusion(StaticDW K3, NoFFN); P3/P5 identity`。
- 假设：ASSAFusion 的主要收益可由 P4 的静态深度卷积跨模态注意力保留，无需三个尺度的动态卷积和门控 FFN。
- 相对变化：仅保留 P4、4 heads 的跨模态注意力，将动态局部投影改为 StaticDW K3，并删除 FFN；P3/P5 恢复恒等双流传递。
- 文件：[Manifest](manifests/ASSA-002.yaml) · [YAML](../../yaml/yolov8s-ASSAFusion-P4-StaticDW-NoFFN.yaml) · [旧 Train](../../train_dronevehicle.py) · —。

### ASSA-003 · P3/P4 部分通道静态 ASSAFusion

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.676`。
- 架构：`P3/P4 PartialChannelASSAFusion(R4, StaticDW K3, NoFFN)`。
- 假设：仅让 P3/P4 的四分之一通道参与静态跨模态注意力，可在接近基线开销的条件下保留有效交互。
- 相对变化：P3/P4 改为 C/4 部分通道、StaticDW K3、无 FFN 的 ASSAFusion，移除 P5 注意力。
- 文件：[Manifest](manifests/ASSA-003.yaml) · [YAML](../../yaml/yolov8s-PartialChannelASSAFusion-P34-R4-StaticDW-NoFFN.yaml) · [旧 Train](../../train_dronevehicle2.py) · —。
