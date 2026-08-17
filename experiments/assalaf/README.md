# ASSALAF 实验谱系

> 本文件由 Manifest 自动生成；模型结构以各实验链接的 YAML 为准。

生成时间：2026-08-17T13:57:11+08:00

## 架构树

```text
ALF-001 [tested] P3/P4 LAF(global/local gates + ASSA cross) feedback; P5 original LAF; fused Refine P3/P4

ALF-002 [tested] P3 LAF(global/local gates + ASSA cross) feedback; P4/P5 original LAF; fused Refine P3/P4
```

## 架构与产物

| ID | 状态 | 父实验 | 唯一改动 | P3 | P4 | P5 | Epoch | Test mAP50-95 | 尝试次数 | YAML | 当前运行 |
|---|---|---|---|---|---|---|---:|---:|---:|---|---|
| ALF-001 | tested | DA-012 | P3/P4 移除 LAF 原 PartialCrossModalValue，以 PartialChannelASSAFusion 的双向残差作为唯一 cross interaction；P5 不变。 | C2f -> LAF reliability gates + ASSA cross replacement -> feedback -> fused Refine | C2f -> LAF reliability gates + ASSA cross replacement -> feedback -> fused Refine | C2f/SPPF -> original DA-012 LAF feedback | 100/100 | 0.703 | 1 | [YAML](../../yaml/yolov8s-ASSAReplaceLAFCross-P34-RefineP34-v1.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/ASSALAF_ReplaceLAFCross_P34_RefineP34_H2-4_R4-StaticDW3-NoFFN_v1) |
| ALF-002 | tested | DA-012 | 仅 P3 移除 LAF 原 PartialCrossModalValue 并以 PartialChannelASSAFusion 替换；P4/P5 完整保持 DA-012。 | C2f -> LAF reliability gates + ASSA cross replacement -> feedback -> fused Refine | C2f -> original DA-012 LAF feedback -> fused Refine | C2f/SPPF -> original DA-012 LAF feedback | 100/100 | 0.705 | 1 | [YAML](../../yaml/yolov8s-ASSAReplaceLAFCross-P3-RefineP34-v1.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/ASSALAF_ReplaceLAFCross_P3_RefineP34_H2_R4-StaticDW3-NoFFN_v1) |

## 实验卡片

### ALF-001 · P3/P4 ASSA 替换 LAF cross 分支

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.703`。
- 架构：`P3/P4 LAF(global/local gates + ASSA cross) feedback; P5 original LAF; fused Refine P3/P4`。
- 假设：ASSA 仅替换而不串联 LAF 的跨模态分支，可保留可靠性门控并避免重复特征重写。
- 相对变化：P3/P4 移除 LAF 原 PartialCrossModalValue，以 PartialChannelASSAFusion 的双向残差作为唯一 cross interaction；P5 不变。
- 文件：[Manifest](manifests/ALF-001.yaml) · [YAML](../../yaml/yolov8s-ASSAReplaceLAFCross-P34-RefineP34-v1.yaml) · [旧 Train](../../train_dronevehicle_assareplace_lafcross_p34.py) · [迁移脚本](../../tools/make_assareplace_laf_checkpoints.py)。

### ALF-002 · 仅 P3 ASSA 替换 LAF cross 分支

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.705`。
- 架构：`P3 LAF(global/local gates + ASSA cross) feedback; P4/P5 original LAF; fused Refine P3/P4`。
- 假设：仅在定位最敏感的 P3 使用 ASSA cross，P4/P5 保留 DA-012，可降低跨尺度过度交互并恢复 van/truck 表现。
- 相对变化：仅 P3 移除 LAF 原 PartialCrossModalValue 并以 PartialChannelASSAFusion 替换；P4/P5 完整保持 DA-012。
- 文件：[Manifest](manifests/ALF-002.yaml) · [YAML](../../yaml/yolov8s-ASSAReplaceLAFCross-P3-RefineP34-v1.yaml) · [旧 Train](../../train_dronevehicle_assareplace_lafcross_p3.py) · [迁移脚本](../../tools/make_assareplace_laf_checkpoints.py)。
