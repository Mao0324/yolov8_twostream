# ProtoHGFNet 实验谱系

> 本文件由 Manifest 自动生成；模型结构以各实验链接的 YAML 为准。

生成时间：2026-08-31T10:19:15+08:00

## 架构树

```text
PHG-001 [created] hard top-k prototype hypergraph fusion on P3/P4/P5
├── PHG-002 [created] hard top-k prototype hypergraph fusion on P3/P4; ADD on P5
└── PHG-003 [created] soft-weighted top-k prototype hypergraph fusion on P3/P4/P5
```

## 架构与产物

| ID | 状态 | 父实验 | 唯一改动 | P3 | P4 | P5 | Epoch | Test mAP50-95 | 尝试次数 | YAML | 当前运行 |
|---|---|---|---|---|---|---|---:|---:|---:|---|---|
| PHG-001 | created | BL-001 | 仅将 baseline 的 P3/P4/P5 ADD 替换为 K=6、k=3 的低秩 ProtoHypergraphFusion；其余架构和训练配置不变。 | prototypes -> hard joint relation -> low-rank modulation -> global modality gate | prototypes -> hard joint relation -> low-rank modulation -> global modality gate | prototypes -> hard joint relation -> low-rank modulation -> global modality gate | 100/100 | 0.702 | 1 | [YAML](../../yaml/protohgf_hard_p345.yaml) | [Run](../../runs/DroneVehicle_OBB/train-labels=official-v1/protohgfnet/mainline/PHG-001__hard-p345/seed=000/attempt=01) |
| PHG-002 | created | PHG-001 | P3/P4 与 PHG-001 相同，P5 恢复 baseline ADD。 | hard prototype hypergraph fusion | hard prototype hypergraph fusion | baseline ADD | 100/100 | 0.701 | 1 | [YAML](../../yaml/protohgf_hard_p34.yaml) | [Run](../../runs/DroneVehicle_OBB/train-labels=official-v1/protohgfnet/mainline/PHG-002__hard-p34/seed=000/attempt=01) |
| PHG-003 | created | PHG-001 | 保持 K=6、k=3 和 P3/P4/P5 位置不变，仅将二值关系改为 top-k 内 softmax 权重。 | soft top-k prototype relation fusion | soft top-k prototype relation fusion | soft top-k prototype relation fusion | 87/100 | — | 1 | [YAML](../../yaml/protohgf_soft_p345.yaml) | [Run](../../runs/DroneVehicle_OBB/train-labels=official-v1/protohgfnet/mainline/PHG-003__soft-p345/seed=000/attempt=01) |

## 实验卡片

### PHG-001 · P3/P4/P5 硬 top-k 原型超图融合

- 状态：`created`，进度 `100/100`，Test mAP50-95 `0.702`。
- 架构：`hard top-k prototype hypergraph fusion on P3/P4/P5`。
- 假设：将像素级双模态交互压缩到少量目标语义原型并限制在硬 top-k 邻域，可减少 DroneVehicle 大面积背景对融合的干扰并提升测试 mAP50。
- 相对变化：仅将 baseline 的 P3/P4/P5 ADD 替换为 K=6、k=3 的低秩 ProtoHypergraphFusion；其余架构和训练配置不变。
- 文件：[Manifest](manifests/PHG-001.yaml) · [YAML](../../yaml/protohgf_hard_p345.yaml) · [旧 Train](../../train_dronevehicle_protohgf_hard_p345.py) · [迁移脚本](../../tools/make_protohgfnet_checkpoints_from_baseline.py)。

### PHG-002 · P3/P4 硬 top-k 原型超图，P5 保留 ADD

- 状态：`created`，进度 `100/100`，Test mAP50-95 `0.701`。
- 架构：`hard top-k prototype hypergraph fusion on P3/P4; ADD on P5`。
- 假设：原型关系对高分辨率小目标特征更有价值，P5 保留 ADD 可降低高层过平滑和额外计算。
- 相对变化：P3/P4 与 PHG-001 相同，P5 恢复 baseline ADD。
- 文件：[Manifest](manifests/PHG-002.yaml) · [YAML](../../yaml/protohgf_hard_p34.yaml) · [旧 Train](../../train_dronevehicle_protohgf_hard_p34.py) · [迁移脚本](../../tools/make_protohgfnet_checkpoints_from_baseline.py)。

### PHG-003 · P3/P4/P5 软 top-k 原型关系对照

- 状态：`created`，进度 `87/100`，Test mAP50-95 `—`。
- 架构：`soft-weighted top-k prototype hypergraph fusion on P3/P4/P5`。
- 假设：虽然论文中硬关系更优，但弱配准 DroneVehicle 上对 top-k 邻居保留相似度置信权重可能缓解错误边的突变影响。
- 相对变化：保持 K=6、k=3 和 P3/P4/P5 位置不变，仅将二值关系改为 top-k 内 softmax 权重。
- 文件：[Manifest](manifests/PHG-003.yaml) · [YAML](../../yaml/protohgf_soft_p345.yaml) · [旧 Train](../../train_dronevehicle_protohgf_soft_p345.py) · [迁移脚本](../../tools/make_protohgfnet_checkpoints_from_baseline.py)。
