# CFGPNet 实验谱系

> 本文件由 Manifest 自动生成；模型结构以各实验链接的 YAML 为准。

生成时间：2026-08-18T09:44:50+08:00

## 架构树

```text
CF-001 [created] CrossCEA reliability exchange + max-selected lightweight ASAF on P3/P4/P5
├── CF-002 [created] CrossCEA reliability exchange + max-selected lightweight ASAF on P3/P4; ADD on P5
└── CF-003 [created] CrossCEA reliability exchange only on P3/P4/P5
```

## 架构与产物

| ID | 状态 | 父实验 | 唯一改动 | P3 | P4 | P5 | Epoch | Test mAP50-95 | 尝试次数 | YAML | 当前运行 |
|---|---|---|---|---|---|---|---:|---:|---:|---|---|
| CF-001 | created | BL-001 | 仅将 baseline 的 P3/P4/P5 ADD 替换为零初始化 CFGPCrossAttentionFusion；backbone、neck、OBB head 与训练配置不变。 | CrossCEA(groups=4) -> candidate max selection | CrossCEA(groups=8) -> candidate max selection | CrossCEA(groups=8) -> candidate max selection | 0/100 | — | 0 | [YAML](../../yaml/cfgpnet_crosscea_asaf_p345.yaml) | — |
| CF-002 | created | CF-001 | P3/P4 与 CF-001 相同，P5 恢复 baseline ADD。 | CrossCEA(groups=4) -> candidate max selection | CrossCEA(groups=8) -> candidate max selection | baseline ADD | 0/100 | — | 0 | [YAML](../../yaml/cfgpnet_crosscea_asaf_p34.yaml) | — |
| CF-003 | created | CF-001 | 保持 P3/P4/P5 CrossCEA，但移除候选选择与模态 MLP，仅对交互后的双流特征求和。 | CrossCEA(groups=4) -> sum | CrossCEA(groups=8) -> sum | CrossCEA(groups=8) -> sum | 0/100 | — | 0 | [YAML](../../yaml/cfgpnet_crosscea_only_p345.yaml) | — |

## 实验卡片

### CF-001 · P3/P4/P5 CrossCEA + 轻量 ASAF 选择聚合

- 状态：`created`，进度 `0/100`，Test mAP50-95 `—`。
- 架构：`CrossCEA reliability exchange + max-selected lightweight ASAF on P3/P4/P5`。
- 假设：交换跨模态空间可靠性并在三个候选融合响应中保留强响应，可降低背景冗余并提升 DroneVehicle 测试集 mAP50。
- 相对变化：仅将 baseline 的 P3/P4/P5 ADD 替换为零初始化 CFGPCrossAttentionFusion；backbone、neck、OBB head 与训练配置不变。
- 文件：[Manifest](manifests/CF-001.yaml) · [YAML](../../yaml/cfgpnet_crosscea_asaf_p345.yaml) · [旧 Train](../../train_dronevehicle_cfgpnet_crosscea_asaf_p345.py) · [迁移脚本](../../tools/make_cfgpnet_checkpoints.py)。

### CF-002 · P3/P4 CrossCEA + 轻量 ASAF，P5 保留 ADD

- 状态：`created`，进度 `0/100`，Test mAP50-95 `—`。
- 架构：`CrossCEA reliability exchange + max-selected lightweight ASAF on P3/P4; ADD on P5`。
- 假设：DroneVehicle 小目标更依赖 P3/P4，保留 P5 的简单 ADD 可减少高层语义过度门控并降低新增计算。
- 相对变化：P3/P4 与 CF-001 相同，P5 恢复 baseline ADD。
- 文件：[Manifest](manifests/CF-002.yaml) · [YAML](../../yaml/cfgpnet_crosscea_asaf_p34.yaml) · [旧 Train](../../train_dronevehicle_cfgpnet_crosscea_asaf_p34.py) · [迁移脚本](../../tools/make_cfgpnet_checkpoints.py)。

### CF-003 · P3/P4/P5 纯 CrossCEA 空间可靠性交换

- 状态：`created`，进度 `0/100`，Test mAP50-95 `—`。
- 架构：`CrossCEA reliability exchange only on P3/P4/P5`。
- 假设：若 ASAF 式强响应选择对无人机小目标造成噪声放大，仅保留跨模态可靠性交换会有更稳定的 mAP50。
- 相对变化：保持 P3/P4/P5 CrossCEA，但移除候选选择与模态 MLP，仅对交互后的双流特征求和。
- 文件：[Manifest](manifests/CF-003.yaml) · [YAML](../../yaml/cfgpnet_crosscea_only_p345.yaml) · [旧 Train](../../train_dronevehicle_cfgpnet_crosscea_only_p345.py) · [迁移脚本](../../tools/make_cfgpnet_checkpoints.py)。
