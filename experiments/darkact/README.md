# DarkAct 实验谱系

> 本文件由 Manifest 自动生成；模型结构以各实验链接的 YAML 为准。

生成时间：2026-07-25T18:43:36+08:00

## 架构树

```text
DA-001 [tested] MAA2D -> C2f/SPPF -> LAFMerge2D
└── DA-002 [tested] StaticMAA2D -> C2f/SPPF -> LAFMergeFeedback2D
    ├── DA-003 [tested] StaticMAA2D + stage features -> full-C PaperLAFMergeFeedback2D
    │   └── DA-008 [planned] C2f/SPPF -> supervised target saliency -> full-C PaperLAF feedback
    │       └── DA-009 [tested] C2f/SPPF -> FP32-safe supervised target saliency -> full-C PaperLAF feedback
    │           ├── DA-010 [tested] C2f/SPPF -> FP32 sqrt(HW) target saliency -> full-C PaperLAF feedback
    │           │   └── DA-014 [planned] P3/P4 soft-centerness supervision -> FP32 sqrt(HW) saliency -> full-C PaperLAF feedback
    │           └── DA-011 [tested] C2f/SPPF -> FP32 L2+temperature target saliency -> full-C PaperLAF feedback
    ├── DA-004 [tested] C2f/SPPF -> StaticMAA2D -> LAFMergeFeedback2D
    │   └── DA-005 [tested] C2f/SPPF -> StaticMAA2D -> per-modal ZeroInitRefine -> LAF feedback
    ├── DA-006 [trained] StaticMAA2D -> C2f/SPPF -> LAF feedback -> fused ZeroInitRefine
    ├── DA-007 [trained] C2f/SPPF -> LAFMergeFeedback2D
    │   └── DA-012 [planned] C2f/SPPF -> LAFMergeFeedback2D -> fused Refine(P3/P4 only)
    └── DA-013 [planned] zero-centered StaticMAA2D -> C2f/SPPF -> LAFMergeFeedback2D
```

## 架构与产物

| ID | 状态 | 父实验 | 唯一改动 | P3 | P4 | P5 | Epoch | Test mAP50-95 | 尝试次数 | YAML | 当前运行 |
|---|---|---|---|---|---|---|---:|---:|---:|---|---|
| DA-001 | tested | — | 首个 DarkAct 单帧 MAA2D 与 partial-channel LAF 迁移。 | MAA2D -> C2f/C2f_Faster -> LAFMerge2D | MAA2D -> C2f/C2f_Faster -> LAFMerge2D | MAA2D -> C2f/SPPF -> LAFMerge2D | 100/100 | 0.673 | 1 | [YAML](../../yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/DarkAct_MAA2DLAFMerge_P345_H2-4-8_StaticSaliency-R4-DW3D1-2-2_v1) |
| DA-002 | tested | DA-001 | 用 StaticMAA2D 替代 MAA2D，并将 LAF 修正反馈到后续 backbone。 | StaticMAA2D -> C2f/C2f_Faster -> LAFMergeFeedback2D | StaticMAA2D -> C2f/C2f_Faster -> LAFMergeFeedback2D | StaticMAA2D -> C2f/SPPF -> LAFMergeFeedback2D | 100/100 | 0.683 | 1 | [YAML](../../yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4-v2.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_MSK3-5-R4-PosBeta-DW3D1-2-2_v2) |
| DA-003 | tested | DA-002 | 将 partial-channel LAF 替换为完整通道 PaperLAFMergeFeedback2D。 | StaticMAA2D context + C2f stage features -> full-C PaperLAF feedback | StaticMAA2D context + C2f stage features -> full-C PaperLAF feedback | StaticMAA2D context + C2f/SPPF stage features -> full-C PaperLAF feedback | 100/100 | 0.675 | 1 | [YAML](../../yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4-v3.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/DarkAct_PaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-StaticMAA_v3) |
| DA-004 | tested | DA-002 | 将 StaticMAA2D 后移到各尺度 C2f/SPPF 之后。 | C2f/C2f_Faster -> StaticMAA2D -> LAFMergeFeedback2D | C2f/C2f_Faster -> StaticMAA2D -> LAFMergeFeedback2D | C2f/SPPF -> StaticMAA2D -> LAFMergeFeedback2D | 100/100 | 0.683 | 1 | [YAML](../../yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4-PostC2f-v1.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_PostC2f-MSK3-5-R4-PosBeta-DW3D1-2-2_v1) |
| DA-005 | tested | DA-004 | 在 MAA 后、LAF 前为 RGB/IR 分别加入零初始化残差精炼。 | C2f/C2f_Faster -> StaticMAA2D -> RGB/IR Refine -> LAF feedback | C2f/C2f_Faster -> StaticMAA2D -> RGB/IR Refine -> LAF feedback | C2f/SPPF -> StaticMAA2D -> RGB/IR Refine -> LAF feedback | 100/100 | 0.685 | 1 | [YAML](../../yaml/yolov8s-DarkAct-MAA2D-Refine-LAFMerge-P345-R4-PostC2f-v1.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/DarkAct_StaticMAA2DRefineLAFMergeFeedback_P345_H2-4-8_PostC2f-RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v1) |
| DA-006 | trained | DA-002 | 在各尺度 LAF 之后、进入 FPN/PAN 之前增加融合特征精炼。 | StaticMAA2D -> C2f -> LAF feedback -> fused Refine -> neck | StaticMAA2D -> C2f -> LAF feedback -> fused Refine -> neck | StaticMAA2D -> C2f/SPPF -> LAF feedback -> fused Refine -> neck | 100/100 | — | 1 | [YAML](../../yaml/yolov8s-DarkAct-MAA2D-LAFMerge-Refine-P345-R4-v1.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/DarkAct_StaticMAA2DLAFMergeFeedbackRefine_P345_H2-4-8_RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v1) |
| DA-007 | trained | DA-002 | 完全移除 P3/P4/P5 的 StaticMAA2D。 | C2f/C2f_Faster -> LAFMergeFeedback2D | C2f/C2f_Faster -> LAFMergeFeedback2D | C2f/SPPF -> LAFMergeFeedback2D | 100/100 | — | 1 | [YAML](../../yaml/yolov8s-DarkAct-LAFMergeFeedback2D-P345-R4-NoStaticMAA-v1.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/DarkAct_LAFMergeFeedback2D_P345_H2-4-8_R4-DW3D1-2-2_v1) |
| DA-008 | planned | DA-003 | 用 OBB mask 监督的单通道目标显著性替换无监督 StaticMAA 输出。 | C2f stage -> target saliency logits + OBB loss -> full-C PaperLAF feedback | C2f stage -> target saliency logits + OBB loss -> full-C PaperLAF feedback | C2f/SPPF stage -> target saliency logits + OBB loss -> full-C PaperLAF feedback | 0/100 | — | 0 | [YAML](../../yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-v1.yaml) | — |
| DA-009 | tested | DA-008 | 仅将 StaticMAAContext2D 内部 matmul/scale/softmax/matmul 改为 FP32，保留 sqrt(active_channels) 缩放与其他结构不变。 | C2f stage -> FP32-safe target saliency logits + OBB loss -> full-C PaperLAF feedback | C2f stage -> FP32-safe target saliency logits + OBB loss -> full-C PaperLAF feedback | C2f/SPPF stage -> FP32-safe target saliency logits + OBB loss -> full-C PaperLAF feedback | 100/100 | 0.704 | 1 | [YAML](../../yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-FP32Safe-v2.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn_v2) |
| DA-010 | tested | DA-009 | 仅将 FP32 目标显著性 attention 的除数从 sqrt(active_channels) 改为 sqrt(H*W)。 | C2f stage -> FP32 sqrt(HW) target saliency + OBB loss -> full-C PaperLAF feedback | C2f stage -> FP32 sqrt(HW) target saliency + OBB loss -> full-C PaperLAF feedback | C2f/SPPF stage -> FP32 sqrt(HW) target saliency + OBB loss -> full-C PaperLAF feedback | 100/100 | 0.705 | 1 | [YAML](../../yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-SqrtHW-v3.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn-SqrtHW_v3) |
| DA-011 | tested | DA-009 | Q/K 沿 H*W 做 L2 normalization，以每尺度、每模态一个可学习 tau 替代 sqrt(C) 缩放，V 保持未归一化。 | C2f stage -> FP32 L2+temperature target saliency + OBB loss -> full-C PaperLAF feedback | C2f stage -> FP32 L2+temperature target saliency + OBB loss -> full-C PaperLAF feedback | C2f/SPPF stage -> FP32 L2+temperature target saliency + OBB loss -> full-C PaperLAF feedback | 100/100 | 0.705 | 1 | [YAML](../../yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-L2Temp-v4.yaml) | [Run](../../DroneVehicle_OBB_FusionTransfer/DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn-L2Norm-LearnTemp0p2_v4) |
| DA-012 | planned | DA-007 | 仅在 P3/P4 的 LAF fused lateral 后增加零初始化残差 Refine，P5 和两条 backbone 反馈路径不变。 | C2f/C2f_Faster -> LAF feedback -> zero-init fused Refine -> FPN | C2f/C2f_Faster -> LAF feedback -> zero-init fused Refine -> FPN | C2f/SPPF -> LAF feedback -> FPN/PAN | 0/100 | — | 0 | [YAML](../../yaml/yolov8s-DarkAct-LAFMergeFeedback2D-Refine-P34-R4-NoStaticMAA-v1.yaml) | — |
| DA-013 | planned | DA-002 | 仅将 StaticMAA2D 的 1+beta*sigmoid(logit) 改为有界的 1+beta*tanh(logit)，位置、容量及 LAF 均不变。 | zero-centered bidirectional MAA -> C2f stage -> LAF feedback | zero-centered bidirectional MAA -> C2f stage -> LAF feedback | zero-centered bidirectional MAA -> C2f/SPPF stage -> LAF feedback | 0/100 | — | 0 | [YAML](../../yaml/yolov8s-DarkAct-ZeroCenteredMAA2D-LAFMerge-P345-R4-v1.yaml) | — |
| DA-014 | planned | DA-010 | 保持 sqrt(HW) 模块和融合结构不变；将硬 OBB mask 换为旋转 soft-centerness，仅监督 P3/P4，权重为 1.0/0.5，gain 在 10 epoch 内升至 0.025，并记录 gate 统计。 | C2f stage -> sqrt(HW) saliency + soft-centerness loss(weight 1.0) -> PaperLAF feedback | C2f stage -> sqrt(HW) saliency + soft-centerness loss(weight 0.5) -> PaperLAF feedback | C2f/SPPF stage -> sqrt(HW) saliency(no auxiliary loss) -> PaperLAF feedback | 0/100 | — | 0 | [YAML](../../yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P34-SoftCenterness-Warmup-v5.yaml) | — |

## 实验卡片

### DA-001 · MAA2D 与 LAFMerge2D 初始迁移版

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.673`。
- 架构：`MAA2D -> C2f/SPPF -> LAFMerge2D`。
- 假设：将 DarkAct 的显著性增强和局部自适应融合迁移到单帧 RGB/IR OBB 检测。
- 相对变化：首个 DarkAct 单帧 MAA2D 与 partial-channel LAF 迁移。
- 文件：[Manifest](manifests/DA-001.yaml) · [YAML](../../yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4.yaml) · — · —。

### DA-002 · StaticMAA2D 与 LAF 反馈版

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.683`。
- 架构：`StaticMAA2D -> C2f/SPPF -> LAFMergeFeedback2D`。
- 假设：论文式静态显著性和跨 stage 反馈比初始 MAA/LAF 迁移更适合单帧双流检测。
- 相对变化：用 StaticMAA2D 替代 MAA2D，并将 LAF 修正反馈到后续 backbone。
- 文件：[Manifest](manifests/DA-002.yaml) · [YAML](../../yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4-v2.yaml) · [旧 Train](../../train_dronevehicle_darkact_maalaf.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_v2.py)。

### DA-003 · Full-C 论文式 PaperLAF 版

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.675`。
- 架构：`StaticMAA2D + stage features -> full-C PaperLAFMergeFeedback2D`。
- 假设：完整通道的论文式 LAF 注意力能够比 C/4 通道近似提供更充分的跨模态交互。
- 相对变化：将 partial-channel LAF 替换为完整通道 PaperLAFMergeFeedback2D。
- 文件：[Manifest](manifests/DA-003.yaml) · [YAML](../../yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4-v3.yaml) · [旧 Train](../../train_dronevehicle_darkact_maalaf_v3.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_v3.py)。

### DA-004 · Post-C2f StaticMAA 与 LAF 反馈版

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.683`。
- 架构：`C2f/SPPF -> StaticMAA2D -> LAFMergeFeedback2D`。
- 假设：在完整 stage 特征上计算 StaticMAA，比在 stage 提炼前计算更符合检测语义。
- 相对变化：将 StaticMAA2D 后移到各尺度 C2f/SPPF 之后。
- 文件：[Manifest](manifests/DA-004.yaml) · [YAML](../../yaml/yolov8s-DarkAct-MAA2D-LAFMerge-P345-R4-PostC2f-v1.yaml) · [旧 Train](../../train_dronevehicle_darkact_maalaf_postc2f.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_postc2f.py)。

### DA-005 · Post-C2f 每模态融合前精炼版

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.685`。
- 架构：`C2f/SPPF -> StaticMAA2D -> per-modal ZeroInitRefine -> LAF feedback`。
- 假设：在 StaticMAA 后、LAF 前分别精炼 RGB/IR，可以提高进入融合模块的模态特征质量。
- 相对变化：在 MAA 后、LAF 前为 RGB/IR 分别加入零初始化残差精炼。
- 文件：[Manifest](manifests/DA-005.yaml) · [YAML](../../yaml/yolov8s-DarkAct-MAA2D-Refine-LAFMerge-P345-R4-PostC2f-v1.yaml) · [旧 Train](../../train_dronevehicle_darkact_maalaf_postc2f_refine.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_postc2f_refine.py)。

### DA-006 · LAF 融合后特征精炼版

- 状态：`trained`，进度 `100/100`，Test mAP50-95 `—`。
- 架构：`StaticMAA2D -> C2f/SPPF -> LAF feedback -> fused ZeroInitRefine`。
- 假设：对 LAF 融合结果做零初始化残差精炼，可以改善送入 FPN/PAN 的共享特征。
- 相对变化：在各尺度 LAF 之后、进入 FPN/PAN 之前增加融合特征精炼。
- 文件：[Manifest](manifests/DA-006.yaml) · [YAML](../../yaml/yolov8s-DarkAct-MAA2D-LAFMerge-Refine-P345-R4-v1.yaml) · [旧 Train](../../train_dronevehicle_darkact_maalaf_postlaf_refine.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_postlaf_refine.py)。

### DA-007 · 无 StaticMAA 的 LAF-only 消融版

- 状态：`trained`，进度 `100/100`，Test mAP50-95 `—`。
- 架构：`C2f/SPPF -> LAFMergeFeedback2D`。
- 假设：如果 StaticMAA 近似退化为常数增益，删除它后保留 LAF 反馈可能维持精度并降低开销。
- 相对变化：完全移除 P3/P4/P5 的 StaticMAA2D。
- 文件：[Manifest](manifests/DA-007.yaml) · [YAML](../../yaml/yolov8s-DarkAct-LAFMergeFeedback2D-P345-R4-NoStaticMAA-v1.yaml) · [旧 Train](../../train_dronevehicle_darkact_laffeedback_no_staticmaa.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_no_staticmaa.py)。

### DA-008 · OBB 目标显著性监督与 PaperLAF 版

- 状态：`planned`，进度 `0/100`，Test mAP50-95 `—`。
- 架构：`C2f/SPPF -> supervised target saliency -> full-C PaperLAF feedback`。
- 假设：用 OBB mask 直接监督显著性图，可以避免 StaticMAA 空间门控退化为近似常数。
- 相对变化：用 OBB mask 监督的单通道目标显著性替换无监督 StaticMAA 输出。
- 文件：[Manifest](manifests/DA-008.yaml) · [YAML](../../yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-v1.yaml) · [旧 Train](../../train_dronevehicle_darkact_target_saliency.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_target_saliency.py)。

### DA-009 · OBB 目标显著性 PaperLAF FP32 注意力安全版

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.704`。
- 架构：`C2f/SPPF -> FP32-safe supervised target saliency -> full-C PaperLAF feedback`。
- 假设：仅将目标显著性通道注意力的累加和 Softmax 改为 FP32，可防止 P3 在 AMP 下溢出为 NaN。
- 相对变化：仅将 StaticMAAContext2D 内部 matmul/scale/softmax/matmul 改为 FP32，保留 sqrt(active_channels) 缩放与其他结构不变。
- 文件：[Manifest](manifests/DA-009.yaml) · [YAML](../../yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-FP32Safe-v2.yaml) · [旧 Train](../../train_dronevehicle_darkact_target_saliency_fp32safe.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_target_saliency_fp32safe.py)。

### DA-010 · OBB 目标显著性 sqrt(HW) 通道注意力版

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.705`。
- 架构：`C2f/SPPF -> FP32 sqrt(HW) target saliency -> full-C PaperLAF feedback`。
- 假设：按实际点积长度 H*W 缩放通道注意力 logits，可减少 P3 Softmax 饱和并保留有效梯度。
- 相对变化：仅将 FP32 目标显著性 attention 的除数从 sqrt(active_channels) 改为 sqrt(H*W)。
- 文件：[Manifest](manifests/DA-010.yaml) · [YAML](../../yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-SqrtHW-v3.yaml) · [旧 Train](../../train_dronevehicle_darkact_target_saliency_sqrthw.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_target_saliency_sqrthw.py)。

### DA-011 · OBB 目标显著性 L2 归一化与可学习温度版

- 状态：`tested`，进度 `100/100`，Test mAP50-95 `0.705`。
- 架构：`C2f/SPPF -> FP32 L2+temperature target saliency -> full-C PaperLAF feedback`。
- 假设：用空间响应的余弦相似度和可学习温度限制 logits 幅值，可在不同分辨率下避免 Softmax 饱和。
- 相对变化：Q/K 沿 H*W 做 L2 normalization，以每尺度、每模态一个可学习 tau 替代 sqrt(C) 缩放，V 保持未归一化。
- 文件：[Manifest](manifests/DA-011.yaml) · [YAML](../../yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P345-L2Temp-v4.yaml) · [旧 Train](../../train_dronevehicle_darkact_target_saliency_l2temp.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_target_saliency_l2temp.py)。

### DA-012 · LAF-only 的 P3/P4 融合后精炼版

- 状态：`planned`，进度 `0/100`，Test mAP50-95 `—`。
- 架构：`C2f/SPPF -> LAFMergeFeedback2D -> fused Refine(P3/P4 only)`。
- 假设：DA-007 的融合特征可能需要轻量局部重整，但低分辨率 P5 不需要额外精炼；只精炼 P3/P4 可改善定位且控制容量。
- 相对变化：仅在 P3/P4 的 LAF fused lateral 后增加零初始化残差 Refine，P5 和两条 backbone 反馈路径不变。
- 文件：[Manifest](manifests/DA-012.yaml) · [YAML](../../yaml/yolov8s-DarkAct-LAFMergeFeedback2D-Refine-P34-R4-NoStaticMAA-v1.yaml) · [旧 Train](../../train_dronevehicle_darkact_laf_refine_p34.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_laf_refine_p34.py)。

### DA-013 · 零中心双向 StaticMAA 门控版

- 状态：`planned`，进度 `0/100`，Test mAP50-95 `—`。
- 架构：`zero-centered StaticMAA2D -> C2f/SPPF -> LAFMergeFeedback2D`。
- 假设：原 sigmoid MAA 只能放大且易退化为常数增益；零中心双向门控可按位置和通道增强目标、抑制背景。
- 相对变化：仅将 StaticMAA2D 的 1+beta*sigmoid(logit) 改为有界的 1+beta*tanh(logit)，位置、容量及 LAF 均不变。
- 文件：[Manifest](manifests/DA-013.yaml) · [YAML](../../yaml/yolov8s-DarkAct-ZeroCenteredMAA2D-LAFMerge-P345-R4-v1.yaml) · [旧 Train](../../train_dronevehicle_darkact_zero_centered_maa.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_zero_centered_maa.py)。

### DA-014 · P3/P4 旋转 Soft-Centerness 目标显著性版

- 状态：`planned`，进度 `0/100`，Test mAP50-95 `—`。
- 架构：`P3/P4 soft-centerness supervision -> FP32 sqrt(HW) saliency -> full-C PaperLAF feedback`。
- 假设：旋转软中心监督可减少硬 OBB 填充的背景污染和边界量化噪声，P3/P4-only 与 gain warmup 可降低辅助任务对检测主任务的干扰。
- 相对变化：保持 sqrt(HW) 模块和融合结构不变；将硬 OBB mask 换为旋转 soft-centerness，仅监督 P3/P4，权重为 1.0/0.5，gain 在 10 epoch 内升至 0.025，并记录 gate 统计。
- 文件：[Manifest](manifests/DA-014.yaml) · [YAML](../../yaml/yolov8s-DarkAct-TargetSaliency-PaperLAF-P34-SoftCenterness-Warmup-v5.yaml) · [旧 Train](../../train_dronevehicle_darkact_target_saliency_soft_centerness.py) · [迁移脚本](../../tools/make_twostream_obb_weights_darkact_target_saliency_soft_centerness.py)。
