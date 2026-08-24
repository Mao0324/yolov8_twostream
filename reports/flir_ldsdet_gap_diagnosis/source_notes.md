# FLIR 与 LDSDet 结果差距诊断：证据备忘

## 证据清单

- 论文 PDF：`LDSDet Long-Range Context and Dynamic Cross-Modal Alignment for Multimodal Object Detection Under Challenging Illumination.pdf`
  - 第 4.1 节：FLIR 划分为 4,129 训练对和 1,013 测试对，注释转换为 YOLO HBB。
  - 第 4.2 节：SGD，momentum 0.937，weight decay 5e-4，lr 1e-2 到 1e-4，200 epoch，batch 16，640×640。
  - Table 4：FLIR HBB Deyolo 为 44.0 mAP / 81.1 mAP50 / 40.1 AP75；LDSDet 为 45.3 / 82.9 / 41.6。
  - Table 5：完整 LDSDet 的消融配置报告 22.8M 参数。该表是消融实验环境，因此报告对“FLIR 精确参数量”保留边界说明。
- 当前测试：`runs/FLIR_Align_HBB_FusionTransfer/FLIR_DarkAct_SemanticDisagreementLAF_P34_R4_NoStaticMAA_scalen_HBB2/test_result/test.txt`
- 当前训练参数：同一 HBB2 运行目录下 `args.yaml`。
- 当前训练曲线：同一 HBB2 运行目录下 `results.csv`。
- 当前混淆矩阵：同一 `test_result` 目录下 `confusion_matrix_normalized.png`。
- 数据集口径：FLIR `coco_annotations` 下原始/去 dog JSON 及 `README_读我.txt`。
- 数据准备记录：当前 FLIR align manifest 与 RGB/IR split 列表。

## 核心计算

- 当前四类 mAP50：(89.4 + 81.1 + 50.1 + 8.75) / 4 = 57.34%，与报告的 57.4% 一致。
- 近似三类 mAP50：(89.4 + 81.1 + 50.1) / 3 = 73.53%。
- dog 口径对 mAP50 表面差距的解释比例：(73.53 - 57.34) / (82.9 - 57.34) = 63.3%。
- 当前四类 mAP：(60.0 + 40.3 + 18.8 + 4.25) / 4 = 30.84%，与报告 30.9% 的差异来自显示精度。
- 近似三类 mAP：(60.0 + 40.3 + 18.8) / 3 = 39.70%。
- dog 口径对 mAP 表面差距的解释比例：(39.70 - 30.84) / (45.3 - 30.84) = 61.3%。
- 当前近似优化更新：ceil(4129/64) × 100 = 6,500。
- 论文近似优化更新：ceil(4129/16) × 200 = 51,800。
- 参数量倍数：22.8M / 4.173067M = 5.46。

## 数据与结论边界

- 论文未在已检查正文中明确声明 FLIR 的类别数，因此“论文三类”是高优先级待验证假设，不是已确认事实。
- 近似三类 AP 是从四类模型的已打印 AP 重算；严格可比结果需要 nc=3 重训/重测。
- 全量图像尺寸扫描未完成；已验证的是划分数量、文件名成对和注释统计，不是每一对图像的像素级完整性。
- 混淆矩阵中的落入背景比例由图像刻度读取，用于定性/半定量定位，不作为精密显著性统计。
