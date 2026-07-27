# DarkAct 目标显著性模块诊断与后续实验建议

## 结论

当前目标显著性分支主要问题不是 attention 数值稳定性，而是“监督目标与融合决策不匹配”。
在 DroneVehicle 测试集上，sqrt(HW) 与 L2+温度版本都只有 0.705 mAP50-95，
与 DA-007 LAF-only 的 0.705 持平；推理速度则由 257.96 FPS 降至约
188–189 FPS。继续只修改 attention 缩放或温度，预期收益有限。

| 架构 | mAP50 | mAP50-95 | FPS |
|---|---:|---:|---:|
| DA-007 LAF-only | 0.838 | 0.705 | 257.96 |
| Target Saliency FP32 | 0.837 | 0.704 | 134.96 |
| Target Saliency sqrt(HW) | 0.838 | 0.705 | 189.16 |
| Target Saliency L2 + learnable temperature | 0.837 | 0.705 | 187.65 |

## 具体不足

1. **填充 OBB 不是显著性真值。** 当前损失把整个旋转框内部设为 1。框内包含道路、
   阴影和车辆间隙，硬边界也会放大一两个特征网格的标注误差。
2. **RGB 与 IR 被强制使用同一目标。** 同一张二值 mask 扩展到两个模态通道，
   无法表达夜间 IR 更可靠、强光或热噪声下 RGB 更可靠等模态差异。
3. **P5 对小目标过粗。** 640 输入下 P5 仅 20×20；大量小车会压缩到极少网格，
   0.5-cell tolerance 反而可能比目标本身更大，监督近似量化噪声。
4. **拥挤目标被 union 合并。** 多个相邻框合为大片前景后，显著性分支失去实例中心、
   边界和目标间空隙信息。
5. **固定辅助损失容易与检测主任务竞争。** BCE 的正样本权重最高为 20，再叠加 Dice
   和固定 gain=0.05；训练早期不可靠的显著性梯度会直接影响融合特征。
6. **计算开销与有效信息不成比例。** full-C PaperLAF 和三层显著性分支明显降低 FPS，
   但其输出只表达类别无关前景，测试结果没有显示出相应精度收益。
7. **现有稳定化实验只治数值，不治目标。** FP32、sqrt(HW)、L2/temperature 改善了
   Softmax 的数值行为，却没有改变错误或过粗的监督语义，所以指标基本重合。

## 推荐的最小可解释改进

建议下一版只改监督，不同时改融合模块：

- 用旋转框内的 soft centerness/高斯热图替代硬填充 mask；框中心接近 1，边缘平滑降至 0，
  多框仍取逐点最大值。
- 辅助监督只保留 P3、P4，先移除 P5；stage weight 建议 `[1.0, 0.5]`。
- gain 从 0 warmup 到 0.02–0.03，避免训练初期压过检测损失。
- 对软标签使用连续权重 BCE/Focal + Soft Dice，不再用 `target > 0.5` 的硬正负划分。

之后再按单变量顺序验证：

1. **Soft target only**：保持 DA-010 架构，仅替换 mask/loss；
2. **P3/P4 only**：在上一步基础上去除 P5 辅助监督；
3. **Modality-aware target**：利用单模态教师置信度或可见度估计，为 RGB/IR 生成不同权重；
4. **轻量融合**：若监督确有收益，再把 full-C PaperLAF 换为 partial-channel LAF，并使用
   有界零中心残差门控。

训练时至少记录每层 gate 的均值、标准差、正负比例、饱和比例，以及前景/背景 gate
对比。如果 gate 长期接近常数或两模态统计完全相同，即使检测指标略涨，也不能说明模块
学到了有效的目标显著性。

## 代码对应位置

- 显著性 logits 与融合：`ultralytics/nn/modules/darkact_target_saliency.py`
- 二值 OBB raster、BCE+Dice 与固定 gain：`ultralytics/models/yolo/obb/target_saliency_train.py`
- sqrt(HW)/L2-temperature 稳定化：`ultralytics/nn/modules/darkact_target_saliency_stable_attention.py`
