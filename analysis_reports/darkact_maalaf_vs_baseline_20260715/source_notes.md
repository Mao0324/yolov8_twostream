# DarkAct MAA2D/LAFMerge 诊断证据记录

生成时间：2026-07-15（Asia/Shanghai）

## 比较口径

- baseline：`runs_baseline/train`
- MAA2D+LAFMerge：`DroneVehicle_OBB_FusionTransfer/DarkAct_MAA2DLAFMerge_P345_H2-4-8_StaticSaliency-R4-DW3D1-2-2_v1`
- 数据集：DroneVehicle 双模态 OBB，test split，8,980 张图像，159,614 个实例，5 类。
- 训练参数：两次均为 100 epochs、batch 16、imgsz 640、SGD、seed 0。`args.yaml` 除 model、device、project/name/save_dir 外一致。
- 注意：MAA2D+LAFMerge 的 `test.txt` 只保留三位小数，因此测试增量是近似值。

## 聚合性能

| 指标 | baseline | MAA2D+LAFMerge | 变化 |
|---|---:|---:|---:|
| test mAP50-95 | 0.669562 | 0.673 | +0.003438（约 +0.34 百分点） |
| test mAP50 | 0.812023 | 0.815 | +0.002977 |
| test Precision | 0.788669 | 0.788 | -0.000669 |
| test Recall | 0.781668 | 0.787 | +0.005332 |
| best val mAP50-95 | 0.70117（epoch 96） | 0.70309（epoch 97） | +0.00192 |
| last-10 val mAP50-95 均值 | 0.700711 | 0.702602 | +0.001891 |
| 参数量 | 14,734,722 | 15,702,371 | +6.57% |
| test inference latency | 3.8523 ms | 5.5 ms | +42.77% |
| 前向 FPS | 259.59 | 181.60 | -30.04% |

## 逐类 test mAP50-95

| class | instances | baseline | MAA2D+LAFMerge | delta |
|---|---:|---:|---:|---:|
| car | 137146 | 0.83585 | 0.837 | +0.00115 |
| truck | 8657 | 0.64594 | 0.654 | +0.00806 |
| bus | 4467 | 0.81501 | 0.818 | +0.00299 |
| van | 4282 | 0.50901 | 0.511 | +0.00199 |
| freight_car | 5062 | 0.54200 | 0.543 | +0.00100 |

## checkpoint 参数诊断

`best.pt` 中的关键学习参数：

| stage | beta_rgb | beta_ir | cross_scale |
|---|---:|---:|---:|
| P3 | 0.03128 | -0.55957 | -0.004696 |
| P4 | -0.00872 | -0.06317 | -0.003702 |
| P5 | -0.000249 | 0.04816 | 0.004135 |

以测试集中均匀抽取的 16 张图像做 6 通道前向，按与 dataloader 一致的 RGB/IR 分别 BGR-to-RGB 及 letterbox 处理：

| stage | MAA RGB 平均增益 | MAA IR 平均增益 | LAF RGB 平均权重 | RGB 亮度与 RGB 权重相关系数 |
|---|---:|---:|---:|---:|
| P3 | 1.0132 | 0.5521 | 0.6845 | 0.9202 |
| P4 | 0.9955 | 0.9625 | 0.7701 | 0.8640 |
| P5 | 0.9999 | 1.0215 | 0.7915 | 0.7940 |

LAF 的 RGB/IR 权重之和为 2，使零初始化时精确等价于 baseline ADD。抽样结果表明 LAF 已学到明显的亮度相关性；但 P3 MAA2D 同时将 IR 平均压到约 0.55，而后续 LAF 再偏向 IR，存在模块间对冲。样本数只有 16，该诊断用于判断机制是否激活，不用于估计整个数据集的精确权重分布。

## 论文与当前实现的结构对照

DarkAct 论文中：

- MAA 使用视频相邻帧差分、归一化与 sine mapping 构造 temporal saliency `E^S`；再用 max/avg pooling、MLP 和可学映射构造 spatial-tolerant query `E^Q`，执行 channel attention。
- LAF 显式使用 MAA 输出 `Y_r/Y_t` 与 Transformer stage feature `F_r/F_t` 构造 dynamic query；使用带 3x3 膨胀卷积与 Sigmoid 的 modality-specific keys；使用双向 cross-modal values，再经 hierarchical max/avg pooling+MLP 得到 `F^V`。
- LAF 位于每个 Transformer stage 之后，融合残差会进入下一 stage。
- 任务是双模态视频人体动作分类，指标为 Top-1/Top-5。论文 ablation 中 full/w/o MAA/w/o LAF/w/o both 的 Top-1 为 74.4/71.2/72.3/70.9，不能直接外推为 OBB mAP 收益。

当前实现中：

- MAA2D 是静态高通局部对比 `|F-AvgPool(F)|`，输出单通道空间 mask，没有 temporal saliency、spatial-tolerant query 与 channel attention。
- LAFMerge2D 是全局通道门控+局部空间门控+C/4 部分通道 cross-modal value 残差，不是论文 Eq.4–Eq.8 的 query/key/value 链路。
- MAA2D 放在 P3/P4/P5 的分支 C2f 之前；LAFMerge2D 仅在 neck 入口合并双流，融合结果不回馈后续 backbone stage。
- 这是面向单帧航拍车辆 OBB 的概念迁移，而不是 DarkAct-Net 的等价重现。

## 稳健性与未解问题

- 只有 seed 0 一次对比，+0.34 百分点可能落在训练随机性范围内；当前不能声称显著改进。
- 5 类中 truck 提升约 0.81 百分点，其余类只提升约 0.10–0.30 百分点，收益不均匀。
- car 占 137,146/159,614=85.9% 实例，但它的 mAP50-95 已约 0.836，存在数据不平衡与天花板效应。
- 当前抽样证明 LAF 与亮度相关，但还没有对全部 test set 按亮度分箱统计 mAP，因此不能确认收益是否集中在极暗样本。

## 图表设计记录

- 逐类收益图：比较/排名，竖向单序列 bar；x=class，y=mAP50-95 delta 百分点；用蓝色单色根，直接值标签，零起点。支持“收益主要来自 truck”的论断。
- 没有使用三个 stage 的权重相关系数绘图，因为只有 3 个类别，用表格更诚实。

