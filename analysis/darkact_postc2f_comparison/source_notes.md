# DarkAct 实验对比源记录

## 分析范围

- 基准：`DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_MSK3-5-R4-PosBeta-DW3D1-2-2_v2`
- 后移：`DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_PostC2f-MSK3-5-R4-PosBeta-DW3D1-2-2_v1`
- 后移加精炼：`DarkAct_StaticMAA2DRefineLAFMergeFeedback_P345_H2-4-8_PostC2f-RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v1`

每个 run 均读取 `args.yaml`、`results.csv`、`test_result/test.txt` 和 `weights/best.pt`。
最佳验证 epoch 定义为 `results.csv` 中 `metrics/mAP50-95(B)` 最大的行。
测试差值使用 `100 * (candidate - V2)` 转为百分点。

## Gate 与权重探针

从 DroneVehicle test 集 8980 对 RGB/IR 中等间隔固定选取 32 对。两模态均按验证预处理
letterbox 到 640，BGR 转 RGB，并除以 255。三个 `best.pt` 使用完全相同的输入。

- Hook：P3/P4/P5 `StaticMAA2D.branches[RGB/IR]` 的 sigmoid gate。
- 空间 std：每个样本、每个通道在 HxW 上的 population std，再对样本和通道平均。
- 样本间 std：同一 C,H,W 坐标在 32 个样本上的 population std，再对坐标平均。
- `beta`、Refine `gamma`、LAF `cross_scale` 直接从 `best.pt` 提取。

## 图表地图

- 验证轨迹：折线图，8 个固定 epoch 锚点，比较收敛轨迹与平台。
- Test 分类 AP：分组柱状图，5 个类别乘 3 个架构，检查整体收益的类别来源。
- 汇总、学习权重与 gamma：精确查询表。

## 数据限制

- 三次实验都只有 seed 0。
- `test.txt` 使用 3 位有效数字保存指标。
- Gate 探针使用固定 32 对图像，用于判断是否存在空间/样本变化，不替代全数据集统计。
