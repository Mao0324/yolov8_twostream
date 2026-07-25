# DarkAct 三个实验架构比较与权重诊断

生成日期：2026-07-16（Asia/Shanghai）

## 结论

- 相对 `runs_baseline/train` 的 test mAP50-95=0.669562，V1/V2/V3 分别约为 0.673/0.683/0.675，即约 +0.344/+1.344/+0.544 个百分点。V2 是唯一个提升达到 1 个百分点且 5 个类都上升的方案。
- V2 的改进核心是分尺度模态门控和 stage-to-stage feedback：16 张 test 样本上，P3 平均权重为 RGB 0.318 / IR 1.682，P4 为 RGB 1.534 / IR 0.466，P5 为 RGB 1.331 / IR 0.669。P3 的 RGB 权重与 RGB 亮度相关系数为 0.916，说明低层融合确实会随可见光可靠性变化。
- V1 的 LAF 门控已激活，但 P3 MAA 学到 `beta_ir=-0.5596`，把 IR 平均增益压到 0.552；后面 LAF 又偏向 IR，形成对冲，所以只有小幅提升。
- V2/V3 的 StaticMAA `beta` 从初始 0.01 学到约 0.16，但 gate 的空间标准差只有 3e-6–1.3e-5，几乎等于常数 0.5。因此它实际更像统一的约 8% 增益，没有学出有意义的空间显著性。
- V1/V2 的 C/4 cross-value 分支基本没学起来：实测该分支输出 RMS 只是 baseline ADD 特征 RMS 的约 1e-6–1e-5；总修正几乎全部来自模态门控。
- V3 的 paper-style LAF 确实离开了零初始，但作用从 P3 到 P5 快速衰减：残差 RMS/原 ADD RMS 约为 0.209/0.137/0.034，输出 BN gamma RMS 约为 0.120/0.100/0.026。P5 已接近弱作用，却支付了完整 full-C 参数和计算。

## 比较口径

- 四次训练都是 DroneVehicle 同一数据划分、100 epochs、batch 64、imgsz 640、SGD、seed 0、deterministic=true。`args.yaml` 的实质差异是 model，device 和输出目录。
- test split 是 8,980 张图、159,614 个实例、5 类；所有 DarkAct 实验都用 `best.pt`、imgsz 640、batch 16 评估。
- baseline `test.txt` 保留了精确 `results_dict` 和逐类 `maps`；三个 DarkAct `test.txt` 只保留三位小数，所以其 test delta 是近似值，最大四舍五入误差约 ±0.05 个百分点。
- 参数量是未 fuse 的 checkpoint 参数量；GFLOPs 用当前 repo 的 6 通道 THOP 路径在 640x640 计算。FPS 来自各自 `test.txt`，受 GPU 运行时状态影响，只用于粗略效率对比。

## 聚合结果

| 实验 | best val mAP50-95 | test mAP50-95 | test delta | val-test gap | Params | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.70117 | 0.669562 | 0 | 3.161 pp | 14.735M | 37.04 | 259.59 |
| V1 | 0.70309 | 0.673 | +0.344 pp | 3.009 pp | 15.702M | 38.56 | 181.60 |
| V2 | 0.70874 | 0.683 | +1.344 pp | 2.574 pp | 16.698M | 40.37 | 200.16 |
| V3 | 0.71097 | 0.675 | +0.544 pp | 3.597 pp | 24.693M | 55.30 | 180.79 |

V3 的 best-val 最高，但 test 比 V2 低 0.8 个百分点，且 val-test gap 最大。这是“高容量 full-C 模块更贴合 val，但未转化为 test 泛化”的证据，不是严格的过拟合因果证明。

## V2 的逐类收益

| class | baseline | V2 | delta |
|---|---:|---:|---:|
| car | 0.83585 | 0.839 | +0.315 pp |
| truck | 0.64594 | 0.662 | +1.606 pp |
| bus | 0.81501 | 0.820 | +0.499 pp |
| van | 0.50901 | 0.531 | +2.199 pp |
| freight_car | 0.54200 | 0.564 | +2.200 pp |

收益并非由某一个类的偶然跳变支撑；五类全部上升，且难类 van/freight_car 的改善最大。

## 权重与激活探针

- 从每个 `best.pt` 中读取 EMA model，而非未平滑的原 model。
- 在 test 集排序列表中等间隔抽取 16 对 RGB/IR 图像，使用与 dataloader 一致的双模态 BGR-to-RGB 和 640 letterbox。
- `MAA gain` 是实际乘子 `1 + beta*gate` 的空间均值；`gate spatial std` 是 sigmoid gate 在空间/通道上的标准差。
- `LAF RGB/IR weight` 是门控权重的平均，两者之和为 2；`correction RMS ratio` 是相对原始 `RGB+IR` 特征 RMS 的修正量 RMS。
- V1/V2 额外独立重算 `cross_scale * cross(rgb,ir)`，确认其实际 RMS 贡献，避免只看标量 `cross_scale` 误判。

## 局限

- 每个架构只有 seed 0 一次，不能声称统计显著；V2 是当前单次实验胜者，不等于已证明在多种子下必然更优。
- 16 张激活探针足以判断分支是否近似退化，但不能代替全量 test 亮度分箱 mAP。
- V2 同时改了 StaticMAA、LAF 门控和 feedback 路径。权重证据强烈指向门控/feedback，但没有 `gate-only` / `no-feedback` 单变量 ablation 时，不能把 +1.344 pp 严格因果分解到某一子模块。

## 建议的下一组 ablation

1. V2 `gate-only + feedback`：移除 cross-value 分支，因为它的实际 RMS 贡献只有 1e-6–1e-5。
2. V2 `gate-only, no StaticMAA`：StaticMAA 现在基本退化成常数约 1.08 倍放大，验证删除它是否不降 mAP。
3. V2 `P3/P4 feedback, P5 ADD`：P5 不再使用高容量融合，优先保留对小目标最重要的 P3/P4。
4. V3 `P3/P4 only`：P5 output BN gamma RMS 只有 0.026，残差 RMS 只有 0.034，可先删除该层 full-C LAF。
5. 保留候选架构后跑至少 3 seeds，报告 mean±std，再做全量 test 亮度分箱与昼/夜分组 mAP。

