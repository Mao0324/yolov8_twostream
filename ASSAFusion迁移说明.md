# ASSAFusion：ASSANet 到 RGB/IR 双流检测的迁移说明

## 参考实现

- 论文：*Adaptive Sparse Self-Attention for Efficient Image Super-resolution and Beyond*，TPAMI 2026，DOI `10.1109/TPAMI.2026.3670856`。
- 官方仓库：<https://github.com/sunny2109/ASSANet>，本次核对提交 `854e1fffe9e355ea50fbd546fe4baf955cacdda0`。
- 论文对应代码：`basicsr/archs/assanet_arch.py` 中的 `SparseSelfAttention`、`AttBlock`，以及 `basicsr/archs/idynamic_dwconv.py` 中的 `IDynamicDWConv`。

## 论文与官方代码的核心机制

ASSA 不是在空间 token 上构造 `HW × HW` 注意力，而是使用转置注意力构造每个 head 内的 `C × C` 矩阵。官方实现的处理顺序是：

1. LayerNorm 后用 `1×1 Conv` 生成 Q 和共享的 K/V 特征。
2. 使用逐像素动态 `3×3` 深度卷积，让每个通道、每个空间位置拥有自己的局部核，从而在进入非局部注意力前引入局部空间变化信息。
3. Q、K 沿空间维做 L2 归一化，再计算转置注意力。
4. 直接对注意力矩阵使用 ReLU，舍弃负相关项；不使用 Softmax，避免把不相关项也平滑成非零权重。
5. 注意力输出经过 `1×1 Conv` 和残差连接，再经过门控深度卷积 FFN 和第二个残差连接。

官方动态卷积依赖 CuPy 自定义 CUDA 算子，且 CPU 路径直接报错。本次在 `ultralytics/nn/modules/assa_fusion.py` 中按 3×3 的 9 个偏移逐项生成动态权重并累加，实现等价的逐像素动态深度卷积。该方式保留动态核行为，可在 CPU、CUDA 和常规 PyTorch 自动求导中运行，也避免 `unfold` 在 P3 大特征图上产生额外的 9 倍展开张量。

## 从自注意力到 RGB/IR 双向交互

输入仍遵循本仓库 `m.f == -3` 的约定，为 `[B, 2C, H, W]`，前一半是 RGB，后一半是 IR。模块内部执行两条对称路径：

```text
RGB_new = RGB + ASSA(Q_RGB, K_IR, V_IR)
IR_new  = IR  + ASSA(Q_IR,  K_RGB, V_RGB)
```

两条路径各自拥有 LayerNorm、局部动态 Q/KV 投影、输出投影和门控 FFN；每个方向还有独立的温度与残差缩放参数。模块最终重新拼接为 `[RGB_new, IR_new]`，由 `tasks.py` 再拆回两条流。因此：

- RGB 和 IR 都读取另一个模态的信息；
- 不生成单一替代特征，不会让任一模态支路中断；
- 每条流保留自己的原始特征残差，交互特征是受控增量；
- 进入共享检测颈部时，才用原有 `ADD` 汇合已经交互过的两条流。

## P3/P4/P5 放置评估

当前配置采用：

| 尺度 | 放置位置 | scale=s 的通道/头数 | 判断 |
| --- | --- | --- | --- |
| P3 | 两支 P3 特征提取结束后、P4 stride-2 卷积前 | 128 / 2 | 有利于小目标和局部边缘互补，但空间尺寸最大，动态卷积显存开销最高 |
| P4 | 两支 P4 特征提取结束后、P5 stride-2 卷积前 | 256 / 4 | 语义与分辨率较平衡，是最值得保留的单点融合位置 |
| P5 | 两支 C2f+SPPF 后、进入共享 neck 前 | 512 / 8 | 空间开销最低、跨模态语义对齐最强，但参数主要集中在此处 |

在“每一次下采样前让下一尺度同时接收跨模态信息”的设计目标下，P3+P4 是合理的；P5 已经没有下一次 backbone 下采样，所以放在末端语义提取后更准确。三个尺度全放适合作为完整方案，但不应跳过消融实验。推荐训练顺序是 `P4 only`、`P3+P4`、`P3+P4+P5`，并同时比较小目标 AP、全类别 mAP、显存和 FPS。

以当前 `scale=s`、640 输入的静态模型统计为例：

| 配置 | 参数量 | GFLOPs |
| --- | ---: | ---: |
| `yaml/baseline.yaml`（换为相同的 5 类 OBB 头统计） | 14,734,722 | 37.0 |
| `yaml/yolov8s-ASSAFusion.yaml` | 21,289,898 | 49.5 |
| 增量 | +6,555,176（+44.5%） | +12.5（+33.8%） |

因此，三尺度设计在功能上合适，但代价不小，尤其 P5 的高通道投影与 FFN。若完整方案速度或参数不满足要求，优先保留 P4，再根据小目标收益决定是否加入 P3；P5 应由实测增益决定，而不是默认必选。

## 使用配置

```python
from ultralytics import YOLO

model = YOLO("yaml/yolov8s-ASSAFusion.yaml")
```

配置中的三个融合层为：

```yaml
- [-3, 1, ASSAFusion, [256, 2]]
- [-3, 1, ASSAFusion, [512, 4]]
- [-3, 1, ASSAFusion, [1024, 8]]
```

第一个参数会随 YOLO 宽度系数缩放，`scale=s` 时分别构造 128、256、512 通道模块。
