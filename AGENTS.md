# Repository instructions for coding agents

## 双卡/多卡 DDP 预训练权重规则

本仓库固定使用经过本地修改的 Ultralytics YOLOv8.1.47。凡训练参数最终包含两个或更多 GPU
（例如 `device="0,1"`，包括远程 Monitor Agent 分配的多卡任务），必须遵守以下规则。

- 必须让自动 DDP 序列化的 `trainer.args.model` 指向实际 `.pt` 权重。
- 标准初始化必须写成 `model = YOLO(checkpoint, task="obb")`，然后调用 `model.train(...)`。
- 禁止在多卡任务中使用 `YOLO(yaml) + model.load(checkpoint)`。这种写法只更新父进程模型；
  自动 DDP 临时脚本仍会记录 YAML，子进程会通过 `weights=None` 随机初始化。
- 如果目标结构必须由新 YAML 构建并部分迁移其他权重，应先在独立步骤生成内嵌目标结构的初始化
  `.pt`，随后通过 `YOLO(init_checkpoint)` 启动多卡训练。不要假设父进程内存中的模型会传给子进程。
- 远程 Monitor Agent 脚本必须继续使用 `resolve_queue_runtime()` 提供的 checkpoint、device、batch
  和 resume 设置，并在 DDP 前通过 `tools.training_monitor` 设置仓库 `PYTHONPATH`。

修改或审查多卡训练脚本时必须验证：

1. 非 resume 情况下，传给 `YOLO(...)` 的是预期 `.pt`，不是模型 YAML。
2. `model.overrides["model"]` 和 `model.ckpt_path` 均解析为该 `.pt`。
3. DDP 临时文件 `~/.config/Ultralytics/DDP/_temp_*.py` 中的 `overrides['model']` 是该 `.pt`。
4. 权重加载日志必须出现在 `DDP:` 启动命令之后。DDP 之前的 `Transferred ...` 只代表父进程，
   `pretrained: true` 也不能证明子进程加载成功。

不得再用旧双卡训练的相似损失或 mAP 曲线反推预训练权重已加载，因为旧脚本可能含有相同的
DDP 丢权重问题。应以 DDP 子进程的模型路径和加载行为为准。

除非用户明确要求，不要为了验证本规则而启动、停止或删除真实训练任务及其输出；优先使用语法检查、
模型元数据检查和 DDP 参数检查。
