#!/usr/bin/env python3
"""
用途：
    使用 M2D-LIF 发布的 DroneVehicle train/val 标注，训练本仓库的双流 OBB baseline。
    模型结构、初始权重和训练参数与 ``train_dronevehicle_baseline.py`` 保持一致。
    训练前会将 M2D-LIF 类别 ID 临时映射到本仓库的类别顺序：
    0->0(car)、1->1(truck)、2->4(freight_car)、3->2(bus)、4->3(van)。

默认输入：
    模型结构与初始权重：
        <本仓库>/pre-pth/yolov8s-obb_twostream_baseline.pt
        该权重内嵌由 <本仓库>/yaml/baseline.yaml 构建的双流 baseline 结构。
    M2D-LIF 训练/验证标注：
        /media/biiteam/新加卷1/biiteam/MCONG/datasets/
        M2D-LIFlabels/DroneVehicle_train_val_labels/labels/{train,val}
    RGB 训练/验证图像：
        /media/biiteam/新加卷1/biiteam/MCONG/datasets/
        DroneVehicle_twostream_3/images/{train,val}
    IR 训练/验证图像：
        /media/biiteam/新加卷1/biiteam/MCONG/datasets/
        DroneVehicle_twostream_3/image/{train,val}

训练设置：
    epochs=100、imgsz=640、workers=8、task=obb。手动运行时默认 batch=64、device=3,5；
    由远程 Monitor Agent 启动时，使用队列分配的 batch、GPU 和断点续训权重。

永久输出：
    <本仓库>/runs/DroneVehicle_OBB/train-labels=m2dlif-v1/baseline/mainline/
    BL-001__add-p345/seed=<seed>/attempt=<attempt>/
    该目录保存训练权重、results.csv 和 Ultralytics 产生的常规结果。

临时输出：
    重映射标签、RGB/IR 数据清单、数据 YAML 和标签缓存仅写入
    ``/tmp/dronevehicle_m2dlif_baseline_*``。训练结束或异常退出后自动清理，
    原始图像和原始标注不会被修改。W&B 和 Comet 默认禁用；训练指标写入远程 Monitor。

运行：
    python -B train_dronevehicle_baseline_m2dlif.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tools.dronevehicle_m2dlif import (
    install_trusted_torch_load,
    prepare_temporary_dataset,
)


os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "pre-pth/yolov8s-obb_twostream_baseline.pt"
# 该历史标记会由 YoloExperimentMonitor 按注册表转换为上述 seed/attempt 布局。
PROJECT = ROOT / "runs_baseline"
EXPERIMENT_NAME = "Baseline_M2DLIFLabels_v1"


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"initial checkpoint not found: {CHECKPOINT}")

    # 临时目录在整个训练过程（包括 DDP 子进程）中保持存活。
    with tempfile.TemporaryDirectory(prefix="dronevehicle_m2dlif_baseline_") as temporary_dir:
        data_yaml = prepare_temporary_dataset(Path(temporary_dir))

        import torch
        from tools.queue_runtime import resolve_queue_runtime
        # 此导入会把仓库和 monitor 目录加入 PYTHONPATH，供 ~/.config/Ultralytics/DDP
        # 下启动的子进程导入本地 ultralytics 和监控训练器。
        from tools.training_monitor import MonitoredOBBTrainer, create_monitor
        from ultralytics import YOLO
        import ultralytics.nn.tasks  # noqa: F401  # 注册本仓库的双流自定义模块。

        install_trusted_torch_load(torch)
        queue = resolve_queue_runtime(str(CHECKPOINT), default_device="3,5", default_batch=64)
        monitor = create_monitor(EXPERIMENT_NAME)
        # 必须直接用 .pt 构造 YOLO。Ultralytics 自动 DDP 只会把 trainer.args
        # 序列化给子进程；若使用 YOLO(yaml) 后再 model.load(pt)，序列化结果仍是
        # YAML，DDP 子进程会以 weights=None 随机初始化，实际丢失预训练权重。
        model = YOLO(queue.checkpoint, task="obb")

        monitor.run(
            model.train,
            trainer=MonitoredOBBTrainer,
            data=str(data_yaml),
            batch=queue.batch,
            epochs=100,
            imgsz=640,
            workers=8,
            device=queue.device,
            project=str(PROJECT),
            name=EXPERIMENT_NAME,
            exist_ok=False,
            task="obb",
            resume=queue.resume or False,
        )

    print("Temporary M2D-LIF labels, manifests, YAML, and dataset caches removed.")


if __name__ == "__main__":
    main()
