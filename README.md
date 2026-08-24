# YOLOV8多模态目标检测
## 前言：环境配置要求
torch                        2.3.1\
torchvision                  0.18.1\
Python 3.8.19\
tensorrt                     8.5.3.1
```
pip install -e .
```
## 1. 数据集DroneVehicle数据集(可见光+热红外)
[DroneVehicle数据集下载地址](https://github.com/VisDrone/DroneVehicle) 

DroneVehicle 数据集由无人机采集的 56,878 张图像组成，其中一半是 RGB 图像，其余是红外图像。我们为这 5 个类别制作了丰富的注释，其中包含定向边界框。其中，汽车在 RGB 图像中有 389,779 个注释，在红外图像中有 428,086 个注释，卡车在 RGB 图像中有 22,123 个注释，在红外图像中有 25,960 个注释，公共汽车在 RGB 图像中有 15,333 个注释，在红外图像中有 16,590 个注释，厢式车在 RGB 图像中有 11,935 个注释，在红外图像中有 12,708 个注释，货车在 RGB 图像中有 13,400 个注释， 以及红外图像中的 17,173 个注释。\
在 DroneVehicle 中，为了在图像边界处对对象进行注释，我们在每张图像的顶部、底部、左侧和右侧设置了一个宽度为 100 像素的白色边框，因此下载的图像比例为 840 x 712。在训练我们的检测网络时，我们可以执行预处理以去除周围的白色边框并将图像比例更改为 640 x 512。
![DroneVehicle数据集图片](./dataset_sample.png)
## 2. 数据集文件格式(labeles: YOLO格式)
```
datasets
├── image
│   ├── test
│   ├── train
│   └── val
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val
```
images 保存的是可见光图片\
image 保存的是热红外图片\
labels 公用一个标签(一般来说使用红外图片标签)
注：DroneVehicle原始数据下载下来需要经过修改文件结构路径，裁剪图片，xml生成txt，以上过程由于图片较多，比较费时间且浪费cpu资源，干脆把处理好直接能下文训练的包含image，images，labels且分别含train，val，test数据集上传了，懒人必备，省的预处理过程，下载直接开始训练，百度网盘需要会员才能上传大文件，遂上传到夸克网盘了，
我用夸克网盘分享了「OBBCrop.zip」，点击链接即可保存。打开「夸克APP」，无需下载在线播放视频，畅享原画5倍速，支持电视投屏。
链接：https://pan.quark.cn/s/71a7d0fef724
提取码：vnep
## 3. 权重文件下载
<details open><summary>目标检测权重</summary>

| 模型                                                                                   | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------- | -------------------- | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640             | 37.3                 | 80.4                        | 0.99                             | 3.2            | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640             | 44.9                 | 128.4                       | 1.20                             | 11.2           | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640             | 50.2                 | 234.7                       | 1.83                             | 25.9           | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640             | 52.9                 | 375.2                       | 2.39                             | 43.7           | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640             | 53.9                 | 479.1                       | 3.53                             | 68.2           | 257.8             |

</details>

<details open><summary>旋转框检测权重</summary>

| 模型                                                                                           | 尺寸<br><sup>(像素) | mAP<sup>test<br>50 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------- | ------------------ | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n-obb](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-obb.pt) | 1024            | 78.0               | 204.77                      | 3.57                             | 3.1            | 23.3              |
| [YOLOv8s-obb](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-obb.pt) | 1024            | 79.5               | 424.88                      | 4.07                             | 11.4           | 76.3              |
| [YOLOv8m-obb](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-obb.pt) | 1024            | 80.5               | 763.48                      | 7.61                             | 26.4           | 208.6             |
| [YOLOv8l-obb](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-obb.pt) | 1024            | 80.7               | 1278.42                     | 11.83                            | 44.5           | 433.8             |
| [YOLOv8x-obb](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-obb.pt) | 1024            | 81.36              | 1759.10                     | 13.23                            | 69.5           | 676.7             |
</details>

## 4. 配置模型yaml文件和数据集yaml文件
分别在yaml文件夹和data文件下进行模型和数据集文件的配置
## 5. 训练
```
python train.py

windows直接运行可能报多进程错误，请运行python train_for_windows.py

```
注：windows直接运行可能报多进程错误，请运行python train_for_windows.py

### 双卡/多卡 DDP 加载预训练权重的重要说明

本仓库使用的 Ultralytics YOLOv8.1.47 在自动 DDP 模式下，只会把 Trainer 的参数序列化到
`~/.config/Ultralytics/DDP/_temp_*.py`，不会把父进程中已经修改过的模型对象直接传给 DDP 子进程。
因此，下面这种“先用 YAML 建模、再调用 `load()`”的写法只适用于单卡，**禁止用于双卡或多卡训练**：

```python
# 错误：父进程会加载权重，但 DDP 子进程仍可能从 YAML 随机初始化。
model = YOLO("yaml/baseline.yaml", task="obb")
model.load("pre-pth/yolov8s-obb_twostream_baseline.pt")
model.train(device="0,1", ...)
```

原因是 `model.load()` 虽然会更新父进程模型和 `self.ckpt`，却不会把
`self.overrides["model"]` 改为该 `.pt` 路径。自动 DDP 生成的临时脚本仍会得到
`model: yaml/baseline.yaml`，子进程随后执行相当于 `get_model(cfg=yaml, weights=None)` 的逻辑，
最终从随机权重开始训练。DDP 启动前出现 `Transferred ... items from pretrained weights`，只能证明
父进程加载成功，不能证明真正训练的 DDP 子进程加载成功。

双卡/多卡训练必须直接用 `.pt` 构造 YOLO：

```python
# 正确：DDP 临时参数中的 model 是 .pt，所有子进程都会加载该权重。
checkpoint = "pre-pth/yolov8s-obb_twostream_baseline.pt"
model = YOLO(checkpoint, task="obb")
model.train(device="0,1", ...)
```

如果新模型结构必须先由 YAML 构建并从其他模型部分迁移权重，应先生成一个内嵌目标结构和迁移后
参数的初始化 `.pt`，再另起训练进程通过 `YOLO(init_checkpoint)` 启动多卡训练，不能在同一个
自动 DDP 训练进程中使用 `YOLO(yaml) + model.load(pt)`。

启动后至少检查以下三项：

1. `args.yaml` 中的 `model` 必须是预期的 `.pt`，不能是 YAML。
2. `~/.config/Ultralytics/DDP/_temp_*.py` 的 `overrides['model']` 必须是预期的 `.pt`。
3. `DDP:` 命令输出之后，子进程应再次出现权重加载信息；仅检查 DDP 之前的 `Transferred ...` 不够。

`pretrained: true` 只是配置开关，也不能单独证明 DDP 子进程已经加载指定权重。

## 6. 测试
```
python test.py  
```
## 7. 打印模型信息
```
python info.py  
```
## 8. obb
推理：detect/obbDetect.py \
热图绘制: detect/hbbHeapmap.py\
onnx推理: detect/obbOnnxdetect.py
## 9. hbb
推理：detect/hbbDetect.py \
热图绘制: detect/hbbHeapmap.py
## 10. onnx
导出onnx格式文件:  export/exportOnnx.py \
判断onnx格式和pth格式的输出是否相同: export/campareOnnxWithPth.py \
删除导出engine文件的meta信息：export/delEngineMeta.py
## 11. 绘制框
绘制hbb真实框: plot/plotGt.py\
绘制obb真实框 plot/plotObb.py\
绘制文件下所有图片的hbb真实框 plot/plotAllGt.py
## 12. int8量化
从训练集随机挑选测试集作为校准集并进行int8量化: int8Calib/int8Claib.py \
删除int8量化的中间文件：int8Calib/delData.sh \
多次生成校准集量化： int8Calib/testClaib.sh
## 13. 检测结果
挑选最好的检测结果：findBestDetect/bestMatch.py \
比较原来模型和新的模型的检测结果： findBestDetect/campareOriginWithNew.py
## 14. 数据处理
裁剪droneVechile数据集的白边： dataProcess/cropData.py \
droneVechile数据集从Dota格式转OBB格式：dataProcess/dotaToOBB.py \
droneVechile数据集从xml格式转yolo格式： xmlToTxt.py


# Contributing 
Feel free to dive in! [Open an issue](https://github.com/mujianyu/TwoStream_Yolov8/issues) or submit PRs.


# Contributors
This project exists thanks to all the people who contribute.\
@ [hopesala](https://github.com/hopesala) <img src="https://avatars.githubusercontent.com/u/8850257?v=4" alt="hopesala" width="50" height="50">
