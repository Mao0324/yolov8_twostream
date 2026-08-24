# FLIR 数据集实验

此目录是 FLIR 数据集命名空间，只存放该数据集的实验 Manifest。方法归属由 Manifest 的 `family` 字段表达，方法谱系文档由注册表生成到 `experiments/<family>/README.md`。

| ID | 方法族 | 任务 | 尺度 | Manifest |
|---|---|---|---|---|
| FLIRDA-001 | DarkAct | HBB (`detect`, car/person/bicycle) | n | [FLIRDA-001.yaml](manifests/FLIRDA-001.yaml) |

FLIR 实验统一使用 `train_new.json/test_new.json` 三类口径，排除样本极少的 dog 类。
