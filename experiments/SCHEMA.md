# 实验清单约定

`experiments/` 是本仓库的实验注册层，不替代可执行的模型 YAML，也不移动历史训练目录。

## 单一信息源

- 模型结构的可执行真源：`files.model_yaml`。
- 实验身份、父子关系、文件链路和运行规则的真源：对应的 Manifest。
- `INDEX.md`、各论文族的 `README.md` 和 `registry.csv` 均由 Manifest 自动生成，不手工修改。
- 历史训练目录只作为只读证据，由 `legacy.run_dir` 链接。
- 通用入口创建的新运行通过 `provenance/resolved_manifest.yaml` 中的永久 ID 自动归档；`name2`、`name3` 等重跑不会变成新的实验。

## 必填字段

```yaml
schema_version: 1
id: DA-001
family: DarkAct
title: 人类可读标题
parent_id: null
lifecycle:
  status: auto
  recovery_confidence: confirmed
hypothesis: 本实验要验证的假设
change_from_parent: 相对父实验的唯一结构变化
architecture:
  summary: 一行结构摘要
  version: v1
  stages: [P3, P4, P5]
  heads: H2-4-8
  flow:
    P3: 模块顺序
    P4: 模块顺序
    P5: 模块顺序
files:
  model_yaml: yaml/model.yaml
  train_entrypoint: train_xxx.py
  migration_script: tools/migrate_xxx.py
  init_checkpoint: pre-pth/model.pt
  module_files: []
  integration_files: []
training:
  profile: experiments/profiles/dronevehicle_obb_100e.yaml
  launch_mode: checkpoint
  trainer_class: ultralytics.models.yolo.obb.train:OBBTrainer
  output:
    project: runs/DroneVehicle_OBB_FusionTransfer
    name: Paper_Module_Stages_Heads_Structure_Version
legacy:
  run_dir: DroneVehicle_OBB_FusionTransfer/旧目录
reports: []
notes: []
```

## 状态规则

`lifecycle.status: auto` 时，工具按当前产物动态判断：

1. 当前运行已跑满且有 `test_result/test.txt`：`tested`。
2. `results.csv` 已达到该运行自己 `args.yaml` 中的目标 epoch：`trained`。
3. `results.csv` 未跑满且最近仍更新：`running`。
4. `results.csv` 未跑满且长时间未更新：`interrupted`。
5. 没有任何运行目录：`planned`。

如自动状态无法覆盖早停等特殊情况，可以显式填写 `running`、`trained`、`tested`、`interrupted` 或 `archived`。

## 常用命令

```bash
python tools/experiment_registry.py validate
python tools/experiment_registry.py render
python tools/experiment_registry.py list --family DarkAct
python tools/experiment_registry.py show DA-005
python tools/train_experiment.py DA-005 --device 1,3 --dry-run
python tools/train_experiment.py DA-005 --device 1,3
```

当前通用入口只接受 `launch_mode: checkpoint`，直接从迁移后的 checkpoint 构建模型，保证多卡 DDP 子进程能够继承迁移权重。每次新运行会在输出目录的 `provenance/` 子目录保存解析后的 Manifest、模型 YAML、数据 YAML、相关源码快照、文件哈希和 Git 状态；即使数据集初始化失败，也会保留这份诊断证据。
