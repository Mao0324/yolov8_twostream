#!/usr/bin/env python3
"""Regenerate the DA016 experiment integration document from the checkpoint registry."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "DA016_EXPERIMENT_INTEGRATION.md"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def display_id(key):
    return {"b_detach": "B-Detach"}.get(key, key.upper())


def _tree_lines(variants):
    children = {"A": []}
    for key, spec in variants.items():
        label = display_id(key)
        parent = str(spec["parent"])
        children.setdefault(parent, []).append(label)
        children.setdefault(label, [])
    lines = ["A：原始 DA016"]

    def append(parent, prefix):
        siblings = children.get(parent, [])
        for index, child in enumerate(siblings):
            last = index == len(siblings) - 1
            connector = "└── " if last else "├── "
            key = next(k for k in variants if display_id(k) == child)
            lines.append(f"{prefix}{connector}{child}：{variants[key]['inherits']}")
            append(child, prefix + ("    " if last else "│   "))

    append("A", "")
    return lines


def render_experiment_map(variants):
    known_ids = {"A", *(display_id(key) for key in variants)}
    for key, spec in variants.items():
        required = {"yaml", "output", "name", "inherits", "parent", "prompt_detection_stages", "train_script"}
        missing = sorted(required - set(spec))
        if missing:
            raise ValueError(f"variant {key} is missing documentation fields: {missing}")
        if str(spec["parent"]) not in known_ids:
            raise ValueError(f"variant {key} has unknown parent {spec['parent']}")

    lines = [
        "# DA016 实验集成关系（自动生成）",
        "",
        "> 本文件由 `tools/update_da016_experiment_map.py` 根据",
        "> `tools/make_da016_prompt_ablation_checkpoints.py::VARIANTS` 自动生成。",
        "> 不要手工编辑本文件；以后新增实验时登记到 `VARIANTS`，构建 checkpoint 后会自动更新。",
        "",
        "## 实验树",
        "",
        "```text",
        *_tree_lines(variants),
        "```",
        "",
        "## 严格变量表",
        "",
        "| ID | 直接父实验 | 唯一新增机制 | Prompt 监督尺度 | Prompt 进入检测尺度 | Detach | Class×Winner 平衡 |",
        "|---|---|---|---|---|---:|---:|",
        "| A | — | 原始 DA016 | 无 | 无 | 否 | 0 |",
    ]
    layer_to_stage = {10: "P3", 15: "P4", 22: "P5"}
    for key, spec in variants.items():
        supervised = "/".join(layer_to_stage[index] for index in sorted(spec["stage_modules"]))
        detection = "/".join(spec["prompt_detection_stages"]) or "无"
        lines.append(
            f"| {display_id(key)} | {spec['parent']} | {spec['inherits']} | {supervised} | {detection} | "
            f"{'是' if spec.get('detach_prompt_features', False) else '否'} | "
            f"{float(spec.get('class_winner_balance_ratio', 0.0)):g} |"
        )

    lines.extend(
        [
            "",
            "## 文件索引",
            "",
            "| ID | YAML | 初始化 checkpoint | 训练入口 | checkpoint 状态 |",
            "|---|---|---|---|---|",
        ]
    )
    for key, spec in variants.items():
        yaml_path = Path(spec["yaml"])
        checkpoint = Path(spec["output"])
        lines.append(
            f"| {display_id(key)} | `{yaml_path.relative_to(ROOT)}` | `{checkpoint.relative_to(ROOT)}` | "
            f"`{spec['train_script']}` | {'已生成' if checkpoint.is_file() else '待生成'} |"
        )

    lines.extend(
        [
            "",
            "## B 分支的因果解释",
            "",
            "- B：Prompt 不进入融合，但 Prompt loss 会反传到 RGB/IR detector features，因此属于训练期多任务正则化。",
            "- B-Detach：Prompt head 仍受 teacher 监督，但输入 feature 被 detach；用于判断 B 的检测收益是否来自辅助梯度。",
            "- G：只在 B 上增加 P5 auxiliary Prompt。比较 `G-B` 得到纯 P5 辅助监督收益。",
            "- H25/H50/H75/H：检测结构与 B 完全相同，只把 Prompt object loss 按 25%/50%/75%/100%",
            "  插值到 `(GT class, teacher-winning modality)` 组等权聚合，检查完全平衡是否过强。",
            "- D/J/C 构成 P3/P4 residual 的二因素消融：D 仅 P3，J 仅 P4，C 同时 P3/P4。",
            "- C/E/F 属于 Prompt residual 分支，不用于替代上述 B 分支因果对照。",
            "",
            "二维交互应使用：",
            "",
            "```text",
            "P5 auxiliary 的纯收益           = G - B",
            "有 P3/P4 residual 时的 P5 收益 = E - C",
            "交互效应                        = (E-C) - (G-B)",
            "```",
            "",
            "P3/P4 residual 二因素应使用：",
            "",
            "```text",
            "P3 residual 单独收益 = D - B",
            "P4 residual 单独收益 = J - B",
            "P3×P4 交互效应       = C - D - J + B",
            "```",
            "",
            "## H 的平衡强度定义",
            "",
            "`p2_object_class_winner_balance_ratio=r` 不是额外 loss 倍率，而是两种聚合方式的插值：",
            "",
            "```text",
            "L_prompt = (1-r) * L_B + r * L_class×winner",
            "```",
            "",
            "其中 B/H25/H50/H75/H 分别为 `r=0/0.25/0.50/0.75/1.00`。H 为完全等权每个",
            "存在的 class×winner 组；四个平衡实验均保持 `p2_object_reliability_gain=0.10` 等其他参数不变。",
            "",
            "## 所有实验共同约束",
            "",
            "- 数据标注：`/media/biiteam/新加卷1/biiteam/MCONG/datasets/M2D-LIFlabels/`。",
            "- 两个冻结的 RGB-only/IR-only teacher，FP16/autocast，`temperature=0.2`，`topk=128`。",
            "- Prompt gate-direction supervision保持关闭；B/G/H25/H50/H75/H/B-Detach 的 Prompt 均不进入检测融合。",
            "- Monitor 训练使用 `resolve_queue_runtime()`；多卡 DDP 必须从嵌入结构的 `.pt` 启动。",
            "- auxiliary-only Prompt 在普通推理时自动跳过；训练和训练期 validation 仍完整计算 Prompt。",
            "",
            "## 更新方法",
            "",
            "新增实验时在 `VARIANTS` 中登记完整父节点、机制、YAML、checkpoint 和训练入口，然后执行：",
            "",
            "```bash",
            "python -B tools/update_da016_experiment_map.py",
            "```",
            "",
            "运行 `tools/make_da016_prompt_ablation_checkpoints.py` 构建任一实验后也会自动执行同一更新。",
            "",
        ]
    )
    return "\n".join(lines)


def update_experiment_map(variants):
    content = render_experiment_map(variants)
    OUTPUT.write_text(content, encoding="utf-8")
    return OUTPUT


def main():
    from tools.make_da016_prompt_ablation_checkpoints import VARIANTS

    output = update_experiment_map(VARIANTS)
    print(f"updated={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
