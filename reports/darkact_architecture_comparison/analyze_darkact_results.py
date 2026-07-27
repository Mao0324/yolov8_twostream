#!/usr/bin/env python3
"""Parse and compare DarkAct-related DroneVehicle OBB test results."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "DroneVehicle_OBB_FusionTransfer"
OUT_DIR = Path(__file__).resolve().parent

CLASS_NAMES = ("car", "truck", "bus", "van", "freight_car")
RARE_CLASS_NAMES = ("truck", "bus", "van", "freight_car")

ARCHITECTURES = (
    (
        "TargetSaliencyPaperLAFMergeFeedback2D",
        "DA-009/010/011",
        "OBB-supervised target saliency + full-C PaperLAF",
    ),
    (
        "StaticMAA2DLAFMergeFeedbackRefine",
        "DA-006",
        "StaticMAA before stage + partial LAF + fused refine",
    ),
    (
        "StaticMAA2DRefineLAFMergeFeedback",
        "DA-005",
        "Post-C2f StaticMAA + per-modal refine + partial LAF",
    ),
    (
        "PostC2f",
        "DA-004",
        "Post-C2f StaticMAA + partial LAF feedback",
    ),
    (
        "StaticMAA2DLAFMergeFeedback",
        "DA-002",
        "StaticMAA before stage + partial LAF feedback",
    ),
    (
        "PaperLAFMergeFeedback2D",
        "DA-003",
        "StaticMAA + full-C PaperLAF feedback",
    ),
    (
        "LAFMergeFeedback2D",
        "DA-007",
        "LAF-only: partial-channel LAF feedback",
    ),
    (
        "MAA2DLAFMerge",
        "DA-001",
        "Initial static-saliency MAA2D + partial LAF",
    ),
)


def architecture_for(run_name: str) -> tuple[str, str]:
    for needle, experiment_id, label in ARCHITECTURES:
        if needle in run_name:
            if "L2Norm-LearnTemp" in run_name:
                return "DA-011", f"{label} (L2 + learnable temperature)"
            if "SqrtHW" in run_name:
                return "DA-010", f"{label} (sqrt(HW) scaling)"
            if "FP32Attn" in run_name and "TargetSaliency" in run_name:
                return "DA-009", f"{label} (FP32 attention)"
            return experiment_id, label
    raise ValueError(f"Unknown DarkAct run name: {run_name}")


def parse_test(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    run_name = path.parents[1].name
    experiment_id, architecture = architecture_for(run_name)
    result = {
        "run_name": run_name,
        "experiment_id": experiment_id,
        "architecture": architecture,
        "test_path": str(path.relative_to(ROOT)),
    }

    weights_match = re.search(r"^weights:\s*(.+)$", text, re.MULTILINE)
    if not weights_match:
        raise ValueError(f"Missing weights line in {path}")
    result["weights_path_recorded"] = weights_match.group(1).strip()
    test_device_match = re.search(r"^device:\s*(.+)$", text, re.MULTILINE)
    result["test_device"] = test_device_match.group(1).strip() if test_device_match else ""

    row_pattern = re.compile(
        r"^\s*(all|car|truck|bus|van|freight_car)\s+"
        r"(\d+)\s+(\d+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        re.MULTILINE,
    )
    rows = {}
    for match in row_pattern.finditer(text):
        name = match.group(1)
        rows[name] = {
            "images": int(match.group(2)),
            "instances": int(match.group(3)),
            "precision": float(match.group(4)),
            "recall": float(match.group(5)),
            "map50": float(match.group(6)),
            "map50_95": float(match.group(7)),
        }
    if set(rows) != {"all", *CLASS_NAMES}:
        raise ValueError(f"Unexpected metric rows in {path}: {sorted(rows)}")

    result.update(
        {
            "images": rows["all"]["images"],
            "instances": rows["all"]["instances"],
            "precision": rows["all"]["precision"],
            "recall": rows["all"]["recall"],
            "map50": rows["all"]["map50"],
            "map50_95": rows["all"]["map50_95"],
            "rare_map50_95": mean(rows[name]["map50_95"] for name in RARE_CLASS_NAMES),
        }
    )
    for name in CLASS_NAMES:
        result[f"{name}_map50_95"] = rows[name]["map50_95"]

    speed_match = re.search(
        r"Speed:\s*([0-9.]+)ms preprocess,\s*([0-9.]+)ms inference,.*?"
        r"([0-9.]+)ms postprocess per image",
        text,
    )
    fps_match = re.search(r"前向传播 FPS:\s*([0-9.]+)", text)
    if not speed_match or not fps_match:
        raise ValueError(f"Missing speed metrics in {path}")
    result.update(
        {
            "preprocess_ms": float(speed_match.group(1)),
            "inference_ms": float(speed_match.group(2)),
            "postprocess_ms": float(speed_match.group(3)),
            "forward_fps": float(fps_match.group(1)),
        }
    )

    args_path = path.parents[1] / "args.yaml"
    args_text = args_path.read_text(encoding="utf-8")
    model_match = re.search(r"^model:\s*(.+)$", args_text, re.MULTILINE)
    device_match = re.search(r"^device:\s*(.+)$", args_text, re.MULTILINE)
    seed_match = re.search(r"^seed:\s*(.+)$", args_text, re.MULTILINE)
    if not model_match:
        raise ValueError(f"Missing model in {args_path}")
    model_source = model_match.group(1).strip()
    if model_source.endswith((".yaml", ".yml")):
        initialization_cohort = "yaml_direct"
    elif "DroneVehicle_OBB_FusionTransfer" in model_source:
        initialization_cohort = "self_run_checkpoint"
    else:
        initialization_cohort = "migrated_pretrained_checkpoint"
    result.update(
        {
            "train_model_source": model_source,
            "initialization_cohort": initialization_cohort,
            "train_device": device_match.group(1).strip() if device_match else "",
            "seed": seed_match.group(1).strip() if seed_match else "",
        }
    )

    csv_path = path.parents[1] / "results.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    metric = "metrics/mAP50-95(B)"
    best_row = max(csv_rows, key=lambda row: float(row[metric]))
    result.update(
        {
            "epochs_completed": len(csv_rows),
            "best_val_epoch": int(float(best_row["epoch"])),
            "best_val_map50_95": float(best_row[metric]),
            "last_val_map50_95": float(csv_rows[-1][metric]),
        }
    )

    weight_path = path.parents[1] / "weights" / "best.pt"
    result["checkpoint_mib"] = weight_path.stat().st_size / (1024 * 1024)
    return result


def is_dominated(row: dict, rows: list[dict]) -> bool:
    """Return True when another run is at least as accurate and fast, and strictly better in one."""
    for other in rows:
        if other is row:
            continue
        no_worse = (
            other["map50_95"] >= row["map50_95"]
            and other["forward_fps"] >= row["forward_fps"]
        )
        strictly_better = (
            other["map50_95"] > row["map50_95"]
            or other["forward_fps"] > row["forward_fps"]
        )
        if no_worse and strictly_better:
            return True
    return False


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def short_label(row: dict) -> str:
    labels = {
        "DA-001": "DA-001 MAA2D+LAF",
        "DA-002": "DA-002 StaticMAA+LAF",
        "DA-003": "DA-003 Full-C PaperLAF",
        "DA-004": "DA-004 Post-C2f MAA",
        "DA-005": "DA-005 Per-modal refine",
        "DA-006": "DA-006 Fused refine",
        "DA-007": "DA-007 LAF-only",
        "DA-009": "DA-009 Target FP32",
        "DA-010": "DA-010 Target sqrt(HW)",
        "DA-011": "DA-011 Target L2+temp",
    }
    return labels[row["experiment_id"]]


def build_artifact(current: list[dict], strict_comparable: list[dict], summary: dict) -> dict:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    chart_rows = []
    for row in current:
        chart_rows.append(
            {
                "model": short_label(row),
                "experiment_id": row["experiment_id"],
                "architecture": row["architecture"],
                "map50_95": row["map50_95"],
                "map50": row["map50"],
                "rare_map50_95": round(row["rare_map50_95"], 5),
                "forward_fps": row["forward_fps"],
                "inference_ms": row["inference_ms"],
                "checkpoint_mib": round(row["checkpoint_mib"], 2),
                "initialization_cohort": row["initialization_cohort"],
                "pareto": "Pareto frontier" if row["pareto_accuracy_fps"] else "Dominated",
            }
        )

    source_results = {
        "id": "darkact_results",
        "label": "DarkAct experiment test outputs",
        "path": "reports/darkact_architecture_comparison/current_checkpoint_results.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": (
                "Parsed every DarkAct*/test_result/test.txt and joined each run's args.yaml "
                "and results.csv using analyze_darkact_results.py; this query reads the reviewed output."
            ),
            "sql": (
                "SELECT * FROM read_csv_auto("
                "'reports/darkact_architecture_comparison/current_checkpoint_results.csv', "
                "header = true)"
            ),
            "tables_used": [
                "reports/darkact_architecture_comparison/current_checkpoint_results.csv"
            ],
            "filters": [
                "Test split only",
                "imgsz=640",
                "Current table excludes YAML-direct runs",
                "Strict initialization comparisons use migrated pretrained checkpoints only",
            ],
            "metric_definitions": [
                "mAP50-95 is the all-class OBB mean average precision reported by the test evaluator across IoU 0.50:0.95.",
                "Forward FPS is copied from each stored test.txt and equals the reciprocal of reported inference time.",
                "Rare-class mAP50-95 is the unweighted mean across truck, bus, van, and freight_car.",
                "Pareto frontier means no other current run has both equal-or-higher mAP50-95 and equal-or-higher stored forward FPS, with one strictly higher.",
            ],
        },
    }
    source_paper = {
        "id": "darkact_paper",
        "label": "DarkAct CVPR 2026 paper",
        "path": "Tan_DarkAct_A_RGB-Thermal_Dataset_and_Fusion_Framework_for_Multimodal_Low-Light_CVPR_2026_paper.pdf",
        "query": {
            "engine": "local-document",
            "description": (
                "Reviewed Sections 4 and 5.4 for the original temporal MAA, LAF equations, "
                "and DarkAct-Net ablation results."
            ),
        },
    }
    source_impl = {
        "id": "darkact_implementation",
        "label": "Single-image DarkAct adaptation implementation",
        "path": "ultralytics/nn/modules/darkact_maalaf_v2.py",
        "query": {
            "engine": "local-code",
            "language": "python",
            "description": (
                "Reviewed StaticMAA2D, LAFMergeFeedback2D, full-channel PaperLAF, "
                "target-saliency attention, and auxiliary OBB mask loss implementations."
            ),
        },
    }
    sources = [source_results, source_paper, source_impl]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "DarkAct 双流 OBB 架构对比与优化建议",
            "description": (
                "基于 DroneVehicle 测试集结果、训练初始化口径、DarkAct 论文与当前实现的技术评估。"
            ),
            "generatedAt": generated_at,
            "cards": [],
            "charts": [
                {
                    "id": "accuracy_speed_tradeoff",
                    "title": "当前 DarkAct 实验的精度与前向吞吐",
                    "subtitle": "存档结果中只有 DA-006 与 DA-007 位于精度—速度 Pareto 前沿。",
                    "type": "scatter",
                    "dataset": "current_runs",
                    "sourceId": "darkact_results",
                    "encodings": {
                        "x": {
                            "field": "forward_fps",
                            "type": "quantitative",
                            "label": "前向 FPS",
                        },
                        "y": {
                            "field": "map50_95",
                            "type": "quantitative",
                            "label": "Test mAP50-95",
                        },
                        "label": {"field": "model", "type": "text", "label": "实验"},
                        "tooltip": [
                            {"field": "model", "type": "text", "label": "实验"},
                            {"field": "map50", "type": "quantitative", "label": "mAP50"},
                            {
                                "field": "rare_map50_95",
                                "type": "quantitative",
                                "label": "非 car 四类均值",
                            },
                            {
                                "field": "checkpoint_mib",
                                "type": "quantitative",
                                "label": "Checkpoint MiB",
                            },
                            {"field": "pareto", "type": "text", "label": "Pareto 状态"},
                        ],
                    },
                    "xAxisTitle": "Stored forward FPS",
                    "yAxisTitle": "Test mAP50-95",
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "current_results",
                    "title": "当前 checkpoint 启动实验的测试结果",
                    "subtitle": "10 个当前结果；DA-005 为 self-run checkpoint，其余 9 个为迁移预训练 checkpoint。",
                    "dataset": "current_runs",
                    "sourceId": "darkact_results",
                    "defaultSort": {"field": "map50_95", "direction": "desc"},
                    "density": "comfortable",
                    "layout": "full",
                    "columns": [
                        {"field": "model", "label": "实验", "type": "text"},
                        {"field": "map50_95", "label": "mAP50-95", "format": "number"},
                        {"field": "map50", "label": "mAP50", "format": "number"},
                        {
                            "field": "rare_map50_95",
                            "label": "非 car 四类均值",
                            "format": "number",
                        },
                        {"field": "forward_fps", "label": "前向 FPS", "format": "number"},
                        {"field": "inference_ms", "label": "推理 ms", "format": "number"},
                        {
                            "field": "checkpoint_mib",
                            "label": "Checkpoint MiB",
                            "format": "number",
                        },
                        {"field": "pareto", "label": "Pareto", "type": "text"},
                    ],
                }
            ],
            "sources": sources,
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# DarkAct 双流 OBB 架构对比与优化建议",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "darkact_results",
                    "body": (
                        "## 当前应保留两条主线，而不是只留一个“最高分模型”\n\n"
                        "- **精度主线：DA-006（StaticMAA + partial LAF + 融合后 Refine）**。"
                        "Test mAP50-95 为 **0.708**，前向 **117.76 FPS**；它与 DA-004 并列最高，"
                        "但存档速度更快，且非 car 四类均值略高。\n"
                        "- **效率主线：DA-007（LAF-only）**。Test mAP50-95 为 **0.705**，"
                        "前向 **257.96 FPS**；相对 DA-006 只低 **0.003**，存档吞吐约为其 **2.19 倍**。\n"
                        "- **结论置信度：有条件可用（Share with caveats）**。精度排序基于单 seed；"
                        "速度来自不同 GPU 编号的历史测试文件，适合判断量级，不足以当作严格硬件基准。"
                    ),
                },
                {
                    "id": "tradeoff_finding",
                    "type": "markdown",
                    "sourceId": "darkact_results",
                    "body": (
                        "## 精度—速度前沿只剩 DA-006 与 DA-007\n\n"
                        "横轴越右表示存档前向吞吐越高，纵轴越上表示 mAP50-95 越高。"
                        "DA-004 虽达到 0.708，但在现有速度记录上被 DA-006 支配；"
                        "Full-C PaperLAF、目标显著性与初始 MAA2D 路线都没有形成新的 Pareto 点。"
                    ),
                },
                {
                    "id": "tradeoff_chart",
                    "type": "chart",
                    "chartId": "accuracy_speed_tradeoff",
                    "layout": "full",
                },
                {
                    "id": "detail_interpretation",
                    "type": "markdown",
                    "sourceId": "darkact_results",
                    "body": (
                        "## 高分差异主要集中在少数类，整体提升仍很小\n\n"
                        "DA-004 的 `freight_car` mAP50-95 最高（0.598），DA-005 的 `truck` 最高（0.697），"
                        "DA-002 的 `bus` 最高（0.842），而 DA-006 在 `van` 上并列最高（0.569）。"
                        "因此若业务特别重视货车/货运车，可保留类定向候选；否则 DA-006 的整体平衡更好。"
                    ),
                },
                {
                    "id": "result_table",
                    "type": "table",
                    "tableId": "current_results",
                    "layout": "full",
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "sourceId": "darkact_results",
                    "body": (
                        "## 比较口径：同一测试集，但初始化必须分层\n\n"
                        "所有结果均为 DroneVehicle OBB `test`：8,980 张图、159,614 个实例、640 输入，"
                        "类别为 car、truck、bus、van、freight_car。当前共有 15 份 `test.txt`："
                        "其中 9 个是迁移预训练 checkpoint 启动，1 个是 self-run checkpoint，5 个旧实验是 YAML-direct。"
                        "四组可严格配对的迁移预训练实验相对 YAML-direct 平均提高 **0.0265 mAP50-95**，"
                        "明显大于当前架构间 **0.007** 的总跨度，所以旧、新结果不能混排。"
                    ),
                },
                {
                    "id": "method",
                    "type": "markdown",
                    "body": (
                        "## 方法：统一解析、分层比较与 Pareto 检查\n\n"
                        "分析脚本逐一解析 `test.txt` 的总体/分类指标和速度，联接 `args.yaml` 的模型来源、"
                        "`results.csv` 的最佳验证 epoch，并记录 checkpoint 大小。排名先按初始化来源分层，"
                        "再比较 mAP50-95、mAP50、非 car 四类均值与前向 FPS；"
                        "Pareto 判定要求不存在另一模型同时具备不低的精度和不低的吞吐。"
                    ),
                },
                {
                    "id": "paper_gap",
                    "type": "markdown",
                    "sourceId": "darkact_paper",
                    "body": (
                        "## 论文中的 MAA 优势不能直接外推到当前单帧 OBB\n\n"
                        "论文 MAA 使用相邻视频帧差分构造 temporal motion saliency；"
                        "完整 DarkAct-Net Top-1 为 74.4%，去掉 MAA 后为 71.2%。"
                        "当前 DroneVehicle 是配对单帧检测，只能以局部静态对比近似运动显著性。"
                        "因此当前 LAF-only 接近最高分并不与论文矛盾，而是说明被迁移掉的时间维度正是 MAA 的核心信息。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 限制与稳健性检查\n\n"
                        "- 所有当前实验都是 seed 0，尚无跨 seed 方差；0.001–0.003 的差值不能视为已确认优势。\n"
                        "- 测速文件使用 GPU 0、6、7，当前环境无法读取历史 GPU 型号；速度只作相对筛选。\n"
                        "- DA-005 的 `args.yaml` 从同一运行目录 `last.pt` 启动，不属于严格同初始化对照。\n"
                        "- DA-007 的目录名为 `_v12`，但内部 `args.yaml` 的 name/save_dir 为 `_v13`；"
                        "其测试权重路径仍指向 `_v12`，建议重测并修正实验登记。"
                    ),
                },
                {
                    "id": "recommendations",
                    "type": "markdown",
                    "body": (
                        "## 后续优化优先级\n\n"
                        "1. **先复现 Pareto 两点。** 对 DA-006、DA-007 各跑 3 个 seed，固定同一 GPU、batch、"
                        "warmup 与测速脚本，报告 mean±std；DA-004 仅在 freight_car 是关键类时加入。\n"
                        "2. **做最高 ROI 的缺失消融：DA-007 + fused Refine，完全不加 StaticMAA。** "
                        "这能直接判断 DA-006 的 0.003 是否来自融合后精炼，并有机会保留 LAF-only 的大部分速度。\n"
                        "3. **若继续 MAA，改成可抑制的零中心门控。** 当前 StaticMAA 使用正 `softplus(beta)` 与 "
                        "`sigmoid(gate)`，只能放大特征；可试 `1 + beta·tanh(logit)` 或 "
                        "`1 + beta·(2·sigmoid(logit)-1)`，保持 beta 零初始化，从而既增强目标也抑制噪声背景。\n"
                        "4. **暂停 Full-C PaperLAF 与现有目标显著性分支的扩展。** 它们约 47.6–47.9 MiB，"
                        "测试只到 0.704–0.705；若重启显著性路线，先改为 P3-only 的 rotated centerness/soft mask，"
                        "并降低固定矩形填充目标带来的背景与边界噪声。\n"
                        "5. **把优化资源转向弱类。** van 和 freight_car 明显低于 car/bus；优先尝试同步 RGB/IR 的"
                        "类均衡采样、配对 OBB Copy-Paste，以及按类别报告 AP 与混淆，而不是继续增加全通道注意力。\n"
                        "6. **调整迁移训练策略。** 当前是 SGD、3 epoch warmup、非 cosine；可对新融合/Refine 参数使用"
                        "更高学习率、backbone 使用 0.1× 学习率，或做 AdamW+cosine 小规模对照。"
                        "多数当前模型最佳验证点出现在 44–62 epoch，也应将 60–80 epoch 作为节省算力的候选。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 还需要回答的三个问题\n\n"
                        "- 最终目标更偏向最高 mAP、实时吞吐，还是 freight_car/van 的类定向能力？\n"
                        "- 统一 GPU 上的端到端延迟、显存峰值、参数量和 FLOPs 是否与存档 FPS 排序一致？\n"
                        "- 是否能按昼/夜、距离、遮挡与 RGB–IR 配准误差切分测试集，验证 LAF 的“光照自适应”是否真实发生？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "current_runs": chart_rows,
                "strict_comparable_runs": [
                    {
                        "model": short_label(row),
                        "map50_95": row["map50_95"],
                        "forward_fps": row["forward_fps"],
                    }
                    for row in strict_comparable
                ],
                "initialization_deltas": summary["initialization_deltas"],
            },
        },
        "sources": sources,
        "package_info": {
            "analysisScript": "reports/darkact_architecture_comparison/analyze_darkact_results.py",
            "summary": "reports/darkact_architecture_comparison/summary.json",
        },
    }
    return artifact


def main() -> None:
    test_paths = sorted(RUN_ROOT.glob("DarkAct*/test_result/test.txt"))
    results = [parse_test(path) for path in test_paths]
    current = [
        row
        for row in results
        if row["initialization_cohort"] != "yaml_direct"
    ]
    strict_comparable = [
        row
        for row in current
        if row["initialization_cohort"] == "migrated_pretrained_checkpoint"
    ]
    for row in current:
        row["pareto_accuracy_fps"] = not is_dominated(row, current)
    current.sort(key=lambda row: (-row["map50_95"], -row["forward_fps"]))

    old_by_id = {
        row["experiment_id"]: row
        for row in results
        if row["initialization_cohort"] == "yaml_direct"
    }
    paired_initialization_deltas = []
    for row in strict_comparable:
        old = old_by_id.get(row["experiment_id"])
        if old:
            paired_initialization_deltas.append(
                {
                    "experiment_id": row["experiment_id"],
                    "old_run": old["run_name"],
                    "current_run": row["run_name"],
                    "old_test_map50_95": old["map50_95"],
                    "current_test_map50_95": row["map50_95"],
                    "delta_map50_95": row["map50_95"] - old["map50_95"],
                }
            )

    class_best = {}
    for class_name in CLASS_NAMES:
        field = f"{class_name}_map50_95"
        best_value = max(row[field] for row in current)
        class_best[class_name] = {
            "map50_95": best_value,
            "runs": [row["run_name"] for row in current if row[field] == best_value],
        }

    summary = {
        "generated_from": {
            "test_glob": "DroneVehicle_OBB_FusionTransfer/DarkAct*/test_result/test.txt",
            "paper": "Tan_DarkAct_A_RGB-Thermal_Dataset_and_Fusion_Framework_for_Multimodal_Low-Light_CVPR_2026_paper.pdf",
        },
        "test_population": {
            "images": current[0]["images"],
            "instances": current[0]["instances"],
            "classes": list(CLASS_NAMES),
        },
        "counts": {
            "all_test_files": len(results),
            "current_checkpoint_runs": len(current),
            "strict_comparable_pretrained_runs": len(strict_comparable),
            "self_run_checkpoint_runs": sum(
                row["initialization_cohort"] == "self_run_checkpoint" for row in results
            ),
            "yaml_direct_runs": sum(
                row["initialization_cohort"] == "yaml_direct" for row in results
            ),
        },
        "best_accuracy": [
            {
                "run_name": row["run_name"],
                "experiment_id": row["experiment_id"],
                "map50_95": row["map50_95"],
                "map50": row["map50"],
                "forward_fps": row["forward_fps"],
            }
            for row in current
            if row["map50_95"] == max(item["map50_95"] for item in current)
        ],
        "pareto_frontier": [
            {
                "run_name": row["run_name"],
                "experiment_id": row["experiment_id"],
                "map50_95": row["map50_95"],
                "forward_fps": row["forward_fps"],
            }
            for row in current
            if row["pareto_accuracy_fps"]
        ],
        "class_best": class_best,
        "initialization_deltas": paired_initialization_deltas,
        "initialization_delta_mean": mean(
            row["delta_map50_95"] for row in paired_initialization_deltas
        ),
        "comparable_accuracy_range": (
            max(row["map50_95"] for row in current)
            - min(row["map50_95"] for row in current)
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "all_test_results.csv", results)
    write_csv(OUT_DIR / "current_checkpoint_results.csv", current)
    write_csv(OUT_DIR / "strict_comparable_pretrained_results.csv", strict_comparable)
    write_csv(OUT_DIR / "initialization_deltas.csv", paired_initialization_deltas)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact = build_artifact(current, strict_comparable, summary)
    (OUT_DIR / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
