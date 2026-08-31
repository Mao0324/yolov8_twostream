#!/usr/bin/env python3
"""Build a reproducible portable report for the DA016 A-H ablations."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "DroneVehicle_OBB_FusionTransfer"
OUT = ROOT / "reports" / "da016_ablation_analysis_20260830"

EXPERIMENTS = {
    "A": (
        "DA016-A_OriginalSemanticDisagreementLAF-P34_M2DLIFLabels_v1",
        "原始 DA016",
        "基线",
    ),
    "B": (
        "DA016-B_DualReliabilityPromptAuxOnly-P34_M2DLIFLabels_v12",
        "A + P3/P4 双 Prompt 辅助监督",
        "A",
    ),
    "C": (
        "DA016-C_DualReliabilityPrompt-ZeroInitBoundedResidual-P34_M2DLIFLabels_v12",
        "B + P3/P4 zero-init 有界 residual",
        "B",
    ),
    "D": (
        "DA016-D_DualReliabilityPrompt-P3ZeroInitBoundedResidual-P4AuxOnly_M2DLIFLabels_v1",
        "B + 仅 P3 residual；P4 辅助监督",
        "B",
    ),
    "E": (
        "DA016-E_DualReliabilityPrompt-P34Residual-P5AuxOnly_M2DLIFLabels_v1",
        "C + P5 Prompt 辅助监督",
        "C",
    ),
    "F": (
        "DA016-F_EPlusP5ZeroInitBoundedResidual_M2DLIFLabels_v1",
        "E + P5 zero-init 有界 residual",
        "E",
    ),
    "G": (
        "DA016-G_BPlusP5DualReliabilityPromptAuxOnly_M2DLIFLabels_v1",
        "B + P5 Prompt 辅助监督",
        "B",
    ),
    "B-Detach": (
        "DA016-BDetach_PromptGradientCausalControl-P34_M2DLIFLabels_v1",
        "B 的 Prompt 特征 detach 因果对照",
        "B",
    ),
    "H": (
        "DA016-H_BPlusClassWinnerBalancedPromptAux-P34_M2DLIFLabels_v1",
        "B + 类别×teacher-winner 平衡辅助监督",
        "B",
    ),
}

CLASS_ORDER = ["all", "car", "truck", "bus", "van", "freight_car"]
DISPLAY_CLASS = {
    "all": "all",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "van": "van",
    "freight_car": "freight_car",
}


def parse_test(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    metadata = {}
    for key in ("labels", "rgb", "ir", "class_map", "split", "imgsz", "batch", "device"):
        match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, flags=re.MULTILINE)
        if match:
            metadata[key] = match.group(1).strip()

    rows: dict[str, dict[str, float]] = {}
    pattern = re.compile(
        r"^\s*(all|car|truck|bus|van|freight_car)\s+"
        r"(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(text):
        cls, images, instances, precision, recall, map50, map5095 = match.groups()
        rows[cls] = {
            "images": int(images),
            "instances": int(instances),
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(map50),
            "map5095": float(map5095),
        }
    missing = [name for name in CLASS_ORDER if name not in rows]
    if missing:
        raise RuntimeError(f"Missing test rows in {path}: {missing}")
    speed = re.search(r"Speed: .*?([0-9.]+)ms inference", text)
    metadata["inference_ms"] = float(speed.group(1)) if speed else None
    return rows, metadata


def read_training(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {str(k).strip(): float(str(v).strip()) for k, v in row.items() if k is not None and v not in (None, "")}
            for row in reader
        ]
    best = max(rows, key=lambda row: row["metrics/mAP50-95(B)"])
    return best, rows[-1]


def pct(value: float) -> float:
    return round(value * 100, 3)


def delta_pp(value: float, baseline: float) -> float:
    return round((value - baseline) * 100, 3)


def source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    definitions: list[str],
    sql: str,
    tables_used: list[str],
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "tables_used": tables_used,
            "filters": [
                "仅使用各实验 test_m2dlif/test.txt 的 test split",
                "imgsz=640，batch=16，M2D-LIF labels，8980 images",
                "不比较不同 GPU 上的推理速度",
            ],
            "metric_definitions": definitions,
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test_by_exp = {}
    metadata_by_exp = {}
    train_by_exp = {}
    last_by_exp = {}

    for exp, (run_name, _, _) in EXPERIMENTS.items():
        run_dir = RUN_ROOT / run_name
        test_by_exp[exp], metadata_by_exp[exp] = parse_test(run_dir / "test_m2dlif" / "test.txt")
        train_by_exp[exp], last_by_exp[exp] = read_training(run_dir / "results.csv")

    # Strict comparability checks. Device is intentionally excluded.
    comparable_keys = ("labels", "rgb", "ir", "class_map", "split", "imgsz", "batch")
    base_meta = metadata_by_exp["A"]
    for exp, meta in metadata_by_exp.items():
        for key in comparable_keys:
            if meta.get(key) != base_meta.get(key):
                raise RuntimeError(f"{exp} differs from A for {key}: {meta.get(key)!r} != {base_meta.get(key)!r}")

    baseline = test_by_exp["A"]["all"]
    overall = []
    for exp, (_, structure, parent) in EXPERIMENTS.items():
        row = test_by_exp[exp]["all"]
        overall.append(
            {
                "experiment": exp,
                "parent": parent,
                "structure": structure,
                "precision_pct": pct(row["precision"]),
                "recall_pct": pct(row["recall"]),
                "map50_pct": pct(row["map50"]),
                "map5095_pct": pct(row["map5095"]),
                "delta_vs_a_pp": delta_pp(row["map5095"], baseline["map5095"]),
            }
        )

    comparisons_spec = [
        ("B − A", "A", "B", "双 Prompt 辅助监督"),
        ("B-Detach − B", "B", "B-Detach", "辅助梯度因果对照"),
        ("C − B", "B", "C", "P3/P4 residual"),
        ("D − B", "B", "D", "仅 P3 residual"),
        ("E − C", "C", "E", "P5 Prompt（P3/P4 residual 背景）"),
        ("F − E", "E", "F", "P5 residual"),
        ("G − B", "B", "G", "P5 Prompt（无 residual 背景）"),
        ("H − B", "B", "H", "类别×winner 平衡监督"),
    ]
    comparisons = []
    for label, parent, child, change in comparisons_spec:
        p = test_by_exp[parent]["all"]
        c = test_by_exp[child]["all"]
        comparisons.append(
            {
                "comparison": label,
                "change": change,
                "delta_precision_pp": delta_pp(c["precision"], p["precision"]),
                "delta_recall_pp": delta_pp(c["recall"], p["recall"]),
                "delta_map50_pp": delta_pp(c["map50"], p["map50"]),
                "delta_map5095_pp": delta_pp(c["map5095"], p["map5095"]),
            }
        )

    h_class_delta = []
    for cls in CLASS_ORDER[1:]:
        b = test_by_exp["B"][cls]
        h = test_by_exp["H"][cls]
        h_class_delta.append(
            {
                "class": DISPLAY_CLASS[cls],
                "instances": int(h["instances"]),
                "b_map50_pct": pct(b["map50"]),
                "h_map50_pct": pct(h["map50"]),
                "delta_map50_pp": delta_pp(h["map50"], b["map50"]),
                "b_map5095_pct": pct(b["map5095"]),
                "h_map5095_pct": pct(h["map5095"]),
                "delta_map5095_pp": delta_pp(h["map5095"], b["map5095"]),
            }
        )

    def prompt_row(exp: str) -> dict:
        best = train_by_exp[exp]
        row = {
            "experiment": exp,
            "best_val_epoch": int(best["epoch"]),
            "best_val_map5095_pct": pct(best["metrics/mAP50-95(B)"]),
        }
        for stage in ("P3", "P4", "P5"):
            acc = best.get(f"p2/{stage}_object_prompt_accuracy")
            corr = best.get(f"p2/{stage}_object_prompt_correlation")
            gain = best.get(f"p2/{stage}_prompt_residual_gain")
            row[f"{stage.lower()}_accuracy_pct"] = pct(acc) if acc is not None else None
            row[f"{stage.lower()}_correlation"] = round(corr, 4) if corr is not None else None
            row[f"{stage.lower()}_residual_gain"] = round(gain, 4) if gain is not None else None
        return row

    prompt_summary = [prompt_row(exp) for exp in EXPERIMENTS]

    summary = [{
        "h_map5095_pct": pct(test_by_exp["H"]["all"]["map5095"]),
        "h_vs_b_map5095_pp": delta_pp(test_by_exp["H"]["all"]["map5095"], test_by_exp["B"]["all"]["map5095"]),
        "h_map50_pct": pct(test_by_exp["H"]["all"]["map50"]),
        "h_vs_b_map50_pp": delta_pp(test_by_exp["H"]["all"]["map50"], test_by_exp["B"]["all"]["map50"]),
        "c_map5095_pct": pct(test_by_exp["C"]["all"]["map5095"]),
        "g_vs_b_map5095_pp": delta_pp(test_by_exp["G"]["all"]["map5095"], test_by_exp["B"]["all"]["map5095"]),
        "f_vs_e_map5095_pp": delta_pp(test_by_exp["F"]["all"]["map5095"], test_by_exp["E"]["all"]["map5095"]),
    }]

    source_inventory = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "test_files": {exp: f"runs/DroneVehicle_OBB_FusionTransfer/{run_name}/test_m2dlif/test.txt" for exp, (run_name, _, _) in EXPERIMENTS.items()},
        "training_files": {exp: f"runs/DroneVehicle_OBB_FusionTransfer/{run_name}/results.csv" for exp, (run_name, _, _) in EXPERIMENTS.items()},
        "comparability": {key: base_meta[key] for key in comparable_keys},
        "datasets": {
            "summary": summary,
            "test_overall": overall,
            "direct_comparisons": comparisons,
            "h_class_delta": h_class_delta,
            "prompt_summary": prompt_summary,
        },
    }
    (OUT / "analysis_data.json").write_text(json.dumps(source_inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    query_dir = OUT / "queries"
    query_dir.mkdir(exist_ok=True)
    test_sql = "SELECT dataset, row_json FROM test_metrics_export ORDER BY dataset, row_json;"
    train_sql = "SELECT dataset, row_json FROM training_metrics_export ORDER BY dataset, row_json;"
    (query_dir / "test_metrics.sql").write_text(test_sql + "\n", encoding="utf-8")
    (query_dir / "training_metrics.sql").write_text(train_sql + "\n", encoding="utf-8")

    # Materialize and execute the exact report queries so provenance is reproducible.
    database = sqlite3.connect(OUT / "analysis_snapshot.sqlite")
    try:
        database.execute("DROP TABLE IF EXISTS test_metrics_export")
        database.execute("DROP TABLE IF EXISTS training_metrics_export")
        database.execute("CREATE TABLE test_metrics_export (dataset TEXT NOT NULL, row_json TEXT NOT NULL)")
        database.execute("CREATE TABLE training_metrics_export (dataset TEXT NOT NULL, row_json TEXT NOT NULL)")
        for dataset, rows in {
            "summary": summary,
            "test_overall": overall,
            "direct_comparisons": comparisons,
            "h_class_delta": h_class_delta,
        }.items():
            database.executemany(
                "INSERT INTO test_metrics_export VALUES (?, ?)",
                [(dataset, json.dumps(row, ensure_ascii=False, sort_keys=True)) for row in rows],
            )
        database.executemany(
            "INSERT INTO training_metrics_export VALUES (?, ?)",
            [("prompt_summary", json.dumps(row, ensure_ascii=False, sort_keys=True)) for row in prompt_summary],
        )
        database.commit()
        database.execute(test_sql).fetchall()
        database.execute(train_sql).fetchall()
    finally:
        database.close()

    with (OUT / "test_overall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall[0]))
        writer.writeheader()
        writer.writerows(overall)
    with (OUT / "training_prompt_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prompt_summary[0]))
        writer.writeheader()
        writer.writerows(prompt_summary)

    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    test_source = source(
        "test_metrics",
        "DA016 A-H test_m2dlif 评估结果",
        "queries/test_metrics.sql",
        "逐个解析九个实验的 test_m2dlif/test.txt，并校验数据集、标签、split、imgsz 与 batch 一致。",
        [
            "mAP50-95 为 IoU 0.50:0.95 上的 OBB AP 均值；报告中的百分数为原始小数乘 100。",
            "差值单位为 percentage point（pp），计算为 (child - parent) × 100。",
            "测试集包含 8980 张图像与 159618 个实例。",
        ],
        test_sql,
        ["test_metrics_export"],
    )
    train_source = source(
        "training_metrics",
        "DA016 A-H 训练与 Prompt 诊断指标",
        "queries/training_metrics.sql",
        "解析每个 results.csv；best validation epoch 由 metrics/mAP50-95(B) 最大值确定。",
        [
            "Prompt accuracy 为目标级 RGB/IR winner 分类准确率。",
            "Prompt correlation 为预测相对可靠性与 teacher 相对目标的相关性。",
            "best validation 仅用于训练诊断，不用于替代 test_m2dlif 的最终比较。",
        ],
        train_sql,
        ["training_metrics_export"],
    )

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "DA016 A–H 严格消融实验结果分析",
            "description": "M2D-LIF labels、同一 DroneVehicle test split 下的双 Prompt、残差与 P5 消融诊断。",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "h_quality",
                    "description": "H 是辅助监督路线的最好结果，并与 C 并列最高 test mAP50–95。",
                    "dataset": "summary",
                    "sourceId": "test_metrics",
                    "metrics": [
                        {"label": "H test mAP50–95 (%)", "field": "h_map5095_pct", "format": "number"},
                        {"label": "H − B (pp)", "field": "h_vs_b_map5095_pp", "format": "number", "signed": True},
                    ],
                },
                {
                    "id": "h_map50",
                    "description": "H 的 mAP50 为所有实验最高。",
                    "dataset": "summary",
                    "sourceId": "test_metrics",
                    "metrics": [
                        {"label": "H test mAP50 (%)", "field": "h_map50_pct", "format": "number"},
                        {"label": "H − B (pp)", "field": "h_vs_b_map50_pp", "format": "number", "signed": True},
                    ],
                },
                {
                    "id": "p5_aux",
                    "description": "在 B 上增加 P5 Prompt 辅助监督的直接效果。",
                    "dataset": "summary",
                    "sourceId": "test_metrics",
                    "metrics": [
                        {"label": "G − B mAP50–95 (pp)", "field": "g_vs_b_map5095_pp", "format": "number", "signed": True},
                    ],
                },
                {
                    "id": "p5_residual",
                    "description": "在 E 上启用 P5 residual 的直接效果。",
                    "dataset": "summary",
                    "sourceId": "test_metrics",
                    "metrics": [
                        {"label": "F − E mAP50–95 (pp)", "field": "f_vs_e_map5095_pp", "format": "number", "signed": True},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "overall_delta",
                    "title": "各实验相对 A 的 test mAP50–95 差值",
                    "subtitle": "单位为百分点；0 代表与原始 DA016 相同。",
                    "intent": "comparison",
                    "question": "哪些 Prompt、残差与 P5 组合真正超过原始 DA016 基线？",
                    "rationale": "水平条形图适合比较九个离散实验相对同一基线的小幅正负变化。",
                    "type": "horizontalBar",
                    "dataset": "test_overall",
                    "sourceId": "test_metrics",
                    "encodings": {
                        "x": {"field": "experiment", "type": "nominal", "label": "实验"},
                        "y": {"field": "delta_vs_a_pp", "type": "quantitative", "label": "Δ mAP50–95", "unit": "pp"},
                        "tooltip": [
                            {"field": "map5095_pct", "type": "quantitative", "label": "mAP50–95", "unit": "%"},
                            {"field": "parent", "type": "nominal", "label": "直接父实验"},
                        ],
                    },
                    "xAxisTitle": "相对 A 的 mAP50–95 差值 (pp)",
                    "valueFormat": "number",
                    "unit": "pp",
                    "referenceLines": [{"value": 0, "label": "A baseline"}],
                    "layout": "full",
                },
                {
                    "id": "h_class_delta_chart",
                    "title": "H 相对 B 的逐类别 test mAP50–95 差值",
                    "subtitle": "类别×teacher-winner 平衡监督主要改善 truck、van 与 freight_car。",
                    "intent": "comparison",
                    "question": "H 的总体收益来自哪些类别，又牺牲了哪些类别？",
                    "rationale": "逐类水平条形图直接暴露少数类收益与 bus 退化，避免总体均值掩盖方向相反的变化。",
                    "type": "horizontalBar",
                    "dataset": "h_class_delta",
                    "sourceId": "test_metrics",
                    "encodings": {
                        "x": {"field": "class", "type": "nominal", "label": "类别"},
                        "y": {"field": "delta_map5095_pp", "type": "quantitative", "label": "H − B", "unit": "pp"},
                        "tooltip": [
                            {"field": "instances", "type": "quantitative", "label": "实例数"},
                            {"field": "b_map5095_pct", "type": "quantitative", "label": "B mAP50–95", "unit": "%"},
                            {"field": "h_map5095_pct", "type": "quantitative", "label": "H mAP50–95", "unit": "%"},
                        ],
                    },
                    "xAxisTitle": "H − B mAP50–95 (pp)",
                    "valueFormat": "number",
                    "unit": "pp",
                    "referenceLines": [{"value": 0, "label": "no change"}],
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "overall_table",
                    "title": "九个实验的 test 指标",
                    "subtitle": "全部取自相同设置的 test_m2dlif 评估。",
                    "dataset": "test_overall",
                    "sourceId": "test_metrics",
                    "defaultSort": {"field": "map5095_pct", "direction": "desc"},
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "experiment", "label": "实验", "type": "text"},
                        {"field": "parent", "label": "父实验", "type": "text"},
                        {"field": "precision_pct", "label": "P (%)", "format": "number"},
                        {"field": "recall_pct", "label": "R (%)", "format": "number"},
                        {"field": "map50_pct", "label": "mAP50 (%)", "format": "number"},
                        {"field": "map5095_pct", "label": "mAP50–95 (%)", "format": "number"},
                        {"field": "delta_vs_a_pp", "label": "vs A (pp)", "format": "number", "movement": True},
                    ],
                },
                {
                    "id": "comparison_table",
                    "title": "严格父子对照的直接增量",
                    "subtitle": "每一行只比较一个新增机制。",
                    "dataset": "direct_comparisons",
                    "sourceId": "test_metrics",
                    "defaultSort": {"field": "delta_map5095_pp", "direction": "desc"},
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "comparison", "label": "对照", "type": "text"},
                        {"field": "change", "label": "唯一新增机制", "type": "text"},
                        {"field": "delta_precision_pp", "label": "ΔP (pp)", "format": "number", "movement": True},
                        {"field": "delta_recall_pp", "label": "ΔR (pp)", "format": "number", "movement": True},
                        {"field": "delta_map50_pp", "label": "ΔmAP50 (pp)", "format": "number", "movement": True},
                        {"field": "delta_map5095_pp", "label": "ΔmAP50–95 (pp)", "format": "number", "movement": True},
                    ],
                },
                {
                    "id": "prompt_table",
                    "title": "最佳 validation epoch 与 Prompt 诊断",
                    "subtitle": "A 无 Prompt；B 与 B-Detach 用于判断辅助梯度是否真正改变 Prompt 表征。",
                    "dataset": "prompt_summary",
                    "sourceId": "training_metrics",
                    "defaultSort": {"field": "best_val_map5095_pct", "direction": "desc"},
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "experiment", "label": "实验", "type": "text"},
                        {"field": "best_val_epoch", "label": "epoch", "format": "number"},
                        {"field": "best_val_map5095_pct", "label": "best val (%)", "format": "number"},
                        {"field": "p3_accuracy_pct", "label": "P3 acc (%)", "format": "number"},
                        {"field": "p3_correlation", "label": "P3 corr", "format": "number"},
                        {"field": "p4_accuracy_pct", "label": "P4 acc (%)", "format": "number"},
                        {"field": "p4_correlation", "label": "P4 corr", "format": "number"},
                        {"field": "p5_accuracy_pct", "label": "P5 acc (%)", "format": "number"},
                        {"field": "p5_correlation", "label": "P5 corr", "format": "number"},
                    ],
                },
            ],
            "sources": [test_source, train_source],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# DA016 A–H 严格消融实验结果分析"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "test_metrics",
                    "body": (
                        "## 结论：H 与 C 并列第一，但两条路线解决的是不同问题\n\n"
                        "**C 与 H 的 test mAP50–95 都是 67.0%，相对 A 提升 0.3 pp。** C 证明 P3/P4 的 Prompt 条件残差可能有益；H 则在完全不让 Prompt 进入融合的前提下，把 mAP50 提到最高的 82.6%。"
                        "\n\n**B 与 B-Detach 的 test mAP50–95 同为 66.8%。** 因而 B 相对 A 的 0.1 pp 不能被解释为 Prompt 辅助梯度带来的因果检测收益。"
                        "\n\n**P5 路线没有通过消融。** G−B 和 F−E 均为 −0.2 pp；E−C 为 0.0 pp。当前证据支持停止继续给 P5 加 Prompt 或 residual。"
                    ),
                },
                {"id": "key_metrics", "type": "metric-strip", "cardIds": ["h_quality", "h_map50", "p5_aux", "p5_residual"]},
                {
                    "id": "ranking_headline",
                    "type": "markdown",
                    "sourceId": "test_metrics",
                    "body": (
                        "## 总体结果：只有 C、E、H 达到 67.0%，但 E 没有提供 P5 增量\n\n"
                        "C、E、H 相对 A 都是 +0.3 pp；不过 E 与其直接父实验 C 完全相同，所以不能把 E 的成绩归因于 P5 Prompt。D、F、G 分别验证了仅 P3 residual、P5 residual、P5 auxiliary 的负结果。"
                    ),
                },
                {"id": "overall_delta_block", "type": "chart", "chartId": "overall_delta", "layout": "full"},
                {"id": "overall_table_block", "type": "table", "tableId": "overall_table", "layout": "full"},
                {
                    "id": "h_class_headline",
                    "type": "markdown",
                    "sourceId": "test_metrics",
                    "body": (
                        "## H 的收益集中在少数类，但 bus 明显退化\n\n"
                        "相对 B，H 的 truck、van、freight_car mAP50–95 分别提高 0.9、0.7、0.6 pp；car 基本不变（−0.1 pp），bus 下降 0.7 pp。这个分布与类别×winner 平衡监督的设计目标一致，但也说明当前平衡强度可能过大。"
                    ),
                },
                {"id": "h_class_delta_block", "type": "chart", "chartId": "h_class_delta_chart", "layout": "full"},
                {
                    "id": "causal_headline",
                    "type": "markdown",
                    "sourceId": "training_metrics",
                    "body": (
                        "## B 的 Prompt 学到了可靠性，但这次没有转化成检测 AP\n\n"
                        "B 在最佳 validation epoch 的 P3/P4 Prompt accuracy 为 68.3%/68.7%，高于 B-Detach 的 66.0%/65.4%；相关性也分别由 0.388/0.407 提高到 0.420/0.454。"
                        "\n\n这说明辅助梯度确实改善了 Prompt 的 teacher-winner 表征，不是简单复制固定 prior。"
                    ),
                },
                {
                    "id": "causal_detection_result",
                    "type": "markdown",
                    "sourceId": "test_metrics",
                    "body": (
                        "### 但检测收益没有通过因果对照\n\n"
                        "B 与 B-Detach 的 test mAP50–95 都是 66.8%，所以不能声称 Prompt 辅助梯度已经改善 detector；B 相对 A 的 +0.1 pp 也可能只是单次训练波动。"
                    ),
                },
                {"id": "prompt_table_block", "type": "table", "tableId": "prompt_table", "layout": "full"},
                {
                    "id": "p5_diagnosis",
                    "type": "markdown",
                    "sourceId": "training_metrics",
                    "body": (
                        "## P5 的问题不是 Prompt 没学会，而是信号与检测目标不对齐\n\n"
                        "G 的 P5 Prompt accuracy/correlation 已达到 70.7%/0.453，F 的 P5 residual gain 也已从 zero-init 学到约 0.024。说明 P5 Prompt head 与 residual 都实际激活了，并非没有训练到。"
                    ),
                },
                {
                    "id": "p5_detection_counterevidence",
                    "type": "markdown",
                    "sourceId": "test_metrics",
                    "body": (
                        "### 但 P5 的检测结果反向\n\n"
                        "G−B 与 F−E 的 test mAP50–95 都下降 0.2 pp，而 E−C 为 0.0 pp。因此，继续加大 P5 参数或监督权重不太可能直接解决问题；更合理的假设是 P5 的粗空间尺度与当前逐目标可靠性监督存在梯度冲突或目标混叠。"
                    ),
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## 范围、数据与指标定义\n\n"
                        "九个实验全部使用同一 M2D-LIF 标注、同一 DroneVehicle RGB/IR test split、imgsz 640、batch 16；测试集共 8980 张图像和 159618 个实例。类别映射一致。"
                        "\n\nmAP50–95 指 OBB 在 IoU 0.50:0.95 上的平均 AP；报告中的 pp 是 mAP 小数差乘 100。由于各 test.txt 只保留三位小数，0.1 pp 量级的差异仅可视为方向性证据。"
                    ),
                    "sourceId": "test_metrics",
                },
                {
                    "id": "design",
                    "type": "markdown",
                    "body": (
                        "## 实验设计：必须按直接父实验解释增量\n\n"
                        "A 是原始 DA016；B 只加 P3/P4 Prompt 辅助监督；C 在 B 上给 P3/P4 加 residual；D 只给 P3 加 residual；E 在 C 上加 P5 auxiliary；F 再给 P5 加 residual；G 在 B 上单独加 P5 auxiliary；B-Detach 隔断 Prompt 辅助梯度；H 在 B 上改为类别×winner 平衡监督。"
                        "\n\n因此，E 的绝对成绩不能被当作 P5 的收益，必须看 E−C；同理，H 必须优先看 H−B，而不是只看 H−A。"
                    ),
                },
                {"id": "comparison_table_block", "type": "table", "tableId": "comparison_table", "layout": "full"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": "training_metrics",
                    "body": (
                        "## 不确定性：validation 与 test 排序不一致\n\n"
                        "每个结构目前只有一个训练运行。H 的最佳 validation mAP50–95 只有 71.9%，低于 B 的 72.4% 与 E 的 72.6%。这表明单次运行的 checkpoint 与数据划分波动不可忽略。"
                    ),
                },
                {
                    "id": "test_precision_limit",
                    "type": "markdown",
                    "sourceId": "test_metrics",
                    "body": (
                        "### 三位小数不足以证明微小提升\n\n"
                        "测试文本只保留三位小数，且缺少逐图预测与 paired bootstrap 区间。因此 +0.1 至 +0.3 pp 目前只能作为方向性证据，不能构成统计稳健的优越性声明。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 下一步：先验证 H，再隔离 P4，不再扩展 P5\n\n"
                        "1. **最高优先级：复跑 A、B、B-Detach、H、C 至少 3 个 seed。** 核心问题是 H−B 是否稳定为正，以及 B−B-Detach 是否仍接近零。\n"
                        "2. **对 H 做平衡强度扫描。** 建议 0.25、0.50、0.75，相对于当前 1.0；目标是保留 truck/van/freight_car 收益，同时追回 bus。\n"
                        "3. **新增 P4-only residual。** D 证明 P3-only 为负，C 的 P3+P4 为正；必须用 `J = B + P4 residual only` 才能判断真正贡献是否来自 P4 或跨尺度交互。\n"
                        "4. **停止 P5 auxiliary/residual 扩展。** 除非先改成尺度感知的目标分配或显式梯度冲突控制。\n"
                        "5. **保存未四舍五入的评估 JSON 和逐图预测。** 后续用 paired bootstrap 给出置信区间，不再仅凭三位小数排名。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 仍需回答的问题\n\n"
                        "- H 的 bus 下降是平衡系数过强，还是 teacher 在 bus 上的 winner 标签噪声更大？\n"
                        "- C 的收益主要来自 P4，还是 P3/P4 residual 的联合优化？\n"
                        "- Prompt accuracy 提升但 AP 不变时，梯度冲突发生在 backbone、融合层还是检测 head？\n"
                        "- H 的 best.pt 与 last.pt 是否保持同样的 test 排序？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "summary": summary,
                "test_overall": overall,
                "direct_comparisons": comparisons,
                "h_class_delta": h_class_delta,
                "prompt_summary": prompt_summary,
            },
        },
        "sources": [test_source, train_source],
        "package_info": {
            "root": "da016_ablation_analysis_20260830",
            "manifestPath": "artifact.json",
            "snapshotPath": "artifact.json",
        },
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT / "artifact.json")


if __name__ == "__main__":
    main()
