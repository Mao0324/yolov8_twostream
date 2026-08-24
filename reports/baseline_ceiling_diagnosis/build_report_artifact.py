#!/usr/bin/env python3
"""Build the canonical portable-report artifact for the diagnosis."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


OUT = Path(__file__).resolve().parent
TITLE = "双流 OBB Baseline 是否过高：跨分支实验诊断"


def load_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def short_label(run_name: str) -> str:
    rules = (
        ("PostC2f-KLDProbIoU05", "DarkACT KLD-0.5"),
        ("PostC2f-MSK3-5", "DarkACT Post-C2f MAA"),
        ("ASSAFusion_P345", "ASSA P345"),
        ("FeedbackRefine_P345", "DarkACT fused refine"),
        ("ZeroCentered", "DarkACT zero-centered"),
        ("LAFMergeFeedback2D_P345", "DarkACT LAF-only"),
        ("StaticMAA2DLAFMergeFeedback_P345", "DarkACT StaticMAA+LAF"),
        ("StaticMAA2DRefine", "DarkACT per-modal refine"),
        ("SqrtHW", "DarkACT target sqrt(HW)"),
        ("ReplaceLAFCross_P3", "ASSA replace-LAF P3"),
        ("CRFormer", "CRFormer P3"),
        ("BottleneckRefine", "Bottleneck refine"),
    )
    for token, label in rules:
        if token in run_name:
            return label
    return run_name[:42]


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> int:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    comparable = load_csv("comparable_pretrained_results.csv")
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")

    top_rows = []
    for rank, row in enumerate(comparable[:12], start=1):
        top_rows.append(
            {
                "rank": rank,
                "model": short_label(row["run_name"]),
                "family": row["family"],
                "run_name": row["run_name"],
                "map50": number(row, "map50"),
                "delta_pp": number(row, "delta_map50_vs_baseline") * 100,
                "map50_95": number(row, "map50_95"),
                "seed": int(float(row["cfg_seed"])),
                "best_epoch": int(float(row["best_fitness_epoch"])),
            }
        )

    class_rows = []
    for row in summary["class_deltas_for_best_observed"]:
        for series, field in (("Baseline", "baseline_map50"), ("Best observed", "best_map50")):
            class_rows.append(
                {
                    "class": row["class"],
                    "series": series,
                    "map50": row[field],
                    "delta_pp": row["delta_map50"] * 100,
                    "baseline_map50_95": row["baseline_map50_95"],
                    "best_map50_95": row["best_map50_95"],
                }
            )

    baseline = summary["baseline"]
    best = summary["best_observed"]
    distribution = summary["variant_distribution"]
    headline = [
        {
            "baseline_map50": baseline["map50"],
            "best_map50": best["map50"],
            "best_delta": best["delta_map50_vs_baseline"],
            "median_gain_pp": (distribution["map50_median"] - baseline["map50"]) * 100,
        }
    ]

    source = {
        "id": "cross_branch_results",
        "label": "Cross-branch committed DroneVehicle test results",
        "path": "reports/baseline_ceiling_diagnosis/comparable_pretrained_results.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": "analyze_results.py discovers committed test outputs across six local branches, deduplicates inherited blobs, parses the final class table, joins args.yaml and results.csv, and writes the reviewed comparable cohort.",
            "sql": "SELECT * FROM read_csv_auto('reports/baseline_ceiling_diagnosis/comparable_pretrained_results.csv', header = true)",
            "tables_used": [
                "reports/baseline_ceiling_diagnosis/comparable_pretrained_results.csv"
            ],
            "filters": [
                "Committed Git artifacts from main, ASSAFusion, CRFormer, DarkACT, exp/cfgpnet-lite, and exp/protohgfnet-lite",
                "DroneVehicle test split only; 8,980 images and 159,614 OBB instances",
                "mAP50 > 0.83",
                "args.pretrained = true",
                "args.model points to a .pt checkpoint",
                "Inherited identical Git blobs deduplicated",
            ],
            "metric_definitions": [
                "mAP50 is the unweighted mean of the five class AP values at IoU 0.50 reported by the OBB evaluator.",
                "mAP50-95 is the unweighted mean AP over IoU thresholds 0.50:0.95 reported by the OBB evaluator.",
                "Percentage-point delta equals (variant mAP - baseline mAP) × 100.",
                "Best validation checkpoint fitness uses 0.1 × mAP50 + 0.9 × mAP50-95, as implemented in ultralytics/utils/metrics.py.",
            ],
        },
    }

    cards = [
        {
            "id": "baseline_card",
            "dataset": "headline",
            "sourceId": "cross_branch_results",
            "description": "Pretrained two-stream baseline on the committed DroneVehicle test split.",
            "metrics": [
                {"label": "Baseline mAP50", "field": "baseline_map50", "format": "number"}
            ],
        },
        {
            "id": "best_card",
            "dataset": "headline",
            "sourceId": "cross_branch_results",
            "description": "Highest committed mAP50 among the comparable cohort.",
            "metrics": [
                {"label": "Best observed mAP50", "field": "best_map50", "format": "number"},
                {"label": "vs baseline", "field": "best_delta", "format": "number", "signed": True},
            ],
        },
        {
            "id": "median_card",
            "dataset": "headline",
            "sourceId": "cross_branch_results",
            "description": "Median absolute mAP50 percentage-point gain across 22 variants.",
            "metrics": [
                {"label": "Median variant gain (pp)", "field": "median_gain_pp", "format": "number", "signed": True}
            ],
        },
    ]

    charts = [
        {
            "id": "top_delta_chart",
            "title": "Top comparable models: mAP50 gain versus baseline",
            "subtitle": "Top 12 of 23 comparable runs; absolute percentage-point delta, single seed (seed 0).",
            "showDescription": True,
            "intent": "comparison",
            "question": "How large are the strongest observed mAP50 gains over the pretrained baseline?",
            "rationale": "Horizontal bars make small signed model deltas and long model labels directly comparable.",
            "type": "bar",
            "dataset": "top_runs",
            "sourceId": "cross_branch_results",
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {"field": "delta_pp", "type": "quantitative", "label": "ΔmAP50", "unit": "pp"},
                "tooltip": [
                    {"field": "family", "type": "text", "label": "Family"},
                    {"field": "map50", "type": "quantitative", "label": "Test mAP50"},
                    {"field": "map50_95", "type": "quantitative", "label": "Test mAP50-95"},
                    {"field": "best_epoch", "type": "quantitative", "label": "Best val epoch"},
                    {"field": "seed", "type": "quantitative", "label": "Seed"},
                ],
            },
            "xAxisTitle": "Model",
            "yAxisTitle": "mAP50 change vs baseline (percentage points)",
            "layout": "full",
            "palette": {"kind": "sequential", "name": "blue"},
            "settings": {"orientation": "horizontal", "sort": "descending", "showValues": True},
        },
        {
            "id": "class_comparison_chart",
            "title": "Per-class AP50: baseline versus highest-mAP50 model",
            "subtitle": "The overall gain is concentrated in truck and freight_car; car and bus are already near saturation.",
            "showDescription": True,
            "intent": "comparison",
            "question": "Is the small aggregate gain caused by class-level saturation?",
            "rationale": "Grouped bars preserve absolute AP50 levels while showing which classes changed.",
            "type": "bar",
            "dataset": "class_comparison",
            "sourceId": "cross_branch_results",
            "encodings": {
                "x": {"field": "class", "type": "nominal", "label": "Class"},
                "y": {"field": "map50", "type": "quantitative", "label": "AP50"},
                "color": {"field": "series", "type": "nominal", "label": "Model"},
                "tooltip": [
                    {"field": "series", "type": "text", "label": "Model"},
                    {"field": "map50", "type": "quantitative", "label": "Class AP50"},
                    {"field": "delta_pp", "type": "quantitative", "label": "Best-minus-baseline (pp)"},
                ],
            },
            "xAxisTitle": "Class",
            "yAxisTitle": "AP50",
            "layout": "full",
            "palette": {"kind": "identity", "name": "baseline-vs-best"},
            "legend": {"position": "top", "title": "Model"},
            "settings": {"groupMode": "grouped", "orientation": "vertical", "showValues": True},
        },
    ]

    tables = [
        {
            "id": "top_runs_table",
            "title": "Highest committed comparable results",
            "subtitle": "Top 12 by mAP50; all rows use the same test population and recorded training recipe.",
            "dataset": "top_runs",
            "sourceId": "cross_branch_results",
            "defaultSort": {"field": "map50", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "rank", "label": "Rank", "format": "number"},
                {"field": "model", "label": "Model", "type": "text"},
                {"field": "family", "label": "Family", "type": "text"},
                {"field": "map50", "label": "mAP50", "format": "number"},
                {"field": "delta_pp", "label": "Δ vs baseline (pp)", "format": "number", "movement": True},
                {"field": "map50_95", "label": "mAP50-95", "format": "number"},
                {"field": "seed", "label": "Seed", "format": "number"},
                {"field": "best_epoch", "label": "Best val epoch", "format": "number"},
            ],
        }
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "cross_branch_results",
            "body": "## 结论：不要人为压低 baseline；先证明 0.8pp 超过随机波动\n\n"
            "- **baseline 较强，但不能据此认定模型已到全局天花板。** 可比 baseline 的 Test mAP50 为 **0.833**、mAP50-95 为 **0.701**；最高结果为 **0.841 / 0.708**，分别提升 **0.8 / 0.7 个百分点**。\n"
            "- **现有提升可能有价值，但证据强度不足。** 22 个可比变体的 mAP50 中位数为 **0.837**，平均只比 baseline 高 **0.43pp**；所有运行都是 seed 0，没有跨 seed 方差或逐图预测，无法判断最高的 0.8pp 是否稳定。\n"
            "- **故意降低 baseline 会破坏对照。** 学习率和训练策略可以调，但目标应是让双方各自达到合理最优；若为了放大相对增益而选择使 baseline 变差的参数，所得“模块提升”不具备可信解释。\n"
            "- **当前更像“部分指标/类别饱和 + 增量模块收益有限 + 单次运行不确定性”，而不是单一的 baseline 过高。** 建议先做 3 seed 复现，再决定是否开展对称的学习率搜索。",
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["baseline_card", "best_card", "median_card"]},
        {
            "id": "observed_gain",
            "type": "markdown",
            "sourceId": "cross_branch_results",
            "body": "## 最高增益只有 0.8pp，而且大多数模型聚集在更窄区间\n\n"
            "23 个可比结果（含 baseline）使用同一测试集、同一输入尺寸和同一记录训练配方。变体 mAP50 分布为 **0.833–0.841**，中位数 **0.837**；mAP50-95 分布为 **0.701–0.708**。图中以 baseline 为零点展示最强 12 个结果，因此读者看到的是绝对百分点差异，而不是被 0.83 起始轴夸大的视觉差异。",
        },
        {"id": "top_delta", "type": "chart", "chartId": "top_delta_chart", "layout": "full"},
        {
            "id": "class_result",
            "type": "markdown",
            "sourceId": "cross_branch_results",
            "body": "## baseline 只在 car 与 bus 上接近饱和，弱类仍有明显空间\n\n"
            "最高 mAP50 模型相对 baseline 的类别变化为：car **0.000pp**、truck **+1.1pp**、bus **0.000pp**、van **+0.3pp**、freight_car **+2.2pp**。所以总分小，部分是 car（0.986）和 bus（0.968）已很难继续提高；但 baseline 的 van（0.671）与 freight_car（0.707）远未饱和，整体理论 headroom 仍为 **16.7pp**。更准确的说法是“简单类的 AP50 饱和掩盖了弱类变化”，而不是“baseline 已高到模块无法发挥”。",
        },
        {"id": "class_chart", "type": "chart", "chartId": "class_comparison_chart", "layout": "full"},
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "cross_branch_results",
            "body": "## 比较口径：只保留正确 checkpoint 初始化且 mAP50 > 0.83 的运行\n\n"
            "分析扫描 `main`、`ASSAFusion`、`CRFormer`、`DarkACT`、`exp/cfgpnet-lite` 与 `exp/protohgfnet-lite` 六个本地分支，共发现 **36** 份不重复提交的测试文件；按用户指定阈值，并额外验证 `args.pretrained=true` 与 `.pt` checkpoint 入口后，保留 **23** 个运行。它们均使用 DroneVehicle `test`：8,980 张图、159,614 个实例、640 输入。当前 CFGPNet 与 ProtoHGFNet 两个分支没有新增已提交的 `test_result/test.txt`，所以结论不包含这两类模型的新测试数值。",
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "cross_branch_results",
            "body": "## 方法：跨分支去重、配置核对与独立重算\n\n"
            "`analyze_results.py` 用 Git blob 去除分支继承造成的重复文件，解析每份 `test.txt` 最后一张总体/分类指标表，再联接相邻 `args.yaml` 与 `results.csv`。对比的 14 个关键训练字段只有 **1 个配置签名**：100 epochs、batch 64、imgsz 640、SGD、lr0 0.01、lrf 0.01、momentum 0.937、weight decay 0.0005、3 epoch warmup、非 cosine、close_mosaic 0、seed 0、deterministic、AMP。验证集最佳点按仓库 fitness 定义 `0.1×mAP50 + 0.9×mAP50-95` 独立重算。",
        },
        {
            "id": "training_curve",
            "type": "markdown",
            "sourceId": "cross_branch_results",
            "body": "## 当前学习率没有显示 baseline 欠训练，但可能不是新增模块的最优策略\n\n"
            "baseline 最佳验证 checkpoint 出现在 epoch **61**（val mAP50 0.8685、mAP50-95 0.7272），到 epoch 100 降为 0.8574 / 0.7208；最高测试模型的最佳验证点在 epoch **49**。这说明 100 epochs 已足够到达峰值，单纯延长训练不太可能解决问题。另一方面，所有迁移 backbone 和随机初始化的新模块共用同一全局 SGD 学习率；这保证了控制变量，却不保证新增模块得到最佳优化。可把分层学习率作为单独实验，但不能用它来刻意压低 baseline。",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "cross_branch_results",
            "body": "## 当前只能“带保留地使用”：0.8pp 尚未通过不确定性检验\n\n"
            "- **单 seed：** 23 个可比运行全部是 seed 0；同一 seed 不能估计训练随机性。\n"
            "- **无逐图预测：** 提交物只有聚合 AP，不能做 paired bootstrap 或检查模型在相同图像上的胜负分布。\n"
            "- **测试集反复筛选：** 36 份已提交测试结果说明 test 已参与大量架构比较；从中挑最高值会产生选择偏差。后续超参数与结构决策应只看 val，新 test/隐藏 holdout 只在方案锁定后评一次。\n"
            "- **最高模型的优势并非全类别一致：** 提升主要来自 truck 与 freight_car；不能把总体 +0.8pp 解读为普遍改善。\n\n"
            "综合判断为 **Share with caveats**：结果方向可用于选择复现对象，但不足以作为稳定优于 baseline 的最终结论。",
        },
        {
            "id": "top_table_context",
            "type": "markdown",
            "sourceId": "cross_branch_results",
            "body": "## 精确结果用于选复现对象，不用于宣称显著性\n\n"
            "下表列出最高 12 个可比运行及其验证集最佳 epoch。由于差异多数只有 0.1–0.8pp，应把它当作复现实验的优先级清单，而不是确定排名。",
        },
        {"id": "top_table", "type": "table", "tableId": "top_runs_table", "layout": "full"},
        {
            "id": "recommendations",
            "type": "markdown",
            "body": "## 推荐实验顺序：先测方差，再调优化器\n\n"
            "1. **锁定当前配方做最小复现。** baseline、最高 DarkACT Post-C2f 模型、ASSA P345 各跑 seeds 0/1/2，共 9 次；报告 Test mAP50、mAP50-95、各类 AP 的 mean±std。若可以保存逐图预测，再做图像级 paired bootstrap 95% CI。\n"
            "2. **开发阶段停止查看 test。** 所有学习率、结构和 epoch 选择仅使用 val；最终配置冻结后，在未参与筛选的 test 或新 holdout 上评一次。\n"
            "3. **只有当模块均值增益稳定后才调学习率。** 先做对称的 `lr0 ∈ {0.005, 0.01, 0.02}` 小搜索，baseline 与模块获得相同预算，并基于 val 选取。若必须使用完全相同的最终配方，应根据两者的平均 val 表现预先锁定，而不是根据哪组能制造更大的 Test 差值。\n"
            "4. **把分层学习率作为独立优化实验。** 迁移 backbone 可尝试 0.1× LR，新融合模块/检测头用 1× LR；同时保留当前全局 LR 对照。该实验回答“模块是否难优化”，不能替代公平 baseline。\n"
            "5. **优先增加更敏感的评估维度。** 主报 mAP50-95、AP75、van/freight_car AP，并按昼夜、距离、遮挡、目标尺度及 RGB–IR 配准质量分层；这些比故意降低 baseline 更能显示模块在哪些困难样本上有效。\n"
            "6. **可小规模验证训练日程，而不是盲目加 epoch。** 当前最佳点在 49–61 epoch；可试 `cos_lr=true` 与最后 10 epoch 关闭 mosaic，但必须对 baseline 和模块同等验证，并继续使用 best checkpoint。",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": "## 仍需回答的问题\n\n"
            "- 0.8pp 在 3–5 个 seed 下的均值、标准差和置信区间是多少？\n"
            "- 目标是总体 mAP50、定位更严格的 mAP50-95，还是 van/freight_car 的类定向收益？\n"
            "- 是否能划出一个不再参与调参的隐藏 holdout，消除当前多次查看 test 带来的选择偏差？\n"
            "- 新模块参数的初始化尺度、梯度范数和学习速度是否明显不同于迁移 backbone？",
        },
    ]

    manifest_sources = [source]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "诊断高 baseline 是否限制模块增益，并评估是否应通过学习率或训练设置改变对照结果。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "top_runs": top_rows,
                "class_comparison": class_rows,
            },
        },
        "sources": manifest_sources,
        "package_info": {
            "analysisScript": "reports/baseline_ceiling_diagnosis/analyze_results.py",
            "notebook": "reports/baseline_ceiling_diagnosis/baseline_ceiling_diagnosis.ipynb",
            "summary": "reports/baseline_ceiling_diagnosis/summary.json",
        },
    }
    (OUT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(OUT / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
