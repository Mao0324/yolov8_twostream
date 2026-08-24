#!/usr/bin/env python3
"""Build a compact, executable notebook for the RGB/IR alignment audit."""

from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).resolve().parent
NOTEBOOK = OUT / "dronevehicle_alignment_quality.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# DroneVehicle RGB/IR 配准质量审计\n\n"
        "## tl;dr\n\n"
        "数据集的 RGB/IR 文件配对、分辨率和共享标签均完整一致；几何上属于较好配准，"
        "但不能把同一车辆严格视为逐像素完全相同的 `(x, y)`。可信跨模态配准样本的"
        "残余平移中位数约 1.74 px，仍有一部分序列达到 5–10 px。"
    ),
    nbf.v4.new_markdown_cell(
        "## Context & Methods\n\n"
        "审计分两层：一是对全部 train/val/test 文件检查同名配对、可读性与尺寸；二是对"
        "400 对均匀抽样图像，用梯度幅值上的相位相关和 translation-only ECC 估计全局"
        "残余位移，并对其中 40 对追加仿射 ECC。标签框尺寸用所有多边形的最小外接旋转矩形"
        "计算。\n\n"
        "### Key Assumptions\n\n"
        "- ECC 相关系数至少 0.50 且位移不超过 10 px 的结果记为可信。\n"
        "- 该估计衡量场景级几何关系，不是每辆车的独立双模态中心标注。\n"
        "- 热辐射、阴影和运动目标会降低跨模态相关性。"
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import csv, json, subprocess, sys\n"
        "from IPython.display import Markdown, display\n\n"
        "REPORT_DIR = Path.cwd()\n"
        "if not (REPORT_DIR / 'analyze_alignment.py').is_file():\n"
        "    REPORT_DIR = REPORT_DIR / 'reports/dronevehicle_alignment_quality'\n"
        "# Re-check every pair and label while reusing the already generated deterministic ECC sample.\n"
        "# Remove --reuse-alignment to regenerate the expensive cross-modal registration estimates.\n"
        "subprocess.run([sys.executable, str(REPORT_DIR / 'analyze_alignment.py'), '--reuse-alignment'], check=True)\n"
        "summary = json.loads((REPORT_DIR / 'summary.json').read_text(encoding='utf-8'))\n"
        "summary['total_exact_triplets'], summary['alignment_sample_count']"
    ),
    nbf.v4.new_markdown_cell("## Data integrity"),
    nbf.v4.new_code_cell(
        "display({\n"
        "    'columns': ['split', 'rgb_files', 'ir_files', 'label_files', 'triplet_coverage_pct', 'dimension_match_pct'],\n"
        "    'rows': [[r[k] for k in ['split', 'rgb_files', 'ir_files', 'label_files', 'triplet_coverage_pct', 'dimension_match_pct']] for r in summary['pairing']]\n"
        "})\n"
        "display(Markdown(f\"**Valid polygon rows:** {summary['box_instances']:,}; malformed rows: {summary['invalid_label_rows']}.\"))"
    ),
    nbf.v4.new_markdown_cell("## Geometric alignment results"),
    nbf.v4.new_code_cell(
        "lines = [\n"
        "    f\"**Trusted estimates:** {summary['trusted_alignment_count']}/{summary['alignment_sample_count']} ({summary['trusted_alignment_rate_pct']:.2f}%).\",\n"
        "    f\"**Residual translation:** median {summary['median_shift_px']:.3f} px; p90 {summary['p90_shift_px']:.3f} px.\",\n"
        "    f\"**Within 2 px:** {summary['share_within_2px_pct']:.2f}%; within 3 px: {summary['share_within_3px_pct']:.2f}%.\",\n"
        "    f\"**Affine check:** median center shift {summary['affine_median_center_shift_px']:.3f} px; p90 maximum corner shift {summary['affine_p90_max_corner_shift_px']:.3f} px.\",\n"
        "    f\"**Object scale:** median target short side {summary['box_size_summary'][0]['short_side_p50_px']:.2f} px; median shift is {summary['median_shift_as_pct_of_median_short_side']:.2f}% of it.\",\n"
        "]\n"
        "display(Markdown('\\n\\n'.join(lines)))\n"
        "display({'columns': list(summary['split_alignment_summary'][0]), 'rows': [list(r.values()) for r in summary['split_alignment_summary']]})"
    ),
    nbf.v4.new_markdown_cell(
        "## Takeaways\n\n"
        "- **近似成立，严格不成立。** 对多数可信样本，可把同一车辆在两模态中的中心理解为"
        "相差约 1–3 px，而不是数学上完全相同的坐标。\n"
        "- **存在序列性例外。** 约 18.8% 的可信估计落在 5–10 px；高位移可视化中道路线、"
        "灯杆和车辆共同出现方向一致的红/绿边缘，说明并非只有车辆热轮廓变化。\n"
        "- **对小目标不可忽略。** 全体目标短边中位数约 25 px，1.74 px 已占约 7%；"
        "5–10 px 对小车的逐像素融合会很明显。\n"
        "- 共享一套 OBB 标签只说明训练管线把两模态当作同坐标监督，不证明两幅原始图像逐像素对齐。"
    ),
]

nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
