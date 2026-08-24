#!/usr/bin/env python3
"""Build the reproducible diagnostic notebook with nbformat."""

from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).resolve().parent
NOTEBOOK = OUT / "baseline_ceiling_diagnosis.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Baseline ceiling diagnosis\n\n"
        "## tl;dr\n\n"
        "This notebook audits every committed test result across the six local "
        "experiment branches, applies the requested `mAP50 > 0.83` filter, and "
        "checks initialization and training-config comparability. Conclusions are "
        "written after the executed outputs below."
    ),
    nbf.v4.new_markdown_cell(
        "## Context & Methods\n\n"
        "The decision is whether to weaken the baseline through learning-rate or "
        "other changes. The controlling comparison is the pretrained two-stream "
        "baseline versus module variants on the same DroneVehicle OBB test split.\n\n"
        "### Key Assumptions\n\n"
        "- A comparable run has test `mAP50 > 0.83`, `args.pretrained: true`, and a "
        "checkpoint (`.pt`) model source.\n"
        "- Aggregate test files cannot estimate seed variance or paired-bootstrap "
        "uncertainty without per-image predictions.\n"
        "- Repeated architecture selection on the test split is treated as a "
        "validation limitation, not as independent confirmation."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import csv, json, subprocess, sys\n"
        "from IPython.display import Markdown, display\n\n"
        "REPORT_DIR = Path.cwd()\n"
        "if not (REPORT_DIR / 'analyze_results.py').is_file():\n"
        "    REPORT_DIR = REPORT_DIR / 'reports/baseline_ceiling_diagnosis'\n"
        "subprocess.run([sys.executable, str(REPORT_DIR / 'analyze_results.py')], check=True)\n"
        "summary = json.loads((REPORT_DIR / 'summary.json').read_text(encoding='utf-8'))\n"
        "summary['comparable_runs'], summary['variant_runs']"
    ),
    nbf.v4.new_markdown_cell("## Data"),
    nbf.v4.new_code_cell(
        "lines = [\n"
        "    f\"**Committed evidence:** {summary['unique_committed_test_artifacts']} unique test artifacts; "
        "{summary['comparable_runs']} comparable runs after filtering.\",\n"
        "    f\"**Test population:** {summary['test_population_signatures'][0]['images']:,} images, "
        "{summary['test_population_signatures'][0]['instances']:,} instances, "
        "split `{summary['test_population_signatures'][0]['split']}`, "
        "image size {summary['test_population_signatures'][0]['imgsz']}.\",\n"
        "    f\"**Training recipes among comparable runs:** {summary['training_config_signature_count']} signature.\",\n"
        "]\n"
        "display(Markdown('\\n\\n'.join(lines)))"
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell(
        "baseline = summary['baseline']\n"
        "best = summary['best_observed']\n"
        "dist = summary['variant_distribution']\n"
        "lines = [\n"
        "    f\"**Baseline:** mAP50={baseline['map50']:.3f}, mAP50-95={baseline['map50_95']:.3f}.\",\n"
        "    f\"**Best observed:** mAP50={best['map50']:.3f} "
        "({best['delta_map50_vs_baseline']*100:+.1f} pp), "
        "mAP50-95={best['map50_95']:.3f} "
        "({best['delta_map50_95_vs_baseline']*100:+.1f} pp).\",\n"
        "    f\"**Across {summary['variant_runs']} comparable variants:** median mAP50={dist['map50_median']:.3f}, "
        "mean gain={dist['mean_delta_map50_vs_baseline']*100:+.2f} pp.\",\n"
        "]\n"
        "display(Markdown('\\n\\n'.join(lines)))"
    ),
    nbf.v4.new_code_cell(
        "with (REPORT_DIR / 'comparable_pretrained_results.csv').open(encoding='utf-8') as handle:\n"
        "    rows = list(csv.DictReader(handle))\n"
        "top = rows[:10]\n"
        "display(Markdown('### Top comparable runs'))\n"
        "display({\n"
        "    'columns': ['family', 'run_name', 'map50', 'map50_95', 'delta_map50_vs_baseline', 'cfg_seed'],\n"
        "    'rows': [[r[k] for k in ['family', 'run_name', 'map50', 'map50_95', 'delta_map50_vs_baseline', 'cfg_seed']] for r in top]\n"
        "})"
    ),
    nbf.v4.new_code_cell(
        "display(Markdown('### Per-class change for the highest-mAP50 model'))\n"
        "display({\n"
        "    'columns': ['class', 'baseline_map50', 'best_map50', 'delta_map50'],\n"
        "    'rows': [[r['class'], r['baseline_map50'], r['best_map50'], r['delta_map50']] "
        "for r in summary['class_deltas_for_best_observed']]\n"
        "})"
    ),
    nbf.v4.new_markdown_cell(
        "## Takeaways\n\n"
        "- The observed best gain is **+0.8 mAP50 percentage points**, while the "
        "comparable variant median is only about **+0.4 pp** above baseline.\n"
        "- Baseline is partly saturated for `car` and `bus`, but not globally: "
        "macro-mAP50 still has 16.7 pp theoretical headroom, and the weak classes "
        "remain far from saturation.\n"
        "- The evidence is **single-seed and test-selected**. Aggregate files do "
        "not establish that +0.8 pp exceeds run-to-run variance.\n"
        "- Do not tune hyperparameters to make the baseline worse. First estimate "
        "variance with repeated seeds; then either keep one common locked recipe or "
        "give baseline and module models the same hyperparameter-search budget."
    ),
]
nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
