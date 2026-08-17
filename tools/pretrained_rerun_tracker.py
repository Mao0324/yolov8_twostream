#!/usr/bin/env python3
"""Track historical reruns and safely launch new checkpoint-based experiments.

``tracked_train`` remains strict for the fixed ``PT-Rxxx`` campaign. New
experiments should use ``checkpoint_train``; for backward compatibility, a
non-PT-R ID accidentally passed to ``tracked_train`` is routed there instead of
failing after the monitor has already created a run.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "experiments/pretrained_rerun_status.json"
TABLE_PATH = ROOT / "experiments/PRETRAINED_RERUNS.md"
COMPLETED_STATUS = "已正确加载预训练权重重新实验"

RECORDS = (
    {
        "id": "PT-R001",
        "experiment": "YOLOv8 two-stream baseline",
        "script": "train_dronevehicle_baseline.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_baseline.pt",
        "project": "runs_baseline",
        "name": "train",
        "legacy_run": "runs_baseline/train",
    },
    {
        "id": "PT-R002",
        "experiment": "ASSAFusion P345",
        "script": "train_dronevehicle_assafusion_p345.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "ASSANet_ASSAFusion_P345_H2-4-8_DynK3-FFN2_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/ASSANet_ASSAFusion_P345_H2-4-8_DynK3-FFN2_v1",
    },
    {
        "id": "PT-R003",
        "experiment": "ASSAFusion P4 StaticDW",
        "script": "train_dronevehicle.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_assafusion_p4_staticdw_noffn.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "ASSANet_ASSAFusion_P4_H4_StaticDW-NoFFN_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/ASSANet_ASSAFusion_P4_H4_StaticDW-NoFFN_v1",
    },
    {
        "id": "PT-R004",
        "experiment": "PartialChannelASSAFusion P34",
        "script": "train_dronevehicle2.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_partialchannel_assafusion_p34_r4_staticdw_noffn.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "ASSANet_PartialChannelASSAFusion_P34_H2-4_R4-StaticDW-NoFFN_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/ASSANet_PartialChannelASSAFusion_P34_H2-4_R4-StaticDW-NoFFN_v1",
    },
    {
        "id": "PT-R005",
        "experiment": "CRFormer FTCrossMerge P3",
        "script": "train_dronevehicle_crformer_ftatt.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_crformer_ftcrossmerge_p3_r4.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "CRFormer_FTCrossMerge_P3_H2_R4-P2-StaticDW-NoFFN_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/CRFormer_FTCrossMerge_P3_H2_R4-P2-StaticDW-NoFFN_v1",
    },
    {
        "id": "PT-R006",
        "experiment": "Baseline BottleneckRefine P345",
        "script": "train_dronevehicle_baseline_add_bottleneck_refine.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_baseline_add_p345_bottleneck_refine.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "YOLOv8_BottleneckRefine_P345_HNA_E0p5-K1-3_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/YOLOv8_BottleneckRefine_P345_HNA_E0p5-K1-3_v1",
    },
    {
        "id": "PT-R007",
        "experiment": "DarkAct MAA2D + LAFMerge2D V1",
        "script": "train_dronevehicle_darkact_maalaf_v1.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_maa2d_lafmerge_p345_h2-4-8_r4.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_MAA2DLAFMerge_P345_H2-4-8_StaticSaliency-R4-DW3D1-2-2_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_MAA2DLAFMerge_P345_H2-4-8_StaticSaliency-R4-DW3D1-2-2_v1",
    },
    {
        "id": "PT-R008",
        "experiment": "DarkAct StaticMAA + LAF Feedback V2",
        "script": "train_dronevehicle_darkact_maalaf.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_staticmaa_laffeedback_p345_h2-4-8_r4_v2.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_MSK3-5-R4-PosBeta-DW3D1-2-2_v2",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_MSK3-5-R4-PosBeta-DW3D1-2-2_v2",
    },
    {
        "id": "PT-R009",
        "experiment": "DarkAct full-C PaperLAF V3",
        "script": "train_dronevehicle_darkact_maalaf_v3.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_staticmaa_paperlaf_fullc_p345_v3.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_PaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-StaticMAA_v3",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_PaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-StaticMAA_v3",
    },
    {
        "id": "PT-R010",
        "experiment": "DarkAct Post-C2f StaticMAA",
        "script": "train_dronevehicle_darkact_maalaf_postc2f.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_staticmaa_laffeedback_postc2f_p345_v1.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_PostC2f-MSK3-5-R4-PosBeta-DW3D1-2-2_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_PostC2f-MSK3-5-R4-PosBeta-DW3D1-2-2_v1",
    },
    {
        "id": "PT-R011",
        "experiment": "DarkAct Post-C2f per-modal Refine",
        "script": "train_dronevehicle_darkact_maalaf_postc2f_refine.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_staticmaa_refine_laffeedback_postc2f_p345_e025_v1.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_StaticMAA2DRefineLAFMergeFeedback_P345_H2-4-8_PostC2f-RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_StaticMAA2DRefineLAFMergeFeedback_P345_H2-4-8_PostC2f-RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v1",
    },
    {
        "id": "PT-R012",
        "experiment": "DarkAct post-LAF fused Refine",
        "script": "train_dronevehicle_darkact_maalaf_postlaf_refine.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_staticmaa_laffeedback_refine_p345_e025_v1.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_StaticMAA2DLAFMergeFeedbackRefine_P345_H2-4-8_RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_StaticMAA2DLAFMergeFeedbackRefine_P345_H2-4-8_RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v1",
    },
    {
        "id": "PT-R013",
        "experiment": "DarkAct NoStaticMAA",
        "script": "train_dronevehicle_darkact_laffeedback_no_staticmaa.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_laffeedback2d_p345_r4_no_staticmaa_v1.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_LAFMergeFeedback2D_P345_H2-4-8_R4-DW3D1-2-2_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_LAFMergeFeedback2D_P345_H2-4-8_R4-DW3D1-2-2_v1",
    },
    {
        "id": "PT-R014",
        "experiment": "DarkAct target saliency PaperLAF",
        "script": "train_dronevehicle_darkact_target_saliency.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_target_saliency_paperlaf_p345_v1.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS_v1",
    },
    {
        "id": "PT-R015",
        "experiment": "DarkAct target saliency PaperLAF FP32 attention",
        "script": "train_dronevehicle_darkact_target_saliency_fp32safe.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_target_saliency_paperlaf_p345_fp32safe_v2.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn_v2",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn_v2",
    },
    {
        "id": "PT-R016",
        "experiment": "DarkAct target saliency sqrt(HW)",
        "script": "train_dronevehicle_darkact_target_saliency_sqrthw.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_target_saliency_paperlaf_p345_sqrthw_v3.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn-SqrtHW_v3",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn-SqrtHW_v3",
    },
    {
        "id": "PT-R017",
        "experiment": "DarkAct target saliency L2 + learnable temperature",
        "script": "train_dronevehicle_darkact_target_saliency_l2temp.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_target_saliency_paperlaf_p345_l2temp_v4.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn-L2Norm-LearnTemp0p2_v4",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn-L2Norm-LearnTemp0p2_v4",
    },
    {
        "id": "PT-R018",
        "experiment": "DarkAct LAF-only fused Refine P34",
        "script": "train_dronevehicle_darkact_laf_refine_p34.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_laf_refine_p34_no_staticmaa_v1.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_LAFMergeFeedback2D_RefineP34_H2-4-8_R4-NoStaticMAA_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_LAFMergeFeedback2D_RefineP34_H2-4-8_R4-NoStaticMAA_v1",
    },
    {
        "id": "PT-R019",
        "experiment": "DarkAct zero-centered StaticMAA",
        "script": "train_dronevehicle_darkact_zero_centered_maa.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_zerocenteredmaa_laffeedback_p345_v1.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_ZeroCenteredStaticMAA2D_LAFMergeFeedback2D_P345_H2-4-8_R4_v1",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_ZeroCenteredStaticMAA2D_LAFMergeFeedback2D_P345_H2-4-8_R4_v1",
    },
    {
        "id": "PT-R020",
        "experiment": "DarkAct target saliency P34 soft-centerness",
        "script": "train_dronevehicle_darkact_target_saliency_soft_centerness.py",
        "checkpoint": "pre-pth/yolov8s-obb_twostream_darkact_target_saliency_soft_centerness_p34_warmup_v5.pt",
        "project": "DroneVehicle_OBB_FusionTransfer",
        "name": "DarkAct_TargetSaliencyPaperLAF_P34SoftCenterness-W1-0p5-Gain0p025-Warmup10-GateStats_v5",
        "legacy_run": "DroneVehicle_OBB_FusionTransfer/DarkAct_TargetSaliencyPaperLAF_P34SoftCenterness-W1-0p5-Gain0p025-Warmup10-GateStats_v5",
    },
)
RECORD_BY_ID = {record["id"]: record for record in RECORDS}


def repo_path(value: str | Path) -> Path:
    """Resolve repository paths and relocate stale absolute paths from older checkouts."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        return (ROOT / path).resolve()

    resolved = path.resolve()
    if resolved.exists():
        return resolved

    matching_roots = [index for index, part in enumerate(path.parts) if part == ROOT.name]
    for index in reversed(matching_roots):
        relocated = ROOT.joinpath(*path.parts[index + 1 :]).resolve()
        if relocated.exists():
            return relocated
    return resolved


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _count_epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _same_path(left: Any, right: Path) -> bool:
    if not left:
        return False
    return repo_path(str(left)) == repo_path(right)


def _candidate_runs(record: Mapping[str, str]) -> Iterable[Path]:
    project = repo_path(record["project"])
    if not project.is_dir():
        return ()
    name = record["name"]
    candidates = []
    for path in project.iterdir():
        if not path.is_dir() or not path.name.startswith(name):
            continue
        suffix = path.name[len(name) :]
        if suffix == "" or suffix.isdigit():
            candidates.append(path)
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def _inspect_correct_run(
    record: Mapping[str, str], run_dir: Path, *, training_returned: bool = False
) -> Dict[str, Any] | None:
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        return None
    try:
        args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    checkpoint = repo_path(record["checkpoint"])
    model_arg = args.get("model")
    resume_arg = args.get("resume")
    uses_initial_checkpoint = _same_path(model_arg, checkpoint)
    uses_resume_checkpoint = (
        isinstance(resume_arg, (str, Path))
        and bool(str(resume_arg).strip())
        and _same_path(model_arg, repo_path(resume_arg))
    )
    if not (uses_initial_checkpoint or uses_resume_checkpoint):
        return None
    completed = _count_epochs(run_dir / "results.csv")
    target = int(args.get("epochs") or 100)
    has_weights = (run_dir / "weights/last.pt").is_file() and (run_dir / "weights/best.pt").is_file()
    # Reaching the configured epoch count is independently verifiable during a
    # later refresh. A normal model.train() return is also a valid completion
    # signal because Ultralytics may finish early when patience is exhausted.
    if has_weights and completed and (completed >= target or training_returned):
        status = COMPLETED_STATUS
    elif completed:
        status = f"正确加载预训练权重重新实验中（{completed}/{target}）"
    else:
        status = "已正确指定预训练 checkpoint，等待首轮结果"
    return {
        "status": status,
        "run_dir": _repo_relative(run_dir),
        "epochs_completed": completed,
        "epochs_target": target,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _refresh_state_from_artifacts(state: Dict[str, Any]) -> None:
    for record in RECORDS:
        previous = state.get(record["id"], {})
        for run_dir in _candidate_runs(record):
            snapshot = _inspect_correct_run(record, run_dir)
            if snapshot is not None:
                # Preserve a successful-return completion across later manual
                # refreshes when an early-stopped run has fewer than `epochs`
                # rows but its verified artifacts still exist.
                if (
                    previous.get("status") == COMPLETED_STATUS
                    and previous.get("run_dir") == snapshot["run_dir"]
                    and snapshot["epochs_completed"] > 0
                    and (run_dir / "weights/last.pt").is_file()
                    and (run_dir / "weights/best.pt").is_file()
                ):
                    snapshot["status"] = COMPLETED_STATUS
                state[record["id"]] = snapshot
                break
        else:
            state.setdefault(
                record["id"],
                {
                    "status": "待正确加载预训练权重重新实验",
                    "run_dir": "",
                    "epochs_completed": 0,
                    "epochs_target": 100,
                    "updated_at": "",
                },
            )


def _markdown_link(label: str, target: str) -> str:
    if not target:
        return "—"
    path = Path(target)
    relative = Path("..") / path if not path.is_absolute() else path
    return f"[{label}]({relative.as_posix()})"


def _render_table(state: Mapping[str, Any]) -> None:
    lines = [
        "# 双卡预训练权重正确重跑记录",
        "",
        "> 本表由 `tools/pretrained_rerun_tracker.py` 自动维护。旧 YAML 启动目录作为 Scratch 历史保留；已有的正确 checkpoint-direct 运行继续保留。修正后的脚本显式使用 `exist_ok=False`，会自动创建 `name2/name3`，不覆盖已有结果。",
        "",
        f"完成判定：实际 Run 的 `args.yaml:model` 必须等于该行预训练 `.pt`，或与 `args.yaml:resume` 的断点一致；训练正常返回（含正常 early-stop）或达到目标 epoch，且 `results.csv` 非空、`best.pt/last.pt` 均存在。满足后状态自动写为“{COMPLETED_STATUS}”。",
        "",
        "| ID | 实验 | 修正后训练脚本 | 迁移 checkpoint | 已有目录 | 正确重跑目录 | 状态 |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in RECORDS:
        snapshot = state.get(record["id"], {})
        lines.append(
            "| {id} | {experiment} | {script} | {checkpoint} | {legacy} | {run} | {status} |".format(
                id=record["id"],
                experiment=record["experiment"],
                script=_markdown_link(record["script"], record["script"]),
                checkpoint=_markdown_link(Path(record["checkpoint"]).name, record["checkpoint"]),
                legacy=_markdown_link(record["legacy_run"], record["legacy_run"]),
                run=_markdown_link(snapshot.get("run_dir", ""), snapshot.get("run_dir", "")),
                status=snapshot.get("status", "待正确加载预训练权重重新实验"),
            )
        )
    lines.extend(
        [
            "",
            "## 手动刷新",
            "",
            "```bash",
            "python tools/pretrained_rerun_tracker.py refresh",
            "```",
            "",
        ]
    )
    temporary = TABLE_PATH.with_suffix(TABLE_PATH.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(TABLE_PATH)


def _locked_update(mutator=None, refresh=True) -> Dict[str, Any]:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        state = json.loads(raw) if raw else {}
        if refresh:
            _refresh_state_from_artifacts(state)
        if mutator is not None:
            mutator(state)
        handle.seek(0)
        handle.truncate()
        json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        _render_table(state)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return state


def _verify_checkpoint_direct(
    model,
    record: Mapping[str, str],
    checkpoint: Path,
    resume: str | Path | bool = False,
) -> None:
    expected = repo_path(record["checkpoint"])
    if resume:
        if isinstance(resume, bool):
            raise ValueError(f"{record['id']} resume requires an explicit checkpoint path")
        resume_checkpoint = repo_path(resume)
        if checkpoint.resolve() != resume_checkpoint:
            raise ValueError(
                f"{record['id']} resume checkpoint mismatch: {checkpoint} != {resume_checkpoint}"
            )
    elif checkpoint.resolve() != expected:
        raise ValueError(f"{record['id']} checkpoint mismatch: {checkpoint} != {expected}")
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint-direct training requires an existing .pt: {checkpoint}")
    actual_model_arg = (getattr(model, "overrides", {}) or {}).get("model")
    if not _same_path(actual_model_arg, checkpoint):
        raise RuntimeError(
            f"{record['id']} is not checkpoint-direct: model.overrides['model']={actual_model_arg!r}"
        )


def checkpoint_train(model, checkpoint: str | Path, **train_args):
    """Train a new experiment from an explicit checkpoint without PT-R bookkeeping.

    This validates the common accidental YAML/direct-load failure while keeping
    the historical rerun registry completely separate from new experiments.
    """
    checkpoint_path = repo_path(checkpoint)
    resume = train_args.get("resume") or False
    if checkpoint_path.suffix != ".pt" or not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint training requires an existing .pt: {checkpoint_path}")
    if resume and not isinstance(resume, bool):
        resume_path = repo_path(resume)
        if checkpoint_path.resolve() != resume_path.resolve():
            raise ValueError(f"resume checkpoint mismatch: {checkpoint_path} != {resume_path}")
    actual_model_arg = (getattr(model, "overrides", {}) or {}).get("model")
    if not _same_path(actual_model_arg, checkpoint_path):
        raise RuntimeError(
            "new experiment is not checkpoint-direct: "
            f"model.overrides['model']={actual_model_arg!r}, expected {checkpoint_path}"
        )
    return model.train(**train_args)


def tracked_train(model, record_id: str, checkpoint: str | Path, **train_args):
    """Run one explicitly registered historical PT-R rerun.

    Unknown ``PT-Rxxx`` IDs still raise because they usually indicate a typo.
    Other IDs are new experiments and safely fall back to ``checkpoint_train``.
    """
    if record_id not in RECORD_BY_ID:
        if str(record_id).upper().startswith("PT-R"):
            raise KeyError(f"unknown historical pretrained rerun record: {record_id}")
        warnings.warn(
            f"{record_id} is a new experiment, not a historical PT-R rerun; "
            "routing to checkpoint_train without historical bookkeeping.",
            stacklevel=2,
        )
        return checkpoint_train(model, checkpoint, **train_args)
    record = RECORD_BY_ID[record_id]
    checkpoint_path = repo_path(checkpoint)
    resume = train_args.get("resume") or False
    _verify_checkpoint_direct(model, record, checkpoint_path, resume=resume)
    expected_epochs = int(train_args.get("epochs") or 0)

    def mark_started(state):
        state[record_id] = {
            "status": (
                "已正确指定断点 checkpoint，等待续训结果"
                if resume
                else "已正确指定预训练 checkpoint，等待首轮结果"
            ),
            "run_dir": "",
            "epochs_completed": 0,
            "epochs_target": expected_epochs,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    _locked_update(mark_started)
    try:
        results = model.train(**train_args)
    except BaseException as exc:
        def mark_interrupted(state):
            state[record_id] = {
                "status": f"重新实验中断：{type(exc).__name__}",
                "run_dir": _repo_relative(Path(model.trainer.save_dir)) if getattr(model, "trainer", None) else "",
                "epochs_completed": 0,
                "epochs_target": expected_epochs,
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }

        _locked_update(mark_interrupted)
        raise

    run_dir = Path(model.trainer.save_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    snapshot = _inspect_correct_run(record, run_dir, training_returned=True)

    def mark_finished(state):
        state[record_id] = snapshot or {
            "status": "训练返回，但预训练权重/轮数/权重文件校验未通过",
            "run_dir": _repo_relative(run_dir),
            "epochs_completed": _count_epochs(run_dir / "results.csv"),
            "epochs_target": expected_epochs,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    _locked_update(mark_finished)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="refresh", choices=("refresh",))
    parser.parse_args()
    state = _locked_update(refresh=True)
    completed = sum(1 for value in state.values() if value.get("status") == COMPLETED_STATUS)
    print(f"updated {TABLE_PATH}: {completed}/{len(RECORDS)} completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
