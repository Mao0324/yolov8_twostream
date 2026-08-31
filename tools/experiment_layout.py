"""Resolve stable dataset/label/family/seed/attempt output directories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_REGISTRY = ROOT / "experiments/run_registry.yaml"
LEGACY_PROJECT_MARKERS = {
    "DroneVehicle_OBB_FusionTransfer",
    "runs/DroneVehicle_OBB_FusionTransfer",
    "runs/DroneVehicle_OBB_CFGPNet",
    "runs/DroneVehicle_OBB_ProtoHGFNet",
    "runs/DroneVehicle_OBB_SingleModalityTeachers",
    "runs/FLIR_Align_HBB_FusionTransfer",
    "runs_baseline",
}
PLANNED_EXPERIMENT_ROOTS = {
    "ASSANet_ASSAFusion_P345_H2-4-8_DynK3-FFN2_v1": (
        "runs/DroneVehicle_OBB/train-labels=official-v1/assanet/mainline/"
        "ASSA-001__assafusion-p345"
    ),
    "DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS_v1": (
        "runs/DroneVehicle_OBB/train-labels=official-v1/darkact/mainline/"
        "DA-008__target-saliency"
    ),
    "DA016-H25_BPlusClassWinnerBalancedPromptAux-r025-P34_M2DLIFLabels_v1": (
        "runs/DroneVehicle_OBB/train-labels=m2dlif-v1/darkact/da016-prompt-ablation/"
        "D16-010__class-winner-r025"
    ),
    "DA016-H75_BPlusClassWinnerBalancedPromptAux-r075-P34_M2DLIFLabels_v1": (
        "runs/DroneVehicle_OBB/train-labels=m2dlif-v1/darkact/da016-prompt-ablation/"
        "D16-012__class-winner-r075"
    ),
    "CRFormer_FTCrossMerge_P3_H2_R4-P2-StaticDW-NoFFN_v1": (
        "runs/DroneVehicle_OBB/train-labels=official-v1/crformer/mainline/"
        "CRF-001__ftcrossmerge-p3"
    ),
}


def _registry_aliases() -> dict[str, Path]:
    payload = yaml.safe_load(RUN_REGISTRY.read_text(encoding="utf-8"))
    aliases: dict[str, Path] = {}
    for record in payload.get("runs", []):
        run_dir = Path(str(record["run_dir"]))
        experiment_root = ROOT / run_dir.parents[1]
        legacy_name = Path(str(record["legacy_run_dir"])).name
        recorded_name = str(record.get("recorded_name") or "")
        for alias in (legacy_name, recorded_name):
            if alias:
                aliases.setdefault(alias, experiment_root)
        attempt = int(record.get("attempt", 1))
        suffix = str(attempt)
        if attempt > 1 and legacy_name.endswith(suffix):
            aliases.setdefault(legacy_name[: -len(suffix)], experiment_root)
    aliases.update({name: ROOT / path for name, path in PLANNED_EXPERIMENT_ROOTS.items()})
    return aliases


def _legacy_project(project: Any) -> bool:
    if project in {None, ""}:
        return False
    text = str(project).replace("\\", "/").rstrip("/")
    return any(text == marker or text.endswith("/" + marker) for marker in LEGACY_PROJECT_MARKERS)


def resolve_seed(train_args: Mapping[str, Any]) -> int:
    raw = (os.getenv("YOLO_QUEUE_SEED") or os.getenv("YOLO_SEED") or "").strip()
    value = raw if raw else train_args.get("seed", 0)
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"training seed must be an integer, got {value!r}") from exc
    if seed < 0:
        raise ValueError(f"training seed must be non-negative, got {seed}")
    return seed


def _resume_attempt(resume: Any, project: Path) -> str | None:
    if not isinstance(resume, (str, os.PathLike)) or not str(resume).strip():
        return None
    checkpoint = Path(resume).expanduser().resolve()
    if checkpoint.parent.name != "weights":
        return None
    attempt_dir = checkpoint.parent.parent
    if attempt_dir.parent.resolve() != project.resolve() or not attempt_dir.name.startswith("attempt="):
        raise ValueError(f"resume checkpoint is outside the selected seed directory: {checkpoint}")
    return attempt_dir.name


def _next_attempt(project: Path) -> str:
    for number in range(1, 10000):
        name = f"attempt={number:02d}"
        if not (project / name).exists():
            return name
    raise RuntimeError(f"no free attempt directory under {project}")


def organize_train_args(experiment_name: str, train_args: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite legacy project/name arguments into the stable seeded layout.

    Already-organized or unrelated calls are returned unchanged.  A legacy
    project with no registry entry is rejected so it cannot silently recreate
    one of the retired flat output roots.
    """

    organized = dict(train_args)
    project_value = organized.get("project")
    if not _legacy_project(project_value):
        return organized
    aliases = _registry_aliases()
    experiment_root = aliases.get(str(experiment_name)) or aliases.get(str(organized.get("name") or ""))
    if experiment_root is None:
        raise KeyError(f"legacy output has no organized registry entry: {experiment_name!r}")
    seed = resolve_seed(organized)
    project = experiment_root / f"seed={seed:03d}"
    resume_name = _resume_attempt(organized.get("resume"), project)
    organized["project"] = str(project)
    organized["name"] = resume_name or _next_attempt(project)
    organized["seed"] = seed
    organized["exist_ok"] = bool(resume_name)
    return organized


def latest_run(experiment: str, *, train_labels: str, require: str | None = None) -> Path:
    """Return the newest registered attempt for an experiment and label set."""

    payload = yaml.safe_load(RUN_REGISTRY.read_text(encoding="utf-8"))
    candidates = []
    for record in payload.get("runs", []):
        if record.get("experiment") != experiment or record.get("train_labels") != train_labels:
            continue
        path = ROOT / str(record["run_dir"])
        if require and not (path / require).is_file():
            continue
        candidates.append((int(record.get("seed", 0)), int(record.get("attempt", 0)), path))
    if not candidates:
        raise FileNotFoundError(f"no registered run for {experiment} labels={train_labels} require={require}")
    return max(candidates)[-1]


__all__ = ("latest_run", "organize_train_args", "resolve_seed")
