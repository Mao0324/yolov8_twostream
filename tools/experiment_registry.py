#!/usr/bin/env python3
"""Validate, query, and render the repository-local experiment registry."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT / "experiments"
MANIFEST_PATTERN = "*/manifests/*.yaml"
ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*-\d{3}$")
ALLOWED_STATUSES = {
    "auto",
    "planned",
    "created",
    "running",
    "interrupted",
    "trained",
    "tested",
    "archived",
}
ALLOWED_LAUNCH_MODES = {"checkpoint"}
ALLOWED_RECOVERY_CONFIDENCE = {"confirmed", "recovered", "inferred"}
SUPPORTED_TRAINER_CLASSES = {
    "ultralytics.models.yolo.obb.train:OBBTrainer",
    "ultralytics.models.yolo.obb.target_saliency_train:SoftCenternessTargetSaliencyOBBTrainer",
    "ultralytics.models.yolo.obb.target_saliency_train:TargetSaliencyOBBTrainer",
}
RUNNING_MTIME_SECONDS = 30 * 60


class RegistryError(RuntimeError):
    """Raised when the experiment registry is invalid or incomplete."""


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RegistryError(f"YAML root must be a mapping: {path}")
    return data


def repo_path(value: Optional[str]) -> Optional[Path]:
    """Resolve a repository-relative path without requiring it to exist."""

    if value in {None, ""}:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return (ROOT / path).resolve()

    resolved = path.resolve()
    if resolved.exists():
        return resolved

    # Run metadata stores absolute paths.  Keep it portable when the same
    # repository has moved to another volume or parent directory.
    matching_roots = [index for index, part in enumerate(path.parts) if part == ROOT.name]
    for index in reversed(matching_roots):
        relocated = ROOT.joinpath(*path.parts[index + 1 :]).resolve()
        if relocated.exists():
            return relocated
    return resolved


def repo_relative(path: Path) -> str:
    """Return a stable repository-relative path when possible."""

    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def discover_manifest_paths() -> List[Path]:
    return sorted(EXPERIMENTS_ROOT.glob(MANIFEST_PATTERN))


def load_registry() -> List[Dict[str, Any]]:
    manifests: List[Dict[str, Any]] = []
    for path in discover_manifest_paths():
        manifest = _read_yaml(path)
        manifest["_manifest_path"] = repo_relative(path)
        manifests.append(manifest)
    return sorted(manifests, key=lambda item: str(item.get("id", "")))


def get_manifest(experiment_id: str, manifests: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    registry = list(manifests) if manifests is not None else load_registry()
    matches = [item for item in registry if item.get("id") == experiment_id]
    if not matches:
        available = ", ".join(str(item.get("id")) for item in registry)
        raise RegistryError(f"unknown experiment id {experiment_id!r}; available: {available}")
    if len(matches) > 1:
        raise RegistryError(f"duplicate experiment id: {experiment_id}")
    return matches[0]


def load_profile(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    training = manifest.get("training") or {}
    profile_path = repo_path(training.get("profile"))
    if profile_path is None or not profile_path.is_file():
        raise RegistryError(f"{manifest.get('id')}: training profile not found: {training.get('profile')}")
    profile = _read_yaml(profile_path)
    profile["_profile_path"] = repo_relative(profile_path)
    return profile


def _required_mapping(
    value: Any,
    label: str,
    experiment_id: str,
    errors: List[str],
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{experiment_id}: {label} must be a mapping")
        return {}
    return value


def _check_path(
    experiment_id: str,
    label: str,
    value: Optional[str],
    errors: List[str],
    warnings: List[str],
    required: bool = True,
    file_only: bool = False,
) -> None:
    if value in {None, ""}:
        if required:
            errors.append(f"{experiment_id}: missing {label}")
        return
    path = repo_path(value)
    exists = path.is_file() if file_only else path.exists()
    if not exists:
        message = f"{experiment_id}: {label} not found: {value}"
        (errors if required else warnings).append(message)


def _validate_parent_graph(manifests: Sequence[Mapping[str, Any]], errors: List[str]) -> None:
    by_id = {str(item.get("id")): item for item in manifests if item.get("id")}
    for experiment_id, manifest in by_id.items():
        parent_id = manifest.get("parent_id")
        if parent_id is not None and parent_id not in by_id:
            errors.append(f"{experiment_id}: unknown parent_id {parent_id}")

    for start_id in by_id:
        seen = set()
        current = start_id
        while current is not None:
            if current in seen:
                errors.append(f"{start_id}: parent graph contains a cycle at {current}")
                break
            seen.add(current)
            current = by_id.get(current, {}).get("parent_id")


def validate_registry(manifests: Optional[Sequence[Dict[str, Any]]] = None) -> Tuple[List[str], List[str]]:
    """Return validation errors and warnings without mutating repository files."""

    registry = list(manifests) if manifests is not None else load_registry()
    errors: List[str] = []
    warnings: List[str] = []

    ids = [str(item.get("id", "")) for item in registry]
    for experiment_id, count in Counter(ids).items():
        if not experiment_id:
            errors.append("manifest missing id")
        elif count > 1:
            errors.append(f"duplicate experiment id: {experiment_id}")

    _validate_parent_graph(registry, errors)

    for manifest in registry:
        experiment_id = str(manifest.get("id", "<missing-id>"))
        if not ID_PATTERN.fullmatch(experiment_id):
            errors.append(f"{experiment_id}: id must match {ID_PATTERN.pattern}")
        if manifest.get("schema_version") != 1:
            errors.append(f"{experiment_id}: schema_version must be 1")
        if not manifest.get("family"):
            errors.append(f"{experiment_id}: missing family")
        if not manifest.get("title"):
            errors.append(f"{experiment_id}: missing title")
        if not manifest.get("hypothesis"):
            errors.append(f"{experiment_id}: missing hypothesis")
        if not manifest.get("change_from_parent"):
            errors.append(f"{experiment_id}: missing change_from_parent")

        lifecycle = _required_mapping(manifest.get("lifecycle"), "lifecycle", experiment_id, errors)
        status = lifecycle.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{experiment_id}: invalid lifecycle.status {status!r}")
        recovery_confidence = lifecycle.get("recovery_confidence")
        if recovery_confidence not in ALLOWED_RECOVERY_CONFIDENCE:
            errors.append(
                f"{experiment_id}: invalid lifecycle.recovery_confidence {recovery_confidence!r}"
            )

        architecture = _required_mapping(manifest.get("architecture"), "architecture", experiment_id, errors)
        stages = architecture.get("stages")
        flow = architecture.get("flow")
        if not architecture.get("summary"):
            errors.append(f"{experiment_id}: missing architecture.summary")
        if not isinstance(stages, list) or not stages:
            errors.append(f"{experiment_id}: architecture.stages must be a non-empty list")
            stages = []
        if not isinstance(flow, dict):
            errors.append(f"{experiment_id}: architecture.flow must be a mapping")
            flow = {}
        for stage in stages:
            if stage not in flow or not flow.get(stage):
                errors.append(f"{experiment_id}: missing architecture.flow.{stage}")

        training = _required_mapping(manifest.get("training"), "training", experiment_id, errors)
        launch_mode = training.get("launch_mode")
        files = _required_mapping(manifest.get("files"), "files", experiment_id, errors)
        # Historical manifests may outlive source YAMLs that were never
        # recovered. Keep them visible in generated indexes; the selected
        # experiment launcher performs its own strict model-YAML check.
        _check_path(
            experiment_id,
            "files.model_yaml",
            files.get("model_yaml"),
            errors,
            warnings,
            required=False,
            file_only=True,
        )
        _check_path(
            experiment_id,
            "files.init_checkpoint",
            files.get("init_checkpoint"),
            errors,
            warnings,
            required=launch_mode == "checkpoint",
            file_only=True,
        )
        _check_path(
            experiment_id,
            "files.train_entrypoint",
            files.get("train_entrypoint"),
            errors,
            warnings,
            required=False,
            file_only=True,
        )
        _check_path(
            experiment_id,
            "files.migration_script",
            files.get("migration_script"),
            errors,
            warnings,
            required=False,
            file_only=True,
        )
        for label in ("module_files", "integration_files"):
            values = files.get(label)
            if not isinstance(values, list):
                errors.append(f"{experiment_id}: files.{label} must be a list")
                continue
            for value in values:
                _check_path(experiment_id, f"files.{label}", value, errors, warnings, file_only=True)

        _check_path(experiment_id, "training.profile", training.get("profile"), errors, warnings, file_only=True)
        if launch_mode not in ALLOWED_LAUNCH_MODES:
            errors.append(f"{experiment_id}: invalid training.launch_mode {launch_mode!r}")
        trainer_class = training.get("trainer_class")
        if not isinstance(trainer_class, str) or not trainer_class or ":" not in trainer_class:
            errors.append(f"{experiment_id}: trainer_class must use module.path:ClassName")
        elif trainer_class not in SUPPORTED_TRAINER_CLASSES:
            errors.append(f"{experiment_id}: unsupported trainer_class {trainer_class!r}")
        output = _required_mapping(training.get("output"), "training.output", experiment_id, errors)
        if not output.get("project") or not output.get("name"):
            errors.append(f"{experiment_id}: training.output requires project and name")
        output_name = str(output.get("name") or "")
        if output_name and (Path(output_name).name != output_name or output_name in {".", ".."}):
            errors.append(f"{experiment_id}: training.output.name must be one path component")

        if training.get("profile"):
            try:
                profile = load_profile(manifest)
            except RegistryError as exc:
                errors.append(str(exc))
            else:
                if profile.get("task") != "obb":
                    errors.append(f"{experiment_id}: profile.task must be 'obb'")
                if not isinstance(profile.get("environment", {}), dict):
                    errors.append(f"{experiment_id}: profile.environment must be a mapping")
                _check_path(experiment_id, "profile.data", profile.get("data"), errors, warnings, file_only=True)
                if not isinstance(profile.get("args"), dict):
                    errors.append(f"{experiment_id}: profile.args must be a mapping")

        legacy = manifest.get("legacy") or {}
        legacy_run = legacy.get("run_dir")
        _check_path(
            experiment_id,
            "legacy.run_dir",
            legacy_run,
            errors,
            warnings,
            required=False,
        )
        if legacy_run:
            args_path = repo_path(legacy_run) / "args.yaml"
            if not args_path.is_file():
                warnings.append(f"{experiment_id}: legacy args.yaml not found: {repo_relative(args_path)}")
            else:
                try:
                    legacy_args = _read_yaml(args_path)
                except RegistryError as exc:
                    warnings.append(str(exc))
                else:
                    args_name = legacy_args.get("name")
                    output_name = str(output.get("name") or "")
                    args_name_text = str(args_name or "")
                    name_matches = args_name_text == output_name or (
                        args_name_text.startswith(output_name)
                        and args_name_text[len(output_name) :].isdigit()
                    )
                    if args_name and not name_matches:
                        errors.append(
                            f"{experiment_id}: output name differs from legacy args.yaml: "
                            f"{output.get('name')!r} != {args_name!r}"
                        )
                    args_model = legacy_args.get("model")
                    if isinstance(args_model, str) and args_model.endswith(('.yaml', '.yml')):
                        expected_model = repo_path(files.get("model_yaml"))
                        actual_model = repo_path(args_model)
                        if expected_model and actual_model and expected_model != actual_model:
                            errors.append(
                                f"{experiment_id}: model_yaml differs from legacy args.yaml: "
                                f"{files.get('model_yaml')} != {args_model}"
                            )

        reports = manifest.get("reports") or []
        if not isinstance(reports, list):
            errors.append(f"{experiment_id}: reports must be a list")
        else:
            for report in reports:
                _check_path(experiment_id, "reports", report, errors, warnings, required=False)

        if files.get("train_entrypoint") is None:
            warnings.append(f"{experiment_id}: historical train entrypoint is missing")
        if files.get("migration_script") is None:
            warnings.append(f"{experiment_id}: historical migration script is missing")

    return errors, warnings


def count_result_epochs(results_csv: Path) -> int:
    if not results_csv.is_file():
        return 0
    with results_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return max(sum(1 for row in csv.reader(handle) if row) - 1, 0)


def read_test_metrics(test_file: Path) -> Tuple[Optional[float], Optional[float]]:
    if not test_file.is_file():
        return None, None
    for line in test_file.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0] == "all":
            try:
                return float(parts[-2]), float(parts[-1])
            except ValueError:
                continue
    return None, None


def _run_activity_mtime(run_dir: Path) -> float:
    """Return a cheap activity timestamp without recursively walking large run directories."""

    paths = (
        run_dir,
        run_dir / "args.yaml",
        run_dir / "results.csv",
        run_dir / "test_result/test.txt",
        run_dir / "weights/best.pt",
        run_dir / "weights/last.pt",
        run_dir / "provenance/resolved_manifest.yaml",
    )
    mtimes = []
    for path in paths:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes, default=0.0)


def _discover_registered_runs(manifest: Mapping[str, Any]) -> List[Path]:
    """Find attempts created by the ID-based launcher from their provenance bundle."""

    project_value = ((manifest.get("training") or {}).get("output") or {}).get("project")
    project = repo_path(project_value)
    if project is None or not project.is_dir():
        return []

    experiment_id = str(manifest.get("id"))
    matches: List[Path] = []
    for child in project.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        provenance = child / "provenance/resolved_manifest.yaml"
        if not provenance.is_file():
            continue
        try:
            payload = _read_yaml(provenance)
        except (OSError, RegistryError, yaml.YAMLError):
            continue
        recorded = payload.get("experiment") or {}
        if isinstance(recorded, dict) and str(recorded.get("id")) == experiment_id:
            matches.append(child.resolve())
    return sorted(matches, key=lambda path: (_run_activity_mtime(path), path.name))


def _discover_named_runs(manifest: Mapping[str, Any]) -> List[Path]:
    """Find pre-registry script runs by their declared project/name pair.

    Historical scripts wrote ``project=DroneVehicle_OBB_FusionTransfer`` while
    newer manifests reserve ``runs/DroneVehicle_OBB_FusionTransfer`` for the
    ID launcher.  Probe both layouts and validate args.yaml's run name so an
    auto-status manifest can adopt completed legacy artifacts safely.
    """

    output = ((manifest.get("training") or {}).get("output") or {})
    project_value = output.get("project")
    output_name = str(output.get("name") or "")
    if not project_value or not output_name:
        return []
    project = repo_path(project_value)
    candidates = [project / output_name] if project is not None else []
    project_parts = Path(str(project_value)).parts
    if len(project_parts) > 1 and project_parts[0] == "runs":
        candidates.append(ROOT.joinpath(*project_parts[1:], output_name))

    matches = []
    for candidate in candidates:
        args_path = candidate / "args.yaml"
        if not candidate.is_dir() or not args_path.is_file():
            continue
        try:
            args = _read_yaml(args_path)
        except (OSError, RegistryError, yaml.YAMLError):
            continue
        recorded_name = str(args.get("name") or candidate.name)
        # Ultralytics may append a numeric collision suffix to args.name even
        # when a historical wrapper has already fixed the outer save folder.
        if recorded_name == candidate.name or (
            recorded_name.startswith(candidate.name) and recorded_name[len(candidate.name) :].isdigit()
        ):
            matches.append(candidate.resolve())
    return matches


def _run_target_epochs(run_dir: Path, fallback: int) -> int:
    """Prefer the immutable per-run configuration over today's shared profile."""

    candidates = (
        (run_dir / "args.yaml", ("epochs",)),
        (run_dir / "provenance/resolved_manifest.yaml", ("resolved", "train_args", "epochs")),
    )
    for path, keys in candidates:
        if not path.is_file():
            continue
        try:
            value: Any = _read_yaml(path)
        except (OSError, RegistryError, yaml.YAMLError):
            continue
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return fallback


def _inspect_run(run_dir: Path, profile_epochs: int, now: float) -> Dict[str, Any]:
    results = run_dir / "results.csv"
    test_file = run_dir / "test_result/test.txt"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    target_epochs = _run_target_epochs(run_dir, profile_epochs)
    completed_epochs = count_result_epochs(results)
    test_map50, test_map50_95 = read_test_metrics(test_file)
    activity_mtime = _run_activity_mtime(run_dir)
    is_complete = bool(target_epochs and completed_epochs >= target_epochs)

    if test_file.is_file() and (is_complete or not target_epochs):
        status = "tested"
    elif is_complete:
        status = "trained"
    elif results.is_file() and completed_epochs:
        status = "running" if now - results.stat().st_mtime <= RUNNING_MTIME_SECONDS else "interrupted"
    elif best.is_file() or last.is_file():
        status = "running" if now - activity_mtime <= RUNNING_MTIME_SECONDS else "interrupted"
    else:
        status = "running" if activity_mtime and now - activity_mtime <= RUNNING_MTIME_SECONDS else "created"

    return {
        "status": status,
        "epochs_completed": completed_epochs,
        "epochs_target": target_epochs,
        "test_map50": test_map50,
        "test_map50_95": test_map50_95,
        "run_dir": repo_relative(run_dir),
        "args": repo_relative(run_dir / "args.yaml"),
        "results": repo_relative(results),
        "best": repo_relative(best),
        "last": repo_relative(last),
        "test": repo_relative(test_file),
        "activity_mtime": activity_mtime,
    }


def artifact_snapshot(manifest: Mapping[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    """Inspect legacy and ID-launched runs without modifying any artifacts."""

    profile = load_profile(manifest)
    profile_epochs = int((profile.get("args") or {}).get("epochs", 0))
    reference_time = time.time() if now is None else now
    legacy_run_value = (manifest.get("legacy") or {}).get("run_dir")
    legacy_run = repo_path(legacy_run_value)
    registered_runs = _discover_registered_runs(manifest)
    named_runs = _discover_named_runs(manifest)

    known_runs = {path.resolve() for path in (*registered_runs, *named_runs)}
    if legacy_run and legacy_run.is_dir():
        known_runs.add(legacy_run.resolve())
    selected_run = max(
        known_runs,
        key=lambda path: (_run_activity_mtime(path), path.name),
        default=None,
    )

    if selected_run is None:
        snapshot: Dict[str, Any] = {
            "status": "planned",
            "epochs_completed": 0,
            "epochs_target": profile_epochs,
            "test_map50": None,
            "test_map50_95": None,
            "run_dir": None,
            "args": None,
            "results": None,
            "best": None,
            "last": None,
            "test": None,
            "activity_mtime": 0.0,
        }
    else:
        snapshot = _inspect_run(selected_run, profile_epochs, reference_time)

    explicit_status = (manifest.get("lifecycle") or {}).get("status", "auto")
    if explicit_status != "auto":
        snapshot["status"] = explicit_status
    snapshot.update(
        {
            "legacy_run_dir": legacy_run_value,
            "registered_run_dirs": [repo_relative(path) for path in registered_runs],
            "attempt_count": len(known_runs),
        }
    )
    return snapshot


def resolve_training(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve a manifest and shared profile into concrete training settings."""

    profile = load_profile(manifest)
    training = manifest["training"]
    output = training["output"]
    args = dict(profile.get("args") or {})
    data_path = repo_path(profile.get("data"))
    project_path = repo_path(output.get("project"))
    args.update(
        {
            "data": str(data_path),
            "project": str(project_path),
            "name": str(output["name"]),
            "task": str(profile["task"]),
        }
    )
    return {
        "task": str(profile["task"]),
        "environment": dict(profile.get("environment") or {}),
        "args": args,
        "launch_mode": training["launch_mode"],
        "trainer_class": training["trainer_class"],
        "model_yaml": str(repo_path(manifest["files"]["model_yaml"])),
        "init_checkpoint": str(repo_path(manifest["files"].get("init_checkpoint")))
        if manifest["files"].get("init_checkpoint")
        else None,
        "profile_path": str(repo_path(training["profile"])),
    }


def _md_link(value: Optional[str], output_file: Path, label: str) -> str:
    if not value:
        return "—"
    target = repo_path(value)
    relative = os.path.relpath(str(target), str(output_file.parent)).replace(os.sep, "/")
    return f"[{label}]({relative})"


def _metric(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}"


def _epoch_progress(snapshot: Mapping[str, Any]) -> str:
    target = snapshot.get("epochs_target") or 0
    completed = snapshot.get("epochs_completed") or 0
    return f"{completed}/{target}" if target else str(completed)


def _family_tree(family_manifests: Sequence[Mapping[str, Any]], snapshots: Mapping[str, Mapping[str, Any]]) -> str:
    by_id = {str(item["id"]): item for item in family_manifests}
    children: Dict[Optional[str], List[str]] = defaultdict(list)
    for item in family_manifests:
        parent_id = item.get("parent_id") if item.get("parent_id") in by_id else None
        children[parent_id].append(str(item["id"]))
    for values in children.values():
        values.sort()

    lines: List[str] = []

    def walk(experiment_id: str, prefix: str, connector: str) -> None:
        item = by_id[experiment_id]
        snapshot = snapshots[experiment_id]
        lines.append(
            f"{prefix}{connector}{experiment_id} [{snapshot['status']}] {item['architecture']['summary']}"
        )
        child_ids = children.get(experiment_id, [])
        child_prefix = prefix + ("    " if connector == "└── " else "│   " if connector else "")
        for index, child_id in enumerate(child_ids):
            child_connector = "└── " if index == len(child_ids) - 1 else "├── "
            walk(child_id, child_prefix, child_connector)

    roots = children.get(None, [])
    for index, root_id in enumerate(roots):
        if index:
            lines.append("")
        walk(root_id, "", "")
    return "\n".join(lines)


def render_index(manifests: Sequence[Dict[str, Any]], generated_at: str) -> str:
    output_file = EXPERIMENTS_ROOT / "INDEX.md"
    snapshots = {item["id"]: artifact_snapshot(item) for item in manifests}
    counts = Counter(snapshot["status"] for snapshot in snapshots.values())
    count_text = "，".join(f"{key}={counts[key]}" for key in sorted(counts))
    lines = [
        "# 实验总索引",
        "",
        "> 本文件由 `python tools/experiment_registry.py render` 自动生成，请修改 Manifest，不要直接编辑本文件。",
        "",
        f"生成时间：{generated_at}",
        "",
        f"当前登记 {len(manifests)} 个实验：{count_text}。",
        "",
        "| ID | 论文族 | 状态 | 父实验 | 相对父实验的唯一改动 | 架构摘要 | Epoch | Test mAP50-95 | 尝试次数 | 文件链路 |",
        "|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for item in manifests:
        snapshot = snapshots[item["id"]]
        files = item["files"]
        link_items = [
            _md_link(item["_manifest_path"], output_file, "Manifest"),
            _md_link(files.get("model_yaml"), output_file, "YAML"),
            _md_link(files.get("train_entrypoint"), output_file, "旧 Train"),
            _md_link(snapshot.get("run_dir"), output_file, "当前 Run"),
        ]
        if snapshot.get("legacy_run_dir") and snapshot.get("legacy_run_dir") != snapshot.get("run_dir"):
            link_items.append(_md_link(snapshot.get("legacy_run_dir"), output_file, "旧 Run"))
        links = " · ".join(link_items)
        lines.append(
            "| {id} | {family} | {status} | {parent} | {change} | {summary} | {epochs} | {metric} | {attempts} | {links} |".format(
                id=item["id"],
                family=item["family"],
                status=snapshot["status"],
                parent=item.get("parent_id") or "—",
                change=str(item["change_from_parent"]).replace("|", "\\|"),
                summary=str(item["architecture"]["summary"]).replace("|", "\\|"),
                epochs=_epoch_progress(snapshot),
                metric=_metric(snapshot.get("test_map50_95")),
                attempts=snapshot.get("attempt_count", 0),
                links=links,
            )
        )
    lines.extend(
        [
            "",
            "## 查询与训练",
            "",
            "```bash",
            "python tools/experiment_registry.py list",
            "python tools/experiment_registry.py show DA-005",
            "python tools/train_experiment.py DA-005 --device 1,3 --dry-run",
            "python tools/train_experiment.py DA-005 --device 1,3",
            "```",
            "",
            "状态为动态快照；训练进行中时重新执行 `render` 可刷新 Epoch 和状态。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_family_readme(
    family: str,
    manifests: Sequence[Dict[str, Any]],
    generated_at: str,
) -> Tuple[Path, str]:
    family_dir = EXPERIMENTS_ROOT / family.lower()
    output_file = family_dir / "README.md"
    snapshots = {item["id"]: artifact_snapshot(item) for item in manifests}
    lines = [
        f"# {family} 实验谱系",
        "",
        "> 本文件由 Manifest 自动生成；模型结构以各实验链接的 YAML 为准。",
        "",
        f"生成时间：{generated_at}",
        "",
        "## 架构树",
        "",
        "```text",
        _family_tree(manifests, snapshots),
        "```",
        "",
        "## 架构与产物",
        "",
        "| ID | 状态 | 父实验 | 唯一改动 | P3 | P4 | P5 | Epoch | Test mAP50-95 | 尝试次数 | YAML | 当前运行 |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for item in manifests:
        snapshot = snapshots[item["id"]]
        flow = item["architecture"]["flow"]
        lines.append(
            "| {id} | {status} | {parent} | {change} | {p3} | {p4} | {p5} | {epochs} | {metric} | {attempts} | {yaml} | {run} |".format(
                id=item["id"],
                status=snapshot["status"],
                parent=item.get("parent_id") or "—",
                change=str(item["change_from_parent"]).replace("|", "\\|"),
                p3=str(flow.get("P3", "—")).replace("|", "\\|"),
                p4=str(flow.get("P4", "—")).replace("|", "\\|"),
                p5=str(flow.get("P5", "—")).replace("|", "\\|"),
                epochs=_epoch_progress(snapshot),
                metric=_metric(snapshot.get("test_map50_95")),
                attempts=snapshot.get("attempt_count", 0),
                yaml=_md_link(item["files"].get("model_yaml"), output_file, "YAML"),
                run=_md_link(snapshot.get("run_dir"), output_file, "Run"),
            )
        )

    lines.extend(["", "## 实验卡片", ""])
    for item in manifests:
        snapshot = snapshots[item["id"]]
        files = item["files"]
        lines.extend(
            [
                f"### {item['id']} · {item['title']}",
                "",
                f"- 状态：`{snapshot['status']}`，进度 `{_epoch_progress(snapshot)}`，Test mAP50-95 `{_metric(snapshot.get('test_map50_95'))}`。",
                f"- 架构：`{item['architecture']['summary']}`。",
                f"- 假设：{item['hypothesis']}",
                f"- 相对变化：{item['change_from_parent']}",
                f"- 文件：{_md_link(item['_manifest_path'], output_file, 'Manifest')} · {_md_link(files.get('model_yaml'), output_file, 'YAML')} · {_md_link(files.get('train_entrypoint'), output_file, '旧 Train')} · {_md_link(files.get('migration_script'), output_file, '迁移脚本')}。",
                "",
            ]
        )
    return output_file, "\n".join(lines).rstrip() + "\n"


def render_registry_csv(manifests: Sequence[Dict[str, Any]]) -> str:
    import io

    output = io.StringIO(newline="")
    fields = [
        "id",
        "family",
        "title",
        "parent_id",
        "status",
        "recovery_confidence",
        "version",
        "stages",
        "heads",
        "architecture_summary",
        "change_from_parent",
        "model_yaml",
        "train_entrypoint",
        "migration_script",
        "init_checkpoint",
        "trainer_class",
        "output_project",
        "output_name",
        "current_run_dir",
        "legacy_run_dir",
        "attempt_count",
        "epochs_completed",
        "epochs_target",
        "test_map50",
        "test_map50_95",
        "manifest",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in manifests:
        snapshot = artifact_snapshot(item)
        writer.writerow(
            {
                "id": item["id"],
                "family": item["family"],
                "title": item["title"],
                "parent_id": item.get("parent_id") or "",
                "status": snapshot["status"],
                "recovery_confidence": item["lifecycle"]["recovery_confidence"],
                "version": item["architecture"].get("version", ""),
                "stages": ",".join(item["architecture"].get("stages", [])),
                "heads": item["architecture"].get("heads", ""),
                "architecture_summary": item["architecture"]["summary"],
                "change_from_parent": item["change_from_parent"],
                "model_yaml": item["files"].get("model_yaml") or "",
                "train_entrypoint": item["files"].get("train_entrypoint") or "",
                "migration_script": item["files"].get("migration_script") or "",
                "init_checkpoint": item["files"].get("init_checkpoint") or "",
                "trainer_class": item["training"].get("trainer_class") or "",
                "output_project": item["training"]["output"]["project"],
                "output_name": item["training"]["output"]["name"],
                "current_run_dir": snapshot.get("run_dir") or "",
                "legacy_run_dir": snapshot.get("legacy_run_dir") or "",
                "attempt_count": snapshot.get("attempt_count", 0),
                "epochs_completed": snapshot["epochs_completed"],
                "epochs_target": snapshot["epochs_target"],
                "test_map50": "" if snapshot["test_map50"] is None else snapshot["test_map50"],
                "test_map50_95": "" if snapshot["test_map50_95"] is None else snapshot["test_map50_95"],
                "manifest": item["_manifest_path"],
            }
        )
    return output.getvalue()


def render_all(manifests: Optional[Sequence[Dict[str, Any]]] = None) -> List[Path]:
    registry = list(manifests) if manifests is not None else load_registry()
    errors, warnings = validate_registry(registry)
    if errors:
        raise RegistryError("registry validation failed:\n- " + "\n- ".join(errors))
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    written: List[Path] = []

    index_path = EXPERIMENTS_ROOT / "INDEX.md"
    index_path.write_text(render_index(registry, generated_at), encoding="utf-8")
    written.append(index_path)

    csv_path = EXPERIMENTS_ROOT / "registry.csv"
    csv_path.write_text(render_registry_csv(registry), encoding="utf-8")
    written.append(csv_path)

    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in registry:
        by_family[str(item["family"])].append(item)
    for family, family_manifests in sorted(by_family.items()):
        output_path, content = render_family_readme(family, sorted(family_manifests, key=lambda item: item["id"]), generated_at)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        written.append(output_path)

    if warnings:
        print("Warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"- {warning}", file=sys.stderr)
    return written


def _print_list(manifests: Sequence[Dict[str, Any]], family: Optional[str]) -> None:
    selected = [item for item in manifests if family is None or str(item["family"]).lower() == family.lower()]
    if not selected:
        raise RegistryError(f"no experiments found for family {family!r}")
    headers = ("ID", "STATUS", "EPOCH", "TEST", "PARENT", "ARCHITECTURE")
    rows = []
    for item in selected:
        snapshot = artifact_snapshot(item)
        rows.append(
            (
                item["id"],
                snapshot["status"],
                _epoch_progress(snapshot),
                _metric(snapshot["test_map50_95"]),
                item.get("parent_id") or "-",
                item["architecture"]["summary"],
            )
        )
    widths = [max(len(str(row[index])) for row in [headers] + rows) for index in range(len(headers))]
    print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate manifests and referenced paths")
    validate_parser.add_argument("--strict", action="store_true", help="treat warnings as errors")

    subparsers.add_parser("render", help="regenerate INDEX.md, family README files, and registry.csv")

    list_parser = subparsers.add_parser("list", help="show a live compact experiment table")
    list_parser.add_argument("--family", help="case-insensitive family filter")

    show_parser = subparsers.add_parser("show", help="show one resolved experiment card")
    show_parser.add_argument("experiment_id")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifests = load_registry()
    if not manifests:
        raise RegistryError(f"no manifests found under {EXPERIMENTS_ROOT}")

    if args.command == "validate":
        errors, warnings = validate_registry(manifests)
        for warning in warnings:
            print(f"WARNING: {warning}")
        for error in errors:
            print(f"ERROR: {error}")
        if errors or (args.strict and warnings):
            return 1
        print(f"OK: {len(manifests)} manifests validated ({len(warnings)} warnings)")
        return 0

    if args.command == "render":
        for path in render_all(manifests):
            print(f"wrote {repo_relative(path)}")
        return 0

    if args.command == "list":
        _print_list(manifests, args.family)
        return 0

    if args.command == "show":
        manifest = dict(get_manifest(args.experiment_id, manifests))
        manifest["observed"] = artifact_snapshot(manifest)
        print(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False).rstrip())
        return 0

    raise RegistryError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
