#!/usr/bin/env python3
"""Train one registered experiment from its permanent experiment ID.

对应模型 YAML 由当前实验 manifest 的 ``training.model_yaml`` 字段指定。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TRAINERS = {
    "ultralytics.models.yolo.obb.train:OBBTrainer",
    "ultralytics.models.yolo.obb.target_saliency_train:TargetSaliencyOBBTrainer",
}
PROTECTED_TRAIN_KEYS = {
    "device",
    "exist_ok",
    "mode",
    "model",
    "name",
    "project",
    "resume",
    "save_dir",
    "task",
}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiment_registry import (  # noqa: E402
    RegistryError,
    get_manifest,
    load_registry,
    repo_path,
    repo_relative,
    resolve_training,
    validate_registry,
)


def _parse_scalar(value: str) -> Any:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise RegistryError(f"invalid YAML scalar {value!r}: {exc}") from exc

    def is_basic(item: Any) -> bool:
        if item is None or isinstance(item, (bool, int, float, str)):
            return True
        if isinstance(item, list):
            return all(is_basic(child) for child in item)
        if isinstance(item, dict):
            return all(isinstance(key, str) and is_basic(child) for key, child in item.items())
        return False

    if not is_basic(parsed):
        raise RegistryError(f"--set value must contain only basic YAML values, got {type(parsed).__name__}")
    return parsed


def _parse_set(values: Sequence[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise RegistryError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise RegistryError(f"--set key is empty: {item!r}")
        if key in PROTECTED_TRAIN_KEYS:
            protected = ", ".join(sorted(PROTECTED_TRAIN_KEYS))
            raise RegistryError(f"--set cannot override control key {key!r}; protected: {protected}")
        overrides[key] = _parse_scalar(raw_value)
    return overrides


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", help="permanent ID such as DA-005")
    parser.add_argument("--device", help="required for a real launch, e.g. 1,3 or cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--name", help="override the registered output name")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra Ultralytics train override")
    parser.add_argument("--dry-run", action="store_true", help="resolve and print without creating a run")
    parser.add_argument(
        "--check-checkpoint",
        action="store_true",
        help="with --dry-run, load the checkpoint and compare its embedded architecture with model_yaml",
    )
    return parser.parse_args()


def _prepend_pythonpath(root: Path) -> None:
    current = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    root_text = str(root)
    if root_text not in current:
        current.insert(0, root_text)
    os.environ["PYTHONPATH"] = os.pathsep.join(current)


def _patch_trusted_torch_load() -> None:
    import torch

    original = torch.load
    if getattr(original, "_experiment_trusted_patch", False):
        return
    supports_weights_only = "weights_only" in inspect.signature(original).parameters

    def trusted_load(*args, **kwargs):
        if supports_weights_only:
            kwargs.setdefault("weights_only", False)
        else:
            kwargs.pop("weights_only", None)
        return original(*args, **kwargs)

    trusted_load._experiment_trusted_patch = True
    torch.load = trusted_load


def _normalize_architecture(value: Any) -> Any:
    """Normalize YAML data while removing loader-only top-level metadata."""

    if not isinstance(value, dict):
        raise RegistryError("model YAML must be a mapping")
    normalized = copy.deepcopy(value)
    for key in ("yaml_file", "ch"):
        normalized.pop(key, None)
    return normalized


def _architecture_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(_normalize_architecture(dict(value)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_and_check_model(resolved: Mapping[str, Any]):
    _patch_trusted_torch_load()
    from ultralytics import YOLO
    from ultralytics.nn.tasks import OBBModel

    launch_mode = resolved["launch_mode"]
    checkpoint = resolved.get("init_checkpoint")
    model_yaml = Path(resolved["model_yaml"])
    if not model_yaml.is_file():
        raise RegistryError(f"model YAML not found: {model_yaml}")

    if launch_mode != "checkpoint":
        raise RegistryError(f"unsupported launch mode: {launch_mode}")
    if not checkpoint or not Path(checkpoint).is_file():
        raise RegistryError(f"checkpoint launch requires an existing init checkpoint: {checkpoint}")
    model = YOLO(checkpoint, task=resolved["task"])
    if resolved["task"] != "obb" or model.task != "obb" or not isinstance(model.model, OBBModel):
        raise RegistryError(
            f"checkpoint task/model mismatch: expected OBBModel task=obb, "
            f"got {type(model.model).__name__} task={model.task!r}"
        )

    if checkpoint:
        disk_yaml = yaml.safe_load(model_yaml.read_text(encoding="utf-8"))
        embedded_yaml = getattr(model.model, "yaml", None)
        if not isinstance(embedded_yaml, dict):
            raise RegistryError(f"checkpoint does not expose embedded model YAML: {checkpoint}")
        disk_hash = _architecture_hash(disk_yaml)
        embedded_hash = _architecture_hash(embedded_yaml)
        if disk_hash != embedded_hash:
            raise RegistryError(
                "checkpoint architecture does not match manifest model_yaml: "
                f"disk={disk_hash[:12]} embedded={embedded_hash[:12]}"
            )
    return model


def _candidate_paths(project: Path, name: str) -> Iterable[Tuple[Path, Path]]:
    reservation_dir = project / ".run-reservations"
    for index in range(1, 10000):
        suffix = "" if index == 1 else str(index)
        candidate = project / f"{name}{suffix}"
        reservation = reservation_dir / f"{candidate.name}.lock"
        yield candidate, reservation


def _select_run_dir(project: Path, name: str, reserve: bool) -> Tuple[Path, Optional[Path]]:
    for candidate, reservation in _candidate_paths(project, name):
        if candidate.exists() or reservation.exists():
            continue
        if not reserve:
            return candidate, reservation
        reservation.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(str(reservation), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                f"run_name={name}\npid={os.getpid()}\n"
                f"created_at={datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            )
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            reservation.unlink(missing_ok=True)
            continue
        return candidate, reservation
    raise RegistryError(f"unable to reserve a run directory under {project}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> str:
    commands = (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "branch", "--show-current"]),
        ("status", ["git", "status", "--short"]),
    )
    sections: List[str] = []
    for label, command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        value = result.stdout.rstrip() if result.returncode == 0 else f"ERROR: {result.stderr.rstrip()}"
        sections.append(f"[{label}]\n{value}")
    return "\n\n".join(sections) + "\n"


def _declared_source_paths(manifest: Mapping[str, Any], resolved: Mapping[str, Any]) -> List[Path]:
    files = manifest["files"]
    values: List[Optional[str]] = [
        manifest.get("_manifest_path"),
        manifest["training"].get("profile"),
        files.get("model_yaml"),
        files.get("train_entrypoint"),
        files.get("migration_script"),
    ]
    values.extend(files.get("module_files") or [])
    values.extend(files.get("integration_files") or [])
    paths = [repo_path(value) for value in values if value]
    paths.extend(
        [
            ROOT / "tools/experiment_registry.py",
            ROOT / "tools/experiment_trainers.py",
            ROOT / "tools/train_experiment.py",
        ]
    )
    checkpoint = resolved.get("init_checkpoint")
    if checkpoint:
        paths.append(Path(checkpoint))
    data = (resolved.get("args") or {}).get("data")
    if data:
        paths.append(Path(str(data)))
    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved_path = path.resolve()
        if resolved_path not in seen:
            seen.add(resolved_path)
            unique.append(resolved_path)
    return unique


def _snapshot_destination(source: Path, source_root: Path) -> Path:
    try:
        return source_root / source.relative_to(ROOT)
    except ValueError:
        parent_hash = hashlib.sha256(str(source.parent).encode("utf-8")).hexdigest()[:12]
        return source_root / "external" / parent_hash / source.name


def _build_snapshot_stage(
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    train_args: Mapping[str, Any],
    save_dir: Path,
) -> Path:
    stage = Path(tempfile.mkdtemp(prefix=f"yolo-{manifest['id'].lower()}-provenance-"))
    source_root = stage / "source"
    hashes: Dict[str, Dict[str, Any]] = {}
    for path in _declared_source_paths(manifest, resolved):
        if not path.is_file():
            continue
        label = repo_relative(path)
        hashes[label] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        if path.suffix.lower() != ".pt":
            destination = _snapshot_destination(path, source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    manifest_payload = copy.deepcopy(dict(manifest))
    manifest_payload.pop("_manifest_path", None)
    resolved_payload = {
        "snapshot_version": 1,
        "experiment": manifest_payload,
        "resolved": {
            "manifest_path": manifest.get("_manifest_path"),
            "repo_root": str(ROOT),
            "model_yaml": resolved["model_yaml"],
            "init_checkpoint": resolved.get("init_checkpoint"),
            "launch_mode": resolved["launch_mode"],
            "trainer_class": resolved["trainer_class"],
            "save_dir": str(save_dir),
            "train_args": dict(train_args),
            "command": shlex.join(sys.argv),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    }
    (stage / "resolved_manifest.yaml").write_text(
        yaml.safe_dump(resolved_payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(Path(resolved["model_yaml"]), stage / "model.yaml")
    (stage / "source_hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (stage / "git_state.txt").write_text(_git_state(), encoding="utf-8")
    return stage


def _apply_cli_overrides(resolved: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    train_args = dict(resolved["args"])
    explicit = {
        "device": args.device,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "seed": args.seed,
    }
    train_args.update({key: value for key, value in explicit.items() if value is not None})
    train_args.update(_parse_set(args.set))
    if args.name:
        train_args["name"] = args.name
    run_name = str(train_args["name"])
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise RegistryError("run name must be one path component and cannot contain path traversal")
    return train_args


def _print_resolution(manifest: Mapping[str, Any], resolved: Mapping[str, Any], train_args: Mapping[str, Any], save_dir: Path) -> None:
    print(f"experiment: {manifest['id']} ({manifest['title']})")
    print(f"architecture: {manifest['architecture']['summary']}")
    print(f"manifest: {manifest['_manifest_path']}")
    print(f"model_yaml: {resolved['model_yaml']}")
    print(f"checkpoint: {resolved.get('init_checkpoint')}")
    print(f"trainer: {resolved['trainer_class']}")
    print(f"save_dir: {save_dir}")
    print("train_args:")
    print(yaml.safe_dump(dict(train_args), allow_unicode=True, sort_keys=True).rstrip())


def main() -> int:
    args = _parse_args()
    manifests = load_registry()
    errors, warnings = validate_registry(manifests)
    if errors:
        raise RegistryError("registry validation failed:\n- " + "\n- ".join(errors))
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    manifest = get_manifest(args.experiment_id, manifests)
    resolved = resolve_training(manifest)
    if resolved["trainer_class"] not in SUPPORTED_TRAINERS:
        supported = ", ".join(sorted(SUPPORTED_TRAINERS))
        raise RegistryError(
            f"trainer_class {resolved['trainer_class']!r} has no static DDP provenance wrapper; "
            f"supported: {supported}"
        )
    train_args = _apply_cli_overrides(resolved, args)
    project = Path(train_args["project"])
    name = str(train_args["name"])
    preview_dir, _ = _select_run_dir(project, name, reserve=False)
    _print_resolution(manifest, resolved, train_args, preview_dir)

    if args.dry_run:
        if args.check_checkpoint:
            _load_and_check_model(resolved)
            print("checkpoint architecture: OK")
        print("dry-run: no run directory created and no training started")
        return 0

    if not args.device:
        raise RegistryError("--device is required for a real training launch")

    for key, value in resolved["environment"].items():
        os.environ[str(key)] = str(value)
    _prepend_pythonpath(ROOT)
    from tools.training_monitor import create_monitor

    monitor = create_monitor(name)
    model = _load_and_check_model(resolved)

    save_dir, reservation = _select_run_dir(project, name, reserve=True)
    train_args["project"] = str(save_dir.parent)
    train_args["name"] = save_dir.name
    train_args["exist_ok"] = True
    # Keep args.model as the checkpoint so DDP children reload migrated weights instead of the YAML alone.
    if resolved.get("init_checkpoint"):
        train_args["model"] = resolved["init_checkpoint"]

    stage: Optional[Path] = None
    try:
        stage = _build_snapshot_stage(manifest, resolved, train_args, save_dir)
        # Materialize before Trainer construction so dataset/config failures remain traceable.
        shutil.copytree(stage, save_dir / "provenance", dirs_exist_ok=True)
        os.environ["YOLO_EXPERIMENT_SNAPSHOT_STAGE"] = str(stage)
        if reservation:
            os.environ["YOLO_EXPERIMENT_RESERVATION"] = str(reservation)

        from tools.experiment_trainers import get_trainer_wrapper

        trainer = get_trainer_wrapper(resolved["trainer_class"])
        print(f"launching {manifest['id']} -> {save_dir}")
        monitor.run(model.train, trainer=trainer, **train_args)
        return 0
    finally:
        if reservation:
            reservation.unlink(missing_ok=True)
        if stage:
            shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
