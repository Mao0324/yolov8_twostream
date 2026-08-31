#!/usr/bin/env python3
"""Plan, apply, and verify the 2026-08-31 experiment-run layout migration.

The migration moves complete run directories atomically.  It never edits files
inside a run, merges two runs, overwrites a destination, or deletes artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_REGISTRY = ROOT / "experiments/run_registry.yaml"


@dataclass(frozen=True)
class RunSpec:
    source: str
    dataset_root: str
    train_labels: str
    family: str
    series: str
    experiment: str
    attempt: int = 1
    seed: int = 0
    task: str = "obb"

    @property
    def target(self) -> str:
        return (
            f"runs/{self.dataset_root}/train-labels={self.train_labels}/"
            f"{self.family}/{self.series}/{self.experiment}/"
            f"seed={self.seed:03d}/attempt={self.attempt:02d}"
        )


def _spec(
    source: str,
    labels: str,
    family: str,
    series: str,
    experiment: str,
    attempt: int = 1,
    *,
    dataset_root: str = "DroneVehicle_OBB",
    task: str = "obb",
) -> RunSpec:
    return RunSpec(source, dataset_root, labels, family, series, experiment, attempt, 0, task)


def run_specs() -> list[RunSpec]:
    official = "official-v1"
    m2dlif = "m2dlif-v1"
    old = "DroneVehicle_OBB_FusionTransfer"
    fusion = "runs/DroneVehicle_OBB_FusionTransfer"
    specs = [
        _spec("runs_baseline/train", official, "baseline", "mainline", "BL-001__add-p345"),
        _spec("runs/Baseline_M2DLIFLabels_v1", m2dlif, "baseline", "mainline", "BL-001__add-p345", 2),
        _spec(
            f"{old}/ASSALAF_AlignFuseFeedbackRefine_P34_H2-4_R4-StaticDW3-NoFFN_v1",
            official,
            "assanet",
            "laf",
            "ASSA-004__align-fuse-feedback-p34",
        ),
    ]

    darkact = [
        ("DarkAct_MAA2DLAFMerge_P345_H2-4-8_StaticSaliency-R4-DW3D1-2-2_v1", "DA-001__maa2d-lafmerge-p345", 1),
        ("DarkAct_MAA2DLAFMerge_P345_H2-4-8_StaticSaliency-R4-DW3D1-2-2_v12", "DA-001__maa2d-lafmerge-p345", 2),
        ("DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_MSK3-5-R4-PosBeta-DW3D1-2-2_v2", "DA-002__staticmaa-laf-feedback-p345", 1),
        ("DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_MSK3-5-R4-PosBeta-DW3D1-2-2_v22", "DA-002__staticmaa-laf-feedback-p345", 2),
        ("DarkAct_PaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-StaticMAA_v3", "DA-003__paper-laf-fullc-p345", 1),
        ("DarkAct_PaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-StaticMAA_v32", "DA-003__paper-laf-fullc-p345", 2),
        ("DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_PostC2f-MSK3-5-R4-PosBeta-DW3D1-2-2_v1", "DA-004__postc2f-staticmaa-laf-p345", 1),
        ("DarkAct_StaticMAA2DLAFMergeFeedback_P345_H2-4-8_PostC2f-MSK3-5-R4-PosBeta-DW3D1-2-2_v12", "DA-004__postc2f-staticmaa-laf-p345", 2),
        ("DarkAct_StaticMAA2DRefineLAFMergeFeedback_P345_H2-4-8_PostC2f-RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v1", "DA-005__postc2f-refine-laf-p345", 1),
        ("DarkAct_StaticMAA2DRefineLAFMergeFeedback_P345_H2-4-8_PostC2f-RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v12", "DA-005__postc2f-refine-laf-p345", 2),
        ("DarkAct_StaticMAA2DLAFMergeFeedbackRefine_P345_H2-4-8_RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v1", "DA-006__laf-feedback-refine-p345", 1),
        ("DarkAct_StaticMAA2DLAFMergeFeedbackRefine_P345_H2-4-8_RefineE0p25-K1-3-G0-MSK3-5-R4-PosBeta-DW3D1-2-2_v12", "DA-006__laf-feedback-refine-p345", 2),
        ("DarkAct_LAFMergeFeedback2D_P345_H2-4-8_R4-DW3D1-2-2_v1", "DA-007__laf-feedback-p345", 1),
        ("DarkAct_LAFMergeFeedback2D_P345_H2-4-8_R4-DW3D1-2-2_v12", "DA-007__laf-feedback-p345", 3),
        ("DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn_v2", "DA-009__target-saliency-fp32", 1),
        ("DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn-SqrtHW_v3", "DA-010__target-saliency-sqrthw", 1),
        ("DarkAct_TargetSaliencyPaperLAFMergeFeedback2D_P345_HNA_FullC-DilK3-PoolK3-OBBMaskS-FP32Attn-L2Norm-LearnTemp0p2_v4", "DA-011__target-saliency-l2temp", 1),
        ("DarkAct_LAFMergeFeedback2D_RefineP34_H2-4-8_R4-NoStaticMAA_v1", "DA-012__laf-refine-p34", 1),
        ("DarkAct_ZeroCenteredStaticMAA2D_LAFMergeFeedback2D_P345_H2-4-8_R4_v1", "DA-013__zero-centered-maa", 1),
        ("DarkAct_TargetSaliencyPaperLAF_P34SoftCenterness-W1-0p5-Gain0p025-Warmup10-GateStats_v5", "DA-014__soft-centerness-p34", 2),
        ("DarkAct_PostC2fStaticMAA_LAFFeedback_P345_KLDProbIoU025_v1", "DA-017__postc2f-kld025", 1),
        ("DarkAct_PostC2fStaticMAA_LAFFeedback_P345_KLDProbIoU05_v1", "DA-018__postc2f-kld05", 2),
    ]
    specs.extend(
        _spec(f"{old}/{name}", official, "darkact", "mainline", experiment, attempt)
        for name, experiment, attempt in darkact
    )
    specs.extend(
        [
            _spec(
                f"{fusion}/DarkAct_DA015_PostC2f_DisagreementLAF_P34_H2-4-8_R4_v1",
                official,
                "darkact",
                "mainline",
                "DA-015__disagreement-laf-p34",
                4,
            ),
            _spec(
                f"{fusion}/DarkAct_DA016_SemanticDisagreementLAF_P34_H2-4-8_R4-NoStaticMAA_v1",
                official,
                "darkact",
                "mainline",
                "DA-016__semantic-disagreement-p34",
                4,
            ),
            _spec(
                f"{fusion}/DarkAct_DA016_SemanticDisagreementLAF_P34_H2-4-8_R4-NoStaticMAA_M2DLIFLabels_v1",
                m2dlif,
                "darkact",
                "mainline",
                "DA-016__semantic-disagreement-p34",
            ),
        ]
    )

    p2d_names = {
        1: "P2D-001_IRPrompt-P34_PostC2f_v1",
        2: "P2D-002_DualPrompt-P34_PostC2f_v2",
        3: "P2D-003_DualPrompt-GDER-P4_PostC2f_v3",
        4: "P2D-004_DualPrompt-P345-GDER-P45_PostC2f_v4",
        5: "P2D-005_DualPrompt-GDER-P345_PostC2f_v5",
        6: "P2D-006_DualPrompt-GDER-P45-NoStaticMAA_v6",
        7: "P2D-007_IRSpatial-RGBGlobal-P34_PostC2f_v7",
        8: "P2D-008_DualPrompt-IdentityGDER-P45-NoStaticMAA_v8",
        9: "P2D-009_DualPrompt-RGBGlobal-IdentityGDER-P45-NoStaticMAA_v9",
        10: "P2D-010_DualPrompt-RGBGlobal-IdentityGDER-FactorizedP4-NoStaticMAA_v10",
        11: "P2D-011_IRPrompt-P34-NoStaticMAA-NoGDER_PostC2f_v11",
        12: "P2D-012_IRPrompt-AsymIdentityGDER-P45-NoStaticMAA_PostC2f_v12",
        13: "P2D-013_IRPrompt-AsymIdentityGDER-P4-NoStaticMAA_PostC2f_v13",
        14: "P2D-014_IRPrompt-AsymModExpert-P4-NoStaticMAA_PostC2f_v14",
        15: "P2D-015_IRPrompt-AttExpert-P4-NoStaticMAA_PostC2f_v15",
    }
    p2d_slugs = {
        1: "irprompt-p34",
        2: "dualprompt-p34",
        3: "dualprompt-gder-p4",
        4: "dualprompt-gder-p45",
        5: "dualprompt-gder-p345",
        6: "dualprompt-gder-p45-no-staticmaa",
        7: "irspatial-rgbglobal-p34",
        8: "dualprompt-identitygder-p45",
        9: "rgbglobal-identitygder-p45",
        10: "factorized-p4",
        11: "irprompt-p34-no-gder",
        12: "asymidentity-p45",
        13: "asymidentity-p4",
        14: "asymmodexpert-p4",
        15: "attexpert-p4",
    }
    specs.extend(
        _spec(
            f"{old}/{p2d_names[index]}",
            official,
            "p2det",
            "mainline",
            f"P2D-{index:03d}__{p2d_slugs[index]}",
        )
        for index in range(1, 16)
    )
    specs.extend(
        [
            _spec(
                f"{old}/P2D-015_IRPrompt-AttExpert-P4-NoStaticMAA_PostC2f_v15_M2DLIFLabels_v1",
                m2dlif,
                "p2det",
                "mainline",
                "P2D-015__attexpert-p4",
            ),
            _spec(
                f"{fusion}/P2D-016_DualReliabilityPrompt-P34-PriorLAF-NoGDER_v16_M2DLIFLabels_v12",
                m2dlif,
                "p2det",
                "mainline",
                "P2D-016__dual-reliability-p34",
                2,
            ),
            _spec(
                f"{fusion}/P2D-017_ObjectReliabilityPrompt-P34-MonotonicPriorLAF-NoGDER_v17_M2DLIFLabels_v22",
                m2dlif,
                "p2det",
                "mainline",
                "P2D-017__object-reliability-p34",
                2,
            ),
            _spec(
                f"{fusion}/P2D-018_TrueObjectReliabilityPrompt-P34-SignPreservingLAF-NoGDER_v18_M2DLIFLabels_v12",
                m2dlif,
                "p2det",
                "mainline",
                "P2D-018__true-object-reliability-p34",
                2,
            ),
        ]
    )

    da016 = [
        ("DA016-A_OriginalSemanticDisagreementLAF-P34_M2DLIFLabels_v1", "D16-001__original", 1),
        ("DA016-B_DualReliabilityPromptAuxOnly-P34_M2DLIFLabels_v1", "D16-002__prompt-aux-p34", 1),
        ("DA016-B_DualReliabilityPromptAuxOnly-P34_M2DLIFLabels_v12", "D16-002__prompt-aux-p34", 2),
        ("DA016-C_DualReliabilityPrompt-ZeroInitBoundedResidual-P34_M2DLIFLabels_v1", "D16-003__prompt-residual-p34", 1),
        ("DA016-C_DualReliabilityPrompt-ZeroInitBoundedResidual-P34_M2DLIFLabels_v12", "D16-003__prompt-residual-p34", 2),
        ("DA016-D_DualReliabilityPrompt-P3ZeroInitBoundedResidual-P4AuxOnly_M2DLIFLabels_v1", "D16-004__p3-residual-p4-aux", 1),
        ("DA016-E_DualReliabilityPrompt-P34Residual-P5AuxOnly_M2DLIFLabels_v1", "D16-005__p34-residual-p5-aux", 1),
        ("DA016-F_EPlusP5ZeroInitBoundedResidual_M2DLIFLabels_v1", "D16-006__p345-residual", 1),
        ("DA016-G_BPlusP5DualReliabilityPromptAuxOnly_M2DLIFLabels_v1", "D16-007__p345-aux", 1),
        ("DA016-BDetach_PromptGradientCausalControl-P34_M2DLIFLabels_v1", "D16-008__prompt-detach-control", 1),
        ("DA016-H_BPlusClassWinnerBalancedPromptAux-P34_M2DLIFLabels_v1", "D16-009__class-winner-r100", 1),
        ("DA016-H50_BPlusClassWinnerBalancedPromptAux-r050-P34_M2DLIFLabels_v1", "D16-011__class-winner-r050", 1),
        ("DA016-J_DualReliabilityPrompt-P4ZeroInitBoundedResidual-P3AuxOnly_M2DLIFLabels_v1", "D16-013__p4-residual-p3-aux", 1),
    ]
    specs.extend(
        _spec(f"{fusion}/{name}", m2dlif, "darkact", "da016-prompt-ablation", experiment, attempt)
        for name, experiment, attempt in da016
    )

    cfgp = [
        ("CFGP_CF001_CrossCEA_ASAF_P345_v1", "CFGP-001__crosscea-asaf-p345"),
        ("CFGP_CF002_CrossCEA_ASAF_P34_v1", "CFGP-002__crosscea-asaf-p34"),
        ("CFGP_CF003_CrossCEAOnly_P345_v1", "CFGP-003__crosscea-only-p345"),
    ]
    specs.extend(
        _spec(f"runs/DroneVehicle_OBB_CFGPNet/{name}", official, "cfgpnet", "mainline", experiment)
        for name, experiment in cfgp
    )
    proto = [
        ("ProtoHGF_PHG001_HardK6K3_P345_v1", "mainline", "PHG-001__hard-p345"),
        ("ProtoHGF_PHG002_HardK6K3_P34_v1", "mainline", "PHG-002__hard-p34"),
        ("ProtoHGF_PHG003_SoftK6K3_P345_v1", "mainline", "PHG-003__soft-p345"),
        ("PHG003_DDP_DEBUG", "debug", "PHG-003__soft-p345"),
    ]
    specs.extend(
        _spec(f"runs/DroneVehicle_OBB_ProtoHGFNet/{name}", official, "protohgfnet", series, experiment)
        for name, series, experiment in proto
    )
    teachers = [
        ("P2D-Teacher_RGBOnly_M2DLIFLabels_v1", "TCH-001__rgb-only", 1),
        ("P2D-Teacher_RGBOnly_M2DLIFLabels_v12", "TCH-001__rgb-only", 2),
        ("P2D-Teacher_RGBOnly_M2DLIFLabels_v13", "TCH-001__rgb-only", 3),
        ("P2D-Teacher_IROnly_M2DLIFLabels_v1", "TCH-002__ir-only", 1),
    ]
    specs.extend(
        _spec(f"runs/DroneVehicle_OBB_SingleModalityTeachers/{name}", m2dlif, "p2det", "teachers", experiment, attempt)
        for name, experiment, attempt in teachers
    )
    specs.append(
        _spec(
            "runs/FLIR_Align_HBB_FusionTransfer/FLIR_DarkAct_SemanticDisagreementLAF_P34_R4_NoStaticMAA_scales_3Class_HBB",
            "aligned-3class-v1",
            "darkact",
            "mainline",
            "FLIR-001__semantic-disagreement-p34",
            dataset_root="FLIR_Align_HBB",
            task="detect",
        )
    )
    return specs


REPORT_MOVES = {
    "DroneVehicle_OBB_FusionTransfer/P2D_bottleneck_analysis": "reports/p2det/bottleneck-analysis-v1",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(directory: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    files = symlinks = total_bytes = 0
    for path in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            symlinks += 1
            payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
            digest.update(b"L\0" + relative.encode() + b"\0" + payload + b"\0")
        elif path.is_file():
            files += 1
            size = path.stat().st_size
            total_bytes += size
            file_hash = _sha256_file(path)
            digest.update(b"F\0" + relative.encode() + b"\0" + str(size).encode() + b"\0" + file_hash.encode() + b"\0")
    return {
        "file_count": files,
        "symlink_count": symlinks,
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def _validate_specs(specs: Iterable[RunSpec], *, migrated: bool) -> None:
    specs = list(specs)
    sources = [spec.source for spec in specs]
    targets = [spec.target for spec in specs]
    if len(sources) != len(set(sources)):
        raise RuntimeError("duplicate source path in migration specification")
    if len(targets) != len(set(targets)):
        raise RuntimeError("duplicate target path in migration specification")
    for spec in specs:
        source = ROOT / spec.source
        target = ROOT / spec.target
        if migrated:
            if source.exists() or not target.is_dir():
                raise RuntimeError(f"invalid migrated state: source={source.exists()} target={target.is_dir()} {spec.source}")
        elif not source.is_dir() or target.exists():
            raise RuntimeError(f"invalid pre-migration state: source={source.is_dir()} target={target.exists()} {spec.source}")


def _record(spec: RunSpec, before: dict[str, int | str]) -> dict[str, object]:
    args_path = ROOT / spec.target / "args.yaml"
    args = {}
    if args_path.is_file():
        with args_path.open("r", encoding="utf-8", errors="replace") as handle:
            loaded = yaml.safe_load(handle)
            if isinstance(loaded, dict):
                args = loaded
    record = asdict(spec)
    record.update(
        {
            "legacy_run_dir": record.pop("source"),
            "run_dir": spec.target,
            "recorded_name": args.get("name"),
            "recorded_project": args.get("project"),
            "inventory": before,
        }
    )
    return record


def plan(specs: list[RunSpec]) -> None:
    _validate_specs(specs, migrated=False)
    print(f"runs: {len(specs)}")
    for spec in specs:
        print(f"{spec.source} -> {spec.target}")
    for source, target in REPORT_MOVES.items():
        source_path, target_path = ROOT / source, ROOT / target
        if not source_path.is_dir() or target_path.exists():
            raise RuntimeError(f"invalid report move: {source} -> {target}")
        print(f"REPORT {source} -> {target}")


def apply(specs: list[RunSpec]) -> None:
    _validate_specs(specs, migrated=False)
    inventories = {spec.source: inventory(ROOT / spec.source) for spec in specs}
    report_inventories = {source: inventory(ROOT / source) for source in REPORT_MOVES}
    for spec in specs:
        source, target = ROOT / spec.source, ROOT / spec.target
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        after = inventory(target)
        if after != inventories[spec.source]:
            raise RuntimeError(f"inventory mismatch after moving {spec.source} -> {spec.target}")
        print(f"MOVED {spec.source} -> {spec.target}")
    for source, target in REPORT_MOVES.items():
        source_path, target_path = ROOT / source, ROOT / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(target_path)
        if inventory(target_path) != report_inventories[source]:
            raise RuntimeError(f"inventory mismatch after moving report {source} -> {target}")
        print(f"MOVED REPORT {source} -> {target}")
    payload = {
        "schema_version": 1,
        "layout_version": "2026-08-31",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rules": {
            "path_axes": ["dataset_task", "train_labels", "family", "series", "experiment", "seed", "attempt"],
            "run_files_modified": False,
        },
        "runs": [_record(spec, inventories[spec.source]) for spec in specs],
        "reports": [
            {
                "legacy_path": source,
                "path": target,
                "inventory": report_inventories[source],
            }
            for source, target in REPORT_MOVES.items()
        ],
    }
    RUN_REGISTRY.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"WROTE {RUN_REGISTRY.relative_to(ROOT)}")


def verify(specs: list[RunSpec]) -> None:
    _validate_specs(specs, migrated=True)
    registry = yaml.safe_load(RUN_REGISTRY.read_text(encoding="utf-8"))
    registered = {entry["run_dir"]: entry for entry in registry["runs"]}
    if set(registered) != {spec.target for spec in specs}:
        raise RuntimeError("run registry paths do not match migration specification")
    for spec in specs:
        observed = inventory(ROOT / spec.target)
        if observed != registered[spec.target]["inventory"]:
            raise RuntimeError(f"registered inventory mismatch: {spec.target}")
    for source, target in REPORT_MOVES.items():
        if (ROOT / source).exists() or not (ROOT / target).is_dir():
            raise RuntimeError(f"invalid migrated report state: {source} -> {target}")
    print(f"OK: {len(specs)} runs and {len(REPORT_MOVES)} report directories verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "apply", "verify"))
    args = parser.parse_args()
    specs = run_specs()
    if args.command == "plan":
        plan(specs)
    elif args.command == "apply":
        apply(specs)
    else:
        verify(specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
