"""Checkpoint discovery and on-demand installation for the benchmark runner."""

from __future__ import annotations

import csv
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
THIRD_PARTY_TABLE = CHECKPOINT_DIR / "THIRD_PARTY.tsv"
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".model", ".bin", ".safetensors")


def _values(value) -> Iterable[str]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _values(child)
    elif isinstance(value, str):
        yield value


def _path(value: str, roots: list[Path]) -> Path | None:
    """Resolve a config value that looks like a checkpoint against candidate roots.

    Most configs are repository-relative; FlowDock's ``model_checkpoint`` is
    relative to ``models_source_dir``. The first root that contains the file
    wins; a missing file is reported under the repository root.
    """
    if not value.lower().endswith(CHECKPOINT_SUFFIXES):
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    for root in roots:
        if (root / candidate).exists():
            return root / candidate
    return roots[0] / candidate


def _sequence_paths(model: str, dataset: str, task: str, fold: int) -> list[Path]:
    """Return dynamic CV paths used by the three sequence-model loaders."""
    dset = dataset.lower()
    if dset not in {"davis_warm", "kiba_warm", "davis_cold_drug", "davis_cold_target", "kiba_cold_drug", "kiba_cold_target"}:
        if model == "deepdta":
            return [REPO_ROOT / "forks" / "DeepDTA-Pytorch" / "pdbbind2020_refined91_results" / "model_refined91.pt"]
        if model == "llf":
            return [REPO_ROOT / "forks" / "LLF" / "models" / "pdbbind" / "best_model_full.pth"]
        if model == "mixingdta":
            root = REPO_ROOT / "forks" / "MixingDTA" / "MEETA" / "results_refined_91"
            return [root / name / "PDBbind_Refined_91" / f"{fold}_fold_valid_best_checkpoint.pth" for name in ("results_none", "results_all_pair", "results_drug", "results_reversed", "result_integration")]
        return []
    prefix = "DAVIS" if "davis" in dset else "KIBA"
    if model == "deepdta":
        root = REPO_ROOT / "forks" / "DeepDTA-Pytorch"
        if "warm" in task or "warm" in dset:
            directory = ("davis_warm_results" if prefix == "DAVIS" else "kiba_warm_results")
            directory = root / directory / (f"davis_warm_fold_{fold}" if prefix == "DAVIS" else f"fold_{fold}")
            filename = "best_model_fold_%d.pt" % fold if prefix == "DAVIS" else "deepdta-warm-fold_%d-prk12-ldk8.pt" % fold
        elif "drug" in task or "drug" in dset:
            directory = root / ("davis_drug_results" if prefix == "DAVIS" else "kiba_drug_results") / (f"davis_drug_fold_{fold}" if prefix == "DAVIS" else f"kiba_drug_fold_{fold}")
            filename = "deepdta-fold_%d-prk12-ldk8.pt" % fold
        else:
            directory = root / ("davis_results" if prefix == "DAVIS" else "kiba_target_cold_results") / (f"davis_target_fold_{fold}" if prefix == "DAVIS" else f"kiba_target_cold_fold_{fold}")
            filename = "deepdta-fold_%d-prk12-ldk8.pt" % fold if prefix == "DAVIS" else "best_model_fold_%d.pt" % fold
        return [directory / filename]
    if model == "llf":
        root = REPO_ROOT / "forks" / "LLF" / "models"
        if "warm" in task or "warm" in dset:
            return [root / f"{dset}/fold_{fold}/best_model.pth"]
        if "drug" in task or "drug" in dset:
            return [root / f"{prefix.lower()}_drug_cv/best_model_fold_{fold}.model"]
        return [root / f"{prefix.lower()}_target_cv/fold_{fold}/best_model.pth"]
    if model == "mixingdta":
        root = REPO_ROOT / "forks" / "MixingDTA"
        if "warm" in task or "warm" in dset:
            cases = ("results_none", "results_all_pair", "results_drug", "results_protein", "results_drug_and_protein", "results_reversed")
            return [root / "model_weights" / case / prefix / f"{fold}_fold_valid_best_checkpoint.pth" for case in cases] + [root / "model_weights" / "result_integration" / prefix / f"{fold}_fold_valid_best_checkpoint.pth"]
        case = "results_cold_drug_case4_protein" if "drug" in task or "drug" in dset else "results_cold_target_case3_drug"
        return [root / "MEETA" / case / prefix / f"{fold}_fold_valid_best_checkpoint.pth"]
    return []


def required_checkpoints(model_cfg, dataset_cfg) -> list[Path]:
    model = str(model_cfg.get("name", "")).lower()
    dataset = str(dataset_cfg.get("name", "")).lower()
    task = str(dataset_cfg.get("task", model_cfg.get("task", ""))).lower()
    fold = int(dataset_cfg.get("fold", model_cfg.get("fold", 1)))
    roots = [REPO_ROOT]
    source_dir = model_cfg.get("models_source_dir")
    if source_dir:
        roots.append(REPO_ROOT / str(source_dir))
    paths = [candidate for value in list(_values(model_cfg)) + list(_values(dataset_cfg)) for candidate in [_path(value, roots)] if candidate is not None]
    paths += _sequence_paths(model, dataset, task, fold)
    return list(dict.fromkeys(paths))


def _third_party_index() -> tuple[dict[Path, str], list[tuple[Path, str]]]:
    """Read THIRD_PARTY.tsv into exact-path and path-prefix lookups."""
    exact: dict[Path, str] = {}
    prefixes: list[tuple[Path, str]] = []
    if not THIRD_PARTY_TABLE.exists():
        return exact, prefixes
    with THIRD_PARTY_TABLE.open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 4 or not row[0] or row[0].startswith("#"):
                continue
            model, source_path, target_rel, url = row[:4]
            hint = f"{model}: {url}"
            for rel in (source_path, target_rel):
                rel = rel.split(" ")[0]
                if not rel.startswith(("checkpoints/", "forks/")):
                    continue
                if "{" in rel or "<" in rel:
                    stem = rel.split("{")[0].split("<")[0]
                    prefixes.append(((REPO_ROOT / stem).resolve(strict=False), hint))
                else:
                    exact[(REPO_ROOT / rel).resolve(strict=False)] = hint
    return exact, prefixes


def _third_party_hint(path: Path) -> str | None:
    exact, prefixes = _third_party_index()
    resolved = path.resolve(strict=False)
    if resolved in exact:
        return exact[resolved]
    for stem, hint in prefixes:
        if str(resolved).startswith(str(stem)):
            return hint
    return None


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def ensure_checkpoints(model_cfg, dataset_cfg) -> None:
    missing = [path for path in required_checkpoints(model_cfg, dataset_cfg) if not path.exists()]
    if not missing:
        return
    external = {path: hint for path in missing for hint in [_third_party_hint(path)] if hint}
    own = [path for path in missing if path not in external]
    if external:
        lines = "\n".join(f"  {_rel(path)}  <- {hint}" for path, hint in external.items())
        print(f"Third-party checkpoints are missing; install them from the official source listed in checkpoints/THIRD_PARTY.tsv:\n{lines}", flush=True)
    if own:
        script = REPO_ROOT / "scripts" / "download_checkpoints.sh"
        if not script.exists():
            raise RuntimeError(f"Missing checkpoints and installer: {script}")
        names = ", ".join(_rel(path) for path in own)
        if os.environ.get("PLABENCH_AUTO_DOWNLOAD_CHECKPOINTS", "1").lower() in {"0", "false", "no"}:
            raise RuntimeError(f"Missing checkpoints ({names}); automatic download disabled")
        print(f"Missing checkpoints detected; downloading archive for: {names}", flush=True)
        subprocess.run(["bash", str(script)], cwd=str(REPO_ROOT), check=True)
    still_missing = [path for path in required_checkpoints(model_cfg, dataset_cfg) if not path.exists()]
    if still_missing:
        raise RuntimeError("Checkpoint installation incomplete: " + ", ".join(map(_rel, still_missing)))
