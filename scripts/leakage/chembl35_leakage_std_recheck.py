#!/usr/bin/env python
"""Cross-validation pass for the ChEMBL35 leakage audit.

Two things the plain ECFP4 run cannot see:
  1. Salts and charge states. `CC(=O)[O-].[Na+]` and `CC(=O)O` score 0.667, which is
     below any sane threshold, yet they are the same compound. Fix: run both sides
     through the standard RDKit curation pipeline (Normalizer -> LargestFragmentChooser
     -> Uncharger) before fingerprinting. This is what ChEMBL itself does.
  2. Tautomers. ECFP4 sees different NH placement and drops to ~0.6. InChI's mobile-H
     layer merges them, so InChIKey block-1 (the skeleton hash) catches what the
     fingerprint misses.

Writes one row per benchmark compound per corpus with the standardized similarity and
the InChIKey block-1 hit flag, so the net increment over the raw run can be measured
rather than assumed.

Usage: python scripts/leakage/chembl35_leakage_std_recheck.py [corpus_name ...]
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

OUT = "results/chembl35_leakage/per_corpus_std"
DEV = os.environ.get("SIM_DEVICE", "cuda:0")
NPROC = int(os.environ.get("SIM_PROCS", "64"))
CHUNK = 200_000

# name -> (path, smiles column or None for a bare one-per-line file)
CORPORA: dict[str, tuple[str, str | None]] = {
    "chembl34_full": ("/tmp/corpora/chembl34_smiles.txt", None),
    "bindingdb_pre2412": ("/tmp/bdb_smiles_pre2412.txt", None),
    "bindingnet_full": ("/bml/Lyuwei_data/Index_for_BindingNetv1_and_BindingNetv2.csv",
                        "Molecule SMILES"),
    "hiqbind": ("data/hiqbind/full.csv", "smiles"),
    "spindr": ("data/spindr/full.csv", "smiles"),
    "bindingmoad": ("data/bindingmoad_flowr/full.csv", "smiles"),
    "pdbbind2020": ("/bmlfast/Lyuwei/1.Datasets/pdbbind_2020_standardized_training.csv",
                    "compound_iso_smiles"),
    "pdbbind2020_general": ("data/pdbbind2020/full.csv", "smiles"),
    "omol25": ("/bmlfast/Lyuwei/1.Datasets/omol25_biomolecules/omol25_ligands.smi", "smiles"),
    "plinder": ("/bmlfast/Lyuwei/1.Datasets/flowr_root_corpora/plinder.smi", None),
    "kinodata": ("/bmlfast/Lyuwei/1.Datasets/flowr_root_corpora/kinodata.smi", None),
    "kiba3d": ("/bmlfast/Lyuwei/1.Datasets/flowr_root_corpora/kiba3d.smi", None),
    "davis3d": ("/bmlfast/Lyuwei/1.Datasets/flowr_root_corpora/davis3d.smi", None),
    "sair": ("/tmp/sair_max.smi", None),
    "sair_pklonly": ("/tmp/sair_pklonly.smi", None),
    "selfcontrol": ("/tmp/bench_self.smi", None),
}

_GEN = None
_NORM = None
_FRAG = None
_UNCHG = None


def _init() -> None:
    """Per-worker RDKit objects. They are not fork-safe to share, so build them lazily."""
    global _GEN, _NORM, _FRAG, _UNCHG
    _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    _NORM = rdMolStandardize.Normalizer()
    _FRAG = rdMolStandardize.LargestFragmentChooser()
    _UNCHG = rdMolStandardize.Uncharger()


def _standardize(smi: str):
    """SMILES -> (2048-bit uint8 fp, InChIKey block-1) after desalt + neutralize."""
    if _GEN is None:
        _init()
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = _UNCHG.uncharge(_FRAG.choose(_NORM.normalize(m)))
        if m is None or m.GetNumHeavyAtoms() == 0:
            return None
        Chem.SanitizeMol(m)
        bits = np.frombuffer(_GEN.GetFingerprint(m).ToBitString().encode(), "u1") - ord("0")
        ik = Chem.MolToInchiKey(m)
        return bits, (ik.split("-")[0] if ik else "")
    except Exception:
        return None


def standardize_batch(smiles: list[str], desc: str, pool: Pool):
    rows = pool.map(_standardize, smiles, chunksize=500)
    n_bad = sum(r is None for r in rows)
    if n_bad:
        print(f"    {desc}: {n_bad}/{len(rows)} unstandardizable", flush=True)
    return rows


def to_gpu(rows) -> torch.Tensor:
    bits = [r[0] for r in rows if r is not None]
    if not bits:
        return torch.empty((0, 2048), device=DEV, dtype=torch.float16)
    return torch.from_numpy(np.array(bits, dtype=np.uint8)).to(DEV, torch.float16)


def read_corpus(path: str, col: str | None) -> list[str]:
    if path.endswith((".csv", ".tsv")) or col is not None:
        sep = "\t" if path.endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep, usecols=[col] if col else None, low_memory=False)
        s = df[col] if col else df.iloc[:, 0]
    else:
        s = pd.Series([ln.strip() for ln in open(path) if ln.strip()])
    return s.dropna().astype(str).unique().tolist()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    wanted = sys.argv[1:] or list(CORPORA)

    pool = Pool(NPROC)
    bench = pd.read_csv("data/chembl35/chembl35_full_input.csv")
    brows = standardize_batch(bench.compound_iso_smiles.tolist(), "benchmark", pool)
    ok = np.array([r is not None for r in brows])
    print(f"benchmark: {ok.sum()}/{len(bench)} standardized", flush=True)
    q = to_gpu(brows)
    q_pop = q.sum(1)
    b_ik = np.array([r[1] if r is not None else "" for r in brows])

    for name in wanted:
        path, col = CORPORA[name]
        smiles = read_corpus(path, col)
        print(f"\n=== {name}: {len(smiles):,} unique raw SMILES", flush=True)

        best = torch.zeros(q.shape[0], device=DEV, dtype=torch.float16)
        ik_set: set[str] = set()
        for i in range(0, len(smiles), CHUNK):
            rows = standardize_batch(smiles[i:i + CHUNK], name, pool)
            ik_set.update(r[1] for r in rows if r is not None and r[1])
            c = to_gpu(rows)
            if c.numel() == 0:
                continue
            inter = q @ c.T
            denom = q_pop[:, None] + c.sum(1)[None, :] - inter
            best = torch.maximum(best, (inter / denom.clamp(min=1)).max(1).values)
            del inter, denom, c
            torch.cuda.empty_cache()

        sim = np.full(len(bench), np.nan)
        sim[ok] = best.float().cpu().numpy()  # q holds only the ok rows, in order
        hit = np.array([k in ik_set if k else False for k in b_ik])

        df = pd.DataFrame({"compound_id": bench.compound_id,
                           f"simstd_{name}": np.round(sim, 4),
                           f"ik1_{name}": hit})
        df.to_csv(f"{OUT}/{name}.csv", index=False)
        v = pd.Series(sim).dropna()
        print(f"  standardized ECFP4: " + "  ".join(
            f">={t:.2f} {int((v >= t).sum())}" for t in (1.0, 0.9, 0.85, 0.8)) +
            f"  median {v.median():.3f}", flush=True)
        print(f"  InChIKey block-1 hits: {int(hit.sum())}  "
              f"(of which ECFP4std < 0.85: {int((hit & (sim < 0.85)).sum())})", flush=True)

    pool.close()


if __name__ == "__main__":
    main()
