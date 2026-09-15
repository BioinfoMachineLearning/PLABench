#!/usr/bin/env python
"""Ligand-level leakage audit for the two CASP16 affinity series.

Same protocol as the ChEMBL35 audit: standardize both sides with the RDKit curation
pipeline (Normalizer -> LargestFragmentChooser -> Uncharger), fingerprint with ECFP4
(Morgan radius 2, 2048 bits), and record for every CASP16 ligand its maximum Tanimoto
similarity to any ligand in each training corpus, plus an InChIKey block-1 exact hit.

L1000 is Chymase (17 ligands), L3000 is Autotaxin (123 ligands).

Writes results/casp16_leakage/{per_corpus.csv,summary.csv}.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.leakage.chembl35_leakage_std_recheck import (  # noqa: E402
    CORPORA, read_corpus, standardize_batch, to_gpu,
)
from multiprocessing import Pool  # noqa: E402

DEV = os.environ.get("SIM_DEVICE", "cuda:0")
CHUNK = 200_000
OUT = "results/casp16_leakage"

SERIES = {
    "L1000": ("data/casp16_data/labels/L1000_exper_affinity.csv", "ligand_smiles"),
    "L3000": ("data/casp16_data/labels/L3000_exper_affinity.csv", "Structure"),
}
# sair_pklonly and selfcontrol are diagnostics for the ChEMBL35 audit, not corpora
SKIP = {"sair_pklonly", "selfcontrol"}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    pool = Pool(int(os.environ.get("SIM_PROCS", "64")))

    frames = []
    for name, (path, col) in SERIES.items():
        d = pd.read_csv(path, encoding="utf-8-sig")
        d = d.rename(columns={"Target ID": "ligand_id", col: "smiles"})
        frames.append(d[["ligand_id", "smiles"]].assign(series=name))
    q = pd.concat(frames, ignore_index=True)
    print(f"CASP16 ligands: {len(q)} "
          f"({', '.join(f'{k} {int((q.series == k).sum())}' for k in SERIES)})", flush=True)

    rows = standardize_batch(q.smiles.tolist(), "casp16", pool)
    ok = np.array([r is not None for r in rows])
    print(f"standardized: {int(ok.sum())}/{len(q)}", flush=True)
    Q = to_gpu(rows)
    q_pop = Q.sum(1)
    q_ik = np.array([r[1] if r is not None else "" for r in rows])

    out = q[["series", "ligand_id", "smiles"]].copy()
    for name in [c for c in CORPORA if c not in SKIP]:
        path, col = CORPORA[name]
        if not os.path.exists(path):
            print(f"=== {name}: MISSING at {path}, skipped", flush=True)
            continue
        smiles = read_corpus(path, col)
        best = torch.zeros(Q.shape[0], device=DEV, dtype=torch.float16)
        ik_set: set[str] = set()
        for i in range(0, len(smiles), CHUNK):
            crows = standardize_batch(smiles[i:i + CHUNK], name, pool)
            ik_set.update(r[1] for r in crows if r is not None and r[1])
            C = to_gpu(crows)
            if C.numel() == 0:
                continue
            inter = Q @ C.T
            denom = q_pop[:, None] + C.sum(1)[None, :] - inter
            best = torch.maximum(best, (inter / denom.clamp(min=1)).max(1).values)
            del inter, denom, C
            torch.cuda.empty_cache()
        sim = np.full(len(q), np.nan)
        sim[ok] = best.float().cpu().numpy()
        out[f"sim_{name}"] = np.round(sim, 4)
        out[f"ik_{name}"] = [k in ik_set if k else False for k in q_ik]
        v = pd.Series(sim).dropna()
        print(f"=== {name:<22} ({len(smiles):>9,} SMILES)  " + "  ".join(
            f">={t:.2f} {int((v >= t).sum()):>3}" for t in (1.0, 0.9, 0.85, 0.8)) +
            f"  median {v.median():.3f}  ik {int(out[f'ik_{name}'].sum())}", flush=True)

    scols = [c for c in out.columns if c.startswith("sim_")]
    icols = [c for c in out.columns if c.startswith("ik_")]
    out["max_sim"] = out[scols].max(axis=1)
    out["top_corpus"] = out[scols].idxmax(axis=1).str.replace("sim_", "", regex=False)
    out["any_inchikey_hit"] = out[icols].any(axis=1)
    out.to_csv(f"{OUT}/per_corpus.csv", index=False)

    print("\n===== union over all corpora =====")
    for s, g in out.groupby("series"):
        print(f"{s}  n={len(g)}  " + "  ".join(
            f">={t:.2f} {int((g.max_sim >= t).sum())}" for t in (1.0, 0.95, 0.9, 0.85, 0.8)) +
            f"  median {g.max_sim.median():.3f}  mean {g.max_sim.mean():.3f}"
            f"  InChIKey exact {int(g.any_inchikey_hit.sum())}")
    print("\ncorpus holding the closest analog, for ligands at >=0.85:")
    print(out[out.max_sim >= 0.85].groupby(["series", "top_corpus"]).size().to_string()
          or "  none")

    # FLOWR.ROOT's own corpora, called out separately
    flowr = [c for c in ("sair", "plinder", "kinodata", "kiba3d", "davis3d",
                         "bindingmoad", "spindr", "hiqbind") if f"sim_{c}" in out]
    out["flowr_max"] = out[[f"sim_{c}" for c in flowr]].max(axis=1)
    print(f"\nFLOWR.ROOT corpora only ({', '.join(flowr)}):")
    for s, g in out.groupby("series"):
        print(f"  {s}  " + "  ".join(f">={t:.2f} {int((g.flowr_max >= t).sum())}"
                                     for t in (1.0, 0.9, 0.85)) +
              f"  median {g.flowr_max.median():.3f}  max {g.flowr_max.max():.3f}")
    out.to_csv(f"{OUT}/per_corpus.csv", index=False)
    print(f"\nwrote {OUT}/per_corpus.csv")
    pool.close()


if __name__ == "__main__":
    main()
