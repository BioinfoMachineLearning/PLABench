#!/usr/bin/env python
"""Does the CASP16 conclusion survive removing the leaked ligands?

Two removal sets on the L3000 (Autotaxin) series:
  sim   - the 26 ligands within ECFP4 Tanimoto 0.85 of any training-corpus ligand
  label - the ligands whose Autotaxin affinity itself is in BindingDB (entered before
          June 2023) or in SAIR, i.e. the label and not merely the structure was public

L1000 is left alone: only one of its 17 ligands has any corpus match at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plabench.analysis.units import dg_to_pkd  # noqa: E402

PRED = {
    "Boltz2": "outputs/boltz2/boltz2_casp16_l3000/latest/predictions.csv",
    "FlowDock": "outputs/flowdock/af3_casp16_l3000_stage1/latest/predictions.csv",
    "FLOWR.ROOT": "outputs/flowr_root/flowr_root_af3_casp16_l3000_stage1/latest/predictions.csv",
    "Graph_RG": "outputs/haiping/haiping_af3_casp16_l3000_stage1/latest/predictions.csv",
    "LCDD-team": "outputs/bapred/af3_casp16_l3000_stage1/latest/predictions.csv",
    "MFE": "outputs/mfe/mfe_af3_casp16_l3000_stage1/latest/predictions.csv",
    "DeepDTA": "outputs/deepdta/sequence_casp16_l3000/latest/predictions.csv",
    "LLF": "outputs/llf/sequence_casp16_l3000/latest/predictions.csv",
    "MixingDTA": "outputs/mixingdta/sequence_casp16_l3000/latest/predictions.csv",
}


def read(path: str) -> pd.DataFrame:
    return (pd.read_csv(path, header=None, names=["pred", "ligand_id"], dtype={"ligand_id": str})
            .assign(ligand_id=lambda d: d.ligand_id.str.strip()))


def main() -> None:
    gt = (pd.read_csv("data/casp16_data/labels/L3000_exper_affinity.csv", encoding="utf-8-sig")
          .rename(columns={"Target ID": "ligand_id", "binding_affinity": "aff"})[["ligand_id", "aff"]])
    gt["ligand_id"] = gt.ligand_id.astype(str).str.strip()
    gt["pKd"] = dg_to_pkd(gt.aff)

    lk = pd.read_csv("results/casp16_leakage/per_corpus.csv")
    lk = lk[lk.series == "L3000"]
    sim_hits = set(lk[lk.max_sim >= 0.85].ligand_id)
    # the on-target-label set, measured in casp16_leak_dating.py / the SAIR parquet check
    label_hits = set(pd.read_csv("results/casp16_leakage/ontarget_label_hits.csv").ligand_id)

    d = gt.copy()
    for name, path in PRED.items():
        d = d.merge(read(path).rename(columns={"pred": name}), on="ligand_id", how="left")
    d["Boltz2 + FLOWR.ROOT"] = (d["Boltz2"] + d["FLOWR.ROOT"]) / 2

    models = list(PRED) + ["Boltz2 + FLOWR.ROOT"]
    subsets = {"all": d,
               "minus similarity hits": d[~d.ligand_id.isin(sim_hits)],
               "minus on-target labels": d[~d.ligand_id.isin(label_hits)]}
    for k, v in subsets.items():
        print(f"{k}: n={len(v)}")

    rows = []
    for m in models:
        rec = {"model": m}
        for k, v in subsets.items():
            g = v[v[m].notna()]
            rec[f"tau_{k}"] = round(kendalltau(g[m], g.pKd)[0], 3)
            rec[f"rho_{k}"] = round(spearmanr(g[m], g.pKd)[0], 3)
            rec[f"r_{k}"] = round(pearsonr(g[m], g.pKd)[0], 3)
        rows.append(rec)
    t = pd.DataFrame(rows)
    cols = ["model"] + [f"{p}_{k}" for p in ("tau", "rho", "r") for k in subsets]
    t = t[cols].sort_values("tau_all", ascending=False)
    t.to_csv("results/casp16_leakage/l3000_sensitivity.csv", index=False)
    print()
    print(t.to_string(index=False))
    print("\nwrote results/casp16_leakage/l3000_sensitivity.csv")


if __name__ == "__main__":
    main()
