#!/usr/bin/env python
"""Re-evaluate every model on the leakage-filtered ChEMBL35 set.

Recomputes the per-target metrics, the N-weighted summary and the per-family Pearson
table on both the original 7,650 rows and the filtered 4,862, so every number in the
paper can be updated and the size of each change can be read off directly.

Bootstrap confidence intervals are resampled over targets, not over rows: the targets
are the independent units here, and a per-row bootstrap would badly understate the
uncertainty on a benchmark where one target contributes 159 ligands and another 16.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plabench.analysis.metrics import calculate_metrics  # noqa: E402
from plabench.analysis.units import dg_to_pkd  # noqa: E402

MODELS = {
    "boltz2": "outputs/boltz2/boltz2_chembl35_full/latest/predictions.csv",
    "flowr_root": "outputs/flowr_root/flowr_root_af3_chembl35_full/latest/predictions.csv",
    "flowdock": "outputs/flowdock/af3_chembl35_full/latest/predictions.csv",
    "haiping": "outputs/haiping/haiping_af3_chembl35_full/latest/predictions.csv",
    "mfe": "outputs/mfe/mfe_af3_chembl35_full/latest/predictions.csv",
    "bapred": "outputs/bapred/af3_chembl35_full/latest/predictions.csv",
    "llf": "outputs/llf/sequence_chembl35_full/latest/predictions.csv",
    "mixingdta": "outputs/mixingdta/sequence_chembl35_full/latest/predictions.csv",
    "deepdta": "outputs/deepdta/sequence_chembl35_full/latest/predictions.csv",
}
METRICS = ["RMSE", "MSE", "Pearson", "Spearman", "Kendall", "CI", "Rm2"]
N_BOOT = 2000
SEED = 20260908


def load_predictions() -> pd.DataFrame:
    frames = []
    for name, path in MODELS.items():
        p = pd.read_csv(path, header=None, names=["prediction", "compound_id"])
        frames.append(p.assign(model=name))
    P = pd.concat(frames).pivot_table(index="compound_id", columns="model",
                                      values="prediction", aggfunc="mean")
    P["ensemble"] = (P.boltz2 + P.flowr_root) / 2
    return P


def per_target(truth: pd.DataFrame, P: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    d = truth.merge(P, left_on="compound_id", right_index=True, how="inner")
    rows = []
    for m in models:
        for t, g in d.groupby("target"):
            g = g[g[m].notna()]
            if len(g) < 3:
                continue
            r = calculate_metrics(g[m].to_numpy(), g.affinity.to_numpy())
            rows.append({"Model": m, "Target": t, "Target_class": g.Target_class.iloc[0],
                         "N": len(g), **r})
    out = pd.DataFrame(rows)
    for c in METRICS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def weighted(pt: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for m, g in pt.groupby("Model"):
        rec = {"Model": m, "N_total": int(g.N.sum()), "N_targets": g.Target.nunique()}
        for c in METRICS:
            v, w = g[c].to_numpy(), g.N.to_numpy()
            ok = ~np.isnan(v)
            rec[c] = round(float(np.average(v[ok], weights=w[ok])), 4) if ok.any() else np.nan
        # bootstrap over targets for the three headline ranking metrics
        idx = np.arange(len(g))
        for c in ("Pearson", "Spearman", "CI"):
            v, w = g[c].to_numpy(), g.N.to_numpy()
            draws = []
            for _ in range(N_BOOT):
                s = rng.choice(idx, size=len(idx), replace=True)
                vv, ww = v[s], w[s]
                ok = ~np.isnan(vv)
                if ok.any():
                    draws.append(np.average(vv[ok], weights=ww[ok]))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            rec[f"{c}_lo"], rec[f"{c}_hi"] = round(float(lo), 4), round(float(hi), 4)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("Pearson", ascending=False)


def per_family(pt: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """N-weighted Pearson per family. A dual-class target such as Hydrolase|Transferase
    counts toward both families, matching the convention of the published table."""
    e = pt.copy()
    e["Target_class"] = e.Target_class.str.split("|")
    e = e.explode("Target_class")
    e["Target_class"] = e.Target_class.str.strip()
    rows = []
    for cls, g in e.groupby("Target_class"):
        rec = {"Target_class": cls,
               "n_targets": g[g.Model == models[0]].Target.nunique(),
               "n_compounds": int(g[g.Model == models[0]].N.sum())}
        for m in models:
            h = g[g.Model == m]
            ok = h.Pearson.notna()
            rec[m] = round(float(np.average(h.Pearson[ok], weights=h.N[ok])), 4) if ok.any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("n_compounds", ascending=False)


# CASP16 contributes two extra single-target series that the published family table
# folds in: Chymase (L1000, n=17) into Protease and Autotaxin (L3000, n=123) into
# Hydrolase. They are unaffected by the ChEMBL35 leakage filter and carry over as is.
#
# LLF's key is the bare `sequence_casp16_l*`, which loads the default
# forks/LLF/models/pdbbind/best_model_full.pth -- the same checkpoint as
# sequence_chembl35_full, and the PDBbind-retrained model the paper describes. The
# suffixed runs (_stage1, _stage2_only, _v1/_v4/_v5/_v6, _protenix_v3) are separate
# training experiments that each override model_path; _stage1 in particular is LLF
# retrained on FLOWR.ROOT's stage-1 data and is not the benchmarked model.
CASP16_FOLD = [("Protease", "casp16_l1000", 17), ("Hydrolase", "casp16_l3000", 123)]
CASP16_KEYS = {
    "casp16_l1000": {"boltz2": "boltz2_casp16_l1000", "flowdock": "af3_casp16_l1000_stage1",
                     "flowr_root": "flowr_root_af3_casp16_l1000_stage1",
                     "haiping": "haiping_af3_casp16_l1000_stage1",
                     "bapred": "af3_casp16_l1000_stage1", "mfe": "mfe_af3_casp16_l1000_stage1",
                     "deepdta": "sequence_casp16_l1000", "llf": "sequence_casp16_l1000",
                     "mixingdta": "sequence_casp16_l1000"},
    "casp16_l3000": {"boltz2": "boltz2_casp16_l3000", "flowdock": "af3_casp16_l3000_stage1",
                     "flowr_root": "flowr_root_af3_casp16_l3000_stage1",
                     "haiping": "haiping_af3_casp16_l3000_stage1",
                     "bapred": "af3_casp16_l3000_stage1", "mfe": "mfe_af3_casp16_l3000_stage1",
                     "deepdta": "sequence_casp16_l3000", "llf": "sequence_casp16_l3000",
                     "mixingdta": "sequence_casp16_l3000"},
}
CASP16_ENS = {
    "casp16_l1000": ("outputs/boltz2/boltz2_casp16_l1000/latest/predictions.csv",
                     "outputs/flowr_root/flowr_root_af3_casp16_l1000_stage1/latest/predictions.csv",
                     "data/casp16_data/labels/L1000_exper_affinity.csv"),
    "casp16_l3000": ("outputs/boltz2/boltz2_casp16_l3000/latest/predictions.csv",
                     "outputs/flowr_root/flowr_root_af3_casp16_l3000_stage1/latest/predictions.csv",
                     "data/casp16_data/labels/L3000_exper_affinity.csv"),
}


def casp16_pearson(models: list[str]) -> dict[str, dict[str, float]]:
    from scipy.stats import pearsonr
    bs = pd.read_csv("results/benchmark_summary.csv")
    out: dict[str, dict[str, float]] = {}
    for ds, keys in CASP16_KEYS.items():
        vals = {}
        for m, key in keys.items():
            r = bs[(bs.Model == m) & (bs.Dataset == key)]
            vals[m] = float(r.Pearson.iloc[0]) if len(r) else np.nan
        bp, fp, gtp = CASP16_ENS[ds]
        rd = lambda p: pd.read_csv(p, header=None, names=["pred", "compound_id"], dtype={"compound_id": str}) \
            .assign(compound_id=lambda d: d.compound_id.str.strip())
        gt = pd.read_csv(gtp, encoding="utf-8-sig").rename(
            columns={"Target ID": "compound_id", "binding_affinity": "affinity"})[["compound_id", "affinity"]]
        gt["compound_id"] = gt.compound_id.astype(str).str.strip()
        gt["affinity"] = dg_to_pkd(gt.affinity)  # the file stores a dG in kcal/mol
        d = rd(bp).rename(columns={"pred": "b"}).merge(
            rd(fp).rename(columns={"pred": "f"}), on="compound_id").merge(gt, on="compound_id")
        vals["ensemble"] = float(pearsonr((d.b + d.f) / 2, d.affinity)[0])
        out[ds] = vals
    return out


def fold_casp16(pf: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    c16 = casp16_pearson(models)
    out = pf.set_index("Target_class").copy()
    for cls, ds, n16 in CASP16_FOLD:
        if cls not in out.index:
            continue
        n0 = out.loc[cls, "n_compounds"]
        for m in models:
            v0, v1 = out.loc[cls, m], c16[ds].get(m, np.nan)
            if np.isnan(v1):
                continue
            out.loc[cls, m] = round((v0 * n0 + v1 * n16) / (n0 + n16), 4)
        out.loc[cls, "n_compounds"] = n0 + n16
        out.loc[cls, "n_targets"] = out.loc[cls, "n_targets"] + 1
    return out.reset_index().sort_values("n_compounds", ascending=False)


def main() -> None:
    tmap = pd.read_csv("data/chembl35/target_class_map.csv") \
        .rename(columns={"UniProt_ID": "target", "Target_class": "Target_class"})
    P = load_predictions()
    models = list(MODELS) + ["ensemble"]

    for tag, path in [("full", "data/chembl35/chembl35_full_input.csv"),
                      ("filtered", "data/chembl35/chembl35_filtered_input.csv")]:
        b = pd.read_csv(path)[["compound_id", "affinity"]]
        b["target"] = b.compound_id.str.split("_CHEMBL").str[0]
        b = b.merge(tmap[["target", "Target_class"]], on="target", how="left")
        b["Target_class"] = b.Target_class.fillna("Unclassified")
        b = b.reset_index(drop=True)

        pt = per_target(b, P, models)
        pt.to_csv(f"results/chembl35/per_target_evaluation_chembl35_{tag}.csv", index=False)
        w = weighted(pt)
        w.to_csv(f"results/chembl35/weighted_summary_chembl35_{tag}.csv", index=False)
        pf = per_family(pt, models)
        pf.to_csv(f"results/chembl35/per_class_pearson_wide_chembl35_{tag}.csv", index=False)
        pc = fold_casp16(pf, models)
        pc.to_csv(f"results/chembl35/per_class_pearson_wide_chembl35_casp16_{tag}.csv", index=False)
        print(f"\n===== {tag}: {len(b):,} rows / {b.target.nunique()} targets =====")
        print(w[["Model", "N_total", "N_targets", "Pearson", "Pearson_lo", "Pearson_hi",
                 "Spearman", "Kendall", "CI", "RMSE"]].to_string(index=False))

        # which model leads each family, and how much of the set the ensemble owns
        tot = pc.n_compounds.sum()
        best = pc.set_index("Target_class")[models].idxmax(axis=1)
        ens = pc.set_index("Target_class").loc[best[best == "ensemble"].index, "n_compounds"]
        print(f"\n-- {tag}: CASP16-folded families, leader per family --")
        for fam, row in pc.set_index("Target_class").iterrows():
            print(f"  {fam:<24} n={int(row.n_compounds):>5}  best={best[fam]:<12} "
                  f"{row[best[fam]]:+.3f}   flowr={row.flowr_root:+.3f} "
                  f"boltz2={row.boltz2:+.3f} ens={row.ensemble:+.3f}")
        print(f"  ensemble leads {len(ens)}/{len(pc)} families, "
              f"{int(ens.sum()):,}/{int(tot):,} compounds = {ens.sum() / tot * 100:.1f}%")


if __name__ == "__main__":
    main()
