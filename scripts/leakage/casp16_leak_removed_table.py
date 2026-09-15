#!/usr/bin/env python
"""CASP16 stage-1 metrics after dropping every ligand whose affinity label was public.

The removal set is the union, over all nine methods, of the triple-level hits found by
scripts/leakage/casp16_triple_leakage.py: same protein, same ligand (full InChIKey) AND an
attached affinity in a training corpus. That is L1012 on L1000 (1 of 17) and 20 ligands
on L3000 (Boltz2's 19 union FLOWR.ROOT's 14). Structural-similarity-only matches are not
removed -- a Tanimoto neighbour with no label is chemistry, not leakage.

Same metric set as tab:casp16_stage1, plus Kendall's tau. Writes
results/tables/casp16_stage1_leak_removed.tex and .csv.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import kendalltau

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plabench.analysis.metrics import calculate_metrics  # noqa: E402
from plabench.analysis.units import dg_to_pkd  # noqa: E402

PRED = {
    "Boltz2": "outputs/boltz2/boltz2_casp16_{s}/latest/predictions.csv",
    "FlowDock": "outputs/flowdock/af3_casp16_{s}_stage1/latest/predictions.csv",
    "FLOWR.ROOT": "outputs/flowr_root/flowr_root_af3_casp16_{s}_stage1/latest/predictions.csv",
    "Graph_RG": "outputs/haiping/haiping_af3_casp16_{s}_stage1/latest/predictions.csv",
    "LCDD-team": "outputs/bapred/af3_casp16_{s}_stage1/latest/predictions.csv",
    "MFE": "outputs/mfe/mfe_af3_casp16_{s}_stage1/latest/predictions.csv",
    "DeepDTA": "outputs/deepdta/sequence_casp16_{s}/latest/predictions.csv",
    "LLF": "outputs/llf/sequence_casp16_{s}/latest/predictions.csv",
    "MixingDTA": "outputs/mixingdta/sequence_casp16_{s}/latest/predictions.csv",
}
COLS = ["MSE", "RMSE", "Pearson", "Spearman", "Kendall", "Rm2", "CI"]
DIRS = {"MSE": "min", "RMSE": "min", "Pearson": "max", "Spearman": "max",
        "Kendall": "max", "Rm2": "max", "CI": "max"}
DISPLAY = {"Graph_RG": "Graph\\_RG"}


def removal_set(series: str) -> set[str]:
    d = pd.read_csv("results/casp16_leakage/triple_detail.csv")
    return set(d[d.series == series].ligand_id.astype(str))


def truth(series: str) -> pd.DataFrame:
    d = (pd.read_csv(f"data/casp16_data/labels/{series}_exper_affinity.csv",
                     encoding="utf-8-sig")
         .rename(columns={"Target ID": "ligand_id"}))
    d["ligand_id"] = d.ligand_id.astype(str).str.strip()
    d["pKd"] = dg_to_pkd(d.binding_affinity)
    return d[["ligand_id", "pKd"]]


def main() -> None:
    res: dict[str, dict[str, dict[str, float]]] = {}
    ns: dict[str, tuple[int, int]] = {}
    for series in ("L1000", "L3000"):
        gt, drop = truth(series), removal_set(series)
        keep = gt[~gt.ligand_id.isin(drop)]
        ns[series] = (len(gt), len(keep))
        res[series] = {}
        for model, tmpl in PRED.items():
            p = pd.read_csv(tmpl.format(s=series.lower()), header=None,
                            names=["pred", "ligand_id"], dtype={"ligand_id": str})
            p["ligand_id"] = p.ligand_id.str.strip()
            d = p.merge(keep, on="ligand_id")
            m = calculate_metrics(d.pred, d.pKd)
            res[series][model] = {c: float(m[c]) for c in COLS if c != "Kendall"}
            res[series][model]["Kendall"] = float(kendalltau(d.pred, d.pKd)[0])

    best = {(s, c): (min if DIRS[c] == "min" else max)(res[s][m][c] for m in PRED)
            for s in res for c in COLS}
    body = ""
    for i, model in enumerate(PRED):
        if i == 6:
            body += "\\midrule\n"
        cells = [f"{{\\textbf{{{res[s][model][c]:.3f}}}}}"
                 if res[s][model][c] == best[(s, c)] else f"{res[s][model][c]:.3f}"
                 for s in ("L1000", "L3000") for c in COLS]
        body += f"{DISPLAY.get(model, model)} & " + " & ".join(cells) + " \\\\\n"

    open("results/tables/casp16_stage1_leak_removed.tex", "w").write(body)
    rows = [{"series": s, "model": m, **res[s][m]} for s in res for m in PRED]
    pd.DataFrame(rows).to_csv("results/tables/casp16_stage1_leak_removed.csv", index=False)

    for s, (n0, n1) in ns.items():
        print(f"{s}: {n0} -> {n1} ligands")
        print(f"{'model':<12}" + "".join(f"{c:>10}" for c in COLS))
        for m in PRED:
            print(f"{m:<12}" + "".join(f"{res[s][m][c]:>10.3f}" for c in COLS))
        print()
    print(body)


if __name__ == "__main__":
    main()
