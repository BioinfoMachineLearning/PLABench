#!/usr/bin/env python
"""Emit Table 4 in the exact style already used in paper/main.tex.

Row order, display names, block separator and the bold-only convention are copied from
the table at main.tex line 504, so the output is a drop-in replacement.

Numbers are aggregated at full precision straight from the per-target evaluation file
and rounded exactly once, at print time. Reading the 4-dp weighted_summary CSV instead
double-rounds: FlowDock's MSE is 2.1325244, which stores as 2.1325 and then formats to
2.132 rather than the correct 2.133.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["MSE", "RMSE", "Pearson", "Spearman", "Kendall", "Rm2", "CI"]
DIRS = ["min", "min", "max", "max", "max", "max", "max"]
# structure-based block, then the sequence-based block, exactly as in main.tex
ORDER = [("boltz2", "Boltz2"), ("flowdock", "FlowDock"), ("flowr_root", "FLOWR.ROOT"),
         ("haiping", "Graph\\_RG"), ("bapred", "LCDD-team"), ("mfe", "MFE"),
         (None, None),
         ("deepdta", "DeepDTA"), ("llf", "LLF"), ("mixingdta", "MixingDTA")]
ENS = ("ensemble", "Boltz2 + FLOWR.ROOT")


def weighted(tag: str) -> pd.DataFrame:
    """N-weighted mean over targets, full precision."""
    pt = pd.read_csv(f"results/chembl35/per_target_evaluation_chembl35_{tag}.csv")
    rows = []
    for m, g in pt.groupby("Model"):
        rec = {"Model": m, "N_total": int(g.N.sum()), "N_targets": g.Target.nunique()}
        for c in METRICS:
            v, w = g[c].to_numpy(float), g.N.to_numpy(float)
            ok = ~np.isnan(v)
            rec[c] = float(np.average(v[ok], weights=w[ok]))
        rows.append(rec)
    return pd.DataFrame(rows).set_index("Model")


def table(tag: str, caption: str, label: str, with_ensemble: bool) -> str:
    df = weighted(tag)
    ranked = df.drop(index="ensemble", errors="ignore")
    best = {m: (ranked[m].min() if d == "min" else ranked[m].max())
            for m, d in zip(METRICS, DIRS)}

    def row(key: str, name: str, bold_ok: bool = True) -> str:
        r = df.loc[key]
        cells = [f"{{\\textbf{{{r[m]:.3f}}}}}" if bold_ok and r[m] == best[m] else f"{r[m]:.3f}"
                 for m in METRICS]
        return f"{name} & " + " & ".join(cells) + " \\\\\n"

    body = ""
    for key, name in ORDER:
        body += "\\midrule\n" if key is None else row(key, name)
    if with_ensemble:
        body += "\\midrule\n" + row(*ENS, bold_ok=False)

    return ("\\begin{table*}[t]\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\begin{center}\n\\resizebox{\\textwidth}{!}{\n"
            "\\begin{tabular}{lccccccc}\n\\toprule\n"
            "Model & MSE $\\downarrow$ & RMSE $\\downarrow$ & Pearson $\\uparrow$ & "
            "Spearman $\\uparrow$ & Kendall $\\uparrow$ & $R_m^2 \\uparrow$ & "
            "CI $\\uparrow$ \\\\\n\\midrule\n" + body +
            "\\bottomrule\n\\end{tabular}\n}\n\\end{center}\n\\end{table*}\n")


CAP_FULL = ("Performance comparison of different models on the ChEMBL35 dataset "
            "(7,650 ligands, 84 targets). Bold values indicate the best performance.")
CAP_FILT = ("Performance comparison of different models on the ChEMBL35 dataset after "
            "ligand-similarity filtering (4,862 ligands, 71 targets; every test ligand "
            "within ECFP4 Tanimoto 0.85 of any ligand in the fourteen audited training "
            "corpora is removed). Bold values indicate the best performance.")


def main() -> None:
    # The filtered arm is Table 4 in the main text, so it owns tab:chembl35_summary.
    # The unfiltered arm is the supplementary comparison.
    for tag, cap, lab in (("full", CAP_FULL, "tab:chembl35_summary_full"),
                          ("filtered", CAP_FILT, "tab:chembl35_summary")):
        for ens in (False, True):
            suffix = "_with_ensemble" if ens else ""
            tex = table(tag, cap, lab, ens)
            open(f"results/tables/table4_{tag}{suffix}.tex", "w").write(tex)
            if not ens:
                print(f"%%%%%%%%%%  {tag}  %%%%%%%%%%")
                print(tex)


if __name__ == "__main__":
    main()
