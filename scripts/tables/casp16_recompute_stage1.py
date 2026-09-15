#!/usr/bin/env python
"""Recompute the CASP16 stage-1 table with the correct dG -> pKd constant.

The pipeline converted CASP16 labels with `* -0.733`, i.e. RT ln(10) at 298.15 K. The
label files were built at 300 K (see plabench/analysis/units), so the ground truth was
0.61% too small. Scale-invariant metrics (Pearson, Spearman, CI) are unaffected; MSE,
RMSE and Rm2 are not. This prints both versions side by side.
"""
from __future__ import annotations

import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plabench.analysis.metrics import calculate_metrics  # noqa: E402
from plabench.analysis.units import DG_KCAL_PER_LOG_UNIT  # noqa: E402

OLD_SCALE = 0.733  # what scripts/collect_results.py and run_benchmark.py used

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
COLS = ["MSE", "RMSE", "Pearson", "Spearman", "Rm2", "CI"]


def labels(series: str) -> pd.DataFrame:
    d = pd.read_csv(f"data/casp16_data/labels/{series}_exper_affinity.csv",
                    encoding="utf-8-sig").rename(columns={"Target ID": "ligand_id"})
    d["ligand_id"] = d.ligand_id.astype(str).str.strip()
    d["old"] = d.binding_affinity * -OLD_SCALE
    d["new"] = -d.binding_affinity / DG_KCAL_PER_LOG_UNIT
    return d[["ligand_id", "old", "new"]]


DISPLAY = {"Graph_RG": "Graph\\_RG"}
DIRS = {"MSE": "min", "RMSE": "min", "Pearson": "max", "Spearman": "max",
        "Rm2": "max", "CI": "max"}


def latex(new: dict[str, dict[str, dict[str, float]]]) -> str:
    best = {(s, c): (min if DIRS[c] == "min" else max)(new[s][m][c] for m in PRED)
            for s in ("L1000", "L3000") for c in COLS}
    body = ""
    for i, model in enumerate(PRED):
        if i == 6:
            body += "\\midrule\n"
        cells = []
        for s in ("L1000", "L3000"):
            for c in COLS:
                v = new[s][model][c]
                cells.append(f"{{\\textbf{{{v:.3f}}}}}" if v == best[(s, c)] else f"{v:.3f}")
        body += f"{DISPLAY.get(model, model)} & " + " & ".join(cells) + " \\\\\n"
    return body


def main() -> None:
    new: dict[str, dict[str, dict[str, float]]] = {}
    for series in ("L1000", "L3000"):
        gt = labels(series)
        new[series] = {}
        print(f"\n=== {series} (n={len(gt)}) ===")
        print(f"{'model':<12}" + "".join(f"{c:>18}" for c in COLS))
        for model, tmpl in PRED.items():
            p = pd.read_csv(tmpl.format(s=series.lower()), header=None,
                            names=["pred", "ligand_id"], dtype={"ligand_id": str})
            p["ligand_id"] = p.ligand_id.str.strip()
            d = p.merge(gt, on="ligand_id")
            a = calculate_metrics(d.pred, d.old)
            b = calculate_metrics(d.pred, d.new)
            new[series][model] = {c: float(b[c]) for c in COLS}
            cells = "".join(f"{a[c]:>8.3f} ->{b[c]:>7.3f}" for c in COLS)
            print(f"{model:<12}{cells}")

    body = latex(new)
    open("results/tables/casp16_stage1.tex", "w").write(body)
    print("\n%%%%%%%%%%  corrected tab:casp16_stage1 body  %%%%%%%%%%")
    print(body)


if __name__ == "__main__":
    main()
