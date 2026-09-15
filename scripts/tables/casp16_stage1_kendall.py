#!/usr/bin/env python
"""Collect CASP16 stage-1 Kendall's tau per model and per ligand series.

Pulls the CASP16 rows out of results/benchmark_summary.csv, pairs each model's
L1000 (Chymase) and L3000 (Autotaxin) runs, and adds the N-weighted average over
all 140 compounds. That average is the black rule in the CASP16 figure; the plot
script reads this file.

A method is named `<model> (<input source>)`, so one model scoring several input
arms gets one row per arm.

Output: results/casp16/casp16_stage1_kendall.csv
"""
from __future__ import annotations

import pandas as pd

SUMMARY = "results/benchmark_summary.csv"
OUT = "results/casp16/casp16_stage1_kendall.csv"


def stage1_rows(df: pd.DataFrame) -> pd.DataFrame:
    """CASP16 stage-1 only: no stage-2, no training-variant reruns."""
    return df[
        df.Dataset.str.contains("casp16", na=False)
        & ~df.Dataset.str.contains("stage2", na=False)
        # v1..v6 are numbered reruns of the same config, superseded by the last one
        & ~df.Dataset.str.contains("v[1-6]", regex=True, na=False)
        # `sequence_casp16_*_stage1` is LLF retrained on FLOWR.ROOT stage-1 data, a
        # training variant with its own model_path -- not the CASP16 stage-1 task.
        # The benchmarked LLF is the bare `sequence_casp16_l*` (pdbbind checkpoint).
        & ~df.Dataset.str.contains("sequence_casp16_.*_stage1", regex=True, na=False)
        & ~df.Model.str.contains("stage2", na=False)
    ]


def method_name(model: str, dataset: str) -> str:
    """`flowdock` + `flowdock_boltz2_casp16_l3000_stage1` -> `flowdock (boltz2)`."""
    base = dataset
    for token in ("_casp16_l1000", "_casp16_l3000", "_stage1", "haiping_", "flowr_root_"):
        base = base.replace(token, "")
    return f"{model} ({base})" if base else model


def main() -> None:
    rows = stage1_rows(pd.read_csv(SUMMARY))
    rows = rows.assign(
        Method=[method_name(m, d) for m, d in zip(rows.Model, rows.Dataset)],
        Series=rows.Dataset.str.extract(r"(l1000|l3000)", expand=False).str.upper(),
    ).dropna(subset=["Series"])

    tau = rows.pivot(index="Method", columns="Series", values="Kendall")
    n = rows.pivot(index="Method", columns="Series", values="N")
    # Only methods that ran both series get a weighted average, and that is the
    # column the figure sorts on, so single-series methods drop out entirely.
    out = pd.DataFrame({"L1000": tau.get("L1000"), "L3000": tau.get("L3000")}).dropna()
    w = n.loc[out.index]
    out["Weighted"] = (out.L1000 * w.L1000 + out.L3000 * w.L3000) / (w.L1000 + w.L3000)

    out.to_csv(OUT)
    print(f"{len(out)} methods -> {OUT}")
    print(out.sort_values("Weighted", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
