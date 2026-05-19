"""Pivot results/per_class_summary_chembl35.csv (long form, N-weighted) to a wide
table with one row per Target_class and one column per method's mean_Pearson.

Outputs:
- results/per_class_pearson_wide_chembl35.csv
- results/per_class_pearson_wide_chembl35.tex  (booktabs, row-best bolded)

Method column order matches §5 weighted overall Pearson rank (strongest left).
Row order is by n_compounds desc (largest class first).
"""
import pandas as pd

LONG = "results/per_class_summary_chembl35.csv"
CSV_OUT = "results/per_class_pearson_wide_chembl35.csv"
TEX_OUT = "results/per_class_pearson_wide_chembl35.tex"

# Order from §5 weighted overall Pearson (chembl35 full 84 targets / 7650 compounds)
METHOD_ORDER = ["flowr_root", "boltz2", "llf", "mfe", "flowdock",
                "bapred", "mixingdta", "deepdta", "haiping"]

PRETTY_CLASS = {  # shorten the few long names for the LaTeX table
    "Family A G protein-coupled receptor": "Family A GPCR",
    "Cytochrome p450": "Cytochrome P450",
    "Transcription factor": "Transcription factor",
    "Unclassified protein": "Unclassified",
    "Other nuclear protein": "Other nuclear",
    "Epigenetic regulator": "Epigenetic regulator",
}


def main():
    df = pd.read_csv(LONG)

    n_info = (df.groupby("Target_class")
                .agg(n_targets=("n_targets", "max"),
                     n_compounds=("n_compounds", "max"))
                .reset_index())

    piv = (df.pivot_table(index="Target_class", columns="Model",
                          values="mean_Pearson")
             .reindex(columns=METHOD_ORDER))

    wide = n_info.merge(piv.reset_index(), on="Target_class")
    wide = wide.sort_values("n_compounds", ascending=False).reset_index(drop=True)

    wide.to_csv(CSV_OUT, index=False)
    print(f"Wrote {CSV_OUT}  ({len(wide)} rows)")

    # LaTeX (booktabs). Bold the per-row max across method columns.
    lines = []
    lines.append(r"\begin{tabular}{l r r " + " ".join(["r"] * len(METHOD_ORDER)) + "}")
    lines.append(r"\toprule")
    def _tex_escape(name):
        return name.replace("_", r"\_")

    header_methods = " & ".join(
        rf"\textbf{{{_tex_escape(m)}}}" for m in METHOD_ORDER
    )
    lines.append(rf"Target class & $n_{{\mathrm{{tgt}}}}$ & $n_{{\mathrm{{cmp}}}}$ & {header_methods} \\")
    lines.append(r"\midrule")

    for _, row in wide.iterrows():
        cls = PRETTY_CLASS.get(row["Target_class"], row["Target_class"])
        vals = [row[m] for m in METHOD_ORDER]
        best = max((v for v in vals if pd.notna(v)), default=None)
        cells = []
        for v in vals:
            if pd.isna(v):
                cells.append("--")
            else:
                s = f"{v:.3f}".replace("-", "$-$")
                if best is not None and v == best:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
        lines.append(rf"{cls} & {int(row['n_targets'])} & {int(row['n_compounds'])} & "
                     + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(TEX_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {TEX_OUT}")

    # Echo the CSV for quick check
    print()
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
