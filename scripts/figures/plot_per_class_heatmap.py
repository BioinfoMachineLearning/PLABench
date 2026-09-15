"""Plot a publication-ready Seaborn heatmap of per-class mean Pearson for the
9 individual benchmark methods + Boltz2/FLOWR.ROOT ensemble on the 15 merged
target classes (chembl35 + CASP16 Chymase/Autotaxin folded in).

n_targets and n_compounds appear as compact text in the y-axis row labels
(e.g., "Kinase  [23 / 1949]") rather than as colored cells.

Columns are split into three groups with black vertical separators:
  Structure-based  |  Structure-independent  |  Ensemble

Output: results/figures/per_class_pearson_heatmap.pdf (vector, paper figure)

Usage:
    python scripts/figures/plot_per_class_heatmap.py                # published 7,650-ligand arm
    python scripts/figures/plot_per_class_heatmap.py filtered       # leakage-filtered 4,862 arm
The argument is the suffix of results/chembl35/per_class_pearson_wide_chembl35_casp16_<arm>.csv,
so the two arms are drawn by the same code and cannot drift apart in style.
"""
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Match the Arial used by the other paper figures. Arial itself is not installed
# here; Liberation Sans is its metric-compatible clone, so widths are identical.
mpl.rcParams["font.family"] = "sans-serif"
FONTS = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
mpl.rcParams["font.sans-serif"] = FONTS
mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType rather than rasterise to Type 3

ARM = sys.argv[1] if len(sys.argv) > 1 else None
CSV = ("results/chembl35/per_class_pearson_wide_chembl35_casp16.csv" if ARM is None
       else f"results/chembl35/per_class_pearson_wide_chembl35_casp16_{ARM}.csv")
OUT_PDF = ("results/figures/per_class_pearson_heatmap.pdf" if ARM is None
           else f"results/figures/per_class_pearson_heatmap_{ARM}.pdf")

# Column groups + their order
STRUCTURE_BASED = ["Boltz2", "FlowDock", "FLOWR.ROOT", "Graph_RG",
                   "LCDD team", "MFE"]
STRUCTURE_INDEPENDENT = ["Deepdta", "LLF", "Mixingdta"]
ENSEMBLE = ["Ensemble"]
ALL_METHODS = STRUCTURE_BASED + STRUCTURE_INDEPENDENT + ENSEMBLE

# Separator positions (where to draw a vertical black line)
SEP_INDEPENDENT = len(STRUCTURE_BASED)                              # before Deepdta
SEP_ENSEMBLE = len(STRUCTURE_BASED) + len(STRUCTURE_INDEPENDENT)    # before Ensemble


# eval_chembl35_filtered.py writes raw model keys and raw ChEMBL class names; the
# published CSV carries the display forms. Normalize so either file can be plotted.
RENAME = {"boltz2": "Boltz2", "flowdock": "FlowDock", "flowr_root": "FLOWR.ROOT",
          "haiping": "Graph_RG", "bapred": "LCDD team", "mfe": "MFE",
          "deepdta": "Deepdta", "llf": "LLF", "mixingdta": "Mixingdta",
          "ensemble": "Ensemble"}
PRETTY_CLASS = {"Family A G protein-coupled receptor": "Family A GPCR",
                "Cytochrome p450": "Cytochrome P450",
                "Unclassified protein": "Unclassified",
                "Other nuclear protein": "Other nuclear"}


def main():
    df = pd.read_csv(CSV).rename(columns=RENAME)
    df["Target_class"] = df["Target_class"].replace(PRETTY_CLASS)
    df = df.sort_values("n_compounds", ascending=False).reset_index(drop=True)

    matrix = df[ALL_METHODS].astype(float)
    matrix.index = [
        f"{cls}  [{int(nt)} / {int(nc):,}]"
        for cls, nt, nc in zip(df["Target_class"], df["n_targets"], df["n_compounds"])
    ]

    vmax = max(abs(matrix.values.min()), abs(matrix.values.max()))

    # sns.set_theme() rewrites rcParams, so the font family has to be (re)applied
    # after it, not at import time.
    sns.set_theme(style="white", context="paper")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = FONTS
    mpl.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax, vmax=vmax,
        annot=True, fmt=".3f",
        annot_kws={"fontsize": 7.5},
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.85, "pad": 0.02},
        square=False,
        ax=ax,
    )

    # Group separators
    ax.axvline(SEP_INDEPENDENT, color="black", linewidth=1.5)
    ax.axvline(SEP_ENSEMBLE, color="black", linewidth=1.5)

    ax.set_xlabel("")
    ax.set_ylabel("Target class  [targets number / compound number]", fontsize=10)
    ax.tick_params(axis="x", labelrotation=35, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    # Top group labels
    sec_ax = ax.secondary_xaxis("top")
    sec_ax.set_xticks([
        SEP_INDEPENDENT / 2,
        (SEP_INDEPENDENT + SEP_ENSEMBLE) / 2,
        (SEP_ENSEMBLE + len(ALL_METHODS)) / 2,
    ])
    sec_ax.set_xticklabels(["Structure-based", "Sequence-based", "Ensemble"],
                           fontsize=10, fontweight="bold")
    sec_ax.tick_params(axis="x", length=0, pad=4)

    plt.tight_layout()
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.savefig(OUT_PDF.replace(".pdf", ".png"), format="png", dpi=200,
                bbox_inches="tight")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
