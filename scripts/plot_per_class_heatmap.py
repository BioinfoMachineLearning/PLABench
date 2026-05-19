"""Plot a publication-ready Seaborn heatmap of per-class mean Pearson for the
9 individual benchmark methods + Boltz2/FLOWR.ROOT ensemble on the 15 merged
target classes (chembl35 + CASP16 Chymase/Autotaxin folded in).

n_targets and n_compounds appear as compact text in the y-axis row labels
(e.g., "Kinase  [23 / 1949]") rather than as colored cells.

Columns are split into three groups with black vertical separators:
  Structure-based  |  Structure-independent  |  Ensemble

Output: results/per_class_pearson_heatmap.pdf (vector, paper figure)
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CSV = "results/per_class_pearson_wide_chembl35_casp16.csv"
OUT_PDF = "results/per_class_pearson_heatmap.pdf"

# Column groups + their order
STRUCTURE_BASED = ["Boltz2", "FlowDock", "FLOWR.ROOT", "Graph_RG",
                   "LCDD team", "MFE"]
STRUCTURE_INDEPENDENT = ["Deepdta", "LLF", "Mixingdta"]
ENSEMBLE = ["Ensemble"]
ALL_METHODS = STRUCTURE_BASED + STRUCTURE_INDEPENDENT + ENSEMBLE

# Separator positions (where to draw a vertical black line)
SEP_INDEPENDENT = len(STRUCTURE_BASED)                              # before Deepdta
SEP_ENSEMBLE = len(STRUCTURE_BASED) + len(STRUCTURE_INDEPENDENT)    # before Ensemble


def main():
    df = pd.read_csv(CSV)
    df = df.sort_values("n_compounds", ascending=False).reset_index(drop=True)

    matrix = df[ALL_METHODS].astype(float)
    matrix.index = [
        f"{cls}  [{int(nt)} / {int(nc):,}]"
        for cls, nt, nc in zip(df["Target_class"], df["n_targets"], df["n_compounds"])
    ]

    vmax = max(abs(matrix.values.min()), abs(matrix.values.max()))

    sns.set_theme(style="white", context="paper")
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
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
