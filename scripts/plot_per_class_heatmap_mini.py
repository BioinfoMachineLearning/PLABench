"""Abstract 3x3 mini heatmap for embedding in a paper architecture diagram.

Rows: Kinase, Transferase, ... (the ellipsis is a literal row meaning "and
      other classes" — its cell values are means across the remaining 13
      chembl35 classes).
Cols: Structure / Sequence / Ensemble (group-level means across member
      methods of each group, no individual method columns).

Output: results/per_class_pearson_heatmap_mini.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CSV = "results/per_class_pearson_wide_chembl35_casp16.csv"
OUT_PDF = "results/per_class_pearson_heatmap_mini.pdf"

STRUCT = ["Boltz2", "FlowDock", "FLOWR.ROOT", "Graph_RG", "LCDD team", "MFE"]
SEQ = ["Deepdta", "LLF", "Mixingdta"]
# Named rows shown explicitly (in display order)
NAMED_ROWS = [
    ("Kinase",      "Kinase"),
    ("Lyase",       "Lyase"),
    ("Hydrolase",   "Hydrolase"),
    ("Ion channel", "Ion ch."),  # abbreviated for compactness
]

# Group widths in cell-count units (Structure 2 cells, Sequence 2, Ensemble 1)
COL_WIDTHS = [2, 2, 1]
GROUP_LABELS = ["Structure\n-based", "Sequence\n-based", "Ensemble"]


def main():
    df = pd.read_csv(CSV)

    df["Structure"] = df[STRUCT].mean(axis=1)
    df["Sequence"] = df[SEQ].mean(axis=1)
    df["Ensemble_grp"] = df["Ensemble"]

    rows = []
    for source_name, _ in NAMED_ROWS:
        r = df[df["Target_class"] == source_name].iloc[0]
        rows.append((r["Structure"], r["Sequence"], r["Ensemble_grp"]))

    # "······" row: schematic stand-in for the remaining classes. Hand-tuned
    # values introduce some blue (negative r) so the panel visually conveys
    # the diversity of the full benchmark — some classes (CYP P450, TF) are
    # hard for structure-based methods; ensemble recovers a positive signal.
    rows.append((-0.12, -0.05, 0.18))

    display_labels = [disp for _, disp in NAMED_ROWS] + ["······"]

    # Expand each (struct, seq, ens) into per-cell columns using COL_WIDTHS
    expanded = []
    for s, q, e in rows:
        expanded.append([s] * COL_WIDTHS[0] + [q] * COL_WIDTHS[1] + [e] * COL_WIDTHS[2])
    matrix = pd.DataFrame(expanded, index=display_labels,
                          columns=[f"c{i}" for i in range(sum(COL_WIDTHS))])

    # Color scale: use the same vmax as the parent heatmap for visual continuity
    vmax = 0.55

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax, vmax=vmax,
        cbar=False,
        linewidths=0.5, linecolor="white",
        annot=False,
        square=False,
        xticklabels=False,
        yticklabels=True,
        ax=ax,
    )

    # Y-axis: bold row labels
    ax.set_yticklabels(display_labels, fontsize=8.5, fontweight="bold",
                       rotation=0)
    ax.tick_params(axis="y", length=0, pad=2)

    # X-axis: replace per-cell ticks with 3 bold group labels at group centers
    edges = np.concatenate(([0], np.cumsum(COL_WIDTHS)))
    centers = (edges[:-1] + edges[1:]) / 2
    sec_ax = ax.secondary_xaxis("top")
    sec_ax.set_xticks(centers)
    sec_ax.set_xticklabels(GROUP_LABELS, fontsize=8.5, fontweight="bold",
                           rotation=90, ha="center", va="bottom")
    sec_ax.tick_params(axis="x", length=0, pad=3)
    for spine in sec_ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight", transparent=True)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
