"""Fig casp16: Kendall's tau on the two CASP16 stage-1 series.

Grouped bars give each series' raw tau (L1000 = Chymase, n=17; L3000 = Autotaxin,
n=123). The black rule across each group is the size-weighted average over all 140
compounds, tau_L1000 * 17/140 + tau_L3000 * 123/140, annotated above the group.
Models are sorted by that weighted average.

Reads results/casp16/casp16_stage1_kendall.csv (written by
scripts/tables/casp16_stage1_kendall.py), taking the AF3 arm for every
structure-based model.

Palette is Okabe-Ito blue and orange, which stays distinguishable in the common
forms of colour blindness and in greyscale print.

Output: results/figures/casp16_kendall.pdf / .png
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Arial is not installed here; Liberation Sans is its metric-compatible clone, so
# glyph widths and line breaks match a real Arial build exactly. Arial is listed
# first so a machine that has it uses it.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Liberation Sans"
mpl.rcParams["mathtext.it"] = "Liberation Sans:italic"
mpl.rcParams["mathtext.bf"] = "Liberation Sans:bold"
mpl.rcParams["mathtext.cal"] = "Liberation Sans:italic"
mpl.rcParams["mathtext.sf"] = "Liberation Sans"
mpl.rcParams["mathtext.tt"] = "Liberation Sans"
mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType, keep text selectable

N1, N3 = 17, 123
ROWS = {"Boltz2": "boltz2 (boltz2)", "FLOWR.ROOT": "flowr_root (af3)",
        "Graph_RG": "haiping (af3)", "LCDD-team": "bapred (af3)",
        "MFE": "mfe (mfe_af3)", "LLF": "llf (sequence)",
        "MixingDTA": "mixingdta (sequence)", "FlowDock": "flowdock (af3)",
        "DeepDTA": "deepdta (sequence)"}
BLUE, ORANGE = "#0072B2", "#E69F00"  # Okabe-Ito
W = 0.38


def main():
    d = pd.read_csv("results/casp16/casp16_stage1_kendall.csv").set_index("Method")
    rec = [(n, d.loc[k, "L1000"], d.loc[k, "L3000"],
            (d.loc[k, "L1000"] * N1 + d.loc[k, "L3000"] * N3) / (N1 + N3))
           for n, k in ROWS.items()]
    rec.sort(key=lambda r: r[3], reverse=True)
    names, t1, t3, wa = zip(*rec)
    x = range(len(names))

    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    b1 = ax.bar([i - W / 2 for i in x], t1, W, color=BLUE, edgecolor="black",
                linewidth=0.7, zorder=3)
    b3 = ax.bar([i + W / 2 for i in x], t3, W, color=ORANGE, edgecolor="black",
                linewidth=0.7, zorder=3)

    line = None
    for i, v in enumerate(wa):
        line, = ax.plot([i - W - 0.03, i + W + 0.03], [v, v], color="black",
                        linewidth=2.4, solid_capstyle="butt", zorder=5)
        # the label sits above the taller bar, not above the rule, so it never
        # collides with a group whose L3000 bar overshoots the average
        ax.annotate(f"{v:.3f}", (i, max(t1[i], t3[i], v)), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom",
                    color="black", fontsize=13, zorder=6)

    # every tau is non-negative on the AF3 arms plotted here, so the axis starts at 0
    ax.set_ylim(0, 0.56)
    ax.set_yticks([round(0.1 * i, 1) for i in range(6)])
    ax.set_yticklabels(["0"] + [f"{0.1 * i:.1f}" for i in range(1, 6)])
    ax.set_ylabel("Kendall's $\\tau$", fontsize=16)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=14,
                       fontstyle="italic")
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="0.88", linestyle=":", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.65, len(names) - 0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.legend([b1, b3, line],
              [f"L1000, Chymase (n = {N1})", f"L3000, Autotaxin (n = {N3})",
               "Size-weighted average"],
              loc="upper right", fontsize=13, frameon=False)

    plt.tight_layout()
    plt.savefig("results/figures/casp16_kendall.pdf", bbox_inches="tight")
    plt.savefig("results/figures/casp16_kendall.png", dpi=200, bbox_inches="tight")
    for n, a, b, v in rec:
        print(f"{n:<12} L1000 {a:+.3f}  L3000 {b:+.3f}  weighted {v:.3f}")


if __name__ == "__main__":
    main()
