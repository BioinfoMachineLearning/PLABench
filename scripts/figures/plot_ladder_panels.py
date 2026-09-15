"""Fig casp16_ladder: what changes along the CASP16 L3000 pose-quality ladder.

Panel a -- Pearson r against the experimental labels for every model, on the same
93 (protein, ligand, label) triples in every cell. Only the input coordinates
change: the four rungs are the Boltz-2 cofolded pose (median ligand RMSD 6.96 A),
the template-pocket cofolded pose (3.72 A), the AF3 pose (0.76 A) and the
deposited crystal pose. Boltz-2 predicts structure and affinity in one pass, so it
has no rung series and appears as a horizontal reference at its native value; the
three sequence-only models are horizontal references for the same reason.

Panel b -- how much the interaction census actually moves. Each series is the
percentage difference between the predicted-pose count and the crystal-pose count,
summed over the same 93 complexes, so 0 means the rung reproduces the crystal
total exactly. Read with results/casp16/pose_ladder/rung_interaction_types.csv:
the totals barely move while the per-contact identity churns.

Reads results/casp16/pose_ladder/{nci_counts_by_rung.json,rung_interaction_types.csv}
and the per-rung predictions under outputs/. Output:
results/figures/fig_ladder_panels.pdf / .png
"""
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plabench.analysis.units import dg_to_pkd  # noqa: E402

# Arial is not installed here; Liberation Sans is its metric-compatible clone, so
# glyph widths match a real Arial build. Arial is listed first for machines that
# do have it.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "custom"
for _k, _v in [("rm", "Liberation Sans"), ("it", "Liberation Sans:italic"),
               ("bf", "Liberation Sans:bold"), ("cal", "Liberation Sans:italic"),
               ("sf", "Liberation Sans"), ("tt", "Liberation Sans")]:
    mpl.rcParams[f"mathtext.{_k}"] = _v
mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType, keep text selectable

LABELS = ["6.96 Å", "3.72 Å", "0.76 Å", "Crystal"]
LABEL_FILE = "data/casp16_data/labels/L3000_exper_affinity.csv"
NCI_JSON = "results/casp16/pose_ladder/nci_counts_by_rung.json"
TYPES_CSV = "results/casp16/pose_ladder/rung_interaction_types.csv"
OUT = "results/figures/fig_ladder_panels"

# Wong colorblind-safe palette, as in the original figure.
BLUE, GREEN, ORANGE, PINK, PURPLE, GOLD = (
    "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#6A3D9A", "#E69F00")
GREY = "0.62"

# Four rungs of predictions per structure-based model, worst pose first.
LADDER = {
    "FLOWR.ROOT": (PINK, "D", ["outputs/flowr_root/flowr_root_boltz2_casp16_l3000_stage1",
                               "outputs/flowr_root/flowr_root_tplpocket_casp16_l3000_stage1",
                               "outputs/flowr_root/flowr_root_af3_casp16_l3000_stage1",
                               "outputs/flowr_root/experimental_casp16_l3000_stage2"]),
    "MFE": (GREEN, "D", ["outputs/mfe/mfe_casp16_l3000_stage1",
                         "outputs/mfe/mfe_tplpocket_casp16_l3000_stage1",
                         "outputs/mfe/mfe_af3_casp16_l3000_stage1",
                         "outputs/mfe/mfe_casp16_l3000_stage2"]),
    "Graph_RG": (PURPLE, "P", ["outputs/haiping/haiping_boltz2_casp16_l3000_stage1",
                               "outputs/haiping/haiping_tplpocket_casp16_l3000_stage1",
                               "outputs/haiping/haiping_af3_casp16_l3000_stage1",
                               "outputs/haiping/haiping_casp16_l3000_stage2"]),
    "LCDD team": (BLUE, "o", ["outputs/bapred/boltz2_casp16_l3000_stage1",
                              "outputs/bapred/tplpocket_casp16_l3000_stage1",
                              "outputs/bapred/af3_casp16_l3000_stage1",
                              "outputs/bapred/experimental_casp16_l3000_stage2"]),
    "FlowDock": (GOLD, "s", ["outputs/flowdock/boltz2_casp16_l3000_stage1",
                             "outputs/flowdock/tplpocket_casp16_l3000_stage1",
                             "outputs/flowdock/af3_casp16_l3000_stage1",
                             "outputs/flowdock/experimental_casp16_l3000_stage2"]),
}
# Models with no rung series: Boltz-2 folds and scores in one pass, the other
# three never see a structure. LLF is the bare run (the PDBbind-retrained model
# the paper benchmarks), not the FLOWR.ROOT stage-1 retrain.
FLAT = {
    "Boltz2": (ORANGE, "outputs/boltz2/boltz2_casp16_l3000"),
    "LLF": (GREY, "outputs/llf/sequence_casp16_l3000"),
    "MixingDTA": (GREY, "outputs/mixingdta/sequence_casp16_l3000"),
    "DeepDTA": (GREY, "outputs/deepdta/sequence_casp16_l3000"),
}

# Panel b: the five interaction classes with enough counts to plot. Salt bridges
# are excluded -- the crystal total is 1, so the percentage is meaningless.
NCI = [("contact", "Contact residues", "#4D4D4D", "o"),
       ("hydrophobic", "Hydrophobic", BLUE, "s"),
       ("hbond", "H-bonds", ORANGE, "^"),
       ("pistack", "π-stacking", GREEN, "D"),
       ("halogen", "Halogen bonds", PINK, "v")]


def truth() -> pd.DataFrame:
    """The 93 ligands with a deposited complex, with labels on the pKd scale."""
    ids = set(json.load(open(NCI_JSON))["af3"])
    d = (pd.read_csv(LABEL_FILE, encoding="utf-8-sig")
         .rename(columns={"Target ID": "ligand_id"}))
    d["ligand_id"] = d.ligand_id.astype(str).str.strip()
    d["pKd"] = dg_to_pkd(d.binding_affinity)
    return d[d.ligand_id.isin(ids)][["ligand_id", "pKd"]]


def pearson(run: str, gt: pd.DataFrame) -> float:
    p = pd.read_csv(f"{run}/latest/predictions.csv", header=None,
                    names=["pred", "ligand_id"], dtype={"ligand_id": str})
    p["ligand_id"] = p.ligand_id.str.strip()
    d = p.merge(gt, on="ligand_id")
    assert len(d) == len(gt), f"{run}: {len(d)} of {len(gt)} ligands matched"
    return float(pearsonr(d.pred, d.pKd)[0])


def declutter(labels, gap):
    """Push the inline labels apart so none overlaps, keeping their order.

    `labels` is (y, name, color); the returned y is where to draw the text, which
    may differ from the series' own y -- the caller draws a leader line then.
    """
    out = sorted(labels)
    for i in range(1, len(out)):
        if out[i][0] - out[i - 1][0] < gap:
            out[i] = (out[i - 1][0] + gap,) + out[i][1:]
    return out


def panel_a(ax, gt):
    x = range(4)
    labels = []
    for name, (color, marker, runs) in LADDER.items():
        y = [pearson(r, gt) for r in runs]
        ax.plot(x, y, color=color, marker=marker, markersize=7, linewidth=2,
                clip_on=False, zorder=3)
        labels.append((y[-1], name, color))
        print(f"{name:<12} " + "  ".join(f"{v:+.3f}" for v in y))

    for name, (color, run) in FLAT.items():
        v = pearson(run, gt)
        # a plot() rather than axhline() so the rule stops at the last rung and
        # leaves the label margin clear
        ax.plot([0, 3], [v, v], color=color, linestyle="--", linewidth=1.8,
                zorder=2)
        labels.append((v, name, color))
        print(f"{name:<12} flat {v:+.3f}")

    ax.set_ylim(0.35, 0.70)
    ax.set_ylabel("Pearson $r$ vs. experimental affinity", fontsize=13)
    for (ty, name, color), (y, _, _) in zip(declutter(labels, 0.016),
                                             sorted(labels)):
        if abs(ty - y) > 1e-9:  # nudged, so show where the label belongs
            ax.plot([3.02, 3.11], [y, ty], color=color, linewidth=0.8,
                    clip_on=False, zorder=2)
        ax.annotate(name, (3, ty), xytext=(12, 0), textcoords="offset points",
                    va="center", color=color, fontsize=12, annotation_clip=False)


def panel_b(ax):
    t = pd.read_csv(TYPES_CSV)
    order = ["boltz2", "tplpocket", "af3"]
    for key, label, color, marker in NCI:
        s = t[t.type == key].set_index("rung").total_pct
        y = [s[r] for r in order] + [0.0]  # the crystal rung is the reference
        ax.plot(range(4), y, color=color, marker=marker, markersize=7,
                linewidth=2, label=label, clip_on=False, zorder=3)
    ax.axhline(0, color="0.55", linewidth=1, zorder=1)
    ax.set_ylabel("Interaction count difference (%)\nrelative to crystal complex",
                  fontsize=13)
    ax.legend(fontsize=11, loc="upper right", frameon=False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.1))
    panel_a(axes[0], truth())
    panel_b(axes[1])

    for ax, tag in zip(axes, "ab"):
        ax.set_xticks(range(4))
        ax.set_xticklabels(LABELS, fontsize=13)
        ax.set_xlabel("Input pose error (median ligand RMSD)", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.12, 3.12)
        ax.annotate(tag, (0, 1), xycoords="axes fraction", xytext=(-52, 22),
                    textcoords="offset points", fontsize=19, fontweight="bold")

    # Leave room on the right of panel a for the inline series labels.
    fig.subplots_adjust(left=0.07, right=0.99, wspace=0.42, bottom=0.13, top=0.93)
    fig.savefig(f"{OUT}.pdf")
    fig.savefig(f"{OUT}.png", dpi=200)
    print(f"Wrote {OUT}.pdf")


if __name__ == "__main__":
    main()
