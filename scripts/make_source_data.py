"""Assemble the Source Data file the manuscript promises.

Nature Communications asks for the numerical values behind every graph and chart,
one file per figure panel, as an Excel workbook or a zipped folder. This writes the
zipped folder, because the values already exist as CSV and round-tripping them
through xlsx only risks changing them.

    python scripts/make_source_data.py            # -> source_data/ and source_data.zip

Most panels are a copy of an existing results CSV. Figure 3a is not: the figure
recomputes Pearson r from the per-rung predictions at draw time and never writes
them down, so this script recomputes them the same way, by importing the figure's
own truth() and pearson().
"""
import importlib.util
import shutil
import zipfile
from pathlib import Path

import pandas as pd

OUT = Path("source_data")

# One file per numbered figure panel and per numbered table, so each name maps to
# exactly one thing in the manuscript. Two things have no entry and should not:
# Figure 1 is a schematic, and Table 1 is the model inventory (names, versions,
# commits), which carries no measured values.
#
# (destination file, source CSV) for everything that is already on disk in the shape
# the manuscript prints it. Tables 2, 3 and 4 are derived further down instead.
COPIES = [
    ("Figure_2_casp16_kendall.csv",
     "results/casp16/casp16_stage1_kendall.csv"),
    ("Figure_3b_interaction_counts_by_rung.csv",
     "results/casp16/pose_ladder/rung_interaction_types.csv"),
    # Figure 4 is the leakage-filtered arm, 15 classes x 9 models + ensemble, which
    # is what its caption describes (n = 5,014 pairs over 73 targets).
    ("Figure_4_per_class_pearson.csv",
     "results/chembl35/per_class_pearson_wide_chembl35_casp16_filtered.csv"),
    # Supplementary tables. main.tex resets the table counter before the
    # Supplementary Information, so these are numbered 1-4 in their own series.
    # The chain-rule arms were trained in a scratch tree that is not part of the
    # Zenodo deposit, so this is the only published copy of their numbers.
    ("Supplementary_Table_1_chain_rule.csv",
     "scratch/concat_ablation/results/arm_comparison.csv"),
    ("Supplementary_Table_2_chain_rule_did.csv",
     "scratch/concat_ablation/results/diff_in_diff.csv"),
    ("Supplementary_Table_3_casp16_official.csv",
     "results/casp16/casp16_submission_vs_plabench_140.csv"),
    ("Supplementary_Table_4_casp16_leak_removed.csv",
     "results/tables/casp16_stage1_leak_removed.csv"),
    # Every aggregate above is an N-weighted mean over targets. This is the file
    # they are all weighted from, for anyone who wants to recompute a cell.
    ("Underlying_per_target_evaluation.csv", "results/per_target_evaluation.csv"),
]

# Table 2 prints the three sequence models on Davis and KIBA under three splits.
TABLE2_DATASETS = ["davis_warm", "kiba_warm", "davis_cold_target", "kiba_cold_target",
                   "davis_cold_drug", "kiba_cold_drug"]

README = """Source Data for "{title}"

One CSV per numbered figure panel and per numbered table. File names match the
manuscript, so Table_3_casp16_stage1.csv holds the values printed in Table 3.
Values are exactly those printed or plotted; nothing is rounded here beyond the
six decimal places the aggregates are stored at.

{listing}
Figure 1 is a schematic and Table 1 is the model inventory, so neither has
underlying numerical data.

Underlying_per_target_evaluation.csv is not a figure or a table. It is the
per-target file the aggregates in Tables 2 to 4 are N-weighted from, included so
any cell can be recomputed.

Regenerate with: python scripts/make_source_data.py
"""


def load(path: str, name: str):
    """Import a script by path so its own numbers can be reused rather than reimplemented."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def table_2() -> pd.DataFrame:
    """Davis and KIBA under the three splits, straight from the scorer's output."""
    d = pd.read_csv("results/benchmark_summary.csv")
    d = d[d.Dataset.isin(TABLE2_DATASETS)].copy()
    d["Dataset"] = pd.Categorical(d.Dataset, TABLE2_DATASETS, ordered=True)
    return d.sort_values(["Dataset", "Model"])


def table_3() -> pd.DataFrame:
    """CASP16 stage 1, recomputed the way casp16_recompute_stage1.py recomputes it.

    That script renders LaTeX and keeps no CSV, so this reuses its label loader and
    its prediction paths rather than growing a second implementation that could drift.
    The values are the 300 K ones, matching the manuscript.
    """
    c = load("scripts/tables/casp16_recompute_stage1.py", "casp16_stage1")
    rows = []
    for series in ("L1000", "L3000"):
        gt = c.labels(series)
        for model, tmpl in c.PRED.items():
            p = pd.read_csv(tmpl.format(s=series.lower()), header=None,
                            names=["pred", "ligand_id"], dtype={"ligand_id": str})
            p["ligand_id"] = p.ligand_id.str.strip()
            d = p.merge(gt, on="ligand_id")
            m = c.calculate_metrics(d.pred, d.new)
            rows.append({"series": series, "model": model, "n": len(d),
                         **{k: round(float(m[k]), 6) for k in c.COLS}})
    return pd.DataFrame(rows)


def table_4() -> pd.DataFrame:
    """ChEMBL35, the leakage-filtered arm the manuscript prints, at full precision.

    emit_table4.weighted() aggregates from the per-target file and rounds once at
    print time; reading the 4-dp summary instead double-rounds.
    """
    t4 = load("scripts/tables/emit_table4.py", "emit_table4")
    df = t4.weighted("filtered").drop(index="ensemble", errors="ignore")
    return df.round(6).reset_index()


def ladder_panel_a() -> pd.DataFrame:
    """Pearson r per model per rung, recomputed the way the figure computes it."""
    ladder = load("scripts/figures/plot_ladder_panels.py", "ladder")
    gt = ladder.truth()
    rows = []
    for name, (_colour, _marker, runs) in ladder.LADDER.items():
        for rung, run in zip(ladder.LABELS, runs):
            rows.append({"model": name, "input_pose": rung, "series": "ladder",
                         "pearson_r": round(ladder.pearson(run, gt), 6)})
    for name, (_colour, run) in ladder.FLAT.items():
        # These models never see the rung poses, so the figure draws one value as a
        # horizontal reference. Repeated here per rung, flagged, so nobody reads the
        # flat line as four measurements.
        r = round(ladder.pearson(run, gt), 6)
        for rung in ladder.LABELS:
            rows.append({"model": name, "input_pose": rung,
                         "series": "reference (pose-independent)", "pearson_r": r})
    return pd.DataFrame(rows)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    written = []
    for dest, src in COPIES:
        if not Path(src).exists():
            print(f"  MISSING {src}")
            continue
        shutil.copy(src, OUT / dest)
        written.append((dest, src))

    for dest, build, note in [
        ("Figure_3a_pose_ladder_pearson.csv", ladder_panel_a,
         "recomputed via plot_ladder_panels.py from outputs/<model>/<rung>/"),
        ("Table_2_davis_kiba.csv", table_2, "results/benchmark_summary.csv"),
        ("Table_3_casp16_stage1.csv", table_3,
         "recomputed via casp16_recompute_stage1.py from outputs/<model>/<series>/"),
        ("Table_4_chembl35.csv", table_4,
         "recomputed via emit_table4.py from per_target_evaluation_chembl35_filtered.csv"),
    ]:
        build().to_csv(OUT / dest, index=False)
        written.append((dest, note))
    written.sort()

    listing = "".join(f"  {d:<48} {s}\n" for d, s in written)
    (OUT / "README.txt").write_text(README.format(
        title="Benchmarking protein-ligand binding affinity prediction", listing=listing))

    with zipfile.ZipFile("source_data.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            z.write(f, f"source_data/{f.name}")

    for d, s in written:
        print(f"  {d:<48} <- {s}")
    print(f"Wrote {OUT}/ and source_data.zip "
          f"({Path('source_data.zip').stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
