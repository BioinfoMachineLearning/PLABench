#!/usr/bin/env python
"""Apply the approved leakage filter to the ChEMBL35 benchmark.

Rule (strict, target-blind): drop any test row whose ligand reaches ECFP4 Tanimoto
>= 0.85 against any ligand in the fourteen audited training corpora, fingerprints
computed after desalting and neutralisation. Then drop targets left with fewer than
ten ligands, since per-target correlations are meaningless below that.

Similarities come from results/chembl35_leakage/per_corpus_std/, one file per corpus,
produced by chembl35_leakage_std_recheck.py.

Nothing is overwritten. The kept rows, the dropped rows and a per-target ledger are
written next to the original input; a per-corpus breakdown of the cut goes to
results/chembl35_leakage/leak_by_corpus.csv.
"""
from __future__ import annotations

import pandas as pd

CUT = 0.85
FLOOR = 10
LEAK_DIR = "results/chembl35_leakage"
STD_DIR = f"{LEAK_DIR}/per_corpus_std"
CORPORA = ["chembl34_full", "bindingdb_pre2412", "bindingnet_full", "hiqbind", "omol25",
           "pdbbind2020", "pdbbind2020_general", "bindingmoad", "spindr",
           "plinder", "kiba3d", "davis3d", "kinodata", "sair"]


def main() -> None:
    b = pd.read_csv("data/chembl35/chembl35_full_input.csv")
    b["target"] = b.compound_id.str.split("_CHEMBL").str[0]
    # chembl35_leakage_std_recheck.py scores each corpus separately. A row is only
    # as clean as its worst corpus, so the cut uses the row-wise maximum; the
    # corpus holding that closest analog goes into the removal ledger.
    per = b[["compound_id"]].copy()
    for c in CORPORA:
        s = pd.read_csv(f"{STD_DIR}/{c}.csv")[["compound_id", f"simstd_{c}"]]
        per = per.merge(s, on="compound_id", how="left", validate="one_to_one")
    cols = [f"simstd_{c}" for c in CORPORA]
    assert per[cols].notna().all().all(), "similarity tables do not cover every row"
    b["std_union"] = per[cols].max(axis=1)
    b["top_corpus"] = per[cols].idxmax(axis=1).str.replace("simstd_", "", regex=False)

    sim_drop = b.std_union >= CUT
    kept = b[~sim_drop]
    counts = kept.groupby("target").size()
    thin = set(counts[counts < FLOOR].index) | (set(b.target) - set(counts.index))
    floor_drop = ~sim_drop & b.target.isin(thin)

    b["drop_reason"] = "keep"
    b.loc[sim_drop, "drop_reason"] = f"similarity>={CUT}"
    b.loc[floor_drop, "drop_reason"] = f"target below {FLOOR} ligands after filtering"
    final = b[b.drop_reason == "keep"].drop(columns=["top_corpus"])
    removed = b[b.drop_reason != "keep"]

    out_cols = ["compound_id", "target_sequence", "compound_iso_smiles", "affinity", "status",
                "Sequence_identity", "Compound_structural_similarity"]
    final[out_cols].to_csv("data/chembl35/chembl35_filtered_input.csv", index=False)
    removed[["compound_id", "target", "affinity", "std_union", "top_corpus", "drop_reason"]] \
        .sort_values(["drop_reason", "std_union"], ascending=[True, False]) \
        .to_csv("data/chembl35/chembl35_removed_rows.csv", index=False)

    ledger = pd.DataFrame({"before": b.groupby("target").size(),
                           "after": final.groupby("target").size()}).fillna(0).astype(int)
    ledger["kept_pct"] = (ledger.after / ledger.before * 100).round(1)
    ledger.sort_values("kept_pct").to_csv("data/chembl35/chembl35_filter_ledger.csv")

    # Two different questions, so two columns. "rows_at_cut" is how much this corpus
    # leaks on its own, ignoring the others, and the column does not sum to the number
    # of removed rows because corpora overlap. "rows_attributed" splits the removals
    # into disjoint buckets by whichever corpus held the closest analog.
    attributed = removed[removed.drop_reason.str.startswith("similarity")].top_corpus.value_counts()
    by_corpus = pd.DataFrame({
        "rows_at_cut": {c: int((per[f"simstd_{c}"] >= CUT).sum()) for c in CORPORA},
        "rows_attributed": attributed,
    }).fillna(0).astype(int)
    by_corpus["pct_of_benchmark"] = (by_corpus.rows_at_cut / len(b) * 100).round(2)
    by_corpus.sort_values("rows_at_cut", ascending=False) \
        .rename_axis("corpus").to_csv(f"{LEAK_DIR}/leak_by_corpus.csv")

    print(f"kept     {len(final):>6,} rows / {final.target.nunique()} targets")
    print(f"removed  {len(removed):>6,} rows  "
          f"({int(sim_drop.sum()):,} by similarity, {int(floor_drop.sum()):,} by the {FLOOR}-ligand floor)")
    print(f"targets dropped: {b.target.nunique() - final.target.nunique()}")
    print("\nremovals attributed to each corpus (the one holding the closest analog):")
    print(removed[removed.drop_reason.str.startswith('similarity')]
          .top_corpus.value_counts().to_string())
    print(f"\naffinity range  before {b.affinity.min():.2f}-{b.affinity.max():.2f} "
          f"(mean {b.affinity.mean():.3f}, sd {b.affinity.std():.3f})")
    print(f"                after  {final.affinity.min():.2f}-{final.affinity.max():.2f} "
          f"(mean {final.affinity.mean():.3f}, sd {final.affinity.std():.3f})")


if __name__ == "__main__":
    main()
