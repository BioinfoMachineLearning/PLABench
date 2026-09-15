# Results

Scored output. Four files sit at the top level because everything else is derived
from them; the rest is grouped by what it describes.

| Path | Contents |
| --- | --- |
| `benchmark_summary.csv` | One row per model and dataset: the headline table |
| `per_target_evaluation.csv` | The same runs broken down by target |
| `weighted_summary.csv` | Target-count-weighted aggregates |
| `benchmark_failures.csv` | Runs that produced no usable prediction, with the reason |

`scripts/collect_results.py` writes the first, second and fourth from
`outputs/`; `scripts/weighted_summary.py` then aggregates the first two into the
third. A run only reaches these files if both `configs/model/<model>.yaml` and
`configs/dataset/<dataset>.yaml` still exist, so archiving a config removes its
rows on the next run and nothing from `archive/configs/` survives here. The
Davis and KIBA rows are the exception the collector leaves alone: they come from
`scripts/cv/collect_kiba_davis_cv.py`.

## `chembl35/`

Per-target and per-protein-family scores for the ChEMBL35 benchmark. `_filtered`
is the published arm, 4,862 ligands over 71 targets; `_full` is the unfiltered
7,650 ligands over 84 targets that the leakage comparison is against. Both
include the Boltz2 + FLOWR.ROOT ensemble. Files carrying no suffix are the same
7,650 ligands without the ensemble row.
`scripts/leakage/eval_chembl35_filtered.py` writes the suffixed pair,
`scripts/tables/` writes the rest.

## `casp16/`

`casp16_stage1_kendall.csv` and `weighted_summary_casp16.csv` hold the stage-1
ranking metrics. `casp16_submission_vs_plabench_140.csv` and
`casp16_official_vs_plabench.csv` compare the three teams' own CASP16 entries
with their PLABench arms, on all 140 ligands and on the 122 the assessors used.
`casp16_af3_vs_stage2_crystal.csv` is the AlphaFold3-versus-crystal contrast.

`pose_ladder/` holds the two interaction tables behind the four-rung pose-quality
figure: how many contacts each rung makes, and of what type.

## `leakage`

`chembl35_leakage/per_corpus_std/` is the audit itself: one file per training
corpus, giving every benchmark ligand's highest ECFP4 Tanimoto against that
corpus and whether its InChIKey skeleton appears there, after desalting and
neutralisation. The filter cuts on the row-wise maximum over the fourteen
corpora, so these files are what makes the filtered benchmark reproducible.
`selfcontrol.csv` and `sair_pklonly.csv` are controls and are not part of the
union.

`leak_by_corpus.csv` summarizes that audit one corpus at a time.
`rows_at_cut` counts the benchmark ligands a corpus reaches on its own, ignoring
the other thirteen, so the column does not sum to the number of removed rows.
`rows_attributed` does sum, because it assigns each removed row to whichever
corpus held its closest analog. SAIR accounts for nearly all of the cut, which
follows from SAIR being built out of ChEMBL 35 in the first place.

The loose CSVs beside it answer the questions the union cannot: how the drop rate
moves with the threshold (`sensitivity_by_threshold.csv`), how much of the
overlap is target-conditioned rather than global, which ligands match a training
compound exactly, and how much protein sequence the benchmark shares with the
corpora.

`casp16_leakage/` is the same audit for CASP16, plus the triple-level version
that requires protein, ligand and affinity label to match before calling a row
leaked.

## `tables/` and `figures/`

The LaTeX tables and vector figures the paper uses, written by `scripts/tables/`
and `scripts/figures/`. Each figure is kept as both PDF and PNG; the PNGs are
what the top-level `README.md` displays. `figures/figure1.pdf` is the drawn
overview panel and has no generating script.

`table4_filtered.tex` is Table 4 and carries `tab:chembl35_summary`;
`table4_full.tex` is the unfiltered comparison under
`tab:chembl35_summary_full`. The `_with_ensemble` variants add the Boltz2 +
FLOWR.ROOT row. All four are rounded once from full precision, so they are the
version to quote: any other copy that rounds a 4-dp CSV a second time lands a
digit off on cells like FlowDock's MSE, 2.1325.

## Not published

`results/unclassified_probe/` is a 229 MB working dump from the pose and affinity
probes. The two tables and the figure the paper takes from it are in
`casp16/pose_ladder/` and `figures/`; the rest is ignored by git. Superseded
passes, draft figure variants and exploratory audits are in `archive/results/`,
which is also not published.
