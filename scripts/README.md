# Scripts

Four things live at the top level: the two halves of the scorer every analysis
reads, and the two checkpoint fetchers the install instructions call. Everything
else is grouped by what it does. Superseded and exploratory scripts are kept in
`archive/scripts/`, which is not published.

| Path | Contents |
| --- | --- |
| `collect_results.py` | The main scorer |
| `weighted_summary.py` | N-weighted aggregates over the scorer's output |
| `download_checkpoints.sh`, `download_third_party.sh` | Weight setup |
| `package_zenodo.sh` | Builds the Zenodo deposit |
| `make_source_data.py` | Builds the Source Data file for the journal |
| `cv/` | Davis and KIBA cross-validation |
| `leakage/` | The ChEMBL35 and CASP16 training-overlap audits |
| `tables/` | Paper tables |
| `figures/` | Paper figures |
| `data_prep/` | Structure conversion for the released inputs |
| `chain_ablation/` | The longest-chain versus concatenation study |

Run everything from the repository root: paths inside the scripts are relative to
it.

## `collect_results.py` and `weighted_summary.py`

`collect_results.py` writes `results/benchmark_summary.csv`,
`results/per_target_evaluation.csv` and `results/benchmark_failures.csv` from
whatever is in `outputs/`; `weighted_summary.py` reads those and writes
`results/weighted_summary.csv`. Every table and figure below reads one of the
four, so run these two first, in that order.

`outputs/` keeps exploratory runs long after their configs move to
`archive/configs/`, so the collector scores a run only when both
`configs/model/<model>.yaml` and `configs/dataset/<dataset>.yaml` exist, and
applies the same test to the rows it would otherwise carry over from the
previous file. Archiving a config is therefore enough to retire its results.

## `package_zenodo.sh`

Builds the four tarballs of the Zenodo deposit, about 6 GB in total: the
benchmark inputs and split partitions, the AlphaFold3 structures, the
PLABench-trained checkpoints, and the per-model predictions with their scored
metrics. `--dry-run` lists the members and sizes without writing anything.

Membership is decided by provenance, following `data/SOURCES.tsv`: PLABench
artifacts ship, corpora other people built are linked. So the PDBbind partition
goes in as `compound_id,split` and not as the rows it was cut from, the CSAR-HiQ
structures stay out while their standardised tables go in, and Davis and KIBA
ship as CSVs without the MixingDTA pickles. The AlphaFold3 archive carries the
modification notice its Output Terms of Use require.

Predictions are filtered by the same live-config rule the scorer uses, so runs
whose configs have moved to `archive/configs/` stay out. The PDBbind 2020
refined 4,465 / 497 split is not in the repository; `PDBBIND_SPLITS` points at
it and defaults to the path it was built at. The copies in
`data/pdbbind2020/{train,val,test}.csv` are header-only stubs and are not what
the deposit ships.

## `make_source_data.py`

Writes `source_data/` and `source_data.zip`: one CSV per numbered figure panel
and per numbered table, named so each file maps to exactly one thing in the
manuscript. Nature Communications accepts a zipped folder as well as a workbook,
and the values already exist as CSV, so nothing is rounded on the way through.

Two items have no file and should not. Figure 1 is a schematic, and Table 1 is
the model inventory, which lists versions and commits rather than measured
values.

Figures 2 and 4 and all four supplementary tables are a copy of an existing
file. Figure 3a and Tables 3 and 4 are not: the scripts that produce them
recompute their numbers while rendering and keep no CSV. Rather than
reimplementing the aggregation and risking a drift from what the manuscript
prints, this imports each script and calls its own functions, so the numbers are
the same ones by construction. Table 4 is the leakage-filtered arm, which is
what `main.tex` prints.

This one goes to the journal, not to Zenodo, and is gitignored.

## `cv/`

| Script | Produces |
| --- | --- |
| `run_deepdta_cv.py`, `run_llf_cv.py`, `run_mixingdta_cv.py` | `outputs/<model>/<config>/fold_<n>/predictions.csv` for the Davis and KIBA five-fold splits |
| `collect_kiba_davis_cv.py` | The Davis and KIBA rows of `results/benchmark_summary.csv`, plus per-fold metrics in `analysis/davis_kiba_cv_metrics.csv` |

## `leakage/`

| Script | Produces |
| --- | --- |
| `chembl35_leakage_std_recheck.py` | `results/chembl35_leakage/per_corpus_std/*.csv`: standardized ligand similarity against 14 training corpora. Also imported as a library by `casp16_ligand_leakage.py` |
| `apply_leakage_filter.py` | `data/chembl35/chembl35_filtered_input.csv`: defines the filtered benchmark and writes the per-target removal ledger |
| `eval_chembl35_filtered.py` | `results/chembl35/per_target_evaluation_chembl35_{full,filtered}.csv` and the wide per-class tables, feeding both the ChEMBL35 table and the family heatmap |
| `casp16_ligand_leakage.py` | `results/casp16_leakage/per_corpus.csv`: which CASP16 ligands appear in which training corpus |
| `casp16_triple_leakage.py` | `results/casp16_leakage/{triples,per_model,triple_detail,label_agreement}.csv`: protein, ligand and label leakage per model |
| `casp16_leak_sensitivity.py` | `results/casp16_leakage/l3000_sensitivity.csv`: how the leaderboard moves when leaked entries are dropped |
| `casp16_leak_removed_table.py` | `results/tables/casp16_stage1_leak_removed.{tex,csv}`: the stage-1 table with leaked complexes removed |
| `casp16_leak_dating.py` | Prints the BindingDB deposition dates behind the temporal-cutoff claim. Writes nothing |

Order: `chembl35_leakage_std_recheck.py` for every corpus, then
`apply_leakage_filter.py`, then `eval_chembl35_filtered.py`. The CASP16 scripts
are independent of the ChEMBL35 ones apart from the shared similarity code.

## `tables/`

| Script | Produces |
| --- | --- |
| `casp16_recompute_stage1.py` | `results/tables/casp16_stage1.tex`: the CASP16 stage-1 table |
| `casp16_stage1_kendall.py` | `results/casp16/casp16_stage1_kendall.csv`: per-series Kendall's tau plus the N-weighted average, the input to the CASP16 figure |
| `emit_table4.py` | `results/tables/table4_{full,filtered}[_with_ensemble].tex`: the ChEMBL35 summary tables |
| `per_class_summary.py` | `results/chembl35/per_class_summary_chembl35.csv` |
| `per_class_pearson_wide_with_casp16.py` | `results/chembl35/per_class_pearson_wide_chembl35_casp16.csv`, the only input to the heatmap |

## `figures/`

| Script | Produces |
| --- | --- |
| `plot_per_class_heatmap.py` | `results/figures/per_class_pearson_heatmap[_filtered].pdf`: the protein-family heatmap |
| `plot_casp16_kendall.py` | `results/figures/casp16_kendall.pdf`: the CASP16 ranking figure |
| `plot_ladder_panels.py` | `results/figures/fig_ladder_panels.pdf`: the four-rung pose-quality ladder and its interaction panel |

The heatmap needs `tables/per_class_summary.py` and then
`tables/per_class_pearson_wide_with_casp16.py` first. The other two are
independent.

## `data_prep/`

| Script | Sets up |
| --- | --- |
| `gen_cofold_structures.py` | Converts AlphaFold3 CIF output into the released `protein.pdb` / `ligand.sdf` / `ligand.mol2` triples |
| `gen_boltz2_ligand_mol2.py` | Extracts Boltz-2 ligand MOL2 files, the first rung of the pose ladder |
| `rescue_af3_conversion_failures.py` | Recovers AlphaFold3 cases whose top-ranked sample will not convert, by sweeping all ten samples in `pair_iptm` order under three conversion strategies. It produced 11 of the released ChEMBL35 ligand structures; pass `--keep_sdf_dir` to write them |

## Machine-specific paths

Model interpreters are environment variables with the benchmark defaults baked
in; see the environment table in the top-level README. Two scripts still hold
absolute paths to data outside the repository and need editing before they run
elsewhere: the training-corpus locations in
`leakage/chembl35_leakage_std_recheck.py` and the Boltz-2 summary CSVs in
`data_prep/gen_boltz2_ligand_mol2.py`. `chain_ablation/` has its own note.
