# Chain-rule ablation

The code behind Supplementary Tables `tab:chainrule` and `tab:did`: retraining the
three sequence models on concatenated multi-chain protein strings versus the
longest resolved chain, three seeds per arm, scored on six test sets.

Switching chain rules moves the mean Pearson correlation by 0.016, less than half
the 0.033 spread from changing the random seed alone. That is why the paper uses
the longest chain throughout.

## Design

Both arms train on the same 90/10 split of PDBbind-2020 refined, with the same
hyperparameters and the same row order. The protein string is the only variable.

Both arms also use a 4,700-residue window, which is wider than either rule needs:
the longest single chain is 1,287 residues and the longest concatenation 4,638, so
nothing is truncated on either side. This matters because LLF's deployed window of
1,200 would have cut 217 concatenated entries against 1 single-chain entry, and an
arm built that way would measure truncation and the chain rule together. Widening
the window changes no weight shape, since both LLF and DeepDTA pool with
`AdaptiveMaxPool1d(1)`.

Because the window differs from the deployed one, these arms are not the paper's
headline numbers and are not meant to replace them. The published checkpoints are
scored alongside as a third reference point.

## Order

Numbered in each script's docstring. Roughly:

1. `build_concat_csvs.py`, `build_concat_testsets.py` rebuild the protein strings
   under both chain rules. `chain_counts.py` records how many chains each entry
   has, so the comparison can be split into single-chain and multi-chain strata.
2. `esm3_extract.py` computes the per-residue ESM3 embeddings both arms need.
3. `build_datasets.py`, `llf_build_data.py` assemble the per-model dataset trees.
4. `deepdta_train.py`, `llf_train.py` train one arm each. `run_fleet.sh` and
   `finish_concat_arm.sh` drive them across seeds.
5. `evaluate_arm.py`, `eval_seq_models.py`, `eval_paper_sets.py` score the arms
   through `plabench.models.*.inference`, the same path that produced the
   manuscript numbers. Scoring the longest-chain arm at the deployed window
   reproduces the published report exactly, which is the check that the shared
   evaluation path changed nothing.
6. `compare_arms.py`, `analyze_seq_arms.py`, `analyze_paper_sets.py` produce the
   paired comparisons in `tab:chainrule`.
7. `diff_in_diff.py` produces `tab:did`. Single-chain test entries have
   byte-identical input in both arms, so any movement there is the retrained model
   rather than the chain rule. The difference of the two differences estimates
   what supplying the missing chains actually buys.

## Reading the DiD

LLF and MEETA come out positive, small and indistinguishable from zero. DeepDTA
comes out significantly negative on CASF-2016: concatenation made the entries whose
input changed worse while the entries whose input did not change got better. That
is what redundancy predicts. DeepDTA max-pools over the sequence axis, and 81% of
multi-chain PDBbind entries are homomers whose chains are at least 95% identical,
so concatenation appends near-duplicate copies that add no new residue and shift
the pooled statistics.

## Paths

These scripts were written as a one-off study and hold absolute paths to the
training corpora, the model checkouts, and their own working tree at
`scratch/concat_ablation/`. That working tree holds 13 GB of ESM3 embeddings and
training runs and is not published. Running the ablation elsewhere means editing
the `W`, `DATA` and `OUT` constants at the top of each file.

The `deepdta` conda environment ships `lib/libstdc++.so.6.0.34` without a
`libstdc++.so.6` symlink, so the loader falls back to `/lib64` and pandas fails on
`GLIBCXX_3.4.29`. Launch it with
`LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6.0.34` rather than editing the
environment.
