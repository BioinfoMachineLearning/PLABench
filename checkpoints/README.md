# PLABench checkpoints

Layout:

- `sequence/{deepdta,llf,mixingdta}/`: sequence models. `pdbbind/` holds the
  single PDBbind 2020 refined model (4465/497, 9:1 train/validation split)
  behind the CASF, CSAR, CASP16 and ChEMBL35 rows of
  `results/benchmark_summary.csv`; `{davis,kiba}/{warm,cold_drug,cold_target}/`
  hold the five-fold models.
- `structure/{mfe,flowr_root,flowdock,bapred,haiping}/`: structure models.
  Boltz-2 lives in the `boltz` CLI cache (`~/.boltz/`), not here.
- `ESM3/`: the ESM3 open-small weight used by MixingDTA's on-the-fly protein
  embedder (reached through the `data/weights` link with `INFRA_PROVIDER` set).

Provenance:

- `MANIFEST.tsv`: every checkpoint, whether it was trained by PLABench or
  comes from a third party, and whether it ships in the Zenodo archive.
- `THIRD_PARTY.tsv`: official download URL, SHA256, size and license for each
  third-party file. These are **not** re-uploaded to Zenodo.
- `LINKS.tsv`: fork-path compatibility links recreated by
  `scripts/download_checkpoints.sh` for archive-shipped files.

Install, from the repository root:

1. `bash scripts/download_third_party.sh` fetches every third-party file from
   its official source and verifies the SHA256 (needs `gdown` for the Google
   Drive items and `HF_TOKEN` for the gated ESM3 weight; Boltz-2 fetches its
   own weights on first use).
2. Set `PLABENCH_CHECKPOINT_URL` to the published Zenodo file URL and run
   `bash scripts/download_checkpoints.sh` for the PLABench-trained weights.
   It also recreates the fork links (`--links-only` re-links without touching
   the archive).
