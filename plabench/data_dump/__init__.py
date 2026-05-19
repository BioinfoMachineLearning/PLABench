"""
Reusable framework for dumping FlowR-style / per-system PLA datasets into the
unified LLF/MixingDTA-compatible CSV schema.

Pipeline (per dataset, see plabench.data_dump.base.BaseDumper):

  Step A. Per-row filters (each row decided independently):
    - Subclass-defined pre_filter (sign, resolution, ...)
    - PDB and SDF files exist
    - SDF parses to >=1 heavy atom
    - 4.5 A single-chain contact (drop multi_homo + multi_hetero + no_contact)
    - non-empty extracted sequence

  Step B. Cross-row dedup:
    - Group by (pdb_id, smiles)
    - Keep ALL entries on the chain with the longest extracted sequence
    - Tiebreak among chains: alphabetically-smallest chain_id_used

  Step C. Output:
    - data/{out_dir}/full.csv, train.csv, val.csv, test.csv (streamed)
    - data/{out_dir}/dump_log.txt (counts + drop reasons)

Usage:
    python -m plabench.data_dump.datasets.hiqbind     # one entry per dataset

Adding a new dataset:
    1. Create plabench/data_dump/datasets/{name}.py
    2. Subclass BaseDumper, set DATASET_TAG / OUT_DIR_NAME / EXTRA_FIELDS
    3. Implement iter_candidates() — yield candidate dicts from source
    4. Optionally override pre_filter() for dataset-specific filters
"""
