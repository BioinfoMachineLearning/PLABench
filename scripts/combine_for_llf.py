"""Combine HiQBind + PDBbind2020 + SPINDR into LLF-ready train / val CSVs.

Pipeline:
  1. Load `data/{hiqbind,pdbbind2020,spindr}/full.csv` (all framework-dump versions).
  2. concat → group by (pdb_id, smiles).
  3. Step A — inherit split from any source:
        if any row has split='test'  → group split = 'test'
        elif any row has split='val' → group split = 'val'
        elif any row has split='train' → 'train'
        else → 'none'
     (avoids leakage where SPINDR/HiQBind mark something as val/test but PDBbind
      has the same (pdb,smi) with split='none' which would default to train)
  4. Step B — priority dedup hiqbind > pdbbind > spindr; keep one row per group
     for actual data values (sequence, affinity, smiles), but use Step A's split.
  5. Output:
        data/hiqbind_pdbbind_spider_llf/train.csv   (split in {train, none})
        data/hiqbind_pdbbind_spider_llf/val.csv     (split in {val, test} —
                                                     user spec: no test set,
                                                     ChEMBL35 will be the test)
        data/hiqbind_pdbbind_spider_llf/combine_log.txt

CSV columns (LLF-compatible):
  compound_id          ← pdb_id
  target_sequence      ← sequence
  compound_iso_smiles  ← smiles
  affinity             ← affinity (-log10(M) standard pX)
  status               ← always 'OK' (no SDF rescue path in framework)
  _src_dataset         ← which dataset this row came from after dedup
  _affinity_source     ← original 'pkd'/'pki'/'pic50'/'pec50'

V2 output (val augmented to ~1000):
  train_v2.csv  ← train minus randomly-sampled 786 rows (seed=42)
  val_v2.csv    ← val (214) + sampled 786 = 1000 rows

V3 output (default; val=2000 + new SMILES standardization upstream):
  train_v3.csv  ← train minus randomly-sampled (2000 - len(val)) rows (seed=42)
  val_v3.csv    ← val + sampled = 2000 rows
  Inputs come from re-dumped sources (data/{hiqbind,pdbbind2020,spindr}/full.csv
  generated AFTER the 2026-05-03 ligand_utils.canonical_smiles() patch:
  Uncharger + RemoveHs + AssignStereochemistry(cleanIt) + canonical roundtrip).
  Both v2 and v3 are written every run; v3 is the default for new training.

Note on leakage policy (per user 2026-05-02 decision):
  "Strings differ → samples differ" — for training, we treat sequence/SMILES
  string mismatch as truly different examples. This loosens dedup (keeps more
  rows) and keeps cross-source canonicalization edge cases (e.g., HiQBind
  metadata SMILES vs PDBbind SDF SMILES of the same molecule but different
  protonation/explicit-H representation that RDKit canonical doesn't unify)
  as distinct training samples. Leakage check uses (pdb_id, smiles) — the
  stricter (pdb_id only) check would catch ~55 cross-source SMILES-string
  mismatches but those are accepted as "different inputs" for training.
  Real evaluation uses ChEMBL35 holdout, val is just a development signal.

Run:
  python scripts/combine_for_llf.py
"""
import os
import csv
import logging
from collections import Counter

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
log = logging.getLogger('combine_for_llf')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(DATA_ROOT, 'hiqbind_pdbbind_spider_llf')

DATASETS = [
    ('hiqbind',     0, 'data/hiqbind/full.csv'),
    ('pdbbind2020', 1, 'data/pdbbind2020/full.csv'),
    ('spindr',      2, 'data/spindr/full.csv'),
]

SPLIT_PRIORITY = {'test': 3, 'val': 2, 'train': 1, 'none': 0}

OUTPUT_FIELDS = [
    'compound_id', 'target_sequence', 'compound_iso_smiles', 'affinity',
    'status', '_src_dataset', '_affinity_source',
    '_target_id', '_chain_id_used', '_uniprot_id',
]


def load_one(ds_tag, prio, rel_path):
    full_path = os.path.join(ROOT, rel_path)
    head = pd.read_csv(full_path, nrows=0).columns.tolist()
    base = ['pdb_id', 'sequence', 'smiles', 'affinity', 'affinity_source', 'split',
            'target_id', 'chain_id_used']
    cols = [c for c in base if c in head]
    if 'uniprot_id' in head:
        cols.append('uniprot_id')
    df = pd.read_csv(full_path, usecols=cols)
    # ensure missing columns exist (SPINDR/PDBbind have no uniprot_id) so concat aligns
    for c in base + ['uniprot_id']:
        if c not in df.columns:
            df[c] = pd.NA
    df['pdb_id'] = df['pdb_id'].astype(str).str.lower()
    df['_ds'] = ds_tag
    df['_priority'] = prio
    log.info('  loaded %-12s  rows=%6d  splits=%s',
             ds_tag, len(df), dict(df['split'].value_counts()))
    return df


def inherit_split(splits):
    """Return the most-strict split present in the group."""
    return max(splits, key=lambda s: SPLIT_PRIORITY.get(s, -1))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    log.info('loading 3 source datasets...')
    parts = [load_one(ds, prio, p) for ds, prio, p in DATASETS]
    all_df = pd.concat(parts, ignore_index=True)
    log.info('concat total: %d rows', len(all_df))

    # Dedup key: (sequence, smiles).  Per user policy "strings differ → samples
    # differ" — same pdb_id with different chain selection across sources gives
    # different sequence strings, those are kept as DISTINCT training samples
    # rather than merged.  Earlier (pdb_id, smiles) key over-merged ~6k rows.
    DEDUP_KEY = ['sequence', 'smiles']

    # Step A — inherited split per dedup-key group
    log.info('computing inherited split per %s group...', tuple(DEDUP_KEY))
    inherited = (
        all_df.groupby(DEDUP_KEY)['split']
        .apply(lambda s: inherit_split(s.tolist()))
        .reset_index()
        .rename(columns={'split': 'inherited_split'})
    )
    log.info('  unique groups: %d', len(inherited))

    # Step B — priority dedup
    log.info('priority dedup (hiqbind > pdbbind2020 > spindr)...')
    dedup = (
        all_df.sort_values('_priority')
        .drop_duplicates(subset=DEDUP_KEY, keep='first')
        .reset_index(drop=True)
    )
    log.info('  dedup rows: %d (removed %d)', len(dedup), len(all_df) - len(dedup))

    # merge inherited split (overrides per-row split)
    dedup = (
        dedup.drop(columns=['split'])
        .merge(inherited, on=DEDUP_KEY)
        .rename(columns={'inherited_split': 'split'})
    )

    log.info('final split distribution: %s', dict(dedup['split'].value_counts()))

    # Step C1 — fill uniprot_id for PDBbind/SPINDR rows by reverse-lookup against
    # HiQBind on shared pdb_id. HiQBind covers ~8200/12378 PDBbind PDBs and
    # ~4900/6065 SPINDR PDBs, so this recovers most of the missing uniprots.
    hiq_pdb2uni = (
        parts[0][parts[0]['uniprot_id'].notna()]
        .drop_duplicates(subset='pdb_id', keep='first')
        .set_index('pdb_id')['uniprot_id']
    )
    n0 = dedup['uniprot_id'].notna().sum()
    miss = dedup['uniprot_id'].isna()
    dedup.loc[miss, 'uniprot_id'] = dedup.loc[miss, 'pdb_id'].map(hiq_pdb2uni)
    n1 = dedup['uniprot_id'].notna().sum()
    log.info('uniprot fill C1 (← HiQBind by pdb_id):       %d → %d (+%d)', n0, n1, n1 - n0)

    # Step C2 — SIFTS reverse-lookup on (pdb_id, chain_id_used) for whatever's
    # still missing. SIFTS PDBs are lowercase (matches our format); chain is
    # single-char uppercase (matches our chain_id_used format). Take first
    # SP_PRIMARY per (PDB, CHAIN) when SIFTS reports multiple residue ranges.
    sifts_path = os.path.join(ROOT, 'data/sifts/pdb_chain_uniprot.tsv.gz')
    if os.path.exists(sifts_path):
        sifts = pd.read_csv(sifts_path, sep='\t', skiprows=1, low_memory=False,
                            usecols=['PDB', 'CHAIN', 'SP_PRIMARY'])
        sifts_map = (
            sifts.dropna(subset=['SP_PRIMARY'])
                 .drop_duplicates(['PDB', 'CHAIN'], keep='first')
                 .set_index(['PDB', 'CHAIN'])['SP_PRIMARY']
        )
        miss = dedup['uniprot_id'].isna()
        keys = list(zip(dedup.loc[miss, 'pdb_id'].astype(str),
                        dedup.loc[miss, 'chain_id_used'].astype(str)))
        dedup.loc[miss, 'uniprot_id'] = [sifts_map.get(k) for k in keys]
        n2 = dedup['uniprot_id'].notna().sum()
        log.info('uniprot fill C2 (← SIFTS by pdb+chain):     %d → %d (+%d, sifts rows=%d)',
                 n1, n2, n2 - n1, len(sifts_map))
    else:
        log.warning('SIFTS file not found at %s — skipping C2', sifts_path)
        n2 = n1

    # split into train / val (per user spec: no test, val = native val + native test)
    train = dedup[dedup['split'].isin(['train', 'none'])].copy()
    val   = dedup[dedup['split'].isin(['val', 'test'])].copy()

    log.info('train: %d   val: %d', len(train), len(val))

    # leakage sanity check (uses same dedup key)
    train_keys = set(zip(*[train[k] for k in DEDUP_KEY]))
    val_keys = set(zip(*[val[k] for k in DEDUP_KEY]))
    leaks = train_keys & val_keys
    log.info('leakage check: train ∩ val = %d (expected 0)', len(leaks))
    assert len(leaks) == 0, 'LEAKAGE — something is broken'

    # rename to LLF-compatible columns
    def to_llf(df):
        out = pd.DataFrame()
        out['compound_id']         = df['pdb_id']
        out['target_sequence']     = df['sequence']
        out['compound_iso_smiles'] = df['smiles']
        out['affinity']            = df['affinity']
        out['status']              = 'OK'
        out['_src_dataset']        = df['_ds']
        out['_affinity_source']    = df['affinity_source']
        out['_target_id']          = df['target_id']
        out['_chain_id_used']      = df['chain_id_used']
        out['_uniprot_id']         = df['uniprot_id']
        return out[OUTPUT_FIELDS]

    train_llf = to_llf(train)
    val_llf   = to_llf(val)
    train_llf.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
    val_llf.to_csv(os.path.join(OUT_DIR, 'val.csv'), index=False)
    log.info('wrote: %s/train.csv  (%d rows)', OUT_DIR, len(train_llf))
    log.info('wrote: %s/val.csv    (%d rows)', OUT_DIR, len(val_llf))

    # V2 / V3 — augment val by random sampling from train (seed=42 for reproducibility)
    SEED = 42

    def make_augmented(target_size, suffix):
        n_sample = max(0, target_size - len(val_llf))
        if n_sample <= 0 or n_sample > len(train_llf):
            log.warning('skipping %s: n_sample=%d invalid', suffix, n_sample)
            return None, None
        sampled = train_llf.sample(n=n_sample, random_state=SEED)
        train_aug = train_llf.drop(sampled.index).reset_index(drop=True)
        val_aug = pd.concat([val_llf, sampled], ignore_index=True)
        train_aug.to_csv(os.path.join(OUT_DIR, f'train_{suffix}.csv'), index=False)
        val_aug.to_csv(os.path.join(OUT_DIR, f'val_{suffix}.csv'), index=False)
        log.info('wrote: %s/train_%s.csv (%d rows)', OUT_DIR, suffix, len(train_aug))
        log.info('wrote: %s/val_%s.csv   (%d rows, +%d sampled from train, seed=%d)',
                 OUT_DIR, suffix, len(val_aug), n_sample, SEED)
        return train_aug, val_aug

    train_v2, val_v2 = make_augmented(1000, 'v2')   # legacy
    train_v3, val_v3 = make_augmented(2000, 'v3')   # default for new training

    # log file
    log_path = os.path.join(OUT_DIR, 'combine_log.txt')
    with open(log_path, 'w') as f:
        f.write('hiqbind + pdbbind2020 + spindr → LLF combined dataset\n')
        f.write('=' * 60 + '\n\n')

        f.write('Source row counts:\n')
        for ds, prio, _ in DATASETS:
            n = (all_df['_ds'] == ds).sum()
            f.write(f'  {ds:12s} (priority {prio}): {n}\n')
        f.write(f'  concat total: {len(all_df)}\n\n')

        f.write(f'Unique (sequence, smiles) groups: {len(inherited)}\n')
        f.write(f'Cross-dataset duplicates removed by priority: '
                f'{len(all_df) - len(dedup)} ({100*(len(all_df)-len(dedup))/len(all_df):.1f}%)\n\n')

        f.write('Final dedup rows by source:\n')
        for ds, n in dedup['_ds'].value_counts().items():
            orig = (all_df['_ds'] == ds).sum()
            f.write(f'  {ds:12s}: {n:6d}  ({100*n/orig:.1f}% of source kept)\n')

        f.write(f'\nFinal split (after inheritance):\n')
        for s, n in dedup['split'].value_counts().items():
            f.write(f'  {s:6s}: {n}\n')

        f.write(f'\nFinal LLF outputs:\n')
        f.write(f'  train.csv: {len(train_llf)} rows  (split in train/none)\n')
        f.write(f'  val.csv:   {len(val_llf)} rows  (split in val/test, per user spec '
                f'no separate test set — ChEMBL35 will be the test)\n')

        f.write(f'\nBy affinity_source (in train):\n')
        for k, v in train['affinity_source'].value_counts().items():
            f.write(f'  {k:12s}: {v}\n')

        f.write(f'\nLeakage check (train ∩ val by (sequence,smiles)): {len(leaks)} ✓\n')

        f.write(f'\nUniprot coverage (after HiQBind reverse-lookup + SIFTS):\n')
        f.write(f'  in train: {train["uniprot_id"].notna().sum()}/{len(train)} '
                f'({100*train["uniprot_id"].notna().sum()/len(train):.1f}%) rows have _uniprot_id\n')
        f.write(f'  in val:   {val["uniprot_id"].notna().sum()}/{len(val)} '
                f'({100*val["uniprot_id"].notna().sum()/len(val):.1f}%) rows have _uniprot_id\n')
        f.write(f'  unique uniprot_id (train): {train["uniprot_id"].nunique()}\n')
        if train_v2 is not None:
            f.write(f'\nV2 (augmented val=1000, legacy):\n')
            f.write(f'  train_v2.csv: {len(train_v2)} rows\n')
            f.write(f'  val_v2.csv:   {len(val_v2)} rows\n')
        if train_v3 is not None:
            f.write(f'\nV3 (augmented val=2000, default — uses post-2026-05-03 SMILES standardization):\n')
            f.write(f'  train_v3.csv: {len(train_v3)} rows\n')
            f.write(f'  val_v3.csv:   {len(val_v3)} rows\n')
    log.info('wrote: %s', log_path)


if __name__ == '__main__':
    main()
