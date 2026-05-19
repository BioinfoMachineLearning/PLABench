"""Combine HiQBind + SPINDR into LLF-ready train / val CSVs (Stage 2 finetune).

This is the post-2026-05-03 stage2 — PDBbind2020 has been dropped per FlowR-
paper convention and per LLF experiments showing that PDBbind general-set
inclusion adds noise on chembl35 ranking (see flowr_data.md §11 and
chembl35-distribution analysis under scripts/compare_train_test_distribution.py).

Stage 1 = pretraining (bindingmoad + bindingnet, dock-derived large set).
Stage 2 = curated finetuning (HiQBind + SPINDR, crystal-quality only).
Cross-stage data is intentionally not re-deduped — "1 是 1, 2 是 2".

Per spec:
  1. dedup KEY = (sequence, smiles)  — same convention as stage1; HiQBind /
     SPINDR both have pdb_id but the (seq, smi) string identity is what the
     model actually consumes.
  2. dedup PRIORITY: hiqbind > spindr  — HiQBind is real PDB crystals,
     SPINDR is cofolded poses; trust crystal data on exact match.
  3. val starts from native val ∪ native test  (per stage convention;
     ChEMBL35 = real test).  Native val (~196) is small, so we augment by
     sampling (1000 - len(val)) rows from train (seed=42) into val_aug.
     train_aug is train minus the sampled rows.  Final outputs ONLY include
     the augmented pair (no native split co-exists, to avoid confusion).
  4. Leakage check key = (sequence, smiles) — must be 0 between train and val.

CSV columns (LLF-compatible, identical to stage1 schema):
  compound_id          ← pdb_id (lowercased)
  target_sequence      ← sequence (longest 4.5Å contact chain from new dump)
  compound_iso_smiles  ← smiles (post-uncharger canonical isomeric)
  affinity             ← float, standard -log10(M) = pX
  status               ← always 'OK'
  _src_dataset         ← 'hiqbind' | 'spindr'
  _affinity_source     ← 'pkd'/'pki'/'pic50'/'pec50'

Run:
  python scripts/combine_for_llf_stage2.py
"""
import os
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
log = logging.getLogger('combine_stage2')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(DATA_ROOT, 'hiqbind_spindr_llf_stage2')

# (tag, priority, full.csv path)  — lower priority number wins in dedup
DATASETS = [
    ('hiqbind', 0, 'data/hiqbind/full.csv'),
    ('spindr',  1, 'data/spindr/full.csv'),
]

SPLIT_PRIORITY = {'test': 3, 'val': 2, 'train': 1, 'none': 0}

OUTPUT_FIELDS = [
    'compound_id', 'target_sequence', 'compound_iso_smiles', 'affinity',
    'status', '_src_dataset', '_affinity_source',
]


def load_one(ds_tag, prio, rel_path):
    full_path = os.path.join(ROOT, rel_path)
    df = pd.read_csv(full_path,
                     usecols=['pdb_id', 'sequence', 'smiles',
                              'affinity', 'affinity_source', 'split'])
    df['pdb_id'] = df['pdb_id'].astype(str).str.lower()
    df['_ds'] = ds_tag
    df['_priority'] = prio
    log.info('  loaded %-12s  rows=%6d  splits=%s  src=%s',
             ds_tag, len(df),
             dict(df['split'].value_counts()),
             dict(df['affinity_source'].value_counts()))
    return df


def inherit_split(splits):
    return max(splits, key=lambda s: SPLIT_PRIORITY.get(s, -1))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    log.info('loading 2 source datasets...')
    parts = [load_one(ds, prio, p) for ds, prio, p in DATASETS]
    all_df = pd.concat(parts, ignore_index=True)
    log.info('concat total: %d rows', len(all_df))

    # drop rows with empty sequence or smiles (would corrupt dedup key)
    pre = len(all_df)
    all_df = all_df[all_df['sequence'].notna() & (all_df['sequence'] != '')]
    all_df = all_df[all_df['smiles'].notna() & (all_df['smiles'] != '')]
    if len(all_df) < pre:
        log.info('  dropped %d rows with empty seq/smiles', pre - len(all_df))

    # Step A — inherited split per (sequence, smiles) group
    log.info('computing inherited split per (sequence, smiles) group...')
    inherited = (
        all_df.groupby(['sequence', 'smiles'])['split']
        .apply(lambda s: inherit_split(s.tolist()))
        .reset_index()
        .rename(columns={'split': 'inherited_split'})
    )
    log.info('  unique (seq, smi) groups: %d', len(inherited))

    # Step B — priority dedup (hiqbind beats spindr on exact match)
    log.info('priority dedup (hiqbind > spindr) by (sequence, smiles)...')
    dedup = (
        all_df.sort_values('_priority')
        .drop_duplicates(subset=['sequence', 'smiles'], keep='first')
        .reset_index(drop=True)
    )
    log.info('  dedup rows: %d (removed %d cross-source duplicates)',
             len(dedup), len(all_df) - len(dedup))

    # merge inherited split (overrides per-row split)
    dedup = (
        dedup.drop(columns=['split'])
        .merge(inherited, on=['sequence', 'smiles'])
        .rename(columns={'inherited_split': 'split'})
    )

    log.info('final split distribution: %s', dict(dedup['split'].value_counts()))

    train = dedup[dedup['split'].isin(['train', 'none'])].copy()
    val   = dedup[dedup['split'].isin(['val', 'test'])].copy()
    log.info('train: %d   val: %d', len(train), len(val))

    # leakage check
    train_keys = set(zip(train['sequence'], train['smiles']))
    val_keys = set(zip(val['sequence'], val['smiles']))
    leaks = train_keys & val_keys
    log.info('leakage check: train ∩ val (key=(seq,smi)) = %d (expected 0)', len(leaks))
    assert len(leaks) == 0, 'LEAKAGE — split inheritance is broken'

    def to_llf(df):
        out = pd.DataFrame()
        out['compound_id']         = df['pdb_id']
        out['target_sequence']     = df['sequence']
        out['compound_iso_smiles'] = df['smiles']
        out['affinity']            = df['affinity']
        out['status']              = 'OK'
        out['_src_dataset']        = df['_ds']
        out['_affinity_source']    = df['affinity_source']
        return out[OUTPUT_FIELDS]

    train_llf = to_llf(train)
    val_llf   = to_llf(val)

    # Augment val to VAL_TARGET by sampling from train (seed for reproducibility)
    VAL_TARGET = 1000
    SEED = 42
    n_native_val = len(val_llf)
    n_sample = max(0, VAL_TARGET - len(val_llf))
    if n_sample > 0 and n_sample <= len(train_llf):
        sampled = train_llf.sample(n=n_sample, random_state=SEED)
        train_out = train_llf.drop(sampled.index).reset_index(drop=True)
        val_out = pd.concat([val_llf, sampled], ignore_index=True)
        log.info('augmented val: native %d + sampled %d (seed=%d) = %d',
                 n_native_val, n_sample, SEED, len(val_out))
    else:
        log.warning('no augment needed: native val %d already ≥ target %d',
                    len(val_llf), VAL_TARGET)
        train_out = train_llf
        val_out = val_llf
        n_sample = 0

    train_out.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
    val_out.to_csv(os.path.join(OUT_DIR, 'val.csv'), index=False)
    log.info('wrote: %s/train.csv  (%d rows)', OUT_DIR, len(train_out))
    log.info('wrote: %s/val.csv    (%d rows)', OUT_DIR, len(val_out))

    log_path = os.path.join(OUT_DIR, 'combine_log.txt')
    with open(log_path, 'w') as f:
        f.write('hiqbind + spindr → LLF combined dataset (Stage 2 finetune, no PDBbind)\n')
        f.write('=' * 60 + '\n\n')

        f.write('Source row counts:\n')
        for ds, prio, _ in DATASETS:
            n = (all_df['_ds'] == ds).sum()
            f.write(f'  {ds:12s} (priority {prio}): {n}\n')
        f.write(f'  concat total: {len(all_df)}\n\n')

        f.write(f'Unique (sequence, smiles) groups: {len(inherited)}\n')
        f.write(f'Cross-dataset exact-match duplicates removed by priority: '
                f'{len(all_df) - len(dedup)} ({100*(len(all_df)-len(dedup))/len(all_df):.2f}%)\n\n')

        f.write('Final dedup rows by source:\n')
        for ds, n in dedup['_ds'].value_counts().items():
            orig = (all_df['_ds'] == ds).sum()
            f.write(f'  {ds:12s}: {n:6d}  ({100*n/orig:.1f}% of source kept)\n')

        f.write(f'\nFinal split (after inheritance):\n')
        for s, n in dedup['split'].value_counts().items():
            f.write(f'  {s:6s}: {n}\n')

        f.write(f'\nNative split (before augment):\n')
        f.write(f'  train: {len(train_llf)} rows  (split in train/none)\n')
        f.write(f'  val:   {n_native_val} rows  (split in val/test, '
                f'merged per stage convention; ChEMBL35 = real test)\n')

        f.write(f'\nFinal LLF outputs (val augmented to {VAL_TARGET}):\n')
        f.write(f'  train.csv: {len(train_out)} rows  (native train minus {n_sample} sampled)\n')
        f.write(f'  val.csv:   {len(val_out)} rows  (native {n_native_val} + sampled {n_sample}, seed={SEED})\n')

        f.write(f'\nBy affinity_source (in train, native):\n')
        for k, v in train['affinity_source'].value_counts().items():
            f.write(f'  {k:12s}: {v}\n')

        f.write(f'\nBy affinity_source (in val, native):\n')
        for k, v in val['affinity_source'].value_counts().items():
            f.write(f'  {k:12s}: {v}\n')

        f.write(f'\nLeakage check (train ∩ val by (sequence, smiles)): {len(leaks)} ✓\n')
    log.info('wrote: %s', log_path)


if __name__ == '__main__':
    main()
