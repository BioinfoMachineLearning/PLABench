"""Combine BindingMOAD + BindingNet into LLF-ready train / val CSVs (Stage 1).

Stage 1 = pretraining set (bigger but more docking-derived),
Stage 2 = curated crystal-quality finetuning set (hiqbind + pdbbind + spindr).
The two stages are intentionally not cross-deduped — user spec:
  "stage1 和 stage2 之间的事情不用管, 1 就是 1, 2 就是 2".

Per-user spec (2026-05-02):
  1. dedup KEY = (sequence, smiles)  [NOT (pdb_id, smiles)]
        — bindingnet has no pdb_id (ChEMBL-derived), so pdb_id key would collapse
          all bindingnet rows into one bucket. (sequence, smiles) is the
          actual training-input identity at the model layer.
  2. dedup PRIORITY: bindingmoad > bindingnet
        — bindingmoad is real PDB crystals, bindingnet is ChEMBL activities
          docked onto PDB templates → trust crystal data first.
  3. INCLUDE bindingnet's `logaff_only` rows (~14.6k)
        — confirmed bindingnet logaff is standard -log10(M) (positive pX).
        - Different from HiQBind logaff which is log10(M) (negative).
  4. val = native val ∪ native test  (per stage2 convention; ChEMBL35 = real test)
  5. NO v2 augment  — val already has ~3.3k rows, plenty of signal.
  6. Leakage check key = (sequence, smiles)  — must be 0 between train and val.

CSV columns (LLF-compatible, identical to stage2):
  compound_id          ← system_id (PDB-id-like for moad, ChEMBL-like for net)
  target_sequence      ← sequence (longest 4.5Å contact chain)
  compound_iso_smiles  ← smiles (canonical isomeric, longest fragment, sanitized)
  affinity             ← float, standard -log10(M) = pX
  status               ← always 'OK'
  _src_dataset         ← 'bindingmoad' | 'bindingnet'
  _affinity_source     ← 'pkd'/'pki'/'pic50'/'logaff_only'

Run:
  python scripts/combine_for_llf_stage1.py
"""
import os
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
log = logging.getLogger('combine_stage1')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(DATA_ROOT, 'bindingmoad_bindingnet_llf_stage1')

# (tag, priority, full.csv path)  — lower priority number wins in dedup
DATASETS = [
    ('bindingmoad', 0, 'data/bindingmoad_flowr/full.csv'),
    ('bindingnet',  1, 'data/bindingnet_high/full.csv'),
]

SPLIT_PRIORITY = {'test': 3, 'val': 2, 'train': 1, 'none': 0}

OUTPUT_FIELDS = [
    'compound_id', 'target_sequence', 'compound_iso_smiles', 'affinity',
    'status', '_src_dataset', '_affinity_source',
]


def load_one(ds_tag, prio, rel_path):
    full_path = os.path.join(ROOT, rel_path)
    df = pd.read_csv(full_path,
                     usecols=['system_id', 'sequence', 'smiles',
                              'affinity', 'affinity_source', 'split'],
                     dtype={'system_id': str})
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

    # drop rows with empty/missing sequence or smiles (would corrupt dedup key)
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

    # Step B — priority dedup (bindingmoad wins over bindingnet on exact match)
    log.info('priority dedup (bindingmoad > bindingnet) by (sequence, smiles)...')
    dedup = (
        all_df.sort_values('_priority')
        .drop_duplicates(subset=['sequence', 'smiles'], keep='first')
        .reset_index(drop=True)
    )
    log.info('  dedup rows: %d (removed %d cross-source duplicates)',
             len(dedup), len(all_df) - len(dedup))

    # merge inherited split (overrides per-row split — prevents val/test leak
    # if e.g. bindingmoad marked a row 'test' but bindingnet has same (seq,smi)
    # marked 'train')
    dedup = (
        dedup.drop(columns=['split'])
        .merge(inherited, on=['sequence', 'smiles'])
        .rename(columns={'inherited_split': 'split'})
    )

    log.info('final split distribution: %s', dict(dedup['split'].value_counts()))

    # split into train / val per stage1 spec
    train = dedup[dedup['split'].isin(['train', 'none'])].copy()
    val   = dedup[dedup['split'].isin(['val', 'test'])].copy()
    log.info('train: %d   val: %d', len(train), len(val))

    # leakage check — key = (sequence, smiles), must be 0
    train_keys = set(zip(train['sequence'], train['smiles']))
    val_keys = set(zip(val['sequence'], val['smiles']))
    leaks = train_keys & val_keys
    log.info('leakage check: train ∩ val (key=(seq,smi)) = %d (expected 0)', len(leaks))
    assert len(leaks) == 0, 'LEAKAGE — split inheritance is broken'

    def to_llf(df):
        out = pd.DataFrame()
        out['compound_id']         = df['system_id']
        out['target_sequence']     = df['sequence']
        out['compound_iso_smiles'] = df['smiles']
        out['affinity']            = df['affinity']
        out['status']              = 'OK'
        out['_src_dataset']        = df['_ds']
        out['_affinity_source']    = df['affinity_source']
        return out[OUTPUT_FIELDS]

    train_llf = to_llf(train)
    val_llf   = to_llf(val)
    train_llf.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
    val_llf.to_csv(os.path.join(OUT_DIR, 'val.csv'), index=False)
    log.info('wrote: %s/train.csv  (%d rows)', OUT_DIR, len(train_llf))
    log.info('wrote: %s/val.csv    (%d rows)', OUT_DIR, len(val_llf))

    # log file
    log_path = os.path.join(OUT_DIR, 'combine_log.txt')
    with open(log_path, 'w') as f:
        f.write('bindingmoad + bindingnet → LLF combined dataset (Stage 1)\n')
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

        f.write(f'\nFinal LLF outputs:\n')
        f.write(f'  train.csv: {len(train_llf)} rows  (split in train/none)\n')
        f.write(f'  val.csv:   {len(val_llf)} rows  (split in val/test, '
                f'merged per stage convention; ChEMBL35 = real test)\n')

        f.write(f'\nBy affinity_source (in train):\n')
        for k, v in train['affinity_source'].value_counts().items():
            f.write(f'  {k:12s}: {v}\n')

        f.write(f'\nBy affinity_source (in val):\n')
        for k, v in val['affinity_source'].value_counts().items():
            f.write(f'  {k:12s}: {v}\n')

        f.write(f'\nLeakage check (train ∩ val by (sequence, smiles)): {len(leaks)} ✓\n')
    log.info('wrote: %s', log_path)


if __name__ == '__main__':
    main()
