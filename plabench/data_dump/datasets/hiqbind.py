"""HiQBind dumper.

Source layout (on /bml NFS):
    metadata: /bml/Lyuwei_data/HiQBind/hiqbind_metadata.csv (32,275 rows)
    structures: /bml/Lyuwei_data/HiQBind/data/{pdb}_{lig}_{chain}_{resi}.{pdb,sdf}
    splits:   /bml/Lyuwei_data/HiQBind/final/splits.npz   (LMDB integer keys)
    LMDB:     /bml/Lyuwei_data/HiQBind/final/             (only used for split idx → system_id)

HiQBind-specific pre-filters (Step A):
    1. Sign == '='                     drop ~933 censored / approximate
    2. Resolution ≤ 2.5 Å              drop ~890 low-quality structures
    3. Affinity present                drop ~4
    4. is_covalent == False            (HiQBind already filtered: 0 covalent in LMDB)

Affinity:
    affinity = -1 * metadata['Log Binding Affinity']  (HiQBind stores log10(M); flip to -log10 = pX)
    affinity_source ← from 'Binding Affinity Measurement' column ('ki' → 'pki', etc.)

target_id:
    prefer dedup'd UniProt ID (single → use it; multi-distinct → ','-joined sorted set)
    fallback to PDB ID if UniProt is empty.

Run:
    python -m plabench.data_dump.datasets.hiqbind
"""
import os
import lmdb
import pickle
import numpy as np
import pandas as pd

from plabench.data_dump.base import BaseDumper

HIQ_ROOT      = '/bml/Lyuwei_data/HiQBind'
META_CSV      = os.path.join(HIQ_ROOT, 'hiqbind_metadata.csv')
DATA_DIR      = os.path.join(HIQ_ROOT, 'data')
LMDB_DIR      = os.path.join(HIQ_ROOT, 'final')
SPLITS_NPZ    = os.path.join(LMDB_DIR, 'splits.npz')

RESOLUTION_MAX_A = 2.5

MEASUREMENT_TO_SOURCE = {
    'ic50': 'pic50',
    'kd':   'pkd',
    'ki':   'pki',
    'ec50': 'pec50',
}


class HiqbindDumper(BaseDumper):
    DATASET_TAG    = 'hiqbind'
    OUT_DIR_NAME   = 'hiqbind'
    EXTRA_FIELDS   = [
        'uniprot_id', 'uniprot_id_count', 'resolution', 'year',
        'affinity_sign', 'affinity_value', 'affinity_unit', 'affinity_db_source',
    ]
    EXPECTED_TOTAL = 32275

    def __init__(self):
        super().__init__()
        self._df = None
        self._sid_to_split = None

    # ── load source data ────────────────────────────────────────────────

    def _load(self):
        self.log.info('reading metadata.csv ...')
        df = pd.read_csv(META_CSV, low_memory=False)
        df['system_id'] = (df['PDBID'].astype(str) + '_' + df['Ligand Name'].astype(str)
                           + '_' + df['Ligand Chain'].astype(str)
                           + '_' + df['Ligand Residue Number'].astype(str))
        df['Resolution_num'] = pd.to_numeric(df['Resolution'], errors='coerce')
        self._df = df

        self.log.info('building split lookup (system_id → split) ...')
        splits = np.load(SPLITS_NPZ)
        idx_to_split = {}
        for s in ('train', 'val', 'test'):
            for i in splits[f'idx_{s}'].tolist():
                idx_to_split[i] = s

        sid_to_split = {}
        env = lmdb.open(LMDB_DIR, readonly=True, lock=False, max_readers=1, subdir=True)
        with env.begin() as txn:
            for k, v in txn.cursor():
                try:
                    idx = int(k.decode())
                    o = pickle.loads(v)
                    sid = o['metadata']['system_id']
                    sid_to_split[sid] = idx_to_split.get(idx, 'none')
                except Exception:
                    continue
        env.close()
        self._sid_to_split = sid_to_split
        self.log.info('  metadata rows: %d  split lookup size: %d', len(df), len(sid_to_split))

    # ── candidate generator ────────────────────────────────────────────

    def iter_candidates(self):
        if self._df is None:
            self._load()
        for _, r in self._df.iterrows():
            sid = r['system_id']
            cand = {
                'system_id':   sid,
                'pdb_id':      str(r['PDBID']),
                'pdb_path':    os.path.join(DATA_DIR, sid + '.pdb'),
                'sdf_path':    os.path.join(DATA_DIR, sid + '.sdf'),
                'compound_id': f"{r['Ligand Name']}:{r['Ligand Chain']}:{r['Ligand Residue Number']}",
                'smiles_raw':  r.get('Ligand SMILES'),
                'split':       self._sid_to_split.get(sid, 'none'),

                # affinity bookkeeping (pre-filtered in pre_filter; final values set here)
                'affinity':         self._compute_affinity(r),
                'affinity_source':  self._affinity_source(r),

                # target_id (UniProt preferred, PDB fallback)
                'target_id':        self._target_id(r),

                # extras
                'uniprot_id':         self._uniprot_dedup_str(r),
                'uniprot_id_count':   self._uniprot_count(r),
                'resolution':         self._safe_float(r.get('Resolution_num')),
                'year':               self._safe_int(r.get('Year')),
                'affinity_sign':      str(r.get('Binding Affinity Sign', '')),
                'affinity_value':     self._safe_float(r.get('Binding Affinity Value')),
                'affinity_unit':      str(r.get('Binding Affinity Unit', '')) if not pd.isna(r.get('Binding Affinity Unit')) else '',
                'affinity_db_source': str(r.get('Binding Affinity Source', '')),

                # raw row for pre_filter
                '_raw_row': r,
            }
            yield cand

    # ── per-row filters ─────────────────────────────────────────────────

    def pre_filter(self, candidate):
        r = candidate['_raw_row']
        # Sign filter
        if r['Binding Affinity Sign'] != '=':
            return 'drop_sign'
        # Resolution
        res = r['Resolution_num']
        if pd.isna(res) or res > RESOLUTION_MAX_A:
            return 'drop_resolution'
        # Affinity available
        log_aff = r.get('Log Binding Affinity')
        if pd.isna(log_aff):
            return 'drop_no_affinity'
        # Measurement type recognized
        meas = str(r.get('Binding Affinity Measurement', '')).strip().lower()
        if meas not in MEASUREMENT_TO_SOURCE:
            return 'drop_unknown_measurement'
        return None

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_affinity(r):
        log_aff = r.get('Log Binding Affinity')
        if pd.isna(log_aff):
            return None
        try:
            return -float(log_aff)
        except Exception:
            return None

    @staticmethod
    def _affinity_source(r):
        meas = str(r.get('Binding Affinity Measurement', '')).strip().lower()
        return MEASUREMENT_TO_SOURCE.get(meas, '')

    @staticmethod
    def _uniprot_parts(r):
        raw = r.get('Protein UniProtID')
        if not isinstance(raw, str) or not raw.strip():
            return []
        return sorted({t.strip() for t in raw.split(',') if t.strip()})

    @classmethod
    def _uniprot_dedup_str(cls, r):
        return ','.join(cls._uniprot_parts(r))

    @classmethod
    def _uniprot_count(cls, r):
        return len(cls._uniprot_parts(r))

    @classmethod
    def _target_id(cls, r):
        s = cls._uniprot_dedup_str(r)
        return s if s else str(r['PDBID'])

    @staticmethod
    def _safe_float(x):
        if x is None or pd.isna(x):
            return ''
        try:
            return float(x)
        except Exception:
            return ''

    @staticmethod
    def _safe_int(x):
        if x is None or pd.isna(x):
            return ''
        try:
            return int(x)
        except Exception:
            return ''


if __name__ == '__main__':
    HiqbindDumper().run()
