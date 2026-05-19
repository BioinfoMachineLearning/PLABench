"""bindingnet_high_flowr_final dumper (framework version with dedup).

Source layout:
    LMDB:     /bmlfast/Lyuwei/1.Datasets/bindingnet_high_flowr_final/
    Prepared: /bml/Lyuwei_data/bindingnet_high_flowr_prepared/   ← /bml mirror, faster than /bmlfast

LMDB metadata fields:
    system_id, is_covalent, pic50, pki, logaff, strain, dG,
    vina_score, gnina_score, apo_type, split, metrics
    (vina_score / apo_type all NaN; dG / strain are docking artifacts, not used)

system_id format: '{CHEMBL_target_id}_{CHEMBL_compound_id}'
    e.g., 'CHEMBL3174_CHEMBL4435063' → target=CHEMBL3174, compound=CHEMBL4435063

Affinity:
    `logaff` is universal (-log10 affinity). When pic50 / pki present, logaff matches them
    to 0.01 precision. ~7% of entries have only logaff (no pic50/pki) — origin not annotated
    by FlowR but the value is still pX-scale; we keep them with affinity_source='logaff_only'.

target_id: ChEMBL target ID (no PDB ID for these docked structures).
pdb_id:    empty (BindingNet structures are cross-docked, no native PDB).

Run:
    python -m plabench.data_dump.datasets.bindingnet
"""
import os
import lmdb
import pickle
import numpy as np

from plabench.data_dump.base import BaseDumper

LMDB_DIR = '/bmlfast/Lyuwei/1.Datasets/bindingnet_high_flowr_final'
PREP_DIR = '/bml/Lyuwei_data/bindingnet_high_flowr_prepared'


def _has_value(x):
    if x is None:
        return False
    try:
        if hasattr(x, 'shape') and x.shape == ():
            x = float(x)
        return not (isinstance(x, float) and np.isnan(x))
    except Exception:
        return False


class BindingnetDumper(BaseDumper):
    DATASET_TAG    = 'bindingnet_high'
    OUT_DIR_NAME   = 'bindingnet_high'
    EXTRA_FIELDS   = []
    EXPECTED_TOTAL = 232035

    def iter_candidates(self):
        splits = np.load(os.path.join(LMDB_DIR, 'splits.npz'))
        idx_to_split = {}
        for s in ('train', 'val', 'test'):
            for i in splits[f'idx_{s}'].tolist():
                idx_to_split[i] = s

        env = lmdb.open(LMDB_DIR, readonly=True, lock=False, max_readers=1, subdir=True)
        try:
            with env.begin() as txn:
                for k, v in txn.cursor():
                    try:
                        idx = int(k.decode())
                        obj = pickle.loads(v)
                        md = obj['metadata']
                    except Exception:
                        continue

                    la = md.get('logaff')
                    if not _has_value(la):
                        continue
                    try:
                        affinity = float(la)
                    except Exception:
                        continue

                    ic = md.get('pic50'); ki = md.get('pki')
                    if _has_value(ic):
                        src = 'pic50'
                    elif _has_value(ki):
                        src = 'pki'
                    else:
                        src = 'logaff_only'

                    sid = md.get('system_id') or ''
                    if '_' not in sid:
                        continue
                    target_id, _, compound_id = sid.partition('_')

                    raw_smi = None
                    try:
                        lig_obj = pickle.loads(obj['ligand'])
                        raw_smi = lig_obj.get('id')
                    except Exception:
                        pass

                    gnina = md.get('gnina_score')
                    if isinstance(gnina, float) and np.isnan(gnina):
                        gnina = None

                    yield {
                        'lmdb_idx':        idx,
                        'system_id':       sid,
                        'pdb_id':          '',                # cross-docked, no native PDB ID
                        'target_id':       target_id,
                        'compound_id':     compound_id,
                        'pdb_path':        os.path.join(PREP_DIR, sid + '.pdb'),
                        'sdf_path':        os.path.join(PREP_DIR, sid + '.sdf'),
                        'smiles_raw':      raw_smi,
                        'affinity':        affinity,
                        'affinity_source': src,
                        'gnina_score':     gnina if gnina is not None else '',
                        'split':           idx_to_split.get(idx, 'none'),
                    }
        finally:
            env.close()


if __name__ == '__main__':
    BindingnetDumper().run()
