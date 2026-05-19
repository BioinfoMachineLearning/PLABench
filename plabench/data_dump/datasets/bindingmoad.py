"""bindingmoad_flowr dumper (framework version with dedup).

Source layout (on /bmlfast NFS, no /bml mirror — bindingmoad prepared dir is small enough):
    LMDB:     /bmlfast/Lyuwei/1.Datasets/bindingmoad_flowr/
    Prepared: /bmlfast/Lyuwei/1.Datasets/bindingmoad_data_prepared/
                  {system_id}.pdb  (full bioassembly)
                  {system_id}.sdf  (single ligand)

LMDB metadata fields:
    system_id, is_covalent, pic50, pec50, pki, pkd, vina_score, gnina_score,
    apo_type, split, metrics
    (pec50 / vina_score / apo_type all NaN; logaff field not present)

system_id format: '{pdb_id}-bio{N}_{lig_resn}:{chain}:{resi}'
    e.g., '6HOQ-bio1_FER:A:404' → pdb_id=6HOQ, biounit=1, ligand FER on chain A residue 404

Affinity:
    Each entry has at most one of pkd/pki/pic50 non-NaN (mutually exclusive,
    verified empirically: 4,545 + 4,406 + 5,488 = 14,439 = union).
    affinity = the one non-NaN pX value (already standard -log10).
    affinity_source = 'pkd' / 'pki' / 'pic50'.

target_id: pdb_id (no UniProt info in LMDB).

Run:
    python -m plabench.data_dump.datasets.bindingmoad
"""
import os
import lmdb
import pickle
import numpy as np

from plabench.data_dump.base import BaseDumper

LMDB_DIR = '/bmlfast/Lyuwei/1.Datasets/bindingmoad_flowr'
PREP_DIR = '/bmlfast/Lyuwei/1.Datasets/bindingmoad_data_prepared'

AFFINITY_PRIORITY = ('pkd', 'pki', 'pic50')   # pec50 always NaN in this dataset


def _has_value(x):
    if x is None:
        return False
    try:
        if hasattr(x, 'shape') and x.shape == ():
            x = float(x)
        return not (isinstance(x, float) and np.isnan(x))
    except Exception:
        return False


class BindingmoadDumper(BaseDumper):
    DATASET_TAG    = 'bindingmoad'
    OUT_DIR_NAME   = 'bindingmoad_flowr'
    EXTRA_FIELDS   = []
    EXPECTED_TOTAL = 33291

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

                    affinity = None
                    src = None
                    for key in AFFINITY_PRIORITY:
                        x = md.get(key)
                        if _has_value(x):
                            try:
                                affinity = float(x)
                                src = key
                            except Exception:
                                continue
                            break
                    if affinity is None:
                        continue

                    sid = md.get('system_id') or ''
                    if '_' not in sid:
                        continue
                    pdb_part, lig_part = sid.split('_', 1)
                    pdb_id = pdb_part.split('-')[0]

                    # SMILES from ligand['id'] (explicit-H SMILES from FlowR)
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
                        'pdb_id':          pdb_id,
                        'target_id':       pdb_id,           # no UniProt — fallback to PDB ID
                        'compound_id':     lig_part,         # 'FER:A:404'
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
    BindingmoadDumper().run()
