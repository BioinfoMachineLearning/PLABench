"""SPINDR dumper.

SPINDR is the FLOWR project's flagship LigPrep + PrepWizard prepared dataset
(arXiv 2504.10564, CC BY 4.0). FLOWR's primary task is ligand generation, so
only ~31% of entries (11,132 / 35,627) carry experimental affinity. We dump
those for sequence-only affinity training.

Source layout:
    /bml/Lyuwei_data/SPINDR/
      ├── data/
      │   ├── {train,val,test}/{system_id}.{pdb,sdf}    ← we read these
      │   └── *.{cif,sdf}                               ← root-level mmCIF set, not used
      └── final_from_smol.tar.gz                         ← LMDB pack (256 MB → 4.5 GB)

LMDB extraction (one-time, before running this dumper):
    mkdir -p /tmp/spindr_lmdb
    tar -xzf /bml/Lyuwei_data/SPINDR/final_from_smol.tar.gz -C /tmp/spindr_lmdb \
        final_from_smol/data.mdb final_from_smol/lock.mdb final_from_smol/splits.npz

system_id format: '{pdb_id}__{biounit}__{model.chain}__{lig_model.lig_chain}'
    e.g., '1a3w__1__1.A__1.D' → pdb_id=1a3w, biounit=1,
                                 protein_chain at model 1.A,
                                 ligand at model 1, chain D

Per-row filters (Step A):
    1. exactly one of pic50/pki/pkd/pec50 must be non-NaN (mutually exclusive)
    2. data/{split}/{system_id}.{pdb,sdf} must exist in the split dir matching metadata.split
    + standard framework filters (single-chain contact, sequence non-empty, ...)

target_id: pdb_id (no UniProt info in SPINDR metadata).
"""
import os
import lmdb
import pickle
import numpy as np

from plabench.data_dump.base import BaseDumper

LMDB_DIR        = '/tmp/spindr_lmdb/final_from_smol'
SPINDR_DATA_ROOT = '/bml/Lyuwei_data/SPINDR/data'

AFFINITY_PRIORITY = ('pkd', 'pki', 'pic50', 'pec50')   # priority order; mutually exclusive in practice


def _has_value(x):
    if x is None:
        return False
    try:
        if hasattr(x, 'shape') and x.shape == ():
            x = float(x)
        return not (isinstance(x, float) and np.isnan(x))
    except Exception:
        return False


class SpindrDumper(BaseDumper):
    DATASET_TAG    = 'spindr'
    OUT_DIR_NAME   = 'spindr'
    EXTRA_FIELDS   = []                  # SPINDR has no rich metadata to expose
    EXPECTED_TOTAL = 35632

    def iter_candidates(self):
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

                    # affinity (mutually exclusive in SPINDR)
                    affinity = None
                    src = None
                    for key in AFFINITY_PRIORITY:
                        x = md.get(key)
                        if _has_value(x):
                            try:
                                affinity = float(x)
                                src = 'p' + key[1:] if key.startswith('p') else key
                                # already named pic50/pki/pkd/pec50 — keep as-is
                                src = key
                            except Exception:
                                continue
                            break
                    if affinity is None:
                        continue   # no affinity — skip silently

                    sid = md.get('system_id')
                    if not sid or '__' not in sid:
                        continue
                    pdb_id = sid.split('__', 1)[0]

                    # split info comes from metadata field directly
                    split = md.get('split') or 'none'
                    pdb_path = os.path.join(SPINDR_DATA_ROOT, split, sid + '.pdb')
                    sdf_path = os.path.join(SPINDR_DATA_ROOT, split, sid + '.sdf')

                    yield {
                        'lmdb_idx':        idx,
                        'system_id':       sid,
                        'pdb_id':          pdb_id,
                        'target_id':       pdb_id,
                        'compound_id':     sid.split('__', 1)[1],   # e.g., '1__1.A__1.D'
                        'pdb_path':        pdb_path,
                        'sdf_path':        sdf_path,
                        'smiles_raw':      None,                    # no SMILES in metadata; framework falls back to mol from SDF
                        'affinity':        affinity,
                        'affinity_source': src,
                        'split':           split,
                    }
        finally:
            env.close()


if __name__ == '__main__':
    SpindrDumper().run()
