"""BaseDumper: orchestrates the full data-dump pipeline for a dataset.

Subclass contract: implement iter_candidates() to yield row-dicts from the
dataset's source. Optionally override pre_filter() for dataset-specific row
rejections (resolution, sign, etc.). Set DATASET_TAG / OUT_DIR_NAME / EXTRA_FIELDS
class attributes.

The candidate dict yielded by iter_candidates() must contain:
    pdb_id, system_id, target_id, compound_id, smiles_raw,
    affinity, affinity_source,
    pdb_path, sdf_path,
    split          ('train' | 'val' | 'test' | 'none')
    {extra fields the subclass declares in EXTRA_FIELDS}

Optional fields (passed through if present): lmdb_idx, gnina_score
"""
from __future__ import annotations
import os
import csv
import gc
import time
import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict

from plabench.data_dump.pdb_utils import parse_protein_atoms, chain_sequence
from plabench.data_dump.ligand_utils import parse_ligand_sdf, canonical_smiles
from plabench.data_dump.contact_filter import classify_contact, CONTACT_CUTOFF_A
from plabench.data_dump.dedup import dedup_keep_longest_chain

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_OUT_ROOT = os.path.join(PROJECT_ROOT, 'data')

BASE_FIELDS = [
    'lmdb_idx', 'system_id', 'dataset_source', 'target_id', 'compound_id',
    'pdb_id', 'sequence', 'chain_id_used', 'seq_len', 'contact_residues',
    'smiles', 'affinity', 'affinity_source', 'gnina_score', 'split',
]


class BaseDumper(ABC):
    DATASET_TAG: str = 'unknown'
    OUT_DIR_NAME: str | None = None     # default = DATASET_TAG
    EXTRA_FIELDS: list[str] = []
    CONTACT_CUTOFF_A: float = CONTACT_CUTOFF_A
    PROGRESS_EVERY: int = 2000
    GC_EVERY: int = 10000
    EXPECTED_TOTAL: int | None = None    # for ETA calc; subclass may set

    def __init__(self):
        out_name = self.OUT_DIR_NAME or self.DATASET_TAG
        self.out_dir = os.path.join(DATA_OUT_ROOT, out_name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.fields = BASE_FIELDS + list(self.EXTRA_FIELDS)
        self.log = logging.getLogger(f'{self.__class__.__name__}')
        if not self.log.handlers:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s [%(name)s] %(message)s')

    # ── Subclass contract ───────────────────────────────────────────────

    @abstractmethod
    def iter_candidates(self):
        """Yield candidate dicts from the dataset's metadata source.

        Each dict must populate at least the fields documented in this module's
        docstring. The base pipeline will compute sequence / chain_id_used /
        seq_len / contact_residues / canonical smiles itself, so the candidate
        only carries the source-level data plus paths to PDB and SDF.
        """
        raise NotImplementedError

    def pre_filter(self, candidate):
        """Override to apply dataset-specific filters (sign, resolution, ...).

        Return None to keep, or a string drop-reason to reject.
        """
        return None

    # ── Pipeline ────────────────────────────────────────────────────────

    def run(self):
        t0 = time.time()
        self.log.info('start: %s → %s', self.DATASET_TAG, self.out_dir)

        staged = []                       # rows passing per-row filters (pre-dedup)
        stats = Counter()
        drop_examples = defaultdict(list)

        for cand in self.iter_candidates():
            stats['source_total'] += 1

            # subclass-level pre-filters
            reason = self.pre_filter(cand)
            if reason is not None:
                stats[reason] += 1
                _record_example(drop_examples, reason, cand)
                self._maybe_progress(stats, t0)
                continue

            row = self._process_candidate(cand, stats, drop_examples)
            if row is not None:
                staged.append(row)
                stats['kept_pre_dedup'] += 1
            self._maybe_progress(stats, t0)
            if stats['source_total'] % self.GC_EVERY == 0:
                gc.collect()

        self.log.info('per-row filters complete: kept %d / %d', stats['kept_pre_dedup'], stats['source_total'])

        # Cross-row dedup (Step B)
        kept, dropped = dedup_keep_longest_chain(staged)
        stats['drop_dedup_other_chains'] = dropped
        stats['kept_final'] = len(kept)
        self.log.info('dedup complete: %d → %d (-%d)', len(staged), len(kept), dropped)

        # Write CSVs
        self._write_csvs(kept)

        # Write log
        self._write_dump_log(kept, stats, drop_examples, t0)

        self.log.info('done: %d rows in %s', len(kept), self.out_dir)

    # ── Internal helpers ────────────────────────────────────────────────

    def _process_candidate(self, cand, stats, drop_examples):
        # File existence
        pdb_path = cand['pdb_path']
        sdf_path = cand['sdf_path']
        if not (os.path.exists(pdb_path) and os.path.exists(sdf_path)):
            stats['drop_no_structure'] += 1
            _record_example(drop_examples, 'drop_no_structure', cand)
            return None

        lig_xyz, lig_mol = parse_ligand_sdf(sdf_path)
        if lig_xyz is None or len(lig_xyz) == 0 or lig_mol is None:
            stats['drop_sdf_fail'] += 1
            _record_example(drop_examples, 'drop_sdf_fail', cand)
            return None

        prot = parse_protein_atoms(pdb_path)
        if not prot:
            stats['drop_no_prot'] += 1
            _record_example(drop_examples, 'drop_no_prot', cand)
            return None

        # SMILES canonicalize
        smi = canonical_smiles(cand.get('smiles_raw'))
        if smi is None:
            try:
                from rdkit import Chem
                smi = canonical_smiles(Chem.MolToSmiles(lig_mol))
            except Exception:
                smi = None
        if smi is None:
            stats['drop_smiles_fail'] += 1
            _record_example(drop_examples, 'drop_smiles_fail', cand)
            return None

        # Contact classification
        kind, ch_used, contact_set = classify_contact(prot, lig_xyz, pdb_path, self.CONTACT_CUTOFF_A)
        if kind != 'single':
            tag = f'drop_{kind}'
            stats[tag] += 1
            _record_example(drop_examples, tag, cand)
            return None

        sequence = chain_sequence(pdb_path, ch_used)
        if not sequence:
            stats['drop_empty_seq'] += 1
            _record_example(drop_examples, 'drop_empty_seq', cand)
            return None

        row = {
            'lmdb_idx':         cand.get('lmdb_idx', ''),
            'system_id':        cand['system_id'],
            'dataset_source':   self.DATASET_TAG,
            'target_id':        cand['target_id'],
            'compound_id':      cand['compound_id'],
            'pdb_id':           cand['pdb_id'],
            'sequence':         sequence,
            'chain_id_used':    ch_used,
            'seq_len':          len(sequence),
            'contact_residues': len(contact_set),
            'smiles':           smi,
            'affinity':         cand['affinity'],
            'affinity_source':  cand['affinity_source'],
            'gnina_score':      cand.get('gnina_score', ''),
            'split':            cand.get('split', 'none'),
        }
        for f in self.EXTRA_FIELDS:
            row[f] = cand.get(f, '')
        return row

    def _maybe_progress(self, stats, t0):
        n = stats['source_total']
        if n == 0 or n % self.PROGRESS_EVERY != 0:
            return
        dt = time.time() - t0
        rate = n / max(dt, 1e-3)
        if self.EXPECTED_TOTAL:
            eta_min = max(0, (self.EXPECTED_TOTAL - n) / max(rate, 1e-3)) / 60.0
            self.log.info('  scanned %d  kept(pre-dedup) %d  (%.0fs, %.1f/s, ETA %.1fmin)',
                          n, stats['kept_pre_dedup'], dt, rate, eta_min)
        else:
            self.log.info('  scanned %d  kept(pre-dedup) %d  (%.0fs, %.1f/s)',
                          n, stats['kept_pre_dedup'], dt, rate)

    def _write_csvs(self, kept):
        # full
        full_path = os.path.join(self.out_dir, 'full.csv')
        with open(full_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            for r in kept:
                w.writerow({k: r.get(k, '') for k in self.fields})
        # splits
        for s in ('train', 'val', 'test'):
            sub = [r for r in kept if r.get('split') == s]
            with open(os.path.join(self.out_dir, f'{s}.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.fields)
                w.writeheader()
                for r in sub:
                    w.writerow({k: r.get(k, '') for k in self.fields})

    def _write_dump_log(self, kept, stats, drop_examples, t0):
        from collections import Counter
        split_counts = Counter(r.get('split', 'none') for r in kept)
        src_counts = Counter(r.get('affinity_source') for r in kept)
        target_counts = Counter(r.get('target_id') for r in kept)

        with open(os.path.join(self.out_dir, 'dump_log.txt'), 'w') as f:
            f.write(f'{self.DATASET_TAG} dump log\n')
            f.write('=' * 60 + '\n')
            f.write(f'duration: {time.time() - t0:.1f}s\n')
            f.write(f'cutoff = {self.CONTACT_CUTOFF_A} A, single-chain-contact only\n')
            f.write(f'dedup = keep all entries on longest chain per (pdb_id, smiles) group\n\n')
            f.write('Counts:\n')
            for k, v in stats.most_common():
                f.write(f'  {k:30s} {v:8d}\n')
            f.write('\nKept rows by split:\n')
            for s in ('train', 'val', 'test', 'none'):
                f.write(f'  {s:10s} {split_counts.get(s, 0):8d}\n')
            f.write('\nKept rows by affinity_source:\n')
            for k, v in src_counts.most_common():
                f.write(f'  {k:15s} {v:8d}\n')
            if target_counts:
                import numpy as np
                sa = np.array(list(target_counts.values()))
                f.write(f'\nUnique target_id (kept rows): {len(target_counts)}\n')
                f.write(f'  compounds-per-target: mean={sa.mean():.1f}  '
                        f'median={int(np.median(sa))}  min={sa.min()}  max={sa.max()}\n')
            f.write('\nDrop reason → first 5 system_ids:\n')
            for reason, lst in drop_examples.items():
                f.write(f'  {reason} ({stats.get(reason, len(lst))}):\n')
                for sid in lst[:5]:
                    f.write(f'    {sid}\n')


def _record_example(drop_examples, reason, cand, cap=10):
    if len(drop_examples[reason]) < cap:
        drop_examples[reason].append(cand.get('system_id', '?'))
