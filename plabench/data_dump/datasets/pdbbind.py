"""PDBbind 2020 (R1, P-L subset) dumper.

Source layout (already extracted, no tar to unpack):
    /bmlfast/Lyuwei/1.Datasets/PDBbind2020r1/
      ├── index/INDEX_general_PL.2020R1.lst    (19,037 entries with affinity + year + resolution)
      ├── P-L/
      │   ├── 1981-2000/{pdb_id}/{pdb_id}_{protein.pdb,ligand.sdf,ligand.mol2,pocket.pdb}
      │   ├── 2001-2010/...
      │   └── 2011-2019/...

Index line format:
    {pdb_id}  {resolution}  {year}  {binding_str}  // {ref}.pdf  ({ligand_name})
    e.g.: 10gs  2.10  1997  Ki=8.6uM  // 10gs.pdf (VWW)

binding_str sub-format: {Type}{op}{value}{unit}
    Type ∈ {Kd, Ki, IC50}        (PL set has no EC50 / Ka)
    op   ∈ {'=', '<', '>', '~'}  (PL set 98.6% are '=')
    unit ∈ {mM, uM, nM, pM, fM}  (always molar prefix; converted to M then -log10)

Affinity unification: stored `affinity` = -log10(value × unit_factor) = standard pX
  (matches bindingmoad/hiqbind/spindr/bindingnet 'affinity' column convention)

Per-row pre-filters (Step A, HiQBind-style strict regression cuts):
  1. Sign == '='                  drop ~270 censored / approximate
  2. Resolution ≤ 2.5 Å           drop ~3,750 lower-quality + ~300 NMR
  3. Affinity parseable           drop ~7 unparseable
  → expect ~15,092 rows surviving Step A pre-filter (before file/contact checks)

target_id / compound_id / pdb_id all set to the PDB ID per user spec — these
columns aren't load-bearing for PDBbind training; the framework just needs them
present for cross-dataset schema parity.

Splits: PDBbind 2020 ships no train/val/test split. All rows get split='none'.
Downstream pipeline can carve CASF-2016 (285 PDBs) as test and re-split train/val.

Run:
    python -m plabench.data_dump.datasets.pdbbind
"""
import os
import re
import math

from plabench.data_dump.base import BaseDumper

PDBBIND_ROOT = '/bmlfast/Lyuwei/1.Datasets/PDBbind2020r1'
INDEX_PATH   = os.path.join(PDBBIND_ROOT, 'index', 'INDEX_general_PL.2020R1.lst')
PL_ROOT      = os.path.join(PDBBIND_ROOT, 'P-L')

# Year-bucket directory layout
YEAR_BUCKETS = (
    ((1, 2000),      '1981-2000'),
    ((2001, 2010),   '2001-2010'),
    ((2011, 2019),   '2011-2019'),
)

# binding_str regex: kd/ki/ic50/ec50/ka, +/-/=/~, value, unit
BIND_RE = re.compile(
    r'^(Kd|Ki|IC50|EC50|Ka|kd|ki|ic50|ec50|ka)([=<>~])([\d.eE+-]+)([a-zA-Z^\-]+)$'
)

MEASUREMENT_TO_SOURCE = {
    'kd': 'pkd', 'ki': 'pki', 'ic50': 'pic50',
    'ec50': 'pec50',  # PL set has none, but kept for completeness
}

# Molar-prefix → conversion factor to M
UNIT_TO_M = {
    'm':  1.0,
    'mm': 1e-3,
    'um': 1e-6, 'μm': 1e-6,
    'nm': 1e-9,
    'pm': 1e-12,
    'fm': 1e-15,
}

LIG_NAME_RE = re.compile(r'\(([^)]+)\)\s*$')   # match the trailing "(VWW)" in the ref column


def _bucket_dir(year: int) -> str:
    for (lo, hi), name in YEAR_BUCKETS:
        if lo <= year <= hi:
            return name
    return ''


def _to_pX(value: float, unit: str):
    """Convert raw value+unit → -log10(M). Returns None on unknown unit / bad value."""
    factor = UNIT_TO_M.get(unit.lower())
    if factor is None or value is None or value <= 0:
        return None
    try:
        return -math.log10(value * factor)
    except Exception:
        return None


class PdbbindDumper(BaseDumper):
    DATASET_TAG    = 'pdbbind2020'
    OUT_DIR_NAME   = 'pdbbind2020'
    EXTRA_FIELDS   = [
        'resolution', 'year',
        'affinity_sign', 'affinity_value', 'affinity_unit',
        'ligand_name',
    ]
    EXPECTED_TOTAL = 19037

    def iter_candidates(self):
        with open(INDEX_PATH) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split(None, 4)        # split into [pdb, res, year, bind, rest]
                if len(parts) < 5:
                    continue
                pdb_id, res_str, year_str, bind_str, rest = parts
                pdb_id = pdb_id.lower()

                # parse resolution (string or "NMR")
                try: resolution = float(res_str)
                except Exception: resolution = None

                # parse year
                try: year = int(year_str)
                except Exception: year = None

                # parse binding_str
                m = BIND_RE.match(bind_str)
                if not m:
                    continue
                meas_raw, sign, val_str, unit = m.groups()
                meas = meas_raw.lower()
                src = MEASUREMENT_TO_SOURCE.get(meas)
                if src is None:
                    continue
                try: value = float(val_str)
                except Exception: continue
                affinity = _to_pX(value, unit)
                if affinity is None:
                    continue

                # ligand 3-letter code in trailing "(...)"
                lig_match = LIG_NAME_RE.search(rest.strip())
                ligand_name = lig_match.group(1) if lig_match else ''

                # locate files
                bucket = _bucket_dir(year) if year else ''
                dir_path = os.path.join(PL_ROOT, bucket, pdb_id) if bucket else ''
                pdb_path = os.path.join(dir_path, f'{pdb_id}_protein.pdb')
                sdf_path = os.path.join(dir_path, f'{pdb_id}_ligand.sdf')

                yield {
                    'system_id':       pdb_id,
                    'pdb_id':          pdb_id,
                    'target_id':       pdb_id,
                    'compound_id':     pdb_id,
                    'pdb_path':        pdb_path,
                    'sdf_path':        sdf_path,
                    'smiles_raw':      None,            # PDBbind index has no SMILES; framework reads from SDF mol
                    'affinity':        affinity,
                    'affinity_source': src,
                    'split':           'none',          # PDBbind ships no split

                    # extras
                    'resolution':      resolution if resolution is not None else '',
                    'year':            year if year is not None else '',
                    'affinity_sign':   sign,
                    'affinity_value':  value,
                    'affinity_unit':   unit,
                    'ligand_name':     ligand_name,
                    '_raw_sign':       sign,
                    '_raw_res':        resolution,
                }

    def pre_filter(self, candidate):
        if candidate['_raw_sign'] != '=':
            return 'drop_sign'
        res = candidate['_raw_res']
        if res is None or res > 2.5:
            return 'drop_resolution'
        return None


if __name__ == '__main__':
    PdbbindDumper().run()
