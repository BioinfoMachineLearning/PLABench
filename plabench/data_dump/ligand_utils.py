"""SDF/SMILES utilities for ligand side."""
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Shared, stateless — safe to reuse across canonical_smiles() calls.
_UNCHARGER = rdMolStandardize.Uncharger()


def parse_ligand_sdf(sdf_path):
    """Return (heavy-atom xyz Nx3 ndarray, RDKit Mol) for the first valid molecule.

    Returns (None, None) on parse failure or empty SDF.
    """
    try:
        sup = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=True)
    except Exception:
        return None, None
    for m in sup:
        if m is None:
            continue
        conf = m.GetConformer()
        coords = []
        for i in range(m.GetNumAtoms()):
            if m.GetAtomWithIdx(i).GetSymbol() == 'H':
                continue
            p = conf.GetAtomPosition(i)
            coords.append((p.x, p.y, p.z))
        return np.asarray(coords, dtype=np.float64), m
    return None, None


def canonical_smiles(raw_smi):
    """Strict canonical isomeric SMILES via RDKit, with v3 standardization.

    Standardization pipeline (matches the test-set input style — chembl35 / CASF /
    CASP16 are all neutral canonical with no explicit H):
      1. neutralize ionized groups via RDKit Uncharger ([O-]→OH, [NH3+]→NH2 etc.)
      2. RemoveHs — drop any explicit [H] atoms
      3. AssignStereochemistry(cleanIt=True, force=True) — strip *fake* stereo
         markers (e.g. P with two equivalent OH neighbours) but keep real ones
      4. canonical isomeric SMILES, longest fragment if multi-component
      5. canonical roundtrip — re-parse and re-canonicalize so the implicit-valence
         field gets recomputed cleanly (LLF's atom_features uses GetImplicitValence,
         which RDKit sets to 0 for atoms with explicit Hs in the original parse)

    Returns None on:
      - empty input
      - parseable but cannot be sanitized (RDKit Chem.SanitizeMol failure)
      - any standardization step raising
      - MolToSmiles failure
    """
    if not raw_smi:
        return None
    m = Chem.MolFromSmiles(raw_smi, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    try:
        m = _UNCHARGER.uncharge(m)
        m = Chem.RemoveHs(m)
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    except Exception:
        return None
    try:
        smi = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None
    if '.' in smi:
        smi = max(smi.split('.'), key=len)
    try:
        m2 = Chem.MolFromSmiles(smi)
        if m2 is not None:
            smi = Chem.MolToSmiles(m2, canonical=True, isomericSmiles=True)
    except Exception:
        pass
    return smi
