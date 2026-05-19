"""4.5 A single-chain ligand-protein contact filter.

A chain "contacts" a ligand iff some standard-AA residue on that chain has any
heavy atom within `cutoff` of any ligand heavy atom. Entries where the ligand
contacts >=2 chains are dropped (multi_homo if those chains share sequence,
multi_hetero otherwise).
"""
from collections import defaultdict
import numpy as np

from plabench.data_dump.pdb_utils import all_chain_sequences

CONTACT_CUTOFF_A = 4.5


def per_chain_contact_residues(prot_atoms, lig_xyz, cutoff=CONTACT_CUTOFF_A):
    """Compute {chain_id: set of contacting res_ids} for the given ligand coords.

    prot_atoms: list of (chain, res_id, res_name, x, y, z) — from pdb_utils.parse_protein_atoms
    lig_xyz:    (N_lig_heavy, 3) float ndarray
    """
    if not prot_atoms or lig_xyz is None or len(lig_xyz) == 0:
        return defaultdict(set)
    pxyz = np.array([(x, y, z) for _, _, _, x, y, z in prot_atoms], dtype=np.float64)
    d2 = ((pxyz[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(-1)
    min_d = np.sqrt(d2.min(axis=1))
    contact = defaultdict(set)
    for i, (ch, rid, _, _, _, _) in enumerate(prot_atoms):
        if min_d[i] <= cutoff:
            contact[ch].add(rid)
    return contact


def classify_contact(prot_atoms, lig_xyz, pdb_path, cutoff=CONTACT_CUTOFF_A):
    """Return ('single', chain_id, contact_residue_set) if exactly one chain
    is in contact, else one of:

        ('no_contact', None, None)
        ('multi_homo',   None, None)   — multiple contact chains, identical sequences
        ('multi_hetero', None, None)   — multiple contact chains, distinct sequences
    """
    contact = per_chain_contact_residues(prot_atoms, lig_xyz, cutoff)
    chains = [c for c, s in contact.items() if len(s) >= 1]

    if len(chains) == 0:
        return 'no_contact', None, None
    if len(chains) == 1:
        c = chains[0]
        return 'single', c, contact[c]

    seqs = all_chain_sequences(pdb_path)
    seq_set = {seqs.get(c, '') for c in chains}
    if len(seq_set) == 1:
        return 'multi_homo', None, None
    return 'multi_hetero', None, None
