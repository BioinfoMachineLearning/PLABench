"""PDB ATOM-record parsing for protein side."""
from plabench.models.mfe.inference import AA_3TO1, extract_sequence_from_atoms


def parse_protein_atoms(pdb_path):
    """Read ATOM records, keep standard AA atoms only.

    Returns list of (chain, res_id, res_name, x, y, z). Empty list on file error.
    """
    out = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                try:
                    rn = line[17:20].strip()
                    if rn not in AA_3TO1:
                        continue
                    ch  = line[21]
                    rid = int(line[22:26].strip())
                    x   = float(line[30:38])
                    y   = float(line[38:46])
                    z   = float(line[46:54])
                    out.append((ch, rid, rn, x, y, z))
                except Exception:
                    continue
    except Exception:
        return []
    return out


def chain_sequence(pdb_path, chain_id):
    """Extract the full ATOM-derived 1-letter sequence of one chain."""
    chains = extract_sequence_from_atoms(pdb_path)   # {chain: [3-letter,...]}
    return ''.join(AA_3TO1.get(r, 'X') for r in chains.get(chain_id, []))


def all_chain_sequences(pdb_path):
    """Return {chain_id: 1-letter sequence string} for all standard-AA chains."""
    chains = extract_sequence_from_atoms(pdb_path)
    return {ch: ''.join(AA_3TO1.get(r, 'X') for r in residues)
            for ch, residues in chains.items()}
