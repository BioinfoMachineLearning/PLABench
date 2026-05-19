"""Cross-row dedup: keep all entries on the longest chain per (pdb_id, smiles).

Single-chain-contact filter (in contact_filter.py) decides which protein chain
a ligand binds. Even after that filter, the same (pdb_id, smiles) pair can
appear in multiple rows because:

  case 3: same ligand bound to multiple chains of a homo-/hetero-multimer
          (each chain produces its own entry)
  case 5: same ligand bound at multiple sites of the SAME chain
          (different ligand residue numbers → different entries, same chain)

For sequence-only models the input is (sequence, smiles); rows with the same
(pdb_id, smiles) on different chains are redundant copies of the same binding
event from the model's perspective. We dedup case 3 by picking the chain with
the longest extracted sequence (so the most complete protein context survives).
We deliberately keep case 5 — multiple ligand copies on the same selected chain
are retained, treating frequency as a signal that this binding mode is common
for the target.
"""
from collections import defaultdict


def dedup_keep_longest_chain(rows):
    """Apply the dedup rule.

    Each row must be a dict with at least: 'pdb_id', 'smiles', 'chain_id_used',
    'seq_len' (int).

    Returns (kept_rows, drop_count).

    Rule:
      group rows by (pdb_id, smiles).
      if group has only one chain_id_used → keep all entries (no inter-chain dedup needed).
      else pick the chain whose seq_len is largest (tiebreak: chain_id_used alphabetic),
           keep all entries on that chain, drop entries on the other chains.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r['pdb_id'], r['smiles'])].append(r)

    kept = []
    dropped = 0
    for (pdb, smi), grp in groups.items():
        chains_in_group = {r['chain_id_used'] for r in grp}
        if len(chains_in_group) <= 1:
            kept.extend(grp)
            continue
        # multi-chain group → pick longest chain
        chain_to_seqlen = {}
        for r in grp:
            chain_to_seqlen.setdefault(r['chain_id_used'], r['seq_len'])
        # tiebreak: longest seq_len, then alphabetic chain id
        best_chain = max(chain_to_seqlen.keys(),
                         key=lambda c: (chain_to_seqlen[c], -ord(c[0]) if c else 0))
        # the lambda above uses negative ord for inverse-alphabetic in case of seq_len tie;
        # but max() of tuple compares left-then-right, so we need: bigger seqlen wins,
        # if equal pick smaller chain alphabetically. Re-derive cleanly:
        best_chain = sorted(
            chain_to_seqlen.keys(),
            key=lambda c: (-chain_to_seqlen[c], c)
        )[0]
        kept_in_group = [r for r in grp if r['chain_id_used'] == best_chain]
        kept.extend(kept_in_group)
        dropped += len(grp) - len(kept_in_group)
    return kept, dropped
