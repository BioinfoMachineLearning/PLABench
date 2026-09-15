#!/usr/bin/env python
"""Date the CASP16 ligands that have an exact twin in ChEMBL 34.

The audit in casp16_ligand_leakage.py says which CASP16 ligands are already in the
training corpora. This asks the follow-up question the paper actually needs answered:
when were they published, and against which target were they measured. A ligand that
appears in ChEMBL 34 with a 2019 autotaxin IC50 is not a blind CASP16 compound.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
DB = "/bmlfast/Lyuwei/1.Datasets/chembl_34/chembl_34_sqlite/chembl_34.db"

_norm = rdMolStandardize.Normalizer()
_frag = rdMolStandardize.LargestFragmentChooser()
_unc = rdMolStandardize.Uncharger()


def skeleton(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = _unc.uncharge(_frag.choose(_norm.normalize(m)))
    return Chem.MolToInchiKey(m).split("-")[0]


QUERY = """
SELECT substr(cs.standard_inchi_key, 1, 14) AS blk, md.chembl_id AS mol,
       cseq.accession, act.standard_type, act.standard_value, act.standard_units, d.year
FROM compound_structures cs
JOIN molecule_dictionary md ON md.molregno = cs.molregno
JOIN activities act ON act.molregno = md.molregno
JOIN assays a ON a.assay_id = act.assay_id
JOIN target_dictionary td ON td.tid = a.tid
LEFT JOIN target_components tc ON tc.tid = td.tid
LEFT JOIN component_sequences cseq ON cseq.component_id = tc.component_id
LEFT JOIN docs d ON d.doc_id = act.doc_id
WHERE substr(cs.standard_inchi_key, 1, 14) IN (%s)
"""


def main() -> None:
    d = pd.read_csv("results/casp16_leakage/per_corpus.csv")
    hits = d[d.sim_chembl34_full >= 0.9999]
    keys: dict[str, str] = {}
    for _, r in hits.iterrows():
        k = skeleton(r.smiles)
        if k:
            keys[k] = f"{r.series}/{r.ligand_id}"
    print(f"CASP16 ligands with an exact ChEMBL 34 twin: {len(hits)} "
          f"({', '.join(sorted(set(hits.series)))})")

    con = sqlite3.connect(DB)
    r = pd.read_sql(QUERY % ",".join("?" * len(keys)), con, params=list(keys))
    con.close()
    r["casp"] = r.blk.map(keys)

    print(f"\nChEMBL 34 activity rows for those ligands: {len(r):,}")
    g = r.groupby(["casp", "accession"]).agg(
        n=("year", "size"), first_year=("year", "min"), last_year=("year", "max"))
    print(g.sort_values("n", ascending=False).head(30).to_string())

    print("\npublication year of the earliest ChEMBL 34 record per CASP16 ligand:")
    print(r.groupby("casp").year.min().to_string())

    for acc, label in (("Q13822", "autotaxin/ENPP2"), ("P23946", "chymase/CMA1")):
        s = r[r.accession == acc]
        if len(s):
            print(f"\n{label} ({acc}): {len(s)} rows, "
                  f"{s.casp.nunique()} CASP16 ligands, years {s.year.min()}-{s.year.max()}")
            print(s.groupby("casp").agg(n=("year", "size"), yr=("year", "min")).to_string())
        else:
            print(f"\n{label} ({acc}): no rows")


if __name__ == "__main__":
    main()
