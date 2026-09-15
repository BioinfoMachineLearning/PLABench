#!/usr/bin/env python
"""Triple-level leakage audit for CASP16: (protein, ligand, affinity) must all match.

A ligand that also appears in a training corpus against a different protein is not
leakage, it is chemistry. What counts is a corpus row where the protein is the CASP16
target, the molecule is the CASP16 molecule, and an affinity label is attached. This
script counts exactly those, per corpus, and then maps corpora onto the models that
trained on them (mapping taken from Table 1 of the paper).

Two ligand-matching strictnesses are reported:
  strict - identical full InChIKey, so stereochemistry must agree
  loose  - identical InChIKey block 1, i.e. same skeleton, stereo-blind

Writes results/casp16_leakage/triples.csv and per_model.csv.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plabench.analysis.units import dg_to_pkd  # noqa: E402
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
OUT = "results/casp16_leakage"
KMER = 10
TARGET = {"L1000": ("P23946", "chymase"), "L3000": ("Q13822", "autotaxin")}
BOLTZ2_CUTOFF = "2023-06-01"   # the cutoff the paper claims for every model

_norm = rdMolStandardize.Normalizer()
_frag = rdMolStandardize.LargestFragmentChooser()
_unc = rdMolStandardize.Uncharger()


def keys(smiles: str) -> tuple[str, str] | None:
    """(full InChIKey, skeleton block) after desalt+neutralise."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = _unc.uncharge(_frag.choose(_norm.normalize(m)))
    ik = Chem.MolToInchiKey(m)
    return ik, ik.split("-")[0]


def load_queries() -> pd.DataFrame:
    rows = []
    for ser, (path, col) in {
        "L1000": ("data/casp16_data/labels/L1000_exper_affinity.csv", "ligand_smiles"),
        "L3000": ("data/casp16_data/labels/L3000_exper_affinity.csv", "Structure"),
    }.items():
        d = pd.read_csv(path, encoding="utf-8-sig").rename(
            columns={"Target ID": "ligand_id", col: "smiles", "binding_affinity": "dG"})
        d["series"] = ser
        d["pKd"] = dg_to_pkd(d.dG)
        rows.append(d[["series", "ligand_id", "smiles", "pKd"]])
    q = pd.concat(rows, ignore_index=True)
    k = [keys(s) for s in q.smiles]
    q["ik"] = [x[0] if x else None for x in k]
    q["blk"] = [x[1] if x else None for x in k]
    return q


def kmers(s: str) -> set[str]:
    return {s[i:i + KMER] for i in range(len(s) - KMER + 1)} if len(s) >= KMER else set()


def target_kmers() -> dict[str, set[str]]:
    out = {}
    for ser in TARGET:
        L = open(f"data/casp16_data/stage1_input/{ser}.txt").read().split("\n")
        out[ser] = kmers("".join(l.strip() for l in L[1:]
                                 if l.strip() and not l.startswith(">")))
    return out


def seq_series(seq: str, tk: dict[str, set[str]]) -> str | None:
    """Which CASP16 target, if any, this corpus sequence is."""
    ks = kmers(str(seq))
    if not ks:
        return None
    for ser, q in tk.items():
        if len(ks & q) / len(ks) > 0.5:
            return ser
    return None


# ---------------------------------------------------------------- corpora readers
def from_structure_csv(path: str, q: pd.DataFrame, tk) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    sc = "sequence" if "sequence" in d else "target_sequence"
    lc = "smiles" if "smiles" in d else "compound_iso_smiles"
    if "uniprot_id" in d:
        acc = {v[0]: k for k, v in TARGET.items()}
        d["series"] = d.uniprot_id.map(acc)
        miss = d.series.isna()
        d.loc[miss, "series"] = [seq_series(s, tk) for s in d.loc[miss, sc]]
    else:
        uniq = {s: seq_series(s, tk) for s in d[sc].astype(str).unique()}
        d["series"] = d[sc].astype(str).map(uniq)
    d = d[d.series.notna()].copy()
    if d.empty:
        return d.assign(ik=None, blk=None, label=np.nan)[["series", "ik", "blk", "label"]]
    k = [keys(s) for s in d[lc].astype(str)]
    d["ik"] = [x[0] if x else None for x in k]
    d["blk"] = [x[1] if x else None for x in k]
    d["label"] = pd.to_numeric(d["affinity"], errors="coerce")
    return d[["series", "ik", "blk", "label"]]


def from_bindingnet(q, tk) -> pd.DataFrame:
    d = pd.read_csv("/bml/Lyuwei_data/Index_for_BindingNetv1_and_BindingNetv2.csv",
                    low_memory=False)
    acc = {v[0]: k for k, v in TARGET.items()}
    d["series"] = d.UniProt.map(acc)
    d = d[d.series.notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["series", "ik", "blk", "label"])
    k = [keys(s) for s in d["Molecule SMILES"].astype(str)]
    d["ik"] = [x[0] if x else None for x in k]
    d["blk"] = [x[1] if x else None for x in k]
    d["label"] = pd.to_numeric(d["-logAffi"], errors="coerce")
    return d[["series", "ik", "blk", "label"]]


def from_sair(q, tk) -> pd.DataFrame:
    s = pd.read_parquet("data/SAIR/filtered_data.parquet",
                        columns=["protein", "SMILES", "pIC50"])
    acc = {v[0]: k for k, v in TARGET.items()}
    s["series"] = s.protein.map(acc)
    s = s[s.series.notna()].copy()
    k = [keys(x) for x in s.SMILES.astype(str)]
    s["ik"] = [x[0] if x else None for x in k]
    s["blk"] = [x[1] if x else None for x in k]
    return s.rename(columns={"pIC50": "label"})[["series", "ik", "blk", "label"]]


def from_chembl34(q, tk) -> pd.DataFrame:
    con = sqlite3.connect("/bmlfast/Lyuwei/1.Datasets/chembl_34/chembl_34_sqlite/"
                          "chembl_34.db")
    sql = """
    SELECT cs.standard_inchi_key AS ik, cseq.accession AS acc,
           act.pchembl_value AS label
    FROM activities act
    JOIN assays a ON a.assay_id = act.assay_id
    JOIN target_dictionary td ON td.tid = a.tid
    JOIN target_components tc ON tc.tid = td.tid
    JOIN component_sequences cseq ON cseq.component_id = tc.component_id
    JOIN compound_structures cs ON cs.molregno = act.molregno
    WHERE cseq.accession IN ('P23946','Q13822')"""
    d = pd.read_sql(sql, con)
    con.close()
    acc = {v[0]: k for k, v in TARGET.items()}
    d["series"] = d.acc.map(acc)
    d["blk"] = d.ik.str[:14]
    d["label"] = pd.to_numeric(d.label, errors="coerce")
    return d[["series", "ik", "blk", "label"]]


def from_bindingdb(q, tk) -> pd.DataFrame:
    """Rows pre-scanned from BindingDB_All.tsv, restricted to the paper's June 2023 cutoff."""
    d = pd.read_csv("/tmp/casp16_bdb_triple.tsv", sep="\t", header=None,
                    names=["lig", "ik", "acc", "date", "ki", "ic50", "kd", "ec50"], dtype=str)
    acc = {v[0]: k for k, v in TARGET.items()}
    d["series"] = d.acc.map(acc)
    d = d[d.series.notna()].copy()
    d = d[pd.to_datetime(d.date, errors="coerce") < BOLTZ2_CUTOFF]

    def num(x):
        try:
            return float(str(x).lstrip("><~ ").strip())
        except Exception:
            return np.nan
    v = d.ki.map(num)
    for c in ("kd", "ic50", "ec50"):
        v = v.fillna(d[c].map(num))
    d["label"] = 9 - np.log10(v)
    d["blk"] = d.ik.str[:14]
    return d[["series", "ik", "blk", "label"]]


CORPORA = {
    "chembl34": from_chembl34,
    "bindingdb": from_bindingdb,
    "bindingnet": from_bindingnet,
    "sair": from_sair,
    "hiqbind": lambda q, t: from_structure_csv("data/hiqbind/full.csv", q, t),
    "spindr": lambda q, t: from_structure_csv("data/spindr/full.csv", q, t),
    "bindingmoad": lambda q, t: from_structure_csv("data/bindingmoad_flowr/full.csv", q, t),
    "pdbbind2020_general": lambda q, t: from_structure_csv("data/pdbbind2020/full.csv", q, t),
    "pdbbind2020_refined": lambda q, t: from_structure_csv(
        "/bmlfast/Lyuwei/1.Datasets/pdbbind_2020_standardized_training.csv", q, t),
}
# ligand-only corpora: the ECFP4 sweep already found zero CASP16 hits in them at any
# threshold, so their triple count is zero by construction
LIGAND_ONLY = ["plinder", "kinodata", "kiba3d", "davis3d", "omol25"]

# training corpora per model, read off Table 1 of the paper
MODEL_CORPORA = {
    "Boltz2": ["chembl34", "bindingdb"],                      # + PubChem, not auditable
    "FLOWR.ROOT": ["sair", "spindr", "plinder", "bindingmoad", "hiqbind", "bindingnet",
                   "kiba3d", "davis3d", "kinodata"],
    "FlowDock": ["pdbbind2020_general", "bindingmoad"],
    "Graph_RG": ["pdbbind2020_general"],
    "MFE": ["pdbbind2020_general"],                           # v2016 is a subset of v2020
    "LCDD-team": [],                                          # training set undisclosed
    "DeepDTA": ["pdbbind2020_refined"],                       # retrained by us
    "LLF": ["pdbbind2020_refined"],
    "MixingDTA": ["pdbbind2020_refined"],
}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    q = load_queries()
    tk = target_kmers()
    print(f"CASP16 ligands: {len(q)} "
          f"({', '.join(f'{s} {int((q.series == s).sum())}' for s in TARGET)})\n")

    recs, hit = [], {}   # hit[corpus][series] = {"strict": set, "loose": set, "lab": set}
    detail = []
    for name, fn in CORPORA.items():
        c = fn(q, tk)
        hit[name] = {}
        for ser in TARGET:
            cs = c[c.series == ser]
            qs = q[q.series == ser]
            ik2lig = {k: v for k, v in zip(qs.ik, qs.ligand_id) if k}
            blk2lig = {k: v for k, v in zip(qs.blk, qs.ligand_id) if k}
            strict = cs[cs.ik.isin(ik2lig)]
            loose = cs[cs.blk.isin(blk2lig)]
            lab = strict[strict.label.notna()]
            hit[name][ser] = {
                "strict": {ik2lig[k] for k in strict.ik.unique()},
                "loose": {blk2lig[k] for k in loose.blk.unique()},
                "lab": {ik2lig[k] for k in lab.ik.unique()},
                "tgt": set(cs.ik.dropna().unique()),   # every ligand the corpus has on this target
            }
            recs.append({"corpus": name, "series": ser,
                         "corpus_rows_on_target": len(cs),
                         "corpus_ligands_on_target": cs.ik.nunique(),
                         "triple_strict": len(hit[name][ser]["strict"]),
                         "triple_loose": len(hit[name][ser]["loose"]),
                         "triple_with_label": len(hit[name][ser]["lab"])})
            for ik, g in lab.groupby("ik"):
                qq = qs[qs.ik == ik].iloc[0]
                detail.append({"corpus": name, "series": ser, "ligand_id": qq.ligand_id,
                               "casp16_pKd": qq.pKd,
                               "corpus_label": float(g.label.median()),
                               "n_rows": len(g)})
    t = pd.DataFrame(recs)
    for name in LIGAND_ONLY:
        hit[name] = {s: {"strict": set(), "loose": set(), "lab": set(), "tgt": set()}
                     for s in TARGET}
        for ser in TARGET:
            t.loc[len(t)] = {"corpus": name, "series": ser, "corpus_rows_on_target": 0,
                             "corpus_ligands_on_target": 0, "triple_strict": 0,
                             "triple_loose": 0, "triple_with_label": 0}
    t.to_csv(f"{OUT}/triples.csv", index=False)
    dd = pd.DataFrame(detail)
    dd.to_csv(f"{OUT}/triple_detail.csv", index=False)
    print("per corpus: CASP16 (protein, ligand) pairs the corpus actually contains")
    print(t.to_string(index=False))

    print("\nthe leaked triples, corpus label against the CASP16 label")
    print(dd.sort_values(["series", "ligand_id", "corpus"]).to_string(index=False))
    for ser, g in dd.groupby("series"):
        u = g.groupby("ligand_id").agg(c=("casp16_pKd", "first"), l=("corpus_label", "median"))
        print(f"  {ser}: {len(u)} ligands with a label; "
              f"mean |corpus - CASP16| = {np.abs(u.c - u.l).mean():.2f} log units, "
              f"{int((np.abs(u.c - u.l) <= 0.5).sum())}/{len(u)} agree within 0.5")

    rows = []
    for model, cs in MODEL_CORPORA.items():
        rec = {"model": model}
        for ser in TARGET:
            for kind in ("strict", "loose", "lab", "tgt"):
                u: set[str] = set()
                for c in cs:
                    u |= hit.get(c, {}).get(ser, {}).get(kind, set())
                rec[f"{ser}_{kind}"] = len(u)
        rows.append(rec)
    pm = pd.DataFrame(rows)
    pm.to_csv(f"{OUT}/per_model.csv", index=False)
    print("\nper model, union over that model's corpora. *_lab = leaked and carrying an "
          "affinity label; *_tgt = how many ligands the corpora hold for that target "
          "in total, leaked or not")
    print(pm.to_string(index=False))

    # how close is the label the model could have memorised to the CASP16 answer?
    print("\nlabel agreement on the leaked triples, per model (pKd log units)")
    agree = []
    for model, cs in MODEL_CORPORA.items():
        for ser in TARGET:
            g = dd[(dd.series == ser) & dd.corpus.isin(cs)]
            if g.empty:
                continue
            u = g.groupby("ligand_id").agg(c=("casp16_pKd", "first"),
                                           l=("corpus_label", "median"))
            e = (u.l - u.c).abs()
            agree.append({"model": model, "series": ser, "n": len(u),
                          "min_abs_err": e.min(), "median_abs_err": e.median(),
                          "max_abs_err": e.max(), "mean_abs_err": e.mean(),
                          "n_within_0.1": int((e <= 0.1).sum()),
                          "n_within_0.5": int((e <= 0.5).sum()),
                          "n_exact": int((e < 1e-6).sum())})
    ag = pd.DataFrame(agree)
    ag.to_csv(f"{OUT}/label_agreement.csv", index=False)
    print(ag.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
