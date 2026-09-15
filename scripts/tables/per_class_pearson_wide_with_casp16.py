"""Build a 15-row per-class Pearson table that folds CASP16 single-target
results into their corresponding chembl35 family rows (N-weighted), with an
additional Ensemble column for the Boltz2 + FLOWR.ROOT per-compound-averaged
ensemble.

Mapping:
  - Chymase (CASP16 L1000, n=17, P23946)   → Protease
  - Autotaxin (CASP16 L3000, n=123, Q13822) → Hydrolase

For every method, per-class Pearson is the N-weighted mean across member
targets:
  merged = (chembl35_class_pearson * chembl35_n + casp16_pearson * casp16_n)
           / (chembl35_n + casp16_n)
The 9 single-method columns are read from results/chembl35/per_class_summary_chembl35.csv
(chembl35 per-target rows, already N-weighted across the class's chembl35
targets) and results/benchmark_summary.csv (per-CASP16-target Pearson),
EXCEPT the Boltz2 row on CASP16 L3000 (Autotaxin), which is recomputed from
the latest predictions to pick up the 2 newly-added compounds (was N=121,
now N=123).

The Ensemble column is computed end-to-end from per-compound predictions:
  ensemble_pred = (boltz2_pred + flowr_root_pred) / 2
  per-target Pearson(ensemble_pred, GT) on each chembl35 target + on Chymase
  and Autotaxin, then N-weighted per class with the same fold-in formula.

Output: results/chembl35/per_class_pearson_wide_chembl35_casp16.csv  (15 rows)
"""
import pandas as pd
from scipy.stats import pearsonr

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plabench.analysis.units import dg_to_pkd  # noqa: E402

PER_CLASS = "results/chembl35/per_class_summary_chembl35.csv"
BENCH = "results/benchmark_summary.csv"
CLASS_MAP = "data/chembl35/target_class_map.csv"
CSV_OUT = "results/chembl35/per_class_pearson_wide_chembl35_casp16.csv"

# Single-method column order (matches the renamed CSV); Ensemble appended last
METHOD_ORDER = ["Boltz2", "FlowDock", "FLOWR.ROOT", "Graph_RG", "LCDD team",
                "MFE", "Deepdta", "LLF", "Mixingdta"]
ENSEMBLE_COL = "Ensemble"

RENAME = {
    "boltz2": "Boltz2",
    "flowdock": "FlowDock",
    "flowr_root": "FLOWR.ROOT",
    "haiping": "Graph_RG",
    "bapred": "LCDD team",
    "mfe": "MFE",
    "deepdta": "Deepdta",
    "llf": "LLF",
    "mixingdta": "Mixingdta",
}

CASP16_L1000_DATASETS = {  # Chymase → Protease
    "boltz2":     "boltz2_casp16_l1000",
    "flowdock":   "af3_casp16_l1000_stage1",
    "flowr_root": "flowr_root_af3_casp16_l1000_stage1",
    "haiping":    "haiping_af3_casp16_l1000_stage1",
    "bapred":     "af3_casp16_l1000_stage1",
    "mfe":        "mfe_af3_casp16_l1000_stage1",
    "deepdta":    "sequence_casp16_l1000",
    "llf":        "sequence_casp16_l1000",
    "mixingdta":  "sequence_casp16_l1000",
}
CASP16_L3000_DATASETS = {  # Autotaxin → Hydrolase
    "boltz2":     "boltz2_casp16_l3000",
    "flowdock":   "af3_casp16_l3000_stage1",
    "flowr_root": "flowr_root_af3_casp16_l3000_stage1",
    "haiping":    "haiping_af3_casp16_l3000_stage1",
    "bapred":     "af3_casp16_l3000_stage1",
    "mfe":        "mfe_af3_casp16_l3000_stage1",
    "deepdta":    "sequence_casp16_l3000",
    "llf":        "sequence_casp16_l3000",
    "mixingdta":  "sequence_casp16_l3000",
}

CASP16_MERGES = [
    ("Protease", 17, CASP16_L1000_DATASETS),
    ("Hydrolase", 123, CASP16_L3000_DATASETS),
]

PRETTY_CLASS = {
    "Family A G protein-coupled receptor": "Family A GPCR",
    "Cytochrome p450": "Cytochrome P450",
    "Unclassified protein": "Unclassified",
    "Other nuclear protein": "Other nuclear",
}

# Prediction file paths used for ensemble + Boltz2 L3000 refresh
PREDS = {
    "boltz2": {
        "chembl35_full": "outputs/boltz2/boltz2_chembl35_full/latest/predictions.csv",
        "casp16_l1000":  "outputs/boltz2/boltz2_casp16_l1000/latest/predictions.csv",
        "casp16_l3000":  "outputs/boltz2/boltz2_casp16_l3000/latest/predictions.csv",
    },
    "flowr_root": {
        "chembl35_full": "outputs/flowr_root/flowr_root_af3_chembl35_full/latest/predictions.csv",
        "casp16_l1000":  "outputs/flowr_root/flowr_root_af3_casp16_l1000_stage1/latest/predictions.csv",
        "casp16_l3000":  "outputs/flowr_root/flowr_root_af3_casp16_l3000_stage1/latest/predictions.csv",
    },
}

# (gt_csv, compound_id_col, affinity_col, stores_dg) per dataset. stores_dg marks the
# CASP16 files, which report a dG in kcal/mol rather than a p-scale affinity.
# chembl35 `affinity` is pChEMBL (positive for strong binders, same sign as
# Boltz2/FLOWR.ROOT predictions). CASP16 `binding_affinity` is stored as
# log10(K_d in M) (negative for strong binders), so it must be NEGATED to
# align signs with model predictions (which output pK_d).
GT_PATHS = {
    "chembl35_full": ("data/chembl35/chembl35_full_input.csv",            "compound_id", "affinity",          False),
    "casp16_l1000":  ("data/casp16_data/labels/L1000_exper_affinity.csv", "Target ID",   "binding_affinity",  True),
    "casp16_l3000":  ("data/casp16_data/labels/L3000_exper_affinity.csv", "Target ID",   "binding_affinity",  True),
}


def load_pred(path):
    df = pd.read_csv(path, header=None, names=["pred", "compound_id"],
                     dtype={"compound_id": str})
    df["compound_id"] = df["compound_id"].str.strip()
    return df


def joined_for_dataset(dataset):
    """Inner-join boltz2 + flowr_root + GT on compound_id; add ensemble col."""
    boltz2 = load_pred(PREDS["boltz2"][dataset]).rename(columns={"pred": "boltz2"})
    flowr = load_pred(PREDS["flowr_root"][dataset]).rename(columns={"pred": "flowr_root"})
    gt_csv, gt_id, gt_aff, stores_dg = GT_PATHS[dataset]
    # utf-8-sig handles potential BOM in CASP16 label CSVs
    gt = pd.read_csv(gt_csv, encoding="utf-8-sig")
    gt = gt.rename(columns={gt_id: "compound_id", gt_aff: "affinity"})
    gt = gt[["compound_id", "affinity"]]
    gt["compound_id"] = gt["compound_id"].astype(str).str.strip()
    if stores_dg:
        gt["affinity"] = dg_to_pkd(gt["affinity"])
    df = boltz2.merge(flowr, on="compound_id").merge(gt, on="compound_id")
    df["ensemble"] = (df["boltz2"] + df["flowr_root"]) / 2.0
    return df


def safe_pearson(x, y):
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


def chembl35_ensemble_per_class():
    """N-weighted Ensemble per chembl35 Target_class (84 targets → 15 classes)."""
    df = joined_for_dataset("chembl35_full")
    df["target"] = df["compound_id"].str.split("_", n=1).str[0]

    per_tgt = []
    for tgt, g in df.groupby("target"):
        per_tgt.append({
            "target": tgt,
            "n_compounds": len(g),
            "ensemble": safe_pearson(g["ensemble"].values, g["affinity"].values),
        })
    per_tgt = pd.DataFrame(per_tgt)

    cm = pd.read_csv(CLASS_MAP)
    cm["Target_class"] = cm["Target_class"].str.split("|")
    cm = cm.explode("Target_class")
    cm["Target_class"] = cm["Target_class"].str.strip()

    merged = per_tgt.merge(cm, left_on="target", right_on="UniProt_ID", how="left")
    if merged["Target_class"].isna().any():
        miss = merged[merged["Target_class"].isna()]["target"].unique()
        raise RuntimeError(f"Targets without class: {miss}")

    rows = []
    for cls, g in merged.groupby("Target_class"):
        m = g["ensemble"].notna()
        ens = (g.loc[m, "ensemble"] * g.loc[m, "n_compounds"]).sum() / g.loc[m, "n_compounds"].sum()
        rows.append({"Target_class": cls, "ensemble_chembl35": ens})
    return pd.DataFrame(rows)


def casp16_single_target_pearson(dataset):
    """Return dict {boltz2, ensemble, n_compounds} for the single CASP16 target."""
    df = joined_for_dataset(dataset)
    return {
        "boltz2":   safe_pearson(df["boltz2"].values, df["affinity"].values),
        "ensemble": safe_pearson(df["ensemble"].values, df["affinity"].values),
        "n_compounds": len(df),
    }


def load_chembl35_wide():
    df = pd.read_csv(PER_CLASS)
    n_info = (df.groupby("Target_class")
                .agg(n_targets=("n_targets", "max"),
                     n_compounds=("n_compounds", "max"))
                .reset_index())
    piv = (df.pivot_table(index="Target_class", columns="Model",
                          values="mean_Pearson")
             .rename(columns=RENAME)
             .reindex(columns=METHOD_ORDER))
    return n_info.merge(piv.reset_index(), on="Target_class")


def casp16_method_pearsons_from_bench(datasets):
    """Pull dataset-level Pearson per method from benchmark_summary.csv."""
    bench = pd.read_csv(BENCH)
    out = {}
    for mid, ds_id in datasets.items():
        m = bench[(bench["Model"] == mid) & (bench["Dataset"] == ds_id)]
        if m.empty:
            raise RuntimeError(f"No row for Model={mid} Dataset={ds_id}")
        out[RENAME[mid]] = float(m.iloc[0]["Pearson"])
    return out


def main():
    wide = load_chembl35_wide()

    # 1) Compute ensemble per chembl35 class (15 values) from per-compound preds
    ens_chembl = chembl35_ensemble_per_class()
    wide = wide.merge(ens_chembl, on="Target_class", how="left")

    # 2) Compute CASP16 single-target Pearsons (boltz2 + ensemble) from preds
    chymase = casp16_single_target_pearson("casp16_l1000")
    autotaxin = casp16_single_target_pearson("casp16_l3000")
    print(f"Chymase (L1000) n={chymase['n_compounds']}: "
          f"Boltz2={chymase['boltz2']:.4f}  Ensemble={chymase['ensemble']:.4f}")
    print(f"Autotaxin (L3000) n={autotaxin['n_compounds']}: "
          f"Boltz2={autotaxin['boltz2']:.4f}  Ensemble={autotaxin['ensemble']:.4f}")

    # 3) Apply CASP16 fold-in for all 9 single methods + Ensemble
    for class_name, n_casp16_default, ds_map in CASP16_MERGES:
        # Pull single-method Pearsons from benchmark_summary, BUT override
        # Boltz2 with the freshly-computed value (catches 2-compound update).
        casp16_p = casp16_method_pearsons_from_bench(ds_map)
        if class_name == "Hydrolase":
            casp16_p["Boltz2"] = autotaxin["boltz2"]
            casp16_ens_p = autotaxin["ensemble"]
            n_casp16 = autotaxin["n_compounds"]
        elif class_name == "Protease":
            casp16_p["Boltz2"] = chymase["boltz2"]
            casp16_ens_p = chymase["ensemble"]
            n_casp16 = chymase["n_compounds"]
        else:
            raise RuntimeError(class_name)

        idx = wide.index[wide["Target_class"] == class_name]
        if len(idx) != 1:
            raise RuntimeError(f"Expected one '{class_name}' row, got {len(idx)}")
        i = idx[0]
        n_old = int(wide.at[i, "n_compounds"])
        n_new = n_old + n_casp16
        wide.at[i, "n_targets"] = int(wide.at[i, "n_targets"]) + 1
        wide.at[i, "n_compounds"] = n_new
        for col in METHOD_ORDER:
            old_p = float(wide.at[i, col])
            new_p = (old_p * n_old + casp16_p[col] * n_casp16) / n_new
            wide.at[i, col] = new_p
        # Ensemble merge (use chembl35 ensemble per-class + casp16 single-target)
        old_ens = float(wide.at[i, "ensemble_chembl35"])
        wide.at[i, "ensemble_chembl35"] = (old_ens * n_old + casp16_ens_p * n_casp16) / n_new

    wide = wide.rename(columns={"ensemble_chembl35": ENSEMBLE_COL})

    for c in METHOD_ORDER + [ENSEMBLE_COL]:
        wide[c] = wide[c].round(4)

    wide = wide.sort_values("n_compounds", ascending=False).reset_index(drop=True)
    wide["Target_class"] = wide["Target_class"].map(
        lambda s: PRETTY_CLASS.get(s, s)
    )

    cols = ["Target_class", "n_targets", "n_compounds"] + METHOD_ORDER + [ENSEMBLE_COL]
    wide = wide[cols]
    wide.to_csv(CSV_OUT, index=False)
    print(f"\nWrote {CSV_OUT}  ({len(wide)} rows)\n")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
