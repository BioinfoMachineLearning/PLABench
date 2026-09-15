"""Chain-rule ablation, step 3: assemble the MEETA dataset directories for the concat arm.

Rather than re-deriving the drug keys, this reuses the deployed pkl files verbatim and
substitutes only element [2], the protein string. Row order in the deployed pkls follows
the source CSVs exactly, which is asserted here before anything is written. The result is
a split that is bit-identical to the longest-chain arm on the drug side, on the labels and
on the row order -- the protein string is the single changed variable.

Ligand embeddings are keyed by SMILES and therefore unchanged; the MolFormer tables are
symlinked rather than recomputed.
"""
import os
import pickle
import shutil
import sys

import pandas as pd
import torch

DB = "/bmlfast/Lyuwei/0.Projects/MixingDTA/DTA_DataBase"
DATA = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation/data"
WORK = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation/emb"

# new dataset name -> (deployed dataset name, [(pkl file, concat csv), ...])
JOBS = {
    "PDBbind_Refined_91_concat": ("PDBbind_Refined_91", [
        ("train_1.pkl", "pdbbind_train_full_concat.csv"),
        ("test_1.pkl", "pdbbind_val_full_concat.csv"),
    ]),
    "CASF2016_Std_concat": ("CASF2016_Std", [("test_1.pkl", "CASF2016_Std_concat.csv")]),
    "CASF2013_Std_concat": ("CASF2013_Std", [("test_1.pkl", "CASF2013_Std_concat.csv")]),
    "CSAR36_Std_concat": ("CSAR36_Std", [("test_1.pkl", "CSAR36_Std_concat.csv")]),
    "CSAR51_Std_concat": ("CSAR51_Std", [("test_1.pkl", "CSAR51_Std_concat.csv")]),
}


def load_esm():
    """Merge the per-GPU shards into one sequence -> embedding table."""
    merged = {}
    for f in sorted(os.listdir(WORK)):
        if f.startswith("esm3_shard") and f.endswith(".pt"):
            merged.update(torch.load(os.path.join(WORK, f), map_location="cpu"))
    print(f"merged ESM3 shards: {len(merged)} sequences")
    return merged


def main():
    esm = load_esm()
    ok = True

    for new_name, (old_name, parts) in JOBS.items():
        out_dir = os.path.join(DB, "datasets", new_name)
        os.makedirs(out_dir, exist_ok=True)

        new_parts, all_prot, all_drug = {}, [], []
        for pkl_name, csv_name in parts:
            with open(os.path.join(DB, "datasets", old_name, pkl_name), "rb") as f:
                rows = pickle.load(f)
            df = pd.read_csv(os.path.join(DATA, csv_name))
            assert len(rows) == len(df), f"{new_name}/{pkl_name}: {len(rows)} vs {len(df)}"

            # the deployed pkl must line up row-for-row with the source CSV
            src = pd.read_csv(os.path.join(
                "/home/lwfvx/Lyuwei/1.Datasets",
                "pdbbind_refined_91" if old_name == "PDBbind_Refined_91"
                else "pdbbind_like_test_normalize",
                {"train_1.pkl": "pdbbind_train_full.csv",
                 "test_1.pkl": "pdbbind_val_full.csv"}[pkl_name]
                if old_name == "PDBbind_Refined_91" else
                {"CASF2016_Std": "CASF-2016_standardized_test.csv",
                 "CASF2013_Std": "CASF-2013_standardized_test.csv",
                 "CSAR36_Std": "CSAR-HIQ_36_standardized_test.csv",
                 "CSAR51_Std": "CSAR-HIQ_51_standardized_test.csv"}[old_name]))
            bad = sum(1 for i, r in enumerate(rows)
                      if r[2] != str(src.target_sequence.iloc[i]))
            if bad:
                print(f"  !! {new_name}/{pkl_name}: {bad} rows do not align with the CSV")
                ok = False
                continue

            newrows = [[r[0], r[1], str(s), r[3]]
                       for r, s in zip(rows, df.target_sequence.astype(str))]
            new_parts[pkl_name] = newrows
            all_prot += [r[2] for r in newrows]
            all_drug += [r[0] for r in newrows]

        if not new_parts:
            continue

        # mappings: drug side copied verbatim so the MolFormer table stays valid
        shutil.copy(os.path.join(DB, "datasets", old_name, "drug_id_2_idx.pkl"),
                    os.path.join(out_dir, "drug_id_2_idx.pkl"))
        shutil.copy(os.path.join(DB, "datasets", old_name, "drug_idx_2_id.pkl"),
                    os.path.join(out_dir, "drug_idx_2_id.pkl"))
        uniq = list(dict.fromkeys(all_prot))
        p2i = {s: i for i, s in enumerate(uniq)}
        with open(os.path.join(out_dir, "protein_id_2_idx.pkl"), "wb") as f:
            pickle.dump(p2i, f)
        with open(os.path.join(out_dir, "protein_idx_2_id.pkl"), "wb") as f:
            pickle.dump({i: s for s, i in p2i.items()}, f)

        for pkl_name, rows in new_parts.items():
            with open(os.path.join(out_dir, pkl_name), "wb") as f:
                pickle.dump(rows, f)
        if "PDBbind" in new_name:  # deployed prep writes valid == test
            with open(os.path.join(out_dir, "valid_1.pkl"), "wb") as f:
                pickle.dump(new_parts["test_1.pkl"], f)

        # embeddings
        sub = {s: esm[s] for s in uniq if s in esm}
        missing = [s for s in uniq if s not in esm]
        torch.save(sub, os.path.join(DB, "ESM3_open_small", f"{new_name}.pt"))
        link = os.path.join(DB, "MolFormer", f"{new_name}.pt")
        if not os.path.exists(link):
            os.symlink(os.path.join(DB, "MolFormer", f"{old_name}.pt"), link)

        print(f"{new_name:28s} rows {sum(len(v) for v in new_parts.values()):5d}  "
              f"unique proteins {len(uniq):5d}  embedded {len(sub):5d}  "
              f"MISSING {len(missing)}")
        if missing:
            ok = False
            pd.DataFrame({"len": [len(s) for s in missing], "seq": missing}).to_csv(
                os.path.join(WORK, f"missing_{new_name}.csv"), index=False)

    print("\nALL EMBEDDINGS PRESENT" if ok else
          "\nINCOMPLETE - see missing_*.csv / failures_*.csv in emb/")


if __name__ == "__main__":
    sys.exit(main())
