
import os
import pandas as pd
import glob
import logging
import yaml
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plabench.analysis.metrics import calculate_metrics  # noqa: E402
from plabench.analysis.units import dg_to_pkd  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(message)s')


MODEL_CONFIG_DIR = "configs/model"
DATASET_CONFIG_DIR = "configs/dataset"


def load_dataset_config(dataset_name, config_dir=DATASET_CONFIG_DIR):
    """Load a dataset YAML config. Returns {} if not found."""
    path = os.path.join(config_dir, f"{dataset_name}.yaml")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logging.warning(f"Failed to parse {path}: {e}")
        return {}


def is_release_run(model_name, dataset_name):
    """True when both halves of a run still have a live Hydra config.

    `outputs/` accumulates exploratory runs whose configs were later moved to
    `archive/configs/`, and those must not reach `results/`. Skipping them at scan
    time is not enough on its own: the merges below deliberately preserve existing
    rows the scan did not reproduce, which would keep an orphan alive forever. The
    same predicate is therefore applied to the preserved frame, so running this
    script twice in a row is a no-op and running it once cleans up after an
    archived experiment.
    """
    return (os.path.exists(os.path.join(MODEL_CONFIG_DIR, f"{model_name}.yaml"))
            and os.path.exists(os.path.join(DATASET_CONFIG_DIR, f"{dataset_name}.yaml")))


# Beside the dataset configs rather than under data/, because the point of a
# manifest is to still be there when data/ has not been unpacked.
MANIFEST_DIR = os.path.join("configs", "manifests")


def _read_manifest(dataset_name):
    path = os.path.join(MANIFEST_DIR, f"{dataset_name}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        ids = {line.strip() for line in f if line.strip() and not line.startswith("#")}
    return ids or None


def _write_manifest(dataset_name, ids):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(os.path.join(MANIFEST_DIR, f"{dataset_name}.txt"), "w") as f:
        f.write(f"# Target IDs {dataset_name} was scored on, written by scripts/collect_results.py\n")
        f.write("# from the input directory. Read back when that directory is not on disk.\n")
        for target_id in sorted(ids):
            f.write(f"{target_id}\n")


def resolve_expected(dataset_name, gt_ids, list_inputs):
    """Narrow a ground-truth ID set to the targets that had model inputs.

    Several datasets score fewer targets than their ground truth holds, because
    only some targets have a structure. `list_inputs` recovers that set from the
    input directory and returns None when the directory is not there.

    The structure directories are the bulk of the Zenodo deposit and nobody
    reproducing the tables alone will unpack them, so the resolved set is also
    committed under data/manifests/. Without it the denominator falls back to the
    full ground truth, and the CASP16 stage 2 rows report 93 predictions out of
    123 targets at 75.6% coverage instead of 93 out of 93 at 100%, with every
    metric unchanged. The manifest is refreshed whenever the directory is present,
    so the two cannot drift apart.
    """
    listed = list_inputs()
    if listed:
        narrowed = gt_ids & listed
        if narrowed:
            _write_manifest(dataset_name, narrowed)
            return narrowed
        logging.warning(f"{dataset_name}: no input matched a ground-truth ID; using the full ground truth.")
        return gt_ids

    stored = _read_manifest(dataset_name)
    if stored:
        narrowed = gt_ids & stored
        if narrowed:
            return narrowed

    logging.warning(f"{dataset_name}: no input directory and no manifest in {MANIFEST_DIR}; "
                    "using the full ground truth as the denominator.")
    return gt_ids


def _subdirs_with(input_dir, suffix=None):
    """Subdirectory names of `input_dir`, or None when it is not on disk.

    With `suffix`, keep only subdirectories holding a `<name><suffix>` file.
    """
    if not input_dir or not os.path.isdir(input_dir):
        return None
    return {d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
            and (suffix is None or os.path.exists(os.path.join(input_dir, d, f"{d}{suffix}")))}


def _protein_pdb_stems(input_dir, last_segment=False):
    """`<id>_protein.pdb` stems in a flat directory, or None when it is not on disk.

    `last_segment` keeps only the part after the final underscore, for the
    Boltz-2 CSAR directories whose files carry a set prefix.
    """
    if not input_dir or not os.path.isdir(input_dir):
        return None
    stems = set()
    for fn in os.listdir(input_dir):
        if fn.endswith("_protein.pdb"):
            stem = fn[: -len("_protein.pdb")]
            stems.add(stem.split("_")[-1] if last_segment else stem)
    return stems


def collect_results(outputs_dir="outputs"):
    os.makedirs("results", exist_ok=True)
    # Find all predictions.csv under any model directory
    pred_files = glob.glob(os.path.join(outputs_dir, "*", "*", "*", "predictions.csv"))

    # Deduplicate: for each model+dataset, prefer timestamp dir over "latest"
    dataset_files = {}
    for f in pred_files:
        parts = f.split(os.sep)
        model_name = parts[-4]
        dataset_name = parts[-3]
        run_name = parts[-2]
        key = f"{model_name}/{dataset_name}"
        if key not in dataset_files:
            dataset_files[key] = []
        dataset_files[key].append((run_name, f))

    deduped = []
    for key, entries in dataset_files.items():
        non_latest = [(r, f) for r, f in entries if r != "latest"]
        if non_latest:
            deduped.extend([f for _, f in non_latest])
        else:
            deduped.extend([f for _, f in entries])
    pred_files = deduped
    
    results = []
    all_failures = []
    
    # Project Root (Assumed to be CWD)
    project_root = os.getcwd()
    
    for pf in pred_files:
        path_parts = pf.split(os.sep)
        # outputs/{model}/{dataset}/{timestamp}/predictions.csv
        model_name = path_parts[-4]
        dataset_name = path_parts[-3]
        timestamp = path_parts[-2]
        run_dir = os.path.dirname(pf)

        if not is_release_run(model_name, dataset_name):
            continue

        # Datasets with multi_protein=true in their YAML config contain multiple
        # protein series mixed together. Pooled metrics across them are misleading
        # (Simpson's paradox), so skip pooled computation here. Per-target metrics
        # and weighted averages are computed via the per-target block below +
        # scripts/weighted_summary.py.
        ds_cfg = load_dataset_config(dataset_name)
        if ds_cfg.get("multi_protein"):
            continue

        # Determine relative Ground Truth path
        gt_rel_path = None
        gt_format = "csv"  # default
        if "l1000" in dataset_name.lower():
            gt_rel_path = os.path.join("data", "casp16_data", "labels", "L1000_exper_affinity.csv")
        elif "l3000" in dataset_name.lower():
            gt_rel_path = os.path.join("data", "casp16_data", "labels", "L3000_exper_affinity.csv")
        elif "casf2013" in dataset_name.lower():
            # Sequence models use CSV GT (standardized test), unlike others using CoreSet.dat
            if "sequence" in model_name.lower() or "sequence" in dataset_name.lower() or "deepdta" in model_name.lower() or "mixingdta" in model_name.lower():
                gt_rel_path = os.path.join("data", "Structure_independent", "CASF-2013_standardized_test.csv")
                gt_format = "csv"
            else:
                gt_rel_path = os.path.join("data", "CASF-2013", "power_scoring", "CoreSet.dat")
                gt_format = "casf_dat"
        elif "casf2016" in dataset_name.lower():
            if "sequence" in model_name.lower() or "sequence" in dataset_name.lower() or "deepdta" in model_name.lower() or "mixingdta" in model_name.lower():
                gt_rel_path = os.path.join("data", "Structure_independent", "CASF-2016_standardized_test.csv")
                gt_format = "csv"
            else:
                gt_rel_path = os.path.join("data", "CASF-2016", "power_scoring", "CoreSet.dat")
                gt_format = "casf_dat"
        elif "csar_hiq51" in dataset_name.lower():
            if "sequence" in model_name.lower() or "sequence" in dataset_name.lower() or "deepdta" in model_name.lower() or "mixingdta" in model_name.lower():
                gt_rel_path = os.path.join("data", "Structure_independent", "CSAR-HIQ_51_standardized_test.csv")
            else:
                gt_rel_path = os.path.join("data", "CSAR-HIQ_51", "index.txt")
            gt_format = "csv"
        elif "csar_hiq36" in dataset_name.lower():
            if "sequence" in model_name.lower() or "sequence" in dataset_name.lower() or "deepdta" in model_name.lower() or "mixingdta" in model_name.lower():
                gt_rel_path = os.path.join("data", "Structure_independent", "CSAR-HIQ_36_standardized_test.csv")
            else:
                gt_rel_path = os.path.join("data", "CSAR-HIQ_36", "index.txt")
            gt_format = "csv"
            
        gt_path = os.path.abspath(gt_rel_path) if gt_rel_path else None
        
        if not gt_path or not os.path.exists(gt_path):
            logging.warning(f"Skipping {dataset_name}: Ground truth not found at {gt_rel_path}")
            continue
            
        try:
            # Load Predictions
            preds_df = pd.read_csv(pf, header=None, names=["prediction", "name"])

            if gt_format == "casf_dat":
                # Parse CoreSet.dat: space-delimited, col 0=PDB, col 3=-logKd/Ki
                gt_rows = []
                with open(gt_path, 'r') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) >= 4:
                            gt_rows.append({"pdb_id": parts[0], "affinity": float(parts[3])})
                gt_df = pd.DataFrame(gt_rows)
                gt_id_col = "pdb_id"
                gt_score_col = "affinity"
            else:
                # Load Ground Truth (CSV)
                gt_df = pd.read_csv(gt_path)
                gt_id_col = gt_df.columns[0]
                if "Target ID" in gt_df.columns: gt_id_col = "Target ID"
                elif "\ufeffTarget ID" in gt_df.columns: gt_id_col = "\ufeffTarget ID"

                gt_score_col = None
                for col in gt_df.columns:
                    if "binding_affinity" in col.lower() or "affinity" in col.lower():
                        gt_score_col = col
                        break
                if not gt_score_col:
                    gt_score_col = gt_df.columns[-1] # Fallback to last column
            
            # Determine EXPECTED Total for this dataset
            # Stage 2 datasets depend on available inputs (some structures missing)
            # We filter GT to only include targets present in the input directory.
            # Stage 1 datasets usually include all targets.
            
            expected_total = len(gt_df)
            expected_gt_ids = set(gt_df[gt_id_col].astype(str))

            # Boltz2 as affinity predictor uses sequence+SMILES input,
            # not limited by available structure files — use full GT count.
            if model_name == "boltz2":
                pass  # keep expected_total = len(gt_df)
            elif "stage2" in dataset_name.lower():
                # data/casp16_data/stage2_input/L3000_prepared/Lxxxx/
                if "l3000" in dataset_name.lower():
                    input_dir = os.path.join("data", "casp16_data", "stage2_input", "L3000_prepared")
                elif "l1000" in dataset_name.lower():
                    input_dir = os.path.join("data", "casp16_data", "stage2_input", "L1000_prepared")
                else:
                    input_dir = None
                expected_gt_ids = resolve_expected(
                    dataset_name, expected_gt_ids, lambda: _subdirs_with(input_dir))
                expected_total = len(expected_gt_ids)

            elif "casf2013" in dataset_name.lower() or "casf2016" in dataset_name.lower():
                # Total based on available inputs, not GT
                year = "2013" if "casf2013" in dataset_name.lower() else "2016"
                if "exp" in dataset_name.lower():
                    # Experimental: coreset subdirectories contain {pdb_id}_protein.pdb
                    input_dir = os.path.join("data", f"CASF-{year}", "coreset")
                    expected_gt_ids = resolve_expected(
                        dataset_name, expected_gt_ids,
                        lambda: _subdirs_with(input_dir, "_protein.pdb"))
                    expected_total = len(expected_gt_ids)
                elif "sequence" in dataset_name.lower() or "deepdta" in dataset_name.lower():
                     # Sequence models use full GT (no structural filtering)
                     expected_total = len(gt_df)
                else:
                    # Boltz2: flat directory with {pdb_id}_protein.pdb
                    input_dir = os.path.join("data", "Boltz2_structures", f"casf{year}")
                    expected_gt_ids = resolve_expected(
                        dataset_name, expected_gt_ids,
                        lambda: _protein_pdb_stems(input_dir))
                    expected_total = len(expected_gt_ids)

            elif "csar_hiq51" in dataset_name.lower() or "csar_hiq36" in dataset_name.lower():
                hiq = "hiq51" if "hiq51" in dataset_name.lower() else "hiq36"
                if "boltz2" in dataset_name.lower():
                    # Boltz2: flat or prefixed naming, so keep the last segment
                    boltz_dir = "csar51" if hiq == "hiq51" else "csar36"
                    input_dir = os.path.join("data", "Boltz2_structures", boltz_dir)
                    expected_gt_ids = resolve_expected(
                        dataset_name, expected_gt_ids,
                        lambda: _protein_pdb_stems(input_dir, last_segment=True))
                    expected_total = len(expected_gt_ids)
                else:
                    # Exp: nested directories with {pdb_id}/{pdb_id}_protein.pdb
                    data_dir_name = "CSAR-HIQ_51" if hiq == "hiq51" else "CSAR-HIQ_36"
                    input_dir = os.path.join("data", data_dir_name)
                    expected_gt_ids = resolve_expected(
                        dataset_name, expected_gt_ids,
                        lambda: _subdirs_with(input_dir, "_protein.pdb"))
                    expected_total = len(expected_gt_ids)

            # Filter GT DF for merging? No, merging works regardless.
            # Just need Expected Set for "Missing" calculation.
            
            # Merge
            merged = pd.merge(preds_df, gt_df, left_on="name", right_on=gt_id_col)
            
            if len(merged) == 0:
                # Try cleaning IDs
                preds_df['name_clean'] = preds_df['name'].astype(str).str.replace(r"['\[\]]", "", regex=True)
                merged = pd.merge(preds_df, gt_df, left_on="name_clean", right_on=gt_id_col)
            
            if len(merged) > 0:
                # Unit conversion: CASP16 affinity is a dG in kcal/mol -> pKd.
                # CASF-2013 is already -logKd/Ki, no conversion needed
                if "l1000" in dataset_name.lower() or "l3000" in dataset_name.lower():
                    merged[gt_score_col] = dg_to_pkd(merged[gt_score_col])
                
                metrics = calculate_metrics(merged["prediction"], merged[gt_score_col])
                metrics['Model'] = model_name
                metrics['Dataset'] = dataset_name
                metrics['Timestamp'] = timestamp
                metrics['N'] = len(merged)
                metrics['Total'] = expected_total
                metrics['Coverage'] = f"{len(merged)/expected_total*100:.1f}%" if expected_total > 0 else "0.0%"
                results.append(metrics)
                logging.info(f"Processed {dataset_name} (N={len(merged)}/{expected_total})")
            
            # --- 2. Collect Failures (Missing + Explicit) ---
            
            # A) Identify Missing Targets relative to EXPECTED set
            # Use the column we successfully merged on ('name' or 'name_clean')
            if 'name_clean' in preds_df.columns and len(merged) > 0:
                pred_ids = set(merged['name_clean'].astype(str))
            else:
                pred_ids = set(merged['name'].astype(str)) if 'name' in preds_df.columns else set()
                
            missing_ids = expected_gt_ids - pred_ids
            
            for mid in missing_ids:
                all_failures.append({
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "TargetID": mid,
                    "ErrorType": "MissingInPreds",
                    "Details": "Sample absent from final predictions.csv (e.g. Inference Crash/OOM or Preprocessing Skip)",
                    "Timestamp": timestamp
                })

            # B) Load Explicit Failure Log (failures.csv)
            fail_csv = os.path.join(run_dir, "failures.csv")
            if os.path.exists(fail_csv):
                try:
                    explicit_fails = pd.read_csv(fail_csv)
                    # Add to list
                    for _, row in explicit_fails.iterrows():
                        # Check if this failure is already covered by "MissingInPreds"?
                        # Usually yes, but explicit log gives BETTER reason.
                        # So we should populate the reason for the missing ones.
                        
                        # Find the entry in all_failures for this ID and update it
                        # Or just append and we deduplicate later?
                        # Better to update:
                        
                        target_id = str(row['TargetID'])
                        reason = row['ErrorType']
                        details = row['Details']
                        
                        # Search in all_failures
                        found = False
                        for fail in all_failures:
                            if fail['Model'] == model_name and fail['Dataset'] == dataset_name and fail['TargetID'] == target_id:
                                fail['ErrorType'] = reason
                                fail['Details'] = details # Overwrite generic message
                                found = True
                                break
                        
                        if not found:
                             # If explicitly failed but not in expected set? Could happen if input list > expected list?
                             pass 
                except Exception as e:
                    logging.warning(f"Error reading failures.csv: {e}")

        except Exception as e:
            logging.error(f"Error processing {dataset_name}: {e}")
            
    # --- Per-target evaluation for multi_protein datasets ---
    # Datasets with multi_protein=true in their YAML contain multiple protein series.
    # Split predictions by compound_id prefix (before first "_"), treat each prefix
    # as one protein target, and compute per-target metrics. GT path comes from the
    # dataset YAML. Uses compound_id column from GT CSV.
    per_target_results = []
    for pf in pred_files:
        path_parts = pf.split(os.sep)
        model_name = path_parts[-4]
        dataset_name = path_parts[-3]
        if not is_release_run(model_name, dataset_name):
            continue
        ds_cfg = load_dataset_config(dataset_name)
        if not ds_cfg.get("multi_protein"):
            continue
        gt_rel = ds_cfg.get("ground_truth_path")
        if not gt_rel:
            logging.warning(f"multi_protein dataset {dataset_name} has no ground_truth_path in config; skipping per-target eval")
            continue
        gt_path = os.path.abspath(gt_rel)
        if not os.path.exists(gt_path):
            logging.warning(f"GT not found for {dataset_name}: {gt_path}")
            continue
        try:
            preds_df = pd.read_csv(pf, header=None, names=["prediction", "name"])
            gt_df = pd.read_csv(gt_path)
            # Auto-detect compound_id column (first col that isn't 'prediction')
            id_col = None
            for col in ["compound_id", "Compound_id", "compound_ID", "Target ID", "TargetID"]:
                if col in gt_df.columns:
                    id_col = col
                    break
            if id_col is None:
                id_col = gt_df.columns[0]
            # Auto-detect score column
            score_col = None
            for col in gt_df.columns:
                if col.lower() in ("affinity", "label", "binding_affinity", "pchembl_value_median"):
                    score_col = col
                    break
            if score_col is None:
                score_col = gt_df.columns[-1]

            merged = pd.merge(preds_df, gt_df, left_on="name", right_on=id_col)
            if len(merged) == 0:
                continue
            # Target = prefix of compound_id before first "_"
            merged["target"] = merged[id_col].astype(str).str.split("_").str[0]
            for target, group in merged.groupby("target"):
                if len(group) < 3:
                    continue
                m = calculate_metrics(group["prediction"], group[score_col])
                m["Model"] = model_name
                m["Dataset"] = dataset_name
                m["Target"] = target
                m["N"] = len(group)
                per_target_results.append(m)
        except Exception as e:
            logging.error(f"Error in per-target eval for {model_name}/{dataset_name}: {e}")

    if per_target_results:
        per_target_csv = os.path.join("results", "per_target_evaluation.csv")
        pt_df = pd.DataFrame(per_target_results)
        cols = ['Model', 'Dataset', 'Target', 'N', 'RMSE', 'MSE', 'Pearson', 'Spearman', 'Kendall', 'CI', 'Rm2']
        for col in cols:
            if col not in pt_df.columns:
                pt_df[col] = None
        pt_df = pt_df[cols].sort_values(by=['Dataset', 'Model', 'Target'])

        if os.path.exists(per_target_csv):
            try:
                existing = pd.read_csv(per_target_csv)
                new_keys = set(tuple(x) for x in pt_df[['Model', 'Dataset', 'Target']].values)
                live = existing.apply(lambda r: is_release_run(r['Model'], r['Dataset']), axis=1)
                if (~live).any():
                    logging.info(f"Dropping {int((~live).sum())} per-target rows from archived configs")
                mask = live & existing.apply(
                    lambda r: (r['Model'], r['Dataset'], r['Target']) not in new_keys, axis=1)
                pt_df = pd.concat([existing[mask], pt_df], ignore_index=True)
                pt_df = pt_df.sort_values(by=['Dataset', 'Model', 'Target'])
            except Exception as e:
                logging.error(f"Error merging per_target_evaluation.csv: {e}")

        pt_df.to_csv(per_target_csv, index=False)
        print(f"\nSaved per-target evaluation ({len(per_target_results)} new rows) to {per_target_csv}")

    # Save Results
    summary_csv = os.path.join("results", "benchmark_summary.csv")
    if results:
        df = pd.DataFrame(results)
        cols = ['Model', 'Dataset', 'N', 'Total', 'Coverage', 'RMSE', 'MSE', 'Pearson', 'Spearman', 'Kendall', 'CI', 'Rm2']
        # Add missing columns with None/NaN if they don't exist
        for col in cols:
            if col not in df.columns:
                df[col] = None
        new_df = df[cols]
        print("\n=== Benchmark Results (Newly Scanned) ===")
        print(new_df.to_markdown(index=False))
        
        # Merge with existing CSV to preserve manually added or CV scripts generated rows
        if os.path.exists(summary_csv):
            try:
                existing_df = pd.read_csv(summary_csv)
                # Create keys for merging
                new_keys = set(tuple(x) for x in new_df[['Model', 'Dataset']].values)
                # Keep existing rows that are NOT in the new scan, as long as they
                # still correspond to a live config. The Davis and KIBA rows come
                # from scripts/cv/collect_kiba_davis_cv.py rather than from this
                # scan, and survive here.
                live = existing_df.apply(lambda r: is_release_run(r['Model'], r['Dataset']), axis=1)
                if (~live).any():
                    logging.info(f"Dropping {int((~live).sum())} summary rows from archived configs")
                mask = live & existing_df.apply(
                    lambda row: (row['Model'], row['Dataset']) not in new_keys, axis=1)
                preserved_df = existing_df[mask]
                
                final_df = pd.concat([preserved_df, new_df], ignore_index=True)
                # Sort by Model then Dataset for clean output
                final_df = final_df.sort_values(by=['Model', 'Dataset'])
            except Exception as e:
                logging.error(f"Error merging with existing benchmark_summary.csv: {e}")
                final_df = new_df
        else:
            final_df = new_df
            
        final_df.to_csv(summary_csv, index=False)
        print(f"\nSaved (and merged) to {summary_csv}")
    
    # Save Failures. The summary above preserves rows this scan did not reproduce,
    # and the same care is needed here: a checkout with no outputs/ scans nothing,
    # which must not be read as "every target succeeded" and take the committed
    # failure list down with it.
    outfile = os.path.join("results", "benchmark_failures.csv")
    if all_failures:
        fail_df = pd.DataFrame(all_failures)
        # Sort by dataset and ID
        fail_df = fail_df.sort_values(by=['Model', 'Dataset', 'TargetID'])
        fail_df.to_csv(outfile, index=False)
        print(f"Saved {len(all_failures)} failure records to {outfile}")
    elif results:
        cols = ['Model', 'Dataset', 'TargetID', 'ErrorType', 'Details', 'Timestamp']
        pd.DataFrame(columns=cols).to_csv(outfile, index=False)
        print(f"No failures detected. Wrote an empty {outfile}")
    else:
        print(f"Nothing scanned. Left {outfile} as it is.")

if __name__ == "__main__":
    collect_results()
