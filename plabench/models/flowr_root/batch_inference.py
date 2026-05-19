"""Batch FLOWR.root affinity prediction.

Loads the model ONCE and iterates over a manifest CSV of (target, pdb, sdf)
pairs across all configured seeds. Replaces the per-target subprocess loop in
flowr_root_wrapper for large datasets (e.g. chembl35_full, 7091 compounds).

Per-(target, seed) outputs land in `{sandbox}/{target_id}/seed_{N}/gen_lig_with_aff.sdf`,
matching the existing per-seed cache layout, so this is fully resumable.

Final aggregation: per-seed median over the four affinity heads
(pic50, pki, pkd, pec50), then median across seeds. Same as
plabench/models/wrappers/flowr_root_wrapper.py (paper §5.4 / Fig 5).
"""
import argparse
import csv
import logging
import math
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch
from rdkit import Chem

# Ensure flowr_root fork is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
FLOWR_ROOT = os.path.join(PROJECT_ROOT, "forks", "flowr_root")
if FLOWR_ROOT not in sys.path:
    sys.path.insert(0, FLOWR_ROOT)

import flowr.util.rdkit as smolRD  # noqa: E402
from flowr.data.dataset import GeometricDataset  # noqa: E402
from flowr.gen.utils import (  # noqa: E402
    get_dataloader,
    load_data_from_pdb,
    load_util,
)
from flowr.predict.predict import predict_affinity_batch  # noqa: E402
from flowr.scriptutil import load_model  # noqa: E402
from flowr.util.pocket import PocketComplexBatch  # noqa: E402

log = logging.getLogger("flowr_root.batch")


# Defaults match `forks/flowr_root/scripts/predict_aff.sl` (paper §5.4 high-quality)
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "linear"
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"
DEFAULT_BUCKET_COST_SCALE = "quadratic"


def make_args(args):
    """Construct SimpleNamespace mirroring predict_from_pdb's get_args() defaults."""
    return SimpleNamespace(
        # Core
        ckpt_path=args.ckpt_path,
        seed=42,
        gpus=1,
        mp_index=0,
        num_workers=0,
        arch=args.arch,
        pocket_type=args.pocket_type,
        pocket_noise=args.pocket_noise,
        cut_pocket=args.cut_pocket,
        pocket_cutoff=args.pocket_cutoff,
        max_pocket_size=1000,
        min_pocket_size=10,
        save_dir=None,
        save_file=None,
        # Sampling
        coord_noise_scale=args.coord_noise_scale,
        pocket_coord_noise_std=0.0,
        sample_mol_sizes=False,
        corrector_iters=0,
        rotation_alignment=False,
        permutation_alignment=False,
        batch_cost=args.batch_cost,
        ligand_time=None,
        pocket_time=None,
        interaction_time=None,
        fixed_interactions=False,
        interaction_conditional=False,
        scaffold_hopping=False,
        scaffold_elaboration=False,
        linker_inpainting=False,
        anisotropic_prior=False,
        ref_ligand_com_prior=False,
        ref_ligand_com_noise_std=1.0,
        fragment_inpainting=False,
        fragment_growing=False,
        max_fragment_cuts=3,
        core_inpainting=False,
        substructure_inpainting=False,
        substructure=None,
        core_growing=False,
        ring_system_indexing=0,
        graph_inpainting=None,
        separate_pocket_interpolation=False,
        separate_interaction_interpolation=False,
        integration_steps=DEFAULT_INTEGRATION_STEPS,
        cat_sampling_noise_level=DEFAULT_CAT_SAMPLING_NOISE_LEVEL,
        ode_sampling_strategy=DEFAULT_ODE_SAMPLING_STRATEGY,
        solver="euler",
        categorical_strategy=DEFAULT_CATEGORICAL_STRATEGY,
        use_sde_simulation=False,
        use_cosine_scheduler=False,
        bucket_cost_scale=DEFAULT_BUCKET_COST_SCALE,
        # IO
        pdb_id=None,
        ligand_id=None,
        pdb_file=None,
        ligand_file=None,
        multiple_ligands=False,
        res_txt_file=None,
        data_path=None,
        splits_path=None,
        dataset=None,
        # Prep flags off (per ablation results — none help)
        protonate_pocket=False,
        compute_interactions=False,
        compute_interaction_recovery=False,
        add_hs=False,
        add_hs_and_optimize=False,
        add_hs_and_optimize_gen_ligs=False,
        kekulize=False,
        use_pdbfixer=False,
        add_bonds_to_protein=False,
        add_hs_to_protein=False,
        lora_finetuned=False,
        canonicalize_conformer=False,
    )


def parse_aff_from_sdf(sdf_path, keys):
    """Read 4 affinity head props from gen_lig_with_aff.sdf and return median.

    Drops NaN/missing heads. Returns None if all 4 are absent or unreadable.
    """
    if not os.path.exists(sdf_path) or os.path.getsize(sdf_path) == 0:
        return None
    try:
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
        for mol in suppl:
            if mol is None:
                continue
            vals = []
            for k in keys:
                if not mol.HasProp(k):
                    continue
                try:
                    v = float(mol.GetProp(k))
                except (ValueError, TypeError):
                    continue
                if math.isnan(v) or math.isinf(v):
                    continue
                vals.append(v)
            if not vals:
                return None
            return statistics.median(vals)
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True,
                        help="CSV with columns: id, pdb_path, sdf_path")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--sandbox_dir", required=True)
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--seeds", default="2,42,512,1000,7777")
    parser.add_argument("--coord_noise_scale", type=float, default=0.1)
    parser.add_argument("--pocket_cutoff", type=float, default=7.0)
    parser.add_argument("--batch_cost", type=int, default=20)
    parser.add_argument("--arch", default="pocket")
    parser.add_argument("--pocket_type", default="holo")
    parser.add_argument("--pocket_noise", default="fix")
    parser.add_argument("--cut_pocket", action="store_true", default=True)
    parser.add_argument("--save_every", type=int, default=100,
                        help="Write partial predictions.csv every N targets")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    base_args = make_args(args)

    torch.set_float32_matmul_precision("high")

    log.info(f"Loading model from {args.ckpt_path}...")
    t0 = time.time()
    (model, hparams, vocab, vocab_charges, vocab_hybridization,
     vocab_aromatic, _, _) = load_model(base_args)
    model = model.to("cuda")
    model.eval()
    log.info(f"Model loaded in {time.time() - t0:.1f}s")

    transform, interpolant = load_util(
        base_args, hparams, vocab, vocab_charges,
        vocab_hybridization, vocab_aromatic,
    )
    log.info("Transform and interpolant ready")

    rows = []
    with open(args.manifest_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    log.info(f"Manifest: {len(rows)} targets")

    seeds = [int(s) for s in args.seeds.split(",")]
    log.info(f"Seeds: {seeds}")

    affinity_keys = ["pic50", "pki", "pkd", "pec50"]

    predictions = []
    failures = []
    t_start = time.time()

    os.makedirs(os.path.dirname(args.predictions_csv) or ".", exist_ok=True)

    for i, row in enumerate(rows, 1):
        tid = row["id"]
        pdb_path = row["pdb_path"]
        sdf_path = row["sdf_path"]

        seed_preds = []
        for seed in seeds:
            seed_dir = os.path.join(args.sandbox_dir, tid, f"seed_{seed}")
            seed_sdf_out = os.path.join(seed_dir, "gen_lig_with_aff.sdf")

            cached = parse_aff_from_sdf(seed_sdf_out, affinity_keys)
            if cached is not None:
                seed_preds.append(cached)
                continue

            os.makedirs(seed_dir, exist_ok=True)

            base_args.pdb_file = pdb_path
            base_args.ligand_file = sdf_path
            base_args.save_dir = seed_dir
            base_args.seed = seed

            try:
                system = load_data_from_pdb(
                    base_args,
                    remove_hs=hparams["remove_hs"],
                    remove_aromaticity=hparams["remove_aromaticity"],
                )
                dataset = GeometricDataset(
                    PocketComplexBatch([system]),
                    data_cls=PocketComplexBatch,
                    transform=transform,
                )
                dataloader = get_dataloader(base_args, dataset, interpolant)
                batch = next(iter(dataloader))
                gen_ligs = predict_affinity_batch(
                    base_args,
                    model=model,
                    prior=batch[0],
                    posterior=batch[1],
                    noise_scale=args.coord_noise_scale,
                    eps=1e-4,
                    seed=seed,
                )
                smolRD.write_sdf_file(seed_sdf_out, gen_ligs, name=False)
                pred = parse_aff_from_sdf(seed_sdf_out, affinity_keys)
                if pred is not None:
                    seed_preds.append(pred)
            except Exception as e:
                log.warning(f"{tid} seed={seed}: {type(e).__name__}: {e}")

        if seed_preds:
            final = statistics.median(seed_preds)
            predictions.append((final, tid))
            if i <= 5 or i % 25 == 0:
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                eta_s = (len(rows) - i) / rate if rate > 0 else 0
                log.info(
                    f"[{i}/{len(rows)}] {tid}: median={final:.3f} "
                    f"({len(seed_preds)}/{len(seeds)} seeds) "
                    f"rate={rate:.2f}/s ETA={eta_s/3600:.1f}h"
                )
        else:
            failures.append((tid, "all_seeds_failed"))
            log.warning(f"[{i}/{len(rows)}] {tid}: ALL seeds failed")

        if i % args.save_every == 0:
            with open(args.predictions_csv, "w") as f:
                for pred, name in predictions:
                    f.write(f"{pred},{name}\n")

    with open(args.predictions_csv, "w") as f:
        for pred, name in predictions:
            f.write(f"{pred},{name}\n")
    log.info(f"Wrote {len(predictions)} predictions to {args.predictions_csv}")

    if failures:
        fail_file = args.predictions_csv.replace(
            "predictions.csv", "failures.csv"
        )
        with open(fail_file, "w") as f:
            f.write("TargetID,ErrorType\n")
            for name, err in failures:
                f.write(f"{name},{err}\n")
        log.info(f"{len(failures)} failures recorded")


if __name__ == "__main__":
    main()
