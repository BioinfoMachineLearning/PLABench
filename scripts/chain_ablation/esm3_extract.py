"""Chain-rule ablation, step 2: ESM3-open-small per-residue embeddings for the
concatenated protein strings.

Identical model, precision and call signature to scripts/prepare_refined_91.py so the two
arms differ only in the input string. Sequences are sharded round-robin after sorting by
length, so each GPU receives a comparable mix of short and long work.

Out-of-memory failures are caught per sequence and written to a CSV rather than aborting
the shard -- they can be re-run afterwards on a larger card and merged in.
"""
import argparse
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["INFRA_PROVIDER"] = "local"

import pandas as pd
import torch

DATA = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation/data"
WORK = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation/emb"
GROUPS = {
    "PDBbind_Refined_91_concat": ["pdbbind_train_full_concat.csv", "pdbbind_val_full_concat.csv"],
    "CASF2016_Std_concat": ["CASF2016_Std_concat.csv"],
    "CASF2013_Std_concat": ["CASF2013_Std_concat.csv"],
    "CSAR36_Std_concat": ["CSAR36_Std_concat.csv"],
    "CSAR51_Std_concat": ["CSAR51_Std_concat.csv"],
}


def all_sequences():
    seqs = set()
    for files in GROUPS.values():
        for f in files:
            df = pd.read_csv(os.path.join(DATA, f))
            seqs.update(df.target_sequence.astype(str).tolist())
    # Sort on (length, string), never on length alone: set iteration order varies between
    # processes under string hash randomization, so a stable sort on a non-unique key gives
    # each worker a different order and the round-robin shards then overlap and leave gaps.
    # Shortest first, so a shard that dies on a long one still delivers everything before it.
    return sorted(seqs, key=lambda s: (len(s), s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--only-seqs", type=str, default=None,
                    help="pickle of a specific sequence list (for OOM re-runs)")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    if args.only_seqs:
        seqs = pd.read_pickle(args.only_seqs)
    else:
        seqs = all_sequences()
    mine = [s for i, s in enumerate(seqs) if i % args.nshards == args.shard]
    total_res = sum(len(s) for s in mine)
    print(f"[shard {args.shard}] {len(mine)} sequences, {total_res:,} residues, "
          f"longest {max(len(s) for s in mine)}", flush=True)

    # gpu = -1 runs on CPU, which is the only way to fit the sequences past ~2,400
    # residues on a 32 GB card. Precision stays fp32 either way, so the embeddings
    # remain consistent with the rest of the table.
    device = torch.device("cpu" if args.gpu < 0 else f"cuda:{args.gpu}")
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, SamplingConfig
    from esm.utils.constants.models import ESM3_OPEN_SMALL

    model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)
    model.eval()

    out, failures = {}, []
    t0 = time.time()
    for i, seq in enumerate(mine):
        try:
            t_prot = model.encode(ESMProtein(sequence=seq))
            with torch.no_grad():
                o = model.forward_and_sample(
                    t_prot, SamplingConfig(return_per_residue_embeddings=True))
            if o.per_residue_embedding is None:
                failures.append({"len": len(seq), "error": "no embedding returned",
                                 "seq": seq})
                continue
            out[seq] = o.per_residue_embedding.squeeze(0).cpu()
        except torch.cuda.OutOfMemoryError as e:
            failures.append({"len": len(seq), "error": f"OOM: {str(e)[:120]}", "seq": seq})
            torch.cuda.empty_cache()
            print(f"[shard {args.shard}] OOM at length {len(seq)}", flush=True)
        except Exception as e:  # noqa: BLE001 - record and keep going
            failures.append({"len": len(seq), "error": repr(e)[:160], "seq": seq})
            torch.cuda.empty_cache()
            print(f"[shard {args.shard}] FAIL len {len(seq)}: {repr(e)[:120]}", flush=True)
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"[shard {args.shard}] {i+1}/{len(mine)}  {el/60:.1f} min  "
                  f"eta {el/(i+1)*(len(mine)-i-1)/60:.1f} min", flush=True)

    suffix = f"{args.tag}{args.shard}"
    torch.save(out, os.path.join(WORK, f"esm3_shard{suffix}.pt"))
    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(WORK, f"failures_shard{suffix}.csv"), index=False)
    print(f"[shard {args.shard}] done: {len(out)} embedded, {len(failures)} failed, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
