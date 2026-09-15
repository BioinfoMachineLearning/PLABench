"""Turn the CASP16 / ChEMBL35 prediction dumps into the numbers the paper decision needs.

Four questions:

  1. Does the harness reproduce the paper's published rows? (If not, the gap to the concat
     arm cannot be read off against the manuscript.)
  2. How far apart are the two chain rules on these sets, against seed noise?
  3. How far is each arm from what the paper currently prints?
  4. Is the arm difference distinguishable from zero under a paired bootstrap over the unit
     the paper aggregates on -- test rows within each CASP16 series, and targets in ChEMBL35?

CASP16 and ChEMBL35 are not chain-selected: CASP16 supplies one target sequence per series,
ChEMBL35 the canonical UniProt sequence per target. Both arms therefore see byte-identical
test input, and the entire difference is the retrained weights.
"""
import os
import sys

import numpy as np
import pandas as pd

W = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation"
PLABENCH = "/bmlfast/Lyuwei/0.Projects/PLABench"
sys.path.insert(0, PLABENCH)
from plabench.analysis.metrics import calculate_metrics  # noqa: E402

SEQ = os.path.join(W, "results", "paper_sets")
RES = os.path.join(W, "results")
SEEDS = (0, 1, 2)
NBOOT = 2000
METRICS = ("RMSE", "Pearson", "Spearman", "CI")
CASP16 = {"casp16_l1000": f"{PLABENCH}/data/Structure_independent/L1000_casp16_test.csv",
          "casp16_l3000": f"{PLABENCH}/data/Structure_independent/L3000_casp16_test.csv"}
CHEMBL = f"{PLABENCH}/data/chembl35/chembl35_full_input.csv"

# what results/{casp16/weighted_summary_casp16,chembl35/weighted_summary_chembl35}.csv currently prints
PAPER = {
    ("llf", "casp16"): dict(RMSE=1.681, Pearson=0.447, Spearman=0.444, CI=0.657),
    ("deepdta", "casp16"): dict(RMSE=1.213, Pearson=0.402, Spearman=0.442, CI=0.656),
    ("meeta", "casp16"): dict(RMSE=1.029, Pearson=0.514, Spearman=0.529, CI=0.687),
    ("llf", "chembl35"): dict(RMSE=1.465, Pearson=0.194, Spearman=0.192, CI=0.566),
    ("deepdta", "chembl35"): dict(RMSE=1.471, Pearson=0.130, Spearman=0.129, CI=0.545),
    ("meeta", "chembl35"): dict(RMSE=1.287, Pearson=0.145, Spearman=0.135, CI=0.547),
}


def load(model, arm, ds):
    f = os.path.join(SEQ, model, f"{arm}_{ds}.csv")
    if not os.path.isfile(f):
        return None
    return pd.read_csv(f, header=None, names=["prediction", "name"]).prediction.values


def arm_names(model, rule):
    return [f"{rule}_L4700_s{s}" for s in SEEDS] if model != "meeta" else [
        "published" if rule == "longest" else "concat"]


def casp16_weighted(preds_by_ds, key):
    """N-weighted mean of the per-series metric, the aggregation the paper uses."""
    num = den = 0.0
    for ds, (y, p) in preds_by_ds.items():
        num += calculate_metrics(p, y)[key] * len(y)
        den += len(y)
    return num / den


def chembl_weighted(y, p, targets, key):
    """Per-target metric over targets with >= 3 compounds, then N-weighted."""
    d = pd.DataFrame({"y": y, "p": p, "t": targets})
    num = den = 0.0
    for _, g in d.groupby("t"):
        if len(g) < 3:
            continue
        v = calculate_metrics(g.p, g.y)[key]
        if isinstance(v, str) or (isinstance(v, float) and np.isnan(v)):
            continue
        num += v * len(g)
        den += len(g)
    return num / den if den else float("nan")


def gather(model):
    """Truth + one averaged prediction per arm, for both keys."""
    out = {}
    c16 = {}
    for ds, f in CASP16.items():
        y = pd.read_csv(f).affinity.values
        cols = {}
        for rule in ("longest", "concat"):
            ps = [load(model, a, ds) for a in arm_names(model, rule)]
            ps = [p for p in ps if p is not None]
            if not ps:
                return None
            cols[rule] = np.mean(ps, axis=0)
            cols[f"{rule}_seeds"] = ps
        pub = load(model, "published", ds)
        c16[ds] = (y, cols, pub)
    out["casp16"] = c16

    d = pd.read_csv(CHEMBL)
    y = d.affinity.values
    tg = d.compound_id.astype(str).str.split("_").str[0].values
    cols = {}
    for rule in ("longest", "concat"):
        ps = [load(model, a, CHEMBL_DS) for a in arm_names(model, rule)]
        ps = [p for p in ps if p is not None]
        if not ps:
            return None
        cols[rule] = np.mean(ps, axis=0)
        cols[f"{rule}_seeds"] = ps
    out["chembl35"] = (y, tg, cols, load(model, "published", CHEMBL_DS))
    return out


CHEMBL_DS = "chembl35_full"


def section_summary(data):
    print("=" * 100)
    print("1-3. ARMS vs THE PUBLISHED ROW  (mean over seeds; sd in parens where >1 seed)")
    print("=" * 100)
    print(f"{'model':8s} {'set':9s} {'metric':9s} {'paper':>8s} {'repro':>8s} "
          f"{'longest':>16s} {'concat':>16s} {'concat-paper':>13s}")
    rows = []
    for model, d in data.items():
        for key in ("casp16", "chembl35"):
            for met in METRICS:
                if key == "casp16":
                    def val(sel):
                        return casp16_weighted(
                            {ds: (y, (c[sel] if isinstance(sel, str) else sel[ds]))
                             for ds, (y, c, _) in d["casp16"].items()}, met)

                    def seed_vals(rule):
                        n = len(d["casp16"]["casp16_l1000"][1][f"{rule}_seeds"])
                        return [casp16_weighted(
                            {ds: (y, c[f"{rule}_seeds"][i])
                             for ds, (y, c, _) in d["casp16"].items()}, met)
                            for i in range(n)]

                    repro = casp16_weighted(
                        {ds: (y, pub) for ds, (y, _, pub) in d["casp16"].items()}, met)
                else:
                    y, tg, c, pub = d["chembl35"]

                    def seed_vals(rule, y=y, tg=tg, c=c):
                        return [chembl_weighted(y, p, tg, met)
                                for p in c[f"{rule}_seeds"]]

                    repro = chembl_weighted(y, pub, tg, met)

                lv, cv = seed_vals("longest"), seed_vals("concat")
                paper = PAPER[(model, key)][met]
                fl = (f"{np.mean(lv):.3f} ({np.std(lv, ddof=1):.3f})" if len(lv) > 1
                      else f"{np.mean(lv):.3f}")
                fc = (f"{np.mean(cv):.3f} ({np.std(cv, ddof=1):.3f})" if len(cv) > 1
                      else f"{np.mean(cv):.3f}")
                print(f"{model:8s} {key:9s} {met:9s} {paper:8.3f} {repro:8.3f} "
                      f"{fl:>16s} {fc:>16s} {np.mean(cv) - paper:+13.3f}")
                rows.append({"model": model, "set": key, "metric": met, "paper": paper,
                             "reproduced_published": round(repro, 3),
                             "longest_mean": round(float(np.mean(lv)), 3),
                             "longest_sd": round(float(np.std(lv, ddof=1)), 3) if len(lv) > 1 else None,
                             "concat_mean": round(float(np.mean(cv)), 3),
                             "concat_sd": round(float(np.std(cv, ddof=1)), 3) if len(cv) > 1 else None,
                             "concat_minus_paper": round(float(np.mean(cv) - paper), 3),
                             "n_seeds": len(cv)})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "paper_sets_summary.csv"), index=False)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def boot_casp16(d, rng, n):
    """Resample rows inside each series; both arms see the same draw."""
    series = [(y, c["longest"], c["concat"]) for _, (y, c, _) in d["casp16"].items()]
    out = []
    for rep in range(n + 1):
        acc = {m: [0.0, 0.0] for m in METRICS}
        den = 0.0
        for y, pl, pc in series:
            i = np.arange(len(y)) if rep == 0 else rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) < 3:
                break
            ml, mc = calculate_metrics(pl[i], y[i]), calculate_metrics(pc[i], y[i])
            for m in METRICS:
                a, b = _num(ml[m]), _num(mc[m])
                if a is None or b is None:
                    break
                acc[m][0] += a * len(y); acc[m][1] += b * len(y)
            den += len(y)
        else:
            out.append({m: (acc[m][0] / den, acc[m][1] / den) for m in METRICS})
    return out[0], out[1:]


def boot_chembl(d, rng, n):
    """Resample the 84 targets. Per-target metrics are fixed under this resampling,
    so score each target once and let the bootstrap only re-weight them."""
    y, tg, c, _ = d["chembl35"]
    df = pd.DataFrame({"y": y, "l": c["longest"], "c": c["concat"], "t": tg})
    ns, vals = [], {m: ([], []) for m in METRICS}
    for _, g in df.groupby("t"):
        if len(g) < 3:
            continue
        ml, mc = calculate_metrics(g.l, g.y), calculate_metrics(g.c, g.y)
        ns.append(len(g))
        for m in METRICS:
            a, b = _num(ml[m]), _num(mc[m])
            vals[m][0].append(np.nan if a is None else a)
            vals[m][1].append(np.nan if b is None else b)
    ns = np.asarray(ns, float)
    arr = {m: (np.asarray(vals[m][0]), np.asarray(vals[m][1])) for m in METRICS}

    def agg(idx):
        w = ns[idx]
        out = {}
        for m in METRICS:
            a, b = arr[m][0][idx], arr[m][1][idx]
            ok = ~(np.isnan(a) | np.isnan(b))
            out[m] = ((a[ok] * w[ok]).sum() / w[ok].sum(),
                      (b[ok] * w[ok]).sum() / w[ok].sum())
        return out

    full = np.arange(len(ns))
    return agg(full), [agg(rng.integers(0, len(ns), len(ns))) for _ in range(n)]


def section_paired(data):
    print("\n" + "=" * 100)
    print("4. CHAIN RULE, seed-ensembled, paired bootstrap over the paper's aggregation unit")
    print("   (CASP16: rows within each series; ChEMBL35: the 84 targets)")
    print("=" * 100)
    print(f"{'model':8s} {'set':9s} {'metric':9s} {'longest':>8s} {'concat':>8s} "
          f"{'diff':>7s} {'95% CI':>18s} {'p':>7s}", flush=True)
    rows = []
    for model, d in data.items():
        for key in ("casp16", "chembl35"):
            rng = np.random.default_rng(0)
            fn = boot_casp16 if key == "casp16" else boot_chembl
            point, reps = fn(d, rng, NBOOT)
            for met in METRICS:
                pl, pc = point[met]
                ds_ = np.array([r[met][1] - r[met][0] for r in reps])
                lo, hi = np.percentile(ds_, [2.5, 97.5])
                p = min(1.0, 2 * min((ds_ <= 0).mean(), (ds_ >= 0).mean()))
                sig = "" if lo <= 0 <= hi else "  *"
                print(f"{model:8s} {key:9s} {met:9s} {pl:8.3f} {pc:8.3f} "
                      f"{pc - pl:+7.3f} [{lo:+.3f},{hi:+.3f}] {p:7.3f}{sig}", flush=True)
                rows.append({"model": model, "set": key, "metric": met,
                             "longest": round(pl, 3), "concat": round(pc, 3),
                             "diff": round(pc - pl, 3), "lo": round(lo, 3),
                             "hi": round(hi, 3), "p": round(p, 3), "n_boot": len(ds_)})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "paper_sets_chain_rule.csv"), index=False)
    print("\n*  = 95% CI on the paired difference excludes zero")


def main():
    data = {}
    for model in ("llf", "deepdta", "meeta"):
        g = gather(model)
        if g is None:
            print(f"skip {model}: predictions incomplete")
            continue
        data[model] = g
    section_summary(data)
    section_paired(data)
    print(f"\nwrote {RES}/paper_sets_{{summary,chain_rule}}.csv")


if __name__ == "__main__":
    sys.exit(main())
