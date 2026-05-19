"""PDB release-date leakage check (opt-in tool).

Verifies whether dataset entries map to PDB structures released on or after a
given cutoff date. Useful before training on dataset X and evaluating on
dataset Y — a leaky PDB in X invalidates the evaluation if Y also uses it.

NOT auto-run by dumpers. Invoke explicitly via CLI or import.

CLI:
    python -m plabench.data_dump.leakage_check \
        --cutoff 2023-06-01 \
        --csv data/spindr/full.csv:pdb_id \
        --csv data/bindingmoad_flowr/full.csv:pdb_id \
        --cache /tmp/pdb_release_dates.json \
        [--report-out leakage_report.json]

Programmatic:
    from plabench.data_dump.leakage_check import (
        fetch_release_dates, find_leaky_pdbs, check_csv
    )
"""
from __future__ import annotations
import os
import json
import time
import math
import argparse
import logging
import urllib.request
from collections import Counter
from typing import Dict, Iterable, Optional, Set

RCSB_GRAPHQL_URL = 'https://data.rcsb.org/graphql'
DEFAULT_BATCH = 200
DEFAULT_TIMEOUT = 30

log = logging.getLogger('leakage_check')


def fetch_release_dates(
    pdb_ids: Iterable[str],
    cache_path: Optional[str] = None,
    batch: int = DEFAULT_BATCH,
) -> Dict[str, Optional[str]]:
    """Query RCSB GraphQL for `initial_release_date` per PDB ID.

    Returns {pdb_id_lower: 'YYYY-MM-DDTHH:MM:SSZ' | None}. None means the PDB ID
    was reported by RCSB as obsolete/withdrawn (or the entry block was missing).

    If `cache_path` is set, an existing JSON cache is loaded first; only IDs not
    already cached are queried; the merged cache is written back at the end.
    """
    targets = {p.lower() for p in pdb_ids if p}
    cache: Dict[str, Optional[str]] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        log.info('loaded cache: %d entries from %s', len(cache), cache_path)

    todo = sorted(targets - set(cache))
    if not todo:
        log.info('all %d PDB IDs already cached', len(targets))
        return {p: cache.get(p) for p in targets}

    log.info('querying RCSB for %d new PDB IDs (batch=%d)...', len(todo), batch)
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        ids_str = '","'.join(p.upper() for p in chunk)
        query = (
            f'{{ entries(entry_ids: ["{ids_str}"]) '
            '{ rcsb_id rcsb_accession_info { initial_release_date } } }'
        )
        payload = json.dumps({'query': query}).encode()
        req = urllib.request.Request(
            RCSB_GRAPHQL_URL, data=payload,
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
                data = json.loads(resp.read())
        except Exception as ex:
            log.error('  batch %d-%d ERROR: %s', i, i + batch, ex)
            continue
        entries = (data.get('data') or {}).get('entries') or []
        # mark all chunk ids as fetched (None if missing from response)
        seen = set()
        for e in entries:
            if not e:
                continue
            pdb = e['rcsb_id'].lower()
            date = (e.get('rcsb_accession_info') or {}).get('initial_release_date')
            cache[pdb] = date
            seen.add(pdb)
        for p in chunk:
            if p not in seen:
                cache[p] = None     # obsolete / withdrawn
        if (i // batch + 1) % 10 == 0:
            log.info('  %d/%d  (%.0fs)', i + len(chunk), len(todo), time.time() - t0)

    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
        log.info('cache → %s (%d total entries)', cache_path, len(cache))
    log.info('fetch done in %.0fs', time.time() - t0)
    return {p: cache.get(p) for p in targets}


def find_leaky_pdbs(pdb_to_date: Dict[str, Optional[str]], cutoff: str) -> Set[str]:
    """Return set of PDB IDs whose release_date >= `cutoff` (ISO date string).

    PDBs with date == None (unresolved / obsolete) are conservatively NOT
    flagged as leaky; caller should inspect them separately if strict.
    """
    return {p for p, d in pdb_to_date.items() if d and d >= cutoff}


def check_csv(
    csv_path: str,
    pdb_id_col: str,
    cutoff: str,
    pdb_to_date: Dict[str, Optional[str]],
) -> Dict:
    """Apply leakage check to one CSV file.

    Returns:
        {
          'csv_path': ..., 'total_rows': N, 'leaky_rows': K,
          'leaky_pct': K/N, 'unresolved_pdbs': M,
          'by_split': {'train': ..., 'val': ..., 'test': ...} (only if 'split' col exists),
          'leaky_examples': [system_id, ...] (up to 10),
        }
    """
    import pandas as pd
    leaky = find_leaky_pdbs(pdb_to_date, cutoff)
    cols = ['split', 'system_id'] if pdb_id_col != 'split' else []
    use = list({pdb_id_col, *cols})
    df = pd.read_csv(csv_path, usecols=lambda c: c in use)
    df[pdb_id_col] = df[pdb_id_col].astype(str).str.lower()
    is_leaky = df[pdb_id_col].isin(leaky)
    n_unresolved = df[~df[pdb_id_col].isin(set(pdb_to_date))].shape[0]
    out = {
        'csv_path':         csv_path,
        'pdb_id_col':       pdb_id_col,
        'cutoff':           cutoff,
        'total_rows':       int(len(df)),
        'leaky_rows':       int(is_leaky.sum()),
        'leaky_pct':        float(is_leaky.mean()),
        'unresolved_pdbs':  int(n_unresolved),
        'leaky_examples':   df.loc[is_leaky, 'system_id'].head(10).tolist()
                            if 'system_id' in df.columns else [],
    }
    if 'split' in df.columns:
        out['by_split'] = (
            df.loc[is_leaky, 'split'].value_counts().to_dict()
            if is_leaky.any() else {}
        )
    return out


# ── CLI ──────────────────────────────────────────────────────────────

def _parse_csv_arg(s: str):
    if ':' not in s:
        raise argparse.ArgumentTypeError('--csv expects "path:column", got: ' + s)
    path, col = s.rsplit(':', 1)
    return path, col


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(name)s] %(message)s')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cutoff', required=True,
                   help='ISO date, e.g. 2023-06-01. PDBs released >= this date count as leaky.')
    p.add_argument('--csv', action='append', required=True, type=_parse_csv_arg,
                   metavar='PATH:COLUMN',
                   help='CSV file + column containing pdb_id (repeatable).')
    p.add_argument('--cache', default=None,
                   help='JSON cache for RCSB queries (re-used across runs).')
    p.add_argument('--report-out', default=None,
                   help='Write per-CSV leakage report to this JSON file.')
    args = p.parse_args()

    import pandas as pd
    all_pdbs: Set[str] = set()
    for path, col in args.csv:
        df = pd.read_csv(path, usecols=[col])
        all_pdbs |= set(df[col].dropna().astype(str).str.lower())
    log.info('union of unique PDB IDs across %d CSVs: %d', len(args.csv), len(all_pdbs))

    pdb_to_date = fetch_release_dates(all_pdbs, cache_path=args.cache)

    reports = []
    for path, col in args.csv:
        r = check_csv(path, col, args.cutoff, pdb_to_date)
        reports.append(r)
        print('-' * 60)
        print(f'CSV: {path}  (col={col})')
        print(f'  total: {r["total_rows"]}, leaky: {r["leaky_rows"]} ({100*r["leaky_pct"]:.2f}%), '
              f'unresolved pdbs: {r["unresolved_pdbs"]}')
        if r.get('by_split'):
            print(f'  by split: {r["by_split"]}')
        if r['leaky_examples']:
            print(f'  examples: {r["leaky_examples"][:5]}')

    if args.report_out:
        with open(args.report_out, 'w') as f:
            json.dump({
                'cutoff': args.cutoff,
                'reports': reports,
            }, f, indent=2)
        print(f'\nreport → {args.report_out}')


if __name__ == '__main__':
    main()
