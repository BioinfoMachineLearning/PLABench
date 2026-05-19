"""Query RCSB PDB + UniProt + ChEMBL REST APIs to check the biological reality
(oligomeric state, cofactors, ligands, sequence coverage) of protein targets
used in a benchmark.

Purpose: benchmarks often use a single chain of a protein as input, but in
nature many proteins are multimers, need cofactors, or bind co-ligands. The
PDB structure referenced may also be only a truncated domain or bound to a
crystallization additive rather than the actual target ligand. This script
produces a per-target biological-reality report that can be compared against
per-target benchmark performance to quantify single-chain bias.

Data sources (with retry/backoff):
  * UniProt (rest.uniprot.org): SUBUNIT + COFACTOR free-text, structured
    cofactor list, PDB cross-references with resolution/method/chain-range,
    full-length sequence length.
  * RCSB PDB (data.rcsb.org):
      - biological assembly: oligomer count, symmetry
      - polymer entity: actual chain length (for coverage sanity check)
      - non-polymer entity: actual ligand comp_id + name (for Apo/Holo +
        crystallization additive detection).
  * ChEMBL (www.ebi.ac.uk/chembl): target_type (informational only — ChEMBL
    'SINGLE PROTEIN' means one sequence, NOT a biological monomer).

Improvements over v1 (per user feedback, 2026-04-23):
  1. Exponential-backoff retry on 429/5xx; 0.2s polite delay between calls
     within one target.
  2. Smart PDB selection when no PDB given: rank UniProt xrefs by
     (method X-ray > EM > NMR > model), then resolution, then coverage.
  3. Sequence coverage ratio: parse UniProt xref 'Chains=X=start-end' and
     divide covered residues by UniProt full-length.
  4. Actual ligand comp_id + name fetched via PDB non-polymer entity API;
     common crystallization additives flagged in a separate column so the
     report distinguishes Apo / Holo with additive / Holo with real ligand.

Input:
    CSV. Auto-detects columns:
      UniProt: UniProt_ID / uniprot / accession
      PDB:     PDB_ID / pdb_id / pdb  (optional; falls back to best xref)
      Name:    Target_name / name     (optional)

Usage:
    python scripts/check_biological_reality.py \\
        --input data/chembl35/Data_S1_target_aaseq.csv \\
        --output results/chembl35_biological_reality.csv

    # Subset
    python scripts/check_biological_reality.py \\
        --input data/chembl35/Data_S1_target_aaseq.csv \\
        --uniprots Q12809,P08183,O60341 \\
        --output /tmp/test.csv
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


UNIPROT_COLS = ["UniProt_ID", "uniprot", "accession", "UniProt", "Target ID"]
PDB_COLS = ["PDB_ID", "pdb_id", "pdb", "PDB"]
NAME_COLS = ["Target_name", "name", "Target_Name"]

OLIGOMER_KEYWORDS = (r"monomer|homodimer|homotrimer|homotetramer|homopentamer|"
                     r"homohexamer|heterodimer|heterotrimer|tetramer|trimer|"
                     r"dimer|pentamer|hexamer|octamer|nonamer|decamer|"
                     r"oligomer|multimer|complex")

# Common PDB crystallization additives / buffers / cryoprotectants / glycans.
# Present ligands outside this set are "interesting" (cofactor or target ligand).
CRYSTALLIZATION_ADDITIVES = {
    # polyols / cryoprotectants
    "GOL", "EDO", "PEG", "PGE", "P6G", "PG4", "PG0", "1PE", "2PE", "MPD", "MRD",
    "BOG", "LMT", "OLC", "OLA", "DDQ", "DMU", "C14",
    # small organics / acids
    "MLI", "CIT", "FMT", "ACE", "ACT", "TRS", "BCT", "IMD", "EPE", "MES", "TAU",
    "BTB", "HEP", "TLA", "MAL", "MLA",
    # anions / salts
    "SO4", "PO4", "NO3", "CLO", "BR", "IOD", "F",
    # small common ions (keep divalent metals like Mg/Ca/Zn separate — they're
    # often catalytic cofactors; handled below)
    "CL", "NA", "K", "HOH", "DOD",
    # reducing / stabilizing agents
    "DMS", "BME", "DTT", "TCE", "DTE", "GSH",
    # glycans / sugars
    "NAG", "MAN", "BMA", "FUC", "GAL", "GLC", "NDG", "BGC", "XYL", "SIA",
}

# Divalent metals that may be either catalytic cofactors OR additives — flag
# but don't auto-exclude. Downstream interpretation is needed.
COMMON_METAL_COFACTORS = {"MG", "CA", "ZN", "MN", "FE", "CU", "NI", "CO"}

HTTP_TIMEOUT = 30
MAX_RETRIES = 3
PER_CALL_DELAY = 0.2  # seconds between sequential API calls within a target


def fetch_json(url, retries=MAX_RETRIES):
    """GET JSON with retry on 429/5xx. Returns dict or raises after N retries."""
    backoff = 1.0
    last_exc = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code == 404:
                raise  # don't retry 404
            if e.code == 429 or 500 <= e.code < 600:
                # jittered backoff
                sleep = backoff + random.uniform(0, 0.5)
                log.debug(f"HTTP {e.code} on {url}, retry {attempt+1}/{retries} in {sleep:.1f}s")
                time.sleep(sleep)
                backoff *= 2
                continue
            raise
        except urllib.error.URLError as e:
            last_exc = e
            sleep = backoff + random.uniform(0, 0.5)
            log.debug(f"URLError on {url}: {e}; retry {attempt+1}/{retries} in {sleep:.1f}s")
            time.sleep(sleep)
            backoff *= 2
    raise last_exc if last_exc else RuntimeError("unknown fetch error")


# ---------------- UniProt ----------------

def _score_pdb_xref(xref):
    """Score a UniProt PDB xref for suitability as a reference structure.

    Higher score = more preferred. Criteria (in order):
      1. Method: X-ray > EM > NMR > Model
      2. Resolution: lower is better (we negate and cap)
      3. Coverage: more covered residues is better
    """
    props = {p["key"]: p["value"] for p in xref.get("properties", []) or []}
    method = (props.get("Method") or "").upper()
    method_score = {"X-RAY": 3, "EM": 2, "ELECTRON MICROSCOPY": 2, "NMR": 1}.get(method, 0)

    # Resolution — parse "3.10 A" -> 3.10 (lower is better)
    res_str = (props.get("Resolution") or "").replace("A", "").replace("Å", "").strip()
    try:
        resolution = float(res_str.split()[0])
    except (ValueError, IndexError):
        resolution = 99.0

    # Coverage — parse "A/B/C=1-350" -> 350 residues covered
    chains_str = props.get("Chains", "")
    covered = 0
    m = re.search(r"=(\d+)-(\d+)", chains_str)
    if m:
        covered = int(m.group(2)) - int(m.group(1)) + 1
    return (method_score, -resolution, covered, chains_str)


def query_uniprot(uniprot_id):
    """Return SUBUNIT text, cofactors, sequence length, and scored PDB xrefs."""
    url = (f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
           f"?fields=cc_subunit,cc_cofactor,xref_pdb,sequence,length")
    try:
        data = fetch_json(url)
    except Exception as e:
        return {"uniprot_error": f"{type(e).__name__}: {str(e)[:100]}"}

    out = {}
    subunit_txt = ""
    cofactors = []
    for c in data.get("comments", []) or []:
        if c.get("commentType") == "SUBUNIT":
            for t in c.get("texts", []) or []:
                subunit_txt += (t.get("value") or "") + " "
        elif c.get("commentType") == "COFACTOR":
            for cof in c.get("cofactors", []) or []:
                cofactors.append(cof.get("name", ""))
    kw = re.search(OLIGOMER_KEYWORDS, subunit_txt, re.IGNORECASE)
    out["uniprot_oligomer_kw"] = kw.group(0).lower() if kw else ""
    out["uniprot_cofactors"] = "; ".join(cofactors)
    out["uniprot_subunit"] = subunit_txt.strip().replace("\n", " ")[:500]

    # Full-length sequence for coverage denominator
    seq_info = data.get("sequence") or {}
    out["uniprot_length"] = seq_info.get("length") or 0

    # Rank PDB xrefs
    ranked = []
    for xref in data.get("uniProtKBCrossReferences", []) or []:
        if xref.get("database") != "PDB":
            continue
        score = _score_pdb_xref(xref)
        props = {p["key"]: p["value"] for p in xref.get("properties", []) or []}
        ranked.append({
            "id": xref.get("id"),
            "method": props.get("Method", ""),
            "resolution": props.get("Resolution", ""),
            "chains": props.get("Chains", ""),
            "score": score,
        })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    out["uniprot_pdb_xrefs_ranked"] = ranked
    # CSV-friendly: top-3 summary for reporting
    out["uniprot_pdb_top3"] = " | ".join(
        f"{r['id']}({r['method']},{r['resolution']})" for r in ranked[:3])
    return out


def coverage_from_xref(xref, uniprot_length):
    """Compute coverage ratio from a UniProt PDB xref using its 'Chains' field."""
    if not uniprot_length:
        return None
    chains_str = xref.get("chains", "") if isinstance(xref, dict) else ""
    # Parse "A/B=1-350" or "A=1-350, B=400-500" etc.
    covered_positions = set()
    for part in chains_str.split(","):
        m = re.search(r"=(\d+)-(\d+)", part)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            covered_positions.update(range(start, end + 1))
    if not covered_positions:
        return None
    return round(len(covered_positions) / uniprot_length, 3)


# ---------------- RCSB PDB ----------------

def query_rcsb_entry(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    try:
        data = fetch_json(url)
    except Exception as e:
        return {"rcsb_entry_error": f"{type(e).__name__}: {str(e)[:100]}"}
    info = data.get("rcsb_entry_info", {}) or {}
    exp = (data.get("exptl") or [{}])[0]
    container = data.get("rcsb_entry_container_identifiers", {}) or {}
    return {
        "rcsb_polymer_composition": info.get("polymer_composition"),
        "rcsb_deposited_chains": info.get("deposited_polymer_entity_instance_count"),
        "rcsb_nonpolymer_entity_ids": info.get("nonpolymer_entity_ids") or [],
        "rcsb_polymer_entity_ids": container.get("polymer_entity_ids") or [],
        "rcsb_experimental_method": exp.get("method"),
        "rcsb_resolution": info.get("resolution_combined", [None])[0],
    }


def query_rcsb_assembly(pdb_id, assembly_id=1):
    url = f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_id.lower()}/{assembly_id}"
    try:
        data = fetch_json(url)
    except Exception as e:
        return {"rcsb_assembly_error": f"{type(e).__name__}: {str(e)[:100]}"}
    asm = data.get("pdbx_struct_assembly", {}) or {}
    out = {
        "oligomer_count": asm.get("oligomeric_count"),
        "oligomer_details": asm.get("oligomeric_details"),
    }
    sym = data.get("rcsb_struct_symmetry") or []
    if sym:
        out["symmetry_symbol"] = sym[0].get("symbol")
        out["symmetry_kind"] = sym[0].get("kind")
    return out


def query_rcsb_polymer_entity(pdb_id, entity_id):
    """Fetch polymer entity info: chain length + UniProt mapping.

    Returns: {"length": int, "uniprot_ids": [str]} or {"error": str}.
    """
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.lower()}/{entity_id}"
    try:
        data = fetch_json(url)
    except Exception as e:
        return {"error": str(e)[:80]}
    poly = data.get("entity_poly", {}) or {}
    length = poly.get("rcsb_sample_sequence_length")
    # Reference sequence identifiers → list of UniProt accessions
    container = data.get("rcsb_polymer_entity_container_identifiers", {}) or {}
    refs = container.get("reference_sequence_identifiers") or []
    uniprots = [r.get("database_accession") for r in refs if r.get("database_name") == "UniProt"]
    return {"length": length, "uniprot_ids": [u for u in uniprots if u]}


def query_rcsb_nonpolymer(pdb_id, entity_id):
    """Fetch actual ligand comp_id (3-letter) and name for one nonpolymer entity."""
    url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id.lower()}/{entity_id}"
    try:
        data = fetch_json(url)
    except Exception:
        return None
    np = data.get("pdbx_entity_nonpoly") or {}
    desc = (data.get("rcsb_nonpolymer_entity") or {}).get("pdbx_description") or ""
    return {
        "comp_id": np.get("comp_id") or "",
        "name": np.get("name") or desc,
    }


def classify_ligand(comp_id):
    """Return one of: 'additive', 'metal_cofactor_or_additive', 'ligand_or_cofactor'."""
    cid = (comp_id or "").upper()
    if cid in CRYSTALLIZATION_ADDITIVES:
        return "additive"
    if cid in COMMON_METAL_COFACTORS:
        return "metal_cofactor_or_additive"
    return "ligand_or_cofactor"


# ---------------- ChEMBL ----------------

def query_chembl(uniprot_id):
    url = (f"https://www.ebi.ac.uk/chembl/api/data/target.json"
           f"?target_components__accession={uniprot_id}&limit=5")
    try:
        data = fetch_json(url)
    except Exception as e:
        return {"chembl_error": f"{type(e).__name__}: {str(e)[:100]}"}
    entries = [f"{t.get('target_chembl_id')}:{t.get('target_type')}"
               for t in data.get("targets", [])]
    return {"chembl_targets": " | ".join(entries)}


# ---------------- Orchestration ----------------

def process_one(uniprot_id, explicit_pdb=None, name=None):
    row = {"uniprot_id": uniprot_id, "target_name": name or "",
           "pdb_id_input": explicit_pdb or ""}

    # 1) UniProt — gives us subunit, cofactor, pdb xrefs, length
    up = query_uniprot(uniprot_id)
    row.update({k: v for k, v in up.items() if k != "uniprot_pdb_xrefs_ranked"})
    time.sleep(PER_CALL_DELAY)

    ranked = up.get("uniprot_pdb_xrefs_ranked") or []
    uniprot_length = up.get("uniprot_length") or 0

    # 2) PDB selection: explicit > best-ranked xref
    chosen_xref = None
    if explicit_pdb:
        row["pdb_id"] = explicit_pdb
        # Find coverage for explicit PDB if it's in xrefs
        for r in ranked:
            if r["id"].lower() == explicit_pdb.lower():
                chosen_xref = r
                break
    elif ranked:
        chosen_xref = ranked[0]
        row["pdb_id"] = chosen_xref["id"]
    else:
        row["pdb_id"] = ""

    if chosen_xref:
        row["pdb_method"] = chosen_xref["method"]
        row["pdb_resolution_xref"] = chosen_xref["resolution"]
        row["pdb_chains_xref"] = chosen_xref["chains"]
        cov = coverage_from_xref(chosen_xref, uniprot_length)
        if cov is not None:
            row["coverage_ratio"] = cov
            row["coverage_source"] = "uniprot_xref"

    pdb_id = row.get("pdb_id")
    if pdb_id:
        entry = query_rcsb_entry(pdb_id)
        polymer_eids = entry.pop("rcsb_polymer_entity_ids", []) or []
        nonpolymer_eids = entry.pop("rcsb_nonpolymer_entity_ids", []) or []
        row.update(entry)
        time.sleep(PER_CALL_DELAY)

        # Fallback coverage: explicit PDB not in UniProt xrefs (e.g., too
        # new). Walk polymer entities, find the one mapped to this UniProt.
        if row.get("coverage_ratio") is None and uniprot_length and polymer_eids:
            for eid in polymer_eids:
                info = query_rcsb_polymer_entity(pdb_id, eid)
                time.sleep(PER_CALL_DELAY)
                if info.get("error"):
                    continue
                if uniprot_id in (info.get("uniprot_ids") or []):
                    length = info.get("length") or 0
                    if length:
                        row["coverage_ratio"] = round(length / uniprot_length, 3)
                        row["pdb_chain_length"] = length
                        row["coverage_source"] = "rcsb_entity"
                        break
            if row.get("coverage_ratio") is None:
                row["coverage_source"] = "unresolved"

        row.update(query_rcsb_assembly(pdb_id, 1))
        time.sleep(PER_CALL_DELAY)

        # Fetch actual ligand comp_ids + names for each nonpolymer entity
        ligands = []
        for eid in nonpolymer_eids:
            info = query_rcsb_nonpolymer(pdb_id, eid)
            time.sleep(PER_CALL_DELAY)
            if info:
                info["class"] = classify_ligand(info["comp_id"])
                ligands.append(info)
        row["pdb_ligands_all"] = "; ".join(f"{l['comp_id']}:{l['name']}" for l in ligands)
        row["pdb_ligands_additives"] = "; ".join(
            l["comp_id"] for l in ligands if l["class"] == "additive")
        row["pdb_ligands_non_additive"] = "; ".join(
            l["comp_id"] for l in ligands if l["class"] != "additive")
        row["pdb_n_ligands_non_additive"] = sum(
            1 for l in ligands if l["class"] != "additive")

    # 3) ChEMBL target_type
    row.update(query_chembl(uniprot_id))
    return row


def detect_col(fieldnames, candidates):
    for c in candidates:
        if c in fieldnames:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="CSV with UniProt_ID (and optional PDB_ID) columns")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--uniprot-col", default=None)
    parser.add_argument("--pdb-col", default=None)
    parser.add_argument("--name-col", default=None)
    parser.add_argument("--uniprots", default=None, help="Optional comma-separated subset")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4; be polite to EBI/UniProt)")
    args = parser.parse_args()

    with open(args.input) as f:
        reader = csv.DictReader(f)
        up_col = args.uniprot_col or detect_col(reader.fieldnames, UNIPROT_COLS)
        pdb_col = args.pdb_col or detect_col(reader.fieldnames, PDB_COLS)
        name_col = args.name_col or detect_col(reader.fieldnames, NAME_COLS)
        if not up_col:
            log.error(f"No UniProt column found. Fields: {reader.fieldnames}")
            sys.exit(1)
        rows = list(reader)

    seen = set()
    targets = []
    for r in rows:
        uid = r[up_col].strip()
        if not uid or uid in seen:
            continue
        seen.add(uid)
        targets.append({
            "uniprot": uid,
            "pdb": (r[pdb_col].strip() if pdb_col and r.get(pdb_col) else None),
            "name": (r[name_col].strip() if name_col and r.get(name_col) else None),
        })

    if args.uniprots:
        keep = {u.strip() for u in args.uniprots.split(",")}
        targets = [t for t in targets if t["uniprot"] in keep]

    log.info(f"Processing {len(targets)} targets with {args.workers} workers")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, t["uniprot"], t["pdb"], t["name"]): t for t in targets}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
                results.append(r)
            except Exception as e:
                t = futs[fut]
                log.warning(f"Fatal error for {t['uniprot']}: {e}")
                results.append({"uniprot_id": t["uniprot"], "target_name": t["name"] or "",
                                "uniprot_error": f"fatal:{str(e)[:100]}"})
            if i % 10 == 0:
                log.info(f"  {i}/{len(targets)} done")

    results.sort(key=lambda r: r.get("uniprot_id", ""))

    fields = [
        "uniprot_id", "target_name",
        "pdb_id_input", "pdb_id",
        "pdb_method", "pdb_resolution_xref", "pdb_chains_xref",
        "uniprot_length", "coverage_ratio", "coverage_source", "pdb_chain_length",
        "oligomer_count", "oligomer_details", "symmetry_symbol", "symmetry_kind",
        "rcsb_polymer_composition", "rcsb_deposited_chains",
        "rcsb_experimental_method", "rcsb_resolution",
        "pdb_ligands_all", "pdb_ligands_additives", "pdb_ligands_non_additive",
        "pdb_n_ligands_non_additive",
        "uniprot_oligomer_kw", "uniprot_cofactors", "uniprot_subunit",
        "uniprot_pdb_top3", "chembl_targets",
        "rcsb_entry_error", "rcsb_assembly_error", "uniprot_error", "chembl_error",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    multimeric = sum(1 for r in results if (r.get("oligomer_count") or 0) and int(r["oligomer_count"]) > 1)
    with_cofactor = sum(1 for r in results if r.get("uniprot_cofactors"))
    low_cov = sum(1 for r in results if (r.get("coverage_ratio") or 1.0) < 0.8)
    with_real_lig = sum(1 for r in results if (r.get("pdb_n_ligands_non_additive") or 0) > 0)
    errors = sum(1 for r in results if any(r.get(k) for k in
                 ("rcsb_entry_error", "rcsb_assembly_error", "uniprot_error", "chembl_error")))
    log.info(f"Saved {len(results)} rows to {args.output}")
    log.info(f"  multimeric (oligomer>1): {multimeric}")
    log.info(f"  with UniProt cofactor: {with_cofactor}")
    log.info(f"  coverage < 0.8 (truncated domain): {low_cov}")
    log.info(f"  PDB with non-additive ligand (Holo): {with_real_lig}")
    log.info(f"  with any API error: {errors}")


if __name__ == "__main__":
    main()
