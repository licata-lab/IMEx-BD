#!/usr/bin/env python3
"""
map_interpro_on_bd.py
=====================

Step 2 of the binding-domain pipeline. Script version of
02_get_api_interpromapping.ipynb + 02_MappingINTERPROonBD.ipynb.

Starting from the interactions with >=1 annotated binding region (output of
count_bindingdomain_imex.py, e.g. per_species/human_bindingdomain.csv) it:

  1. splits every interaction into single interactors (role ftA / ftB), keeps the
     target-species ones and parses binding regions into type / start / end;
  2. retrieves InterPro annotations from the InterPro REST API:
       - isoforms (P12345-2)   -> /protein/uniprot/<canonical>/?isoforms=<isoform>
       - canonicals (P12345)   -> /entry/interpro/protein/uniprot/<acc>/  (paginated)
     every response is cached as JSON, so re-runs only query what is missing;
  3. maps InterPro entries onto the binding regions (overlap of ranges on the same
     protein) and computes lengths, overlap and percentages;
  4. writes the full mapped dataset, the 100x100 dataset and, if requested, a
     dataset with custom filters.

Usage
-----
    python map_interpro_on_bd.py \
        --input  imex_2025_script/per_species/human_bindingdomain.csv \
        --taxid  9606 \
        --cache-dir interpro_api_cache \
        --outdir INTERPROtoBD_mapping
humanbindingdomain.csv output file from the 1 step
    # + custom filtered dataset (any combination of filters)
    ... --max-length-diff 500 --min-perc-domain 50 --feature-types "sufficient binding region"

    # quick test on 20 proteins / offline re-run from cache only
    ... --limit 20
    ... --no-api

Main options
------------
    --taxid T [T ...]        taxid(s) of the target species (default 9606)
    --interpro-types ...     InterPro entry types used for the mapping (default: domain)
    --workers N              parallel API requests (default 4; be kind to EBI)
    --no-api                 use only the cached JSON, never call the API
    --limit N                query/map only the first N accessions (debug)
    --range-mode MODE        how binding-region ranges are parsed (default: numeric)
                               numeric  : keep only ranges with BOTH ends numeric ('10-50',
                                          '111..111-205..205'); any range with n / c / ? / < / >
                                          on either end is dropped
                               notebook : first and last number found in the range
                                          (old behaviour: 'n-40' -> 40-40, '150-c' -> 150-150)
                               termini  : n -> 1, c -> protein length, ? dropped
    --inclusive-overlap      overlap test with <= (inclusive coordinates)
                             (default: notebook behaviour, strict <)
  custom filter (the filtered dataset is written only if at least one is given):
    --min-perc-domain X      PERCENTAGE_OF_DOMAIN  >= X
    --min-perc-feature X     PERCENTAGE_OF_FEATURE >= X
    --max-length-diff N      FEATURE_LENGTH - IPR_LENGTH < N  
    --feature-types T [...]  keep only these binding-region types
    --interpro-ids ID [...]  keep only these InterPro IDs

Cache layout (--cache-dir)
--------------------------
    isoform_api/<isoform>.json      first API call (isoforms)
    canonical_api/<acc>.json        second API call (canonicals, all pages merged)
A response with no data is cached as {"_no_results": true} and not re-queried.
Empty (0-byte) files and canonical files with an unfollowed "next" page are re-queried.

Output (--outdir)
-----------------
    bindingregions_tomap.csv          binding regions per interactor, parsed (fta/ftb x row)
    interpro_api_all.csv              all InterPro entries retrieved (all types)
    interpro_api_mapping.csv          entries of --interpro-types used for the mapping
    INTERPRO_mapped.csv               full mapped dataset (+ lengths, overlap, percentages)
    INTERPRO_mapped100x100.csv        100% of domain AND 100% of feature
    INTERPRO_mapped_filtered.csv      custom filters (only if given)
    *_counts.csv                      counts per INTERPRO_NAME for each mapped dataset
    not_retrieved.txt                 accessions requested but without InterPro entries
    summary.json                      numbers for the Methods (regenerate, never hand-write)
    run.log                           API / parsing warnings

Requirements: pandas, numpy (API calls use the standard library only)
"""

import argparse
import json
import logging
import re
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

API_BASE = "https://www.ebi.ac.uk/interpro/api"
ISOFORM_URL = API_BASE + "/protein/uniprot/{canonical}/?isoforms={isoform}"
CANONICAL_URL = API_BASE + "/entry/interpro/protein/uniprot/{acc}/?page_size=200"

# same pattern used in step 1
BINDING_PATTERN = r"(?:(?:direct|sufficient|necessary)\s+)?binding(?:-associated)? region[^|]*"

INTERPRO_COLS = ["Accession_id", "Protein_accession", "Length", "Accession_interpro",
                 "Integrated_interpro", "Name", "Type", "Source_database",
                 "start", "end", "dc-status", "API"]

log = logging.getLogger("map_interpro")


# ---------------------------------------------------------------------------
# ACCESSION HELPERS
# ---------------------------------------------------------------------------

def strip_pro(acc):
    """P12345-PRO_0000012345 -> P12345."""
    return re.sub(r"-PRO_\d+$", "", acc)


def is_isoform(acc):
    return bool(re.search(r"-\d+$", str(acc))) and "-PRO_" not in str(acc)


def canonical(acc):
    """Collapse isoform (-N) and processed-chain (-PRO_xxxx) suffixes to the canonical accession."""
    return re.sub(r"-\d+$", "", strip_pro(str(acc)))


# ---------------------------------------------------------------------------
# STEP 1 - BINDING REGIONS PER INTERACTOR
# ---------------------------------------------------------------------------

def taxid_regex(taxids):
    """taxid:9606 but not taxid:96060."""
    return r"taxid:(?:" + "|".join(str(t) for t in taxids) + r")(?!\d)"


def split_binding_regions(text):
    # explode on the comma, copying the binding region type into the different rows
    if ":" in text:
        prefix, regions = text.split(":", 1)  # split only on the first ':'
        return [f"{prefix}:{region.strip()}" for region in regions.split(",")]
    return [text]


def parse_matches(text):
    """
    Notebook parsing: splitting the matches of feature in 'FEATURE TYPE', 'FEATURE RANGE START',
    'FEATURE RANGE END' (if we have binding region with range 111..111-205..205, keeping the two
    values nearest to -). NB: 'n-205' -> 205-205, '?-50' -> 50-50.
    """
    parts = text.split(":", 1)
    feature_type = parts[0]
    numbers = re.findall(r"\d+", parts[1]) if len(parts) > 1 else []
    return (feature_type,
            int(numbers[0]) if numbers else np.nan,
            int(numbers[-1]) if numbers else np.nan,
            False)


def _parse_pos(token, side):
    """One end of a PSI-MI range: '123', '123..130', '<123', '>123', 'n', 'c', '?'."""
    token = token.strip().lower()
    nums = re.findall(r"\d+", token)
    if nums:
        # fuzzy range a..b: first number for the start, last number for the end (as notebook)
        return int(nums[0] if side == "start" else nums[-1]), False
    if token == "n" and side == "start":
        return 1, False                    # N-terminal start -> residue 1
    if token == "c" and side == "end":
        return np.nan, True                # C-terminal end -> protein length, filled after merge
    return np.nan, False                   # '?', 'n' as end, 'c' as start -> undetermined


_EXACT_POS = re.compile(r"^(\d+)(?:\.\.(\d+))?$")    # '123' or fuzzy '123..130'


def parse_matches_numeric(text):
    """
    --range-mode numeric (default): keep only ranges whose start AND end are numbered.
    'n', 'c', '?', '<N', '>N' on either end -> NaN -> the row is dropped.
    Fuzzy positions 'a..b' are numbered: start takes a, end takes b (as the notebook).
    """
    parts = text.split(":", 1)
    feature_type = parts[0]
    rng = parts[1].strip() if len(parts) > 1 else ""
    m = re.match(r"^([^-]+)-([^-]+)$", rng)
    if not m:
        return feature_type, np.nan, np.nan, False
    s, e = _EXACT_POS.match(m.group(1).strip()), _EXACT_POS.match(m.group(2).strip())
    if not (s and e):
        return feature_type, np.nan, np.nan, False
    start = int(s.group(1))
    end = int(e.group(2) or e.group(1))
    return feature_type, start, end, False


def parse_matches_termini(text):
    """--range-mode termini: parse start and end separately; n -> 1, c -> length, ? dropped."""
    parts = text.split(":", 1)
    feature_type = parts[0]
    rng = parts[1] if len(parts) > 1 else ""
    m = re.match(r"^\s*(.+?)-(.+?)\s*$", rng)
    if not m:
        return feature_type, np.nan, np.nan, False
    start, _ = _parse_pos(m.group(1), "start")
    end, end_is_cterm = _parse_pos(m.group(2), "end")
    if np.isnan(start):
        return feature_type, np.nan, np.nan, False
    return feature_type, start, end, end_is_cterm


def build_binding_table(bd, taxids, range_mode):
    """
    split the single interactors within each interaction, add role A/B, and build a new
    dataset (similar to Prisca's rangeMapping_complete.txt)
    """
    parts = []
    for role, id_col, feat_col, tax_col in [
        ("ftA", "#ID(s) interactor A", "Feature(s) interactor A", "Taxid interactor A"),
        ("ftB", "ID(s) interactor B", "Feature(s) interactor B", "Taxid interactor B"),
    ]:
        d = bd[["Interaction identifier(s)", id_col, feat_col, tax_col]].copy()
        d.columns = ["INTERACTION_ID", "UNIPROT_ID", "FEATURE", "TAXID_INTERACTOR"]
        d["FEATURE"] = d["FEATURE"].fillna("")
        d["ROLE"] = role
        # extract only binding(-associated) regions
        d["matches"] = d["FEATURE"].str.findall(BINDING_PATTERN, flags=re.I)
        # deleting all the matches_ft empty
        parts.append(d[d["matches"].str.len() > 0])

    # concat the two df (dropping duplicates, not informative at this stage)
    total = pd.concat(parts, ignore_index=True)
    total["matches"] = total["matches"].apply(tuple)       # hashable for drop_duplicates
    total = total.drop_duplicates()
    n_interactor_rows = len(total)

    total = total.explode("matches")
    # taking rows with the target taxid
    total = total[total["TAXID_INTERACTOR"].fillna("").str.contains(taxid_regex(taxids), regex=True)].copy()

    # remove parenthetical annotations (MINT/MI etc.)
    total["matches"] = total["matches"].str.replace(r"\([^)]*\)", "", regex=True)
    total["matches"] = total["matches"].apply(split_binding_regions)
    total = total.explode("matches").reset_index(drop=True)

    parser = {"numeric": parse_matches_numeric, "notebook": parse_matches,
              "termini": parse_matches_termini}[range_mode]
    parsed = pd.DataFrame(total["matches"].apply(parser).tolist(),
                          columns=["FEATURE_TYPE", "FEATURE_RANGE_START", "FEATURE_RANGE_END", "END_IS_CTERM"],
                          index=total.index)
    total = pd.concat([total, parsed], axis=1)
    for c in ["FEATURE_RANGE_START", "FEATURE_RANGE_END"]:
        total[c] = total[c].astype("Int64")               # integers with NA, no 10.0 in the CSV
    total["FEATURE_TYPE"] = total["FEATURE_TYPE"].str.strip()

    # extracting only the uniprot (non-UniProt interactors -> NaN, never mapped)
    total["UNIPROT_ID"] = total["UNIPROT_ID"].str.extract(r"uniprotkb:(.*)", expand=False)

    # accessions to query (before the range filter). Processed chains (-PRO_) are not queried:
    # InterPro has no entry for them and their feature coordinates may refer to the mature chain,
    # so they are never mapped (their rows are only counted in the summary)
    api_list = sorted({x for x in total["UNIPROT_ID"].dropna() if "-PRO_" not in x})

    # mask delete nan value in feature range start / end derived from non annotated binding region
    # such as ?-?, c-c, n-n (numeric mode: also n-40, 150-c, ?-50, <10-50 ...)
    if range_mode == "termini":
        keep = total.FEATURE_RANGE_START.notna() & (total.FEATURE_RANGE_END.notna() | total.END_IS_CTERM)
    else:
        keep = total.FEATURE_RANGE_START.notna() | total.FEATURE_RANGE_END.notna()
    n_dropped_ranges = int((~keep).sum())
    total = total[keep]

    # dropping again duplicates
    tomap = total[["INTERACTION_ID", "UNIPROT_ID", "FEATURE", "ROLE", "FEATURE_TYPE",
                   "FEATURE_RANGE_START", "FEATURE_RANGE_END", "END_IS_CTERM"]].drop_duplicates()
    tomap = tomap[tomap["UNIPROT_ID"].notna()].reset_index(drop=True)
    return tomap, api_list, n_interactor_rows, n_dropped_ranges


# ---------------------------------------------------------------------------
# STEP 2 - INTERPRO API
# ---------------------------------------------------------------------------

_SSL_CTX = ssl.create_default_context()


def http_get_json(url, retries=5, timeout=60):
    """GET a JSON from the InterPro API. Returns None for 204/404 (no data)."""
    delay = 5
    for attempt in range(1, retries + 1):
        try:
            req = request.Request(url, headers={"Accept": "application/json"})
            with request.urlopen(req, context=_SSL_CTX, timeout=timeout) as res:
                if res.status == 204:
                    return None
                body = res.read()
                return json.loads(body) if body else None
        except HTTPError as e:
            if e.code in (204, 404):
                return None
            if e.code in (408, 429) or e.code >= 500:
                # timeout / rate limit / server error: wait and retry
                log.debug(f"HTTP {e.code} on {url} (attempt {attempt}), retrying in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise
        except (URLError, TimeoutError, ConnectionError) as e:
            log.debug(f"{type(e).__name__} on {url} (attempt {attempt}): {e}")
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError(f"giving up after {retries} attempts: {url}")


def _read_cache(path):
    if not path.exists() or path.stat().st_size == 0:     # 0-byte file = failed call
        return None, False
    with open(path) as fh:
        return json.load(fh), True


def _write_cache(path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(data if data is not None else {"_no_results": True}, fh)
    tmp.replace(path)                                       # atomic: no half-written JSON


def fetch_isoform(acc, cache_dir, use_api):
    path = cache_dir / "isoform_api" / f"{acc}.json"
    data, cached = _read_cache(path)
    if cached:
        return acc, data
    if not use_api:
        return acc, None
    data = http_get_json(ISOFORM_URL.format(canonical=canonical(acc), isoform=acc))
    _write_cache(path, data)
    return acc, data


def fetch_canonical(acc, cache_dir, use_api):
    path = cache_dir / "canonical_api" / f"{acc}.json"
    data, cached = _read_cache(path)
    if cached and not (isinstance(data, dict) and data.get("next")):
        return acc, data
    if cached and not use_api:
        log.warning(f"{acc}: cached canonical JSON has more pages ('next') but --no-api is set")
        return acc, data
    if not use_api:
        return acc, None
    # follow pagination and merge all pages in a single JSON
    url, results, count = CANONICAL_URL.format(acc=acc), [], 0
    while url:
        page = http_get_json(url)
        if not page:
            break
        results.extend(page.get("results", []))
        count = page.get("count", len(results))
        url = page.get("next")
    data = {"count": count, "next": None, "previous": None, "results": results} if results else None
    _write_cache(path, data)
    return acc, data


def retrieve_all(accessions, cache_dir, use_api, workers):
    """Isoforms -> first API, canonicals -> second API. Returns {acc: json or None}."""
    (cache_dir / "isoform_api").mkdir(parents=True, exist_ok=True)
    (cache_dir / "canonical_api").mkdir(parents=True, exist_ok=True)
    jobs = [(fetch_isoform if is_isoform(a) else fetch_canonical, a) for a in accessions]
    out, failed = {}, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, a, cache_dir, use_api): a for fn, a in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            acc = futs[fut]
            try:
                out[acc] = fut.result()[1]
            except Exception as e:           # API failure: not cached, retried on next run
                log.warning(f"{acc}: API call failed ({e})")
                failed.append(acc)
                out[acc] = None
            if i % 100 == 0 or i == len(jobs):
                print(f"  API: {i}/{len(jobs)}", file=sys.stderr, end="\r")
    print(file=sys.stderr)
    return out, failed


# flatten the JSON, take only the information we need and put them in a df

def json_to_rows_firstapi(info):
    """First API call (isoforms): one row per InterPro fragment."""
    rows = []
    # Checkpoint 1: empty element or not a dict
    if not info or not isinstance(info, dict) or info.get("_no_results"):
        return rows
    # Checkpoint 2: essential fields missing (no results)
    if "accession" not in info:
        return rows
    # Checkpoint 3/4: features missing, empty or not a dict
    if not isinstance(info.get("features"), dict) or not info["features"]:
        log.debug(f"Accession {info.get('accession')} - no feature present")
        return rows

    base = {"Accession_id": info.get("accession"),
            "Protein_accession": info.get("protein_acc"),
            "Length": info.get("length")}
    found_interpro = False
    for feature_key, feature in info["features"].items():
        # Checkpoint 5: feature is not a dict
        if not isinstance(feature, dict):
            continue
        # keep only interpro
        if feature.get("source_database") != "interpro":
            continue
        found_interpro = True
        meta = {"Accession_interpro": feature.get("accession"),
                "Integrated_interpro": feature.get("integrated"),
                "Name": feature.get("name"),
                "Type": feature.get("type"),
                "Source_database": feature.get("source_database")}
        # Checkpoint 6: locations missing or empty -> add the row anyway, without fragments
        if not feature.get("locations"):
            rows.append({**base, **meta, "start": None, "end": None, "dc-status": None})
            continue
        # locations is a LIST
        for location in feature["locations"]:
            if not isinstance(location, dict) or not location.get("fragments"):
                continue
            # fragments is a LIST: create a new row for each fragment
            for fragment in location["fragments"]:
                if isinstance(fragment, dict):
                    rows.append({**base, **meta, "start": fragment.get("start"),
                                 "end": fragment.get("end"), "dc-status": fragment.get("dc-status")})
    # info message if there are no interpro features
    if not found_interpro:
        log.debug(f"Accession {info.get('accession')} - no interpro feature found")
    return rows


def json_to_rows_secondapi(outer_dict):
    """Second API call (canonicals): one row per InterPro fragment."""
    rows = []
    if not isinstance(outer_dict, dict) or "results" not in outer_dict:
        return rows
    for entry in outer_dict["results"]:
        if not isinstance(entry, dict) or not entry.get("metadata"):
            continue
        metadata = entry["metadata"]
        # keep only interpro
        if metadata.get("source_database") != "interpro":
            continue
        # common metadata fields
        base_metadata = {"Accession_interpro": metadata.get("accession"),
                         "Integrated_interpro": metadata.get("integrated"),
                         "Name": metadata.get("name"),
                         "Type": metadata.get("type"),
                         "Source_database": metadata.get("source_database")}
        # Checkpoint: proteins missing or empty
        if not entry.get("proteins"):
            log.debug(f"no protein for {metadata.get('accession')}")
            continue
        for protein in entry["proteins"]:
            if not isinstance(protein, dict):
                continue
            protein_data = {"Accession_id": protein.get("accession", "").upper(),
                            "Protein_accession": protein.get("accession", "").upper(),
                            "Length": protein.get("protein_length")}
            # Checkpoint: entry_protein_locations missing -> add a row without fragments
            if not protein.get("entry_protein_locations"):
                rows.append({**base_metadata, **protein_data, "start": None, "end": None, "dc-status": None})
                continue
            for location in protein["entry_protein_locations"]:
                if not isinstance(location, dict) or not location.get("fragments"):
                    continue
                for fragment in location["fragments"]:
                    if isinstance(fragment, dict):
                        rows.append({**base_metadata, **protein_data, "start": fragment.get("start"),
                                     "end": fragment.get("end"), "dc-status": fragment.get("dc-status")})
    return rows


def build_interpro_table(api_results):
    rows = []
    for acc, data in api_results.items():
        if is_isoform(acc):
            rows += [{**r, "API": "isoform"} for r in json_to_rows_firstapi(data)]
        else:
            rows += [{**r, "API": "canonical"} for r in json_to_rows_secondapi(data)]
    df = pd.DataFrame(rows, columns=INTERPRO_COLS)
    # no duplicates
    return df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 3 - MAPPING
# ---------------------------------------------------------------------------

def map_interpro(tomap, interpro_clean, inclusive):
    """merging two df and then delete the rows not in the range"""
    mapped = tomap.merge(interpro_clean, on="UNIPROT_ID", how="inner")

    # C-terminal ends (--range-mode termini only) -> protein length
    cterm = mapped["END_IS_CTERM"].astype(bool)
    mapped.loc[cterm, "FEATURE_RANGE_END"] = mapped.loc[cterm, "PROTEIN_LENGTH"]

    for c in ["FEATURE_RANGE_START", "FEATURE_RANGE_END", "INTERPRO_RANGE_START", "INTERPRO_RANGE_END"]:
        mapped[c] = pd.to_numeric(mapped[c], errors="coerce")
    mapped = mapped.dropna(subset=["FEATURE_RANGE_START", "FEATURE_RANGE_END",
                                   "INTERPRO_RANGE_START", "INTERPRO_RANGE_END"])
    for c in ["FEATURE_RANGE_START", "FEATURE_RANGE_END", "INTERPRO_RANGE_START", "INTERPRO_RANGE_END"]:
        mapped[c] = mapped[c].astype(int)

    fs, fe = mapped.FEATURE_RANGE_START, mapped.FEATURE_RANGE_END
    is_, ie = mapped.INTERPRO_RANGE_START, mapped.INTERPRO_RANGE_END
    if inclusive:
        ov = (fs <= ie) & (is_ <= fe)
    else:
        ov = (fs < ie) & (is_ < fe)                      # notebook criterion
    mapped = mapped[ov].copy()

    # add InterPro length, feature length, overlap, and percentages
    # inclusive coordinates: length = end - start + 1; overlap = max(0, min(ends) - max(starts) + 1)
    mapped["FEATURE_LENGTH"] = mapped.FEATURE_RANGE_END - mapped.FEATURE_RANGE_START + 1
    mapped["IPR_LENGTH"] = mapped.INTERPRO_RANGE_END - mapped.INTERPRO_RANGE_START + 1
    mapped["OVERLAP"] = (np.minimum(mapped.INTERPRO_RANGE_END, mapped.FEATURE_RANGE_END)
                         - np.maximum(mapped.INTERPRO_RANGE_START, mapped.FEATURE_RANGE_START)
                         + 1).clip(lower=0)
    mapped["PERCENTAGE_OF_DOMAIN"] = (mapped.OVERLAP / mapped.IPR_LENGTH * 100).round(2)
    mapped["PERCENTAGE_OF_FEATURE"] = (mapped.OVERLAP / mapped.FEATURE_LENGTH * 100).round(2)

    cols = ["INTERACTION_ID", "UNIPROT_ID", "FEATURE", "ROLE", "FEATURE_TYPE",
            "FEATURE_RANGE_START", "FEATURE_RANGE_END", "PROTEIN_LENGTH", "INTERPRO_ID",
            "INTERPRO_NAME", "INTERPRO_RANGE_START", "INTERPRO_RANGE_END",
            "FEATURE_LENGTH", "IPR_LENGTH", "OVERLAP", "PERCENTAGE_OF_DOMAIN", "PERCENTAGE_OF_FEATURE"]
    return mapped[cols].reset_index(drop=True)


def custom_filter(mapped, args):
    """Custom filtered dataset. Returns None if no filter was given."""
    m = pd.Series(True, index=mapped.index)
    used = {}
    if args.min_perc_domain is not None:
        m &= mapped.PERCENTAGE_OF_DOMAIN >= args.min_perc_domain
        used["min_perc_domain"] = args.min_perc_domain
    if args.min_perc_feature is not None:
        m &= mapped.PERCENTAGE_OF_FEATURE >= args.min_perc_feature
        used["min_perc_feature"] = args.min_perc_feature
    if args.max_length_diff is not None:
        # discard entries whose annotated binding region is much larger than the InterPro domain
        m &= (mapped.FEATURE_LENGTH - mapped.IPR_LENGTH) < args.max_length_diff
        used["max_length_diff"] = args.max_length_diff
    if args.feature_types:
        m &= mapped.FEATURE_TYPE.str.lower().isin([t.lower() for t in args.feature_types])
        used["feature_types"] = args.feature_types
    if args.interpro_ids:
        m &= mapped.INTERPRO_ID.isin(args.interpro_ids)
        used["interpro_ids"] = args.interpro_ids
    if not used:
        return None, None
    return mapped[m].copy(), used


def save_counts(df, path):
    df.value_counts("INTERPRO_NAME").reset_index().to_csv(path, index=False)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="interactions with binding regions (step-1 CSV, or MITAB .txt tab-separated)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cache-dir", required=True, help="folder for the InterPro API JSON cache")
    ap.add_argument("--taxid", nargs="+", default=["9606"])
    ap.add_argument("--interpro-types", nargs="+", default=["domain"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--range-mode", choices=["numeric", "notebook", "termini"], default="numeric")
    ap.add_argument("--inclusive-overlap", action="store_true")
    ap.add_argument("--min-perc-domain", type=float, default=None)
    ap.add_argument("--min-perc-feature", type=float, default=None)
    ap.add_argument("--max-length-diff", type=int, default=None)
    ap.add_argument("--feature-types", nargs="+", default=None)
    ap.add_argument("--interpro-ids", nargs="+", default=None)
    args = ap.parse_args()

    outdir, cache_dir = Path(args.outdir), Path(args.cache_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s",
                        handlers=[logging.FileHandler(outdir / "run.log", mode="w")])
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    log.addHandler(console)

    # --- input ---
    sep = "," if args.input.endswith(".csv") else "\t"
    bd = pd.read_csv(args.input, sep=sep, dtype=str)
    print(f"Input: {bd.shape[0]} interactions", file=sys.stderr)

    # --- binding regions per interactor ---
    tomap, api_list, n_int_rows, n_dropped = build_binding_table(bd, args.taxid, args.range_mode)
    if args.limit:
        api_list = api_list[:args.limit]
        tomap = tomap[tomap.UNIPROT_ID.isin(api_list)]
    tomap.drop(columns=["END_IS_CTERM"] if args.range_mode != "termini" else []).to_csv(
        outdir / "bindingregions_tomap.csv", index=False)
    print(f"Binding regions to map: {len(tomap)} rows, {tomap.UNIPROT_ID.nunique()} accessions; "
          f"{len(api_list)} accessions to query "
          f"({sum(map(is_isoform, api_list))} isoforms)", file=sys.stderr)

    # --- InterPro API ---
    api_results, failed = retrieve_all(api_list, cache_dir, not args.no_api, args.workers)
    interpro_all = build_interpro_table(api_results)
    interpro_all.to_csv(outdir / "interpro_api_all.csv", index=False)
    interpro_sel = interpro_all[interpro_all["Type"].isin(args.interpro_types)].copy()
    interpro_sel.to_csv(outdir / "interpro_api_mapping.csv", index=False)

    # rename columns
    interpro_clean = interpro_sel[["Accession_id", "Length", "Accession_interpro", "Name", "start", "end"]].rename(
        columns={"Accession_id": "UNIPROT_ID", "Length": "PROTEIN_LENGTH", "Accession_interpro": "INTERPRO_ID",
                 "Name": "INTERPRO_NAME", "start": "INTERPRO_RANGE_START", "end": "INTERPRO_RANGE_END"})

    # --- mapping ---
    mapped = map_interpro(tomap, interpro_clean, args.inclusive_overlap)
    mapped.to_csv(outdir / "INTERPRO_mapped.csv", index=False)
    save_counts(mapped, outdir / "INTERPRO_mapped_counts.csv")

    # 100% percentage of domain and feature (total overlap)
    interpro_100 = mapped[(mapped.PERCENTAGE_OF_FEATURE == 100.0) & (mapped.PERCENTAGE_OF_DOMAIN == 100.0)].copy()
    interpro_100.to_csv(outdir / "INTERPRO_mapped100x100.csv", index=False)
    save_counts(interpro_100, outdir / "INTERPRO_mapped100x100_counts.csv")

    filtered, used_filters = custom_filter(mapped, args)
    if filtered is not None:
        filtered.to_csv(outdir / "INTERPRO_mapped_filtered.csv", index=False)
        save_counts(filtered, outdir / "INTERPRO_mapped_filtered_counts.csv")

    # ============================================================
    #  SUMMARY NUMBERS for the Methods (regenerate, never hand-write)
    # ============================================================
    retrieved_acc = set(interpro_all["Accession_id"].dropna())
    domain_acc = set(interpro_sel["Accession_id"].dropna())
    not_retrieved = set(api_list) - retrieved_acc
    (outdir / "not_retrieved.txt").write_text("\n".join(sorted(not_retrieved)) + "\n")

    tot_keys = set(tomap["UNIPROT_ID"].dropna())
    matched = tot_keys & set(interpro_clean["UNIPROT_ID"].dropna())
    n_pro = int(tomap.UNIPROT_ID.str.contains("-PRO_", na=False).sum())

    def acc_genes(df):
        s = df["UNIPROT_ID"].dropna()
        return {"accessions": int(s.nunique()), "canonical_genes": int(s.map(canonical).nunique())}

    summary = {
        "input": args.input,
        "taxid": args.taxid,
        "options": {k: v for k, v in vars(args).items() if k not in ("input", "outdir", "cache_dir")},
        "binding_regions": {
            "interactor_rows_with_binding": n_int_rows,
            "ranges_dropped_not_numbered": n_dropped,
            "rows_to_map": len(tomap),
            "accessions_to_map": len(tot_keys),
            "canonical_genes_to_map": len({canonical(x) for x in tot_keys}),
            "rows_on_PRO_chains": n_pro,
        },
        "api_retrieval": {
            "requested": len(api_list),
            "requested_isoforms": sum(map(is_isoform, api_list)),
            "retrieved_with_interpro_entries": len(retrieved_acc),
            "retrieved_with_mapping_types": len(domain_acc),
            "isoforms_retrieved": sum(1 for x in retrieved_acc if is_isoform(x)),
            "not_retrieved": len(not_retrieved),
            "not_retrieved_isoforms": sum(1 for x in not_retrieved if is_isoform(x)),
            "not_retrieved_canonicals": sum(1 for x in not_retrieved if not is_isoform(x)),
            "api_failures": len(failed),
            "entry_types": interpro_all["Type"].value_counts().to_dict(),
        },
        "mapping": {
            "accessions_with_interpro_domain": len(matched),
            "canonical_genes_with_interpro_domain": len({canonical(x) for x in matched}),
            "isoforms_mapped_independently": sum(1 for x in matched if is_isoform(x)),
            "full_mapped": {"rows": len(mapped), **acc_genes(mapped)},
            "100x100": {"rows": len(interpro_100), **acc_genes(interpro_100)},
        },
    }
    if filtered is not None:
        summary["mapping"]["filtered"] = {"filters": used_filters, "rows": len(filtered), **acc_genes(filtered)}
    with open(outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(json.dumps({k: summary[k] for k in ("binding_regions", "api_retrieval", "mapping")},
                     indent=2, default=str))
    if failed:
        print(f"WARNING: {len(failed)} API calls failed - re-run the same command to retry them", file=sys.stderr)


if __name__ == "__main__":
    main()
