#!/usr/bin/env python3
"""
map_diseases.py
===============

Step 5 of the binding-domain pipeline. Script version of 04_Mapping_Diseases.ipynb.

Takes the LP/P and US variants mapped in step 4 (map_variants_on_bd.py), extracts the
associated OMIM diseases and assigns each disease its MeSH top-level class(es), using
only real cross-references:
    - DisGeNET  (diseaseClasses_MSH, from the cached DisGeNET JSON responses)
    - wikidata  (top-level MeSH codes, from the OMIM -> MeSH wikidata table)
C23 (Pathological Conditions, Signs and Symptoms) is excluded; diseases without a MeSH
class are labelled "Unclassified".

By default only variants falling in the overlap between binding region and InterPro
domain are used, and only for domain-region pairs overlapping by >= 80% (same criterion
as ora_interpro_bd.py). A variant is kept if at least one of its pairs passes.

The DisGeNET and wikidata queries are NOT run here (as in the notebook): the script reads
the cached results and writes the lists of OMIM IDs to query, including the ones not in the
cache yet.

Usage
-----
    python map_diseases.py \
        --variants-dir  /home/alessia/BindingDomainDataset/new_data_analysis/Variant_mapping \
        --disgenet-dirs /home/alessia/BindingDomainDataset/new_data_analysis/disgenet_api_results/lp_diseases \
                        /home/alessia/BindingDomainDataset/new_data_analysis/disgenet_api_results/us_diseases \
        --disgenet-notfound-dir /home/alessia/BindingDomainDataset/new_data_analysis/disgenet_api_results/notfound \
        --wikidata-mesh /home/alessia/BindingDomainDataset/new_data_analysis/output_python_mesh.csv \
        --outdir        /home/alessia/BindingDomainDataset/new_data_analysis/Disease_mapping_overlap80

Options
-------
    --source WHAT          overlap : variants in binding region x InterPro overlap (default)
                           bd      : all variants in binding regions (notebook behaviour)
    --min-perc X           minimum overlap percentage of the domain-region pair (default 80;
                           only with --source overlap; 0 = whole overlap dataset)
    --perc-on WHAT         domain (default) | feature | both   (as in ora_interpro_bd.py)
    --disgenet-dirs D ...  folders of DisGeNET JSON with disease classes (lp, us, ...)
    --disgenet-notfound-dir D   folder of DisGeNET JSON with cross-references for the OMIM IDs
                           without a class (used to write the UMLS CUI list for wikidata)
    --wikidata-mesh FILE   CSV with columns OMIM, top_codes ('C05;C16')
    --top-classes N        number of classes in top_mesh_classes.csv (default 10)

Input (from step 4, inside --variants-dir)
------------------------------------------
    --source overlap : Variant_onINTERPRO_BD/LPonINTERPROBD_overlap.csv, USonINTERPROBD_overlap.csv
    --source bd      : Variant_onBD/LP_onBD.csv, VUS_onBD.csv

Output (--outdir)
-----------------
    variants_LP_used.csv / variants_US_used.csv   variant rows actually used (after the filter)
    list_diseases_lp.csv / list_diseases_us.csv   diseases with n. rows and n. unique variants
    list_diseases_lp.txt / list_diseases_us.txt   OMIM_xxxx lists to query DisGeNET
    list_diseases_nf.txt                          OMIM IDs queried but without a DisGeNET class
    omim_not_in_disgenet_cache.txt                OMIM IDs never queried (to query before re-running)
    listCUIs.txt / listOMIMIDs_nf.txt             UMLS CUIs of the not-found OMIM IDs (for wikidata)
    disease_to_mesh_class.csv                     one row per disease, Mesh_class = list
    disease_to_mesh_class_long.csv                one row per disease-class (for plotting in R)
    top_mesh_classes.csv                          top MeSH classes (n. diseases), Unclassified excluded
    disease_summary.json                          parameters and counts

Requirements: pandas, numpy
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# canonical MeSH top-level disease categories (C-tree); C23 intentionally excluded
MESH_C_MAP = {
    "C01": "Infections",
    "C04": "Neoplasms",
    "C05": "Musculoskeletal Diseases",
    "C06": "Digestive System Diseases",
    "C07": "Stomatognathic Diseases",
    "C08": "Respiratory Tract Diseases",
    "C09": "Otorhinolaryngologic Diseases",
    "C10": "Nervous System Diseases",
    "C11": "Eye Diseases",
    "C12": "Male Urogenital Diseases",
    "C13": "Female Urogenital Diseases and Pregnancy Complications",
    "C14": "Cardiovascular Diseases",
    "C15": "Hemic and Lymphatic Diseases",
    "C16": "Congenital, Hereditary, and Neonatal Diseases and Abnormalities",
    "C17": "Skin and Connective Tissue Diseases",
    "C18": "Nutritional and Metabolic Diseases",
    "C19": "Endocrine System Diseases",
    "C20": "Immune System Diseases",
    "C21": "Disorders of Environmental Origin",
    "C22": "Animal Diseases",
    "C24": "Occupational Diseases",
    "C25": "Chemically-Induced Disorders",
    "C26": "Wounds and Injuries",
}

INPUT_FILES = {
    "overlap": {"LP": "Variant_onINTERPRO_BD/LPonINTERPROBD_overlap.csv",
                "US": "Variant_onINTERPRO_BD/USonINTERPROBD_overlap.csv"},
    "bd":      {"LP": "Variant_onBD/LP_onBD.csv",
                "US": "Variant_onBD/VUS_onBD.csv"},
}
VAR_KEY = ["UNIPROT_ID", "AA_change"]


# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------

#read multiple json from a folder
def read_json_files_fromfolder(folder_path):
    json_data = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            if os.path.getsize(file_path) > 0:
                with open(file_path) as file:
                    json_data.append(json.load(file))
    return json_data


def requested_omim(json_files):
    """OMIM IDs present in a folder of DisGeNET responses (queried, with or without payload)."""
    out = set()
    for info in json_files:
        if isinstance(info, dict) and isinstance(info.get("requestpar"), dict):
            d = info["requestpar"].get("disease")
            if d:
                out.add(d)
    return out


def extract_disease_class(results_protein):
    rows = []
    # Checkpoint 0: empty or non-list input
    if not results_protein or not isinstance(results_protein, list):
        return pd.DataFrame(columns=["Disease", "Mesh_class"])

    for info in results_protein:
        # Checkpoint 1: empty element or not a dict
        if not info or not isinstance(info, dict):
            continue
        base = {"Disease": (info.get("requestpar") or {}).get("disease")}
        # Checkpoint 2: essential fields missing
        if not info.get("payload"):
            continue
        for classes in info["payload"]:
            row = base.copy()
            row["Mesh_class"] = classes.get("diseaseClasses_MSH")
            rows.append(row)
    return pd.DataFrame(rows, columns=["Disease", "Mesh_class"])


def extract_disease_crossref(results_disease):
    rows = []
    # Checkpoint 0: empty or non-list input
    if not results_disease or not isinstance(results_disease, list):
        return pd.DataFrame(columns=["Disease", "UMLS_Disease_classes", "vocabulary_type", "ID_voc"])

    for info in results_disease:
        # Checkpoint 1: empty element or not a dict
        if not info or not isinstance(info, dict):
            continue
        # Checkpoint 2: essential fields missing
        if not info.get("payload"):
            continue
        # Iterate through payload list (it's a list, not a dict!)
        for payload_item in info["payload"]:
            base = {
                "Disease": info["requestpar"].get("disease"),
                "UMLS_Disease_classes": payload_item.get("diseaseClasses_UMLS_ST", []),
            }
            # Now iterate through diseaseCodes
            for crossref in payload_item.get("diseaseCodes", []):
                row = base.copy()
                row["vocabulary_type"] = crossref.get("vocabulary")
                row["ID_voc"] = crossref.get("code")
                rows.append(row)
    return pd.DataFrame(rows, columns=["Disease", "UMLS_Disease_classes", "vocabulary_type", "ID_voc"])


def codes_from_disgenet(mesh_class):
    """DisGeNET diseaseClasses_MSH -> set of C## codes."""
    if not isinstance(mesh_class, (list, tuple)):
        return set()
    out = set()
    for item in mesh_class:
        out.update(re.findall(r"C\d{2}", str(item)))
    return out


def write_list(items, path):
    with open(path, "w") as f:
        for x in sorted(items):
            f.write(f"{x}\n")


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

def load_variants(path, source, min_perc, perc_on):
    df = pd.read_csv(path, dtype={"FEATURE_RANGE_START": "Int64", "FEATURE_RANGE_END": "Int64"})
    n_before = len(df.drop_duplicates(subset=VAR_KEY))
    if source == "overlap" and min_perc > 0:
        if perc_on == "domain":
            ok = df["PERCENTAGE_OF_DOMAIN"] >= min_perc
        elif perc_on == "feature":
            ok = df["PERCENTAGE_OF_FEATURE"] >= min_perc
        else:  # both, on the same domain-region pair
            ok = (df["PERCENTAGE_OF_DOMAIN"] >= min_perc) & (df["PERCENTAGE_OF_FEATURE"] >= min_perc)
        df = df[ok].copy()
    # drop variants without an associated disease
    df = df[df["Disease_name"].notna() & (df["Disease_name"] != "-")].copy()
    #extract OMIM ID and then redirect through API
    df["OMIM_ID"] = df["Disease_name"].str.extract(r"\[MIM:(\d+)\]")[0]
    return df, n_before


def disease_list(df):
    """Diseases with n. rows (notebook value_counts) and n. unique variants."""
    rows = df.value_counts("Disease_name").rename("count")
    uniq = df.drop_duplicates(subset=VAR_KEY).value_counts("Disease_name").rename("n_unique_variants")
    return pd.concat([rows, uniq], axis=1).fillna(0).astype(int).reset_index()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants-dir", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--source", choices=["overlap", "bd"], default="overlap")
    ap.add_argument("--min-perc", type=float, default=80.0)
    ap.add_argument("--perc-on", choices=["domain", "feature", "both"], default="domain")
    ap.add_argument("--disgenet-dirs", nargs="+", required=True)
    ap.add_argument("--disgenet-notfound-dir", default=None)
    ap.add_argument("--wikidata-mesh", default=None)
    ap.add_argument("--top-classes", type=int, default=10)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- variants and OMIM IDs ---
    var, n_before = {}, {}
    for cat, rel in INPUT_FILES[args.source].items():
        var[cat], n_before[cat] = load_variants(args.variants_dir / rel, args.source, args.min_perc, args.perc_on)
        var[cat].to_csv(args.outdir / f"variants_{cat}_used.csv", index=False)
        disease_list(var[cat]).to_csv(args.outdir / f"list_diseases_{cat.lower()}.csv", index=False)

    diseases_lp = set("OMIM_" + var["LP"].OMIM_ID.dropna().astype(str))
    diseases_us = set("OMIM_" + var["US"].OMIM_ID.dropna().astype(str))
    diseases_total = diseases_lp | diseases_us
    write_list(diseases_lp, args.outdir / "list_diseases_lp.txt")
    write_list(diseases_us, args.outdir / "list_diseases_us.txt")

    # --- Extracting information from json (DisGeNET disease classes) ---
    disgenet_json = []
    for d in args.disgenet_dirs:
        disgenet_json += read_json_files_fromfolder(d)
    queried = requested_omim(disgenet_json)
    df_classes = extract_disease_class(disgenet_json)
    found = set(df_classes.Disease.dropna())

    # queried but without classes -> to look up through cross-references / wikidata
    not_found = sorted(x.replace("OMIM_", "") for x in (diseases_total & queried) - found)
    write_list(not_found, args.outdir / "list_diseases_nf.txt")
    # never queried: new diseases in this variant set, query DisGeNET and re-run
    not_queried = sorted(diseases_total - queried)
    write_list(not_queried, args.outdir / "omim_not_in_disgenet_cache.txt")
    if not_queried:
        print(f"WARNING: {len(not_queried)} OMIM IDs are not in the DisGeNET cache "
              f"(see omim_not_in_disgenet_cache.txt)", file=sys.stderr)

    # --- Extracting information crossref from json DISGENET (Diseases classification not found) ---
    n_cui = None
    if args.disgenet_notfound_dir:
        df_nf = extract_disease_crossref(read_json_files_fromfolder(args.disgenet_notfound_dir))
        df_nf = df_nf[df_nf.Disease.str.replace("OMIM_", "", regex=False).isin(not_found)]
        # keep only the UMLS CUIs
        df_UMLS = df_nf[df_nf.vocabulary_type == "UMLS"].copy()
        write_list(set(df_UMLS.ID_voc.astype(str)), args.outdir / "listCUIs.txt")
        write_list(set(df_UMLS.Disease.str.replace("OMIM_", "", regex=False)), args.outdir / "listOMIMIDs_nf.txt")
        n_cui = int(df_UMLS.ID_voc.nunique())

    # ------------------------------------------------------------
    # OMIM -> MeSH disease class (DisGeNET + wikidata)
    # ------------------------------------------------------------
    # --- DisGeNET branch: OMIM -> set of C## codes ---
    disgenet = df_classes.copy()
    disgenet["codes"] = disgenet["Mesh_class"].apply(codes_from_disgenet)
    disgenet = disgenet[["Disease", "codes"]]   # Disease is "OMIM_xxxx"

    # --- wikidata branch: OMIM -> set of C## codes (from top_codes) ---
    branches = [disgenet]
    if args.wikidata_mesh:
        mesh_throughwikidata = pd.read_csv(args.wikidata_mesh)
        wd = mesh_throughwikidata[mesh_throughwikidata["top_codes"].notna()].copy()
        # OMIM read as float if the column has NaN (154700.0) -> force integer string
        wd = wd[pd.to_numeric(wd["OMIM"], errors="coerce").notna()]
        wd["Disease"] = "OMIM_" + pd.to_numeric(wd["OMIM"]).astype("int64").astype(str)
        wd["codes"] = wd["top_codes"].apply(lambda s: {c.strip() for c in str(s).split(";") if c.strip()})
        branches.append(wd[["Disease", "codes"]])

    # --- union of the two branches per OMIM, then map codes -> class names (drops C23/unknown) ---
    allcodes = pd.concat(branches, ignore_index=True)
    omim_codes = allcodes.groupby("Disease")["codes"].apply(lambda s: set().union(*s)).reset_index()
    omim_codes["Mesh_class"] = omim_codes["codes"].apply(
        lambda codes: sorted({MESH_C_MAP[c] for c in codes if c in MESH_C_MAP}))
    omim_codes["OMIM_ID"] = omim_codes["Disease"].str.replace("OMIM_", "", regex=False)
    omim_to_mesh = omim_codes[["OMIM_ID", "Mesh_class"]]

    # --- attach the MeSH class list to every disease in the variant dataset ---
    disease_dataset = pd.concat(
        [var["US"][["Disease_name", "OMIM_ID"]], var["LP"][["Disease_name", "OMIM_ID"]]],
        ignore_index=True).drop_duplicates()
    disease_mesh = disease_dataset.merge(omim_to_mesh, on="OMIM_ID", how="left")
    # diseases with no MeSH class -> "Unclassified"
    disease_mesh["Mesh_class"] = disease_mesh["Mesh_class"].apply(
        lambda v: v if isinstance(v, list) and len(v) > 0 else ["Unclassified"])

    # n. unique variants per disease (LP and US), useful to weight / filter in R
    for cat in ["LP", "US"]:
        n = var[cat].drop_duplicates(subset=VAR_KEY).value_counts("Disease_name").rename(f"n_variants_{cat}")
        disease_mesh = disease_mesh.merge(n, left_on="Disease_name", right_index=True, how="left")
        disease_mesh[f"n_variants_{cat}"] = disease_mesh[f"n_variants_{cat}"].fillna(0).astype(int)

    covered = disease_mesh["Mesh_class"].apply(lambda v: v != ["Unclassified"])
    disease_mesh.to_csv(args.outdir / "disease_to_mesh_class.csv", index=False)
    # tidy long version (one row per disease-class) for plotting in R
    exploded = disease_mesh.explode("Mesh_class")
    exploded.to_csv(args.outdir / "disease_to_mesh_class_long.csv", index=False)

    # --- top MeSH disease classes (excluding Unclassified), n. of diseases ---
    top = (exploded[exploded["Mesh_class"] != "Unclassified"]
           .value_counts("Mesh_class").head(args.top_classes).reset_index())
    top.to_csv(args.outdir / "top_mesh_classes.csv", index=False)

    summary = {
        "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "variants": {cat: {"unique_before_filter": n_before[cat],
                           "unique_used_with_disease": int(len(var[cat].drop_duplicates(subset=VAR_KEY))),
                           "proteins": int(var[cat].UNIPROT_ID.nunique()),
                           "diseases": int(var[cat].Disease_name.nunique()),
                           "omim_ids": int(var[cat].OMIM_ID.nunique())} for cat in var},
        "omim": {"lp": len(diseases_lp), "us": len(diseases_us), "total": len(diseases_total),
                 "with_disgenet_class": len(diseases_total & found),
                 "queried_without_class": len(not_found),
                 "not_in_disgenet_cache": len(not_queried),
                 "umls_cuis_for_not_found": n_cui},
        "mesh": {"diseases_total": int(disease_mesh["Disease_name"].nunique()),
                 "omim_total": int(disease_mesh["OMIM_ID"].nunique()),
                 "omim_with_mesh": int(disease_mesh.loc[covered, "OMIM_ID"].nunique()),
                 "diseases_without_omim": int(disease_mesh["OMIM_ID"].isna().sum()),
                 "rows_covered_pct": round(covered.mean() * 100, 1) if len(covered) else None},
    }
    with open(args.outdir / "disease_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    for cat, s in summary["variants"].items():
        print(f"[{cat}] unique variants {s['unique_before_filter']} -> {s['unique_used_with_disease']} "
              f"(filter + disease) | {s['proteins']} proteins | {s['omim_ids']} OMIM")
    m = summary["mesh"]
    print(f"diseases total      : {m['omim_total']} OMIM")
    print(f"diseases with MeSH  : {m['omim_with_mesh']}")
    print(f"rows covered        : {m['rows_covered_pct']}%")


if __name__ == "__main__":
    main()
