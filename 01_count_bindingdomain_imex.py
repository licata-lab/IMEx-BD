#!/usr/bin/env python3
"""
count_bindingdomain_imex.py
===========================

Script version of 01_count_bindingdomain_IMEX.ipynb.

Takes the full IntAct MITAB 2.7 dump (intact.txt, all species in a single file)
and extract the
7 reference species datasets. For the whole dump and for each species it:

  1. extracts the interactions carrying >=1 annotated binding region
     (sufficient / necessary / direct / binding-associated region);
  2. counts binding-region types per species (side A, side B, total);
  3. maps InterPro IDs to the protein on which they are annotated;
  4. builds the InterPro domain ranking per species (+ a species x domain matrix).

Usage
-----
    python count_bindingdomain_imex.py \
        --intact intact.txt \
        --interpro-name INTERPRO_id_names.csv \
        --outdir output_directory

Options
-------
    --chunksize N       rows read per chunk from intact.txt (default 200000)
    --strict-interpro   take InterPro IDs only from the binding-region features AND only from
                        the interactor belonging to the target species (default: notebook
                        behaviour = any IPR in the Feature(s) column, both interactors)

Species / taxids are defined in the SPECIES dict below: edit it to add taxids
(e.g. 511145 for E. coli K-12 MG1655, 4932 for S. cerevisiae).

Output tree (inside --outdir)
-----------------------------
    global_summary.json                        counts on the whole dump
    intact_bindingdomain.csv                   all interactions with >=1 binding region
    binding_region_counts_byspecies.csv        species-only, "binding total" (as in the notebook)
    binding_region_counts_byspecies_full.csv   all collections / categories (total, ftA, ftB)
    interpro_counts_allspecies_long.csv        ranking of all species, long format
    interpro_counts_species_matrix.csv         INTERPRO_ID x species (n. proteins)
    per_species/<sp>_bindingdomain.csv         interactions with binding, per species
    per_species/<sp>_interpro_mapped.csv       unique (INTERPRO_ID, UNIPROT_ID) pairs
    per_species/<sp>_interpro_counts.csv       InterPro ranking (n. distinct proteins per domain)

Requirements: pandas, numpy
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# dict with the taxa to search for
SPECIES = {
    "human":      [9606],
    "mouse":      [10090],
    "c_elegans":  [6239],
    "e_coli":     [83333],
    "rat":        [10116],
    "yeast":      [559292],
    "drosophila": [7227],
}

# same pattern used for the per-species files
BINDING_PATTERN = r"(?:(?:direct|sufficient|necessary)\s+)?binding(?:-associated)? region[^|]*"

# counts, species-agnostic (full dump)
GLOBAL_TYPES = {
    "sufficient binding region": r"sufficient binding region",
    "necessary binding region":  r"necessary binding region",
    "binding-associated region": r"binding-associated region",
}

# binding region types to recognize (lowercase for robust comparison)
SPECIES_TYPES = {
    "sufficient binding region": r"sufficient binding region",
    "necessary binding region":  r"necessary binding region",
    "binding-associated region": r"binding-associated region",
    "direct binding region":     r"direct binding region",
}

IPR_PATTERN = r"IPR\d+"

# MITAB 2.7 columns used
ID_A, ID_B = "#ID(s) interactor A", "ID(s) interactor B"
TAX_A, TAX_B = "Taxid interactor A", "Taxid interactor B"
FEAT_A, FEAT_B = "Feature(s) interactor A", "Feature(s) interactor B"

# helper columns added by the script (never written to the CSV exports)
HELPER_COLS = ["matches_ftA", "matches_ftB", "has_bind_A", "has_bind_B",
               "is_A_target_species", "is_B_target_species"]


# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------

def taxid_regex(taxids):
    """taxid:9606 but not taxid:96060 (negative lookahead on the next digit)."""
    return r"taxid:(?:" + "|".join(str(t) for t in taxids) + r")(?!\d)"


def load_binding_subset(intact_path, chunksize):
    """
    Read intact.txt in chunks and keep only the interactions carrying >=1
    annotated binding region (side A or B). Along the way count, per species,
    ALL interactions (with or without binding) -> denominator 'total_interaction'.
    """
    species_rx = {sp: taxid_regex(t) for sp, t in SPECIES.items()}
    total_interaction = {sp: 0 for sp in SPECIES}
    n_rows = 0
    kept = []

    reader = pd.read_csv(intact_path, sep="\t", dtype=str, chunksize=chunksize)
    for i, chunk in enumerate(reader, 1):
        n_rows += len(chunk)

        # make string columns safe
        fA = chunk[FEAT_A].fillna("").astype(str)
        fB = chunk[FEAT_B].fillna("").astype(str)
        has_bind = (fA.str.contains(BINDING_PATTERN, flags=re.I, regex=True) |
                    fB.str.contains(BINDING_PATTERN, flags=re.I, regex=True))

        # species totals: interaction where interactor A or B belongs to the species
        tA = chunk[TAX_A].fillna("").astype(str)
        tB = chunk[TAX_B].fillna("").astype(str)
        for sp, rx in species_rx.items():
            total_interaction[sp] += int((tA.str.contains(rx, regex=True) |
                                          tB.str.contains(rx, regex=True)).sum())

        kept.append(chunk[has_bind])
        print(f"  chunk {i}: {n_rows:,} rows read", file=sys.stderr, end="\r")
    print(file=sys.stderr)

    bd = pd.concat(kept, ignore_index=True)
    return bd, n_rows, total_interaction


def add_match_columns(df):
    """matches per feature cell (list of matches per row) + boolean presence columns."""
    df[FEAT_A] = df[FEAT_A].fillna("").astype(str)
    df[FEAT_B] = df[FEAT_B].fillna("").astype(str)
    df[TAX_A] = df[TAX_A].fillna("").astype(str)
    df[TAX_B] = df[TAX_B].fillna("").astype(str)
    df["matches_ftA"] = df[FEAT_A].str.findall(BINDING_PATTERN, flags=re.I)
    df["matches_ftB"] = df[FEAT_B].str.findall(BINDING_PATTERN, flags=re.I)
    df["has_bind_A"] = df["matches_ftA"].str.len() > 0
    df["has_bind_B"] = df["matches_ftB"].str.len() > 0
    return df


def global_summary(bd, n_rows):
    """Counts on the whole dump (first cell of the notebook)."""
    # 1) total binding-region ANNOTATIONS = every match on both sides, across all rows
    n_annotations = int(bd["matches_ftA"].str.len().sum() + bd["matches_ftB"].str.len().sum())
    per_type = {name: int(bd[FEAT_A].str.count(patt, flags=re.I).sum() +
                          bd[FEAT_B].str.count(patt, flags=re.I).sum())
                for name, patt in GLOBAL_TYPES.items()}

    # 2) number of INTERACTIONS carrying >=1 binding annotation
    n_interactions = len(bd)

    # 3) how many species have at least one annotated binding domain
    #    = distinct taxids among interactions with binding (side A or B)
    tax = pd.concat([bd[TAX_A], bd[TAX_B]])
    taxids = tax.str.extractall(r"taxid:(-?\d+)")[0].astype(int)
    species = sorted(set(taxids[taxids > 0]))   # keep real organisms only (drop -1 in vitro, etc.)

    summary = {
        "rows_in_dump": n_rows,
        "binding_region_annotations_total": n_annotations,
        "per_type": per_type,
        "sum_of_canonical_types": sum(per_type.values()),
        "interactions_with_binding": n_interactions,
        "n_species_with_binding": len(species),
        "taxids_with_binding": [int(t) for t in species],
    }

    print(f"Binding-region annotations (total):        {n_annotations}")
    for k, v in per_type.items():
        print(f"    {k:32s} {v}")
    print(f"  sum of the 3 canonical types:            {sum(per_type.values())}")
    print(f"Interactions with >=1 binding annotation:  {n_interactions}")
    print(f"Species with >=1 annotated binding domain: {len(species)}")
    return summary


def species_subset(bd, taxids):
    """Interactions with binding (A or B) and at least one interactor of the target species."""
    rx = taxid_regex(taxids)
    isA = bd[TAX_A].str.contains(rx, regex=True)
    isB = bd[TAX_B].str.contains(rx, regex=True)
    df = bd[isA | isB].copy()
    # flag which interactor belongs to the target species (robust)
    df["is_A_target_species"] = isA[isA | isB]
    df["is_B_target_species"] = isB[isA | isB]
    return df


def count_binding_types(df, species_only):
    """
    For each binding type, number of rows where side A (resp. B) has >=1 region of that type.
    species_only=True -> side A counted only if A is the target species (same for B);
    species_only=False -> no species mask (the notebook's 'binding total' collection).
    """
    if species_only:
        maskA = df["is_A_target_species"].fillna(False).astype(bool)
        maskB = df["is_B_target_species"].fillna(False).astype(bool)
    else:
        maskA = pd.Series(True, index=df.index)
        maskB = pd.Series(True, index=df.index)

    # typesA / typesB = normalized list of matched region strings (lower/strip)
    typesA = df["matches_ftA"].apply(lambda L: [m.strip().lower() for m in L if m.strip()])
    typesB = df["matches_ftB"].apply(lambda L: [m.strip().lower() for m in L if m.strip()])

    out = {"binding total": {}, "binding ftA": {}, "binding ftB": {}}
    for key in SPECIES_TYPES:
        key_norm = key.lower()
        # side A: only rows where maskA is True, then check if any region contains key_norm
        nA = int(typesA[maskA].apply(lambda types: any(key_norm in t for t in types)).sum())
        nB = int(typesB[maskB].apply(lambda types: any(key_norm in t for t in types)).sum())
        out["binding ftA"][key] = nA
        out["binding ftB"][key] = nB
        out["binding total"][key] = nA + nB
    return out


def map_interpro(df, strict):
    """
    Map InterPro domains to the protein on which they are annotated.
    Do it independently for interactor A and interactor B, then concatenate,
    so each InterPro ID is attributed to the correct protein and no side is lost.
    """
    parts = []
    for id_col, feat_col, match_col, target_col in [
        (ID_A, FEAT_A, "matches_ftA", "is_A_target_species"),
        (ID_B, FEAT_B, "matches_ftB", "is_B_target_species"),
    ]:
        if strict:
            # only IPRs inside binding-region features, only on the target-species interactor
            src = df[match_col].apply(lambda L: "|".join(L))
            keep = df[target_col].astype(bool)
        else:
            # notebook behaviour: any IPR in the Feature(s) column, both interactors
            src = df[feat_col]
            keep = pd.Series(True, index=df.index)
        side = pd.DataFrame({"UNIPROT_ID": df[id_col],
                             "INTERPRO_ID": src.str.findall(IPR_PATTERN)})[keep]
        parts.append(side)

    # keep only rows that actually carry at least one InterPro ID
    mapped = pd.concat(parts, ignore_index=True)
    mapped = mapped[mapped["INTERPRO_ID"].str.len() > 0]
    mapped = mapped.explode("INTERPRO_ID")[["INTERPRO_ID", "UNIPROT_ID"]]
    return mapped.drop_duplicates().reset_index(drop=True)


def count_interpro(mapped, interpro_names):
    """Ranking: n. of distinct proteins carrying each InterPro domain."""
    cont_domain = (mapped.value_counts("INTERPRO_ID")
                   .rename("Count").reset_index())
    if interpro_names is not None:
        cont_domain = cont_domain.merge(interpro_names, on="INTERPRO_ID", how="left")
    else:
        cont_domain["INTERPRO_NAME"] = np.nan
    cont_domain.insert(0, "Rank", cont_domain["Count"].rank(method="min", ascending=False).astype(int))
    return cont_domain


def export(df, path):
    """Write CSV without the helper columns (lists/bools added by the script)."""
    df.drop(columns=[c for c in HELPER_COLS if c in df.columns]).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intact", required=True, help="full IntAct MITAB 2.7 dump (intact.txt)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--interpro-names", default=None,
                    help="TSV without header: INTERPRO_ID <tab> INTERPRO_NAME")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--strict-interpro", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    per_sp_dir = outdir / "per_species"
    per_sp_dir.mkdir(parents=True, exist_ok=True)

    # InterPro IDs that were already mapped in the dataset
    interpro_names = None
    if args.interpro_names:
        interpro_names = pd.read_csv(args.interpro_names, sep="\t", header=None, dtype=str)
        interpro_names.columns = ["INTERPRO_ID", "INTERPRO_NAME"]

    # --- full IntAct dump (all species in a single file) ---
    print(f"Reading {args.intact} ...", file=sys.stderr)
    bd, n_rows, total_interaction = load_binding_subset(args.intact, args.chunksize)
    bd = add_match_columns(bd)

    summary = global_summary(bd, n_rows)
    summary["total_interaction_per_species"] = total_interaction
    summary["species_taxids"] = SPECIES
    summary["strict_interpro"] = args.strict_interpro
    with open(outdir / "global_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # CSV: only interactions carrying >=1 annotated binding region
    export(bd, outdir / "intact_bindingdomain.csv")

    # --- per species ---
    data, rankings = [], []
    for sp, taxids in SPECIES.items():
        df_sp = species_subset(bd, taxids)
        print(f"{sp}: total {total_interaction[sp]}, filtered and species-only {df_sp.shape}")
        export(df_sp, per_sp_dir / f"{sp}_bindingdomain.csv")

        # counting features for each dataset (both collections of the notebook)
        for collection, species_only in [("imex dataframes binding total", False),
                                         ("imex dataframes species-only binding", True)]:
            counts = count_binding_types(df_sp, species_only)
            for binding_category, binding_type in counts.items():
                for type_binding, count in binding_type.items():
                    data.append({
                        "collection": collection,
                        "species": sp,
                        "binding_category": binding_category,
                        "binding_type": type_binding,
                        "count": count,
                        "total_binding": len(df_sp),
                        "total_interaction": total_interaction[sp],
                    })

        # InterPro mapping + ranking
        mapped = map_interpro(df_sp, args.strict_interpro)
        mapped.to_csv(per_sp_dir / f"{sp}_interpro_mapped.csv", index=False)
        cont_domain = count_interpro(mapped, interpro_names)
        cont_domain.to_csv(per_sp_dir / f"{sp}_interpro_counts.csv", index=False)
        rankings.append(cont_domain.assign(species=sp))


    # build the df_binding_counts dataframe
    df_binding_counts = pd.DataFrame(data)
    # percentage relative to total binding (ftA + ftB)
    df_binding_counts["total_fta+ftb"] = (df_binding_counts
        .groupby(["collection", "binding_category", "species"])["count"].transform("sum"))
    df_binding_counts["perc x binding"] = (df_binding_counts["count"] /
                                           df_binding_counts["total_fta+ftb"].replace(0, np.nan)) * 100
    df_binding_counts.to_csv(outdir / "binding_region_counts_byspecies_full.csv", index=False)

    # filter to save only "binding total" (species-only collection, as in the notebook)
    df_filtered = df_binding_counts[
        (df_binding_counts["collection"] == "imex dataframes species-only binding") &
        (df_binding_counts["binding_category"] == "binding total")]
    df_filtered.to_csv(outdir / "binding_region_counts_byspecies.csv", index=False)

    # rankings: long + species x domain matrix
    long = pd.concat(rankings, ignore_index=True)
    long.to_csv(outdir / "interpro_counts_allspecies_long.csv", index=False)
    matrix = (long.pivot_table(index="INTERPRO_ID", columns="species", values="Count",
                               aggfunc="sum", fill_value=0)
              .reindex(columns=list(SPECIES), fill_value=0))
    matrix["n_species"] = (matrix[list(SPECIES)] > 0).sum(axis=1)
    matrix["total"] = matrix[list(SPECIES)].sum(axis=1)
    matrix = matrix.reset_index()
    if interpro_names is not None:
        matrix = matrix.merge(interpro_names, on="INTERPRO_ID", how="left")
    matrix.sort_values("total", ascending=False).to_csv(
        outdir / "interpro_counts_species_matrix.csv", index=False)

    print(f"Done. Output in {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
