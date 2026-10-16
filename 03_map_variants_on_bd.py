#!/usr/bin/env python3
"""
map_variants_on_bd.py
=====================

Step 3 of the binding-domain pipeline. Script version of 03_LP_VUS_mapping.ipynb.

Maps UniProt humsavar missense variants (LP/P and US) onto:
  1. the IMEx binding regions (variant position inside the binding region);
  2. the overlap between binding region and InterPro domain (variant position inside
     the intersection of the two ranges), full mapping and 100x100 subset;
then counts the top diseases and runs a per-disease enrichment test (are the LP/P
variants of a disease more often inside binding regions than the others?).

Variant positions (humsavar) are numbered on the canonical sequence, so isoform-keyed
binding regions (P12345-2) are NOT collapsed onto the canonical: they simply do not match
the canonical Swiss-Prot AC and are left out (mapping them would mix two coordinate systems).

Usage
-----
    python map_variants_on_bd.py \
        --step2-dir INTERPROtoBD_mapping \
        --humsavar  humsavar.txt  \
        --outdir    Variant_mapping
humsavar.txt file retrieved from uniprot
INTERPROtoBD_mapping output_directory of the 2 step
Options
-------
    --interpro-mapped FILE   InterPro-BD mapping to use for the overlap
                             (default: <step2-dir>/INTERPRO_mapped.csv; e.g. pass
                             INTERPRO_mapped_filtered.csv to work on a filtered mapping)
    --top-diseases N         number of diseases in the top-disease tables (default 20)
    --min-variants N         minimum variants per disease in the enrichment test (default 10)

Input
-----
    <step2-dir>/bindingregions_tomap.csv   binding regions per interactor (from map_interpro_on_bd.py)
    <step2-dir>/INTERPRO_mapped.csv        InterPro domains mapped on binding regions
    humsavar.txt                           ';'-separated, header:
                                           Gene_name;Swiss-Prot_AC;FTId;AA_change;Variant_category;dbSNP;Disease_name

Output (--outdir)
-----------------
    Variant_onBD/LP_onBD.csv                          LP/P variants inside binding regions
    Variant_onBD/VUS_onBD.csv                         US variants inside binding regions
    Variant_onINTERPRO_BD/LPonINTERPROBD_overlap.csv  LP/P inside binding region AND InterPro domain
    Variant_onINTERPRO_BD/USonINTERPROBD_overlap.csv  US   inside binding region AND InterPro domain
    Variant_onINTERPRO_BD/LPonINTERPROBD_100.csv      same, only 100x100 domain-region pairs
    Variant_onINTERPRO_BD/USonINTERPROBD_100.csv
    top_disease_LP.csv / top_disease_US.csv           top diseases on the overlap
    ORA_disease_bindingregion.csv                     per-disease enrichment inside binding regions
    top_disease_protein_coverage.csv                  binding-region coverage of the proteins of the top diseases
    variant_summary.json                              mapping rates and counts

The mapped tables have one row per (interaction, binding region[, domain], variant): the same
variant appears several times if its protein has several interactions/regions. All counts in
the summary are on UNIQUE variants (Swiss-Prot_AC + AA_change).

Requirements: pandas, numpy, scipy, statsmodels
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# humsavar category -> prefixes of the output files (as in the notebook)
CATEGORIES = {
    "LP/P": {"bd": "LP_onBD.csv",  "overlap": "LPonINTERPROBD_overlap.csv",
             "100": "LPonINTERPROBD_100.csv", "top": "top_disease_LP.csv"},
    "US":   {"bd": "VUS_onBD.csv", "overlap": "USonINTERPROBD_overlap.csv",
             "100": "USonINTERPROBD_100.csv", "top": "top_disease_US.csv"},
}
VAR_KEY = ["Swiss-Prot_AC", "AA_change"]


# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------

def load_variants(path):
    # Gene_name;Swiss-Prot_AC;FTId;AA_change;Variant_category;dbSNP;Disease_name
    variants = pd.read_csv(path, delimiter=";", dtype=str)
    # new column PositionAA: extract the substitution position to map it onto the binding domain
    variants["PositionAA"] = pd.to_numeric(variants.AA_change.str.extract(r"(\d+)", expand=False),
                                           errors="coerce")
    variants = variants[variants["PositionAA"].notna()].copy()
    variants["PositionAA"] = variants["PositionAA"].astype(int)
    return variants


def load_binding_regions(path):
    bd = pd.read_csv(path)
    # C-terminal ends without a length (step 2 --range-mode termini) cannot be used here
    bd = bd.dropna(subset=["FEATURE_RANGE_START", "FEATURE_RANGE_END"]).copy()
    bd["FEATURE_RANGE_START"] = bd["FEATURE_RANGE_START"].astype(int)
    bd["FEATURE_RANGE_END"] = bd["FEATURE_RANGE_END"].astype(int)
    return bd.drop(columns=[c for c in ["END_IS_CTERM"] if c in bd.columns])


# ---------------------------------------------------------------------------
# MAPPING
# ---------------------------------------------------------------------------

def variants_on_bd(binding_domain, var):
    """Keep variants whose AA position falls inside the binding region (canonical coordinates only)."""
    m = binding_domain.merge(var, left_on="UNIPROT_ID", right_on="Swiss-Prot_AC", how="inner")
    condition = ((m.PositionAA >= m.FEATURE_RANGE_START) & (m.PositionAA <= m.FEATURE_RANGE_END))
    return m[condition].copy()


def variants_on_overlap(interpro_mapped, var):
    """
    Approach 2 (overlap): keep variants whose AA position falls inside the intersection
    of the InterPro domain and the binding region.
    """
    m = interpro_mapped.merge(var, left_on="UNIPROT_ID", right_on="Swiss-Prot_AC", how="inner")
    condition = (
        (m.PositionAA <= m[["INTERPRO_RANGE_END", "FEATURE_RANGE_END"]].astype(int).min(axis=1)) &
        (m.PositionAA >= m[["INTERPRO_RANGE_START", "FEATURE_RANGE_START"]].astype(int).max(axis=1))
    )
    return m[condition].copy()


def clean_bd_table(df):
    df = df.drop(columns=["Swiss-Prot_AC", "PositionAA"])
    df.insert(2, "Gene_name", df.pop("Gene_name"))
    return df


def top_diseases(overlap, n):
    """
    top diseases among variants mapped on the overlap (InterPro intersect binding region).
    count             = notebook definition: unique (variant, binding region) pairs
                        -> a variant inside k distinct regions is counted k times
    n_unique_variants = each variant counted once
    """
    key = ["UNIPROT_ID", "AA_change", "FEATURE_RANGE_START", "FEATURE_RANGE_END"]
    dedup = overlap.drop_duplicates(subset=key)
    dedup = dedup[dedup["Disease_name"] != "-"]
    counts = dedup.value_counts("Disease_name").rename("count")
    uniq = (dedup.drop_duplicates(subset=["UNIPROT_ID", "AA_change"])
            .value_counts("Disease_name").rename("n_unique_variants"))
    prots = dedup.groupby("Disease_name")["UNIPROT_ID"].nunique().rename("n_proteins")
    out = pd.concat([counts, uniq, prots], axis=1).fillna(0).astype(int)
    return out.sort_values("count", ascending=False).head(n).reset_index()


# ---------------------------------------------------------------------------
# ENRICHMENT ANALYSIS OF THE AA IN THE BINDING DOMAIN RESPECT TO THE OTHER VARIANTS FALLING OUTSIDE
# ---------------------------------------------------------------------------

def ora_disease(binding_domain, lp, min_variants):
    # --- universe: unique LP/P variants on proteins carrying at least one binding region ---
    prot_bd = set(binding_domain["UNIPROT_ID"])
    lp_u = lp.drop_duplicates(subset=VAR_KEY).copy()
    univ = lp_u[lp_u["Swiss-Prot_AC"].isin(prot_bd)].copy()

    # --- label INSIDE: the variant falls within at least one binding region of its protein ---
    hit = variants_on_bd(binding_domain, univ)
    inside_keys = set(map(tuple, hit[VAR_KEY].values))
    univ["INSIDE_BD"] = [(a, c) in inside_keys for a, c in zip(univ["Swiss-Prot_AC"], univ["AA_change"])]
    univ = univ[univ["Disease_name"] != "-"]

    # --- Fisher's exact test per disease: that disease against all the others ---
    tot_in = int(univ["INSIDE_BD"].sum())
    tot_out = int((~univ["INSIDE_BD"]).sum())

    rows = []
    for dz, sub in univ.groupby("Disease_name"):
        a = int(sub["INSIDE_BD"].sum())          # this disease, inside
        b = int((~sub["INSIDE_BD"]).sum())       # this disease, outside
        if (a + b) < min_variants:
            continue
        c = tot_in - a                           # all other diseases, inside
        d = tot_out - b                          # all other diseases, outside
        OR, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        # Haldane-Anscombe correction when any cell is zero (same as the InterPro ORA)
        if min(a, b, c, d) == 0:
            OR = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        frac_in = a / (a + b)
        frac_bg = tot_in / (tot_in + tot_out) if (tot_in + tot_out) else np.nan
        rows.append(dict(Disease_name=dz, n_inside=a, n_outside=b, n_total=a + b,
                         n_proteins=int(sub["Swiss-Prot_AC"].nunique()),
                         frac_inside=frac_in, fold_enrichment=frac_in / frac_bg if frac_bg else np.nan,
                         odds_ratio=OR, log2_odds_ratio=np.log2(OR), p_value=p))

    ora_dz = pd.DataFrame(rows)
    if len(ora_dz):
        ora_dz["fdr"] = multipletests(ora_dz["p_value"], method="fdr_bh")[1]
        ora_dz = ora_dz.sort_values("log2_odds_ratio", ascending=False).reset_index(drop=True)
    stats = {"universe_variants": len(univ), "inside": tot_in, "outside": tot_out,
             "background_rate_inside": round(tot_in / len(univ), 4) if len(univ) else None,
             "diseases_tested": len(ora_dz),
             "significant_fdr_0.05": int((ora_dz.fdr < 0.05).sum()) if len(ora_dz) else 0}
    return ora_dz, univ, stats


def protein_coverage(binding_domain, univ, ora_dz, n_top=10):
    """how much of each protein is covered by binding regions? (for the top-scoring diseases)"""
    if ora_dz.empty:
        return pd.DataFrame()
    top_dz = ora_dz.head(n_top)["Disease_name"].tolist()
    rows = []
    for dz in top_dz:
        for p in univ[univ["Disease_name"] == dz]["Swiss-Prot_AC"].unique():
            bd_p = binding_domain[binding_domain["UNIPROT_ID"] == p]
            if len(bd_p) == 0:
                continue
            covered = set()
            for s, e in zip(bd_p.FEATURE_RANGE_START, bd_p.FEATURE_RANGE_END):
                covered.update(range(s, e + 1))
            span = bd_p.FEATURE_RANGE_END.max() - bd_p.FEATURE_RANGE_START.min() + 1
            rows.append(dict(Disease_name=dz, UNIPROT_ID=p, n_regions=len(bd_p),
                             n_distinct_regions=len(bd_p[["FEATURE_RANGE_START", "FEATURE_RANGE_END"]].drop_duplicates()),
                             covered_aa=len(covered), span=int(span)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step2-dir", required=True, type=Path)
    ap.add_argument("--humsavar", required=True)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--interpro-mapped", default=None)
    ap.add_argument("--top-diseases", type=int, default=20)
    ap.add_argument("--min-variants", type=int, default=10)
    args = ap.parse_args()

    out_bd = args.outdir / "Variant_onBD"
    out_ov = args.outdir / "Variant_onINTERPRO_BD"
    out_bd.mkdir(parents=True, exist_ok=True)
    out_ov.mkdir(parents=True, exist_ok=True)

    binding_domain = load_binding_regions(args.step2_dir / "bindingregions_tomap.csv")
    interpro_mapped = pd.read_csv(args.interpro_mapped or args.step2_dir / "INTERPRO_mapped.csv")
    variants = load_variants(args.humsavar)

    summary = {"inputs": {"step2_dir": str(args.step2_dir), "humsavar": args.humsavar,
                          "interpro_mapped": str(args.interpro_mapped or args.step2_dir / "INTERPRO_mapped.csv")},
               "humsavar_categories": variants.value_counts("Variant_category").to_dict()}
    prot_bd = set(binding_domain["UNIPROT_ID"])

    for cat, files in CATEGORIES.items():
        var = variants[variants["Variant_category"] == cat].copy()
        var_u = var.drop_duplicates(subset=VAR_KEY)

        # --- variants on binding regions ---
        bd_var = variants_on_bd(binding_domain, var)
        n_on_bd_prot = int(var_u["Swiss-Prot_AC"].isin(prot_bd).sum())
        n_bd = len(bd_var.drop_duplicates(subset=VAR_KEY))
        clean_bd_table(bd_var).to_csv(out_bd / files["bd"], index=None)

        # --- variants on the overlap binding region x InterPro domain ---
        ov = variants_on_overlap(interpro_mapped, var)
        ov_100 = ov[(ov["PERCENTAGE_OF_FEATURE"] == 100.0) & (ov["PERCENTAGE_OF_DOMAIN"] == 100.0)].copy()
        top_diseases(ov, args.top_diseases).to_csv(args.outdir / files["top"], index=False)
        ov.drop(columns=["Swiss-Prot_AC", "PositionAA"]).to_csv(out_ov / files["overlap"], index=None)
        ov_100.drop(columns=["Swiss-Prot_AC", "PositionAA"]).to_csv(out_ov / files["100"], index=None)

        summary[cat] = {
            "A_all_unique": len(var_u),                          # denominator A: all variants
            "B_on_proteins_with_binding_region": n_on_bd_prot,   # denominator B
            "inside_binding_region": n_bd,
            "mapping_rate_on_A": round(n_bd / len(var_u), 4) if len(var_u) else None,
            "mapping_rate_on_B": round(n_bd / n_on_bd_prot, 4) if n_on_bd_prot else None,
            "proteins_inside_binding_region": int(bd_var["UNIPROT_ID"].nunique()),
            "inside_overlap_interpro": len(ov.drop_duplicates(subset=VAR_KEY)),
            "proteins_inside_overlap_interpro": int(ov["UNIPROT_ID"].nunique()),
            "inside_overlap_100x100": len(ov_100.drop_duplicates(subset=VAR_KEY)),
            "proteins_inside_overlap_100x100": int(ov_100["UNIPROT_ID"].nunique()),
        }
        s = summary[cat]
        print(f"[{cat}] A. all (unique): {s['A_all_unique']} | B. on proteins with binding region: "
              f"{s['B_on_proteins_with_binding_region']}")
        print(f"      inside binding region: {n_bd} ({s['mapping_rate_on_A']:.1%} of A, "
              f"{(s['mapping_rate_on_B'] or 0):.1%} of B) in {s['proteins_inside_binding_region']} proteins")
        print(f"      inside BD x InterPro overlap: {s['inside_overlap_interpro']} "
              f"({s['proteins_inside_overlap_interpro']} proteins); 100x100: {s['inside_overlap_100x100']} "
              f"({s['proteins_inside_overlap_100x100']} proteins)")

    # --- per-disease enrichment (LP/P) ---
    lp = variants[variants["Variant_category"] == "LP/P"].copy()
    ora_dz, univ, stats = ora_disease(binding_domain, lp, args.min_variants)
    ora_dz.to_csv(args.outdir / "ORA_disease_bindingregion.csv", index=False)
    protein_coverage(binding_domain, univ, ora_dz).to_csv(
        args.outdir / "top_disease_protein_coverage.csv", index=False)
    summary["ora_disease_LP"] = {"min_variants": args.min_variants, **stats}
    print(f"ORA diseases: {stats['diseases_tested']} tested, {stats['significant_fdr_0.05']} with FDR < 0.05 "
          f"(background inside rate {stats['background_rate_inside']})")

    with open(args.outdir / "variant_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
