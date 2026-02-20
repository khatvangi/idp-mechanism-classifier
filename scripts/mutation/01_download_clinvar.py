#!/usr/bin/env python3
"""
task 2: download ClinVar variant_summary.txt.gz, extract missense variants
for our 23 IDP-related genes. outputs a clean CSV with protein positions.
"""

import gzip
import os
import re
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
VARIANTS = DATA / "variants"
VARIANTS.mkdir(parents=True, exist_ok=True)

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
CLINVAR_LOCAL = DATA / "variant_summary.txt.gz"

# our 23 genes
TARGET_GENES = {
    "SNCA", "APP", "MAPT", "TARDBP", "FUS", "HNRNPA1", "PRNP", "HTT",
    "TIA1", "EWSR1", "TAF15", "HNRNPA2B1", "ATXN3", "DDX4", "NPM1", "IAPP",
    "SOD1", "TTR", "LMNA", "VCP", "AR", "SQSTM1", "CRYAB",
}

# 3-letter to 1-letter AA code
AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Sec": "U", "Pyl": "O", "Ter": "*",
}

# regex to extract protein change: p.Xxx123Yyy or p.(Xxx123Yyy)
PROTEIN_CHANGE_RE = re.compile(r"p\.?\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?")

# clinical significance mapping
def classify_significance(clinsig):
    """map ClinVar clinical significance to our labels."""
    clinsig_lower = clinsig.lower()
    if "pathogenic" in clinsig_lower and "likely" not in clinsig_lower and "conflicting" not in clinsig_lower:
        return "pathogenic"
    elif "likely pathogenic" in clinsig_lower and "conflicting" not in clinsig_lower:
        return "likely_pathogenic"
    elif "benign" in clinsig_lower and "likely" not in clinsig_lower and "conflicting" not in clinsig_lower:
        return "benign"
    elif "likely benign" in clinsig_lower and "conflicting" not in clinsig_lower:
        return "likely_benign"
    elif "uncertain" in clinsig_lower:
        return "vus"
    elif "conflicting" in clinsig_lower:
        return "conflicting"
    else:
        return "other"


# review status to star rating
def review_stars(review_status):
    """convert review status text to star count."""
    rs = review_status.lower()
    if "practice guideline" in rs:
        return 4
    if "reviewed by expert panel" in rs:
        return 3
    if "criteria provided, multiple submitters, no conflicts" in rs:
        return 2
    if "criteria provided, conflicting" in rs:
        return 1
    if "criteria provided, single submitter" in rs:
        return 1
    return 0  # no assertion criteria


def download_clinvar():
    """download variant_summary.txt.gz if not cached."""
    if CLINVAR_LOCAL.exists():
        size_mb = CLINVAR_LOCAL.stat().st_size / 1e6
        print(f"  using cached {CLINVAR_LOCAL} ({size_mb:.1f} MB)")
        return
    print(f"  downloading {CLINVAR_URL}...")
    urllib.request.urlretrieve(CLINVAR_URL, CLINVAR_LOCAL)
    size_mb = CLINVAR_LOCAL.stat().st_size / 1e6
    print(f"  saved to {CLINVAR_LOCAL} ({size_mb:.1f} MB)")


def parse_clinvar():
    """parse variant_summary.txt.gz, extract missense variants for target genes."""
    print("  parsing variant_summary.txt.gz...")

    rows = []
    n_total = 0
    n_gene_match = 0
    n_snv = 0
    n_protein_parsed = 0

    with gzip.open(CLINVAR_LOCAL, "rt") as f:
        header = f.readline().strip().split("\t")
        # find column indices
        col_idx = {name.strip("#"): i for i, name in enumerate(header)}

        for line in f:
            n_total += 1
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                continue

            gene = fields[col_idx.get("GeneSymbol", -1)]
            if gene not in TARGET_GENES:
                continue
            n_gene_match += 1

            # only single nucleotide variants
            var_type = fields[col_idx.get("Type", -1)]
            if var_type != "single nucleotide variant":
                continue
            n_snv += 1

            # parse protein change from Name field
            name_field = fields[col_idx.get("Name", -1)]
            match = PROTEIN_CHANGE_RE.search(name_field)
            if not match:
                continue

            ref_aa3, pos_str, alt_aa3 = match.groups()
            ref_aa = AA3_TO_1.get(ref_aa3)
            alt_aa = AA3_TO_1.get(alt_aa3)
            if not ref_aa or not alt_aa or ref_aa == "*" or alt_aa == "*":
                continue  # skip nonsense
            if ref_aa == alt_aa:
                continue  # skip synonymous

            position = int(pos_str)
            n_protein_parsed += 1

            clinsig = fields[col_idx.get("ClinicalSignificance", -1)]
            review_status = fields[col_idx.get("ReviewStatus", -1)]
            phenotype = fields[col_idx.get("PhenotypeList", -1)]
            variation_id = fields[col_idx.get("VariationID", -1)]

            label = classify_significance(clinsig)
            stars = review_stars(review_status)

            rows.append({
                "gene": gene,
                "position": position,
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
                "label": label,
                "clinical_significance": clinsig,
                "review_stars": stars,
                "phenotype": phenotype,
                "variation_id": variation_id,
            })

    print(f"  total lines: {n_total:,}")
    print(f"  gene matches: {n_gene_match:,}")
    print(f"  SNVs: {n_snv:,}")
    print(f"  protein change parsed: {n_protein_parsed:,}")
    print(f"  rows extracted: {len(rows):,}")

    return pd.DataFrame(rows)


def deduplicate(df):
    """
    deduplicate: same gene+position+alt_aa may appear multiple times
    (different submissions). keep the row with highest confidence label.
    priority: pathogenic > likely_pathogenic > vus > likely_benign > benign
    """
    label_priority = {
        "pathogenic": 5,
        "likely_pathogenic": 4,
        "conflicting": 3,
        "vus": 2,
        "likely_benign": 1,
        "benign": 0,
        "other": -1,
    }
    df["label_priority"] = df["label"].map(label_priority)
    # also prefer higher star rating
    df["sort_key"] = df["label_priority"] * 10 + df["review_stars"]

    # keep highest priority per (gene, position, alt_aa)
    df = df.sort_values("sort_key", ascending=False)
    df = df.drop_duplicates(subset=["gene", "position", "ref_aa", "alt_aa"], keep="first")
    df = df.drop(columns=["label_priority", "sort_key"])

    return df


def main():
    print("=" * 60)
    print("ClinVar missense variant extraction for 23 IDP genes")
    print("=" * 60)

    # step 1: download
    print("\nstep 1: download ClinVar")
    download_clinvar()

    # step 2: parse
    print("\nstep 2: parse and extract")
    df = parse_clinvar()

    # step 3: filter by review status (>=1 star)
    print(f"\nstep 3: filter by review status")
    print(f"  before filter: {len(df)} variants")
    print(f"  star distribution: {df['review_stars'].value_counts().sort_index().to_dict()}")
    df_filtered = df[df["review_stars"] >= 1].copy()
    print(f"  after >=1 star: {len(df_filtered)} variants")

    # step 4: deduplicate
    print(f"\nstep 4: deduplicate")
    df_dedup = deduplicate(df_filtered)
    print(f"  after dedup: {len(df_dedup)} unique variants")

    # step 5: summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # collapse labels for training: P+LP = pathogenic, B+LB = benign
    df_dedup["train_label"] = df_dedup["label"].map({
        "pathogenic": "pathogenic",
        "likely_pathogenic": "pathogenic",
        "benign": "benign",
        "likely_benign": "benign",
        "vus": "vus",
        "conflicting": "conflicting",
        "other": "other",
    })

    print(f"\n  label distribution:")
    for label, count in df_dedup["train_label"].value_counts().items():
        print(f"    {label:<15} {count:>6}")

    print(f"\n  per-gene breakdown (pathogenic / benign / vus / other):")
    print(f"  {'gene':<12} {'path':>6} {'benign':>6} {'vus':>6} {'other':>6} {'total':>6}")
    print("  " + "-" * 50)
    for gene in sorted(TARGET_GENES):
        gdf = df_dedup[df_dedup["gene"] == gene]
        tl = gdf["train_label"]
        n_path = (tl == "pathogenic").sum()
        n_ben = (tl == "benign").sum()
        n_vus = (tl == "vus").sum()
        n_other = ((tl == "conflicting") | (tl == "other")).sum()
        print(f"  {gene:<12} {n_path:>6} {n_ben:>6} {n_vus:>6} {n_other:>6} {len(gdf):>6}")

    # step 6: save
    out_path = VARIANTS / "clinvar_idp_missense.csv"
    df_dedup.to_csv(out_path, index=False)
    print(f"\n  saved {len(df_dedup)} variants to {out_path}")

    # also save training-ready subset (pathogenic + benign only)
    train_df = df_dedup[df_dedup["train_label"].isin(["pathogenic", "benign"])].copy()
    train_path = VARIANTS / "clinvar_train_ready.csv"
    train_df.to_csv(train_path, index=False)
    print(f"  saved {len(train_df)} training-ready variants to {train_path}")
    print(f"    pathogenic: {(train_df['train_label'] == 'pathogenic').sum()}")
    print(f"    benign: {(train_df['train_label'] == 'benign').sum()}")

    # spot checks
    print("\n  spot checks:")
    snca_a53t = df_dedup[(df_dedup["gene"] == "SNCA") & (df_dedup["position"] == 53)]
    if len(snca_a53t) > 0:
        print(f"    SNCA A53T: label={snca_a53t.iloc[0]['label']}, stars={snca_a53t.iloc[0]['review_stars']} ✓")
    mapt_p301l = df_dedup[(df_dedup["gene"] == "MAPT") & (df_dedup["position"] == 301)]
    if len(mapt_p301l) > 0:
        print(f"    MAPT P301L: label={mapt_p301l.iloc[0]['label']}, stars={mapt_p301l.iloc[0]['review_stars']} ✓")
    fus_p525l = df_dedup[(df_dedup["gene"] == "FUS") & (df_dedup["position"] == 525)]
    if len(fus_p525l) > 0:
        print(f"    FUS P525L: label={fus_p525l.iloc[0]['label']}, stars={fus_p525l.iloc[0]['review_stars']} ✓")


if __name__ == "__main__":
    main()
