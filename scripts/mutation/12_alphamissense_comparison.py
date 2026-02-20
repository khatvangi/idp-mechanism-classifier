#!/usr/bin/env python3
"""
12_alphamissense_comparison.py — compare AlphaMissense with ESM2 LLR

downloads AlphaMissense predictions from Zenodo for our 23 genes,
merges with ClinVar variants, computes per-gene AUROC.

key question: does AlphaMissense also fail for GoF non-amyloid genes?
if yes → strengthens our finding (structural features don't help either)
if no → AlphaMissense captures something ESM2 LLR misses (structural context?)
"""

import os
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import urllib.request

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
VARIANTS = DATA / "variants"
AM_DIR = DATA / "alphamissense"
AM_DIR.mkdir(exist_ok=True)

# our 23 genes and their UniProt IDs
GENE_TO_UNIPROT = {
    "SNCA": "P37840", "APP": "P05067", "MAPT": "P10636", "TARDBP": "Q13148",
    "FUS": "P35637", "HNRNPA1": "P09651", "PRNP": "P04156", "HTT": "P42858",
    "TIA1": "P31483", "EWSR1": "Q01844", "TAF15": "Q92804", "HNRNPA2B1": "P22626",
    "ATXN3": "P54252", "DDX4": "Q9NQI0", "NPM1": "P06748", "IAPP": "P10997",
    "SOD1": "P00441", "TTR": "P02766", "LMNA": "P02545", "VCP": "P55072",
    "AR": "P10275", "SQSTM1": "Q13501", "CRYAB": "P02511",
}
UNIPROT_TO_GENE = {v: k for k, v in GENE_TO_UNIPROT.items()}

# mechanism groups
GOF_NONAMYLOID = {"FUS", "TARDBP", "HNRNPA1", "TIA1", "HNRNPA2B1", "EWSR1", "TAF15"}
GOF_AMYLOID = {"SNCA", "TTR", "PRNP", "IAPP"}
LOF_STRUCTURED = {"SOD1", "VCP", "LMNA", "CRYAB"}

AM_FILE = AM_DIR / "AlphaMissense_aa_substitutions.tsv.gz"
AM_URL = "https://zenodo.org/records/10813168/files/AlphaMissense_aa_substitutions.tsv.gz"


def download_alphamissense():
    """download the full AlphaMissense predictions (if not already present)."""
    if AM_FILE.exists():
        print(f"  AlphaMissense file already exists: {AM_FILE}")
        return

    print(f"  downloading AlphaMissense from Zenodo...")
    print(f"  URL: {AM_URL}")
    print(f"  this is a large file (~4 GB compressed) — may take a while")
    urllib.request.urlretrieve(AM_URL, AM_FILE)
    print(f"  downloaded to {AM_FILE}")


def extract_our_genes():
    """stream through the gzipped file, extract rows for our 23 proteins."""
    out_file = AM_DIR / "am_our_genes.csv"
    if out_file.exists():
        print(f"  filtered file already exists: {out_file}")
        return pd.read_csv(out_file)

    print(f"  streaming through AlphaMissense file (filtering to our genes)...")
    our_uniprots = set(GENE_TO_UNIPROT.values())
    rows = []
    n_total = 0

    with gzip.open(AM_FILE, "rt") as f:
        # skip comment lines starting with #
        for line in f:
            if line.startswith("#"):
                continue
            # header line
            header = line.strip().split("\t")
            break

        for line in f:
            n_total += 1
            parts = line.strip().split("\t")
            uniprot_id = parts[0]  # first column is uniprot accession

            if uniprot_id in our_uniprots:
                rows.append(parts)

            if n_total % 50_000_000 == 0:
                print(f"    processed {n_total:,} lines, found {len(rows)} matches...")

    print(f"  processed {n_total:,} total lines, found {len(rows)} for our genes")

    df = pd.DataFrame(rows, columns=header)
    df.to_csv(out_file, index=False)
    print(f"  saved to {out_file}")
    return df


def parse_variant(variant_str):
    """parse 'A123B' format into (ref_aa, position, alt_aa)."""
    ref = variant_str[0]
    alt = variant_str[-1]
    pos = int(variant_str[1:-1])
    return ref, pos, alt


if __name__ == "__main__":
    # step 1: download
    print("step 1: downloading AlphaMissense...")
    download_alphamissense()

    # step 2: extract our genes
    print("\nstep 2: extracting our 23 genes...")
    am = extract_our_genes()
    print(f"  {len(am)} AlphaMissense predictions for our genes")

    # parse variant column
    # column name varies — check what we got
    print(f"  columns: {list(am.columns)}")

    # typical columns: uniprot_id, protein_variant, am_pathogenicity, am_class
    # find the variant column
    var_col = None
    for col in am.columns:
        if "variant" in col.lower():
            var_col = col
            break
    if var_col is None:
        # try common alternatives
        for col in am.columns:
            sample = str(am[col].iloc[0])
            if len(sample) >= 3 and sample[0].isalpha() and sample[-1].isalpha():
                var_col = col
                break

    # find pathogenicity score column
    score_col = None
    for col in am.columns:
        if "pathogenicity" in col.lower() and "class" not in col.lower():
            score_col = col
            break
        elif "score" in col.lower():
            score_col = col
            break

    # find uniprot column
    uniprot_col = am.columns[0]  # usually first

    print(f"  uniprot column: {uniprot_col}")
    print(f"  variant column: {var_col}")
    print(f"  score column: {score_col}")

    if var_col is None or score_col is None:
        print("  ERROR: could not identify variant or score columns")
        print(f"  sample row: {am.iloc[0].to_dict()}")
        exit(1)

    # parse variants
    parsed = am[var_col].apply(parse_variant)
    am["ref_aa"] = [p[0] for p in parsed]
    am["position"] = [p[1] for p in parsed]
    am["alt_aa"] = [p[2] for p in parsed]
    am["am_score"] = pd.to_numeric(am[score_col], errors="coerce")
    am["gene"] = am[uniprot_col].map(UNIPROT_TO_GENE)

    # step 3: merge with our ClinVar variants
    print("\nstep 3: merging with ClinVar variants...")
    variants = pd.read_csv(VARIANTS / "esm2_features.csv")
    # exclude HTT
    variants = variants[variants["gene"] != "HTT"].copy()
    print(f"  ClinVar variants (excl HTT): {len(variants)}")

    merged = variants.merge(
        am[["gene", "position", "alt_aa", "am_score"]],
        on=["gene", "position", "alt_aa"],
        how="left"
    )

    n_matched = merged["am_score"].notna().sum()
    print(f"  matched: {n_matched}/{len(merged)} ({n_matched/len(merged):.1%})")

    # step 4: compute AUROCs
    print(f"\n{'='*65}")
    print("AlphaMissense vs ESM2 LLR comparison")
    print(f"{'='*65}")

    valid = merged.dropna(subset=["am_score"])
    y = valid["target"].values

    if y.sum() > 0 and (1 - y).sum() > 0:
        auroc_am = roc_auc_score(y, valid["am_score"])
        auroc_esm2 = roc_auc_score(y, valid["esm2_llr"])
        print(f"\n  overall (n={len(valid)}, P={int(y.sum())}):")
        print(f"    AlphaMissense:  AUROC={auroc_am:.4f}")
        print(f"    ESM2 LLR:       AUROC={auroc_esm2:.4f}")

    # per-gene
    print(f"\n  {'gene':<12} {'n':>5} {'P':>4} {'AM_AUROC':>10} {'ESM2_AUROC':>12}")
    print("  " + "-" * 50)

    gene_results = []
    for gene in sorted(valid["gene"].unique()):
        gdf = valid[valid["gene"] == gene]
        gy = gdf["target"].values
        n_path = int(gy.sum())

        if n_path == 0 or n_path == len(gy):
            print(f"  {gene:<12} {len(gdf):>5} {n_path:>4}       n/a")
            continue

        am_auc = roc_auc_score(gy, gdf["am_score"])
        esm2_auc = roc_auc_score(gy, gdf["esm2_llr"])
        delta = am_auc - esm2_auc

        print(f"  {gene:<12} {len(gdf):>5} {n_path:>4} {am_auc:>10.4f} {esm2_auc:>12.4f}  "
              f"{'↑' if delta > 0.02 else '↓' if delta < -0.02 else '≈'}{abs(delta):.3f}")

        gene_results.append({"gene": gene, "n": len(gdf), "P": n_path,
                             "am_auroc": am_auc, "esm2_auroc": esm2_auc,
                             "delta": delta})

    # per-mechanism
    mechanisms = {
        "gof_nonamyloid": GOF_NONAMYLOID,
        "gof_amyloid": GOF_AMYLOID,
        "lof_structured": LOF_STRUCTURED,
    }

    print(f"\n  per-mechanism:")
    for mech, genes in mechanisms.items():
        mdf = valid[valid["gene"].isin(genes)]
        my = mdf["target"].values
        if my.sum() > 0 and (1 - my).sum() > 0:
            am_auc = roc_auc_score(my, mdf["am_score"])
            esm2_auc = roc_auc_score(my, mdf["esm2_llr"])
            print(f"    {mech:<20} AM={am_auc:.4f}  ESM2={esm2_auc:.4f}  "
                  f"(n={len(mdf)}, P={int(my.sum())})")

    # save
    pd.DataFrame(gene_results).to_csv(VARIANTS / "alphamissense_comparison.csv", index=False)
    merged.to_csv(VARIANTS / "variants_with_alphamissense.csv", index=False)
    print(f"\n  saved to {VARIANTS / 'alphamissense_comparison.csv'}")
    print(f"  saved to {VARIANTS / 'variants_with_alphamissense.csv'}")
