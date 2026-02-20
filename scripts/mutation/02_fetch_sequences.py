#!/usr/bin/env python3
"""
task 3: fetch UniProt canonical sequences for all 23 genes.
cross-validate ClinVar protein positions against UniProt sequences.
"""

import csv
import re
import time
from pathlib import Path

import requests
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
SEQS_DIR = PROJECT / "data" / "sequences"
SEQS_DIR.mkdir(parents=True, exist_ok=True)
VARIANTS = PROJECT / "data" / "variants"

# gene → UniProt ID mapping
# using canonical (longest/most referenced) isoforms
GENE_UNIPROT = {
    "SNCA": "P37840",
    "APP": "P05067",
    "MAPT": "P10636",
    "TARDBP": "Q13148",
    "FUS": "P35637",
    "HNRNPA1": "P09651",
    "PRNP": "P04156",
    "HTT": "P42858",
    "TIA1": "P31483",
    "EWSR1": "Q01844",
    "TAF15": "Q92804",
    "HNRNPA2B1": "P22626",
    "ATXN3": "P54252",
    "DDX4": "Q9NQI0",
    "NPM1": "P06748",
    "IAPP": "P10997",
    "SOD1": "P00441",
    "TTR": "P02766",
    "LMNA": "P02545",
    "VCP": "P55072",
    "AR": "P10275",
    "SQSTM1": "Q13501",
    "CRYAB": "P02511",
}


def fetch_uniprot_fasta(uniprot_id):
    """fetch FASTA sequence from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    header = lines[0]
    seq = "".join(lines[1:])
    return header, seq


def cross_validate_positions(variants_df, sequences):
    """check that ClinVar ref_aa matches UniProt sequence at stated position."""
    n_match = 0
    n_mismatch = 0
    n_out_of_range = 0
    n_no_seq = 0
    mismatches = []

    for _, row in variants_df.iterrows():
        gene = row["gene"]
        pos = row["position"]
        ref_aa = row["ref_aa"]

        if gene not in sequences:
            n_no_seq += 1
            continue

        seq = sequences[gene]
        if pos < 1 or pos > len(seq):
            n_out_of_range += 1
            continue

        # ClinVar uses 1-based positions
        seq_aa = seq[pos - 1]
        if seq_aa == ref_aa:
            n_match += 1
        else:
            n_mismatch += 1
            if len(mismatches) < 20:
                mismatches.append((gene, pos, ref_aa, seq_aa))

    total = n_match + n_mismatch + n_out_of_range + n_no_seq
    print(f"\n  cross-validation results:")
    print(f"    match: {n_match} ({n_match / total * 100:.1f}%)")
    print(f"    mismatch: {n_mismatch} ({n_mismatch / total * 100:.1f}%)")
    print(f"    out of range: {n_out_of_range} ({n_out_of_range / total * 100:.1f}%)")
    print(f"    no sequence: {n_no_seq}")

    if mismatches:
        print(f"\n  sample mismatches (likely isoform issues):")
        for gene, pos, ref, seq_aa in mismatches[:10]:
            print(f"    {gene} pos {pos}: ClinVar={ref}, UniProt={seq_aa}")

    return n_match, n_mismatch, n_out_of_range


def main():
    print("=" * 60)
    print("fetching UniProt sequences for 23 IDP genes")
    print("=" * 60)

    sequences = {}
    info_rows = []

    for gene, uniprot_id in sorted(GENE_UNIPROT.items()):
        print(f"  {gene} ({uniprot_id})...", end=" ", flush=True)
        try:
            header, seq = fetch_uniprot_fasta(uniprot_id)
            sequences[gene] = seq
            info_rows.append({
                "gene": gene,
                "uniprot_id": uniprot_id,
                "length": len(seq),
                "header": header[:80],
            })
            print(f"{len(seq)} aa")
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(0.3)

    # save combined FASTA
    fasta_path = SEQS_DIR / "all_proteins.fasta"
    with open(fasta_path, "w") as f:
        for gene in sorted(sequences):
            f.write(f">{gene}|{GENE_UNIPROT[gene]}\n")
            seq = sequences[gene]
            # wrap at 80 characters
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")
    print(f"\n  saved {len(sequences)} sequences to {fasta_path}")

    # save info CSV
    info_df = pd.DataFrame(info_rows)
    info_path = SEQS_DIR / "protein_info.csv"
    info_df.to_csv(info_path, index=False)
    print(f"  saved protein info to {info_path}")

    # cross-validate ClinVar positions
    print("\n  cross-validating ClinVar positions against UniProt...")
    variants_df = pd.read_csv(VARIANTS / "clinvar_idp_missense.csv")
    n_match, n_mismatch, n_oor = cross_validate_positions(variants_df, sequences)

    # flag valid variants (position matches UniProt)
    valid_flags = []
    for _, row in variants_df.iterrows():
        gene = row["gene"]
        pos = row["position"]
        ref_aa = row["ref_aa"]
        if gene in sequences:
            seq = sequences[gene]
            if 1 <= pos <= len(seq) and seq[pos - 1] == ref_aa:
                valid_flags.append(True)
            else:
                valid_flags.append(False)
        else:
            valid_flags.append(False)

    variants_df["position_valid"] = valid_flags
    n_valid = sum(valid_flags)
    print(f"\n  {n_valid} / {len(variants_df)} variants have valid positions ({n_valid / len(variants_df) * 100:.1f}%)")

    # save updated variants with validity flag
    variants_df.to_csv(VARIANTS / "clinvar_idp_missense.csv", index=False)
    print(f"  updated {VARIANTS / 'clinvar_idp_missense.csv'} with position_valid flag")

    # per-gene validity
    print(f"\n  per-gene position validity:")
    for gene in sorted(GENE_UNIPROT):
        gdf = variants_df[variants_df["gene"] == gene]
        n_total = len(gdf)
        n_val = gdf["position_valid"].sum()
        pct = n_val / n_total * 100 if n_total > 0 else 0
        print(f"    {gene:<12} {n_val:>4} / {n_total:>4} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
