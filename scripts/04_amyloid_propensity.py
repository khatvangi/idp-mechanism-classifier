#!/usr/bin/env python3
"""
axis 3: amyloid propensity scoring for all IDP proteins.
uses sequence-based heuristics since TANGO/WALTZ are web-only.
detects aggregation-prone regions (APRs) via sliding window with
hydrophobicity + beta-sheet propensity + low charge criteria.
"""

import csv
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
FASTA = PROJECT / "sequences" / "all_proteins.fasta"
OUT = PROJECT / "features" / "amyloid_propensity.csv"

# -- amino acid scales --

HYDROPATHY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "D": -3.5, "E": -3.5, "N": -3.5, "Q": -3.5, "K": -3.9, "R": -4.5,
}

# chou-fasman beta-sheet propensity
BETA_PROP = {
    "V": 1.70, "I": 1.60, "Y": 1.47, "F": 1.38, "W": 1.37, "L": 1.30,
    "T": 1.19, "C": 1.19, "Q": 1.10, "M": 1.05, "R": 0.93, "N": 0.89,
    "H": 0.87, "A": 0.83, "S": 0.75, "G": 0.75, "K": 0.74, "P": 0.55,
    "D": 0.54, "E": 0.37,
}

CHARGE_MAP = {"K": 1.0, "R": 1.0, "H": 0.5, "D": -1.0, "E": -1.0}


def read_fasta(path):
    seqs = {}
    name = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs


def detect_aprs(seq, window=7, hydro_thresh=0.8, beta_thresh=1.0, charge_thresh=0.3):
    """
    detect aggregation-prone regions using sliding window.
    criteria (all must be met):
      - mean hydrophobicity > hydro_thresh
      - mean beta-sheet propensity > beta_thresh
      - |mean charge| < charge_thresh (low charge = no electrostatic repulsion)
    returns: list of (start, end, score) tuples.
    """
    n = len(seq)
    aprs = []
    apr_profile = np.zeros(n)

    for i in range(n - window + 1):
        w = seq[i:i + window]
        h = np.mean([HYDROPATHY.get(aa, 0) for aa in w])
        b = np.mean([BETA_PROP.get(aa, 1.0) for aa in w])
        c = abs(np.mean([CHARGE_MAP.get(aa, 0) for aa in w]))

        # aggregation propensity score for this window
        score = max(0, h) * b  # hydrophobicity × beta propensity

        if h > hydro_thresh and b > beta_thresh and c < charge_thresh:
            aprs.append((i, i + window, score))
            apr_profile[i:i + window] = np.maximum(apr_profile[i:i + window], score)

    # merge overlapping APRs
    merged = []
    if aprs:
        current_start, current_end, current_score = aprs[0]
        for s, e, sc in aprs[1:]:
            if s <= current_end:
                current_end = max(current_end, e)
                current_score = max(current_score, sc)
            else:
                merged.append((current_start, current_end, current_score))
                current_start, current_end, current_score = s, e, sc
        merged.append((current_start, current_end, current_score))

    return merged, apr_profile


def compute_amyloid_features(name, seq):
    """compute all amyloid propensity features for one protein."""
    n = len(seq)

    # per-residue beta propensity
    beta_scores = np.array([BETA_PROP.get(aa, 1.0) for aa in seq])
    hydro_scores = np.array([HYDROPATHY.get(aa, 0) for aa in seq])

    # detect APRs
    aprs, apr_profile = detect_aprs(seq)

    # count APR residues (merged)
    apr_residues = set()
    for s, e, _ in aprs:
        apr_residues.update(range(s, e))
    apr_frac = len(apr_residues) / n if n > 0 else 0.0

    # max APR score
    max_apr_score = max((sc for _, _, sc in aprs), default=0.0)

    # mean amyloid propensity over whole sequence
    # = mean(max(0, hydro) * beta) — captures hydrophobic + beta-prone stretches
    amyloid_propensity = np.mean(np.maximum(0, hydro_scores) * beta_scores)

    # LARKS-like detection: short stretches (4-7 aa) with alternating hydrophobic residues
    # and low complexity (proxy for amyloid-like reversible kinked segments)
    larks_count = 0
    for i in range(n - 5):
        w = seq[i:i + 6]
        n_hydro = sum(1 for aa in w if HYDROPATHY.get(aa, 0) > 1.0)
        n_gly_ser = sum(1 for aa in w if aa in ("G", "S"))
        # LARKS-like: mix of hydrophobic and small residues, low charge
        if n_hydro >= 2 and n_gly_ser >= 2:
            c = abs(sum(CHARGE_MAP.get(aa, 0) for aa in w))
            if c < 0.5:
                larks_count += 1

    # proline breaker score: prolines disrupt beta-sheets
    # more prolines = less amyloid prone
    proline_frac = sum(1 for aa in seq if aa == "P") / n

    return {
        "name": name,
        "length": n,
        "n_aprs": len(aprs),
        "apr_fraction": round(apr_frac, 4),
        "max_apr_score": round(max_apr_score, 4),
        "mean_amyloid_propensity": round(amyloid_propensity, 4),
        "mean_beta_propensity": round(float(beta_scores.mean()), 4),
        "mean_hydrophobicity": round(float(hydro_scores.mean()), 4),
        "larks_count": larks_count,
        "proline_breaker_frac": round(proline_frac, 4),
        # top APR regions (for interpretability)
        "top_apr_regions": str([(s+1, e, round(sc, 2)) for s, e, sc in sorted(aprs, key=lambda x: -x[2])[:3]]),
    }


def main():
    seqs = read_fasta(FASTA)
    print(f"loaded {len(seqs)} sequences")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for name, seq in seqs.items():
        seq = seq.upper()
        print(f"  {name} ({len(seq)} aa)...")
        r = compute_amyloid_features(name, seq)
        results.append(r)

    # write CSV
    fields = list(results[0].keys())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(f"\nsaved to {OUT}")

    # print summary
    print(f"\n{'name':<30} {'APRs':>5} {'APR%':>6} {'amyl_prop':>10} {'beta':>6} {'LARKS':>6}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<30} {r['n_aprs']:>5} {r['apr_fraction']:>6.3f} "
              f"{r['mean_amyloid_propensity']:>10.4f} {r['mean_beta_propensity']:>6.3f} "
              f"{r['larks_count']:>6}")


if __name__ == "__main__":
    main()
