#!/usr/bin/env python3
"""
axis 1: polymer physics features for all IDP proteins.
computes kappa, SCD, NCPR, FCR, GRAVY, Das-Pappu region, composition features.
uses localCIDER for proper kappa/SCD + custom composition analysis.
"""

import csv
import sys
from pathlib import Path

# try localcider first (proper implementation), fall back to manual
try:
    from localcider.sequenceParameters import SequenceParameters
    HAS_CIDER = True
except ImportError:
    HAS_CIDER = False
    print("warning: localcider not found, using manual kappa/SCD computation")

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
FASTA = PROJECT / "sequences" / "all_proteins.fasta"
OUT = PROJECT / "features" / "polymer_physics.csv"


# -- amino acid scales --

CHARGE_MAP = {"K": 1.0, "R": 1.0, "H": 0.5, "D": -1.0, "E": -1.0}

HYDROPATHY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "D": -3.5, "E": -3.5, "N": -3.5, "Q": -3.5, "K": -3.9, "R": -4.5,
}

# chou-fasman beta-sheet propensity (used later in amyloid script too)
BETA_PROPENSITY = {
    "V": 1.70, "I": 1.60, "Y": 1.47, "F": 1.38, "W": 1.37, "L": 1.30,
    "T": 1.19, "C": 1.19, "Q": 1.10, "M": 1.05, "R": 0.93, "N": 0.89,
    "H": 0.87, "A": 0.83, "S": 0.75, "G": 0.75, "K": 0.74, "P": 0.55,
    "D": 0.54, "E": 0.37,
}


def read_fasta(path):
    """read FASTA, return dict of name -> sequence."""
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


def compute_ncpr(seq):
    """net charge per residue."""
    return sum(CHARGE_MAP.get(aa, 0) for aa in seq) / len(seq)


def compute_fcr(seq):
    """fraction of charged residues."""
    return sum(1 for aa in seq if aa in ("K", "R", "H", "D", "E")) / len(seq)


def compute_fpos(seq):
    """fraction of positive residues."""
    return sum(1 for aa in seq if aa in ("K", "R")) / len(seq)


def compute_fneg(seq):
    """fraction of negative residues."""
    return sum(1 for aa in seq if aa in ("D", "E")) / len(seq)


def compute_gravy(seq):
    """grand average of hydropathy."""
    return sum(HYDROPATHY.get(aa, 0) for aa in seq) / len(seq)


def compute_kappa_manual(seq):
    """manual kappa: blob-based charge patterning (Das & Pappu 2013)."""
    n = len(seq)
    blob_size = 5
    n_blobs = n // blob_size
    if n_blobs < 2:
        return 0.0
    blob_sigmas = []
    for i in range(n_blobs):
        s = i * blob_size
        e = min(s + blob_size, n)
        blob = seq[s:e]
        n_pos = sum(1 for aa in blob if aa in ("K", "R"))
        n_neg = sum(1 for aa in blob if aa in ("D", "E"))
        # sigma = (f+ - f-) for blob
        blob_sigmas.append((n_pos - n_neg) / len(blob))

    # kappa = variance of blob sigmas / max variance
    var = np.var(blob_sigmas)
    # max variance happens when all positive in one blob, all negative in another
    f_pos = compute_fpos(seq)
    f_neg = compute_fneg(seq)
    sigma_global = f_pos - f_neg
    max_var = (f_pos + f_neg) ** 2 if (f_pos + f_neg) > 0 else 1.0
    return min(1.0, var / (max_var + 1e-10))


def compute_scd_manual(seq):
    """manual SCD: sequence charge decoration (Sawle & Ghosh 2015)."""
    n = len(seq)
    charges = [CHARGE_MAP.get(aa, 0) for aa in seq]
    scd = 0.0
    for i in range(n):
        if abs(charges[i]) < 0.01:
            continue
        for j in range(i + 1, n):
            if abs(charges[j]) < 0.01:
                continue
            scd += charges[i] * charges[j] * np.sqrt(abs(j - i))
    return scd / n


def classify_das_pappu(f_pos, f_neg):
    """classify into Das-Pappu diagram region (R1-R5)."""
    fcr = f_pos + f_neg
    ncpr = abs(f_pos - f_neg)

    if fcr < 0.25:
        return "R1"  # weak polyampholyte/polyelectrolyte: globules, tadpoles
    elif fcr >= 0.25 and ncpr < 0.25:
        return "R3"  # strong polyampholyte: coils, hairpins (kappa-dependent)
    elif fcr >= 0.25 and ncpr >= 0.25 and f_pos > f_neg:
        return "R4"  # positive polyelectrolyte: swollen coils
    elif fcr >= 0.25 and ncpr >= 0.25 and f_neg > f_pos:
        return "R5"  # negative polyelectrolyte: swollen coils
    else:
        return "R2"  # boundary


def compute_aromatic_clustering(seq):
    """measure aromatic clustering: std of inter-aromatic distances.
    high = clustered (bad for liquid LLPS); low = uniform (good for liquid LLPS).
    based on Martin et al. 2020 (Science): patterning determines liquid vs solid."""
    aro_positions = [i for i, aa in enumerate(seq) if aa in ("F", "Y", "W")]
    if len(aro_positions) < 3:
        return 0.0
    gaps = np.diff(aro_positions)
    # clustering = coefficient of variation of inter-aromatic gaps
    # high CV = irregular spacing (clustered); low CV = regular (uniform)
    return float(np.std(gaps) / (np.mean(gaps) + 1e-10))


def analyze_protein(name, seq):
    """compute all polymer physics features for one protein."""
    n = len(seq)

    # -- basic charge features --
    f_pos = compute_fpos(seq)
    f_neg = compute_fneg(seq)
    ncpr = compute_ncpr(seq)
    fcr = compute_fcr(seq)
    gravy = compute_gravy(seq)

    # -- kappa and SCD --
    if HAS_CIDER:
        try:
            sp = SequenceParameters(seq)
            kappa = sp.get_kappa()
            scd = compute_scd_manual(seq)  # localcider doesn't have SCD
        except Exception as e:
            print(f"  localcider failed for {name}: {e}, using manual")
            kappa = compute_kappa_manual(seq)
            scd = compute_scd_manual(seq)
    else:
        kappa = compute_kappa_manual(seq)
        scd = compute_scd_manual(seq)

    # -- Das-Pappu classification --
    dp_region = classify_das_pappu(f_pos, f_neg)

    # -- composition features --
    def frac(aas):
        return sum(1 for aa in seq if aa in aas) / n

    aromatic_frac = frac("FYW")
    glycine_frac = frac("G")
    proline_frac = frac("P")
    qn_frac = frac("QN")  # solidification promoters (Wang 2018)
    serine_frac = frac("S")  # solidification promoter
    tyr_frac = frac("Y")  # primary LLPS sticker (Martin 2020)
    arg_frac = frac("R")  # LLPS sticker (Wang 2018)

    # -- sticker features --
    aromatic_clustering = compute_aromatic_clustering(seq)

    # -- beta-sheet propensity (mean) --
    mean_beta = np.mean([BETA_PROPENSITY.get(aa, 1.0) for aa in seq])

    return {
        "name": name,
        "length": n,
        "ncpr": round(ncpr, 4),
        "fcr": round(fcr, 4),
        "f_pos": round(f_pos, 4),
        "f_neg": round(f_neg, 4),
        "gravy": round(gravy, 4),
        "kappa": round(kappa, 4),
        "scd": round(scd, 4),
        "das_pappu_region": dp_region,
        "aromatic_frac": round(aromatic_frac, 4),
        "glycine_frac": round(glycine_frac, 4),
        "proline_frac": round(proline_frac, 4),
        "qn_frac": round(qn_frac, 4),
        "serine_frac": round(serine_frac, 4),
        "tyr_frac": round(tyr_frac, 4),
        "arg_frac": round(arg_frac, 4),
        "aromatic_clustering": round(aromatic_clustering, 4),
        "mean_beta_propensity": round(mean_beta, 4),
    }


def main():
    seqs = read_fasta(FASTA)
    print(f"loaded {len(seqs)} sequences from {FASTA}")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for name, seq in seqs.items():
        print(f"  analyzing {name} ({len(seq)} aa)...")
        r = analyze_protein(name, seq.upper())
        results.append(r)

    # write CSV
    fields = list(results[0].keys())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(f"\nsaved {len(results)} proteins to {OUT}")

    # print summary table
    print(f"\n{'name':<30} {'len':>4} {'ncpr':>7} {'kappa':>6} {'scd':>8} {'DP':>3} {'aro%':>5} {'QN%':>5} {'G%':>5}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<30} {r['length']:>4} {r['ncpr']:>7.4f} {r['kappa']:>6.4f} "
              f"{r['scd']:>8.4f} {r['das_pappu_region']:>3} {r['aromatic_frac']:>5.3f} "
              f"{r['qn_frac']:>5.3f} {r['glycine_frac']:>5.3f}")


if __name__ == "__main__":
    main()
