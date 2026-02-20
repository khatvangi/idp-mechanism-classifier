#!/usr/bin/env python3
"""
tasks 4+5+6 combined: compute disorder predictions, local window features,
and mutation-specific features for all validated variant positions.
outputs a single feature matrix ready for modeling.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import metapredict as meta

# add condensate-maturation-theory to path for grammar
GRAMMAR_ROOT = Path("/storage/kiran-stuff/condensate-maturation-theory/condensate_maturation")
sys.path.insert(0, str(GRAMMAR_ROOT))
from src.sequence import ProteinSequence
from src.grammar import SequenceGrammar

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
VARIANTS = DATA / "variants"
SEQS_DIR = DATA / "sequences"
DISORDER_DIR = DATA / "disorder"
DISORDER_DIR.mkdir(parents=True, exist_ok=True)

# kyte-doolittle hydrophobicity scale
KD_HYDRO = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "D": -3.5, "E": -3.5, "N": -3.5, "Q": -3.5, "K": -3.9, "R": -4.5,
}

# chou-fasman beta-sheet propensity
CF_BETA = {
    "V": 1.70, "I": 1.60, "Y": 1.47, "F": 1.38, "W": 1.37, "L": 1.30,
    "T": 1.19, "C": 1.19, "Q": 1.10, "M": 1.05, "R": 0.93, "N": 0.89,
    "H": 0.87, "A": 0.83, "S": 0.75, "G": 0.75, "K": 0.74, "D": 0.54,
    "P": 0.55, "E": 0.37,
}

# BLOSUM62 matrix (upper triangle, standard)
_BLOSUM62_STR = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def parse_blosum62():
    """parse BLOSUM62 into dict of (AA1, AA2) → score."""
    lines = _BLOSUM62_STR.strip().split("\n")
    aas = lines[0].split()
    matrix = {}
    for line in lines[1:]:
        parts = line.split()
        aa1 = parts[0]
        scores = [int(x) for x in parts[1:]]
        for j, aa2 in enumerate(aas):
            matrix[(aa1, aa2)] = scores[j]
    return matrix


BLOSUM62 = parse_blosum62()

# grantham distance table (from Grantham 1974)
GRANTHAM = {
    ("S", "R"): 110, ("S", "L"): 145, ("S", "P"): 74, ("S", "T"): 58,
    ("S", "A"): 99, ("S", "V"): 124, ("S", "G"): 56, ("S", "I"): 142,
    ("S", "F"): 155, ("S", "Y"): 144, ("S", "C"): 112, ("S", "H"): 89,
    ("S", "Q"): 68, ("S", "N"): 46, ("S", "K"): 121, ("S", "D"): 65,
    ("S", "E"): 80, ("S", "M"): 135, ("S", "W"): 177,
    ("R", "L"): 102, ("R", "P"): 103, ("R", "T"): 71, ("R", "A"): 112,
    ("R", "V"): 96, ("R", "G"): 125, ("R", "I"): 97, ("R", "F"): 97,
    ("R", "Y"): 77, ("R", "C"): 180, ("R", "H"): 29, ("R", "Q"): 43,
    ("R", "N"): 86, ("R", "K"): 26, ("R", "D"): 96, ("R", "E"): 54,
    ("R", "M"): 91, ("R", "W"): 101,
    ("L", "P"): 98, ("L", "T"): 92, ("L", "A"): 96, ("L", "V"): 32,
    ("L", "G"): 138, ("L", "I"): 5, ("L", "F"): 22, ("L", "Y"): 36,
    ("L", "C"): 198, ("L", "H"): 99, ("L", "Q"): 113, ("L", "N"): 153,
    ("L", "K"): 107, ("L", "D"): 172, ("L", "E"): 138, ("L", "M"): 15,
    ("L", "W"): 61,
    ("P", "T"): 38, ("P", "A"): 27, ("P", "V"): 68, ("P", "G"): 42,
    ("P", "I"): 95, ("P", "F"): 114, ("P", "Y"): 110, ("P", "C"): 169,
    ("P", "H"): 77, ("P", "Q"): 76, ("P", "N"): 91, ("P", "K"): 103,
    ("P", "D"): 108, ("P", "E"): 93, ("P", "M"): 87, ("P", "W"): 147,
    ("T", "A"): 58, ("T", "V"): 69, ("T", "G"): 59, ("T", "I"): 89,
    ("T", "F"): 103, ("T", "Y"): 92, ("T", "C"): 149, ("T", "H"): 47,
    ("T", "Q"): 42, ("T", "N"): 65, ("T", "K"): 78, ("T", "D"): 85,
    ("T", "E"): 65, ("T", "M"): 81, ("T", "W"): 128,
    ("A", "V"): 64, ("A", "G"): 60, ("A", "I"): 94, ("A", "F"): 113,
    ("A", "Y"): 112, ("A", "C"): 195, ("A", "H"): 86, ("A", "Q"): 91,
    ("A", "N"): 111, ("A", "K"): 106, ("A", "D"): 126, ("A", "E"): 107,
    ("A", "M"): 84, ("A", "W"): 148,
    ("V", "G"): 109, ("V", "I"): 29, ("V", "F"): 50, ("V", "Y"): 55,
    ("V", "C"): 192, ("V", "H"): 84, ("V", "Q"): 96, ("V", "N"): 133,
    ("V", "K"): 97, ("V", "D"): 152, ("V", "E"): 121, ("V", "M"): 21,
    ("V", "W"): 88,
    ("G", "I"): 135, ("G", "F"): 153, ("G", "Y"): 147, ("G", "C"): 159,
    ("G", "H"): 98, ("G", "Q"): 87, ("G", "N"): 80, ("G", "K"): 127,
    ("G", "D"): 94, ("G", "E"): 98, ("G", "M"): 127, ("G", "W"): 184,
    ("I", "F"): 21, ("I", "Y"): 33, ("I", "C"): 198, ("I", "H"): 94,
    ("I", "Q"): 109, ("I", "N"): 149, ("I", "K"): 102, ("I", "D"): 168,
    ("I", "E"): 134, ("I", "M"): 10, ("I", "W"): 61,
    ("F", "Y"): 22, ("F", "C"): 205, ("F", "H"): 100, ("F", "Q"): 116,
    ("F", "N"): 158, ("F", "K"): 102, ("F", "D"): 177, ("F", "E"): 140,
    ("F", "M"): 28, ("F", "W"): 40,
    ("Y", "C"): 194, ("Y", "H"): 83, ("Y", "Q"): 99, ("Y", "N"): 143,
    ("Y", "K"): 85, ("Y", "D"): 160, ("Y", "E"): 122, ("Y", "M"): 36,
    ("Y", "W"): 37,
    ("C", "H"): 174, ("C", "Q"): 154, ("C", "N"): 139, ("C", "K"): 202,
    ("C", "D"): 154, ("C", "E"): 170, ("C", "M"): 196, ("C", "W"): 215,
    ("H", "Q"): 24, ("H", "N"): 68, ("H", "K"): 32, ("H", "D"): 81,
    ("H", "E"): 40, ("H", "M"): 87, ("H", "W"): 115,
    ("Q", "N"): 46, ("Q", "K"): 53, ("Q", "D"): 61, ("Q", "E"): 29,
    ("Q", "M"): 101, ("Q", "W"): 130,
    ("N", "K"): 94, ("N", "D"): 23, ("N", "E"): 42, ("N", "M"): 142,
    ("N", "W"): 174,
    ("K", "D"): 101, ("K", "E"): 56, ("K", "M"): 95, ("K", "W"): 110,
    ("D", "E"): 45, ("D", "M"): 160, ("D", "W"): 181,
    ("E", "M"): 126, ("E", "W"): 152,
    ("M", "W"): 67,
}


def get_grantham(aa1, aa2):
    """get Grantham distance (symmetric)."""
    if aa1 == aa2:
        return 0
    return GRANTHAM.get((aa1, aa2), GRANTHAM.get((aa2, aa1), 100))  # default 100


# amino acid properties
CHARGED_POS = set("KRH")
CHARGED_NEG = set("DE")
CHARGED = CHARGED_POS | CHARGED_NEG
AROMATIC = set("FWY")
POLAR = set("STNQHKRDE")
NONPOLAR = set("AVILMFWPG")

# grammar config (from maturation grammar script)
GRAMMAR_CONFIG = {
    "window_length": 12,
    "alpha_pi": 1.0, "alpha_pol": 1.0,
    "pi_activation_z": 1.0, "pol_activation_z": 1.0,
    "require_channel_presence": True,
    "grammar_z_aggregation": "mean",
    "grammar_zscore_mode": "protein_local",
    "lambda_charge": 1.0, "ionic_strength": 0.15,
    "pH": 7.0, "temperature": 300.0,
    "r_c": 0.8, "charge_model": "abs_suppress",
}


def read_fasta(path):
    """read multi-FASTA file into dict."""
    seqs = {}
    current_name = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    seqs[current_name] = "".join(current_seq)
                # extract gene name (before |)
                current_name = line[1:].split("|")[0].strip()
                current_seq = []
            else:
                current_seq.append(line)
    if current_name:
        seqs[current_name] = "".join(current_seq)
    return seqs


def compute_disorder(sequences):
    """compute per-residue disorder predictions for all proteins."""
    print("\n  computing disorder predictions...")
    disorder_dict = {}
    for gene, seq in sorted(sequences.items()):
        scores = meta.predict_disorder(seq)
        disorder_dict[gene] = np.array(scores)
        n_disordered = (np.array(scores) > 0.5).sum()
        print(f"    {gene:<12} {len(seq):>4} aa, {n_disordered:>4} disordered ({n_disordered/len(seq)*100:.0f}%)")

    # save to CSV
    rows = []
    for gene, scores in disorder_dict.items():
        for i, s in enumerate(scores):
            rows.append({"gene": gene, "position": i + 1, "disorder_score": round(s, 4)})
    pd.DataFrame(rows).to_csv(DATA / "disorder" / "per_residue_disorder.csv", index=False)
    print(f"    saved to {DATA / 'disorder' / 'per_residue_disorder.csv'}")

    return disorder_dict


def compute_grammar_scores(sequences):
    """compute per-position grammar scores by aggregating pairwise window interactions.

    for each residue, finds which grammar windows contain it,
    then averages that window's interaction scores across all partner windows.
    positions not covered by any sticker window get the protein-level mean.
    """
    print("\n  computing grammar scores...")
    grammar_dict = {}  # gene → {position → (p_lock, p_pi, p_polar)}

    for gene, seq in sorted(sequences.items()):
        try:
            protein = ProteinSequence(name=gene, sequence=seq)
            n_stickers = len(protein.sticker_positions())

            if n_stickers < 2:
                # not enough stickers for pairwise grammar — fill with zeros
                grammar_dict[gene] = {pos: (0.0, 0.0, 0.0) for pos in range(len(seq))}
                print(f"    {gene:<12} {n_stickers} stickers (too few, zeros)")
                continue

            grammar = SequenceGrammar(protein, GRAMMAR_CONFIG)
            scores = grammar.compute_all()

            p_lock_mat = scores["p_lock_matrix"]
            p_pi_mat = scores["p_pi_matrix"]
            p_pol_mat = scores["p_polar_matrix"]
            n_w = p_lock_mat.shape[0]

            # per-window mean interaction scores (mean across all partners)
            win_mean_lock = np.zeros(n_w)
            win_mean_pi = np.zeros(n_w)
            win_mean_pol = np.zeros(n_w)
            for i in range(n_w):
                partners = [j for j in range(n_w) if j != i]
                if partners:
                    win_mean_lock[i] = np.mean([p_lock_mat[i, j] for j in partners])
                    win_mean_pi[i] = np.mean([p_pi_mat[i, j] for j in partners])
                    win_mean_pol[i] = np.mean([p_pol_mat[i, j] for j in partners])

            # map windows to residue positions
            # each window covers [start, end) around a sticker residue
            windows = protein.all_windows(L=GRAMMAR_CONFIG["window_length"])
            # windows: list of (window_id, start, end, subsequence)

            # build position → list of window indices
            pos_to_wins = {pos: [] for pos in range(len(seq))}
            for wid, start, end, _ in windows:
                for pos in range(start, end):
                    if 0 <= pos < len(seq):
                        pos_to_wins[pos].append(wid)

            # protein-level mean (fallback for uncovered positions)
            global_lock = float(win_mean_lock.mean()) if n_w > 0 else 0.0
            global_pi = float(win_mean_pi.mean()) if n_w > 0 else 0.0
            global_pol = float(win_mean_pol.mean()) if n_w > 0 else 0.0

            # aggregate per position
            pos_scores = {}
            for pos in range(len(seq)):
                wids = pos_to_wins[pos]
                if wids:
                    pos_scores[pos] = (
                        float(np.mean([win_mean_lock[w] for w in wids])),
                        float(np.mean([win_mean_pi[w] for w in wids])),
                        float(np.mean([win_mean_pol[w] for w in wids])),
                    )
                else:
                    pos_scores[pos] = (global_lock, global_pi, global_pol)

            grammar_dict[gene] = pos_scores
            n_covered = sum(1 for wids in pos_to_wins.values() if wids)
            print(f"    {gene:<12} {n_w} windows, {n_covered}/{len(seq)} positions covered")
        except Exception as e:
            print(f"    {gene:<12} FAILED: {e}")
            grammar_dict[gene] = {pos: (0.0, 0.0, 0.0) for pos in range(len(seq))}

    return grammar_dict


def compute_features(variants_df, sequences, disorder_dict, grammar_dict):
    """compute all features for each variant position."""
    print("\n  computing features for each variant...")
    window = 15  # ±15 residues

    feature_rows = []
    for idx, row in variants_df.iterrows():
        gene = row["gene"]
        pos = row["position"]  # 1-based
        ref_aa = row["ref_aa"]
        alt_aa = row["alt_aa"]
        seq = sequences[gene]
        pos_0 = pos - 1  # 0-based

        # --- local window features ---
        start = max(0, pos_0 - window)
        end = min(len(seq), pos_0 + window + 1)
        win_seq = seq[start:end]
        win_len = len(win_seq)

        # local hydrophobicity
        local_hydro = np.mean([KD_HYDRO.get(aa, 0) for aa in win_seq])

        # local charge density
        local_charge = sum(1 for aa in win_seq if aa in CHARGED) / win_len

        # local aromatic density
        local_aromatic = sum(1 for aa in win_seq if aa in AROMATIC) / win_len

        # local glycine fraction
        local_gly = sum(1 for aa in win_seq if aa == "G") / win_len

        # local proline fraction
        local_pro = sum(1 for aa in win_seq if aa == "P") / win_len

        # local beta propensity
        local_beta = np.mean([CF_BETA.get(aa, 1.0) for aa in win_seq])

        # local Q/N fraction
        local_qn = sum(1 for aa in win_seq if aa in "QN") / win_len

        # disorder features
        disorder_scores = disorder_dict.get(gene, np.zeros(len(seq)))
        position_disorder = disorder_scores[pos_0] if pos_0 < len(disorder_scores) else 0
        local_disorder = np.mean(disorder_scores[start:end])
        in_idr = int(position_disorder > 0.5)

        # relative position
        relative_pos = pos / len(seq)

        # grammar features at position
        gram = grammar_dict.get(gene, {})
        gram_scores = gram.get(pos_0, (0.0, 0.0, 0.0))
        local_p_lock, local_p_pi, local_p_polar = gram_scores

        # is sticker (aromatic in potential LCD context)
        is_sticker = int(ref_aa in AROMATIC)

        # --- mutation-specific features ---
        blosum = BLOSUM62.get((ref_aa, alt_aa), 0)
        grantham = get_grantham(ref_aa, alt_aa)

        # charge change categories
        ref_charged = ref_aa in CHARGED
        alt_charged = alt_aa in CHARGED
        ref_pos = ref_aa in CHARGED_POS
        alt_pos = alt_aa in CHARGED_POS
        ref_neg = ref_aa in CHARGED_NEG
        alt_neg = alt_aa in CHARGED_NEG

        if ref_pos and alt_neg or ref_neg and alt_pos:
            charge_change = "charge_flip"
        elif ref_charged and not alt_charged:
            charge_change = "charge_loss"
        elif not ref_charged and alt_charged:
            charge_change = "charge_gain"
        else:
            charge_change = "neutral"

        # hydrophobicity change
        hydro_change = KD_HYDRO.get(alt_aa, 0) - KD_HYDRO.get(ref_aa, 0)

        # size change (approximate molecular weight)
        MW = {
            "G": 57, "A": 71, "V": 99, "L": 113, "I": 113, "P": 97, "F": 147,
            "W": 186, "M": 131, "S": 87, "T": 101, "C": 103, "Y": 163, "H": 137,
            "D": 115, "N": 114, "E": 129, "Q": 128, "K": 128, "R": 156,
        }
        size_change = MW.get(alt_aa, 110) - MW.get(ref_aa, 110)

        # proline changes
        creates_proline = int(alt_aa == "P" and ref_aa != "P")
        destroys_proline = int(ref_aa == "P" and alt_aa != "P")

        # glycine changes
        creates_glycine = int(alt_aa == "G" and ref_aa != "G")

        # aromatic change
        ref_arom = ref_aa in AROMATIC
        alt_arom = alt_aa in AROMATIC
        if not ref_arom and alt_arom:
            aromatic_change = "gain"
        elif ref_arom and not alt_arom:
            aromatic_change = "loss"
        else:
            aromatic_change = "none"

        # polarity change
        ref_polar = ref_aa in POLAR
        alt_polar = alt_aa in POLAR
        if ref_polar and not alt_polar:
            polarity_change = "polar_to_nonpolar"
        elif not ref_polar and alt_polar:
            polarity_change = "nonpolar_to_polar"
        else:
            polarity_change = "same"

        # --- protein-level features ---
        protein_length = len(seq)
        total_disorder_frac = (disorder_scores > 0.5).sum() / len(disorder_scores)

        feature_rows.append({
            # identifiers
            "gene": gene,
            "position": pos,
            "ref_aa": ref_aa,
            "alt_aa": alt_aa,
            "label": row["train_label"],
            # local window (16 features)
            "local_hydrophobicity": round(local_hydro, 4),
            "local_charge_density": round(local_charge, 4),
            "local_aromatic_density": round(local_aromatic, 4),
            "local_glycine_frac": round(local_gly, 4),
            "local_proline_frac": round(local_pro, 4),
            "local_beta_propensity": round(local_beta, 4),
            "local_qn_frac": round(local_qn, 4),
            "local_disorder": round(local_disorder, 4),
            "position_disorder": round(position_disorder, 4),
            "in_idr": in_idr,
            "relative_position": round(relative_pos, 4),
            "local_p_lock": round(local_p_lock, 4),
            "local_p_pi": round(local_p_pi, 4),
            "local_p_polar": round(local_p_polar, 4),
            "is_sticker": is_sticker,
            # mutation-specific (10 features)
            "blosum62": blosum,
            "grantham_distance": grantham,
            "charge_change": charge_change,
            "hydrophobicity_change": round(hydro_change, 2),
            "size_change": size_change,
            "creates_proline": creates_proline,
            "destroys_proline": destroys_proline,
            "creates_glycine": creates_glycine,
            "aromatic_change": aromatic_change,
            "polarity_change": polarity_change,
            # protein-level (2 features)
            "protein_length": protein_length,
            "total_disorder_frac": round(total_disorder_frac, 4),
        })

    return pd.DataFrame(feature_rows)


def main():
    print("=" * 60)
    print("computing features for all validated variant positions")
    print("=" * 60)

    # load sequences
    sequences = read_fasta(SEQS_DIR / "all_proteins.fasta")
    print(f"  loaded {len(sequences)} protein sequences")

    # load variants (valid positions only)
    all_variants = pd.read_csv(VARIANTS / "clinvar_idp_missense.csv")
    variants = all_variants[all_variants["position_valid"] == True].copy()
    print(f"  loaded {len(variants)} valid variants")

    # step 1: disorder prediction
    disorder_dict = compute_disorder(sequences)

    # step 2: grammar scores
    grammar_dict = compute_grammar_scores(sequences)

    # step 3: compute all features
    features_df = compute_features(variants, sequences, disorder_dict, grammar_dict)

    # save
    out_path = VARIANTS / "feature_matrix.csv"
    features_df.to_csv(out_path, index=False)
    print(f"\n  saved {len(features_df)} variants × {len(features_df.columns)} columns to {out_path}")

    # summary
    print(f"\n  label distribution in feature matrix:")
    for label, count in features_df["label"].value_counts().items():
        print(f"    {label:<15} {count:>6}")

    # spot checks
    print(f"\n  spot checks:")
    snca_a53t = features_df[(features_df["gene"] == "SNCA") & (features_df["position"] == 53)]
    if len(snca_a53t) > 0:
        r = snca_a53t.iloc[0]
        print(f"    SNCA A53T: in_idr={r['in_idr']}, disorder={r['position_disorder']:.2f}, "
              f"blosum={r['blosum62']}, grantham={r['grantham_distance']}")

    fus_p525l = features_df[(features_df["gene"] == "FUS") & (features_df["position"] == 525)]
    if len(fus_p525l) > 0:
        r = fus_p525l.iloc[0]
        print(f"    FUS P525L: in_idr={r['in_idr']}, disorder={r['position_disorder']:.2f}, "
              f"p_lock={r['local_p_lock']:.3f}, destroys_proline={r['destroys_proline']}")

    sod1_example = features_df[(features_df["gene"] == "SOD1")].head(1)
    if len(sod1_example) > 0:
        r = sod1_example.iloc[0]
        print(f"    SOD1 pos {r['position']}: in_idr={r['in_idr']}, disorder={r['position_disorder']:.2f}")


if __name__ == "__main__":
    main()
