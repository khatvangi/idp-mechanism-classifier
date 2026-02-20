#!/usr/bin/env python3
"""
axis 2: maturation grammar scores for all IDP proteins.
reuses the SequenceGrammar from condensate-maturation-theory project.
computes p_lock, p_pi, p_polar scalar summaries per protein.
"""

import csv
import sys
from pathlib import Path

import numpy as np

# add condensate-maturation-theory to path
GRAMMAR_ROOT = Path("/storage/kiran-stuff/condensate-maturation-theory/condensate_maturation")
sys.path.insert(0, str(GRAMMAR_ROOT))

from src.sequence import ProteinSequence
from src.grammar import SequenceGrammar

PROJECT = Path(__file__).resolve().parent.parent
FASTA = PROJECT / "sequences" / "all_proteins.fasta"
OUT = PROJECT / "features" / "maturation_grammar.csv"

# grammar config (from grammar_ranking.py, validated at tau=0.929)
GRAMMAR_CONFIG = {
    "window_length": 12,
    "alpha_pi": 1.0,
    "alpha_pol": 1.0,
    "pi_activation_z": 1.0,
    "pol_activation_z": 1.0,
    "require_channel_presence": True,
    "grammar_z_aggregation": "mean",
    "grammar_zscore_mode": "protein_local",
    "lambda_charge": 1.0,
    "ionic_strength": 0.15,
    "pH": 7.0,
    "temperature": 300.0,
    "r_c": 0.8,
    "charge_model": "abs_suppress",
}

# lcd regions for each protein (from literature)
# format: (start_0indexed, end_0indexed) or None for full-length
LCD_REGIONS = {
    "fus_full_wt": (0, 163),         # QGSY-rich LCD
    "fus_P525L": (0, 163),
    "tdp43_full_wt": (273, 414),     # glycine-rich LCD
    "tdp43_M337V": (273, 414),
    "tdp43_Q331K": (273, 414),
    "hnrnpa1_full_wt": (185, 372),   # PrLD
    "hnrnpa1_D262V": (185, 372),
    "tau_2n4r_wt": (243, 368),       # MTBR (aggregation-prone)
    "tau_P301L": (243, 368),
    "alpha_synuclein_ensemble": (60, 95),  # NAC region
    "asyn_A53T": (60, 95),
    "asyn_E46K": (60, 95),
    # ab42, prp, httex1: score full sequence (short enough)
    # -- expanded panel --
    # iapp: 37 aa, too short for lcd subregion
    "tia1_full_wt": (291, 386),          # Q-rich PrLD (C-terminal)
    "tia1_P362L": (291, 386),
    "ewsr1_full_wt": (0, 264),           # QGSY-rich LCD (N-terminal, similar to FUS)
    "taf15_full_wt": (0, 204),           # QGSY-rich LCD (N-terminal)
    "hnrnpa2b1_full_wt": (189, 353),     # Gly-rich PrLD
    "ddx4_full_wt": (0, 236),            # N-terminal RGG-rich IDR
    "npm1_full_wt": (117, 294),          # C-terminal acidic/disordered region
    # ataxin3: polyQ region is short, score full sequence
}


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


def compute_grammar_scores(name, seq):
    """compute grammar scores for a sequence, return scalar summaries."""
    protein = ProteinSequence(name=name, sequence=seq)

    # check if sequence has stickers
    n_stickers = len(protein.sticker_positions())
    if n_stickers < 2:
        return {
            "n_stickers": n_stickers,
            "n_windows": 0,
            "lockability": 0.0,
            "lockability_per_pair": 0.0,
            "mean_p_lock": 0.0,
            "max_p_lock": 0.0,
            "mean_p_pi": 0.0,
            "mean_p_polar": 0.0,
            "mean_g_charge": 1.0,
            "mean_p_lock_eff": 0.0,
        }

    grammar = SequenceGrammar(protein, GRAMMAR_CONFIG)
    scores = grammar.compute_all()

    p_lock = scores["p_lock_matrix"]
    p_pi = scores["p_pi_matrix"]
    p_pol = scores["p_polar_matrix"]
    g_ch = scores["g_charge_matrix"]
    p_eff = scores["p_lock_eff_matrix"]

    n_w = p_lock.shape[0]
    n_pairs = n_w * (n_w - 1) // 2

    # upper triangle values (no diagonal)
    upper_lock = []
    upper_pi = []
    upper_pol = []
    upper_gch = []
    upper_eff = []
    lockability = 0.0
    for i in range(n_w):
        for j in range(i + 1, n_w):
            upper_lock.append(p_lock[i, j])
            upper_pi.append(p_pi[i, j])
            upper_pol.append(p_pol[i, j])
            upper_gch.append(g_ch[i, j])
            upper_eff.append(p_eff[i, j])
            lockability += p_lock[i, j] * g_ch[i, j]

    upper_lock = np.array(upper_lock) if upper_lock else np.array([0.0])
    upper_pi = np.array(upper_pi) if upper_pi else np.array([0.0])
    upper_pol = np.array(upper_pol) if upper_pol else np.array([0.0])
    upper_gch = np.array(upper_gch) if upper_gch else np.array([1.0])
    upper_eff = np.array(upper_eff) if upper_eff else np.array([0.0])

    return {
        "n_stickers": n_stickers,
        "n_windows": n_w,
        "lockability": float(lockability),
        "lockability_per_pair": float(lockability / n_pairs) if n_pairs > 0 else 0.0,
        "mean_p_lock": float(upper_lock.mean()),
        "max_p_lock": float(upper_lock.max()),
        "mean_p_pi": float(upper_pi.mean()),
        "mean_p_polar": float(upper_pol.mean()),
        "mean_g_charge": float(upper_gch.mean()),
        "mean_p_lock_eff": float(upper_eff.mean()),
    }


def main():
    seqs = read_fasta(FASTA)
    print(f"loaded {len(seqs)} sequences")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for name, seq in seqs.items():
        seq = seq.upper()
        print(f"\n  {name} ({len(seq)} aa)...")

        # full-length grammar
        full = compute_grammar_scores(name, seq)
        full = {f"full_{k}": v for k, v in full.items()}

        # lcd-region grammar (if defined)
        lcd_range = LCD_REGIONS.get(name)
        if lcd_range:
            lcd_seq = seq[lcd_range[0]:lcd_range[1]]
            lcd_name = f"{name}_lcd"
            print(f"    lcd region: {lcd_range[0]+1}-{lcd_range[1]} ({len(lcd_seq)} aa)")
            lcd = compute_grammar_scores(lcd_name, lcd_seq)
            lcd = {f"lcd_{k}": v for k, v in lcd.items()}
            has_lcd = True
        else:
            # no lcd region: set lcd columns to zero using the original (unprefixed) keys
            lcd = {
                "lcd_n_stickers": 0, "lcd_n_windows": 0,
                "lcd_lockability": 0.0, "lcd_lockability_per_pair": 0.0,
                "lcd_mean_p_lock": 0.0, "lcd_max_p_lock": 0.0,
                "lcd_mean_p_pi": 0.0, "lcd_mean_p_polar": 0.0,
                "lcd_mean_g_charge": 1.0, "lcd_mean_p_lock_eff": 0.0,
            }
            has_lcd = False

        row = {"name": name, "has_lcd_region": has_lcd}
        row.update(full)
        row.update(lcd)
        results.append(row)

        # print summary
        print(f"    full: lockability={full['full_lockability']:.2f}, "
              f"mean_p_lock={full['full_mean_p_lock']:.4f}, "
              f"p_pi={full['full_mean_p_pi']:.4f}, p_polar={full['full_mean_p_polar']:.4f}")
        if has_lcd:
            print(f"    lcd:  lockability={lcd['lcd_lockability']:.2f}, "
                  f"mean_p_lock={lcd['lcd_mean_p_lock']:.4f}")

    # write CSV
    fields = list(results[0].keys())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
