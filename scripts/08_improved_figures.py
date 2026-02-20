#!/usr/bin/env python3
"""
improved publication-quality figures.
1. main landscape: no-ESM2 version with biplot arrows (most interpretable)
2. improved radar: separate subplots per mechanism class for readability
3. maturation grammar comparison: the strongest result, deserves its own figure
4. feature importance: which features drive separation
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT = Path(__file__).resolve().parent.parent
FEATURES = PROJECT / "features"
FIGURES = PROJECT / "figures"

MECH_COLORS = {
    "amyloid": "#d62728",
    "condensate_maturation": "#1f77b4",
    "dual": "#9467bd",
    "template_misfolding": "#ff7f0e",
    "polyQ_collapse": "#2ca02c",
    "functional_condensate": "#17becf",
}
MECH_LABELS = {
    "amyloid": "amyloid",
    "condensate_maturation": "condensate maturation",
    "dual": "dual pathway",
    "template_misfolding": "template misfolding",
    "polyQ_collapse": "polyQ collapse",
    "functional_condensate": "functional condensate",
}
DISPLAY = {
    "alpha_synuclein_ensemble": "α-syn", "asyn_A53T": "α-syn A53T",
    "asyn_E46K": "α-syn E46K", "ab42_wt": "Aβ42",
    "iapp_mature_wt": "IAPP",
    "tau_2n4r_wt": "Tau", "tau_P301L": "Tau P301L",
    "tdp43_full_wt": "TDP-43", "tdp43_M337V": "TDP-43 M337V",
    "tdp43_Q331K": "TDP-43 Q331K", "fus_full_wt": "FUS",
    "fus_P525L": "FUS P525L", "hnrnpa1_full_wt": "hnRNPA1",
    "hnrnpa1_D262V": "hnRNPA1 D262V",
    "tia1_full_wt": "TIA1", "tia1_P362L": "TIA1 P362L",
    "ewsr1_full_wt": "EWSR1", "taf15_full_wt": "TAF15",
    "hnrnpa2b1_full_wt": "hnRNPA2B1",
    "prp_mature_wt": "PrP",
    "httex1_q21_wt": "Htt Q21", "httex1_q46": "Htt Q46",
    "ataxin3_q14_wt": "Ataxin-3", "ataxin3_q72": "Ataxin-3 Q72",
    "ddx4_full_wt": "DDX4", "npm1_full_wt": "NPM1",
}
WT_NAMES = {"alpha_synuclein_ensemble", "ab42_wt", "iapp_mature_wt",
            "tau_2n4r_wt",
            "tdp43_full_wt", "fus_full_wt", "hnrnpa1_full_wt",
            "tia1_full_wt", "ewsr1_full_wt", "taf15_full_wt", "hnrnpa2b1_full_wt",
            "prp_mature_wt",
            "httex1_q21_wt", "ataxin3_q14_wt",
            "ddx4_full_wt", "npm1_full_wt"}

WT_MUTANT_PAIRS = [
    ("alpha_synuclein_ensemble", "asyn_A53T", "A53T"),
    ("alpha_synuclein_ensemble", "asyn_E46K", "E46K"),
    ("tau_2n4r_wt", "tau_P301L", "P301L"),
    ("tdp43_full_wt", "tdp43_M337V", "M337V"),
    ("tdp43_full_wt", "tdp43_Q331K", "Q331K"),
    ("fus_full_wt", "fus_P525L", "P525L"),
    ("hnrnpa1_full_wt", "hnrnpa1_D262V", "D262V"),
    ("httex1_q21_wt", "httex1_q46", "Q21→Q46"),
    ("tia1_full_wt", "tia1_P362L", "P362L"),
    ("ataxin3_q14_wt", "ataxin3_q72", "Q14→Q72"),
]


def load_data():
    return pd.read_csv(FEATURES / "combined_features.csv")


def figure1_landscape_biplot(df):
    """main figure: PCA biplot without ESM2, with feature loading arrows."""
    feature_cols = [
        "ncpr", "fcr", "gravy", "kappa", "scd",
        "aromatic_frac", "glycine_frac", "proline_frac", "qn_frac", "serine_frac",
        "tyr_frac", "arg_frac", "aromatic_clustering", "mean_beta_propensity",
        "full_mean_p_lock", "full_mean_p_pi", "full_mean_p_polar",
        "lcd_mean_p_lock", "lcd_lockability_per_pair",
        "n_aprs", "apr_fraction", "max_apr_score", "mean_amyloid_propensity", "larks_count",
        "mean_llr", "emb_norm", "residue_emb_variance",
    ]
    cols = [c for c in feature_cols if c in df.columns]

    X = df[cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    loadings = pca.components_.T  # (n_features, 2)

    # select top features for biplot arrows
    # choose features with highest combined loading magnitude
    loading_mag = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_idx = np.argsort(loading_mag)[-8:]  # top 8 feature arrows

    # readable names for the biplot arrows
    nice_names = {
        "ncpr": "NCPR", "fcr": "FCR", "gravy": "GRAVY", "kappa": "κ", "scd": "SCD",
        "aromatic_frac": "aromatic", "glycine_frac": "glycine", "proline_frac": "proline",
        "qn_frac": "Q/N", "serine_frac": "serine", "tyr_frac": "tyrosine",
        "arg_frac": "arginine", "aromatic_clustering": "arom. clustering",
        "mean_beta_propensity": "β-propensity",
        "full_mean_p_lock": "p_lock", "full_mean_p_pi": "p_π",
        "full_mean_p_polar": "p_polar",
        "lcd_mean_p_lock": "LCD p_lock", "lcd_lockability_per_pair": "LCD lockability",
        "n_aprs": "# APRs", "apr_fraction": "APR fraction",
        "max_apr_score": "max APR", "mean_amyloid_propensity": "amyloid prop.",
        "larks_count": "LARKS", "mean_llr": "ESM2 LLR",
        "emb_norm": "emb. norm", "residue_emb_variance": "emb. variance",
    }

    fig, ax = plt.subplots(figsize=(13, 10))

    # plot data points
    for idx, (_, row) in enumerate(df.iterrows()):
        mech = row["mechanism"]
        color = MECH_COLORS.get(mech, "gray")
        is_wt = row["name"] in WT_NAMES
        size = 250 if is_wt else 120
        marker = "o" if is_wt else "^"
        edge = "black" if is_wt else "none"
        alpha = 1.0 if is_wt else 0.7

        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=color, s=size, marker=marker,
                   edgecolors=edge, linewidths=1.5 if is_wt else 0.5, alpha=alpha, zorder=5)

        if is_wt:
            # offset labels to avoid overlap
            offsets = {
                "fus_full_wt": (12, -15),
                "prp_mature_wt": (12, -15),
                "taf15_full_wt": (-60, -15),
            }
            xt, yt = offsets.get(row["name"], (10, 8))
            ax.annotate(DISPLAY.get(row["name"], row["name"]),
                       (X_pca[idx, 0], X_pca[idx, 1]),
                       textcoords="offset points", xytext=(xt, yt),
                       fontsize=11, fontweight="bold", color=color)

    # draw mutation arrows
    for wt_name, mut_name, label in WT_MUTANT_PAIRS:
        wt_idx = df[df["name"] == wt_name].index[0]
        mut_idx = df[df["name"] == mut_name].index[0]

        dx = X_pca[mut_idx, 0] - X_pca[wt_idx, 0]
        dy = X_pca[mut_idx, 1] - X_pca[wt_idx, 1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist > 0.05:
            ax.annotate("", xy=(X_pca[mut_idx, 0], X_pca[mut_idx, 1]),
                       xytext=(X_pca[wt_idx, 0], X_pca[wt_idx, 1]),
                       arrowprops=dict(arrowstyle="->", color="gray",
                                      lw=1.5, connectionstyle="arc3,rad=0.1"))
            mid_x = (X_pca[wt_idx, 0] + X_pca[mut_idx, 0]) / 2
            mid_y = (X_pca[wt_idx, 1] + X_pca[mut_idx, 1]) / 2
            ax.annotate(label, (mid_x, mid_y), fontsize=7, color="gray",
                       ha="center", va="bottom")

    # biplot arrows (feature loadings)
    # scale arrows to be visible
    arrow_scale = 7.0
    for i in top_idx:
        dx = loadings[i, 0] * arrow_scale
        dy = loadings[i, 1] * arrow_scale
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="-|>", color="darkgreen",
                                  lw=1.0, alpha=0.6))
        # label at arrow tip
        fname = nice_names.get(cols[i], cols[i])
        # slight offset for readability
        ax.text(dx * 1.08, dy * 1.08, fname, fontsize=8, color="darkgreen",
               ha="center", va="center", fontstyle="italic", alpha=0.8)

    # legend
    for mech, color in MECH_COLORS.items():
        ax.scatter([], [], c=color, s=120, label=MECH_LABELS.get(mech, mech))
    ax.scatter([], [], c="gray", s=250, marker="o", edgecolors="black",
               linewidths=1.5, label="wild-type")
    ax.scatter([], [], c="gray", s=120, marker="^", label="mutant")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95, ncol=2)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=13)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=13)
    ax.set_title("IDP disease mechanism landscape\n(27 interpretable sequence features, no ESM2)",
                fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color="black", linewidth=0.3, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_landscape_biplot.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {FIGURES / 'fig1_landscape_biplot.png'}")


def figure2_maturation_comparison(df):
    """the strongest result: maturation grammar across protein classes."""
    # select WT proteins only
    wt_df = df[df["name"].isin(WT_NAMES)].copy()
    wt_df["display"] = wt_df["name"].map(DISPLAY)
    wt_df["mech_color"] = wt_df["mechanism"].map(MECH_COLORS)

    # sort by lcd_lockability_per_pair for visual clarity
    wt_df = wt_df.sort_values("lcd_lockability_per_pair", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # panel A: lcd lockability per pair
    ax = axes[0]
    bars = ax.barh(range(len(wt_df)), wt_df["lcd_lockability_per_pair"],
                   color=wt_df["mech_color"].values, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(wt_df)))
    ax.set_yticklabels(wt_df["display"].values, fontsize=10)
    ax.set_xlabel("LCD lockability per pair", fontsize=11)
    ax.set_title("(A) LCD maturation potential", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    # mark proteins with no LCD
    for i, (_, row) in enumerate(wt_df.iterrows()):
        if not row["has_lcd_region"] or row["lcd_lockability_per_pair"] == 0:
            ax.annotate("no LCD", (0.01, i), fontsize=8, color="gray", fontstyle="italic", va="center")

    # panel B: lcd_mean_p_lock vs lcd_mean_p_pi
    ax = axes[1]
    for _, row in wt_df.iterrows():
        color = MECH_COLORS.get(row["mechanism"], "gray")
        ax.scatter(row["lcd_mean_p_lock"], row["lcd_mean_p_pi"],
                  c=color, s=200, edgecolors="black", linewidths=1.5, zorder=5)
        ax.annotate(DISPLAY.get(row["name"], row["name"]),
                   (row["lcd_mean_p_lock"], row["lcd_mean_p_pi"]),
                   textcoords="offset points", xytext=(8, 5),
                   fontsize=9, color=color, fontweight="bold")
    ax.set_xlabel("LCD p_lock (contact potential)", fontsize=11)
    ax.set_ylabel("LCD p_π (aromatic stacking)", fontsize=11)
    ax.set_title("(B) lock vs π-stacking in LCDs", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # panel C: mutation effects on lockability
    ax = axes[2]
    mutations = [
        ("hnRNPA1\nD262V", "hnrnpa1_full_wt", "hnrnpa1_D262V", "lcd_lockability_per_pair"),
        ("TDP-43\nQ331K", "tdp43_full_wt", "tdp43_Q331K", "lcd_lockability_per_pair"),
        ("TDP-43\nM337V", "tdp43_full_wt", "tdp43_M337V", "lcd_lockability_per_pair"),
        ("FUS\nP525L", "fus_full_wt", "fus_P525L", "lcd_lockability_per_pair"),
        ("TIA1\nP362L", "tia1_full_wt", "tia1_P362L", "lcd_lockability_per_pair"),
    ]

    labels = []
    wt_vals = []
    mut_vals = []
    colors = []

    for label, wt_name, mut_name, feat in mutations:
        wt_val = df[df["name"] == wt_name][feat].values[0]
        mut_val = df[df["name"] == mut_name][feat].values[0]
        labels.append(label)
        wt_vals.append(wt_val)
        mut_vals.append(mut_val)
        colors.append(MECH_COLORS.get(df[df["name"] == wt_name]["mechanism"].values[0], "gray"))

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, wt_vals, width, label="wild-type", color="lightgray", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, mut_vals, width, label="mutant", color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("LCD lockability per pair", fontsize=11)
    ax.set_title("(C) mutation effects on maturation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # annotate percent change
    for i in range(len(labels)):
        if wt_vals[i] > 0:
            pct = (mut_vals[i] - wt_vals[i]) / wt_vals[i] * 100
            y_pos = max(wt_vals[i], mut_vals[i]) + 0.01
            ax.text(x[i], y_pos, f"{pct:+.1f}%", ha="center", fontsize=9, fontweight="bold",
                   color="darkred" if pct > 0 else "darkblue")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_maturation_grammar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {FIGURES / 'fig2_maturation_grammar.png'}")


def figure3_amyloid_vs_maturation(df):
    """key scatter: amyloid propensity vs maturation grammar (the two axes that matter)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for _, row in df.iterrows():
        mech = row["mechanism"]
        color = MECH_COLORS.get(mech, "gray")
        is_wt = row["name"] in WT_NAMES
        size = 250 if is_wt else 100
        marker = "o" if is_wt else "^"
        edge = "black" if is_wt else "none"

        ax.scatter(row["mean_amyloid_propensity"], row["full_mean_p_lock"],
                  c=color, s=size, marker=marker,
                  edgecolors=edge, linewidths=1.5 if is_wt else 0.5,
                  alpha=1.0 if is_wt else 0.7, zorder=5)

        if is_wt:
            offsets = {
                "alpha_synuclein_ensemble": (12, 5),
                "ab42_wt": (12, 5),
                "tau_2n4r_wt": (-50, -15),
                "tdp43_full_wt": (12, 5),
                "fus_full_wt": (12, -12),
                "hnrnpa1_full_wt": (12, 5),
                "prp_mature_wt": (12, 5),
                "httex1_q21_wt": (12, 5),
            }
            xt, yt = offsets.get(row["name"], (10, 5))
            ax.annotate(DISPLAY.get(row["name"], row["name"]),
                       (row["mean_amyloid_propensity"], row["full_mean_p_lock"]),
                       textcoords="offset points", xytext=(xt, yt),
                       fontsize=10, fontweight="bold", color=color)

    # draw mutation arrows for condensate maturation proteins
    for wt_name, mut_name, label in WT_MUTANT_PAIRS:
        wt_row = df[df["name"] == wt_name].iloc[0]
        mut_row = df[df["name"] == mut_name].iloc[0]
        dx = mut_row["mean_amyloid_propensity"] - wt_row["mean_amyloid_propensity"]
        dy = mut_row["full_mean_p_lock"] - wt_row["full_mean_p_lock"]
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0.001:
            ax.annotate("", xy=(mut_row["mean_amyloid_propensity"], mut_row["full_mean_p_lock"]),
                       xytext=(wt_row["mean_amyloid_propensity"], wt_row["full_mean_p_lock"]),
                       arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    # quadrant labels
    ax.text(0.95, 0.95, "high amyloid\nhigh maturation\n(TDP-43-like)", transform=ax.transAxes,
           fontsize=9, ha="right", va="top", color="gray", fontstyle="italic", alpha=0.6)
    ax.text(0.05, 0.95, "low amyloid\nhigh maturation\n(FUS-like)", transform=ax.transAxes,
           fontsize=9, ha="left", va="top", color="gray", fontstyle="italic", alpha=0.6)
    ax.text(0.95, 0.05, "high amyloid\nlow maturation\n(α-syn-like)", transform=ax.transAxes,
           fontsize=9, ha="right", va="bottom", color="gray", fontstyle="italic", alpha=0.6)
    ax.text(0.05, 0.05, "low amyloid\nlow maturation\n(Htt-like)", transform=ax.transAxes,
           fontsize=9, ha="left", va="bottom", color="gray", fontstyle="italic", alpha=0.6)

    # legend
    for mech, color in MECH_COLORS.items():
        ax.scatter([], [], c=color, s=120, label=MECH_LABELS.get(mech, mech))
    ax.legend(loc="center left", fontsize=9, framealpha=0.95)

    ax.set_xlabel("mean amyloid propensity", fontsize=13)
    ax.set_ylabel("maturation grammar (p_lock)", fontsize=13)
    ax.set_title("amyloid propensity vs maturation grammar\n(the two axes that separate IDP disease mechanisms)",
                fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_amyloid_vs_maturation.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {FIGURES / 'fig3_amyloid_vs_maturation.png'}")


def figure4_key_features_barplot(df):
    """bar plot of key features across WT proteins, grouped by mechanism."""
    features = [
        ("mean_amyloid_propensity", "amyloid propensity"),
        ("full_mean_p_lock", "maturation grammar (p_lock)"),
        ("lcd_lockability_per_pair", "LCD lockability"),
        ("qn_frac", "Q/N content"),
        ("aromatic_frac", "aromatic fraction"),
        ("proline_frac", "proline fraction"),
    ]

    # order: amyloid, condensate, dual, template, polyQ, functional
    wt_order = ["ab42_wt", "alpha_synuclein_ensemble", "iapp_mature_wt",
                "tdp43_full_wt", "fus_full_wt", "hnrnpa1_full_wt",
                "tia1_full_wt", "ewsr1_full_wt", "taf15_full_wt", "hnrnpa2b1_full_wt",
                "tau_2n4r_wt",
                "prp_mature_wt",
                "httex1_q21_wt", "ataxin3_q14_wt",
                "ddx4_full_wt", "npm1_full_wt"]
    wt_df = df[df["name"].isin(WT_NAMES)].set_index("name").loc[wt_order]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for idx, (feat, label) in enumerate(features):
        ax = axes[idx // 3, idx % 3]
        colors = [MECH_COLORS.get(wt_df.loc[n, "mechanism"], "gray") for n in wt_order]
        labels = [DISPLAY.get(n, n) for n in wt_order]

        ax.bar(range(len(wt_order)), wt_df[feat].values,
               color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(wt_order)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"({chr(65 + idx)}) {label}", fontsize=11, fontweight="bold")

    fig.suptitle("key feature comparison across IDP disease mechanisms",
                fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_feature_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {FIGURES / 'fig4_feature_comparison.png'}")


def main():
    df = load_data()

    print("generating improved figures...")
    figure1_landscape_biplot(df)
    figure2_maturation_comparison(df)
    figure3_amyloid_vs_maturation(df)
    figure4_key_features_barplot(df)
    print("\ndone — all improved figures saved")


if __name__ == "__main__":
    main()
