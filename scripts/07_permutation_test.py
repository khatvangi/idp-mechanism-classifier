#!/usr/bin/env python3
"""
phase 4 analysis: permutation test for mechanism separation.
- shuffles mechanism labels 10,000 times
- computes mean inter-class distance each time
- reports p-value: are real distances significantly larger than expected by chance?
also: feature correlation matrix and local mutation analysis.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
FEATURES = PROJECT / "features"
FIGURES = PROJECT / "figures"


def load_combined():
    """load combined features and select numeric columns for analysis."""
    df = pd.read_csv(FEATURES / "combined_features.csv")
    # select the same features used in landscape PCA
    feature_cols = [
        "ncpr", "fcr", "gravy", "kappa", "scd",
        "aromatic_frac", "glycine_frac", "proline_frac", "qn_frac", "serine_frac",
        "tyr_frac", "arg_frac", "aromatic_clustering", "mean_beta_propensity",
        "full_mean_p_lock", "full_mean_p_pi", "full_mean_p_polar",
        "lcd_mean_p_lock", "lcd_lockability_per_pair",
        "n_aprs", "apr_fraction", "max_apr_score", "mean_amyloid_propensity", "larks_count",
        "mean_llr", "emb_norm", "residue_emb_variance",
    ]
    # add ESM2 PCA columns
    esm_cols = [c for c in df.columns if c.startswith("esm2_pc")]
    all_cols = [c for c in feature_cols + esm_cols if c in df.columns]
    return df, all_cols


def compute_mean_inter_class_distance(X, labels):
    """compute mean euclidean distance between class centroids."""
    unique_labels = np.unique(labels)
    centroids = []
    for lab in unique_labels:
        mask = labels == lab
        centroids.append(X[mask].mean(axis=0))

    # mean pairwise distance between centroids
    n = len(centroids)
    total_dist = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += np.linalg.norm(centroids[i] - centroids[j])
            count += 1
    return total_dist / count if count > 0 else 0


def permutation_test(df, all_cols, n_perm=10000):
    """test if mechanism separation is significant via label permutation."""
    print("=" * 60)
    print("PERMUTATION TEST: mechanism label significance")
    print("=" * 60)

    X = df[all_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels = df["mechanism"].values
    real_dist = compute_mean_inter_class_distance(X_scaled, labels)
    print(f"\n  real mean inter-class distance: {real_dist:.4f}")

    # permutation
    perm_dists = []
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        shuffled = rng.permutation(labels)
        d = compute_mean_inter_class_distance(X_scaled, shuffled)
        perm_dists.append(d)

    perm_dists = np.array(perm_dists)
    p_value = (perm_dists >= real_dist).mean()

    print(f"  permutation distribution: mean={perm_dists.mean():.4f}, "
          f"std={perm_dists.std():.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  z-score: {(real_dist - perm_dists.mean()) / perm_dists.std():.2f}")

    # plot permutation distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_dists, bins=50, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.axvline(real_dist, color="red", linewidth=2, linestyle="--", label=f"observed = {real_dist:.3f}")
    ax.set_xlabel("mean inter-class centroid distance", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(f"permutation test (n={n_perm}): p = {p_value:.4f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "permutation_test.png", dpi=150)
    plt.close()
    print(f"  saved: {FIGURES / 'permutation_test.png'}")

    return p_value, real_dist, perm_dists


def permutation_test_wt_only(df, all_cols, n_perm=10000):
    """same test but only on WT proteins (8 independent points)."""
    print("\n" + "=" * 60)
    print("PERMUTATION TEST: WT-only (8 independent proteins)")
    print("=" * 60)

    wt_names = ["alpha_synuclein_ensemble", "ab42_wt", "iapp_mature_wt",
                "tau_2n4r_wt",
                "tdp43_full_wt", "fus_full_wt", "hnrnpa1_full_wt",
                "tia1_full_wt", "ewsr1_full_wt", "taf15_full_wt", "hnrnpa2b1_full_wt",
                "prp_mature_wt",
                "httex1_q21_wt", "ataxin3_q14_wt",
                "ddx4_full_wt", "npm1_full_wt"]
    wt_df = df[df["name"].isin(wt_names)].copy()

    X = wt_df[all_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels = wt_df["mechanism"].values
    real_dist = compute_mean_inter_class_distance(X_scaled, labels)
    print(f"\n  real mean inter-class distance (WT only): {real_dist:.4f}")

    # permutation
    perm_dists = []
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        shuffled = rng.permutation(labels)
        d = compute_mean_inter_class_distance(X_scaled, shuffled)
        perm_dists.append(d)

    perm_dists = np.array(perm_dists)
    p_value = (perm_dists >= real_dist).mean()

    print(f"  permutation distribution: mean={perm_dists.mean():.4f}, "
          f"std={perm_dists.std():.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  z-score: {(real_dist - perm_dists.mean()) / perm_dists.std():.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_dists, bins=50, alpha=0.7, color="coral", edgecolor="black", linewidth=0.5)
    ax.axvline(real_dist, color="darkred", linewidth=2, linestyle="--", label=f"observed = {real_dist:.3f}")
    ax.set_xlabel("mean inter-class centroid distance", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(f"permutation test WT-only (n={n_perm}, 8 proteins): p = {p_value:.4f}",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "permutation_test_wt_only.png", dpi=150)
    plt.close()
    print(f"  saved: {FIGURES / 'permutation_test_wt_only.png'}")

    return p_value


def permutation_test_no_esm2(df, n_perm=10000):
    """permutation test WITHOUT ESM2 features — only interpretable features."""
    print("\n" + "=" * 60)
    print("PERMUTATION TEST: no ESM2 (interpretable features only)")
    print("=" * 60)

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

    labels = df["mechanism"].values
    real_dist = compute_mean_inter_class_distance(X_scaled, labels)
    print(f"\n  real distance (no ESM2): {real_dist:.4f}")

    perm_dists = []
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        shuffled = rng.permutation(labels)
        d = compute_mean_inter_class_distance(X_scaled, shuffled)
        perm_dists.append(d)

    perm_dists = np.array(perm_dists)
    p_value = (perm_dists >= real_dist).mean()

    print(f"  permutation distribution: mean={perm_dists.mean():.4f}, std={perm_dists.std():.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  z-score: {(real_dist - perm_dists.mean()) / perm_dists.std():.2f}")

    return p_value


def feature_correlation_analysis(df, all_cols):
    """compute and visualize inter-feature correlations."""
    print("\n" + "=" * 60)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 60)

    X = df[all_cols]

    # Spearman correlations (rank-based, robust to outliers)
    corr = X.corr(method="spearman")

    # find highly correlated pairs (|r| > 0.8)
    print("\n  highly correlated feature pairs (|ρ| > 0.8):")
    pairs_found = False
    for i in range(len(all_cols)):
        for j in range(i + 1, len(all_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.8:
                print(f"    {all_cols[i]:30s} ↔ {all_cols[j]:30s}  ρ = {r:+.3f}")
                pairs_found = True
    if not pairs_found:
        print("    none found")

    # cross-axis correlations (polymer physics vs maturation grammar)
    pp_cols = ["ncpr", "fcr", "gravy", "kappa", "scd", "aromatic_frac",
               "glycine_frac", "proline_frac", "qn_frac", "serine_frac"]
    mg_cols = ["full_mean_p_lock", "full_mean_p_pi", "full_mean_p_polar",
               "lcd_mean_p_lock", "lcd_lockability_per_pair"]
    ap_cols = ["n_aprs", "apr_fraction", "max_apr_score", "mean_amyloid_propensity"]

    print("\n  cross-axis correlations (max |ρ| between axis pairs):")
    for name_a, cols_a, name_b, cols_b in [
        ("polymer_physics", pp_cols, "maturation_grammar", mg_cols),
        ("polymer_physics", pp_cols, "amyloid_propensity", ap_cols),
        ("maturation_grammar", mg_cols, "amyloid_propensity", ap_cols),
    ]:
        max_r = 0
        max_pair = ("", "")
        for ca in cols_a:
            for cb in cols_b:
                if ca in corr.columns and cb in corr.columns:
                    r = abs(corr.loc[ca, cb])
                    if r > max_r:
                        max_r = r
                        max_pair = (ca, cb)
        print(f"    {name_a:25s} ↔ {name_b:25s}  max |ρ| = {max_r:.3f} ({max_pair[0]} ↔ {max_pair[1]})")

    # save correlation heatmap
    fig, ax = plt.subplots(figsize=(16, 14))

    # only plot interpretable features (not ESM2 PCA which are already orthogonal)
    interp_cols = [c for c in all_cols if not c.startswith("esm2_pc")]
    interp_corr = X[interp_cols].corr(method="spearman")

    im = ax.imshow(interp_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(interp_cols)))
    ax.set_yticks(range(len(interp_cols)))
    ax.set_xticklabels(interp_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(interp_cols, fontsize=7)
    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    ax.set_title("feature correlation matrix (Spearman, interpretable features only)",
                fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_correlations.png", dpi=150)
    plt.close()
    print(f"  saved: {FIGURES / 'feature_correlations.png'}")


def local_mutation_analysis(df):
    """analyze mutation effects using per-feature z-scores within each mechanism class."""
    print("\n" + "=" * 60)
    print("LOCAL MUTATION ANALYSIS")
    print("=" * 60)

    # key features for mutation analysis
    features = {
        "kappa": "charge patterning",
        "scd": "SCD",
        "full_mean_p_lock": "maturation grammar",
        "lcd_lockability_per_pair": "LCD lockability",
        "mean_amyloid_propensity": "amyloid propensity",
        "apr_fraction": "APR fraction",
        "mean_llr": "ESM2 LLR",
        "qn_frac": "Q/N fraction",
    }

    wt_mutant_pairs = [
        ("alpha_synuclein_ensemble", "asyn_A53T", "α-syn A53T"),
        ("alpha_synuclein_ensemble", "asyn_E46K", "α-syn E46K"),
        ("tau_2n4r_wt", "tau_P301L", "Tau P301L"),
        ("tdp43_full_wt", "tdp43_M337V", "TDP-43 M337V"),
        ("tdp43_full_wt", "tdp43_Q331K", "TDP-43 Q331K"),
        ("fus_full_wt", "fus_P525L", "FUS P525L"),
        ("hnrnpa1_full_wt", "hnrnpa1_D262V", "hnRNPA1 D262V"),
        ("httex1_q21_wt", "httex1_q46", "Htt Q21→Q46"),
    ]

    print(f"\n  {'mutation':>20s}", end="")
    for feat, label in features.items():
        print(f"  {label[:12]:>12s}", end="")
    print()
    print("  " + "-" * (20 + 14 * len(features)))

    for wt_name, mut_name, label in wt_mutant_pairs:
        wt = df[df["name"] == wt_name].iloc[0]
        mut = df[df["name"] == mut_name].iloc[0]

        print(f"  {label:>20s}", end="")
        for feat in features:
            delta = mut[feat] - wt[feat]
            # express as fraction of WT value (when non-zero)
            if abs(wt[feat]) > 1e-10:
                pct = delta / abs(wt[feat]) * 100
                print(f"  {pct:>+11.1f}%", end="")
            else:
                print(f"  {delta:>+11.4f}", end="")
        print()


def landscape_without_esm2(df):
    """generate PCA landscape using only interpretable features (no ESM2)."""
    print("\n" + "=" * 60)
    print("PCA LANDSCAPE WITHOUT ESM2")
    print("=" * 60)

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

    print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"  total: {pca.explained_variance_ratio_.sum():.1%}")

    # loadings
    loadings = pd.DataFrame(pca.components_.T, index=cols, columns=["PC1", "PC2"])
    loadings["abs_PC1"] = loadings["PC1"].abs()
    loadings["abs_PC2"] = loadings["PC2"].abs()

    print("\n  top PC1 drivers (no ESM2):")
    for _, row in loadings.nlargest(5, "abs_PC1").iterrows():
        print(f"    {row.name:30s} {row['PC1']:+.3f}")
    print("  top PC2 drivers (no ESM2):")
    for _, row in loadings.nlargest(5, "abs_PC2").iterrows():
        print(f"    {row.name:30s} {row['PC2']:+.3f}")

    # mechanism colors
    MECH_COLORS = {
        "amyloid": "#d62728",
        "condensate_maturation": "#1f77b4",
        "dual": "#9467bd",
        "template_misfolding": "#ff7f0e",
        "polyQ_collapse": "#2ca02c",
    }
    DISPLAY = {
        "alpha_synuclein_ensemble": "α-syn", "asyn_A53T": "α-syn A53T",
        "asyn_E46K": "α-syn E46K", "ab42_wt": "Aβ42",
        "tau_2n4r_wt": "Tau", "tau_P301L": "Tau P301L",
        "tdp43_full_wt": "TDP-43", "tdp43_M337V": "TDP-43 M337V",
        "tdp43_Q331K": "TDP-43 Q331K", "fus_full_wt": "FUS",
        "fus_P525L": "FUS P525L", "hnrnpa1_full_wt": "hnRNPA1",
        "hnrnpa1_D262V": "hnRNPA1 D262V", "prp_mature_wt": "PrP",
        "httex1_q21_wt": "Htt Q21", "httex1_q46": "Htt Q46",
    }

    wt_names = {"alpha_synuclein_ensemble", "ab42_wt", "tau_2n4r_wt",
                "tdp43_full_wt", "fus_full_wt", "hnrnpa1_full_wt",
                "prp_mature_wt", "httex1_q21_wt"}

    fig, ax = plt.subplots(figsize=(12, 9))

    for idx, (_, row) in enumerate(df.iterrows()):
        mech = row["mechanism"]
        color = MECH_COLORS.get(mech, "gray")
        is_wt = row["name"] in wt_names
        size = 200 if is_wt else 100
        marker = "o" if is_wt else "^"
        edge = "black" if is_wt else "none"

        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=color, s=size, marker=marker,
                   edgecolors=edge, linewidths=1.5 if is_wt else 0, alpha=1.0 if is_wt else 0.7, zorder=5)

        if is_wt:
            ax.annotate(DISPLAY.get(row["name"], row["name"]),
                       (X_pca[idx, 0], X_pca[idx, 1]),
                       textcoords="offset points", xytext=(10, 8),
                       fontsize=10, fontweight="bold", color=color)

    for mech, color in MECH_COLORS.items():
        ax.scatter([], [], c=color, s=100, label=mech.replace("_", " "))
    ax.legend(loc="best", fontsize=9)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title("IDP mechanism landscape — NO ESM2\n(27 interpretable features only)",
                fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "landscape_no_esm2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  saved: {FIGURES / 'landscape_no_esm2.png'}")


def main():
    df, all_cols = load_combined()

    # 1. permutation test with all features (all 16 proteins)
    p_all, real_dist, perm_dists = permutation_test(df, all_cols)

    # 2. permutation test WT-only (8 independent proteins)
    p_wt = permutation_test_wt_only(df, all_cols)

    # 3. permutation test without ESM2
    p_no_esm = permutation_test_no_esm2(df)

    # 4. feature correlation analysis
    feature_correlation_analysis(df, all_cols)

    # 5. local mutation analysis
    local_mutation_analysis(df)

    # 6. landscape without ESM2
    landscape_without_esm2(df)

    # summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_total = len(df)
    n_wt = len([n for n in df["name"] if not any(m in n for m in ["A53T", "E46K", "P301L", "M337V", "Q331K", "P525L", "D262V", "q46", "q72", "P362L", "D290V"])])
    print(f"  permutation p-value (all {n_total}, all features):     {p_all:.4f}")
    print(f"  permutation p-value ({n_wt} WT only, all features):  {p_wt:.4f}")
    print(f"  permutation p-value (all {n_total}, no ESM2):          {p_no_esm:.4f}")
    print(f"\n  interpretation:")
    if p_all < 0.05:
        print(f"    all-feature separation is significant (p={p_all:.4f})")
    else:
        print(f"    all-feature separation is NOT significant (p={p_all:.4f})")
    if p_wt < 0.05:
        print(f"    WT-only separation is significant (p={p_wt:.4f})")
    else:
        print(f"    WT-only separation is NOT significant (p={p_wt:.4f})")
        print(f"    → separation likely driven by pseudo-replication of mutants")


if __name__ == "__main__":
    main()
