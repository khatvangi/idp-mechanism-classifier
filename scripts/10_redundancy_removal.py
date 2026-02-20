#!/usr/bin/env python3
"""
task 13: remove redundant features (|ρ|>0.8) and re-test mechanism separation.
strategy:
  1. compute Spearman correlations on WT-only data (honest, no pseudo-replication)
  2. greedy removal: iteratively drop the feature with most |ρ|>0.8 connections
  3. re-run permutation test with reduced feature set
  4. compare full vs reduced results
  5. also test no-ESM2 reduced set
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
FIGURES.mkdir(parents=True, exist_ok=True)

# mechanism labels
MECHANISM = {
    "alpha_synuclein_ensemble": "amyloid",
    "asyn_A53T": "amyloid", "asyn_E46K": "amyloid",
    "ab42_wt": "amyloid", "iapp_mature_wt": "amyloid",
    "tau_2n4r_wt": "dual", "tau_P301L": "dual",
    "tdp43_full_wt": "condensate_maturation",
    "tdp43_M337V": "condensate_maturation", "tdp43_Q331K": "condensate_maturation",
    "fus_full_wt": "condensate_maturation", "fus_P525L": "condensate_maturation",
    "hnrnpa1_full_wt": "condensate_maturation", "hnrnpa1_D262V": "condensate_maturation",
    "tia1_full_wt": "condensate_maturation", "tia1_P362L": "condensate_maturation",
    "ewsr1_full_wt": "condensate_maturation",
    "taf15_full_wt": "condensate_maturation",
    "hnrnpa2b1_full_wt": "condensate_maturation",
    "prp_mature_wt": "template_misfolding",
    "httex1_q21_wt": "polyQ_collapse", "httex1_q46": "polyQ_collapse",
    "ataxin3_q14_wt": "polyQ_collapse", "ataxin3_q72": "polyQ_collapse",
    "ddx4_full_wt": "functional_condensate",
    "npm1_full_wt": "functional_condensate",
}

MECH_COLORS = {
    "amyloid": "#d62728",
    "condensate_maturation": "#1f77b4",
    "dual": "#9467bd",
    "template_misfolding": "#ff7f0e",
    "polyQ_collapse": "#2ca02c",
    "functional_condensate": "#17becf",
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

WT_NAMES = [
    "alpha_synuclein_ensemble", "ab42_wt", "iapp_mature_wt",
    "tau_2n4r_wt",
    "tdp43_full_wt", "fus_full_wt", "hnrnpa1_full_wt",
    "tia1_full_wt", "ewsr1_full_wt", "taf15_full_wt", "hnrnpa2b1_full_wt",
    "prp_mature_wt",
    "httex1_q21_wt", "ataxin3_q14_wt",
    "ddx4_full_wt", "npm1_full_wt",
]

# all interpretable features (no ESM2 PCA)
INTERP_FEATURES = [
    "ncpr", "fcr", "gravy", "kappa", "scd",
    "aromatic_frac", "glycine_frac", "proline_frac", "qn_frac", "serine_frac",
    "tyr_frac", "arg_frac", "aromatic_clustering", "mean_beta_propensity",
    "full_mean_p_lock", "full_mean_p_pi", "full_mean_p_polar",
    "lcd_mean_p_lock", "lcd_lockability_per_pair",
    "n_aprs", "apr_fraction", "max_apr_score", "mean_amyloid_propensity", "larks_count",
    "mean_llr", "emb_norm", "residue_emb_variance",
]


def compute_mean_inter_class_distance(X, labels):
    """mean euclidean distance between class centroids."""
    unique = np.unique(labels)
    centroids = [X[labels == lab].mean(axis=0) for lab in unique]
    n = len(centroids)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(centroids[i] - centroids[j])
            count += 1
    return total / count if count > 0 else 0


def run_permutation_test(X, labels, n_perm=10000, rng_seed=42):
    """permutation test returning p-value, z-score, and distribution."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    real_dist = compute_mean_inter_class_distance(X_scaled, labels)

    rng = np.random.default_rng(rng_seed)
    perm_dists = np.array([
        compute_mean_inter_class_distance(X_scaled, rng.permutation(labels))
        for _ in range(n_perm)
    ])

    p_value = (perm_dists >= real_dist).mean()
    z_score = (real_dist - perm_dists.mean()) / perm_dists.std() if perm_dists.std() > 0 else 0

    return p_value, z_score, real_dist, perm_dists


def greedy_redundancy_removal(corr_matrix, features, threshold=0.8):
    """
    iteratively remove the most-connected feature from pairs with |ρ|>threshold.
    returns the reduced feature list.
    """
    remaining = list(features)
    removed = []

    while True:
        # count edges for each remaining feature
        edge_count = {f: 0 for f in remaining}
        for i, fi in enumerate(remaining):
            for j in range(i + 1, len(remaining)):
                fj = remaining[j]
                if abs(corr_matrix.loc[fi, fj]) > threshold:
                    edge_count[fi] += 1
                    edge_count[fj] += 1

        # find the feature with most edges
        max_edges = max(edge_count.values())
        if max_edges == 0:
            break  # no more redundant pairs

        # among features with max_edges, remove the one with highest total |ρ|
        # (tie-breaking by total correlation burden)
        candidates = [f for f, e in edge_count.items() if e == max_edges]
        if len(candidates) > 1:
            total_corr = {}
            for f in candidates:
                total_corr[f] = sum(
                    abs(corr_matrix.loc[f, g])
                    for g in remaining if g != f and abs(corr_matrix.loc[f, g]) > threshold
                )
            to_remove = max(candidates, key=lambda f: total_corr[f])
        else:
            to_remove = candidates[0]

        remaining.remove(to_remove)
        removed.append(to_remove)

    return remaining, removed


def main():
    df = pd.read_csv(FEATURES / "combined_features.csv")
    print(f"loaded {len(df)} proteins × {len(df.columns)} columns")

    # ESM2 PCA columns
    esm_cols = [c for c in df.columns if c.startswith("esm2_pc")]
    all_features = [c for c in INTERP_FEATURES + esm_cols if c in df.columns]
    interp_only = [c for c in INTERP_FEATURES if c in df.columns]

    print(f"total features: {len(all_features)} ({len(interp_only)} interpretable + {len(esm_cols)} ESM2 PCA)")

    # === step 1: compute correlations on WT-only ===
    print("\n" + "=" * 60)
    print("STEP 1: feature correlations (16 WT only)")
    print("=" * 60)

    wt_df = df[df["name"].isin(WT_NAMES)].copy()
    print(f"  using {len(wt_df)} WT proteins for correlation")

    # Spearman on all features
    corr_all = wt_df[all_features].corr(method="spearman")
    corr_interp = wt_df[interp_only].corr(method="spearman")

    # === step 2: greedy redundancy removal ===
    print("\n" + "=" * 60)
    print("STEP 2: greedy redundancy removal (|ρ| > 0.8)")
    print("=" * 60)

    # a) all features (interpretable + ESM2)
    reduced_all, removed_all = greedy_redundancy_removal(corr_all, all_features, threshold=0.8)
    print(f"\n  all features: {len(all_features)} → {len(reduced_all)} (removed {len(removed_all)})")
    print(f"  removed: {removed_all}")
    print(f"  kept: {reduced_all}")

    # b) interpretable only (no ESM2)
    reduced_interp, removed_interp = greedy_redundancy_removal(corr_interp, interp_only, threshold=0.8)
    print(f"\n  interpretable only: {len(interp_only)} → {len(reduced_interp)} (removed {len(removed_interp)})")
    print(f"  removed: {removed_interp}")
    print(f"  kept: {reduced_interp}")

    # verify no remaining pairs above threshold
    for label, feats, corr in [("all_reduced", reduced_all, corr_all),
                                ("interp_reduced", reduced_interp, corr_interp)]:
        max_r = 0
        for i, fi in enumerate(feats):
            for j in range(i + 1, len(feats)):
                fj = feats[j]
                r = abs(corr.loc[fi, fj])
                if r > max_r:
                    max_r = r
        print(f"  {label}: max remaining |ρ| = {max_r:.3f}")

    # === step 3: permutation tests ===
    print("\n" + "=" * 60)
    print("STEP 3: permutation tests (10,000 permutations)")
    print("=" * 60)

    labels_all = df["mechanism"].values
    labels_wt = wt_df["mechanism"].values

    conditions = [
        # (name, data_df, features, label_array)
        ("all 26, full features", df, all_features, labels_all),
        ("all 26, reduced features", df, reduced_all, labels_all),
        ("all 26, interp only (no ESM2)", df, interp_only, labels_all),
        ("all 26, interp reduced", df, reduced_interp, labels_all),
        ("16 WT, full features", wt_df, all_features, labels_wt),
        ("16 WT, reduced features", wt_df, reduced_all, labels_wt),
        ("16 WT, interp only (no ESM2)", wt_df, interp_only, labels_wt),
        ("16 WT, interp reduced", wt_df, reduced_interp, labels_wt),
    ]

    results = []
    for name, data, feats, labs in conditions:
        avail = [f for f in feats if f in data.columns]
        X = data[avail].values.astype(float)
        p, z, real_d, perm_d = run_permutation_test(X, labs)
        results.append({
            "condition": name,
            "n_proteins": len(data),
            "n_features": len(avail),
            "p_value": p,
            "z_score": z,
            "real_dist": real_d,
        })
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"\n  {name}:")
        print(f"    {len(avail)} features, observed dist = {real_d:.4f}")
        print(f"    p = {p:.4f}, z = {z:.2f}  [{sig}]")

    # === step 4: summary table ===
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"\n  {'condition':<35s} {'n':>3} {'feat':>4} {'p-value':>9} {'z':>6} {'sig':>4}")
    print("  " + "-" * 65)
    for r in results:
        p = r["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {r['condition']:<35s} {r['n_proteins']:>3} {r['n_features']:>4} {p:>9.4f} {r['z_score']:>6.2f} {sig:>4}")

    # === step 5: PCA landscape with reduced features ===
    print("\n" + "=" * 60)
    print("STEP 5: PCA landscapes with reduced features")
    print("=" * 60)

    # plot both: reduced-all and reduced-interp
    for label, feats, suffix in [
        ("reduced (all)", reduced_all, "reduced_all"),
        ("reduced (interpretable, no ESM2)", reduced_interp, "reduced_interp"),
    ]:
        avail = [f for f in feats if f in df.columns]
        X = df[avail].values.astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        ev1, ev2 = pca.explained_variance_ratio_[:2]
        print(f"\n  {label}: PC1={ev1:.1%}, PC2={ev2:.1%}, total={ev1+ev2:.1%}")

        # top loadings
        loadings = pd.DataFrame(pca.components_.T, index=avail, columns=["PC1", "PC2"])
        loadings["abs_PC1"] = loadings["PC1"].abs()
        loadings["abs_PC2"] = loadings["PC2"].abs()
        print(f"  top PC1 drivers:")
        for _, row in loadings.nlargest(5, "abs_PC1").iterrows():
            print(f"    {row.name:30s} {row['PC1']:+.3f}")
        print(f"  top PC2 drivers:")
        for _, row in loadings.nlargest(5, "abs_PC2").iterrows():
            print(f"    {row.name:30s} {row['PC2']:+.3f}")

        # plot
        fig, ax = plt.subplots(figsize=(12, 9))

        wt_set = set(WT_NAMES)
        for idx, (_, row) in enumerate(df.iterrows()):
            mech = row["mechanism"]
            color = MECH_COLORS.get(mech, "gray")
            is_wt = row["name"] in wt_set
            size = 200 if is_wt else 100
            marker = "o" if is_wt else "^"
            edge = "black" if is_wt else "none"

            ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=color, s=size, marker=marker,
                       edgecolors=edge, linewidths=1.5 if is_wt else 0,
                       alpha=1.0 if is_wt else 0.7, zorder=5)

            if is_wt:
                ax.annotate(DISPLAY.get(row["name"], row["name"]),
                           (X_pca[idx, 0], X_pca[idx, 1]),
                           textcoords="offset points", xytext=(10, 8),
                           fontsize=9, fontweight="bold", color=color)

        # biplot arrows (top 8 features)
        top_feats = loadings.assign(importance=loadings["abs_PC1"] + loadings["abs_PC2"]).nlargest(8, "importance")
        scale = np.abs(X_pca).max() * 0.8
        for fname, row in top_feats.iterrows():
            ax.annotate("", xy=(row["PC1"] * scale, row["PC2"] * scale), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, alpha=0.6))
            ax.annotate(fname, (row["PC1"] * scale, row["PC2"] * scale),
                       fontsize=7, color="gray", alpha=0.8, ha="center")

        for mech, color in MECH_COLORS.items():
            ax.scatter([], [], c=color, s=100, label=mech.replace("_", " "))
        ax.legend(loc="best", fontsize=9)

        ax.set_xlabel(f"PC1 ({ev1:.1%} variance)", fontsize=12)
        ax.set_ylabel(f"PC2 ({ev2:.1%} variance)", fontsize=12)
        ax.set_title(f"IDP mechanism landscape — {label}\n({len(avail)} features, redundancy removed)",
                    fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / f"landscape_{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved: {FIGURES / f'landscape_{suffix}.png'}")

        # save loadings
        loadings.to_csv(FEATURES / f"pca_loadings_{suffix}.csv")

    # === step 6: permutation test visualization (reduced WT-only) ===
    print("\n" + "=" * 60)
    print("STEP 6: WT-only permutation test plots (reduced features)")
    print("=" * 60)

    for label, feats, suffix in [
        ("reduced (all)", reduced_all, "reduced_all"),
        ("reduced (interpretable)", reduced_interp, "reduced_interp"),
    ]:
        avail = [f for f in feats if f in wt_df.columns]
        X_wt = wt_df[avail].values.astype(float)
        p, z, real_d, perm_d = run_permutation_test(X_wt, labels_wt)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(perm_d, bins=50, alpha=0.7, color="coral", edgecolor="black", linewidth=0.5)
        ax.axvline(real_d, color="darkred", linewidth=2, linestyle="--",
                   label=f"observed = {real_d:.3f}")
        ax.set_xlabel("mean inter-class centroid distance", fontsize=11)
        ax.set_ylabel("count", fontsize=11)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"permutation test: 16 WT, {label}\n{len(avail)} features, p = {p:.4f} [{sig}]",
                    fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGURES / f"permutation_wt_{suffix}.png", dpi=150)
        plt.close()
        print(f"  p = {p:.4f} [{sig}], saved: permutation_wt_{suffix}.png")

    # === step 7: save reduced feature list for downstream use ===
    with open(FEATURES / "reduced_features.txt", "w") as f:
        f.write("# reduced feature set after greedy redundancy removal (|rho|>0.8)\n")
        f.write(f"# computed on {len(wt_df)} WT proteins\n\n")
        f.write("# all reduced (interpretable + ESM2 PCA):\n")
        for feat in reduced_all:
            f.write(f"{feat}\n")
        f.write(f"\n# interpretable reduced (no ESM2):\n")
        for feat in reduced_interp:
            f.write(f"{feat}\n")
    print(f"\n  saved reduced feature lists to {FEATURES / 'reduced_features.txt'}")


if __name__ == "__main__":
    main()
