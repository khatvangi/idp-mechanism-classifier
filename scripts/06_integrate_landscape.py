#!/usr/bin/env python3
"""
step 6: integrate all four feature axes into mechanism landscape.
- combine polymer physics + maturation grammar + amyloid propensity + ESM2 embeddings
- PCA and UMAP visualization
- mutation arrows (WT → mutant)
- mechanism clustering validation
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# try UMAP, fall back gracefully
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

PROJECT = Path(__file__).resolve().parent.parent
FEATURES = PROJECT / "features"
FIGURES = PROJECT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# -- known mechanism labels --
MECHANISM = {
    "alpha_synuclein_ensemble": "amyloid",
    "asyn_A53T": "amyloid",
    "asyn_E46K": "amyloid",
    "ab42_wt": "amyloid",
    "iapp_mature_wt": "amyloid",
    "tau_2n4r_wt": "dual",
    "tau_P301L": "dual",
    "tdp43_full_wt": "condensate_maturation",
    "tdp43_M337V": "condensate_maturation",
    "tdp43_Q331K": "condensate_maturation",
    "fus_full_wt": "condensate_maturation",
    "fus_P525L": "condensate_maturation",
    "hnrnpa1_full_wt": "condensate_maturation",
    "hnrnpa1_D262V": "condensate_maturation",
    "tia1_full_wt": "condensate_maturation",
    "tia1_P362L": "condensate_maturation",
    "ewsr1_full_wt": "condensate_maturation",
    "taf15_full_wt": "condensate_maturation",
    "hnrnpa2b1_full_wt": "condensate_maturation",
    "prp_mature_wt": "template_misfolding",
    "httex1_q21_wt": "polyQ_collapse",
    "httex1_q46": "polyQ_collapse",
    "ataxin3_q14_wt": "polyQ_collapse",
    "ataxin3_q72": "polyQ_collapse",
    "ddx4_full_wt": "functional_condensate",
    "npm1_full_wt": "functional_condensate",
}

# mechanism colors
MECH_COLORS = {
    "amyloid": "#d62728",               # red
    "condensate_maturation": "#1f77b4",  # blue
    "dual": "#9467bd",                   # purple
    "template_misfolding": "#ff7f0e",    # orange
    "polyQ_collapse": "#2ca02c",         # green
    "functional_condensate": "#17becf",  # cyan
}

# WT-mutant pairs (WT_name, mutant_name, mutation_label)
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

# display names (short)
DISPLAY = {
    "alpha_synuclein_ensemble": "α-syn",
    "asyn_A53T": "α-syn A53T",
    "asyn_E46K": "α-syn E46K",
    "ab42_wt": "Aβ42",
    "iapp_mature_wt": "IAPP",
    "tau_2n4r_wt": "Tau",
    "tau_P301L": "Tau P301L",
    "tdp43_full_wt": "TDP-43",
    "tdp43_M337V": "TDP-43 M337V",
    "tdp43_Q331K": "TDP-43 Q331K",
    "fus_full_wt": "FUS",
    "fus_P525L": "FUS P525L",
    "hnrnpa1_full_wt": "hnRNPA1",
    "hnrnpa1_D262V": "hnRNPA1 D262V",
    "tia1_full_wt": "TIA1",
    "tia1_P362L": "TIA1 P362L",
    "ewsr1_full_wt": "EWSR1",
    "taf15_full_wt": "TAF15",
    "hnrnpa2b1_full_wt": "hnRNPA2B1",
    "prp_mature_wt": "PrP",
    "httex1_q21_wt": "Htt Q21",
    "httex1_q46": "Htt Q46",
    "ataxin3_q14_wt": "Ataxin-3",
    "ataxin3_q72": "Ataxin-3 Q72",
    "ddx4_full_wt": "DDX4",
    "npm1_full_wt": "NPM1",
}


def load_all_features():
    """load and merge all feature CSVs + ESM2 embeddings."""
    # polymer physics
    pp = pd.read_csv(FEATURES / "polymer_physics.csv")

    # maturation grammar
    mg = pd.read_csv(FEATURES / "maturation_grammar.csv")

    # amyloid propensity
    ap = pd.read_csv(FEATURES / "amyloid_propensity.csv")
    ap = ap.drop(columns=["top_apr_regions"], errors="ignore")

    # merge on name
    df = pp.merge(mg, on="name").merge(ap, on="name", suffixes=("", "_ap"))

    # drop duplicate length columns
    for col in df.columns:
        if col.endswith("_ap"):
            df = df.drop(columns=[col])

    # ESM2 summary
    esm = pd.read_csv(FEATURES / "esm2_embeddings_summary.csv")
    df = df.merge(esm, on="name", suffixes=("", "_esm"))
    for col in df.columns:
        if col.endswith("_esm"):
            df = df.drop(columns=[col])

    # add mechanism labels
    df["mechanism"] = df["name"].map(MECHANISM)
    df["display_name"] = df["name"].map(DISPLAY)

    return df


def select_features_for_landscape(df):
    """select numeric features for PCA/UMAP.
    choose interpretable features from each axis."""
    feature_cols = [
        # axis 1: polymer physics
        "ncpr", "fcr", "gravy", "kappa", "scd",
        "aromatic_frac", "glycine_frac", "proline_frac", "qn_frac", "serine_frac",
        "tyr_frac", "arg_frac", "aromatic_clustering", "mean_beta_propensity",
        # axis 2: maturation grammar
        "full_mean_p_lock", "full_mean_p_pi", "full_mean_p_polar",
        "lcd_mean_p_lock", "lcd_lockability_per_pair",
        # axis 3: amyloid propensity
        "n_aprs", "apr_fraction", "max_apr_score", "mean_amyloid_propensity", "larks_count",
        # axis 4: ESM2
        "mean_llr", "emb_norm", "residue_emb_variance",
    ]
    # keep only columns that exist
    available = [c for c in feature_cols if c in df.columns]
    return available


def load_esm2_pca_features(n_components=10):
    """reduce ESM2 1280-dim embeddings to top PCA components."""
    emb = np.load(FEATURES / "esm2_embeddings.npy")  # (16, 1280)
    names = [n.strip() for n in open(FEATURES / "esm2_names.txt")]

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)

    pca = PCA(n_components=min(n_components, emb.shape[0]))
    emb_pca = pca.fit_transform(emb_scaled)

    # create dataframe with ESM2 PCA components
    cols = [f"esm2_pc{i+1}" for i in range(emb_pca.shape[1])]
    esm_df = pd.DataFrame(emb_pca, columns=cols)
    esm_df["name"] = names

    print(f"  ESM2 PCA: {pca.n_components_} components, "
          f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    for i, ev in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"    PC{i+1}: {ev:.3f}")

    return esm_df, cols


def plot_mechanism_landscape_pca(df, feature_cols, esm_pca_cols):
    """main PCA landscape plot with mutation arrows."""
    # combine interpretable features + ESM2 PCA components
    all_cols = feature_cols + esm_pca_cols
    X = df[all_cols].values.astype(float)

    # standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df["pc1"] = X_pca[:, 0]
    df["pc2"] = X_pca[:, 1]

    print(f"\n  landscape PCA: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")

    # feature loadings
    loadings = pd.DataFrame(
        pca.components_.T, index=all_cols, columns=["PC1", "PC2"]
    )
    loadings["abs_PC1"] = loadings["PC1"].abs()
    loadings["abs_PC2"] = loadings["PC2"].abs()
    print("\n  top PC1 features:")
    for _, row in loadings.nlargest(5, "abs_PC1").iterrows():
        print(f"    {row.name}: {row['PC1']:.3f}")
    print("  top PC2 features:")
    for _, row in loadings.nlargest(5, "abs_PC2").iterrows():
        print(f"    {row.name}: {row['PC2']:.3f}")

    # -- plot --
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # plot WT proteins (larger markers)
    wt_names = {"alpha_synuclein_ensemble", "ab42_wt", "iapp_mature_wt",
                "tau_2n4r_wt",
                "tdp43_full_wt", "fus_full_wt", "hnrnpa1_full_wt",
                "tia1_full_wt", "ewsr1_full_wt", "taf15_full_wt", "hnrnpa2b1_full_wt",
                "prp_mature_wt",
                "httex1_q21_wt", "ataxin3_q14_wt",
                "ddx4_full_wt", "npm1_full_wt"}

    for _, row in df.iterrows():
        mech = row["mechanism"]
        color = MECH_COLORS.get(mech, "gray")
        is_wt = row["name"] in wt_names
        size = 200 if is_wt else 100
        marker = "o" if is_wt else "^"
        edge = "black" if is_wt else "none"
        alpha = 1.0 if is_wt else 0.7

        ax.scatter(row["pc1"], row["pc2"], c=color, s=size, marker=marker,
                   edgecolors=edge, linewidths=1.5 if is_wt else 0, alpha=alpha, zorder=5)

        # label WT proteins
        if is_wt:
            ax.annotate(row["display_name"], (row["pc1"], row["pc2"]),
                       textcoords="offset points", xytext=(10, 8),
                       fontsize=10, fontweight="bold", color=color)

    # draw mutation arrows
    for wt_name, mut_name, label in WT_MUTANT_PAIRS:
        wt_row = df[df["name"] == wt_name].iloc[0]
        mut_row = df[df["name"] == mut_name].iloc[0]

        dx = mut_row["pc1"] - wt_row["pc1"]
        dy = mut_row["pc2"] - wt_row["pc2"]
        dist = np.sqrt(dx**2 + dy**2)

        if dist > 0.01:  # only draw if meaningful displacement
            ax.annotate("", xy=(mut_row["pc1"], mut_row["pc2"]),
                       xytext=(wt_row["pc1"], wt_row["pc2"]),
                       arrowprops=dict(arrowstyle="->", color="gray",
                                      lw=1.5, connectionstyle="arc3,rad=0.1"))
            # label the mutation
            mid_x = (wt_row["pc1"] + mut_row["pc1"]) / 2
            mid_y = (wt_row["pc2"] + mut_row["pc2"]) / 2
            ax.annotate(label, (mid_x, mid_y), fontsize=7, color="gray",
                       ha="center", va="bottom")

    # legend
    for mech, color in MECH_COLORS.items():
        label = mech.replace("_", " ")
        ax.scatter([], [], c=color, s=100, label=label)
    ax.scatter([], [], c="gray", s=200, marker="o", edgecolors="black",
               linewidths=1.5, label="WT (circles)")
    ax.scatter([], [], c="gray", s=100, marker="^", label="mutant (triangles)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title("IDP disease mechanism landscape\n(4-axis: polymer physics + maturation grammar + amyloid propensity + ESM2)",
                fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "mechanism_landscape_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  saved: {FIGURES / 'mechanism_landscape_pca.png'}")

    return pca, loadings


def plot_radar_chart(df):
    """spider/radar chart showing each protein's profile across 4 axes."""
    # select one representative feature per axis
    axis_features = {
        "charge\npatterning\n(κ)": "kappa",
        "maturation\ngrammar\n(p_lock)": "full_mean_p_lock",
        "amyloid\npropensity": "mean_amyloid_propensity",
        "ESM2\nconservation\n(-LLR)": "mean_llr",
        "aromatic\ncontent": "aromatic_frac",
        "Q/N\ncontent": "qn_frac",
    }

    # only WT proteins
    wt_names = ["alpha_synuclein_ensemble", "ab42_wt", "iapp_mature_wt",
                "tau_2n4r_wt",
                "tdp43_full_wt", "fus_full_wt", "hnrnpa1_full_wt",
                "tia1_full_wt", "ewsr1_full_wt", "taf15_full_wt", "hnrnpa2b1_full_wt",
                "prp_mature_wt",
                "httex1_q21_wt", "ataxin3_q14_wt",
                "ddx4_full_wt", "npm1_full_wt"]

    wt_df = df[df["name"].isin(wt_names)].copy()

    # standardize features for radar
    axis_names = list(axis_features.keys())
    axis_cols = list(axis_features.values())

    # flip LLR so higher = more conserved
    wt_df = wt_df.copy()
    wt_df["mean_llr"] = -wt_df["mean_llr"]  # negate so higher = more conserved

    scaler = StandardScaler()
    vals = scaler.fit_transform(wt_df[axis_cols].values)

    # normalize to 0-1 range for radar
    vals = (vals - vals.min(axis=0)) / (vals.max(axis=0) - vals.min(axis=0) + 1e-10)

    n_axes = len(axis_names)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, (_, row) in enumerate(wt_df.iterrows()):
        name = row["name"]
        mech = row["mechanism"]
        color = MECH_COLORS.get(mech, "gray")
        values = vals[i].tolist()
        values += values[:1]  # close
        ax.plot(angles, values, color=color, linewidth=2, label=DISPLAY.get(name, name))
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_names, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("IDP mechanism profiles (4 axes)", fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(FIGURES / "radar_mechanism_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {FIGURES / 'radar_mechanism_profiles.png'}")


def plot_mutation_effects(df):
    """bar chart showing feature changes for each mutation."""
    features_to_compare = [
        ("kappa", "charge patterning (κ)"),
        ("full_mean_p_lock", "maturation grammar (p_lock)"),
        ("mean_amyloid_propensity", "amyloid propensity"),
        ("mean_llr", "ESM2 log-likelihood"),
        ("aromatic_frac", "aromatic content"),
        ("qn_frac", "Q/N content"),
    ]

    n_mut = len(WT_MUTANT_PAIRS)
    n_feat = len(features_to_compare)

    fig, axes = plt.subplots(n_feat, 1, figsize=(12, 3 * n_feat), sharex=True)

    for fi, (feat, feat_label) in enumerate(features_to_compare):
        ax = axes[fi]
        labels = []
        deltas = []
        colors = []

        for wt_name, mut_name, mut_label in WT_MUTANT_PAIRS:
            wt_val = df[df["name"] == wt_name][feat].values[0]
            mut_val = df[df["name"] == mut_name][feat].values[0]
            delta = mut_val - wt_val

            protein = DISPLAY.get(wt_name, wt_name).split()[0] if " " in DISPLAY.get(wt_name, wt_name) else DISPLAY.get(wt_name, wt_name)
            labels.append(f"{protein}\n{mut_label}")
            deltas.append(delta)
            colors.append(MECH_COLORS.get(MECHANISM.get(wt_name, ""), "gray"))

        bars = ax.bar(range(n_mut), deltas, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(f"Δ {feat_label}", fontsize=9)
        ax.set_xticks(range(n_mut))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_title("mutation effects across 4 feature axes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "mutation_effects.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {FIGURES / 'mutation_effects.png'}")


def plot_axis_separation(df):
    """2D scatter plots showing how pairs of axes separate mechanisms."""
    pairs = [
        ("mean_amyloid_propensity", "full_mean_p_lock", "amyloid propensity", "maturation grammar (p_lock)"),
        ("kappa", "full_mean_p_lock", "charge patterning (κ)", "maturation grammar (p_lock)"),
        ("aromatic_frac", "qn_frac", "aromatic fraction", "Q/N fraction"),
        ("scd", "mean_amyloid_propensity", "SCD", "amyloid propensity"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    wt_names = {"alpha_synuclein_ensemble", "ab42_wt", "iapp_mature_wt",
                "tau_2n4r_wt",
                "tdp43_full_wt", "fus_full_wt", "hnrnpa1_full_wt",
                "tia1_full_wt", "ewsr1_full_wt", "taf15_full_wt", "hnrnpa2b1_full_wt",
                "prp_mature_wt",
                "httex1_q21_wt", "ataxin3_q14_wt",
                "ddx4_full_wt", "npm1_full_wt"}

    for idx, (xf, yf, xlabel, ylabel) in enumerate(pairs):
        ax = axes[idx // 2, idx % 2]

        for _, row in df.iterrows():
            mech = row["mechanism"]
            color = MECH_COLORS.get(mech, "gray")
            is_wt = row["name"] in wt_names
            size = 150 if is_wt else 60
            marker = "o" if is_wt else "^"
            edge = "black" if is_wt else "none"

            ax.scatter(row[xf], row[yf], c=color, s=size, marker=marker,
                      edgecolors=edge, linewidths=1 if is_wt else 0, zorder=5)

            if is_wt:
                ax.annotate(DISPLAY.get(row["name"], row["name"]),
                           (row[xf], row[yf]),
                           textcoords="offset points", xytext=(8, 5),
                           fontsize=8, color=color)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)

    # shared legend
    for mech, color in MECH_COLORS.items():
        axes[0, 0].scatter([], [], c=color, s=100, label=mech.replace("_", " "))
    axes[0, 0].legend(fontsize=8, loc="best")

    fig.suptitle("mechanism separation across feature pairs", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "axis_separation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {FIGURES / 'axis_separation.png'}")


def compute_mechanism_distances(df, feature_cols, esm_pca_cols):
    """compute inter-mechanism distances in feature space."""
    all_cols = feature_cols + esm_pca_cols
    mechanisms = df["mechanism"].unique()

    # standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df[all_cols].values.astype(float))

    print("\n  inter-mechanism distances (euclidean in standardized feature space):")
    print(f"  {'':>25}", end="")
    for m in sorted(mechanisms):
        print(f" {m[:12]:>13}", end="")
    print()

    for m1 in sorted(mechanisms):
        print(f"  {m1:>25}", end="")
        idx1 = df["mechanism"] == m1
        centroid1 = X[idx1].mean(axis=0)
        for m2 in sorted(mechanisms):
            idx2 = df["mechanism"] == m2
            centroid2 = X[idx2].mean(axis=0)
            dist = np.linalg.norm(centroid1 - centroid2)
            print(f" {dist:>13.2f}", end="")
        print()


def main():
    print("=" * 60)
    print("IDP mechanism landscape integration")
    print("=" * 60)

    # load all features
    print("\nloading features...")
    df = load_all_features()
    print(f"  merged dataframe: {df.shape[0]} proteins × {df.shape[1]} features")

    # ESM2 PCA reduction
    print("\nreducing ESM2 embeddings...")
    esm_pca_df, esm_pca_cols = load_esm2_pca_features(n_components=10)
    df = df.merge(esm_pca_df, on="name")

    # select features
    feature_cols = select_features_for_landscape(df)
    print(f"\n  selected {len(feature_cols)} interpretable features + {len(esm_pca_cols)} ESM2 PCA components")

    # save combined feature matrix
    out_path = FEATURES / "combined_features.csv"
    df.to_csv(out_path, index=False)
    print(f"  saved combined features to {out_path}")

    # --- plots ---
    print("\ngenerating visualizations...")

    # 1. main PCA landscape
    pca, loadings = plot_mechanism_landscape_pca(df, feature_cols, esm_pca_cols)

    # 2. radar chart
    plot_radar_chart(df)

    # 3. mutation effects
    plot_mutation_effects(df)

    # 4. axis separation plots
    plot_axis_separation(df)

    # 5. inter-mechanism distances
    compute_mechanism_distances(df, feature_cols, esm_pca_cols)

    # save loadings
    loadings.to_csv(FEATURES / "pca_loadings.csv")

    print("\n" + "=" * 60)
    print("DONE — all figures saved to", FIGURES)
    print("=" * 60)


if __name__ == "__main__":
    main()
