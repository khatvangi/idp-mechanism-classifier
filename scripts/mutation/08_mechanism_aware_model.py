#!/usr/bin/env python3
"""
mechanism-aware pathogenicity prediction.
tests whether IDP-specific features can rescue prediction for
gain-of-toxic-function (GoF) genes where ESM2 conservation fails.

key hypothesis: GoF genes (FUS, TARDBP, HNRNPA1, TIA1) need different
features than LoF genes (LMNA, SOD1, CRYAB, VCP).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
VARIANTS = DATA / "variants"
FIGURES = PROJECT / "figures"

# gene mechanism groupings
GOF_NONAMYLOID = {"FUS", "TARDBP", "HNRNPA1", "TIA1", "HNRNPA2B1", "EWSR1", "TAF15"}
GOF_AMYLOID = {"SNCA", "TTR", "PRNP", "IAPP"}
LOF_STRUCTURED = {"SOD1", "VCP", "LMNA", "CRYAB"}
REPEAT = {"HTT", "ATXN3", "AR"}
CONDENSATE = {"DDX4", "NPM1", "SQSTM1", "MAPT"}

# IDP-specific features (no ESM2, no gene-level confounds)
IDP_FEATURES = [
    "local_hydrophobicity", "local_charge_density", "local_aromatic_density",
    "local_glycine_frac", "local_proline_frac", "local_beta_propensity",
    "local_qn_frac", "local_disorder", "position_disorder", "in_idr",
    "relative_position", "local_p_lock", "local_p_pi", "local_p_polar",
    "is_sticker", "blosum62", "grantham_distance", "hydrophobicity_change",
    "size_change", "creates_proline", "destroys_proline", "creates_glycine",
]

# ESM2-only features
ESM2_FEATURES = [
    "esm2_llr", "esm2_ref_logprob", "esm2_alt_logprob",
    "esm2_entropy", "esm2_rank_ref", "esm2_rank_alt",
]

# GoF-specialized features: mutation-specific properties that matter for
# gain-of-toxic-function (aggregation, phase separation disruption)
GOF_SPECIALIZED = [
    # these capture changes in sticker/spacer character
    "hydrophobicity_change",    # gaining hydrophobicity → more sticker-like
    "local_aromatic_density",   # aromatics are key stickers
    "local_qn_frac",           # Q/N are polar stickers
    "is_sticker",              # mutation at a sticker position?
    "aromatic_change",         # loss/gain of aromatic character
    "creates_proline",         # proline kinks disrupt helices, alter aggregation
    "destroys_proline",        # losing proline changes backbone flexibility
    "local_p_lock",            # grammar locking probability at this position
    "local_p_pi",              # aromatic channel interaction
    "local_p_polar",           # polar channel interaction
    "local_charge_density",    # charge context (affects phase separation)
    "charge_change",           # charge gain/loss/flip
    "grantham_distance",       # physico-chemical severity
    "blosum62",                # evolutionary substitution score
    "position_disorder",       # how disordered is the position
    "relative_position",       # where in the protein (NLS is C-terminal for FUS)
]

CATEGORICAL_FEATURES = ["charge_change", "aromatic_change", "polarity_change"]


def load_data():
    """load ESM2 features + handcrafted features."""
    df = pd.read_csv(VARIANTS / "esm2_features.csv")
    df = df[df["label"] != "conflicting"].copy()
    df["target"] = (df["label"] == "pathogenic").astype(int)

    # assign mechanism group
    def get_mechanism(gene):
        if gene in GOF_NONAMYLOID:
            return "gof_nonamyloid"
        elif gene in GOF_AMYLOID:
            return "gof_amyloid"
        elif gene in LOF_STRUCTURED:
            return "lof_structured"
        elif gene in REPEAT:
            return "repeat"
        elif gene in CONDENSATE:
            return "condensate"
        return "other"

    df["mechanism"] = df["gene"].apply(get_mechanism)
    return df


def logo_cv_subset(df, feature_cols, subset_name):
    """LOGO-CV on a subset of the data."""
    genes = df["gene"].values
    y = df["target"].values
    unique_genes = sorted(set(genes))

    # one-hot encode categoricals if present
    cat_present = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if cat_present:
        df_enc = pd.get_dummies(df, columns=cat_present, drop_first=True)
        # add one-hot columns to feature list
        extra = []
        for cat in cat_present:
            extra.extend([c for c in df_enc.columns if c.startswith(cat + "_")])
        feature_cols = [f for f in feature_cols if f not in cat_present] + extra
    else:
        df_enc = df

    # ensure all requested features exist
    available = [f for f in feature_cols if f in df_enc.columns]
    X = df_enc[available].values.astype(np.float32)

    all_preds = np.full(len(y), np.nan)
    per_gene = []

    for fold_gene in unique_genes:
        test_mask = genes == fold_gene
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        n_test, n_pos = len(y_test), int(y_test.sum())
        if n_test < 3 or y_train.sum() < 2:
            continue

        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos, eval_metric="logloss",
            use_label_encoder=False, random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        all_preds[test_mask] = y_prob

        if n_pos > 0 and n_pos < n_test:
            per_gene.append({"gene": fold_gene, "auroc": roc_auc_score(y_test, y_prob),
                             "n": n_test, "n_path": n_pos})

    # overall metrics on predicted samples
    valid = ~np.isnan(all_preds)
    y_valid = y[valid]
    p_valid = all_preds[valid]

    if y_valid.sum() > 0 and (1 - y_valid).sum() > 0:
        auc = roc_auc_score(y_valid, p_valid)
        ap = average_precision_score(y_valid, p_valid)
    else:
        auc, ap = np.nan, np.nan

    return auc, ap, per_gene, all_preds


def test_single_features_gof(df_gof):
    """test individual feature discriminative power for GoF genes.
    uses simple AUROC of the raw feature value (no ML).
    """
    print(f"\n  single-feature AUROC for GoF non-amyloid genes:")
    print(f"  n={len(df_gof)}, P={(df_gof['target']==1).sum()}, "
          f"B={(df_gof['target']==0).sum()}")

    if df_gof["target"].sum() == 0 or (1 - df_gof["target"]).sum() == 0:
        print("  cannot test — single class")
        return []

    results = []
    features_to_test = [
        "esm2_llr", "esm2_entropy", "esm2_ref_logprob",
        "hydrophobicity_change", "local_charge_density", "local_aromatic_density",
        "local_qn_frac", "grantham_distance", "blosum62",
        "local_p_lock", "local_p_pi", "local_p_polar",
        "position_disorder", "creates_proline", "destroys_proline",
        "is_sticker", "relative_position", "size_change",
        "local_hydrophobicity", "local_proline_frac", "local_glycine_frac",
        "local_beta_propensity",
    ]

    for feat in features_to_test:
        if feat not in df_gof.columns:
            continue
        vals = df_gof[feat].values
        y = df_gof["target"].values

        # handle NaN
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            continue

        # AUROC: does higher value = more pathogenic?
        try:
            auc = roc_auc_score(y[valid], vals[valid])
        except:
            continue

        # effect size (Cohen's d)
        path_vals = vals[y == 1]
        ben_vals = vals[y == 0]
        pooled_std = np.sqrt(((len(path_vals) - 1) * path_vals.std()**2 +
                               (len(ben_vals) - 1) * ben_vals.std()**2) /
                              (len(path_vals) + len(ben_vals) - 2))
        d = (path_vals.mean() - ben_vals.mean()) / pooled_std if pooled_std > 0 else 0

        # Mann-Whitney U test
        try:
            u_stat, p_val = stats.mannwhitneyu(path_vals, ben_vals, alternative="two-sided")
        except:
            p_val = 1.0

        results.append({
            "feature": feat, "auroc": auc, "cohens_d": d, "p_value": p_val,
            "mean_path": path_vals.mean(), "mean_ben": ben_vals.mean(),
        })

    results.sort(key=lambda x: abs(x["auroc"] - 0.5), reverse=True)
    print(f"\n  {'Feature':<30} {'AUROC':>7} {'d':>7} {'p':>10} "
          f"{'path':>8} {'ben':>8}")
    print(f"  {'-'*78}")
    for r in results:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"  {r['feature']:<30} {r['auroc']:>7.3f} {r['cohens_d']:>+7.2f} "
              f"{r['p_value']:>10.4f} {r['mean_path']:>8.2f} {r['mean_ben']:>8.2f} {sig}")

    return results


def mechanism_aware_ensemble(df):
    """test a mechanism-aware approach:
    - for LoF genes: use ESM2 LLR (works well)
    - for GoF genes: use best-performing features
    - compare to ESM2 alone
    """
    print("\n" + "=" * 70)
    print("MECHANISM-AWARE ENSEMBLE")
    print("=" * 70)

    genes = df["gene"].values
    y = df["target"].values

    # baseline: ESM2 LLR alone on everything
    llr = df["esm2_llr"].values
    valid = ~np.isnan(llr)
    baseline_auc = roc_auc_score(y[valid], llr[valid])
    print(f"\n  baseline ESM2 LLR (all genes): AUROC={baseline_auc:.4f}")

    # strategy: use ESM2 LLR for LoF/amyloid, IDP features for GoF
    # "oracle" mechanism labeling (we know the mechanism)
    ensemble_preds = np.full(len(y), np.nan)

    # LoF + amyloid + repeat: just use ESM2 LLR (scaled to [0,1])
    lof_mask = df["mechanism"].isin(["lof_structured", "gof_amyloid", "repeat", "condensate"])
    llr_vals = df.loc[lof_mask, "esm2_llr"].values
    # scale LLR to pseudo-probability
    llr_min, llr_max = llr_vals.min(), llr_vals.max()
    if llr_max > llr_min:
        ensemble_preds[lof_mask.values] = (llr_vals - llr_min) / (llr_max - llr_min)

    # GoF non-amyloid: train XGBoost with IDP + ESM2 combined features
    # using LOGO-CV within this subset
    gof_mask = df["mechanism"] == "gof_nonamyloid"
    df_gof = df[gof_mask].copy()

    if len(df_gof) > 0 and df_gof["target"].sum() > 0:
        all_features = IDP_FEATURES + ESM2_FEATURES
        cat_present = [c for c in CATEGORICAL_FEATURES if c in df_gof.columns]
        if cat_present:
            df_gof_enc = pd.get_dummies(df_gof, columns=cat_present, drop_first=True)
            extra = []
            for cat in cat_present:
                extra.extend([c for c in df_gof_enc.columns if c.startswith(cat + "_")])
            all_features = [f for f in all_features if f not in cat_present] + extra
        else:
            df_gof_enc = df_gof

        available = [f for f in all_features if f in df_gof_enc.columns]
        X_gof = df_gof_enc[available].values.astype(np.float32)
        y_gof = df_gof["target"].values
        genes_gof = df_gof["gene"].values

        gof_preds = np.full(len(y_gof), np.nan)
        for fold_gene in sorted(set(genes_gof)):
            test_mask = genes_gof == fold_gene
            train_mask = ~test_mask
            if test_mask.sum() < 3 or y_gof[train_mask].sum() < 2:
                continue

            scale_pos = (y_gof[train_mask] == 0).sum() / max((y_gof[train_mask] == 1).sum(), 1)
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                scale_pos_weight=scale_pos, eval_metric="logloss",
                use_label_encoder=False, random_state=42, verbosity=0,
            )
            model.fit(X_gof[train_mask], y_gof[train_mask])
            gof_preds[test_mask] = model.predict_proba(X_gof[test_mask])[:, 1]

        # fill into ensemble
        gof_indices = np.where(gof_mask.values)[0]
        for i, idx in enumerate(gof_indices):
            if not np.isnan(gof_preds[i]):
                ensemble_preds[idx] = gof_preds[i]

    # evaluate ensemble
    valid = ~np.isnan(ensemble_preds)
    y_valid = y[valid]
    p_valid = ensemble_preds[valid]

    if y_valid.sum() > 0 and (1 - y_valid).sum() > 0:
        ens_auc = roc_auc_score(y_valid, p_valid)
        ens_ap = average_precision_score(y_valid, p_valid)
        print(f"  mechanism-aware ensemble: AUROC={ens_auc:.4f}, AUPRC={ens_ap:.4f}")
    else:
        ens_auc = np.nan

    # per-mechanism breakdown
    for mech_name in ["lof_structured", "gof_amyloid", "gof_nonamyloid", "repeat", "condensate"]:
        mask = (df["mechanism"] == mech_name).values & valid
        if mask.sum() == 0:
            continue
        y_m = y[mask]
        p_m = ensemble_preds[mask]
        if y_m.sum() > 0 and (1 - y_m).sum() > 0:
            auc = roc_auc_score(y_m, p_m)
            print(f"    {mech_name:<20} AUROC={auc:.3f} (n={mask.sum()}, P={y_m.sum()})")

    return ens_auc, baseline_auc


def test_gof_per_gene_features(df):
    """for each GoF gene with pathogenic variants, test which individual
    features have the best discrimination. this reveals what's TRULY
    different about pathogenic mutations in these genes."""
    print("\n" + "=" * 70)
    print("PER-GENE FEATURE ANALYSIS (GoF genes)")
    print("=" * 70)

    for gene in sorted(GOF_NONAMYLOID):
        gdf = df[df["gene"] == gene]
        n_path = (gdf["target"] == 1).sum()
        n_ben = (gdf["target"] == 0).sum()

        if n_path < 2 or n_ben < 2:
            continue

        print(f"\n  {gene} (n={len(gdf)}, P={n_path}, B={n_ben}):")

        features = [
            "esm2_llr", "esm2_entropy", "hydrophobicity_change",
            "local_charge_density", "local_qn_frac", "grantham_distance",
            "blosum62", "local_p_lock", "relative_position",
            "position_disorder", "creates_proline", "destroys_proline",
            "is_sticker", "size_change", "local_aromatic_density",
        ]

        for feat in features:
            if feat not in gdf.columns:
                continue
            try:
                auc = roc_auc_score(gdf["target"], gdf[feat])
                path_mean = gdf[gdf["target"] == 1][feat].mean()
                ben_mean = gdf[gdf["target"] == 0][feat].mean()
                marker = " <<<" if abs(auc - 0.5) > 0.15 else ""
                print(f"    {feat:<30} AUROC={auc:.3f}  P={path_mean:>7.2f}  B={ben_mean:>7.2f}{marker}")
            except:
                pass


def test_positional_features_fus_tardbp(df):
    """FUS and TARDBP have a strong positional signal — pathogenic mutations
    cluster in specific regions (NLS for FUS, LCD for TARDBP).
    test whether position-based features can capture this."""
    print("\n" + "=" * 70)
    print("POSITIONAL CLUSTERING ANALYSIS")
    print("=" * 70)

    # FUS: pathogenic mutations cluster in NLS (501-526)
    fus = df[df["gene"] == "FUS"].copy()
    if fus["target"].sum() > 0:
        fus["in_nls"] = ((fus["position"] >= 501) & (fus["position"] <= 526)).astype(int)
        fus["in_lcd"] = (fus["position"] <= 165).astype(int)
        fus["in_rgg"] = (((fus["position"] >= 165) & (fus["position"] <= 267)) |
                          ((fus["position"] >= 371) & (fus["position"] <= 422)) |
                          ((fus["position"] >= 453) & (fus["position"] <= 501))).astype(int)

        print(f"\n  FUS pathogenic variant positions:")
        fus_path = fus[fus["target"] == 1].sort_values("position")
        for _, r in fus_path.iterrows():
            region = "NLS" if r["in_nls"] else "LCD" if r["in_lcd"] else "RGG" if r["in_rgg"] else "other"
            print(f"    pos {r['position']:>3} {r['ref_aa']}→{r['alt_aa']}  "
                  f"LLR={r['esm2_llr']:>5.2f}  disorder={r['position_disorder']:.2f}  region={region}")

        # can NLS membership predict pathogenicity?
        if fus["target"].sum() > 0 and (1 - fus["target"]).sum() > 0:
            auc_nls = roc_auc_score(fus["target"], fus["in_nls"])
            print(f"\n    NLS membership → pathogenicity AUROC: {auc_nls:.3f}")
            print(f"    (pathogenic in NLS: {fus[fus['target']==1]['in_nls'].sum()}/{fus['target'].sum()}, "
                  f"all in NLS: {fus['in_nls'].sum()}/{len(fus)})")

    # TARDBP: pathogenic mutations cluster in LCD (274-414)
    tdp = df[df["gene"] == "TARDBP"].copy()
    if tdp["target"].sum() > 0:
        tdp["in_lcd"] = ((tdp["position"] >= 274) & (tdp["position"] <= 414)).astype(int)
        tdp["in_rrm"] = (((tdp["position"] >= 104) & (tdp["position"] <= 176)) |
                          ((tdp["position"] >= 191) & (tdp["position"] <= 262))).astype(int)

        print(f"\n  TARDBP pathogenic variant positions:")
        tdp_path = tdp[tdp["target"] == 1].sort_values("position")
        for _, r in tdp_path.iterrows():
            region = "LCD" if r["in_lcd"] else "RRM" if r["in_rrm"] else "NTD"
            print(f"    pos {r['position']:>3} {r['ref_aa']}→{r['alt_aa']}  "
                  f"LLR={r['esm2_llr']:>5.2f}  disorder={r['position_disorder']:.2f}  region={region}")

        if tdp["target"].sum() > 0 and (1 - tdp["target"]).sum() > 0:
            auc_lcd = roc_auc_score(tdp["target"], tdp["in_lcd"])
            print(f"\n    LCD membership → pathogenicity AUROC: {auc_lcd:.3f}")
            print(f"    (pathogenic in LCD: {tdp[tdp['target']==1]['in_lcd'].sum()}/{tdp['target'].sum()}, "
                  f"all in LCD: {tdp['in_lcd'].sum()}/{len(tdp)})")


def plot_mechanism_comparison(df):
    """plot comparing ESM2 performance across mechanism groups."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    mechanisms = {
        "Loss of Function\n(LMNA, SOD1, CRYAB, VCP)": df[df["mechanism"] == "lof_structured"],
        "Toxic Aggregation (amyloid)\n(SNCA, TTR, PRNP)": df[df["mechanism"] == "gof_amyloid"],
        "Toxic Aggregation (non-amyloid)\n(FUS, TARDBP, HNRNPA1, TIA1)": df[df["mechanism"] == "gof_nonamyloid"],
        "Repeat Expansion\n(AR, HTT, ATXN3)": df[df["mechanism"] == "repeat"],
    }

    colors_path = "#d73027"
    colors_ben = "#4575b4"

    for idx, (mech_name, mdf) in enumerate(mechanisms.items()):
        ax = axes[idx // 2, idx % 2]

        if len(mdf) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        path = mdf[mdf["target"] == 1]["esm2_llr"]
        ben = mdf[mdf["target"] == 0]["esm2_llr"]

        # histogram
        bins = np.linspace(-2, 18, 40)
        if len(ben) > 0:
            ax.hist(ben, bins=bins, alpha=0.5, color=colors_ben,
                     label=f"benign/VUS (n={len(ben)})", density=True)
        if len(path) > 0:
            ax.hist(path, bins=bins, alpha=0.5, color=colors_path,
                     label=f"pathogenic (n={len(path)})", density=True)

        # AUROC
        if len(path) > 0 and len(ben) > 0:
            auc = roc_auc_score(mdf["target"], mdf["esm2_llr"])
            ax.text(0.95, 0.95, f"AUROC={auc:.3f}", transform=ax.transAxes,
                     ha="right", va="top", fontsize=12, fontweight="bold",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        ax.set_xlabel("ESM2 LLR")
        ax.set_ylabel("Density")
        ax.set_title(mech_name)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGURES / "mechanism_aware_llr_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def main():
    print("=" * 70)
    print("MECHANISM-AWARE PATHOGENICITY PREDICTION")
    print("=" * 70)

    df = load_data()
    print(f"  loaded {len(df)} variants")
    print(f"  mechanism distribution:")
    for mech, count in df["mechanism"].value_counts().items():
        n_path = (df[df["mechanism"] == mech]["target"] == 1).sum()
        print(f"    {mech:<20} n={count:>4}, P={n_path:>3}")

    # === TEST 1: single feature discriminative power in GoF genes ===
    print("\n" + "=" * 70)
    print("TEST 1: SINGLE-FEATURE DISCRIMINATION IN GoF NON-AMYLOID GENES")
    print("=" * 70)

    df_gof = df[df["mechanism"] == "gof_nonamyloid"]
    gof_results = test_single_features_gof(df_gof)

    # === TEST 2: per-gene feature analysis ===
    test_gof_per_gene_features(df)

    # === TEST 3: positional clustering in FUS/TARDBP ===
    test_positional_features_fus_tardbp(df)

    # === TEST 4: mechanism-specific LOGO-CV ===
    print("\n" + "=" * 70)
    print("TEST 4: MECHANISM-SPECIFIC LOGO-CV")
    print("=" * 70)

    for mech_name, mech_genes in [("LoF structured", LOF_STRUCTURED),
                                    ("GoF amyloid", GOF_AMYLOID),
                                    ("GoF non-amyloid", GOF_NONAMYLOID)]:
        df_mech = df[df["gene"].isin(mech_genes)]
        if df_mech["target"].sum() < 2:
            print(f"\n  {mech_name}: insufficient pathogenic variants")
            continue

        print(f"\n  {mech_name} ({len(df_mech)} variants, "
              f"{df_mech['target'].sum()} pathogenic):")

        # ESM2 only
        auc_e, _, _, _ = logo_cv_subset(df_mech, ESM2_FEATURES, f"{mech_name}_esm2")
        print(f"    ESM2 features LOGO-CV:     AUROC={auc_e:.4f}" if not np.isnan(auc_e) else "")

        # IDP only
        auc_i, _, _, _ = logo_cv_subset(df_mech, IDP_FEATURES, f"{mech_name}_idp")
        print(f"    IDP features LOGO-CV:      AUROC={auc_i:.4f}" if not np.isnan(auc_i) else "")

        # combined
        auc_c, _, _, _ = logo_cv_subset(df_mech, IDP_FEATURES + ESM2_FEATURES,
                                         f"{mech_name}_combined")
        print(f"    Combined LOGO-CV:          AUROC={auc_c:.4f}" if not np.isnan(auc_c) else "")

        # raw ESM2 LLR (no ML)
        if df_mech["target"].sum() > 0 and (1 - df_mech["target"]).sum() > 0:
            auc_raw = roc_auc_score(df_mech["target"], df_mech["esm2_llr"])
            print(f"    ESM2 LLR raw:              AUROC={auc_raw:.4f}")

    # === TEST 5: mechanism-aware ensemble ===
    ens_auc, base_auc = mechanism_aware_ensemble(df)

    # === PLOTS ===
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_mechanism_comparison(df)

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY: CAN WE RESCUE GoF PREDICTION?")
    print("=" * 70)


if __name__ == "__main__":
    main()
