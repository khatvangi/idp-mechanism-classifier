#!/usr/bin/env python3
"""
approach B: IDR vs structured stratified analysis.
trains separate XGBoost models for IDR-only and structured-only subsets.
compares feature importances between the two contexts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
VARIANTS = DATA / "variants"
FIGURES = PROJECT / "figures"

# same features as approach A, minus protein_length and total_disorder_frac
# (gene-level confounds in LOGO-CV)
NUMERIC_FEATURES = [
    "local_hydrophobicity", "local_charge_density", "local_aromatic_density",
    "local_glycine_frac", "local_proline_frac", "local_beta_propensity",
    "local_qn_frac", "local_disorder", "position_disorder",
    "relative_position", "local_p_lock", "local_p_pi", "local_p_polar",
    "is_sticker", "blosum62", "grantham_distance", "hydrophobicity_change",
    "size_change", "creates_proline", "destroys_proline", "creates_glycine",
]
CATEGORICAL_FEATURES = ["charge_change", "aromatic_change", "polarity_change"]


def prepare_data(df):
    """prepare features and binary targets. excludes gene-level confounds."""
    df = df[df["label"] != "conflicting"].copy()
    df["target"] = (df["label"] == "pathogenic").astype(int)
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

    feature_cols = NUMERIC_FEATURES.copy()
    for cat in CATEGORICAL_FEATURES:
        cat_cols = [c for c in df_encoded.columns if c.startswith(cat + "_")]
        feature_cols.extend(cat_cols)

    return df_encoded, feature_cols


def logo_cv_subset(df_encoded, feature_cols, subset_mask, subset_name):
    """run LOGO-CV on a subset (IDR or structured)."""
    df_sub = df_encoded[subset_mask].copy()
    X = df_sub[feature_cols].values.astype(np.float32)
    y = df_sub["target"].values
    genes = df_sub["gene"].values

    unique_genes = sorted(set(genes))
    print(f"\n  {subset_name}: {len(X)} samples, {y.sum()} pathogenic, "
          f"{(1-y).sum():.0f} benign/VUS, {len(unique_genes)} genes")

    all_preds = np.zeros(len(y))
    all_true = y.copy()
    per_gene = []

    for fold_gene in unique_genes:
        test_mask = genes == fold_gene
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        n_test = len(y_test)
        n_pos = y_test.sum()
        n_neg = n_test - n_pos

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

        if n_pos > 0 and n_neg > 0:
            gene_auc = roc_auc_score(y_test, y_prob)
            per_gene.append({"gene": fold_gene, "auroc": gene_auc,
                             "n": n_test, "n_path": int(n_pos)})
            print(f"    {fold_gene:<12} n={n_test:>4} P={n_pos:>3}  AUROC={gene_auc:.3f}")

    # overall
    # only evaluate on samples that were predicted (exclude genes with too few)
    valid = all_preds > 0  # crude check: genes that were actually predicted
    # more robust: check if gene was included in any fold
    evaluated_genes = {r["gene"] for r in per_gene}
    valid = np.array([g in evaluated_genes for g in genes])
    y_eval = all_true[valid]
    p_eval = all_preds[valid]

    if y_eval.sum() > 0 and (1 - y_eval).sum() > 0:
        auc = roc_auc_score(y_eval, p_eval)
        ap = average_precision_score(y_eval, p_eval)
        print(f"\n  {subset_name} overall: AUROC={auc:.4f}, AUPRC={ap:.4f}")
    else:
        auc, ap = np.nan, np.nan
        print(f"\n  {subset_name}: cannot compute overall metrics (single class)")

    return auc, ap, per_gene, all_preds, all_true


def feature_importance_subset(df_encoded, feature_cols, subset_mask, subset_name):
    """train full model on subset, extract importances."""
    df_sub = df_encoded[subset_mask]
    X = df_sub[feature_cols].values.astype(np.float32)
    y = df_sub["target"].values

    if y.sum() < 2 or (1 - y).sum() < 2:
        return None

    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        use_label_encoder=False, random_state=42, verbosity=0,
    )
    model.fit(X, y)

    return pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)


def plot_stratified(idr_imp, struct_imp):
    """compare feature importances between IDR and structured models."""
    if idr_imp is None or struct_imp is None:
        print("  cannot plot — missing importance data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # IDR importance
    top = idr_imp.head(12)
    axes[0].barh(top["feature"][::-1], top["importance"][::-1], color="#d73027")
    axes[0].set_title("IDR Variants — Feature Importance")
    axes[0].set_xlabel("Importance (gain)")

    # structured importance
    top = struct_imp.head(12)
    axes[1].barh(top["feature"][::-1], top["importance"][::-1], color="#4575b4")
    axes[1].set_title("Structured Variants — Feature Importance")
    axes[1].set_xlabel("Importance (gain)")

    plt.tight_layout()
    out = FIGURES / "approach_b_stratified.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved figure to {out}")


def main():
    print("=" * 60)
    print("approach B: IDR vs structured stratified analysis")
    print("=" * 60)

    df = pd.read_csv(VARIANTS / "feature_matrix.csv")
    print(f"  loaded {len(df)} variants")

    df_encoded, feature_cols = prepare_data(df)

    idr_mask = df_encoded["in_idr"] == 1
    struct_mask = df_encoded["in_idr"] == 0

    print(f"\n  IDR variants: {idr_mask.sum()}")
    print(f"  structured variants: {struct_mask.sum()}")

    # run LOGO-CV on each subset
    idr_auc, idr_ap, idr_genes, _, _ = logo_cv_subset(
        df_encoded, feature_cols, idr_mask, "IDR")
    struct_auc, struct_ap, struct_genes, _, _ = logo_cv_subset(
        df_encoded, feature_cols, struct_mask, "Structured")

    # also run LOGO-CV on full data without gene-level confounds
    # (approach A used protein_length; this version removes it)
    print("\n" + "=" * 60)
    print("  approach A' (no gene-level confounds):")
    full_mask = pd.Series(True, index=df_encoded.index)
    full_auc, full_ap, full_genes, _, _ = logo_cv_subset(
        df_encoded, feature_cols, full_mask, "Full (no confounds)")

    # feature importance comparison
    print("\n  computing feature importances per subset...")
    idr_imp = feature_importance_subset(df_encoded, feature_cols, idr_mask, "IDR")
    struct_imp = feature_importance_subset(df_encoded, feature_cols, struct_mask, "Structured")

    if idr_imp is not None:
        print(f"\n  IDR top 5 features:")
        for _, r in idr_imp.head(5).iterrows():
            print(f"    {r['feature']:<35} {r['importance']:.4f}")

    if struct_imp is not None:
        print(f"\n  Structured top 5 features:")
        for _, r in struct_imp.head(5).iterrows():
            print(f"    {r['feature']:<35} {r['importance']:.4f}")

    # summary
    print(f"\n  === SUMMARY ===")
    print(f"  Full (no confounds):  AUROC={full_auc:.4f}" if not np.isnan(full_auc) else "  Full: N/A")
    print(f"  IDR-only:            AUROC={idr_auc:.4f}" if not np.isnan(idr_auc) else "  IDR: N/A")
    print(f"  Structured-only:     AUROC={struct_auc:.4f}" if not np.isnan(struct_auc) else "  Structured: N/A")

    # save
    results = {
        "full_auroc": full_auc, "full_auprc": full_ap,
        "idr_auroc": idr_auc, "idr_auprc": idr_ap,
        "struct_auroc": struct_auc, "struct_auprc": struct_ap,
    }
    pd.DataFrame([results]).to_csv(VARIANTS / "results_approach_b.csv", index=False)

    plot_stratified(idr_imp, struct_imp)


if __name__ == "__main__":
    main()
