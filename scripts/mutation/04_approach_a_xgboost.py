#!/usr/bin/env python3
"""
approach A: XGBoost classifier on handcrafted IDP features.
uses leave-one-gene-out cross-validation (LOGO-CV).
binary classification: pathogenic vs benign (VUS treated as noisy-benign).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
VARIANTS = DATA / "variants"
FIGURES = PROJECT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# numeric feature columns (from 03_disorder_and_features.py)
NUMERIC_FEATURES = [
    "local_hydrophobicity", "local_charge_density", "local_aromatic_density",
    "local_glycine_frac", "local_proline_frac", "local_beta_propensity",
    "local_qn_frac", "local_disorder", "position_disorder", "in_idr",
    "relative_position", "local_p_lock", "local_p_pi", "local_p_polar",
    "is_sticker", "blosum62", "grantham_distance", "hydrophobicity_change",
    "size_change", "creates_proline", "destroys_proline", "creates_glycine",
    "protein_length", "total_disorder_frac",
]

# categorical features to one-hot encode
CATEGORICAL_FEATURES = ["charge_change", "aromatic_change", "polarity_change"]


def prepare_data(df):
    """prepare feature matrix and binary labels.

    label mapping:
      pathogenic → 1
      benign, likely_benign, vus → 0 (VUS as noisy-benign proxy)
      conflicting → excluded
    """
    # filter out conflicting
    df = df[df["label"] != "conflicting"].copy()

    # binary target: pathogenic=1, everything else=0
    df["target"] = (df["label"] == "pathogenic").astype(int)

    # one-hot encode categoricals
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

    # collect all feature columns
    feature_cols = NUMERIC_FEATURES.copy()
    for cat in CATEGORICAL_FEATURES:
        # find one-hot columns
        cat_cols = [c for c in df_encoded.columns if c.startswith(cat + "_")]
        feature_cols.extend(cat_cols)

    X = df_encoded[feature_cols].values.astype(np.float32)
    y = df_encoded["target"].values
    genes = df_encoded["gene"].values
    positions = df_encoded["position"].values

    print(f"  prepared {len(X)} samples, {X.shape[1]} features")
    print(f"  class balance: {y.sum()} pathogenic ({y.mean()*100:.1f}%), "
          f"{(1-y).sum():.0f} benign/VUS ({(1-y).mean()*100:.1f}%)")

    return X, y, genes, positions, feature_cols, df_encoded


def logo_cv(X, y, genes, feature_cols):
    """leave-one-gene-out cross-validation."""
    unique_genes = sorted(set(genes))
    print(f"\n  LOGO-CV with {len(unique_genes)} folds...")

    all_preds = np.zeros(len(y))
    all_true = np.zeros(len(y))
    per_gene_results = []

    for fold_gene in unique_genes:
        test_mask = genes == fold_gene
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        n_test = len(y_test)
        n_pos = y_test.sum()
        n_neg = n_test - n_pos

        if n_test < 5:
            print(f"    {fold_gene:<12} skipped (only {n_test} samples)")
            continue

        # class weight to handle imbalance
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        all_preds[test_mask] = y_prob
        all_true[test_mask] = y_test

        # per-gene metrics (only if both classes present)
        if n_pos > 0 and n_neg > 0:
            gene_auc = roc_auc_score(y_test, y_prob)
            gene_ap = average_precision_score(y_test, y_prob)
            print(f"    {fold_gene:<12} n={n_test:>4} (P={n_pos:>3}, B={n_neg:>4})  "
                  f"AUROC={gene_auc:.3f}  AUPRC={gene_ap:.3f}")
        else:
            gene_auc = np.nan
            gene_ap = np.nan
            print(f"    {fold_gene:<12} n={n_test:>4} (P={n_pos:>3}, B={n_neg:>4})  "
                  f"single-class — cannot compute AUROC")

        per_gene_results.append({
            "gene": fold_gene,
            "n_samples": n_test,
            "n_pathogenic": int(n_pos),
            "n_benign_vus": int(n_neg),
            "auroc": gene_auc,
            "auprc": gene_ap,
        })

    return all_preds, all_true, per_gene_results


def compute_feature_importance(X, y, genes, feature_cols):
    """train on all data, extract feature importances."""
    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return imp_df, model


def plot_results(all_preds, all_true, per_gene_results, imp_df):
    """generate evaluation plots."""
    # filter to samples that were actually predicted (non-zero or where test had both classes)
    mask = np.ones(len(all_true), dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. ROC curve
    fpr, tpr, _ = roc_curve(all_true[mask], all_preds[mask])
    overall_auc = roc_auc_score(all_true[mask], all_preds[mask])
    axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f"AUROC = {overall_auc:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve (LOGO-CV)")
    axes[0, 0].legend()

    # 2. PR curve
    prec, rec, _ = precision_recall_curve(all_true[mask], all_preds[mask])
    overall_ap = average_precision_score(all_true[mask], all_preds[mask])
    baseline = all_true[mask].mean()
    axes[0, 1].plot(rec, prec, 'r-', linewidth=2, label=f"AUPRC = {overall_ap:.3f}")
    axes[0, 1].axhline(baseline, color='k', linestyle='--', alpha=0.5,
                         label=f"baseline = {baseline:.3f}")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve (LOGO-CV)")
    axes[0, 1].legend()

    # 3. per-gene AUROC
    gene_df = pd.DataFrame(per_gene_results)
    gene_df_valid = gene_df.dropna(subset=["auroc"]).sort_values("auroc", ascending=True)
    colors = ['#d73027' if auc < 0.6 else '#fee090' if auc < 0.7 else '#91bfdb' if auc < 0.8 else '#4575b4'
              for auc in gene_df_valid["auroc"]]
    axes[1, 0].barh(gene_df_valid["gene"], gene_df_valid["auroc"], color=colors)
    axes[1, 0].axvline(0.5, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel("AUROC")
    axes[1, 0].set_title("Per-Gene AUROC (LOGO-CV)")
    axes[1, 0].set_xlim(0, 1)

    # 4. feature importance (top 15)
    top_imp = imp_df.head(15)
    axes[1, 1].barh(top_imp["feature"][::-1], top_imp["importance"][::-1], color="#4575b4")
    axes[1, 1].set_xlabel("Feature Importance (gain)")
    axes[1, 1].set_title("Top 15 Features")

    plt.tight_layout()
    out_path = FIGURES / "approach_a_xgboost.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  saved figure to {out_path}")

    return overall_auc, overall_ap


def main():
    print("=" * 60)
    print("approach A: XGBoost on handcrafted IDP features")
    print("=" * 60)

    # load feature matrix
    df = pd.read_csv(VARIANTS / "feature_matrix.csv")
    print(f"  loaded {len(df)} variants from feature_matrix.csv")

    # prepare data
    X, y, genes, positions, feature_cols, df_encoded = prepare_data(df)

    # LOGO-CV
    all_preds, all_true, per_gene_results = logo_cv(X, y, genes, feature_cols)

    # overall metrics
    overall_auc = roc_auc_score(all_true, all_preds)
    overall_ap = average_precision_score(all_true, all_preds)
    print(f"\n  === OVERALL RESULTS ===")
    print(f"  AUROC: {overall_auc:.4f}")
    print(f"  AUPRC: {overall_ap:.4f}")
    print(f"  baseline AUPRC (prevalence): {all_true.mean():.4f}")

    # classification report at threshold=0.5
    y_pred = (all_preds > 0.5).astype(int)
    print(f"\n  classification report (threshold=0.5):")
    print(classification_report(all_true, y_pred, target_names=["benign/VUS", "pathogenic"]))

    # feature importance
    print("\n  computing feature importance (full model)...")
    imp_df, full_model = compute_feature_importance(X, y, genes, feature_cols)
    print(f"\n  top 10 features:")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:.4f}")

    # save results
    gene_df = pd.DataFrame(per_gene_results)
    gene_df.to_csv(VARIANTS / "results_approach_a.csv", index=False)
    imp_df.to_csv(VARIANTS / "feature_importance_a.csv", index=False)
    print(f"\n  saved per-gene results to {VARIANTS / 'results_approach_a.csv'}")
    print(f"  saved feature importance to {VARIANTS / 'feature_importance_a.csv'}")

    # also save predictions for comparison
    pred_df = pd.DataFrame({
        "gene": genes,
        "position": positions,
        "true_label": all_true,
        "pred_prob": all_preds,
    })
    # merge back ref_aa/alt_aa for analysis
    df_filt = df[df["label"] != "conflicting"]
    pred_df["ref_aa"] = df_filt["ref_aa"].values
    pred_df["alt_aa"] = df_filt["alt_aa"].values
    pred_df.to_csv(VARIANTS / "predictions_approach_a.csv", index=False)

    # plot
    plot_results(all_preds, all_true, per_gene_results, imp_df)

    # IDR vs structured comparison (preview for approach B)
    print(f"\n  === IDR vs STRUCTURED BREAKDOWN ===")
    df_for_analysis = df_filt.copy()
    df_for_analysis["pred_prob"] = all_preds
    df_for_analysis["target"] = all_true

    idr_mask = df_for_analysis["in_idr"] == 1
    struct_mask = df_for_analysis["in_idr"] == 0

    for name, mask in [("IDR", idr_mask), ("Structured", struct_mask)]:
        sub = df_for_analysis[mask]
        n = len(sub)
        n_path = (sub["target"] == 1).sum()
        if n_path > 0 and n_path < n:
            auc = roc_auc_score(sub["target"], sub["pred_prob"])
            ap = average_precision_score(sub["target"], sub["pred_prob"])
            print(f"  {name}: n={n}, P={n_path}, AUROC={auc:.3f}, AUPRC={ap:.3f}")
        else:
            print(f"  {name}: n={n}, P={n_path} (single class)")


if __name__ == "__main__":
    main()
