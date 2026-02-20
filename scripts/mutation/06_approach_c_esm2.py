#!/usr/bin/env python3
"""
approach C: ESM2-based pathogenicity prediction.
computes per-variant ESM2 log-likelihood ratio (LLR) and embedding features.
then runs XGBoost with LOGO-CV using ESM2 features alone, and combined with
handcrafted features from approach A.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
VARIANTS = DATA / "variants"
SEQS_DIR = DATA / "sequences"
FIGURES = PROJECT / "figures"

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"


def read_fasta(path):
    """read multi-FASTA into dict."""
    seqs = {}
    name = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name = line[1:].split("|")[0].strip()
                buf = []
            else:
                buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs


def compute_per_residue_logits(model, tokenizer, seq, device):
    """compute per-residue log probabilities from ESM2.
    returns log_probs array of shape (seq_len, vocab_size).

    uses the unmasked (autoregressive-style) logits as a fast approximation.
    for true masked marginals, we'd need one forward pass per position.
    """
    inputs = tokenizer(seq, return_tensors="pt", padding=False, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # (seq_len+2, vocab_size)

    # log softmax over vocab dimension
    log_probs = torch.log_softmax(logits, dim=-1)

    # strip [CLS] (position 0) and [EOS] (last position)
    # positions 1..n correspond to residues 0..n-1
    residue_log_probs = log_probs[1:-1]  # (seq_len, vocab_size)

    return residue_log_probs, input_ids


def compute_esm2_features(model, tokenizer, sequences, variants_df, device):
    """compute ESM2 LLR and embedding features for all variants.

    per-variant features:
    - esm2_llr: log P(ref_aa) - log P(alt_aa) at variant position
    - esm2_ref_logprob: log P(ref_aa) — how conserved is the WT residue
    - esm2_alt_logprob: log P(alt_aa) — how tolerable is the mutant
    - esm2_entropy: entropy of the position distribution — how constrained is it
    - esm2_rank_ref: rank of ref_aa among all AAs (0=most likely)
    - esm2_rank_alt: rank of alt_aa among all AAs
    """
    print("\n  computing ESM2 per-residue features...")

    # cache per-gene logits (only 23 proteins to process)
    gene_logprobs = {}
    for gene, seq in sorted(sequences.items()):
        if len(seq) > 1022:
            # ESM2 max length is 1022 tokens (1024 with CLS/EOS)
            # truncate for now (only affects HTT at 3142 aa)
            print(f"    {gene:<12} {len(seq)} aa — TRUNCATING to 1022 for ESM2")
            seq_trunc = seq[:1022]
        else:
            seq_trunc = seq

        log_probs, input_ids = compute_per_residue_logits(model, tokenizer, seq_trunc, device)
        gene_logprobs[gene] = log_probs.cpu().numpy()
        print(f"    {gene:<12} {len(seq_trunc)} aa processed")

    # map AA characters to ESM2 token IDs
    aa_to_token = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        tokens = tokenizer.encode(aa, add_special_tokens=False)
        if tokens:
            aa_to_token[aa] = tokens[0]

    # compute features for each variant
    esm2_rows = []
    n_truncated = 0
    for _, row in variants_df.iterrows():
        gene = row["gene"]
        pos = row["position"]  # 1-based
        ref_aa = row["ref_aa"]
        alt_aa = row["alt_aa"]
        pos_0 = pos - 1  # 0-based

        log_probs = gene_logprobs.get(gene)
        if log_probs is None or pos_0 >= len(log_probs):
            # position beyond truncated sequence (HTT)
            n_truncated += 1
            esm2_rows.append({
                "esm2_llr": 0.0,
                "esm2_ref_logprob": 0.0,
                "esm2_alt_logprob": 0.0,
                "esm2_entropy": 0.0,
                "esm2_rank_ref": 10,
                "esm2_rank_alt": 10,
            })
            continue

        pos_log_probs = log_probs[pos_0]  # (vocab_size,)

        # get token IDs
        ref_token = aa_to_token.get(ref_aa)
        alt_token = aa_to_token.get(alt_aa)

        if ref_token is None or alt_token is None:
            esm2_rows.append({
                "esm2_llr": 0.0, "esm2_ref_logprob": 0.0,
                "esm2_alt_logprob": 0.0, "esm2_entropy": 0.0,
                "esm2_rank_ref": 10, "esm2_rank_alt": 10,
            })
            continue

        ref_lp = float(pos_log_probs[ref_token])
        alt_lp = float(pos_log_probs[alt_token])
        llr = ref_lp - alt_lp  # positive = ref preferred over alt

        # entropy at this position
        probs = np.exp(pos_log_probs)
        # only consider standard AA tokens
        aa_tokens = list(aa_to_token.values())
        aa_probs = probs[aa_tokens]
        aa_probs = aa_probs / aa_probs.sum()  # renormalize
        entropy = -np.sum(aa_probs * np.log(aa_probs + 1e-10))

        # rank of ref and alt among standard AAs
        aa_lps = [(pos_log_probs[t], aa) for aa, t in aa_to_token.items()]
        aa_lps.sort(reverse=True)
        rank_ref = next(i for i, (_, aa) in enumerate(aa_lps) if aa == ref_aa)
        rank_alt = next(i for i, (_, aa) in enumerate(aa_lps) if aa == alt_aa)

        esm2_rows.append({
            "esm2_llr": round(llr, 4),
            "esm2_ref_logprob": round(ref_lp, 4),
            "esm2_alt_logprob": round(alt_lp, 4),
            "esm2_entropy": round(entropy, 4),
            "esm2_rank_ref": rank_ref,
            "esm2_rank_alt": rank_alt,
        })

    if n_truncated > 0:
        print(f"    {n_truncated} variants beyond ESM2 max length (set to defaults)")

    return pd.DataFrame(esm2_rows)


def logo_cv(X, y, genes, feature_cols, model_name="model"):
    """leave-one-gene-out CV, returns predictions and per-gene metrics."""
    unique_genes = sorted(set(genes))
    all_preds = np.zeros(len(y))
    per_gene = []

    for fold_gene in unique_genes:
        test_mask = genes == fold_gene
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        n_test = len(y_test)
        n_pos = int(y_test.sum())

        if n_test < 5 or y_train.sum() < 2:
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
            gene_auc = roc_auc_score(y_test, y_prob)
            per_gene.append({"gene": fold_gene, "auroc": gene_auc,
                             "n": n_test, "n_path": n_pos})

    return all_preds, per_gene


def plot_comparison(results_dict, figpath):
    """plot ROC/PR curves for multiple approaches."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = {"ESM2 only": "#d73027", "Handcrafted only": "#4575b4",
              "Combined": "#1a9850", "ESM2 LLR only": "#f46d43"}

    # ROC curves
    for name, (y_true, y_pred, auc, ap) in results_dict.items():
        if np.isnan(auc):
            continue
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        axes[0].plot(fpr, tpr, color=colors.get(name, "gray"),
                     linewidth=2, label=f"{name}: {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves (LOGO-CV)")
    axes[0].legend(fontsize=9)

    # PR curves
    for name, (y_true, y_pred, auc, ap) in results_dict.items():
        if np.isnan(ap):
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        axes[1].plot(rec, prec, color=colors.get(name, "gray"),
                     linewidth=2, label=f"{name}: {ap:.3f}")
    baseline = list(results_dict.values())[0][0].mean()
    axes[1].axhline(baseline, color='k', linestyle='--', alpha=0.3, label=f"baseline: {baseline:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR Curves (LOGO-CV)")
    axes[1].legend(fontsize=9)

    # per-gene AUROC comparison
    bar_data = {}
    for name, (_, _, _, _) in results_dict.items():
        pass  # we'll add gene-level data in the bar chart from per_gene results

    # bar chart of overall metrics
    names = list(results_dict.keys())
    aucs = [results_dict[n][2] for n in names]
    aps = [results_dict[n][3] for n in names]
    x = np.arange(len(names))
    w = 0.35
    axes[2].bar(x - w/2, aucs, w, label="AUROC", color="#4575b4")
    axes[2].bar(x + w/2, aps, w, label="AUPRC", color="#d73027")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=15, ha="right")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Overall Metrics")
    axes[2].legend()
    axes[2].axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved figure to {figpath}")


def main():
    print("=" * 60)
    print("approach C: ESM2-based pathogenicity prediction")
    print("=" * 60)

    # load sequences
    sequences = read_fasta(SEQS_DIR / "all_proteins.fasta")
    print(f"  loaded {len(sequences)} protein sequences")

    # load feature matrix (has handcrafted features + labels)
    df = pd.read_csv(VARIANTS / "feature_matrix.csv")
    print(f"  loaded {len(df)} variants")

    # filter out conflicting
    df = df[df["label"] != "conflicting"].copy()
    df["target"] = (df["label"] == "pathogenic").astype(int)
    print(f"  after filtering: {len(df)} samples ({df['target'].sum()} pathogenic)")

    # load ESM2 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    print(f"  loading {MODEL_NAME}...")

    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    mlm_model = EsmForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    print("  model loaded")

    # compute ESM2 features
    esm2_df = compute_esm2_features(mlm_model, tokenizer, sequences, df, device)

    # save ESM2 features
    esm2_out = VARIANTS / "esm2_features.csv"
    esm2_combined = pd.concat([df.reset_index(drop=True), esm2_df.reset_index(drop=True)], axis=1)
    esm2_combined.to_csv(esm2_out, index=False)
    print(f"\n  saved ESM2 features to {esm2_out}")

    # spot checks
    print(f"\n  spot checks:")
    snca = esm2_combined[(esm2_combined["gene"] == "SNCA") & (esm2_combined["position"] == 53)]
    if len(snca) > 0:
        r = snca.iloc[0]
        print(f"    SNCA A53T: LLR={r['esm2_llr']:.2f}, ref_rank={r['esm2_rank_ref']}, "
              f"alt_rank={r['esm2_rank_alt']}, entropy={r['esm2_entropy']:.2f}")
    sod1 = esm2_combined[(esm2_combined["gene"] == "SOD1") & (esm2_combined["position"] == 94)]
    if len(sod1) > 0:
        r = sod1.iloc[0]
        print(f"    SOD1 G94D: LLR={r['esm2_llr']:.2f}, ref_rank={r['esm2_rank_ref']}, "
              f"alt_rank={r['esm2_rank_alt']}, entropy={r['esm2_entropy']:.2f}")

    # prepare data for three model variants
    genes = df["gene"].values
    y = df["target"].values

    # ESM2-only features
    esm2_feature_cols = ["esm2_llr", "esm2_ref_logprob", "esm2_alt_logprob",
                          "esm2_entropy", "esm2_rank_ref", "esm2_rank_alt"]
    X_esm2 = esm2_df[esm2_feature_cols].values.astype(np.float32)

    # handcrafted features (same as approach A, without gene-level confounds)
    handcrafted_cols = [
        "local_hydrophobicity", "local_charge_density", "local_aromatic_density",
        "local_glycine_frac", "local_proline_frac", "local_beta_propensity",
        "local_qn_frac", "local_disorder", "position_disorder", "in_idr",
        "relative_position", "local_p_lock", "local_p_pi", "local_p_polar",
        "is_sticker", "blosum62", "grantham_distance", "hydrophobicity_change",
        "size_change", "creates_proline", "destroys_proline", "creates_glycine",
    ]
    # one-hot encode categoricals
    cat_cols = ["charge_change", "aromatic_change", "polarity_change"]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    extra_cols = []
    for cat in cat_cols:
        extra_cols.extend([c for c in df_encoded.columns if c.startswith(cat + "_")])
    all_handcrafted = handcrafted_cols + extra_cols
    X_hand = df_encoded[all_handcrafted].values.astype(np.float32)

    # combined
    X_combined = np.hstack([X_hand, X_esm2])
    combined_cols = all_handcrafted + esm2_feature_cols

    # --- run LOGO-CV for each variant ---
    print(f"\n  === ESM2-only LOGO-CV ===")
    preds_esm2, genes_esm2 = logo_cv(X_esm2, y, genes, esm2_feature_cols, "ESM2")
    auc_esm2 = roc_auc_score(y, preds_esm2) if y.sum() > 0 else np.nan
    ap_esm2 = average_precision_score(y, preds_esm2) if y.sum() > 0 else np.nan
    print(f"  ESM2-only:   AUROC={auc_esm2:.4f}, AUPRC={ap_esm2:.4f}")
    for r in genes_esm2:
        print(f"    {r['gene']:<12} n={r['n']:>4} P={r['n_path']:>3}  AUROC={r['auroc']:.3f}")

    print(f"\n  === Handcrafted-only LOGO-CV ===")
    preds_hand, genes_hand = logo_cv(X_hand, y, genes, all_handcrafted, "Handcrafted")
    auc_hand = roc_auc_score(y, preds_hand) if y.sum() > 0 else np.nan
    ap_hand = average_precision_score(y, preds_hand) if y.sum() > 0 else np.nan
    print(f"  Handcrafted: AUROC={auc_hand:.4f}, AUPRC={ap_hand:.4f}")

    print(f"\n  === Combined LOGO-CV ===")
    preds_comb, genes_comb = logo_cv(X_combined, y, genes, combined_cols, "Combined")
    auc_comb = roc_auc_score(y, preds_comb) if y.sum() > 0 else np.nan
    ap_comb = average_precision_score(y, preds_comb) if y.sum() > 0 else np.nan
    print(f"  Combined:    AUROC={auc_comb:.4f}, AUPRC={ap_comb:.4f}")

    # also test ESM2 LLR alone (just the single feature)
    print(f"\n  === ESM2 LLR alone (no ML, just threshold) ===")
    llr_values = esm2_df["esm2_llr"].values
    auc_llr = roc_auc_score(y, llr_values) if y.sum() > 0 else np.nan
    ap_llr = average_precision_score(y, llr_values) if y.sum() > 0 else np.nan
    print(f"  ESM2 LLR:    AUROC={auc_llr:.4f}, AUPRC={ap_llr:.4f}")

    # summary
    print(f"\n  {'='*60}")
    print(f"  SUMMARY OF ALL APPROACHES")
    print(f"  {'='*60}")
    print(f"  {'Approach':<25} {'AUROC':>8} {'AUPRC':>8}")
    print(f"  {'-'*45}")
    print(f"  {'ESM2 LLR (no ML)':<25} {auc_llr:>8.4f} {ap_llr:>8.4f}")
    print(f"  {'ESM2 features':<25} {auc_esm2:>8.4f} {ap_esm2:>8.4f}")
    print(f"  {'Handcrafted IDP':<25} {auc_hand:>8.4f} {ap_hand:>8.4f}")
    print(f"  {'Combined':<25} {auc_comb:>8.4f} {ap_comb:>8.4f}")
    print(f"  {'Random baseline':<25} {'0.5000':>8} {y.mean():>8.4f}")

    # save results
    results = {
        "approach": ["ESM2_LLR", "ESM2_features", "Handcrafted", "Combined"],
        "auroc": [auc_llr, auc_esm2, auc_hand, auc_comb],
        "auprc": [ap_llr, ap_esm2, ap_hand, ap_comb],
    }
    pd.DataFrame(results).to_csv(VARIANTS / "results_approach_c.csv", index=False)

    # feature importance for combined model
    print("\n  feature importance (combined model, full data)...")
    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    full_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        use_label_encoder=False, random_state=42, verbosity=0,
    )
    full_model.fit(X_combined, y)
    imp_df = pd.DataFrame({
        "feature": combined_cols,
        "importance": full_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\n  top 10 features (combined model):")
    for _, r in imp_df.head(10).iterrows():
        print(f"    {r['feature']:<35} {r['importance']:.4f}")
    imp_df.to_csv(VARIANTS / "feature_importance_combined.csv", index=False)

    # plot
    results_dict = {
        "ESM2 only": (y, preds_esm2, auc_esm2, ap_esm2),
        "Handcrafted only": (y, preds_hand, auc_hand, ap_hand),
        "Combined": (y, preds_comb, auc_comb, ap_comb),
        "ESM2 LLR only": (y, llr_values, auc_llr, ap_llr),
    }
    plot_comparison(results_dict, FIGURES / "approach_c_esm2_comparison.png")


if __name__ == "__main__":
    main()
