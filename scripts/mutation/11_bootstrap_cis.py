#!/usr/bin/env python3
"""
11_bootstrap_cis.py — bootstrap confidence intervals for per-gene AUROCs

computes 1000 bootstrap resamples per gene.
reports 95% CI for each per-gene AUROC.
essential for paper: per-gene AUROCs on 18 pathogenic (FUS) need error bars.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
VARIANTS = DATA / "variants"

N_BOOTSTRAP = 1000
np.random.seed(42)

EVAL_SETS = {
    "clean": {"allowed_labels": {"pathogenic", "benign"}},
    "full_vus": {"allowed_labels": {"pathogenic", "benign", "vus"}},
}


def bootstrap_auroc(y_true, y_pred, n_boot=N_BOOTSTRAP):
    """compute bootstrap 95% CI for AUROC.
    returns: (median, ci_low, ci_high, n_valid, bootstrap_p)
    bootstrap_p = one-sided p-value testing H0: AUROC <= 0.5
    """
    aurocs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_pred[idx]
        # need both classes in bootstrap sample
        if y_b.sum() > 0 and (1 - y_b).sum() > 0:
            aurocs.append(roc_auc_score(y_b, p_b))
    aurocs = np.array(aurocs)
    if len(aurocs) < 100:
        return np.nan, np.nan, np.nan, 0, np.nan
    # one-sided p-value: fraction of bootstrap resamples where AUROC <= 0.5
    boot_p = np.mean(aurocs <= 0.5)
    return np.median(aurocs), np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5), len(aurocs), boot_p


if __name__ == "__main__":
    results = []
    mechanisms = {
        "gof_nonamyloid": {"FUS", "TARDBP", "HNRNPA1", "TIA1", "HNRNPA2B1", "EWSR1", "TAF15"},
        "gof_amyloid": {"SNCA", "TTR", "PRNP", "IAPP"},
        "lof_structured": {"SOD1", "VCP", "LMNA", "CRYAB"},
        "repeat": {"AR", "ATXN3"},
        "condensate": {"DDX4", "NPM1", "SQSTM1", "MAPT"},
    }

    for eval_name, cfg in EVAL_SETS.items():
        df = pd.read_csv(VARIANTS / "esm2_features.csv")
        df = df[df["gene"] != "HTT"].copy()
        df = df[df["label"].isin(cfg["allowed_labels"])].copy()
        df["target"] = (df["label"] == "pathogenic").astype(int)
        print(f"\nloaded {len(df)} variants for {eval_name} (HTT excluded)")

        print(f"\n[{eval_name}] {'gene':<12} {'n':>5} {'P':>4} {'AUROC':>7} {'95% CI':>16} {'n_boot':>7} {'p':>8}")
        print("-" * 75)
        # track indices of gene-level results for BH correction later
        gene_result_indices = []
        for gene in sorted(df["gene"].unique()):
            gdf = df[df["gene"] == gene]
            y = gdf["target"].values
            pred = gdf["esm2_llr"].values
            n_path = int(y.sum())
            n_total = len(y)

            if n_path == 0 or n_path == n_total:
                print(f"{gene:<12} {n_total:>5} {n_path:>4}    n/a")
                results.append({"evaluation_set": eval_name, "entity_type": "gene",
                                "entity": gene, "n": n_total, "P": n_path,
                                "auroc": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                                "n_boot": 0, "bootstrap_p": np.nan,
                                "bh_adjusted_p": np.nan, "bh_significant": False})
                continue

            auroc_point = roc_auc_score(y, pred)
            med, lo, hi, n_valid, boot_p = bootstrap_auroc(y, pred)

            print(f"{gene:<12} {n_total:>5} {n_path:>4} {auroc_point:>7.4f} [{lo:>6.3f}, {hi:>6.3f}] {n_valid:>7} {boot_p:>8.4f}")
            results.append({"evaluation_set": eval_name, "entity_type": "gene",
                            "entity": gene, "n": n_total, "P": n_path,
                            "auroc": auroc_point, "ci_low": lo, "ci_high": hi,
                            "n_boot": n_valid, "bootstrap_p": boot_p,
                            "bh_adjusted_p": np.nan, "bh_significant": False})
            gene_result_indices.append(len(results) - 1)

        # --- BH correction across per-gene p-values for this evaluation set ---
        if gene_result_indices:
            raw_pvals = np.array([results[i]["bootstrap_p"] for i in gene_result_indices])
            # handle any NaN p-values: treat as non-significant (p=1.0)
            pvals_for_bh = np.where(np.isnan(raw_pvals), 1.0, raw_pvals)
            reject, adjusted, _, _ = multipletests(pvals_for_bh, alpha=0.05, method="fdr_bh")
            for k, idx in enumerate(gene_result_indices):
                results[idx]["bh_adjusted_p"] = adjusted[k]
                results[idx]["bh_significant"] = bool(reject[k])

            # print BH correction summary
            n_genes_tested = len(gene_result_indices)
            n_survive = int(reject.sum())
            print(f"\n[{eval_name}] BH correction summary ({n_genes_tested} genes tested, alpha=0.05):")
            print(f"  {n_survive}/{n_genes_tested} genes survive FDR correction")
            for k, idx in enumerate(gene_result_indices):
                gene_name = results[idx]["entity"]
                raw_p = results[idx]["bootstrap_p"]
                adj_p = adjusted[k]
                sig = "*" if reject[k] else ""
                print(f"  {gene_name:<12} raw_p={raw_p:.4f}  bh_adj_p={adj_p:.4f} {sig}")

        print(f"\n[{eval_name}] {'mechanism':<20} {'n':>5} {'P':>4} {'AUROC':>7} {'95% CI':>16}")
        print("-" * 65)
        for mech, genes in mechanisms.items():
            mdf = df[df["gene"].isin(genes)]
            y = mdf["target"].values
            pred = mdf["esm2_llr"].values
            n_path = int(y.sum())

            if n_path == 0 or n_path == len(y):
                print(f"{mech:<20} {len(y):>5} {n_path:>4}    n/a")
                continue

            auroc_point = roc_auc_score(y, pred)
            med, lo, hi, n_valid, boot_p = bootstrap_auroc(y, pred)
            print(f"{mech:<20} {len(y):>5} {n_path:>4} {auroc_point:>7.4f} [{lo:>6.3f}, {hi:>6.3f}]")

            results.append({"evaluation_set": eval_name, "entity_type": "mechanism",
                            "entity": mech, "n": len(y), "P": n_path,
                            "auroc": auroc_point, "ci_low": lo, "ci_high": hi,
                            "n_boot": n_valid, "bootstrap_p": boot_p,
                            "bh_adjusted_p": np.nan, "bh_significant": False})

    # save
    out = pd.DataFrame(results)
    out.to_csv(VARIANTS / "bootstrap_cis.csv", index=False)
    print(f"\nsaved to {VARIANTS / 'bootstrap_cis.csv'}")
