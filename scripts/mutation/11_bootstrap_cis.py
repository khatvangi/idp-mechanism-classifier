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

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
VARIANTS = DATA / "variants"

N_BOOTSTRAP = 1000
np.random.seed(42)


def bootstrap_auroc(y_true, y_pred, n_boot=N_BOOTSTRAP):
    """compute bootstrap 95% CI for AUROC."""
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
        return np.nan, np.nan, np.nan, 0
    return np.median(aurocs), np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5), len(aurocs)


if __name__ == "__main__":
    # load data
    df = pd.read_csv(VARIANTS / "esm2_features.csv")
    # exclude HTT
    df = df[df["gene"] != "HTT"].copy()
    print(f"loaded {len(df)} variants (HTT excluded)")

    # per-gene bootstrap CIs for ESM2 LLR
    print(f"\n{'gene':<12} {'n':>5} {'P':>4} {'AUROC':>7} {'95% CI':>16} {'n_boot':>7}")
    print("-" * 55)

    results = []
    for gene in sorted(df["gene"].unique()):
        gdf = df[df["gene"] == gene]
        y = gdf["target"].values
        pred = gdf["esm2_llr"].values
        n_path = int(y.sum())
        n_total = len(y)

        if n_path == 0 or n_path == n_total:
            print(f"{gene:<12} {n_total:>5} {n_path:>4}    n/a")
            results.append({"gene": gene, "n": n_total, "P": n_path,
                            "auroc": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                            "n_boot": 0})
            continue

        auroc_point = roc_auc_score(y, pred)
        med, lo, hi, n_valid = bootstrap_auroc(y, pred)

        print(f"{gene:<12} {n_total:>5} {n_path:>4} {auroc_point:>7.4f} [{lo:>6.3f}, {hi:>6.3f}] {n_valid:>7}")
        results.append({"gene": gene, "n": n_total, "P": n_path,
                        "auroc": auroc_point, "ci_low": lo, "ci_high": hi,
                        "n_boot": n_valid})

    # per-mechanism group bootstrap
    mechanisms = {
        "gof_nonamyloid": {"FUS", "TARDBP", "HNRNPA1", "TIA1", "HNRNPA2B1", "EWSR1", "TAF15"},
        "gof_amyloid": {"SNCA", "TTR", "PRNP", "IAPP"},
        "lof_structured": {"SOD1", "VCP", "LMNA", "CRYAB"},
        "repeat": {"AR", "ATXN3"},
        "condensate": {"DDX4", "NPM1", "SQSTM1", "MAPT"},
    }

    print(f"\n{'mechanism':<20} {'n':>5} {'P':>4} {'AUROC':>7} {'95% CI':>16}")
    print("-" * 55)
    for mech, genes in mechanisms.items():
        mdf = df[df["gene"].isin(genes)]
        y = mdf["target"].values
        pred = mdf["esm2_llr"].values
        n_path = int(y.sum())

        if n_path == 0 or n_path == len(y):
            print(f"{mech:<20} {len(y):>5} {n_path:>4}    n/a")
            continue

        auroc_point = roc_auc_score(y, pred)
        med, lo, hi, n_valid = bootstrap_auroc(y, pred)
        print(f"{mech:<20} {len(y):>5} {n_path:>4} {auroc_point:>7.4f} [{lo:>6.3f}, {hi:>6.3f}]")

        results.append({"gene": f"GROUP:{mech}", "n": len(y), "P": n_path,
                        "auroc": auroc_point, "ci_low": lo, "ci_high": hi,
                        "n_boot": n_valid})

    # save
    out = pd.DataFrame(results)
    out.to_csv(VARIANTS / "bootstrap_cis.csv", index=False)
    print(f"\nsaved to {VARIANTS / 'bootstrap_cis.csv'}")
