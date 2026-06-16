#!/usr/bin/env python3
"""
10_two_step_predictor.py — functional region-aware pathogenicity predictor

the two-step approach:
  step 1: is the mutation in a known functional/critical region?
  step 2: if yes, apply region-specific scoring; if no, use ESM2 LLR

tests:
  A) region membership alone as predictor
  B) region membership + ESM2 LLR (logistic regression)
  C) compare to ESM2 LLR alone (baseline)

evaluation: leave-one-gene-out cross-validation (LOGO-CV)
critical question: does this beat ESM2 LLR alone, especially for GoF genes?
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no GPU needed

# minimum benign variant count for a gene to be considered adequately powered
# for per-gene AUROC reporting. genes below this threshold get a power_flag
# and are reported in supplementary only, not main text.
MIN_BENIGN_FOR_POWER = 5

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
VARIANTS = DATA / "variants"
FIGURES = PROJECT / "figures"

EVAL_SETS = {
    "clean": {"allowed_labels": {"pathogenic", "benign"}},
    "full_vus": {"allowed_labels": {"pathogenic", "benign", "vus"}},
}

# ── functional region annotations ─────────────────────────────────────────────
# format: {gene: {"region_name": (start, end), ...}}
# all coordinates are 0-indexed, [start, end) — matching UniProt minus 1
# "critical" regions for GoF genes are tagged with * prefix
PROTEIN_REGIONS = {
    "FUS": {
        "QGSY-rich LCD": (0, 165),
        "RGG1": (165, 267),
        "RRM": (282, 371),
        "RGG2": (371, 422),
        "Zinc finger": (422, 453),
        "RGG3": (453, 501),
        "*NLS": (501, 526),                 # critical: PY-NLS, transportin-1 binding
    },
    "TARDBP": {
        "NTD": (0, 76),
        "NLS": (82, 98),
        "RRM1": (104, 176),
        "RRM2": (191, 262),
        "*Glycine-rich LCD": (274, 414),     # critical: aggregation-prone LCD
    },
    "HNRNPA1": {
        "RRM1": (15, 89),
        "RRM2": (105, 183),
        "*PrLD": (186, 372),                 # critical: prion-like domain
    },
    "TIA1": {
        "RRM1": (0, 92),
        "RRM2": (93, 183),
        "RRM3": (197, 274),
        "*PrLD": (292, 386),                 # critical: prion-like domain
    },
    "HNRNPA2B1": {
        "RRM1": (0, 85),
        "RRM2": (104, 182),
        "*PrLD": (190, 341),                 # critical: prion-like domain
    },
    "EWSR1": {
        "*LCD": (0, 280),                    # critical: low-complexity domain (SYGQ-rich)
        "RRM": (350, 418),
        "RGG": (418, 656),
    },
    "TAF15": {
        "*LCD": (0, 210),                    # critical: low-complexity domain (SYGQ-rich)
        "RRM": (234, 310),
        "RGG": (310, 592),
    },
    "SNCA": {
        "N-terminal (lipid)": (0, 60),
        "*NAC region": (60, 95),             # critical: aggregation-prone core
        "C-terminal (acidic)": (95, 140),
    },
    "TTR": {
        "Signal peptide": (0, 20),
        "*Beta core": (20, 127),             # critical: structured amyloid-prone
        "Tail": (127, 147),
    },
    "PRNP": {
        "Signal peptide": (0, 22),
        "Octarepeat": (51, 91),
        "*Hydrophobic core + helices": (106, 228),  # critical: misfolding-prone
        "GPI anchor": (231, 253),
    },
    "IAPP": {
        "Signal peptide": (0, 22),
        "*Amyloid core": (22, 67),           # critical: amyloid-forming region (20-29 is core)
        "C-peptide": (67, 89),
    },
    "SOD1": {
        "*Beta barrel": (0, 154),            # critical: entire structured protein
    },
    "LMNA": {
        "Head domain": (0, 28),
        "*Rod domain": (28, 381),            # critical: coiled-coil (most mutations here)
        "Ig-like fold": (428, 549),
        "Tail": (549, 664),
    },
    "VCP": {
        "*N domain": (0, 187),               # critical: substrate binding (most disease mutations)
        "D1 ATPase": (209, 460),
        "D2 ATPase": (481, 763),
        "C-terminal": (763, 806),
    },
    "AR": {
        "NTD (IDR)": (0, 537),
        "*DBD": (537, 625),                  # critical: DNA-binding domain (many mutations)
        "Hinge": (625, 669),
        "*LBD": (669, 920),                  # critical: ligand-binding domain
    },
    "CRYAB": {
        "N-terminal": (0, 67),
        "*Alpha-crystallin domain": (67, 150),  # critical: chaperone function
        "C-terminal extension": (150, 175),
    },
    "HTT": {
        "Exon 1 (polyQ)": (0, 90),
        "HEAT repeats": (90, 3144),          # massive structured domain
    },
    "ATXN3": {
        "*Josephin domain": (0, 182),        # critical: ubiquitin protease
        "UIM1": (224, 243),
        "UIM2": (244, 263),
        "PolyQ": (291, 362),
    },
    "DDX4": {
        "*IDR N-terminal": (0, 236),         # critical: phase-separating IDR
        "DEAD-box helicase": (236, 662),
    },
    "NPM1": {
        "Oligomerization": (0, 117),
        "Acidic regions": (117, 240),
        "*C-terminal (nucleolar)": (240, 294),  # critical: nucleolar localization
    },
    "SQSTM1": {
        "PB1 domain": (0, 102),
        "ZZ domain": (122, 167),
        "TRAF6-binding": (225, 250),
        "LIR": (335, 344),
        "*UBA domain": (387, 436),           # critical: ubiquitin binding
    },
    "MAPT": {
        "N-terminal inserts": (0, 150),
        "Proline-rich": (150, 244),
        "*MTBD": (244, 368),                 # critical: microtubule-binding domain (R1-R4)
        "C-terminal": (368, 441),
    },
    "APP": {
        "Extracellular": (0, 624),
        "TM": (624, 648),
        "*Abeta region": (672, 713),         # critical: amyloid beta peptide
        "AICD": (713, 770),
    },
}

# mechanism group assignments
GOF_NONAMYLOID = {"FUS", "TARDBP", "HNRNPA1", "TIA1", "HNRNPA2B1", "EWSR1", "TAF15"}
GOF_AMYLOID = {"SNCA", "TTR", "PRNP", "IAPP"}
LOF_STRUCTURED = {"SOD1", "VCP", "LMNA", "CRYAB"}
REPEAT = {"HTT", "ATXN3", "AR"}
CONDENSATE = {"DDX4", "NPM1", "SQSTM1", "MAPT"}


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
    return "unknown"


def annotate_regions(df):
    """add region membership columns to variant dataframe."""
    # in_any_critical: is the variant in a *-prefixed "critical" region?
    # in_any_annotated: is the variant in any annotated region?
    # critical_region_name: name of the critical region (if applicable)
    in_critical = []
    in_annotated = []
    critical_name = []

    for _, row in df.iterrows():
        gene = row["gene"]
        pos = int(row["position"]) - 1  # 0-indexed

        regions = PROTEIN_REGIONS.get(gene, {})
        found_critical = False
        found_any = False
        crit_name = ""

        for rname, (start, end) in regions.items():
            if start <= pos < end:
                found_any = True
                if rname.startswith("*"):
                    found_critical = True
                    crit_name = rname.lstrip("*")

        in_critical.append(int(found_critical))
        in_annotated.append(int(found_any))
        critical_name.append(crit_name)

    df["in_critical_region"] = in_critical
    df["in_any_region"] = in_annotated
    df["critical_region_name"] = critical_name
    return df


def load_eval_dataset(eval_name):
    """Load one named evaluation dataset."""
    if eval_name not in EVAL_SETS:
        raise ValueError(f"unknown eval set: {eval_name}")

    df = pd.read_csv(VARIANTS / "esm2_features.csv")
    df = df[df["gene"] != "HTT"].copy()
    df = df[df["label"].isin(EVAL_SETS[eval_name]["allowed_labels"])].copy()
    df["target"] = (df["label"] == "pathogenic").astype(int)
    df["mechanism"] = df["gene"].apply(get_mechanism)
    return annotate_regions(df)


def logo_cv_two_step(df, method="membership_only"):
    """leave-one-gene-out CV for two-step predictor.

    methods:
      "membership_only" — predict using in_critical_region flag alone
      "membership_plus_llr" — logistic regression on in_critical_region + esm2_llr
      "esm2_only" — raw ESM2 LLR (baseline)
      "llr_within_critical" — ESM2 LLR but flipped: high LLR in non-critical regions
                              = pathogenic; low LLR in critical regions may also = pathogenic
    """
    genes = sorted(df["gene"].unique())
    all_preds = []
    all_truth = []
    all_genes_out = []

    for test_gene in genes:
        train = df[df["gene"] != test_gene]
        test = df[df["gene"] == test_gene]

        if test["target"].nunique() < 2:
            continue  # can't compute AUROC

        y_train = train["target"].values
        y_test = test["target"].values

        if method == "membership_only":
            # predict: in_critical_region = pathogenic
            preds = test["in_critical_region"].values.astype(float)

        elif method == "esm2_only":
            preds = test["esm2_llr"].values

        elif method == "membership_plus_llr":
            # logistic regression: in_critical_region + esm2_llr
            X_train = train[["in_critical_region", "esm2_llr"]].values
            X_test = test[["in_critical_region", "esm2_llr"]].values

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf.fit(X_train_sc, y_train)
            preds = clf.predict_proba(X_test_sc)[:, 1]

        elif method == "membership_plus_llr_plus_features":
            # logistic regression: region + llr + charge + position
            feature_cols = ["in_critical_region", "esm2_llr",
                            "local_charge_density", "relative_position",
                            "position_disorder"]
            X_train = train[feature_cols].values
            X_test = test[feature_cols].values

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf.fit(X_train_sc, y_train)
            preds = clf.predict_proba(X_test_sc)[:, 1]

        else:
            raise ValueError(f"unknown method: {method}")

        all_preds.extend(preds)
        all_truth.extend(y_test)
        all_genes_out.extend([test_gene] * len(y_test))

    return np.array(all_truth), np.array(all_preds), np.array(all_genes_out)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    methods = [
        "esm2_only",
        "membership_only",
        "membership_plus_llr",
        "membership_plus_llr_plus_features",
    ]

    all_summary = []
    all_mechanism_rows = []
    all_gene_rows = []
    figure_payload = None
    annotated_eval_sets = {}

    for eval_name in ["clean", "full_vus"]:
        print(f"\nloading data for {eval_name}...")
        df = load_eval_dataset(eval_name)
        annotated_eval_sets[eval_name] = df.copy()

        print(f"  total variants: {len(df)}")
        print(f"  pathogenic: {(df['target'] == 1).sum()}")
        print(f"  benign: {(df['label'] == 'benign').sum()}")
        print(f"  vus: {(df['label'] == 'vus').sum()}")
        print(f"  in critical region: {df['in_critical_region'].sum()} "
              f"({df['in_critical_region'].mean():.1%})")

        print("\n  per-mechanism critical region coverage:")
        for mech in ["gof_nonamyloid", "gof_amyloid", "lof_structured", "repeat", "condensate"]:
            mdf = df[df["mechanism"] == mech]
            if len(mdf) == 0:
                continue
            n_crit = mdf["in_critical_region"].sum()
            n_path_crit = ((mdf["target"] == 1) & (mdf["in_critical_region"] == 1)).sum()
            n_path_total = (mdf["target"] == 1).sum()
            print(f"    {mech:<20} n={len(mdf):>4}  in_critical={n_crit:>4}  "
                  f"path_in_critical={n_path_crit:>3}/{n_path_total:>3} "
                  f"({n_path_crit/max(n_path_total,1):.0%})")

        results = {}
        print(f"\n{'='*70}")
        print(f"LOGO-CV results ({eval_name}, excluding HTT)")
        print(f"{'='*70}")

        for method in methods:
            y_true, y_pred, genes_out = logo_cv_two_step(df, method)
            auroc = roc_auc_score(y_true, y_pred)
            prec, rec, _ = precision_recall_curve(y_true, y_pred)
            auprc = auc(rec, prec)
            results[method] = {"auroc": auroc, "auprc": auprc,
                               "y_true": y_true, "y_pred": y_pred,
                               "genes": genes_out}
            all_summary.append({
                "evaluation_set": eval_name,
                "method": method,
                "auroc": auroc,
                "auprc": auprc,
            })
            print(f"  {method:<40} AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

        print(f"\n--- per-mechanism AUROC ({eval_name}) ---")
        for method in methods:
            y_true = results[method]["y_true"]
            y_pred = results[method]["y_pred"]
            genes_out = results[method]["genes"]

            print(f"\n  {method}:")
            for mech in ["gof_nonamyloid", "gof_amyloid", "lof_structured", "repeat", "condensate"]:
                mech_genes = {
                    "gof_nonamyloid": GOF_NONAMYLOID,
                    "gof_amyloid": GOF_AMYLOID,
                    "lof_structured": LOF_STRUCTURED,
                    "repeat": REPEAT,
                    "condensate": CONDENSATE,
                }[mech]
                mask = np.array([g in mech_genes for g in genes_out])
                if mask.sum() == 0:
                    continue
                ym = y_true[mask]
                pm = y_pred[mask]
                if ym.sum() > 0 and (1 - ym).sum() > 0:
                    a = roc_auc_score(ym, pm)
                    print(f"    {mech:<20} AUROC={a:.4f} (n={mask.sum()}, P={int(ym.sum())})")
                    all_mechanism_rows.append({
                        "evaluation_set": eval_name,
                        "method": method,
                        "mechanism": mech,
                        "n": int(mask.sum()),
                        "n_pathogenic": int(ym.sum()),
                        "auroc": a,
                    })
                else:
                    print(f"    {mech:<20} n/a (n={mask.sum()}, P={int(ym.sum())})")

        print(f"\n--- per-gene AUROC ({eval_name}) ---")
        for method in methods:
            y_true = results[method]["y_true"]
            y_pred = results[method]["y_pred"]
            genes_out = results[method]["genes"]
            print(f"\n  {method}:")
            n_underpowered = 0
            for gene in sorted(set(genes_out)):
                mask = np.array([g == gene for g in genes_out])
                if mask.sum() == 0:
                    continue
                ym = y_true[mask]
                pm = y_pred[mask]
                # count benign variants for power assessment
                n_benign = int((1 - ym).sum())
                n_path = int(ym.sum())
                adequately_powered = n_benign >= MIN_BENIGN_FOR_POWER and n_path >= 1
                power_flag = "powered" if adequately_powered else "underpowered"
                if not adequately_powered:
                    n_underpowered += 1
                if ym.sum() > 0 and (1 - ym).sum() > 0:
                    a = roc_auc_score(ym, pm)
                    marker = "" if adequately_powered else " [UNDERPOWERED]"
                    print(f"    {gene:<12} AUROC={a:.4f} (n={mask.sum()}, P={n_path}, B={n_benign}){marker}")
                    all_gene_rows.append({
                        "evaluation_set": eval_name,
                        "method": method,
                        "gene": gene,
                        "n": int(mask.sum()),
                        "n_pathogenic": n_path,
                        "n_benign": n_benign,
                        "auroc": a,
                        "power_flag": power_flag,
                    })
                else:
                    print(f"    {gene:<12} n/a (n={mask.sum()}, P={n_path}, B={n_benign}) [UNDERPOWERED]")
            if n_underpowered > 0:
                print(f"    ({n_underpowered} genes underpowered: <{MIN_BENIGN_FOR_POWER} benign variants)")

        if eval_name == "clean":
            figure_payload = (df, results)

    if figure_payload is None:
        raise RuntimeError("clean evaluation results were not generated")

    df, results = figure_payload
    print("\ngenerating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # panel A: overall ROC comparison
    ax = axes[0]
    colors = {"esm2_only": "#666666", "membership_only": "#e74c3c",
              "membership_plus_llr": "#2ecc71",
              "membership_plus_llr_plus_features": "#3498db"}
    labels = {"esm2_only": f"ESM2 LLR ({results['esm2_only']['auroc']:.3f})",
              "membership_only": f"Region only ({results['membership_only']['auroc']:.3f})",
              "membership_plus_llr": f"Region + LLR ({results['membership_plus_llr']['auroc']:.3f})",
              "membership_plus_llr_plus_features": f"Region + LLR + feat ({results['membership_plus_llr_plus_features']['auroc']:.3f})"}

    from sklearn.metrics import roc_curve
    for method in methods:
        fpr, tpr, _ = roc_curve(results[method]["y_true"], results[method]["y_pred"])
        ax.plot(fpr, tpr, color=colors[method], label=labels[method], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("overall LOGO-CV (excl. HTT)")
    ax.legend(fontsize=8)

    # panel B: per-mechanism AUROC bars
    ax = axes[1]
    mech_order = ["gof_nonamyloid", "gof_amyloid", "lof_structured"]
    mech_labels = {"gof_nonamyloid": "GoF\nnon-amyloid",
                   "gof_amyloid": "GoF\namyloid",
                   "lof_structured": "LoF\nstructured"}
    bar_methods = ["esm2_only", "membership_plus_llr"]
    x = np.arange(len(mech_order))
    width = 0.35

    for i, method in enumerate(bar_methods):
        y_true = results[method]["y_true"]
        y_pred = results[method]["y_pred"]
        genes_out = results[method]["genes"]
        aurocs = []
        for mech in mech_order:
            mech_genes = {"gof_nonamyloid": GOF_NONAMYLOID,
                          "gof_amyloid": GOF_AMYLOID,
                          "lof_structured": LOF_STRUCTURED}[mech]
            mask = np.array([g in mech_genes for g in genes_out])
            ym = y_true[mask]
            pm = y_pred[mask]
            if ym.sum() > 0 and (1 - ym).sum() > 0:
                aurocs.append(roc_auc_score(ym, pm))
            else:
                aurocs.append(0.5)
        bar_label = "ESM2 LLR" if method == "esm2_only" else "Region + LLR"
        ax.bar(x + i * width - width / 2, aurocs, width,
               label=bar_label,
               color=colors[method], alpha=0.8)

    ax.axhline(0.5, color="k", linestyle="--", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([mech_labels[m] for m in mech_order])
    ax.set_ylabel("AUROC")
    ax.set_title("per-mechanism comparison")
    ax.legend()
    ax.set_ylim(0.3, 1.0)

    # panel C: per-gene AUROC for GoF non-amyloid
    ax = axes[2]
    gof_genes = sorted([g for g in GOF_NONAMYLOID
                        if (df[df["gene"] == g]["target"] == 1).sum() >= 3])
    x = np.arange(len(gof_genes))
    for i, method in enumerate(["esm2_only", "membership_plus_llr"]):
        y_true = results[method]["y_true"]
        y_pred = results[method]["y_pred"]
        genes_out = results[method]["genes"]
        aurocs = []
        for gene in gof_genes:
            mask = np.array([g == gene for g in genes_out])
            ym = y_true[mask]
            pm = y_pred[mask]
            if ym.sum() > 0 and (1 - ym).sum() > 0:
                aurocs.append(roc_auc_score(ym, pm))
            else:
                aurocs.append(0.5)
        bar_label = "ESM2 LLR" if method == "esm2_only" else "Region + LLR"
        ax.bar(x + i * width - width / 2, aurocs, width,
               label=bar_label,
               color=colors[method], alpha=0.8)

    ax.axhline(0.5, color="k", linestyle="--", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(gof_genes, rotation=45, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("GoF non-amyloid genes")
    ax.legend()
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    fig.savefig(FIGURES / "two_step_predictor.png", dpi=150, bbox_inches="tight")
    print(f"  saved {FIGURES / 'two_step_predictor.png'}")

    # ── save results ──────────────────────────────────────────────────────────
    pd.DataFrame(all_summary).to_csv(VARIANTS / "results_two_step.csv", index=False)
    print(f"  saved {VARIANTS / 'results_two_step.csv'}")
    pd.DataFrame(all_mechanism_rows).to_csv(VARIANTS / "results_two_step_by_mechanism.csv", index=False)
    print(f"  saved {VARIANTS / 'results_two_step_by_mechanism.csv'}")
    pd.DataFrame(all_gene_rows).to_csv(VARIANTS / "results_two_step_by_gene.csv", index=False)
    print(f"  saved {VARIANTS / 'results_two_step_by_gene.csv'}")

    # save annotated variants with explicit evaluation-set filenames.
    clean_df = annotated_eval_sets["clean"]
    full_vus_df = annotated_eval_sets["full_vus"]
    clean_df.to_csv(VARIANTS / "variants_with_regions.csv", index=False)
    print(f"  saved {VARIANTS / 'variants_with_regions.csv'} (clean primary set)")
    clean_df.to_csv(VARIANTS / "variants_with_regions_clean.csv", index=False)
    print(f"  saved {VARIANTS / 'variants_with_regions_clean.csv'}")
    full_vus_df.to_csv(VARIANTS / "variants_with_regions_full_vus.csv", index=False)
    print(f"  saved {VARIANTS / 'variants_with_regions_full_vus.csv'}")

    print("\ndone.")
