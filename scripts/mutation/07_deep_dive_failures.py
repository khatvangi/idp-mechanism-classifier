#!/usr/bin/env python3
"""
deep dive: why does ESM2 LLR fail for certain genes?
analyzes FUS, TARDBP, and other poor performers in detail.
investigates position-level conservation landscapes, regional effects,
mechanism-specific patterns, and false positive/negative analysis.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
VARIANTS = DATA / "variants"
SEQS_DIR = DATA / "sequences"
DISORDER_DIR = DATA / "disorder"
FIGURES = PROJECT / "figures"

# known functional regions (0-indexed, [start, end))
# from UniProt annotations + literature
PROTEIN_REGIONS = {
    "FUS": {
        "QGSY-rich LCD": (0, 165),       # prion-like domain, phase separation
        "RGG1": (165, 267),               # RNA-binding, methylation sites
        "RRM": (282, 371),                # RNA recognition motif
        "RGG2": (371, 422),               # RNA-binding
        "Zinc finger": (422, 453),        # DNA/RNA binding
        "RGG3": (453, 501),               # RNA-binding
        "NLS": (501, 526),                # PY-NLS (nuclear localization signal)
    },
    "TARDBP": {
        "NTD": (0, 76),                   # N-terminal domain
        "NLS": (82, 98),                  # nuclear localization signal
        "RRM1": (104, 176),               # RNA recognition motif 1
        "RRM2": (191, 262),               # RNA recognition motif 2
        "Glycine-rich LCD": (274, 414),   # low-complexity, aggregation-prone
    },
    "SOD1": {
        "Beta barrel": (0, 80),           # mostly structured
        "Active loop": (80, 84),          # Cu/Zn binding
        "Electrostatic loop": (121, 142), # loop VII
        "Beta barrel 2": (84, 154),       # rest of barrel
    },
    "LMNA": {
        "Head domain": (0, 28),           # N-terminal head
        "Coil 1A": (28, 67),              # rod domain segments
        "Linker L1": (67, 78),
        "Coil 1B": (78, 219),
        "Linker L12": (219, 231),
        "Coil 2": (231, 381),
        "Ig-like fold": (428, 549),       # immunoglobulin-like
        "Tail": (549, 664),              # C-terminal tail (includes CAAX)
    },
    "TTR": {
        "Signal peptide": (0, 20),        # cleaved
        "Beta strand A": (20, 36),
        "Loop AB": (36, 44),
        "Beta strand B": (44, 54),
        "BC loop": (54, 63),
        "Beta strand C-H": (63, 127),     # core beta sandwich
        "EF helix": (85, 96),             # short alpha helix
        "Tail": (127, 147),
    },
    "SNCA": {
        "N-terminal (lipid)": (0, 60),    # lipid-binding amphipathic helix
        "NAC region": (60, 95),           # aggregation-prone core
        "C-terminal (acidic)": (95, 140), # acidic tail, disordered
    },
    "AR": {
        "NTD (IDR)": (0, 537),           # N-terminal disordered domain
        "DBD": (537, 625),               # DNA-binding domain (Zn fingers)
        "Hinge": (625, 669),             # flexible linker
        "LBD": (669, 920),              # ligand-binding domain
    },
    "VCP": {
        "N domain": (0, 187),            # substrate binding
        "D1 ATPase": (209, 460),         # first ATPase ring
        "D1-D2 linker": (460, 481),
        "D2 ATPase": (481, 763),         # second ATPase ring
        "C-terminal": (763, 806),        # regulatory tail
    },
    "PRNP": {
        "Signal peptide": (0, 22),
        "Octarepeat": (51, 91),          # copper-binding octarepeat
        "Hydrophobic core": (106, 126),  # highly conserved
        "Helix 1": (144, 156),
        "Helix 2-3 (PrP-C)": (172, 228), # structured domain
        "GPI anchor": (231, 253),
    },
}


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


def load_all_data():
    """load ESM2 features, disorder predictions, and sequences."""
    esm2_df = pd.read_csv(VARIANTS / "esm2_features.csv")
    disorder_df = pd.read_csv(DISORDER_DIR / "per_residue_disorder.csv")
    sequences = read_fasta(SEQS_DIR / "all_proteins.fasta")

    # filter conflicting, add binary target
    esm2_df = esm2_df[esm2_df["label"] != "conflicting"].copy()
    esm2_df["target"] = (esm2_df["label"] == "pathogenic").astype(int)

    return esm2_df, disorder_df, sequences


def analyze_per_gene_detail(df):
    """detailed per-gene ESM2 LLR analysis."""
    print("\n" + "=" * 70)
    print("DETAILED PER-GENE ESM2 LLR ANALYSIS")
    print("=" * 70)

    results = []
    for gene in sorted(df["gene"].unique()):
        gdf = df[df["gene"] == gene]
        n = len(gdf)
        n_path = (gdf["target"] == 1).sum()
        n_ben = (gdf["target"] == 0).sum()

        llr_path = gdf[gdf["target"] == 1]["esm2_llr"]
        llr_ben = gdf[gdf["target"] == 0]["esm2_llr"]

        if n_path > 0 and n_ben > 0:
            auc = roc_auc_score(gdf["target"], gdf["esm2_llr"])
            # mean LLR difference
            mean_diff = llr_path.mean() - llr_ben.mean()
            # median LLR for each class
            med_path = llr_path.median()
            med_ben = llr_ben.median()

            results.append({
                "gene": gene, "n": n, "n_path": n_path, "n_ben": n_ben,
                "auroc": auc, "mean_llr_diff": mean_diff,
                "med_llr_path": med_path, "med_llr_ben": med_ben,
                "mean_llr_path": llr_path.mean(), "mean_llr_ben": llr_ben.mean(),
            })
            print(f"  {gene:<12} n={n:>4} P={n_path:>3} B={n_ben:>3}  "
                  f"AUROC={auc:.3f}  pathLLR={llr_path.mean():.2f}±{llr_path.std():.2f}  "
                  f"benLLR={llr_ben.mean():.2f}±{llr_ben.std():.2f}  "
                  f"Δ={mean_diff:.2f}")
        else:
            print(f"  {gene:<12} n={n:>4} P={n_path:>3} B={n_ben:>3}  single class")

    return pd.DataFrame(results)


def analyze_regions(df, gene, regions, sequences):
    """analyze ESM2 performance per functional region for a specific gene."""
    gdf = df[df["gene"] == gene].copy()
    seq = sequences.get(gene, "")

    print(f"\n  {gene} regional analysis ({len(gdf)} variants, {len(seq)} aa):")
    print(f"  {'Region':<25} {'n':>4} {'P':>3} {'B':>4} {'AUROC':>7} "
          f"{'LLR_P':>7} {'LLR_B':>7} {'Δ':>6}")
    print(f"  {'-'*75}")

    region_results = []
    for region_name, (start, end) in sorted(regions.items(), key=lambda x: x[1][0]):
        # positions are 1-based in df, region bounds are 0-based
        mask = (gdf["position"] >= start + 1) & (gdf["position"] <= end)
        rdf = gdf[mask]

        n = len(rdf)
        n_path = (rdf["target"] == 1).sum()
        n_ben = (rdf["target"] == 0).sum()

        if n < 2:
            print(f"  {region_name:<25} {n:>4} — too few")
            continue

        mean_llr_path = rdf[rdf["target"] == 1]["esm2_llr"].mean() if n_path > 0 else np.nan
        mean_llr_ben = rdf[rdf["target"] == 0]["esm2_llr"].mean() if n_ben > 0 else np.nan

        if n_path > 0 and n_ben > 0:
            auc = roc_auc_score(rdf["target"], rdf["esm2_llr"])
            diff = mean_llr_path - mean_llr_ben
            print(f"  {region_name:<25} {n:>4} {n_path:>3} {n_ben:>4} "
                  f"{auc:>7.3f} {mean_llr_path:>7.2f} {mean_llr_ben:>7.2f} {diff:>6.2f}")
        else:
            auc = np.nan
            diff = np.nan
            print(f"  {region_name:<25} {n:>4} {n_path:>3} {n_ben:>4}  single class")

        region_results.append({
            "gene": gene, "region": region_name,
            "start": start, "end": end,
            "n": n, "n_path": n_path, "n_ben": n_ben,
            "auroc": auc, "llr_diff": diff,
        })

    return region_results


def false_positive_negative_analysis(df, gene):
    """identify most confidently wrong predictions for a gene."""
    gdf = df[df["gene"] == gene].copy()
    if gdf["target"].nunique() < 2:
        return

    print(f"\n  {gene} false positive/negative analysis:")

    # false negatives: pathogenic but LOW LLR (ESM2 thinks it's fine)
    path_df = gdf[gdf["target"] == 1].sort_values("esm2_llr")
    print(f"\n    worst false negatives (pathogenic with LOW LLR — ESM2 says 'tolerable'):")
    for _, r in path_df.head(8).iterrows():
        print(f"      pos {r['position']:>4} {r['ref_aa']}→{r['alt_aa']}  "
              f"LLR={r['esm2_llr']:>6.2f}  disorder={r['position_disorder']:.2f}  "
              f"idr={r['in_idr']}  grantham={r['grantham_distance']}")

    # false positives: benign/VUS but HIGH LLR (ESM2 says it should be damaging)
    ben_df = gdf[gdf["target"] == 0].sort_values("esm2_llr", ascending=False)
    print(f"\n    worst false positives (benign/VUS with HIGH LLR — ESM2 says 'damaging'):")
    for _, r in ben_df.head(8).iterrows():
        print(f"      pos {r['position']:>4} {r['ref_aa']}→{r['alt_aa']}  "
              f"LLR={r['esm2_llr']:>6.2f}  disorder={r['position_disorder']:.2f}  "
              f"idr={r['in_idr']}  grantham={r['grantham_distance']}")


def conservation_vs_disorder_landscape(df, sequences, disorder_df):
    """analyze the relationship between conservation (ESM2 ref logprob),
    disorder, and pathogenicity across all genes."""
    print("\n" + "=" * 70)
    print("CONSERVATION vs DISORDER vs PATHOGENICITY")
    print("=" * 70)

    # bin variants by disorder level
    bins = [(0, 0.3, "ordered"), (0.3, 0.5, "boundary"), (0.5, 0.8, "moderate IDR"),
            (0.8, 1.01, "strong IDR")]

    print(f"\n  {'Disorder bin':<20} {'n':>5} {'P':>4} {'AUROC':>7} "
          f"{'meanLLR_P':>10} {'meanLLR_B':>10}")
    print(f"  {'-'*65}")

    for lo, hi, name in bins:
        mask = (df["position_disorder"] >= lo) & (df["position_disorder"] < hi)
        sub = df[mask]
        n = len(sub)
        n_path = (sub["target"] == 1).sum()
        n_ben = (sub["target"] == 0).sum()

        if n_path > 0 and n_ben > 0:
            auc = roc_auc_score(sub["target"], sub["esm2_llr"])
            mean_p = sub[sub["target"] == 1]["esm2_llr"].mean()
            mean_b = sub[sub["target"] == 0]["esm2_llr"].mean()
            print(f"  {name:<20} {n:>5} {n_path:>4} {auc:>7.3f} {mean_p:>10.2f} {mean_b:>10.2f}")
        else:
            print(f"  {name:<20} {n:>5} {n_path:>4}  single class")

    # is conservation (ref_logprob) itself less informative in IDRs?
    print(f"\n  mean ESM2 ref_logprob by disorder level:")
    for lo, hi, name in bins:
        mask = (df["position_disorder"] >= lo) & (df["position_disorder"] < hi)
        sub = df[mask]
        if len(sub) > 0:
            mean_ref = sub["esm2_ref_logprob"].mean()
            std_ref = sub["esm2_ref_logprob"].std()
            mean_entropy = sub["esm2_entropy"].mean()
            print(f"  {name:<20} ref_logprob={mean_ref:.2f}±{std_ref:.2f}  "
                  f"entropy={mean_entropy:.2f}")


def mechanism_analysis(df):
    """group genes by disease mechanism and compare ESM2 performance."""
    print("\n" + "=" * 70)
    print("MECHANISM-GROUPED ANALYSIS")
    print("=" * 70)

    # disease mechanism grouping (from literature)
    mechanisms = {
        "Toxic aggregation (amyloid)": ["SNCA", "TTR", "PRNP", "IAPP"],
        "Toxic aggregation (non-amyloid)": ["FUS", "TARDBP", "HNRNPA1", "TIA1",
                                              "HNRNPA2B1", "EWSR1", "TAF15"],
        "Loss of function (structured)": ["SOD1", "VCP", "LMNA", "CRYAB"],
        "Repeat expansion related": ["HTT", "ATXN3", "AR"],
        "Phase separation / condensate": ["DDX4", "NPM1", "SQSTM1", "MAPT"],
    }

    for mech_name, genes in mechanisms.items():
        mask = df["gene"].isin(genes)
        sub = df[mask]
        n = len(sub)
        n_path = (sub["target"] == 1).sum()
        n_ben = (sub["target"] == 0).sum()

        if n_path > 0 and n_ben > 0:
            auc = roc_auc_score(sub["target"], sub["esm2_llr"])
            mean_diff = (sub[sub["target"] == 1]["esm2_llr"].mean() -
                        sub[sub["target"] == 0]["esm2_llr"].mean())
            print(f"\n  {mech_name}:")
            print(f"    genes: {', '.join(genes)}")
            print(f"    n={n}, P={n_path}, B={n_ben}")
            print(f"    AUROC={auc:.3f}, mean LLR diff={mean_diff:.2f}")

            # per-gene in this group
            for g in genes:
                gsub = sub[sub["gene"] == g]
                gn = len(gsub)
                gp = (gsub["target"] == 1).sum()
                gb = gn - gp
                if gp > 0 and gb > 0:
                    gauc = roc_auc_score(gsub["target"], gsub["esm2_llr"])
                    print(f"      {g:<12} n={gn:>4} P={gp:>3}  AUROC={gauc:.3f}")
                elif gn > 0:
                    print(f"      {g:<12} n={gn:>4} P={gp:>3}  (single class)")
        else:
            print(f"\n  {mech_name}: n={n}, P={n_path} — insufficient data")


def plot_protein_landscape(df, disorder_df, sequences, gene, regions):
    """plot ESM2 LLR landscape across protein with functional annotations."""
    gdf = df[df["gene"] == gene]
    seq = sequences.get(gene, "")
    gdis = disorder_df[disorder_df["gene"] == gene].sort_values("position")

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})

    # panel 1: ESM2 LLR by position, colored by pathogenicity
    path_df = gdf[gdf["target"] == 1]
    ben_df = gdf[gdf["target"] == 0]

    axes[0].scatter(ben_df["position"], ben_df["esm2_llr"], c="#4575b4",
                     alpha=0.4, s=20, label=f"benign/VUS (n={len(ben_df)})", zorder=2)
    axes[0].scatter(path_df["position"], path_df["esm2_llr"], c="#d73027",
                     alpha=0.7, s=40, marker="^", label=f"pathogenic (n={len(path_df)})", zorder=3)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("ESM2 LLR\n(+ve = WT preferred)")
    axes[0].set_title(f"{gene} — ESM2 LLR landscape ({len(seq)} aa)")
    axes[0].legend(loc="upper right")

    # annotate notable variants
    if len(path_df) > 0:
        # label the top 3 pathogenic with lowest LLR (most confusing)
        worst = path_df.nsmallest(3, "esm2_llr")
        for _, r in worst.iterrows():
            axes[0].annotate(f"{r['ref_aa']}{int(r['position'])}{r['alt_aa']}",
                              (r["position"], r["esm2_llr"]),
                              fontsize=7, ha="center", va="bottom",
                              color="#d73027", fontweight="bold")

    # panel 2: disorder score
    if len(gdis) > 0:
        axes[1].fill_between(gdis["position"], gdis["disorder_score"],
                              color="#ff7f0e", alpha=0.4)
        axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_ylabel("Disorder\nscore")
        axes[1].set_ylim(0, 1)

    # panel 3: functional regions
    region_colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
    for i, (rname, (start, end)) in enumerate(sorted(regions.items(), key=lambda x: x[1][0])):
        rect = Rectangle((start + 1, 0), end - start, 1,
                           facecolor=region_colors[i], alpha=0.7, edgecolor="black", linewidth=0.5)
        axes[2].add_patch(rect)
        mid = (start + end) / 2 + 1
        axes[2].text(mid, 0.5, rname, ha="center", va="center", fontsize=7,
                      rotation=0 if (end - start) > 40 else 45)
    axes[2].set_xlim(0, len(seq) + 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Regions")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Position")

    plt.tight_layout()
    out = FIGURES / f"deep_dive_{gene.lower()}_landscape.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_mechanism_summary(gene_results):
    """summary plot of per-gene ESM2 performance with mechanism coloring."""
    if len(gene_results) == 0:
        return

    mech_colors = {
        "SNCA": "#d73027", "TTR": "#d73027", "PRNP": "#d73027",  # amyloid
        "FUS": "#f46d43", "TARDBP": "#f46d43", "HNRNPA1": "#f46d43",
        "TIA1": "#f46d43", "HNRNPA2B1": "#f46d43",  # non-amyloid aggregation
        "SOD1": "#4575b4", "VCP": "#4575b4", "LMNA": "#4575b4",
        "CRYAB": "#4575b4",  # loss of function
        "AR": "#91bfdb",  # repeat expansion
        "SQSTM1": "#fee090", "SNCA": "#d73027",  # condensate
    }

    gene_results = gene_results.sort_values("auroc", ascending=True)
    colors = [mech_colors.get(g, "gray") for g in gene_results["gene"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(gene_results["gene"], gene_results["auroc"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="random")
    ax.axvline(0.7, color="green", linestyle="--", alpha=0.3, label="good (0.7)")
    ax.set_xlabel("AUROC (ESM2 LLR)")
    ax.set_title("Per-Gene ESM2 LLR Performance\n(colored by disease mechanism)")
    ax.set_xlim(0, 1)

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d73027", label="Toxic aggregation (amyloid)"),
        Patch(facecolor="#f46d43", label="Toxic aggregation (non-amyloid)"),
        Patch(facecolor="#4575b4", label="Loss of function (structured)"),
        Patch(facecolor="#91bfdb", label="Repeat expansion"),
        Patch(facecolor="#fee090", label="Condensate/phase separation"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # annotate sample sizes
    for i, (_, r) in enumerate(gene_results.iterrows()):
        ax.text(r["auroc"] + 0.01, i, f"n={r['n']}, P={r['n_path']}",
                va="center", fontsize=7)

    plt.tight_layout()
    out = FIGURES / "deep_dive_per_gene_mechanism.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_conservation_disorder_2d(df):
    """2D scatter: conservation (entropy) vs disorder, colored by pathogenicity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # all variants
    ben = df[df["target"] == 0]
    path = df[df["target"] == 1]

    axes[0].scatter(ben["position_disorder"], ben["esm2_entropy"],
                     c="#4575b4", alpha=0.2, s=10, label="benign/VUS")
    axes[0].scatter(path["position_disorder"], path["esm2_entropy"],
                     c="#d73027", alpha=0.5, s=20, label="pathogenic")
    axes[0].set_xlabel("Disorder score")
    axes[0].set_ylabel("ESM2 entropy (higher = less conserved)")
    axes[0].set_title("Conservation vs Disorder")
    axes[0].legend()

    # LLR vs disorder
    axes[1].scatter(ben["position_disorder"], ben["esm2_llr"],
                     c="#4575b4", alpha=0.2, s=10, label="benign/VUS")
    axes[1].scatter(path["position_disorder"], path["esm2_llr"],
                     c="#d73027", alpha=0.5, s=20, label="pathogenic")
    axes[1].set_xlabel("Disorder score")
    axes[1].set_ylabel("ESM2 LLR")
    axes[1].set_title("ESM2 LLR vs Disorder")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    out = FIGURES / "deep_dive_conservation_vs_disorder.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def analyze_idr_conservation_paradox(df):
    """the key question: are pathogenic IDR mutations at positions
    that ESM2 considers UNCONSERVED (low LLR)?

    if yes, this explains the failure: IDR positions are variable by nature,
    so ESM2 can't distinguish pathogenic from benign based on conservation.
    the signal must come from something else (motif disruption, charge pattern, etc.)
    """
    print("\n" + "=" * 70)
    print("THE CONSERVATION PARADOX IN IDRs")
    print("=" * 70)

    idr = df[df["in_idr"] == 1]
    struct = df[df["in_idr"] == 0]

    print(f"\n  IDR variants: {len(idr)} ({(idr['target']==1).sum()} pathogenic)")
    print(f"  Structured variants: {len(struct)} ({(struct['target']==1).sum()} pathogenic)")

    # mean ESM2 metrics in IDR vs structured
    for name, sub in [("IDR", idr), ("Structured", struct)]:
        path_sub = sub[sub["target"] == 1]
        ben_sub = sub[sub["target"] == 0]

        print(f"\n  {name}:")
        print(f"    mean ESM2 entropy:    path={path_sub['esm2_entropy'].mean():.2f}, "
              f"ben={ben_sub['esm2_entropy'].mean():.2f}")
        print(f"    mean ESM2 ref_logprob: path={path_sub['esm2_ref_logprob'].mean():.2f}, "
              f"ben={ben_sub['esm2_ref_logprob'].mean():.2f}")
        print(f"    mean ESM2 LLR:        path={path_sub['esm2_llr'].mean():.2f}, "
              f"ben={ben_sub['esm2_llr'].mean():.2f}")

        if len(path_sub) > 0 and len(ben_sub) > 0:
            # how many pathogenic mutations have LOWER LLR than median benign?
            med_ben_llr = ben_sub["esm2_llr"].median()
            n_below = (path_sub["esm2_llr"] < med_ben_llr).sum()
            print(f"    pathogenic below benign median LLR: "
                  f"{n_below}/{len(path_sub)} ({n_below/len(path_sub)*100:.0f}%)")

    # the paradox test: in IDRs, is the LLR separation between path/ben
    # significantly smaller than in structured regions?
    if ((idr["target"] == 1).sum() > 0 and (idr["target"] == 0).sum() > 0 and
        (struct["target"] == 1).sum() > 0 and (struct["target"] == 0).sum() > 0):

        idr_sep = (idr[idr["target"] == 1]["esm2_llr"].mean() -
                   idr[idr["target"] == 0]["esm2_llr"].mean())
        struct_sep = (struct[struct["target"] == 1]["esm2_llr"].mean() -
                      struct[struct["target"] == 0]["esm2_llr"].mean())

        print(f"\n  LLR separation (path - ben):")
        print(f"    IDR:        {idr_sep:.2f}")
        print(f"    Structured: {struct_sep:.2f}")
        print(f"    ratio:      {idr_sep/struct_sep:.2f}x" if struct_sep != 0 else "")

        idr_auc = roc_auc_score(idr["target"], idr["esm2_llr"])
        struct_auc = roc_auc_score(struct["target"], struct["esm2_llr"])
        print(f"\n  AUROC:")
        print(f"    IDR:        {idr_auc:.3f}")
        print(f"    Structured: {struct_auc:.3f}")

        # what predicts pathogenicity in IDRs if not conservation?
        print(f"\n  in IDRs, what features distinguish path from ben?")
        features_to_check = [
            "esm2_llr", "esm2_entropy", "grantham_distance", "blosum62",
            "hydrophobicity_change", "creates_proline", "destroys_proline",
            "local_p_lock", "local_charge_density", "local_qn_frac",
            "local_aromatic_density",
        ]
        for feat in features_to_check:
            if feat in idr.columns:
                p_val = idr[idr["target"] == 1][feat].mean()
                b_val = idr[idr["target"] == 0][feat].mean()
                diff = p_val - b_val
                # effect size (Cohen's d approximation)
                pooled_std = idr[feat].std()
                d = diff / pooled_std if pooled_std > 0 else 0
                marker = "***" if abs(d) > 0.5 else "**" if abs(d) > 0.3 else "*" if abs(d) > 0.2 else ""
                print(f"    {feat:<30} path={p_val:>7.2f}  ben={b_val:>7.2f}  "
                      f"d={d:>+.2f} {marker}")


def main():
    print("=" * 70)
    print("DEEP DIVE: WHY DOES ESM2 FAIL FOR CERTAIN GENES?")
    print("=" * 70)

    df, disorder_df, sequences = load_all_data()
    print(f"  loaded {len(df)} variants (after filtering conflicting)")

    # 1. detailed per-gene analysis
    gene_results = analyze_per_gene_detail(df)

    # 2. mechanism-based grouping
    mechanism_analysis(df)

    # 3. conservation vs disorder analysis
    conservation_vs_disorder_landscape(df, disorder_df, sequences)

    # 4. the conservation paradox in IDRs
    analyze_idr_conservation_paradox(df)

    # 5. per-gene regional analysis for key failing genes
    all_region_results = []
    for gene in ["FUS", "TARDBP", "LMNA", "SOD1", "TTR", "SNCA", "AR", "VCP", "PRNP"]:
        if gene in PROTEIN_REGIONS:
            results = analyze_regions(df, gene, PROTEIN_REGIONS[gene], sequences)
            all_region_results.extend(results)

    # 6. false positive/negative analysis for failing genes
    for gene in ["FUS", "TARDBP", "LMNA", "VCP", "SOD1"]:
        false_positive_negative_analysis(df, gene)

    # 7. plots
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # protein landscapes for key genes
    for gene in ["FUS", "TARDBP", "LMNA", "SOD1", "SNCA", "TTR", "AR", "VCP", "PRNP"]:
        if gene in PROTEIN_REGIONS:
            plot_protein_landscape(df, disorder_df, sequences, gene, PROTEIN_REGIONS[gene])

    # mechanism summary
    plot_mechanism_summary(gene_results)

    # conservation vs disorder 2D
    plot_conservation_disorder_2d(df)

    # save regional results
    if all_region_results:
        pd.DataFrame(all_region_results).to_csv(
            VARIANTS / "regional_esm2_analysis.csv", index=False)
        print(f"  saved regional analysis to {VARIANTS / 'regional_esm2_analysis.csv'}")

    print("\n  DONE")


if __name__ == "__main__":
    main()
