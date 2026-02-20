# phase 3 self-critique — ralph-wiggin honesty

## what we built
a 37-feature landscape (27 interpretable + 10 ESM2 PCA components) mapping 16 IDP sequences (8 WT + 8 mutants) in mechanism space. PCA shows PC1 (31.7%) + PC2 (22.5%) = 54.2% variance explained. five mechanism classes separate visually in PC1-PC2 space.

## what actually worked

### 1. condensate maturation grammar differentiates LCD proteins
the strongest result. FUS lcd_mean_p_lock=0.493, TDP-43=0.417, hnRNPA1=0.465 — all high lockability. α-synuclein lcd_mean_p_lock=0.173, Aβ42=0 (no LCD). 3× difference between condensate and amyloid proteins. this is not trivial — these scores come from amino acid pairwise interaction potentials, not from the label.

### 2. D262V hnRNPA1 increases maturation potential
full_lockability: 876 → 974 (+11.2%). lcd_lockability_per_pair: 0.348 → 0.392 (+12.7%). this matches Mackenzie et al. 2017 — D262V accelerates fibrillar conversion. the grammar detects this from sequence alone.

### 3. polyQ expansion produces distinct signature
Htt Q21→Q46: qn_frac 0.30→0.45, proline_frac 0.34→0.27, GRAVY -1.54→-1.97. massive shift in PCA space. correctly isolated from all other mechanisms.

### 4. amyloid proteins have highest APR scores
Aβ42: max_apr_score=3.66, apr_fraction=0.60. α-syn: max_apr_score=2.33, apr_fraction=0.46. FUS: max_apr_score=1.97, apr_fraction=0.06. Htt: max_apr_score=0.0, apr_fraction=0.0. ranking matches known amyloid propensity literature.

---

## what's WRONG (and this matters more)

### 1. n=16 is fundamentally underpowered
37 features, 16 data points. PCA will ALWAYS find projections that separate 5 classes with 8 unique WT proteins. this is not a validation — it's a mathematical guarantee. any random 37 features would likely produce visual separation with n=16. the landscape CANNOT fail.

**severity: CRITICAL**. this undermines every "separation" claim. we need >30 WT proteins minimum for any statistical claim.

### 2. ESM2 PC1 dominates the landscape
PCA loadings show esm2_pc1 has the highest absolute loading on PC1 (0.281). Htt proteins are extreme ESM2 outliers because polyQ repeats produce distinctive embeddings. the landscape is partially just "how weird does ESM2 think this sequence is?" — which is sequence composition by another name.

**severity: HIGH**. ESM2 embeddings encode general protein properties (fold type, function), not mechanism-specific features. the 10 PCA components add 10 features that may just be capturing length + composition effects we already have from polymer physics.

### 3. mutation effects are microscopic for point mutations
single AA changes in 400+ residue proteins produce near-zero feature deltas:
- TDP-43 M337V: Δkappa=0.000, Δp_lock=0.000, Δamyloid_propensity=+0.013
- FUS P525L: Δkappa=0.000, Δp_lock=0.000 (NLS mutation = invisible to all our features)
- τ P301L: Δkappa=0.000, Δp_lock=0.000, Δamyloid_propensity=+0.011

mutation arrows in PCA are essentially invisible. the framework has NO sensitivity to point mutations except through ESM2 LLR (which fails in IDRs anyway) and crude composition changes.

**severity: HIGH**. we claimed mutations "shift proteins between mechanisms" but most mutations produce negligible shifts.

### 4. maturation grammar is blind to point mutations
the window-based grammar (window=12) averages over local neighborhoods. one residue change in a 12-residue window barely perturbs the mean. TDP-43 M337V has IDENTICAL grammar scores to WT. this means the maturation grammar axis provides zero mutation resolution for the most clinically relevant cases.

**severity: MODERATE**. the grammar correctly separates WT proteins but cannot detect mutation effects.

### 5. the "4 orthogonal axes" aren't orthogonal
- aromatic_frac correlates with maturation grammar (p_pi) — aromatics ARE stickers
- GRAVY correlates with amyloid propensity — hydrophobicity drives both
- glycine_frac correlates with p_lock — glycine reduces lockability
these correlations mean the axes are not independent. we're partially double-counting.

**severity: MODERATE**. inflates apparent information content.

### 6. class imbalance: condensate_maturation has 7 members, others have 1-4
condensate maturation: 7 proteins (3 TDP-43 + 2 FUS + 2 hnRNPA1)
amyloid: 4 proteins (3 α-syn + 1 Aβ42)
dual: 2 (tau WT + P301L)
template_misfolding: 1 (PrP)
polyQ_collapse: 2 (Htt Q21 + Q46)

n=1 for template misfolding means PrP's position is uninterpretable. we cannot validate template misfolding as a distinct mechanism. and the mutants aren't independent — they're 1-AA variants of their WT.

**severity: HIGH**. effectively we have 8 independent data points, not 16.

### 7. FUS is an outlier within its own class
FUS sits at PC2≈-5, far below TDP-43 (PC2≈+0.5) and hnRNPA1 (PC2≈-2). the "condensate maturation" class is not tight. FUS has extreme glycine content (0.289 vs TDP-43's 0.133) and different LCD architecture (QGSY-rich vs glycine-rich). calling these the same mechanism class may be wrong — FUS and TDP-43 have different LCD architectures and different maturation kinetics.

**severity: MODERATE**. suggests condensate_maturation should be subtyped.

### 8. tau's "boundary" position is weak
tau sits at PC1=2.5, PC2=0.8. amyloid centroid is roughly (0, 4). condensate centroid is roughly (-2, -1). tau is NOT equidistant — it's closer to neither and shifted rightward. the "boundary" interpretation is subjective. tau is simply different from both, not "between" them.

**severity: MODERATE**. the tau prediction is the weakest of the four.

---

## what we'd do differently

1. **expand to 25-30 WT proteins** — add SOD1, IAPP, TIA1, DDX4, LAF1, EWSR1, Ataxin-3, p53, NPM1, CPEB4, G3BP1, hnRNPA2. this is the single most important improvement.

2. **remove ESM2 from the main PCA** — run the landscape with only the 27 interpretable features. ESM2 is likely redundant given the composition features we already have. compare the two landscapes.

3. **compute per-residue mutation effects** — instead of whole-protein averages, compute features in a local window (±20 residues) around each mutation site. this would rescue mutation sensitivity.

4. **use TANGO/WALTZ web API** — our APR heuristic is crude. real amyloid propensity tools would give better separation.

5. **statistical test** — permutation test: shuffle mechanism labels 10,000 times, compute inter-class distances each time. are the real distances significantly larger? this is the minimum bar for publishability.

6. **check feature correlations** — compute Pearson/Spearman matrix between all 37 features. remove redundant features (|r| > 0.8) before PCA.

---

## bottom line
the landscape produces visually appealing separation, but with n=16 and 37 features, this is expected by chance. the strongest genuine signal is the maturation grammar separating LCD proteins from amyloid formers. the weakest signal is mutation effects, where most point mutations are invisible. before calling this publishable, we need (a) more proteins, (b) permutation testing, and (c) local mutation feature computation.
