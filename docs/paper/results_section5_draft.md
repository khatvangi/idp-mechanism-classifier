# Results Section 5: The Two-Step Predictor

## Functional Region Annotation Rescues Conservation-Based Prediction

The preceding three sections established that conservation fails for GoF
non-amyloid genes because pathogenicity is a property of functional regions, not
individual residues. A natural question is whether combining functional region
annotation with conservation can rescue overall prediction. We tested this with
a two-step predictor: (1) annotate each variant's membership in a critical
functional region (NLS, LCD, PrLD, conserved helix, or other literature-defined
motif; Table S1), then (2) train a logistic regression using region membership
and ESM2 LLR as features, evaluated by leave-one-gene-out cross-validation
(LOGO-CV).

On the clean-label dataset (14 evaluable genes with both confirmed pathogenic and
benign variants; 653 variants: 567 pathogenic, 86 benign), the two-step
predictor achieves AUROC = 0.873, compared to 0.748 for ESM2 LLR alone — a gain
of +0.125 overall (Figure 5C). The improvement is concentrated in the GoF
non-amyloid group: ΔAUROC = +0.329 (0.493 to 0.822; n = 55, P = 41). In the
larger VUS-included sensitivity analysis (15 evaluable genes, 3,102 variants,
672 pathogenic), the qualitative pattern holds: ESM2 LLR = 0.676, two-step =
0.766, ΔAUROC = +0.090. Bootstrap analysis across 1,000 resamples yields
P(two-step > ESM2) < 0.001 for the GoF non-amyloid group in both evaluation
sets.

For LoF structured genes, the two-step predictor does not improve — and slightly
degrades — performance. Clean-label LoF estimates are unreliable (n = 238, only
5 benign variants; ESM2 AUROC = 0.81, two-step = 0.65), but the more stable
VUS-included set (n = 1,487, 338 pathogenic) shows a modest gap: ESM2 = 0.76,
two-step = 0.74 (Δ = −0.028). The degradation occurs because annotated critical
regions for LoF genes (e.g., the SOD1 beta barrel spans the entire 154-residue
protein) cover most of the protein length, providing no discriminative signal.
Region membership helps where conservation fails (GoF non-amyloid) without
substantially degrading performance where conservation works (LoF), though the
trade-off should be acknowledged.

Per-gene decomposition reveals the source of the GoF gain (Figure 5B). FUS shows
the largest improvement, driven by the NLS annotation. TARDBP gains from LCD
membership. TIA1 gains substantially, though this is based on a single
pathogenic variant and should not be interpreted quantitatively. For LoF genes
where ESM2 already performs well (LMNA, SOD1), region annotation provides no
benefit and in some cases slightly reduces performance, as expected — the region
annotations for these genes are less discriminative than conservation alone.

The logistic regression coefficients quantify the relative contribution of each
feature: β_region = 2.00 [bootstrap 95% CI: 1.79, 2.23] vs β_ESM2 = 0.18
[0.14, 0.21], computed on standardized features. Region membership contributes
approximately 11× more than ESM2 LLR to the log-odds of pathogenicity,
confirming that functional region annotation is the dominant predictor and ESM2
provides only marginal refinement.

### Non-circularity of the region annotations

A critical methodological concern is whether the functional region annotations
are circular — that is, whether they were defined using the same ClinVar variant
data being predicted. Four lines of evidence argue against circularity.

First, the region annotations are derived from published functional studies that
precede our analysis: the FUS PY-NLS from Dormann et al. (2010) and Kwiatkowski
et al. (2009), the TARDBP LCD from Johnson et al. (2009), the HNRNPA1 PrLD from
Kim et al. (2013), and other regions from similarly independent sources (Table
S1). These studies defined functional regions based on biochemical assays
(transportin-1 binding, phase separation, aggregation), not variant
pathogenicity.

Second, TARDBP provides a direct test of the positional-proxy concern. If region
annotations were merely labeling "the end of the protein where pathogenic
variants cluster," any positional feature would work equally well. But the
TARDBP LCD spans residues 275–414, occupying the middle-to-C-terminal portion
of a 414-residue protein — not an extreme terminus. A simple relative position
feature achieves AUROC = 0.66 for TARDBP, substantially below LCD membership
(0.71). The LCD annotation captures functional biology (aggregation-prone
composition), not positional clustering.

Third, a permutation test comparing the annotated region AUROC to 10,000
length-matched random segments per gene provides gene-level significance
estimates (Figure S5). For FUS, the NLS AUROC (0.916) exceeds 99.8% of random
segments (p = 0.002). However, individual-gene significance is limited for the
other GoF genes: TARDBP's LCD shows p = 0.091, consistent with the LCD spanning
34% of the protein where length-matched random segments can partially approximate
the true region; HNRNPA1's PrLD shows p = 0.170, reflecting both the large
region size (51% of the protein) and the minimal pathogenic sample (n = 3). The
pooled GoF non-amyloid result, which independently samples random segments per
gene, remains significant (p < 0.001), indicating that the pattern is robust as
a class-level observation even where individual-gene significance is limited by
sample size and region extent.

Fourth, boundary sensitivity analysis shows that the GoF non-amyloid AUROC is
stable across narrow (±10 residue contraction), standard, and wide (±20 residue
expansion) region definitions: 0.850, 0.869, and 0.857, respectively (range =
0.019; Figure 5D). These boundaries are nested (narrow ⊂ standard ⊂ wide), so
the three estimates are not independent; a more stringent test would shift
boundaries laterally rather than expand/contract them. Nonetheless, if the
annotations were overfit to exact positions of known pathogenic variants, narrow
boundaries should substantially outperform wide ones. The stability indicates
that the signal comes from being inside the functional region, not from
fine-tuned boundary placement.

### Limitations

The two-step predictor requires curated region annotations from the literature.
For well-studied genes (FUS, TARDBP, HNRNPA1), these annotations exist and are
functionally grounded. For newly discovered IDP disease genes, such annotations
may not be available, limiting the approach's generalizability. Critically, the
LOGO-CV evaluates whether the *global weight* of region membership vs ESM2
generalizes across genes, but the region definitions themselves are gene-specific
and hardcoded — they are not learned or cross-validated. The approach therefore
cannot be applied to a novel gene without pre-existing functional annotation.
The predictor should be understood as demonstrating the *type* of information
needed — functional region membership — rather than as a fully automated
clinical tool.

The clean-label evaluation set is heavily imbalanced (567 pathogenic vs 86
benign across evaluable genes), which limits the precision of per-mechanism
AUROC estimates, particularly for LoF structured genes where only 5 confirmed
benign variants are available. Clean-label results should be interpreted
alongside the more balanced VUS-included sensitivity analysis.

Statistical comparisons between methods use bootstrap resampling (1,000
resamples) rather than the DeLong test (DeLong et al. 1988) for paired AUROC
comparison, because bootstrap naturally handles the LOGO-CV structure where
predictions are generated across multiple folds. The qualitative conclusions —
that region membership improves GoF non-amyloid prediction while modestly
trading LoF performance — are consistent across both evaluation sets.

### Relationship to high-performance IDR predictors

Farquhar 2026 (Research Square preprint) reports AUC = 0.982 for pathogenicity
prediction in IDRs by combining ESM2 embeddings (1280-dim, contributing 83.7%
of feature importance) with AlphaMissense and additional sequence features. Our
ESM2 LLR baseline uses only a single scalar (the log-likelihood ratio), not the
full 1280-dim embedding, explaining much of the performance gap. Furthermore,
Farquhar may use within-gene evaluation splits, which would inflate performance
relative to our more conservative LOGO-CV that prevents gene-specific patterns
from entering training. Importantly, even high-performing tools may still exhibit
mechanism-specific blind spots for GoF non-amyloid genes — a hypothesis that has
not been tested in the Farquhar framework. Our contribution is not a competing
predictor but a diagnostic characterization of *where and why* conservation-based
approaches fail, and what type of information (functional region context) is
needed to rescue them.

This is consistent with the paper's central message: residue-level conservation
is the wrong level of analysis for GoF non-amyloid mutations, and the right
level is the functional motif.
