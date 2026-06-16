# critical evaluation: IDP mechanism classifier manuscript

## date: 2026-03-16
## scope: all manuscript sections, statistical methods, data, figures

---

## 1. executive summary

the manuscript presents a genuine and important finding: conservation-based
pathogenicity predictors (ESM2, AlphaMissense, EVE) systematically fail for
gain-of-toxic-function mutations in intrinsically disordered regions, and
functional region membership rescues prediction. the FUS NLS analysis is
publication-quality. the statistical infrastructure (bootstrap CIs, permutation
tests, boundary sensitivity, LOGO-CV) is thorough.

however, the manuscript had several issues requiring correction before
submission. all critical and high-severity issues have been fixed in the
section drafts (2026-03-16). remaining concerns are structural/interpretive
rather than factual.

**overall verdict: publishable after corrections (now applied), with the
framing shift from "predictor paper" to "diagnostic paper" recommended.**

---

## 2. numerical audit

### 2.1 numbers verified against CSV outputs

all headline numbers in the corrected manuscript now trace to specific cells
in the canonical output files. verification script run on 2026-03-16:

| claim | file | value | match |
|-------|------|-------|-------|
| clean ESM2 overall AUROC | results_two_step.csv | 0.748 | yes |
| clean two-step overall AUROC | results_two_step.csv | 0.873 | yes |
| clean GoF non-amyloid ESM2 | results_two_step_by_mechanism.csv | 0.493 | yes |
| clean GoF non-amyloid two-step | results_two_step_by_mechanism.csv | 0.822 | yes |
| VUS-included ESM2 overall | results_two_step.csv | 0.676 | yes |
| VUS-included two-step overall | results_two_step.csv | 0.766 | yes |
| clean evaluable genes | bootstrap_cis.csv | 14 | yes |
| clean evaluable variants | bootstrap_cis.csv | 653 (567P / 86B) | yes |
| VUS-included evaluable genes | bootstrap_cis.csv | 15 | yes |
| VUS-included evaluable variants | bootstrap_cis.csv | 3,102 (672P) | yes |

### 2.2 numbers corrected (2026-03-16)

**section 5 had mixed eval-set numbers.** the original draft cited overall
two-step AUROC = 0.784 and ESM2 = 0.676. the 0.676 comes from the VUS-included
set; the 0.784 comes from `membership_plus_llr_plus_features` (a 5-feature
model) on VUS-included — not the 2-feature two-step model described in the text.
the actual 2-feature two-step on VUS-included is 0.766. the clean-label two-step
is 0.873.

**corrected**: section 5 now leads with clean-label (0.873 vs 0.748, Δ=+0.125)
and reports VUS-included (0.766 vs 0.676, Δ=+0.090) as explicit sensitivity.

**LoF performance was misreported as "comparable at 0.76 for both methods."**
actual clean-label: ESM2 = 0.813, two-step = 0.650 (degradation). VUS-included:
0.763 vs 0.735 (Δ=−0.028). now disclosed honestly.

---

## 3. statistical methodology assessment

### 3.1 class imbalance — severity: HIGH

the clean-label set has 672 pathogenic vs 106 benign (6.3:1). per-gene
imbalance is extreme:

| gene | pathogenic | benign | AUROC meaningful? |
|------|-----------|--------|-------------------|
| LMNA | 187 | 1 | no (1 benign) |
| SOD1 | 105 | 0 | unevaluable |
| TTR | 109 | 2 | marginal |
| VCP | 44 | 1 | no (1 benign) |
| TARDBP | 18 | 1 | no (1 benign) |
| AR | 139 | 29 | marginal |
| FUS | 18 | 7 | marginal |

the LoF structured mechanism group has 238 clean variants: 233 pathogenic, 5
benign. AUROC estimated from 5 minority-class observations is dominated by the
rank position of those 5 variants. a single misranked benign variant swings the
AUROC by ~0.1.

**consequence**: clean-label per-gene and per-mechanism AUROCs for genes with
<5 benign variants should be treated as illustrative, not inferential. the
VUS-included set provides more stable estimates.

**status**: disclosed in section 1 (new class-imbalance paragraph). sections 2–4
now specify VUS-included for per-gene AUROCs with rationale.

### 3.2 bootstrap CI reliability

| gene | n_boot (of 1000) | CI width | interpretation |
|------|------------------|----------|----------------|
| HNRNPA1 (clean) | 684 | [0.0, 1.0] | zero information |
| CRYAB (clean) | 910 | [1.0, 1.0] | meaningless (2P) |
| TIA1 (clean) | 686 | [0.0, 1.0] | zero information |
| FUS (clean) | 999 | [0.25, 0.74] | wide but informative |
| LMNA (full_vus) | 1000 | [0.79, 0.85] | tight, reliable |

genes where >10% of bootstrap resamples lack both classes produce unreliable CIs.
the 316/1000 failure rate for HNRNPA1 means the "95% CI" is actually computed
from ~684 resamples, all of which happened to include at least 1 of 3 pathogenic
variants — a non-representative subset.

**status**: HNRNPA1 CI corrected from [0.0, 0.9] to [0.0, 1.0] in section 4.

### 3.3 P(two-step > ESM2) = 1.000

reporting a probability as exactly 1.000 from 1,000 bootstrap resamples is
statistically incorrect. you cannot estimate probabilities below 1/n_bootstrap.

**status**: corrected to "P < 0.001" in section 5.

### 3.4 multiple comparisons

the paper tests 4 methods × 5 mechanism groups × 2 eval sets = 40 AUROC
comparisons, plus ~15 genes × 4 methods × 2 sets = 120 per-gene comparisons.
no family-wise correction is applied. the paper's exploratory-to-confirmatory
structure partially mitigates this: the mechanism split (section 1) is the
primary finding; per-gene analyses (sections 2–4) are explicitly secondary.

**assessment**: acceptable given the structure, but should be noted in methods.
the key GoF non-amyloid result (AUROC ≈ 0.50) is so far from the baseline
(0.75) that no correction would change the qualitative conclusion.

### 3.5 LOGO-CV: what is actually cross-validated?

region annotations are hardcoded per gene in script 10 (lines 45–173). when
gene X is held out, the logistic regression trains the *global weight* of
region membership vs ESM2 on other genes, then applies gene X's own hardcoded
region boundaries.

what IS cross-validated:
- the relative importance of region membership vs ESM2 LLR
- the logistic regression intercept and scaling

what is NOT cross-validated:
- the region boundaries themselves
- the binary critical/non-critical designation

**consequence**: the approach cannot generalize to a new gene without
pre-existing functional region annotations.

**status**: disclosed in section 5 limitations (new paragraph).

### 3.6 missing statistical tests

| test | status | recommendation |
|------|--------|----------------|
| DeLong test for paired AUROC comparison | not performed | justified: bootstrap handles LOGO-CV fold structure (noted in section 5) |
| calibration analysis | not performed | would strengthen clinical utility claims; note as future work |
| sensitivity to mechanism group assignment | not performed | moving MAPT or AR between groups could change results; recommend in discussion |
| label noise sensitivity | not performed | recommend in discussion |

---

## 4. per-gene permutation tests

this is where the data is most honest and where reviewers will probe hardest.

| gene | region | region length | % of protein | AUROC | p-value | significant? |
|------|--------|--------------|-------------|-------|---------|-------------|
| FUS | NLS | 25 aa | 4.8% | 0.916 | 0.002 | yes |
| TARDBP | LCD | 140 aa | 33.8% | 0.71 | 0.091 | no |
| HNRNPA1 | PrLD | 186 aa | 50.0% | 0.67 | 0.170 | no |
| **pooled GoF** | — | — | — | 0.82 | <0.001 | **yes** |

the pooled result is robust. the individual-gene results show a clear pattern:
significance scales inversely with region size as a fraction of protein length.
this is expected — a random segment covering 50% of the protein will frequently
overlap with pathogenic variant clusters by chance.

**interpretation**: the permutation test validates that *specific functional
regions*, not random protein segments, drive the signal. FUS is the strongest
case. TARDBP and HNRNPA1 are weaker individually but contribute to the pooled
significance.

**status**: all three p-values now reported honestly in section 5 with
explanation.

---

## 5. study design and bias assessment

### 5.1 selection bias

the 22 genes were selected because they are IDP-associated disease genes with
known mechanisms. this means the "discovery" that conservation fails for GoF
non-amyloid genes is partly a consequence of the selection criterion — these
genes were chosen precisely because their pathogenicity operates through
disordered-region mechanisms.

**recommendation**: frame as "characterizing a known clinical challenge" rather
than "discovering a new phenomenon." the contribution is the quantification
(AUROC = 0.50 for GoF), the three-predictor convergence, and the region-
membership rescue — not the observation that conservation is weak in IDRs.

### 5.2 ClinVar ascertainment bias

ClinVar over-represents variants in well-studied genes and clinically observed
variants. the base rate of pathogenicity in this dataset (~17% overall, ~86% in
clean labels) does not reflect population frequencies. AUROC is independent of
base rate, but precision/recall and clinical utility estimates would be affected.

### 5.3 region annotation circularity

the non-circularity argument (section 5) has four components. assessment:

1. **regions from pre-2026 studies**: STRONG. dormann 2010, kim 2013, johnson
   2009 predate this analysis by >10 years.

2. **TARDBP positional control**: MODERATE. relative position AUROC = 0.66 vs
   LCD membership 0.71 — a modest difference (Δ=0.05) that depends on the
   specific metric and threshold.

3. **permutation test**: MIXED. FUS p=0.002 is strong. TARDBP p=0.091 and
   HNRNPA1 p=0.170 are not individually significant. pooled p<0.001 saves the
   class-level claim.

4. **boundary sensitivity**: MODERATE. range = 0.019 is impressive, but
   boundaries are nested (narrow ⊂ standard ⊂ wide), not independent.

**overall circularity risk**: LOW for FUS, MODERATE for TARDBP/HNRNPA1.
the strongest defense is point 1 (temporal independence of region definitions).

### 5.4 mechanism group assignment

the 5 mechanism groups are assigned from literature knowledge, not derived from
the data. this is defensible but introduces researcher degrees of freedom:

- MAPT (tau) is assigned to "condensate" but has a validated dual
  amyloid/condensate pathway (VALIDATION.md prediction 4). reassignment would
  change condensate group results.

- AR is in "repeat" but its pathogenic mutations target DBD/LBD (structured
  domains), not the polyQ tract. AR may belong in LoF structured.

- APP gets mechanism "unknown" in script 10, still entering LOGO-CV.

**recommendation**: add a sensitivity analysis moving 1-2 borderline genes
between groups (MAPT, AR) and reporting whether the mechanism split holds. this
would cost minimal effort and substantially strengthen the robustness claim.

---

## 6. evidence quality: GRADE-like assessment

### claim 1: "conservation-based prediction is mechanism-dependent"

| criterion | assessment | direction |
|-----------|-----------|-----------|
| study design | observational (retrospective ClinVar analysis) | start LOW |
| risk of bias | mechanism labels post-hoc, ClinVar ascertainment | — |
| inconsistency | low: ESM2, AlphaMissense, EVE all fail for GoF | upgrade ↑ |
| indirectness | low: directly tests variant-level prediction | — |
| imprecision | moderate: small n for GoF genes, wide CIs | downgrade ↓ |
| publication bias | N/A (primary analysis) | — |
| large effect | yes: AUROC drops from 0.75 to 0.50 for GoF | upgrade ↑ |
| **overall** | **MODERATE** | |

### claim 2: "region membership rescues GoF prediction"

| criterion | assessment | direction |
|-----------|-----------|-----------|
| study design | observational | start LOW |
| risk of bias | regions from same literature that identified genes | — |
| inconsistency | high: FUS strong, TARDBP moderate, HNRNPA1 weak | downgrade ↓ |
| indirectness | low | — |
| imprecision | moderate: TARDBP/HNRNPA1 CIs very wide | — |
| large effect | yes for FUS (0.50→0.92) | upgrade ↑ |
| **overall** | **LOW-to-MODERATE** | |

### claim 3: "disorder is not the problem; mechanism is"

| criterion | assessment | direction |
|-----------|-----------|-----------|
| study design | observational | start LOW |
| risk of bias | low | — |
| inconsistency | low: non-monotonic pattern robust | upgrade ↑ |
| dose-response | yes: mechanism×disorder interaction | upgrade ↑ |
| **overall** | **MODERATE** | |

---

## 7. logical structure assessment

### 7.1 strengths

1. **the self-critique (PHASE3_CRITIQUE.md) is exceptional.** correctly
   identified n=16 as underpowered, ESM2 PC1 dominance, mutation invisibility,
   and axis non-orthogonality. this led directly to the superior mutation-level
   analysis. reviewers will see a research group that holds itself to high
   standards.

2. **the experimental validation (VALIDATION.md) is honest.** "3 strong, 1
   supported, 3 partial" with clear documentation of where the framework fails
   (M337V invisible, interaction-dependent effects undetectable). the honest
   conclusion — "this is a sequence composition landscape, not a mechanism
   predictor" — is a model of scientific self-assessment.

3. **the "disorder is not the problem" argument (section 6) is the most
   analytically sophisticated section.** the non-monotonic disorder-AUROC
   pattern, the mechanism×disorder interaction, and the Cohen's d analysis
   collectively make a strong case that is both novel and well-supported.

4. **the masking control (script 09, figure S1) closes a loophole.** by showing
   that true masked marginals (AUROC = 0.408) perform no better than the fast
   approximation (0.417), the paper rules out the hypothesis that the ESM2
   failure is a computational artifact.

5. **HTT exclusion is handled transparently.** no attempt to salvage fabricated
   zero-values for the 170 truncated variants.

### 7.2 weaknesses

1. **the two-step "predictor" framing oversells the contribution.** the
   approach requires pre-existing gene-specific region annotations that cannot
   be learned or transferred to novel genes. calling it a "predictor" implies
   prospective utility that does not exist. the contribution is diagnostic
   (characterizing where conservation fails) rather than predictive (providing
   a tool that works better).

   **recommendation**: reframe as a *diagnostic analysis* that reveals the type
   of information needed, not as a competing clinical tool.

2. **HNRNPA1 gets a full results section (section 4) for n=3 pathogenic
   variants.** bootstrap CI = [0.0, 1.0]. the p_lock AUROC of 0.72 could be
   random noise. this section is already well-caveated (2026-03-16 corrections
   strengthened the caveats further), but its placement as a main-text section
   gives it more weight than the evidence supports.

   **recommendation**: consider moving to supplementary, or merging into a
   shorter paragraph within section 3 (TARDBP) under a "other GoF genes" header.

3. **AlphaMissense's 0.75 AUROC on clean GoF non-amyloid is presented as
   supporting the conservation-failure thesis, but it's actually a meaningful
   counterexample.** a method that achieves 0.75 AUROC is not "failing" in any
   clinical sense. the reframing (2026-03-16 correction) helps, but the tension
   remains: if AlphaMissense works at 0.75, why is the failure "general"?

   the answer (gene-level heterogeneity within the class) is correct but should
   be more prominent. the failure is not that all conservation-based methods fail
   across the board, but that *no single method works reliably across all genes
   in the class*.

4. **the comparison with Farquhar 2026 (AUC = 0.982) is now discussed in
   section 5 limitations.** the key unanswered question is whether Farquhar's
   full-embedding approach also fails for GoF non-amyloid specifically. this
   would either strengthen the paper's thesis (even 1280-dim embeddings have
   the blind spot) or weaken it (the blind spot is an artifact of using LLR
   rather than embeddings). testing this would substantially strengthen the
   manuscript.

### 7.3 missing elements

| element | impact | effort to add |
|---------|--------|---------------|
| sensitivity analysis on mechanism group membership (move MAPT, AR) | moderate | low (re-run script 10 with modified groups) |
| Farquhar-style ESM2 embedding features for GoF genes | high | moderate (extract per-residue embeddings, train classifier) |
| calibration plot (predicted prob vs observed frequency) | low-moderate | low (add to figure 5) |
| per-gene confusion matrices at operating thresholds | low | low (add to supplement) |
| external validation cohort | high | high (requires independent variant set) |

---

## 8. figure assessment

### 8.1 figures that work well

- **figure 1** (mechanism split): clearly communicates the core finding.
  per-gene AUROC bars with mechanism coloring are immediately interpretable.
- **figure 2** (FUS NLS): the strongest figure. protein schematic + LLR
  landscape + charge analysis + ROC curves tell a complete story.
- **figure 6** (disorder is not the problem): the heatmap (panel B) and
  Cohen's d analysis (panel C) are analytically sophisticated.
- **figure S1** (masked marginals): closes the computational-artifact loophole
  cleanly.

### 8.2 figures that need attention

- **figure 4** (HNRNPA1): n=3. the p_lock landscape and D314V analysis are
  interesting but the figure gives more visual weight to HNRNPA1 than the
  evidence supports. consider supplementary.
- **figure 5** (two-step): numbers in the figure panels need to match the
  corrected manuscript text. verify that panel C shows clean-label ROCs and
  panel D shows the correct boundary AUROCs. **the figure script may need
  re-running if it was generated with the old numbers.**

### 8.3 missing figures

- class imbalance visualization (bar chart of P vs B per gene) — would make
  the imbalance tangible for reviewers.
- calibration plot for the two-step predictor.

---

## 9. LoF trade-off: the hidden cost

the two-step predictor degrades LoF structured performance:

| eval set | LoF ESM2 | LoF two-step | Δ |
|----------|---------|-------------|---|
| clean | 0.813 | 0.650 | −0.163 |
| VUS-included | 0.763 | 0.735 | −0.028 |

the clean-label degradation is extreme but unreliable (5 benign variants). the
VUS-included degradation is modest (−0.028) and may not be clinically meaningful.

**mechanism of degradation**: LoF critical regions are too large.
- SOD1: beta barrel = entire 154-aa protein → in_critical_region = 1 for ALL
  variants → zero discriminative power from region membership
- LMNA: rod domain = 353/664 residues (53%) → most variants in-region
- the logistic regression, trained to upweight region membership (β=2.00 vs
  β=0.18 for ESM2), then underweights ESM2 for LoF genes where ESM2 is the
  only useful signal.

**recommendation**: the two-step predictor could be improved by scaling region
annotation informativeness per gene (e.g., by region-to-protein length ratio).
this is beyond the current manuscript scope but worth noting in the discussion.

---

## 10. what the paper gets right

1. **the core finding is real.** ESM2/conservation fails for GoF non-amyloid
   IDR mutations at AUROC ≈ 0.50. this is not an artifact of evaluation
   protocol, small samples, or computational approximation (masked marginals
   confirm).

2. **the three-predictor convergence is convincing.** ESM2, AlphaMissense, and
   EVE all show degraded performance for GoF non-amyloid, spanning sequence
   conservation, structural context, and coevolution.

3. **the FUS NLS story is publication-quality standalone.** 17/18 pathogenic in
   25-aa NLS, charge disruption mechanism, ortholog conservation, masked
   marginal validation, permutation p=0.002.

4. **the self-assessment infrastructure sets a standard.** PHASE3_CRITIQUE.md,
   VALIDATION.md, superseded-script documentation, and the honest-negative
   results in sections 3 and 6 demonstrate intellectual rigor.

5. **the "disorder is not the problem" analysis (section 6) is a genuine
   contribution.** the mechanism×disorder interaction has not been shown this
   clearly elsewhere.

---

## 11. recommended framing for submission

the strongest version of this paper is a **diagnostic paper**:

> "conservation-based variant pathogenicity predictors have a mechanism-specific
> blind spot for gain-of-toxic-function mutations in intrinsically disordered
> regions. we characterize exactly where, why, and for which genes this blind
> spot exists, and show that functional region annotation — not improved
> conservation metrics — is the information needed to rescue prediction."

this framing:
- is fully supported by the data
- does not require the two-step predictor to be a practical clinical tool
- positions against Farquhar 2026 (they optimize prediction; we characterize
  the failure mode that even optimized tools may have)
- makes the FUS NLS analysis the centerpiece rather than the two-step AUROC
- honestly acknowledges that TARDBP and HNRNPA1 provide supporting rather than
  definitive evidence

---

## 12. corrections applied (2026-03-16)

### files modified

| file | changes |
|------|---------|
| `docs/paper/results_section1_draft.md` | class imbalance paragraph; AlphaMissense reframed; eval-set policy stated |
| `docs/paper/results_section2_draft.md` | eval-set note (VUS-included for per-gene, n=168) |
| `docs/paper/results_section3_draft.md` | eval-set note (VUS-included, n=84; clean TARDBP has 1 benign) |
| `docs/paper/results_section4_draft.md` | supplementary-quality framing; CI corrected to [0.0, 1.0]; no-conclusions caveat |
| `docs/paper/results_section5_draft.md` | all numbers fixed to clean-label primary; LoF degradation disclosed; permutation p-values honest; P<0.001; DeLong note; Farquhar 2026 engagement; LOGO-CV generalizability caveat; boundary nesting noted |
| `MANUSCRIPT_STATUS.md` | exact numbers from CSVs; LoF trade-off; permutation significance flags; 7-item caveats checklist |

### verification

all corrected numbers verified against `data/variants/results_two_step.csv`,
`results_two_step_by_mechanism.csv`, and `bootstrap_cis.csv` by automated
cross-check (2026-03-16).

---

## 13. remaining action items

| item | severity | owner | status |
|------|----------|-------|--------|
| re-run figure 5 script with clean-label primary numbers | moderate | human | todo |
| consider moving section 4 (HNRNPA1) to supplementary | moderate | human | decision needed |
| sensitivity analysis on mechanism group assignment | moderate | human/claude | todo |
| test Farquhar 2026 framework on GoF non-amyloid genes | high (impact) | human | optional |
| add class-imbalance bar chart to supplement | low | human/claude | todo |
| finalize title, author list, bibliography | required | human | todo |
| select target journal and write cover letter | required | human | todo |
