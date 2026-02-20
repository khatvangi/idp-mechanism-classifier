# IDP Mutation Vulnerability Analysis

## executive summary

we tested whether IDP-specific biophysical features (condensate grammar, amyloid propensity, disorder profiles, local composition) and evolutionary conservation (ESM2 protein language model) can predict which missense mutations in IDP-related genes are pathogenic. three approaches were tested on 4,011 ClinVar variants across 23 genes using leave-one-gene-out cross-validation (LOGO-CV).

**the answer:** conservation (ESM2 LLR) is the only useful signal (AUROC=0.70), but it has a systematic blind spot for gain-of-toxic-function mutations in disordered regions (AUROC=0.42 for FUS/TARDBP/HNRNPA1/TIA1). IDP-specific biophysical features add nothing (AUROC=0.52). pathogenicity in GoF genes is determined by motif-level functional region membership (NLS, LCD, PrLD), not by residue-level biophysics.

---

## 1. data and methods

### 1.1 variant dataset

- **source:** ClinVar variant_summary.txt.gz (Feb 2026)
- **genes:** 23 IDP-related genes (SNCA, APP, MAPT, TARDBP, FUS, HNRNPA1, PRNP, HTT, TIA1, EWSR1, TAF15, HNRNPA2B1, ATXN3, DDX4, NPM1, IAPP, SOD1, TTR, LMNA, VCP, AR, SQSTM1, CRYAB)
- **variant type:** missense only, ≥1 star review status
- **position validation:** cross-validated against UniProt canonical sequences (94.7% match rate; MAPT 17%, HNRNPA2B1 14% due to isoform mismatches)
- **final dataset:** 4,011 validated variants → 3,664 after excluding conflicting
- **labels:** 672 pathogenic (P + LP), 2,992 benign/VUS proxy (B + LB + VUS)

### 1.2 feature sets

**handcrafted IDP features (27 numeric + 3 categorical):**
- local window (±15 residues): hydrophobicity, charge density, aromatic density, glycine/proline fractions, β-propensity, Q/N fraction
- disorder: per-residue metapredict score, local disorder, in_idr flag
- maturation grammar: per-position p_lock, p_pi, p_polar (from SequenceGrammar)
- mutation-specific: BLOSUM62, Grantham distance, charge/hydrophobicity/size changes, proline creation/destruction
- protein-level: length, total disorder fraction

**ESM2 features (6):**
- esm2_llr: log P(ref) - log P(alt) at variant position
- esm2_ref_logprob, esm2_alt_logprob: individual residue log-probs
- esm2_entropy: position-level entropy across 20 amino acids
- esm2_rank_ref, esm2_rank_alt: rank of WT/mutant residue

### 1.3 evaluation

- **cross-validation:** leave-one-gene-out (LOGO-CV) — train on 22 genes, test on held-out gene. prevents gene-level information leakage.
- **classifier:** XGBoost (200 trees, depth 4, class-weighted)
- **metrics:** AUROC (primary), AUPRC (secondary)
- **baseline:** ESM2 LLR as raw score (no ML)

---

## 2. results

### 2.1 approach comparison

| approach | AUROC | AUPRC | notes |
|----------|-------|-------|-------|
| **ESM2 LLR (raw, no ML)** | **0.696** | **0.337** | single number, no training |
| ESM2 features (XGBoost) | 0.600 | 0.238 | ML hurts — overfits on small folds |
| handcrafted IDP features | 0.518 | 0.182 | essentially random |
| combined (IDP + ESM2) | 0.557 | 0.200 | noisy features dilute ESM2 |
| mechanism-aware ensemble | **0.738** | **0.395** | routes by mechanism |
| random baseline | 0.500 | 0.183 | |

**key findings:**
1. raw ESM2 LLR beats all ML models — the additional features are noise
2. handcrafted IDP features (grammar, composition, disorder) contribute nothing
3. a mechanism-aware ensemble gains +4% over ESM2 alone

### 2.2 the mechanism split

| disease mechanism | genes | ESM2 LLR AUROC | mean LLR Δ |
|-------------------|-------|----------------|------------|
| loss of function (structured) | LMNA, SOD1, CRYAB, VCP | **0.763** | **+2.98** |
| repeat expansion | AR, HTT, ATXN3 | 0.745 | +2.71 |
| toxic aggregation (amyloid) | SNCA, TTR, PRNP, IAPP | 0.655 | +1.49 |
| phase separation / condensate | DDX4, NPM1, SQSTM1, MAPT | 0.802 | +3.97 |
| **toxic aggregation (non-amyloid)** | **FUS, TARDBP, HNRNPA1, TIA1** | **0.417** | **-0.87** |

the non-amyloid toxic aggregation group has **negative** LLR difference — pathogenic mutations at these genes occur at positions ESM2 considers MORE variable than benign sites. conservation is anti-predictive.

### 2.3 per-gene ESM2 LLR performance

genes where ESM2 works (AUROC > 0.6):
- LMNA: 0.822 (n=907, P=187) — coiled-coil LoF
- CRYAB: 0.797 (n=160, P=2) — small heat shock protein LoF
- SQSTM1: 0.794 (n=354, P=4) — autophagy receptor
- SOD1: 0.673 (n=154, P=105) — enzyme LoF
- TTR: 0.669 (n=195, P=109) — amyloid (transthyretin)
- AR: 0.635 (n=312, P=139) — repeat expansion/LoF
- SNCA: 0.621 (n=33, P=5) — amyloid (synucleinopathy)

genes where ESM2 fails (AUROC < 0.5):
- **FUS: 0.417** (n=168, P=18) — NLS mutations
- **TARDBP: 0.412** (n=84, P=18) — LCD mutations
- **HNRNPA1: 0.324** (n=39, P=3) — PrLD mutations
- **TIA1: 0.067** (n=121, P=1) — single pathogenic variant

---

## 3. deep dive: why ESM2 fails for GoF genes

### 3.1 FUS — the NLS smoking gun

17 of 18 pathogenic FUS mutations cluster in the PY-NLS (nuclear localization signal, residues 501-526). this 26-residue motif is in a fully disordered C-terminal region (disorder > 0.90).

**NLS membership alone predicts FUS pathogenicity: AUROC = 0.916**

ESM2 fails because:
- the NLS is disordered → ESM2 sees positions as variable
- the pathogenic mechanism is **mislocalization** (disrupting transportin-1 binding) not structural destabilization
- specific R/K/P/Y pattern matters for function but individual positions aren't strongly conserved

false negatives (pathogenic, low LLR):
- Q519R (LLR=2.48), R514G (3.59), G509D (3.64), K510E (3.67)
- all in NLS, all disrupt basic charge pattern needed for transportin-1

false positives (benign/VUS, high LLR):
- G230C (LLR=15.27), G246C (14.15), G255C (13.93)
- all G→C in RGG regions — ESM2 thinks glycine is sacred, but function is preserved

per-gene discriminative features:
- local_charge_density: AUROC=0.944 (NLS is R/K-rich)
- relative_position: AUROC=0.934 (NLS is extreme C-terminus)
- hydrophobicity_change: AUROC=0.620

### 3.2 TARDBP — the LCD pattern

18 of 18 pathogenic TARDBP mutations in the glycine-rich LCD (residues 274-414). all in a fully disordered region (disorder 0.86-0.99).

**LCD membership predicts TARDBP pathogenicity: AUROC = 0.705**

ESM2 fails because:
- LCD is intrinsically low-complexity → no strong conservation signal
- pathogenic mechanism is **altered phase separation** → gain-of-toxic-function
- mutations like I383V (LLR=2.06) and N352S (LLR=2.68) are conservative substitutions that subtly alter aggregation propensity

per-gene discriminative features:
- local_qn_frac: AUROC=0.711 (LCD is Q/N-enriched)
- position_disorder: AUROC=0.687

### 3.3 HNRNPA1 — the exception

only 3 pathogenic variants (D262V, P288A, one other) but they show a unique pattern:
- **local_p_lock: AUROC=0.722** — the maturation grammar locking probability
- **hydrophobicity_change: AUROC=0.704** — mutations increase hydrophobicity
- **esm2_entropy: AUROC=0.880** — pathogenic at high-entropy (unconserved) positions

this is the ONLY gene where IDP-specific biophysics (sticker-spacer framework) predicts pathogenicity. mutations at high p_lock positions that increase hydrophobicity gain sticker character, promoting aberrant phase transition.

### 3.4 the fundamental problem

each GoF gene has different critical regions and different discriminative features:

| gene | critical region | best feature | AUROC |
|------|----------------|--------------|-------|
| FUS | NLS (501-526) | local_charge_density | 0.944 |
| TARDBP | LCD (274-414) | local_qn_frac | 0.711 |
| HNRNPA1 | PrLD (185-372) | local_p_lock | 0.722 |

no universal feature works across all three. the pathogenicity signal is at the motif/region level, and each gene's functional architecture is different.

---

## 4. conservation vs disorder landscape

### 4.1 disorder level and ESM2 performance

| disorder level | n | pathogenic | AUROC | meaning |
|----------------|---|------------|-------|---------|
| ordered (<0.3) | 1842 | 497 | 0.664 | conservation works moderately |
| **boundary (0.3-0.5)** | **320** | **79** | **0.817** | **best: threshold positions are conserved** |
| moderate IDR (0.5-0.8) | 334 | 31 | 0.760 | still decent |
| strong IDR (>0.8) | 1168 | 65 | 0.566 | weak: IDR positions are variable |

the **order-disorder boundary** is the sweet spot for conservation-based prediction. these positions can flip between ordered and disordered states; evolutionary pressure keeps them at the boundary, making pathogenic disruptions detectable.

### 4.2 IDR conservation paradox

in IDRs:
- mean LLR separation (path - ben): **1.45**
- in structured regions: **2.31**
- ratio: 0.63× (37% weaker signal in IDRs)

pathogenic IDR mutations tend to:
- increase hydrophobicity (Cohen's d = +0.22)
- occur in charged local context (d = +0.25)
- be at lower-entropy positions (d = -0.39) — somewhat conserved within IDRs

but these effect sizes are too small to be practically useful.

---

## 5. mechanism-aware ensemble

routing predictions by disease mechanism:
- LoF / amyloid / repeat / condensate genes → raw ESM2 LLR
- GoF non-amyloid genes → XGBoost with IDP + ESM2 features

**result: AUROC = 0.738 (vs 0.696 baseline)**

this 4% gain comes from slightly better handling of amyloid genes with the XGBoost model. GoF non-amyloid remains at 0.457 — the mechanism-specific models don't truly fix the problem, they just don't drag down the LoF predictions.

---

## 6. implications

### 6.1 for variant interpretation tools

existing conservation-based tools (CADD, PolyPhen-2, EVE, ESM2) have a **systematic blind spot** for gain-of-toxic-function mutations in intrinsically disordered regions. this affects a specific but clinically important set of genes: FUS, TARDBP, HNRNPA1, TIA1 — all linked to ALS/FTD.

recommendation: variant interpretation for these genes should not rely solely on conservation scores. functional region annotation (is the mutation in the NLS? in the LCD? in the PrLD?) provides stronger signal.

### 6.2 for IDP biophysics

the sticker-spacer framework and condensate grammar do not predict pathogenicity as single-residue features. the exception is HNRNPA1, where p_lock (locking probability at sticker positions) correlates with pathogenicity (AUROC=0.722). this suggests the grammar captures something real about the sticker landscape of prion-like domains, but only in specific sequence contexts.

### 6.3 for the maturation project

the maturation grammar's predictive value appears limited to proteins with classic PrLD architecture (HNRNPA1). for FUS, the pathogenic mechanism is NLS disruption (mislocalization), not altered maturation. for TARDBP, it's LCD aggregation propensity. the grammar is a condensate maturation tool, not a general pathogenicity predictor.

### 6.4 for a future paper

the publishable finding is the mechanism split itself:

> "conservation-based pathogenicity predictors systematically fail for gain-of-toxic-function mutations in intrinsically disordered regions"

this is a negative result with positive implications — it explains why clinical variant interpretation is unreliable for ALS/FTD genes and points toward motif-level annotation as the solution.

---

## 7. what the data does NOT support

1. ~~IDP biophysical features predict pathogenicity~~ — they don't (AUROC=0.52)
2. ~~maturation grammar predicts pathogenicity~~ — only for HNRNPA1, not generally
3. ~~ESM2 is universally predictive~~ — fails for GoF genes (AUROC=0.42)
4. ~~combining IDP + ESM2 features helps~~ — it hurts (noise dilution)
5. ~~IDR mutations are harder to predict in general~~ — not true. boundary mutations (disorder 0.3-0.5) are the EASIEST to predict (AUROC=0.82). the failure is mechanism-specific, not disorder-general.

---

## 8. connection to original landscape project

the original protein-level mechanism landscape (16 WT proteins, 6 disease classes) failed at p=0.0597 (not significant). the binary condensate-vs-amyloid test worked (p=0.008).

the mutation-level analysis explains why:
- proteins in the same disease class (e.g., FUS and TARDBP are both ALS/FTD) have mutations at DIFFERENT types of functional regions (NLS vs LCD)
- the landscape correctly separates condensate-forming from amyloid-forming proteins as groups, but cannot resolve within-mechanism diversity at the protein level (n=16 is too small)
- the mutation data confirms the landscape intuition — different mechanisms, different features — but at 4,011 variants instead of 16 proteins

---

## 9. open questions

1. **can motif-level annotation rescue GoF prediction?** if we label NLS/LCD/PrLD regions in each gene, does "mutation in critical region" + ESM2 LLR become a strong predictor?
2. **does true masked marginal ESM2 improve over unmasked logits?** our LLR uses the fast approximation. per-position masked inference (O(n) forward passes) would give true marginals and likely improve AUROC.
3. **is HNRNPA1's p_lock signal real or noise?** only 3 pathogenic variants — n is too small for confidence.
4. **what about charge patterning?** FUS NLS pathogenicity is about disrupting a charge pattern (R/K residues). can sequence-level charge pattern features (κ, SCD from Das-Pappu) predict NLS vulnerability?
5. **can we expand to more genes?** VUS-as-benign is noisy. more genes with well-characterized pathogenic mutations would strengthen the mechanism split finding.
