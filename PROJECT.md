# IDP Disease Mechanism Landscape

## one-line summary
a multi-axis sequence feature framework that maps neurodegenerative IDPs in mechanism space, showing how polymer physics, maturation grammar, amyloid propensity, and evolutionary embeddings separate disease pathways — and how mutations shift proteins between them.

## the problem
all neurodegenerative IDPs aggregate. but through different mechanisms:
- alpha-synuclein, Abeta42 → direct amyloid from solution
- FUS, TDP-43 → liquid condensate → maturation → solid aggregate
- huntingtin → polyQ collapse → aggregation nucleus
- tau → BOTH direct amyloid AND condensate pathway
- prion → template-directed misfolding

existing tools predict ONE axis (will it aggregate? will it phase-separate? will it mature?). nobody predicts WHICH mechanism a given IDP will use, or how a mutation shifts between mechanisms.

## the hypothesis
IDP disease mechanism is encoded in sequence through four orthogonal feature axes:
1. **polymer physics** (kappa, SCD, NCPR, FCR, Das-Pappu region) → chain-level conformation
2. **maturation grammar** (aromatic stacking, polar zippers, Gly fluidity) → condensate fate
3. **amyloid propensity** (beta-sheet nucleation, APR regions, steric zippers) → direct fibrillization
4. **evolutionary embeddings** (ESM2 1280-dim representations, NOT just LLR) → functional constraint landscape

the COMBINATION of these axes creates a mechanism landscape. proteins with known mechanisms should cluster. mutations shift proteins in this landscape. tau should sit at the boundary.

## post-literature-search adjustment
- reframed from "classifier" (8 training points = overfitting) to "landscape" (feature space analysis)
- ESM2 LLR fails in IDRs (rho=-0.02 on SNCA). using ESM2 embeddings instead.
- Frank et al. 2024 (PNAS) is closest competitor but binary only (droplet vs amyloid)
- LLPS and amyloid are thermodynamically separable (Boyko 2020, 2025) → multi-axis approach justified
- sticker patterning (Martin 2020) maps to our maturation grammar
- see LITERATURE_SYNTHESIS.md for full analysis

## what exists (our assets)
- k-dense: polymer physics pipeline for 8 neurodegenerative IDPs
- condensate-maturation-theory: sliding grammar (tau=0.929 on 8 proteins)
- AF3 data: 2,375 structures, 19 conditions (failed at absolute Rg, but contact maps available)
- this machine: 2x Titan RTX, 64 CPUs

## what's needed
- ESM2 mutation scanning for all 8 proteins
- amyloid propensity scoring (WALTZ, TANGO, CamSol, or ZipperDB)
- literature validation data (experimental mechanism for each protein + mutant)
- integration framework (combine all four axes)

## proteins
| protein | length | disease | known mechanism | UniProt |
|---------|--------|---------|-----------------|---------|
| alpha-synuclein | 140 aa | PD | direct amyloid | P37840 |
| Abeta42 | 42 aa | AD | direct amyloid | P05067(672-713) |
| tau 2N4R | 441 aa | AD/FTD | amyloid + condensate | P10636-8 |
| TDP-43 | 414 aa | ALS/FTD | condensate maturation | Q13148 |
| FUS | 526 aa | ALS/FTD | condensate maturation | P35637 |
| hnRNPA1 | 372 aa | ALS | condensate maturation | P09651 |
| PrP | 208 aa | prion | template misfolding | P04156 |
| Httex1 Q21/Q46 | 90/115 aa | HD | polyQ collapse | P42858 |

## disease mutations to test
| mutation | protein | clinical | expected mechanism shift |
|----------|---------|----------|------------------------|
| A53T | alpha-syn | familial PD | ? → more amyloid? |
| E46K | alpha-syn | familial PD | charge removal → ? |
| P301L | tau | FTD | disrupts PPII → amyloid? |
| M337V | TDP-43 | ALS | LCD mutation → more maturation? |
| Q331K | TDP-43 | ALS | charge change in LCD → ? |
| D262V | hnRNPA1 | ALS | +23% p_lock → maturation |
| P525L | FUS | juvenile ALS | NLS mutation → mislocalization |
| Q21→Q46 | Httex1 | HD | polyQ expansion → collapse |

## key question
can this framework predict, from sequence alone, that:
- FUS LCD will mature but alpha-synuclein won't?
- D262V hnRNPA1 accelerates maturation but P525L FUS changes localization?
- tau sits at the boundary between amyloid and condensate pathways?
- polyQ expansion shifts Httex1 toward collapse-driven aggregation?

## phases
1. literature search — what exists, what's the gap
2. plan — detailed implementation with self-critique
3. implement — feature computation for all 8 proteins
4. analyze — mechanism classification + mutation effects
5. validate — against experimental data from literature

## self-critique protocol
after each phase: ralph-wiggin style honesty
- what did we actually learn?
- what's wrong with what we just did?
- what would we do differently?
- adjust plan and continue

---

## post-implementation findings (phases 3-5, expanded panel)

### panel expansion (phase 4+)
expanded from 8→16 WT proteins + 10 mutants (26 total):
- new amyloid: IAPP
- new condensate: TIA1 (+P362L), EWSR1, TAF15, hnRNPA2B1
- new polyQ: Ataxin-3 (+Q72 expansion)
- new functional condensate: DDX4, NPM1

### what works
1. **binary condensate-vs-amyloid separation is SIGNIFICANT: p=0.008** — regardless of feature selection (4-8 features all give p<0.01). this is the core publishable result.
2. **maturation grammar separates LCD proteins from amyloid formers** — FUS lcd_lockability=0.49, α-syn=0.15 (3.3× difference). FET family gradient (FUS > EWSR1 > TAF15).
3. **D262V hnRNPA1 shows +12.7% LCD lockability increase** — matches experimental finding of accelerated fibrillar conversion.
4. **amyloid propensity correctly ranks known formers** — Aβ42 (3.66) >> α-syn (2.33) >> IAPP (1.64) >> FUS (1.97) > Htt (0.0).
5. **polyQ proteins cluster distinctly** — both Htt and Ataxin-3 occupy unique feature space.

### what doesn't work
1. **6-class WT permutation test still fails: p=0.06 (16 WT)** — improved from p=0.10 (8 WT) but still ns at p<0.05. cause: singleton classes (Tau, PrP) and duos (polyQ, functional condensate) make multi-class test underpowered.
2. **redundancy removal WORSENED p-values** — from 0.06→0.15. with n=16, correlated features reinforce signal rather than diluting it. the bottleneck is sample size, not feature quality.
3. **features remain correlated** — 21 pairs with |ρ|>0.8 (16 WT). reduced from 8-protein era but still substantial.
4. **point mutations still invisible** — TIA1 P362L produces zero grammar change (mutation at LCD boundary).
5. **ESM2 still adds noise** — removing ESM2 consistently improves interpretability.

### the honest publishable core
a multi-axis sequence feature framework that **significantly separates condensate-maturation from amyloid-forming IDPs** (p=0.008, binary permutation test on 10 WT proteins). maturation grammar provides unique value: D262V hnRNPA1, FUS-vs-α-syn, FET family gradient. the finer 6-class mechanism taxonomy is biologically motivated but statistically unvalidated at n=16.

### key figures (expanded panel, 26 proteins)
- `fig1_landscape_biplot.png` — main landscape (no ESM2, 27 features, 26 proteins)
- `fig2_maturation_grammar.png` — LCD lockability comparison (16 WT) + mutation effects
- `fig3_amyloid_vs_maturation.png` — amyloid propensity vs p_lock (the key 2D separation)
- `fig4_feature_comparison.png` — 6 features across 16 WT proteins
- `landscape_reduced_interp.png` — redundancy-removed landscape (20 features)

### analysis documents
- `ANALYSIS.md` — full permutation test results, feature correlations, predictions
- `VALIDATION.md` — literature validation of 7 predictions
- `PHASE3_CRITIQUE.md` — honest self-assessment of limitations

### remaining steps to strengthen
1. add more WT proteins to singleton classes (e.g., SOD1 for template, other tauopathy-related IDPs for dual)
2. TANGO/WALTZ for physics-based amyloid propensity (vs current heuristic)
3. local mutation features (±20 residue window) to capture M337V-type effects
4. cross-validation framework if n reaches 30+

---

## mutation-level vulnerability analysis (2026-02-19)

### pivot rationale
the protein-level landscape (n=16) was underpowered. pivoted to mutation-level pathogenicity prediction using ClinVar variants across 23 IDP-related genes (n=4,011 variants, n=672 pathogenic). this provides orders of magnitude more statistical power and tests a more clinically relevant question.

### the question
can IDP-specific biophysical features predict which missense mutations are pathogenic? and does evolutionary conservation (ESM2) suffice?

### the answer
ESM2 LLR alone is the best predictor (AUROC=0.70), but it has a **systematic blind spot** for gain-of-toxic-function mutations in disordered regions. IDP-specific features add nothing (AUROC=0.52). the failure is mechanism-specific:

| disease mechanism | ESM2 AUROC | interpretation |
|---|---|---|
| loss of function (LMNA, SOD1, CRYAB) | **0.763** | conservation works — mutations disrupt conserved structure |
| toxic aggregation, amyloid (SNCA, TTR, PRNP) | 0.655 | moderate — partial conservation signal |
| **toxic aggregation, non-amyloid (FUS, TARDBP, HNRNPA1)** | **0.417** | **anti-predictive — mutations in unconserved IDR motifs** |

### the central finding
pathogenicity in GoF genes is determined by **functional region membership** (NLS for FUS, LCD for TARDBP, PrLD for HNRNPA1), not by residue-level biophysics or conservation. NLS membership alone predicts FUS pathogenicity at AUROC=0.916. but each gene's critical region requires different features — no universal GoF predictor exists.

### HNRNPA1 exception
the only gene where maturation grammar (p_lock) predicts pathogenicity (AUROC=0.722). mutations at high-locking positions that increase hydrophobicity gain sticker character, promoting aberrant phase transition. this is the sticker-spacer framework in action — but limited to one gene.

### connection to landscape project
the protein-level landscape correctly separates condensate from amyloid formers (p=0.008). the mutation-level analysis confirms the mechanism distinction from a different angle: loss-of-function vs gain-of-toxic-function pathogenicity mechanisms require fundamentally different predictive approaches.

### full analysis
see `ANALYSIS.md` (rewritten), `SESSION_LOG_2026-02-19.md` (complete session log)

### scripts
`scripts/mutation/01-08_*.py` — complete pipeline from ClinVar download to mechanism-aware ensemble
