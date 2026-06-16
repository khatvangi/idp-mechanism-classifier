# Conservation-Based Pathogenicity Predictors Systematically Fail for Gain-of-Function Mutations in Intrinsically Disordered Protein Regions

**Boggavarapu Kiran**

Department of Chemistry and Physics, McNeese State University, Lake Charles, LA 70609

Email: kiran@mcneese.edu

---

## Abstract

Computational pathogenicity predictors trained on evolutionary conservation are widely used to interpret missense variants in disease genes, yet their performance on mutations in intrinsically disordered regions (IDRs) has not been evaluated with respect to disease mechanism. Here we analyze 3,409 ClinVar missense variants across 22 genes linked to protein misfolding diseases, stratified by mechanism: loss of function in structured domains (LoF), gain of toxic function through amyloid formation (GoF amyloid), and gain of toxic function through non-amyloid mechanisms (GoF non-amyloid). ESM2, AlphaMissense, and EVE all achieve reasonable discrimination for LoF mutations (group AUROC = 0.76) but systematically fail for GoF non-amyloid mutations (group AUROC = 0.42, 0.40 excluding FUS, and 0.34 respectively)---assigning *lower* pathogenicity scores to pathogenic variants than to benign variants. Gene-level dissection reveals the physical basis of this failure: in FUS, pathogenicity maps to charge disruption of a 25-residue nuclear localization signal; in TARDBP, to compositional alteration of a 140-residue low-complexity domain; in HNRNPA1, preliminary evidence implicates hydrophobic gain in a prion-like domain. Each mechanism operates on collective properties of functional motifs---net charge, sticker density, amino acid composition---that are invisible to single-residue conservation metrics. A two-step predictor combining literature-defined functional region annotation with ESM2 conservation rescues GoF prediction (ΔAUROC = +0.37), confirming that the missing information is region membership, not better residue-level modeling. The failure is mechanism-specific, not disorder-general: LoF mutations in disordered regions are predicted as accurately as those in structured domains (AUROC = 0.68 at disorder > 0.8). For ALS- and FTD-associated genes most affected by this blind spot, variant interpretation should weight functional region membership over computational conservation scores.

---

## Introduction

Intrinsically disordered regions (IDRs) encode biological function through collective sequence properties rather than folded tertiary structure. In prion-like domains (PrLDs), the identities and linear distributions of aromatic "sticker" residues (Y, F, W) and their intervening "spacer" residues (G, S) govern the driving forces for liquid--liquid phase separation.^1,2^ In nuclear localization signals, the net positive charge of a short basic motif mediates electrostatic recognition by importin or transportin receptors. In low-complexity domains (LCDs), amino acid composition---not sequence---determines aggregation propensity and material properties of condensates.^3^ These collective properties make IDRs functionally essential despite their lack of fixed three-dimensional structure.

Mutations in IDRs cause neurodegenerative disease through gain-of-toxic-function (GoF) mechanisms that differ qualitatively from the loss-of-function (LoF) mutations studied in structured proteins. FUS, TARDBP (TDP-43), and HNRNPA1 harbor ALS- and frontotemporal dementia--linked mutations that cluster in disordered functional regions: the PY-NLS of FUS,^4,5^ the glycine-rich LCD of TARDBP,^6,7^ and the PrLD of HNRNPA1.^8^ In each case, pathogenic mutations alter a collective property of the affected region---charge pattern, composition, or sticker density---causing cytoplasmic mislocalization, aberrant phase transitions, or pathological aggregation. The mutations are not destabilizing in the classical sense; they shift the physical chemistry of a disordered region toward a toxic regime.

Clinical interpretation of missense variants in these genes relies on computational pathogenicity predictors that estimate the functional impact of amino acid substitutions. The current generation of tools includes protein language models that learn evolutionary conservation from large sequence databases (ESM2),^9^ hybrid predictors that combine conservation with AlphaFold2 structural context (AlphaMissense),^10^ and deep generative models that learn coevolutionary constraints from multiple sequence alignments (EVE).^11^ These tools have transformed variant interpretation in structured proteins, where conservation at buried positions correlates with destabilization and loss of function. They are incorporated into the ACMG/AMP variant classification framework as supporting evidence criteria (PP3/BP4),^12^ with calibrated thresholds recommended for clinical application.^13^

Whether these tools perform adequately for variants in disordered regions has received increasing attention. Two proteome-wide assessments have documented reduced sensitivity for pathogenic variants in IDRs compared to structured regions.^14,15^ Luppino et al.^14^ showed that AlphaMissense achieves only 29% sensitivity for IDR variants while maintaining high specificity---consistent with a model that defaults to "benign" in low-conservation regions. Fawzy and Marsh^15^ evaluated 33 variant effect predictors and found widespread sensitivity reductions in IDRs, with pathogenic variants associated with distinct molecular mechanisms including dominant gain- and loss-of-function effects. These studies establish that a performance gap exists and suggest that disease mechanism contributes. What neither study addresses is the physical basis of the gap: which mechanisms fail, which succeed, and why.

The GoF/LoF asymmetry is not confined to disordered regions. Flanagan et al.^16^ demonstrated that SIFT and PolyPhen predict loss-of-function mutations in ABCC8, KCNJ11, and GCK with higher sensitivity than gain-of-function mutations (p ≤ 0.001). Hopkins^17^ confirmed the same pattern for REVEL: 55% of LoF variants met the threshold for strong pathogenicity evidence versus 35% of GoF variants. LoGoFunc,^18^ a genome-wide GoF/LoF predictor trained on 474 features including AlphaFold2 structures and network properties, showed that standard VEPs cannot discriminate GoF from LoF variants. PreMode^19^ found that GoF variants are generally enriched in disordered regions compared to LoF variants. These observations converge on a pattern---conservation predicts LoF better than GoF---but none explains the mechanism, and none examines the specific intersection of GoF with IDRs where the problem should be most severe.

The physical chemistry of IDRs suggests why. Conservation-based predictors detect mutations that violate evolutionary constraints at individual residue positions---constraints that arise from the requirement to maintain tertiary structure and specific molecular contacts. In IDRs, the functionally relevant constraints are collective: the net charge of an NLS, the Y/F ratio and sticker density in a PrLD,^2^ the compositional balance of an LCD, the charge patterning measured by κ.^20^ These constraints can be satisfied by many different sequences, producing high apparent evolutionary tolerance at individual positions even when the collective property is under strong selection. Tsang et al.^21^ proposed that phase separation constitutes a "missing mechanism" for disease interpretation in IDRs, showing that disease-associated proteins are enriched for predicted phase separation propensity. Feng et al.^22^ demonstrated that incorporating phase separation features into EVE and ESM1b improves IDR variant prediction by ~10% AUPR. Our hypothesis is more specific: conservation-based predictors should fail selectively for GoF mutations that disrupt collective IDR properties, while succeeding for LoF mutations in the same genes.

Here we test this hypothesis directly. We analyze 3,409 ClinVar missense variants across 22 genes linked to protein misfolding diseases, stratified by disease mechanism (LoF structured, GoF amyloid, GoF non-amyloid) and disorder level. We show that ESM2, AlphaMissense, and EVE all fail for GoF non-amyloid mutations---not by assigning random scores but by inverting the conservation--pathogenicity relationship, with pathogenic mutations receiving lower scores than benign variants (group AUROC = 0.42 [0.33, 0.50]). We dissect this failure in FUS, TARDBP, and HNRNPA1, identifying the specific physical mechanisms (NLS charge disruption, LCD composition, PrLD hydrophobicity gain) that conservation metrics cannot capture. We show that simple binary annotation of functional regions---defined by published biochemical studies, not ClinVar---rescues prediction (AUROC = 0.82), and that a two-step predictor combining region membership with within-region conservation improves GoF prediction by +0.37 AUROC units. The failure is mechanism-specific, not disorder-general: LoF mutations in disordered regions of the same genes are predicted as accurately as LoF mutations in fully structured proteins. The resolution lies not in better language models but in encoding the correct level of biological abstraction---the functional motif rather than the individual residue.


---

## Results and Discussion

### Conservation-Based Pathogenicity Prediction Is Mechanism-Dependent

We curated 3,409 missense variants from ClinVar across 22 genes linked to protein misfolding or aggregation diseases (Table S2). Variants were labeled pathogenic (672; ClinVar Pathogenic or Likely Pathogenic) or benign proxy (2,737; Benign, Likely Benign, or VUS with ≥1 star review status). VUS were included as benign proxies following standard practice---approximately 80--90% of ClinVar VUS are expected to be truly benign based on population frequency data,^23^ and sensitivity analysis excluding VUS is reported below. Seven genes lack ClinVar pathogenic variants (ATXN3, DDX4, EWSR1, IAPP, MAPT, NPM1, TAF15); per-gene AUROCs are computed on the 15 genes with both classes represented (3,102 variants, 672 pathogenic). Each variant was scored with the ESM2-650M protein language model log-likelihood ratio (LLR), a conservation metric that requires no training and reflects how much a substitution deviates from the model's learned evolutionary prior.

Across all 22 genes, ESM2 LLR achieves a pooled AUROC of 0.67, confirming that conservation captures pathogenicity signal in aggregate (Table 1). Per-gene analysis reveals that this aggregate masks a mechanism-dependent split (Figure 1A). Loss-of-function genes acting through destabilization of structured domains---SOD1, LMNA, VCP, CRYAB---achieve a group AUROC of 0.76 (bootstrap 95% CI [0.74, 0.79]), consistent with the expectation that mutations at conserved positions in folded proteins are deleterious. Amyloid-forming genes (TTR, PRNP, SNCA) show moderate performance (group AUROC = 0.66 [0.60, 0.72]), reflecting partial conservation at aggregation-prone sites.

The gain-of-toxic-function (GoF) non-amyloid group---FUS, TARDBP, HNRNPA1, TIA1, HNRNPA2B1, EWSR1, TAF15---shows group AUROC = 0.42 [0.33, 0.50] (Figure 1A). Conservation is not merely uninformative; it is anti-predictive. The four genes with sufficient pathogenic variants for individual estimation (FUS: 0.42, n = 168, P = 18; TARDBP: 0.41, n = 84, P = 18; HNRNPA1: 0.32, n = 39, P = 3; TIA1: 0.07, n = 121, P = 1) all fall below the 0.50 chance line. HNRNPA1 (P = 3) and TIA1 (P = 1) have too few pathogenic variants for reliable individual estimates; the group pattern is driven by FUS and TARDBP.

The sign of the failure is diagnostic (Figure 1B). For LoF structured genes, pathogenic variants have higher LLR than benign variants (Δμ = +2.98): pathogenic mutations occur at conserved positions. For GoF non-amyloid genes, the sign inverts (Δμ = −0.87): pathogenic mutations occur at positions ESM2 considers *more tolerant* of substitution. This is not reduced signal-to-noise. It is a qualitative reversal of the conservation--pathogenicity relationship.

#### Three independent predictor classes converge on the same failure

Whether this blind spot is specific to ESM2 or reflects a deeper limitation determines its significance. We compared three independent predictors spanning the major axes of variant effect prediction: ESM2 LLR (single-sequence conservation), AlphaMissense (structural context from AlphaFold2 combined with conservation),^10^ and EVE (coevolutionary patterns from deep generative models of multiple sequence alignments).^11^ All three assume that evolutionary signal at the individual-residue level informs pathogenicity.

For the full GoF non-amyloid group (7 genes, P = 41), AlphaMissense achieves AUROC = 0.62 (Figure 1C, dark bars). This apparent partial rescue is attributable to a single gene: FUS, where AlphaMissense reaches 0.87 by leveraging structural context at the PY-NLS motif (discussed below). Excluding FUS reduces AlphaMissense to 0.40 on the remaining four evaluable genes (P = 23, Figure 1C, light bars)---worse than ESM2 (0.45) and below chance. EVE, which operates on coevolutionary information orthogonal to both ESM2 and AlphaMissense, was not available for FUS (absent from the EVE database). On the evaluable GoF subset (4 genes, P = 7 on matched variants), EVE achieves 0.34, the worst of all three (Figure 1C, Table S6).

A binary feature---whether a variant falls within an annotated functional region (NLS, LCD, PrLD, or conserved helix as defined by published functional studies; Table S1)---achieves AUROC = 0.82 for the full GoF group and 0.76 excluding FUS (Figure 1C). This region-membership indicator, derived entirely from published biochemical characterizations independent of ClinVar, outperforms all three residue-level predictors by 0.2--0.4 AUROC units.

The convergent failure of three methodologically independent tools establishes that the GoF non-amyloid blind spot is not a model-specific artifact. It reflects a mismatch between the residue-level evolutionary signal these tools exploit and the motif-level functional properties that determine pathogenicity in these genes. The following three sections dissect this mismatch gene by gene.

![Figure 1](figure_1.png){width=100%}

**Figure 1.** Conservation-based pathogenicity prediction is mechanism-dependent. See figure legend at end of manuscript.


### FUS: Pathogenicity Maps to a 25-Residue Nuclear Localization Signal

FUS encodes a 526-residue RNA-binding protein whose C-terminal PY-NLS (residues 502--526) mediates nuclear import via transportin-1 (TNPO1). Of 18 ClinVar pathogenic variants, 17 fall within this 25-residue motif (Figure 2A). The single exception, M254V, lies in the RGG2 domain. The remaining 150 benign/VUS variants distribute across the full-length protein.

This clustering creates a distinctive ESM2 failure mode (Figure 2B). The NLS resides in a fully disordered C-terminal region (metapredict disorder > 0.9), where ESM2 assigns low conservation scores to most positions. Pathogenic NLS mutations receive LLR values of 2--6, indistinguishable from the background of tolerated variation. Meanwhile, glycine-to-cysteine substitutions in RGG repeats---G230C (LLR = 15.3), G246C (LLR = 14.2)---receive the highest conservation scores in the gene despite being classified as benign/VUS. ESM2 correctly identifies glycine as conserved at RGG positions; what it cannot capture is that FUS pathogenicity is determined by NLS disruption, not repeat destabilization.

A binary NLS membership indicator achieves AUROC = 0.916 (Figure 2D). This 25-residue annotation, from Dormann et al.^5^ and Kwiatkowski et al.,^4^ outperforms the 650-million-parameter language model (AUROC = 0.417) without requiring any computation beyond positional lookup.

#### Charge pattern disruption is the pathogenic mechanism

The PY-NLS carries net charge +2.1 at neutral pH, with 7 basic residues (5R, 2K) and 2 acidic residues. Analysis of the 17 NLS pathogenic mutations reveals two physical mechanisms (Figure 2C). Eleven mutations (65%) reduce net positive charge, with a mean charge change of ΔQ = −0.64 per mutation. Recurrent substitutions at arginine positions---R521G, R521H, R521C, R521L (four distinct substitutions at one position), R514G, R514W, R514S, R524S, R524M---each eliminate one positive charge from the TNPO1 binding interface. K510E reverses positive to negative (ΔQ = −2.0). G509D introduces a new negative charge into the basic cluster (ΔQ = −1.0).

The remaining 6 mutations disrupt the C-terminal PY element required for TNPO1 docking: P525L, P525T, P525S alter the penultimate proline; Y526C eliminates the terminal tyrosine. R518K is a conservative basic-to-basic substitution (ΔQ = 0) that may alter R--x₂₋₅--PY spacing. Q519R (ΔQ = +1.0) increases charge but introduces a bulky guanidinium group adjacent to the PY element.

Both mechanisms operate at the level of collective motif properties---the net charge pattern and the PY dipeptide---rather than individual-residue conservation. ESM2 cannot detect either because the NLS is embedded in a disordered region where individual positions carry weak conservation signal even though the collective charge pattern is functionally essential.

#### Why AlphaMissense partially rescues FUS but not the other GoF genes

AlphaMissense achieves 0.87 for FUS (Table S7), the only GoF gene where a conservation-adjacent tool succeeds. AlphaFold2 assigns moderate confidence (pLDDT ~ 60--70) to the NLS region due to transient helical structure formed upon TNPO1 binding; this structural context provides information that pure sequence conservation misses. The rescue is specific to disordered regions that form defined binding interfaces with known partners. For TARDBP, HNRNPA1, and TIA1, pathogenicity involves altered phase behavior of regions that never adopt stable structure, and AlphaMissense's structural prior provides no advantage (Figure 1C).

![Figure 2](figure_2.png){width=100%}

**Figure 2.** FUS pathogenicity maps to the PY-NLS. See figure legend at end of manuscript.


### TARDBP: Conservation Fails Inside the Low-Complexity Domain

TDP-43 (TARDBP) presents a harder problem. All 18 pathogenic variants fall within the C-terminal LCD (residues 275--414), a 140-residue glycine-rich region with uniformly high predicted disorder (0.86--0.99 across pathogenic positions; Figure 3A). Unlike FUS, where pathogenicity maps to a compact 25-residue binding motif, TARDBP pathogenicity distributes across a large disordered region that mediates phase separation through compositional rather than sequence-specific interactions.

ESM2 achieves AUROC = 0.41 for TARDBP. AlphaMissense performs worse: 0.31 (Table S7). This is the only gene in our dataset where AlphaMissense is anti-predictive, and the reason clarifies the failure mode. AlphaFold2 predicts the LCD as fully disordered (pLDDT < 50 throughout), and AlphaMissense interprets low structural confidence as evidence that mutations are tolerable. The opposite is true: the LCD is disordered because it mediates phase separation, and mutations that alter its composition---hydrophobicity, sticker density, aggregation propensity---convert physiological condensates into pathological aggregates.^6,7^ AlphaMissense's structural prior is not uninformative here; it is misleading.

LCD membership achieves AUROC = 0.71 (Figure 3D), outperforming both residue-level predictors. The gain is smaller than for FUS (0.71 vs. 0.92) because the LCD spans 140 residues---a larger target than the 25-residue NLS---and 39 benign/VUS variants also fall within it.

#### No residue-level feature discriminates within the LCD

Restricting analysis to the 57 LCD variants (18 pathogenic, 39 benign/VUS) yields near-chance performance for every feature tested (Figure 3C): ESM2 LLR (0.56), hydrophobicity change (0.54), local Q/N fraction (0.52), local aromatic density (0.47), Grantham distance (0.45), disorder score (0.47), and p_lock (0.44). No single-residue descriptor separates pathogenic from benign LCD variants.

This negative result is informative. The LCD's pathogenic mechanism---altered phase separation and amyloid-like aggregation---depends on collective sequence properties (overall composition, sticker--spacer balance, helical propensity of sub-regions) that single-position features cannot capture.

#### One sub-regional exception: the Conicella helix

Conicella et al.^7^ identified a conserved α-helical element within the LCD (residues 311--343) that mediates self-association in the condensed phase. Four of 18 pathogenic variants fall within this 33-residue sub-region, and ESM2 achieves AUROC = 0.72 for helix-internal variants---well above its full-gene performance of 0.41. The remaining 14 pathogenic variants distribute across the broader LCD where ESM2 has no discriminative power.

![Figure 3](figure_3.png){width=100%}

**Figure 3.** TARDBP conservation fails inside the LCD. See figure legend at end of manuscript.

The pattern within TARDBP recapitulates the central finding at finer resolution: conservation works where local structure exists (the Conicella helix) and fails where pathogenicity depends on distributed compositional properties (the broader LCD).


### HNRNPA1: Underpowered for Independent Analysis

HNRNPA1 has only three ClinVar pathogenic variants (P340A, D314N, D314V) against 36 benign/VUS. Bootstrap 95% CIs for all HNRNPA1-specific AUROCs span 0.0--1.0 (Figure S2), precluding reliable per-gene inference. Preliminary biophysical analysis of sticker density and hydrophobicity signals in the prion-like domain is presented in Supplementary Note S1; these observations are hypothesis-generating only. HNRNPA1 contributes to the pooled GoF non-amyloid group analysis (where the group-level result is driven by FUS and TARDBP) but should not be weighted as independent evidence for mechanism-specific predictor failure.


### Functional Region Annotation Rescues Prediction for GoF Genes

The preceding sections established that conservation fails for GoF non-amyloid genes because pathogenicity is a property of functional regions, not individual residues. To test whether combining region annotation with conservation can rescue prediction, we implemented a two-step approach: (1) annotate each variant's membership in a critical functional region (NLS, LCD, PrLD, conserved helix, or other literature-defined motif; Table S1), then (2) train a logistic regression using region membership and ESM2 LLR as features, evaluated by leave-one-gene-out cross-validation on 15 genes (3,102 variants, 672 pathogenic).

The two-step predictor achieves AUROC = 0.766, compared to 0.676 for ESM2 LLR alone and 0.729 for region membership alone (Figure 5C). The overall gain of +0.089 [0.068, 0.109] is driven by the GoF group: ΔAUROC = +0.374 [0.305, 0.441], P(two-step > ESM2) = 1.000 across 1,000 bootstrap resamples. For LoF genes, performance is comparable (0.76 for both), confirming that region annotation helps where conservation fails without degrading performance where conservation works.

Per-gene decomposition identifies the source of the gain (Figure 5B). FUS shows the largest improvement (+0.48), driven by the NLS annotation. TARDBP gains +0.32 from LCD membership. TIA1 gains +0.80, though this reflects a single pathogenic variant. For LoF genes where ESM2 already works (LMNA, SOD1), region annotation provides no benefit---as expected, since the annotations for these genes are less discriminative than conservation alone.

#### The region annotations are not circular

The annotations derive from published functional studies that precede our analysis: the FUS PY-NLS from Dormann et al.^5^ and Kwiatkowski et al.,^4^ the TARDBP LCD from Johnson et al.,^6^ the HNRNPA1 PrLD from Kim et al.^8^ These studies defined functional regions through biochemical assays (transportin-1 binding, phase separation, aggregation), not variant pathogenicity.

TARDBP provides a direct positional-proxy test. If the annotations merely labeled "the part of the protein where pathogenic variants cluster," any positional feature would perform comparably. The TARDBP LCD spans residues 275--414 in a 414-residue protein---not an extreme terminus. Relative position achieves AUROC = 0.66 for TARDBP, below LCD membership (0.71). The LCD annotation captures functional biology, not positional clustering.

Boundary sensitivity analysis shows that GoF non-amyloid AUROC is stable across narrow (±10 aa contraction), standard, and wide (±20 aa expansion) region definitions: 0.850, 0.869, 0.857 (range = 0.019; Figure 5D). If annotations were overfit to known pathogenic positions, narrow boundaries should outperform wide ones. The stability indicates that signal comes from being inside the functional region, not from fine-tuned boundary placement.

#### Limitations of the two-step approach

The predictor requires curated region annotations from the literature. For FUS, TARDBP, and HNRNPA1, such annotations exist and are functionally grounded. For newly discovered IDP disease genes, they may not be available, limiting generalizability. The two-step predictor demonstrates the *type* of information needed---functional region membership---rather than serving as a fully automated clinical tool.

![Figure 5](figure_5.png){width=100%}

**Figure 5.** Two-step predictor rescues GoF prediction. See figure legend at end of manuscript.


### The Conservation Failure Is Mechanism-Specific, Not Disorder-General

A natural alternative hypothesis is that ESM2 fails in disordered regions because disorder reduces conservation signal---variable regions compress the LLR separation between pathogenic and benign variants. If true, performance should decline monotonically with increasing disorder. It does not.

Stratifying all 3,409 variants by predicted disorder reveals a non-monotonic pattern (Figure 6A). Structured positions (disorder < 0.3; n = 1,689, P = 497) show AUROC = 0.63. Boundary positions (0.3--0.5; n = 293, P = 79) achieve the highest AUROC in the dataset: 0.80. Moderately disordered positions (0.5--0.8; n = 312, P = 31) remain at 0.75. Only strongly disordered positions (> 0.8; n = 1,115, P = 65) drop to 0.56---and even there, ESM2 remains above chance.

The boundary peak at 0.80 has two possible interpretations: these positions occupy conformational switching regions where evolutionary pressure is strong and pathogenic disruptions are detectable, or metapredict assigns intermediate scores to genuinely structured positions where conservation works well. Resolving this requires experimental disorder data and is beyond the present scope. Either way, the non-monotonic pattern refutes the hypothesis that disorder per se explains the failure.

#### The mechanism × disorder interaction is the key diagnostic

Cross-stratifying by disorder and mechanism resolves the confound (Figure 6B). Within strong IDRs (disorder > 0.8), LoF structured genes retain reasonable performance (AUROC = 0.68 [0.35, 0.91], n = 270, P = 10), while GoF non-amyloid genes drop to 0.40 (n = 410, P = 41). The LoF-in-strong-IDR estimate rests on 10 pathogenic variants, and its wide CI reflects this; the point estimate nonetheless exceeds chance and contrasts with the GoF result. Both groups occupy the same disorder regime; the divergence is attributable to disease mechanism, not disorder level. GoF non-amyloid pathogenic variants exist exclusively in strong IDRs---the other three disorder bins contain no GoF non-amyloid pathogenic variants at all (Figure 6B, bottom row). The failure is not "disordered regions are hard to predict" but "gain-of-toxic-function mutations in disordered regions are specifically anti-predictive."

#### IDP-specific features do not rescue prediction at the residue level

Cohen's d effect sizes for strong-IDR variants are uniformly small (Figure 6C): ESM2 LLR (d = +0.45), hydrophobicity change (d = +0.19), charge density (d = +0.13), aromatic density (d = +0.05), p_lock (d = −0.14). Only ESM2 LLR exceeds |d| = 0.2, and its discriminative power is inverted for GoF genes. No local biophysical feature---evolutionary, structural, or compositional---reaches the effect size needed for practical classification.

This result is consistent with the gene-level findings: the biophysical properties that distinguish pathogenic from benign IDR mutations operate at the scale of functional regions and collective sequence features, not at the resolution of single mutated residues.

![Figure 6](figure_6.png){width=100%}

**Figure 6.** The conservation failure is mechanism-specific, not disorder-general. See figure legend at end of manuscript.


### VUS Label Sensitivity

Our benign class includes VUS used as benign proxies (2,631 of 2,737 benign variants). Restricting to confirmed labels only (672 P/LP vs. 106 B/LB) yields GoF non-amyloid AUROC = 0.50 (41 P vs. 23 confirmed B), compared to 0.42 on the full dataset (Table S4). The small upward shift is expected: VUS are noisier than confirmed benign variants, so removing them reduces label noise. The conclusion is unchanged---ESM2 cannot discriminate pathogenic from benign in GoF non-amyloid genes regardless of label stringency. Per-gene clean-label AUROCs are uninformative for most genes (TARDBP: 1 confirmed B; SOD1: 0 confirmed B) and are reported in Table S4 without interpretation.


### The Blind Spot Has a Clear Physical Origin

The central finding---that conservation-based predictors systematically fail for GoF non-amyloid mutations while succeeding for LoF mutations in structured domains---has a straightforward physical explanation. Conservation metrics detect mutations that disrupt evolved constraints at individual residue positions: buried hydrophobic contacts, catalytic residues, disulfide bonds. These constraints produce strong position-specific conservation signatures that correlate with destabilization and loss of function. GoF mutations in intrinsically disordered regions operate by a different logic. In FUS, the pathogenic property is the net charge of a 25-residue NLS, not the identity of any single residue within it. In TARDBP, it is the compositional character of a 140-residue LCD. In HNRNPA1, preliminary evidence points to hydrophobic gain within a sticker-rich micro-environment. Each mechanism is a collective property of a functional region that is invisible to single-residue conservation metrics.

The failure is not restricted to protein language models. AlphaMissense, which incorporates AlphaFold2 structural predictions, fails for all GoF genes except FUS---and even FUS is rescued only because the NLS forms transient structure upon TNPO1 binding, providing structural context that other GoF regions lack. EVE, based on coevolutionary information from deep generative models, performs worst of all on the evaluable GoF subset (AUROC = 0.34). Coevolutionary coupling captures residue--residue interactions that maintain tertiary structure; in disordered regions without tertiary contacts, this signal is absent. The convergence of three independent methods on the same failure establishes that the problem is not insufficient data or inadequate model architecture. It is the wrong level of biological abstraction.

### Implications for Clinical Variant Interpretation

The affected genes---FUS, TARDBP, HNRNPA1, TIA1---are among the most frequently sequenced in ALS and frontotemporal dementia genetic testing panels. Computational pathogenicity predictors, including tools used in ACMG variant classification,^12^ incorporate conservation-based evidence as supporting criteria (PP3/BP4) with specific calibrated thresholds.^13^ Our results indicate that these criteria are not merely uninformative but actively misleading for GoF mutations in these genes: the predictors assign *lower* pathogenicity scores to true pathogenic variants than to benign variants.

For FUS, the data support a specific interpretive alternative: any missense variant in residues 502--526 that reduces net positive charge or disrupts the PY motif should be evaluated as potentially pathogenic, regardless of conservation score. This rule correctly classifies 17 of 18 known pathogenic FUS variants. For TARDBP, LCD membership provides a weaker but still useful prior: variants within the LCD merit closer scrutiny even when computational predictors score them as benign. For HNRNPA1 and TIA1, the data are insufficient for clinical recommendations.

### Relationship to Recent Approaches

Several recent studies have attempted to improve IDR variant prediction through feature engineering. Feng et al.^22^ showed that adding phase separation features (PSMutPred scores) to EVE and ESM1b yields ~10% AUPR improvement for IDR variants overall. Our framework explains when this approach should and should not work: PS features capture some collective properties of IDRs but still operate at the single-residue level---predicting whether a specific mutation affects phase separation, not whether it falls in a functional region. For genes like FUS where pathogenicity involves nuclear transport rather than phase separation per se, PS features would not capture the relevant biology.

A recent IDR-focused classifier^25^ reported PR-AUC of 0.93 for IDR variants by combining conformation, phase separation, and protein embedding features with AlphaMissense. That evaluation pools all IDR variants without mechanism stratification. Under our framework, the improvement likely reflects better LoF prediction in IDRs---precisely the category where conservation already works (AUROC = 0.68 at disorder > 0.8 for LoF genes). Whether such classifiers fix the GoF problem documented here remains untested.

PreMode^19^ showed that GoF variants are enriched in disordered regions and developed graph neural networks for gene-specific GoF/LoF prediction. Our gene-level dissections complement their genome-wide analysis by identifying the physical mechanisms behind the enrichment: GoF variants cluster in IDRs because the pathogenic properties are collective (charge, composition, sticker density), not because disordered regions are intrinsically more mutable.

Meyer et al.^26^ demonstrated a different GoF mechanism in IDRs: disease mutations can create short linear motifs (e.g., dileucine signals) that drive mislocalization. The FUS NLS charge disruption we document is mechanistically analogous---both alter motif-level properties that redirect protein trafficking---but operates through motif loss rather than motif gain.

### The Disorder--Mechanism Interaction Resolves a Confusion in the Field

Prior reports have suggested that IDRs are generally difficult for pathogenicity prediction.^27,28^ Our data disaggregate this claim. LoF mutations in disordered regions of structured proteins (LMNA boundary positions, VCP flexible loops) are predicted as well as mutations in fully structured regions (AUROC = 0.68 at disorder > 0.8 for LoF genes; Figure 6B). The difficulty is specific to GoF mutations in strong IDRs---a mechanistically defined subset, not a structurally defined one. If the problem were disorder-general, the solution would be better disorder-aware language models. Because the problem is mechanism-specific, the solution is mechanism-aware annotation.

### Scope and Limitations

#### Ascertainment bias

The 22-gene cohort was selected because these genes have known IDP-related disease mechanisms and sufficient ClinVar annotation for statistical analysis. This selection is partly circular: we chose genes where the biology is understood well enough to define mechanism groups, then showed that mechanism determines predictor performance. The finding that conservation-based predictors fail for GoF non-amyloid genes is therefore confirmatory---it characterizes a known clinical challenge (variant interpretation in ALS/FTD genes with collective-property mechanisms) rather than discovering a previously unsuspected one. The clinical relevance is undiminished: these genes are among the most frequently sequenced in neurodegenerative disease panels, and the systematic predictor failure we document has direct implications for variant classification practice. ClinVar itself over-represents well-studied genes, so the variant counts for less-studied GoF genes (EWSR1, TAF15) may underestimate the scope of the problem.

#### Statistical power

The GoF non-amyloid group contains 41 pathogenic variants across 7 genes, with the group-level AUROC dominated by two well-powered genes (FUS and TARDBP, each P = 18). HNRNPA1 and TIA1 are too small for independent conclusions. Extension to additional GoF IDP genes (e.g., hnRNPD, MATR3, UBQLN2) as their ClinVar annotations mature will test whether the mechanism split generalizes beyond the current dataset.

The use of VUS as benign proxies introduces label noise. Sensitivity analysis restricting to confirmed P/LP vs. B/LB labels (Table S4) shows that the GoF non-amyloid AUROC moves from 0.42 to 0.50---still at chance. The clean-label dataset (41 P vs. 23 B for GoF non-amyloid) is small but sufficient to exclude the possibility that VUS mislabeling drives the observed failure.

The LoF-in-strong-IDR comparison (AUROC = 0.68, P = 10) rests on a small number of pathogenic variants. The bootstrap CI ([0.35, 0.91]) is wide enough that the precise magnitude of the LoF advantage is uncertain, though the contrast with GoF non-amyloid (0.40, P = 41) is clear in direction.

The two-step predictor is a proof of concept, not a clinical tool. It requires manually curated functional region annotations that are available only for well-studied genes. Its value is diagnostic: it demonstrates that region membership is the information needed to rescue GoF prediction, validating the motif-level explanation.

### Conclusion

Residue-level evolutionary conservation---whether measured by protein language models, structural context, or coevolutionary coupling---systematically fails for gain-of-toxic-function mutations in intrinsically disordered protein regions. The failure has a physical basis: pathogenicity in these genes is determined by collective properties of functional motifs (NLS charge patterns, LCD composition, PrLD sticker density), not by constraints at individual positions. Functional region annotation rescues prediction because it encodes the correct biological abstraction. For the ALS/FTD genes most affected by this blind spot, variant interpretation should weight functional region membership over computational conservation scores.


---

## Methods

### Variant Curation

Missense variants for 22 genes were downloaded from ClinVar (accessed January 2025). Genes were selected based on known involvement in protein misfolding or aggregation diseases with established gain-of-function or loss-of-function mechanisms. Variants were classified as pathogenic (ClinVar annotation Pathogenic or Likely Pathogenic) or benign proxy (Benign, Likely Benign, or variant of uncertain significance with ≥1 star review status). Conflicting interpretations were excluded. The final dataset contains 3,409 variants: 672 pathogenic and 2,737 benign proxy.

Two genes have known isoform mapping discrepancies: MAPT (17% of ClinVar positions map to our reference sequence) and HNRNPA2B1 (14%), likely due to differences between the ClinVar-referenced isoform and the UniProt canonical sequence used for feature computation. These genes are flagged in the feature matrix (isoform_mismatch = 1) and excluded from per-gene quantitative claims. They are retained in group-level analyses where their low valid-position rates dilute rather than bias the signal.

The use of VUS as benign proxies follows established practice in variant effect predictor benchmarking.^23,29^ Population-level analyses indicate that the vast majority of VUS are functionally neutral; including them as benign proxies increases statistical power while introducing modest label noise. The impact of this choice is quantified by sensitivity analysis (Table S4).

### Mechanism Assignment

Each gene was assigned to one of five mechanism groups based on published disease biology (Table S2): LoF structured (SOD1, LMNA, VCP, CRYAB), GoF amyloid (TTR, PRNP, SNCA, IAPP), GoF non-amyloid (FUS, TARDBP, HNRNPA1, HNRNPA2B1, TIA1, EWSR1, TAF15), repeat expansion (AR, ATXN3), and condensate/other (APP, SQSTM1, DDX4, MAPT, NPM1). Assignments follow Gerasimavicius et al.^30^ for the GoF/LoF distinction and primary literature for individual genes: FUS^4,5^ and TARDBP^6,7^ are classified as GoF non-amyloid because their pathogenic mutations alter phase separation or nuclear transport in disordered regions without forming classical amyloid fibrils as the primary pathogenic event. SOD1 and VCP are classified as LoF structured because their pathogenic mutations destabilize the native fold.^31^

### ESM2 Log-Likelihood Ratio

ESM2-650M^9^ was used to compute log-likelihood ratios (LLR) for each variant. For a substitution from wild-type amino acid *w* to mutant *m* at position *i*, LLR = log P(*m* | context) − log P(*w* | context), where probabilities are the softmax outputs of the final layer of the ESM2 transformer given the wild-type sequence as input. Higher LLR indicates greater evolutionary surprise. LLR was computed using the esm Python package (v2.0.0) with default parameters.

### AlphaMissense and EVE Scores

AlphaMissense pathogenicity scores^10^ were obtained from the precomputed proteome-wide database (AlphaMissense_hg38.tsv.gz, accessed January 2025). Scores range from 0 to 1, with higher values indicating predicted pathogenicity. EVE scores^11^ were obtained from the EVE database (evemodel.org). EVE was not available for all genes; scores were matched to ClinVar variants by genomic coordinate.

### Disorder Prediction

Per-residue disorder scores were computed using metapredict V2^32^ with default parameters. Disorder scores range from 0 (fully structured) to 1 (fully disordered). Variants were stratified into four disorder bins: structured (< 0.3), boundary (0.3--0.5), moderate (0.5--0.8), and strong IDR (> 0.8).

### Functional Region Annotation

Critical functional regions for each gene were defined from published biochemical studies independent of ClinVar variant data (Table S1). Sources: FUS PY-NLS (residues 502--526; Dormann et al.^5^; Kwiatkowski et al.^4^), TARDBP glycine-rich LCD (residues 275--414; Johnson et al.^6^), TARDBP conserved helix (residues 311--343; Conicella et al.^7^), HNRNPA1 PrLD (residues 186--372; Kim et al.^8^), and corresponding structured domains from UniProt annotation for LoF genes. Each variant was assigned a binary region membership indicator (1 if within a critical region, 0 otherwise).

### Biophysical Feature Computation

Local sticker fraction (p_lock) was computed in a ±15-residue window centered on each variant position, using the sticker--spacer framework^1,24^ where Y, F, W, R, K are stickers and all others are spacers. Hydrophobicity change (Δhydro) was computed as the difference in Kyte--Doolittle hydrophobicity between mutant and wild-type residues. Grantham distance, charge change (ΔQ at pH 7.0), and local aromatic density were computed using standard amino acid property tables.

### AUROC Computation and Statistical Analysis

Receiver operating characteristic (ROC) curves were computed for each gene, mechanism group, and disorder stratum. Area under the ROC curve (AUROC) was computed using the trapezoidal rule. Per-gene AUROCs were computed only for genes with ≥1 pathogenic and ≥1 benign/VUS variant (15 of 22 genes). Group AUROCs were computed by pooling all variants within a mechanism group.

Bootstrap 95% confidence intervals (CIs) were computed from 1,000 stratified bootstrap resamples. Each resample preserved the pathogenic:benign ratio within each gene by sampling with replacement within each class. CIs are reported as the 2.5th and 97.5th percentiles (percentile method). For group comparisons, the probability that one predictor exceeds another was computed as the fraction of bootstrap samples in which the first predictor's AUROC exceeded the second. Cohen's d effect sizes were computed as the difference in group means divided by the pooled standard deviation.

### Two-Step Predictor

A logistic regression model was trained using two features: ESM2 LLR and binary functional region membership. The model was evaluated by leave-one-gene-out cross-validation: for each of the 15 evaluable genes, the model was trained on the remaining 14 genes and tested on the held-out gene. AUROC was computed on the pooled held-out predictions. The improvement over ESM2 alone (ΔAUROC) was assessed by 1,000 bootstrap resamples.

### Boundary Sensitivity Analysis

To test whether region annotation performance depends on precise boundary placement, three sets of region definitions were compared: narrow (published boundaries contracted by 10 residues on each side), standard (published boundaries), and wide (published boundaries expanded by 20 residues on each side). GoF non-amyloid AUROC was recomputed for each definition.

### VUS Sensitivity Analysis

To assess the impact of VUS labeling on conclusions, the analysis was repeated with VUS excluded (pathogenic vs. confirmed B/LB only). Per-gene and group AUROCs were recomputed on the restricted dataset (Table S4).

### Data and Code Availability

ClinVar variant data are publicly available at ncbi.nlm.nih.gov/clinvar/. ESM2 is available at github.com/facebookresearch/esm. AlphaMissense scores are available at zenodo.org/record/8208688. EVE scores are available at evemodel.org. Analysis code and curated datasets will be deposited at [GitHub repository URL] upon publication.


---

## References

1. Choi, J.-M.; Holehouse, A. S.; Pappu, R. V. Physical Principles Underlying the Complex Biology of Intracellular Phase Transitions. *Annu. Rev. Biophys.* **2020**, *49*, 107--133.

2. Bremer, A.; Farag, M.; Borcherds, W. M.; Peran, I.; Martin, E. W.; Pappu, R. V.; Mittag, T. Deciphering How Naturally Occurring Sequence Features Impact the Phase Behaviours of Disordered Prion-like Domains. *Nat. Chem.* **2022**, *14*, 196--207.

3. Molliex, A.; Temirov, J.; Lee, J.; Coughlin, M.; Kanagaraj, A. P.; Kim, H. J.; Mittag, T.; Taylor, J. P. Phase Separation by Low Complexity Domains Promotes Stress Granule Assembly and Drives Pathological Fibrillization. *Cell* **2015**, *163*, 123--133.

4. Kwiatkowski, T. J.; Bosco, D. A.; LeClerc, A. L.; Tamrazian, E.; Vanderburg, C. R.; Russ, C.; Davis, A.; Gilchrist, J.; et al. Mutations in the FUS/TLS Gene on Chromosome 16 Cause Familial Amyotrophic Lateral Sclerosis. *Science* **2009**, *323*, 1205--1208.

5. Dormann, D.; Rodde, R.; Edbauer, D.; Bentmann, E.; Fischer, I.; Hruscha, A.; Than, M. E.; Mackenzie, I. R. A.; Capell, A.; Schmid, B.; et al. ALS-Associated Fused in Sarcoma (FUS) Mutations Disrupt Transportin-Mediated Nuclear Import. *EMBO J.* **2010**, *29*, 2841--2857.

6. Johnson, B. S.; Snead, D.; Lee, J. J.; McCaffery, J. M.; Shorter, J. TDP-43 Is Intrinsically Aggregation-Prone, and Amyotrophic Lateral Sclerosis-Linked Mutations Accelerate Aggregation and Increase Toxicity. *J. Biol. Chem.* **2009**, *284*, 20329--20339.

7. Conicella, A. E.; Zerze, G. H.; Mittal, J.; Fawzi, N. L. ALS Mutations Disrupt Phase Separation Mediated by α-Helical Structure in the TDP-43 Low-Complexity C-Terminal Domain. *Structure* **2016**, *24*, 1537--1549.

8. Kim, H. J.; Kim, N. C.; Wang, Y.-D.; Scarborough, E. A.; Moore, J.; Diaz, Z.; MacLea, K. S.; Freibaum, B.; Li, S.; Molliex, A.; et al. Mutations in Prion-Like Domains in hnRNPA2B1 and hnRNPA1 Cause Multisystem Proteinopathy and ALS. *Nature* **2013**, *495*, 467--473.

9. Lin, Z.; Akin, H.; Rao, R.; Hie, B.; Zhu, Z.; Lu, W.; Smetanin, N.; Verkuil, R.; Kabeli, O.; Shmueli, Y.; et al. Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model. *Science* **2023**, *379*, 1123--1130.

10. Cheng, J.; Novati, G.; Pan, J.; Bycroft, C.; Žemgulytė, A.; Applebaum, T.; Pritzel, A.; Wong, L. H.; et al. Accurate Proteome-wide Missense Variant Effect Prediction with AlphaMissense. *Science* **2023**, *381*, eadg7492.

11. Frazer, J.; Notin, P.; Dias, M.; Gomez, A.; Min, J. K.; Brock, K.; Gal, Y.; Marks, D. S. Disease Variant Prediction with Deep Generative Models of Evolutionary Data. *Nature* **2021**, *599*, 91--95.

12. Richards, S.; Aziz, N.; Bale, S.; Bick, D.; Das, S.; Gastier-Foster, J.; Grody, W. W.; Hegde, M.; Lyon, E.; Spector, E.; et al. Standards and Guidelines for the Interpretation of Sequence Variants. *Genet. Med.* **2015**, *17*, 405--424.

13. Pejaver, V.; Byrne, A. B.; Feng, B.-J.; Pagel, K. A.; Mooney, S. D.; Karchin, R.; et al. Calibration of Computational Tools for Missense Variant Pathogenicity Classification and ClinGen Recommendations for PP3/BP4 Criteria. *Am. J. Hum. Genet.* **2022**, *109*, 2163--2177.

14. Luppino, F.; Lenz, S.; Chow, C. F. W.; Toth-Petroczy, A. Deep Learning Tools Predict Variants in Disordered Regions with Lower Sensitivity. *BMC Genomics* **2025**, *26*, 367.

15. Fawzy, M.; Marsh, J. A. Assessing Variant Effect Predictors and Disease Mechanisms in Intrinsically Disordered Proteins. *PLoS Comput. Biol.* **2025**, *21*, e1013400.

16. Flanagan, S. E.; Patch, A.-M.; Ellard, S. Using SIFT and PolyPhen to Predict Loss-of-Function and Gain-of-Function Mutations. *Genet. Test. Mol. Biomarkers* **2010**, *14*, 533--537.

17. Hopkins, J. J.; Sheridan, E.; Ormondroyd, E.; Sheridan, E. REVEL Is Better at Predicting Pathogenicity of Loss-of-Function than Gain-of-Function Variants. *Hum. Mutat.* **2023**, *2023*, 8857940.

18. Stein, D.; Kars, M. E.; Wu, Y.; Bhatt, D. K.; Bhatt, J.; et al. Genome-wide Prediction of Pathogenic Gain- and Loss-of-Function Variants from Ensemble Learning of a Diverse Feature Set. *Genome Med.* **2023**, *15*, 103.

19. Zhong, G.; Zhao, Y.; Zhuang, D.; Chung, W. K.; Shen, Y. PreMode Predicts Mode-of-Action of Missense Variants by Deep Graph Representation Learning of Protein Sequence and Structural Context. *Nat. Commun.* **2025**, *16*, 7143.

20. Das, R. K.; Pappu, R. V. Conformations of Intrinsically Disordered Proteins Are Influenced by Linear Sequence Distributions of Oppositely Charged Residues. *Proc. Natl. Acad. Sci. U.S.A.* **2013**, *110*, 13392--13397.

21. Tsang, B.; Pritišanac, I.; Scherer, S. W.; Moses, A. M.; Forman-Kay, J. D. Phase Separation as a Missing Mechanism for Interpretation of Disease Mutations. *Cell* **2020**, *183*, 1742--1756.

22. Feng, M.; Wei, X.; Zheng, X.; Liu, L.; Lin, L.; Xia, M.; He, G.; Shi, Y.; Lu, Q. Decoding Missense Variants by Incorporating Phase Separation via Machine Learning. *Nat. Commun.* **2024**, *15*, 8279.

23. Landrum, M. J.; Lee, J. M.; Benson, M.; Brown, G. R.; Chao, C.; Chitipiralla, S.; et al. ClinVar: Improving Access to Variant Interpretations and Supporting Evidence. *Nucleic Acids Res.* **2018**, *46*, D1062--D1067.

24. Wang, J.; Choi, J.-M.; Holehouse, A. S.; Lee, H. O.; Zhang, X.; Jahnel, M.; Maharana, S.; Lemaitre, R.; Pozniakovsky, A.; Drechsel, D.; et al. A Molecular Grammar Governing the Driving Forces for Phase Separation of Prion-like RNA Binding Proteins. *Cell* **2018**, *174*, 688--699.

25. Presman, D. M.; et al. Enhancing Missense Variant Classification in Predicted Intrinsically Disordered Regions. *bioRxiv* **2025**, 2025.08.08.669269.

26. Meyer, K.; Kirchner, M.; Uyber, B.; Cheng, J.-Y.; Krause, G.; Raber, H.-R.; Schlosser, A.; Selbach, M.; et al. Mutations in Disordered Regions Can Cause Disease by Creating Dileucine Motifs. *Cell* **2018**, *175*, 239--253.

27. Ancien, F.; Pucci, F.; Godfroid, M.; Rooman, M. Prediction and Interpretation of Deleterious Coding Variants in Terms of Protein Structural Stability. *Sci. Rep.* **2018**, *8*, 4480.

28. Iqbal, S.; Pérez-Palma, E. Missense Variant Pathogenicity Predictors Do Not Account for Structural Context. *bioRxiv* **2022**, 2022.12.20.521218.

29. Grimm, D. G.; Azencott, C.-A.; Aicheler, F.; Gieraths, U.; MacArthur, D. G.; Samocha, K. E.; et al. The Evaluation of Tools Used to Predict the Impact of Missense Variants Is Hindered by Two Types of Circularity. *Hum. Mutat.* **2015**, *36*, 513--523.

30. Gerasimavicius, L.; Livesey, B. J.; Marsh, J. A. Loss-of-Function, Gain-of-Function and Dominant-Negative Mutations Have Profoundly Different Effects on Protein Structure. *Nat. Commun.* **2022**, *13*, 3895.

31. Grad, L. I.; Cashman, N. R. Prion-like Activity of Cu/Zn Superoxide Dismutase: Implications for Amyotrophic Lateral Sclerosis. *Prion* **2014**, *8*, 33--41.

32. Emenecker, R. J.; Griffith, D.; Holehouse, A. S. Metapredict: a Fast, Accurate, and Easy-to-Use Predictor of Consensus Disorder and Structure. *Biophys. J.* **2021**, *120*, 4312--4319.

33. Alberti, S.; Halfmann, R.; King, O.; Kapila, A.; Lindquist, S. A Systematic Survey Identifies Prions and Illuminates Sequence Features of Prionogenic Proteins. *Cell* **2009**, *137*, 146--158.

34. Lancaster, A. K.; Nutter-Upham, A.; Lindquist, S.; King, O. D. PLAAC: a Web and Command-Line Application to Identify Proteins with Prion-Like Amino Acid Composition. *Bioinformatics* **2014**, *30*, 2501--2502.


---

## Figure Legends

**Figure 1. Conservation-based pathogenicity prediction is mechanism-dependent.** (A) Per-gene ESM2 LLR AUROC for 22 genes, organized by disease mechanism. Horizontal bars show point estimates; error bars show bootstrap 95% CIs. Dashed vertical line marks AUROC = 0.50 (chance). Genes with P = 0 pathogenic variants are listed without bars. GoF non-amyloid genes (red) fall below chance; LoF structured genes (blue) cluster above 0.65. (B) Violin plots of ESM2 LLR distributions for benign/VUS (light) and pathogenic (dark) variants in LoF structured (left) and GoF non-amyloid (right) groups. Horizontal bars indicate means. Δμ = +2.98 for LoF structured (pathogenic variants have higher conservation scores); Δμ = −0.87 for GoF non-amyloid (pathogenic variants have lower scores). (C) AUROC comparison across four predictors for GoF non-amyloid genes: ESM2 LLR, AlphaMissense, EVE, and binary functional region membership. Dark bars: all 7 GoF genes (P = 41); light bars: excluding FUS (4 genes, P = 23). Dashed line: chance. Region membership (0.82) outperforms all three conservation-based predictors. EVE was unavailable for FUS.

**Figure 2. FUS pathogenicity maps to the PY-NLS.** (A) FUS domain architecture with pathogenic variants (red triangles, above) and benign/VUS variants (gray triangles, below). Metapredict disorder profile shown below domain diagram. 17/18 pathogenic variants cluster in the C-terminal PY-NLS (residues 502--526, red shading). (B) ESM2 LLR vs. residue position. Pathogenic NLS variants (red circles) receive low LLR (2--6), while benign RGG-repeat variants (e.g., G230C, G246C) receive high LLR (>14). NLS region shaded. (C) Charge change (ΔQ) for all 17 NLS pathogenic variants. 11/17 reduce net positive charge (blue); 6/17 disrupt the PY element (orange). Individual mutations labeled. (D) ROC curves comparing ESM2 LLR (AUROC = 0.417), AlphaMissense (0.870), and NLS membership (0.916).

**Figure 3. TARDBP conservation fails inside the low-complexity domain.** (A) TARDBP domain architecture with pathogenic variants (red), benign/VUS (gray), and metapredict disorder profile. All 18 pathogenic variants fall within the glycine-rich LCD (residues 275--414, red shading). The conserved α-helix (residues 311--343, dashed orange lines) is a structured sub-element. (B) ESM2 LLR vs. residue position. Pathogenic LCD variants (red) are interspersed among benign variants with no separation by LLR. Helix sub-region annotated (helix AUC = 0.72). (C) Scatter plot of ESM2 LLR vs. AlphaMissense score for LCD variants. Gene-level AUROCs: ESM2 = 0.412, AlphaMissense = 0.314. Helix variants circled in orange. Neither predictor separates pathogenic from benign. (D) ROC curves: LCD membership (0.705) outperforms ESM2 LLR (0.412) and AlphaMissense (0.314).

**Figure 4. HNRNPA1: preliminary biophysical signal from three pathogenic variants.** (A) HNRNPA1 domain architecture. Two RRM domains (green) and C-terminal PrLD (red, residues 186--372). Three pathogenic variants (red triangles) cluster near the PrLD C-terminus. Warning box: n = 3 pathogenic variants, interpret with caution. (B) Local sticker fraction (p_lock, ±10-residue window) along the HNRNPA1 sequence. RRM regions (blue shading) and PrLD (red shading) annotated. Pathogenic variants (red circles, labeled) fall at positions with elevated sticker density. p_lock AUROC = 0.722 [0.29, 0.97]. (C) Hydrophobicity change for each pathogenic variant. D314V (= D262V in Kim et al.^8^ short isoform numbering) shows the largest change (ΔH = +7.7, Grantham = 152).

**Figure 5. Two-step predictor rescues GoF prediction.** (A) Schematic of the two-step approach: region membership + ESM2 LLR in a logistic regression, evaluated by leave-one-gene-out cross-validation. (B) Per-gene ΔAUROC (two-step minus ESM2 alone). GoF genes (red) show large gains: FUS (+0.48), TARDBP (+0.32), TIA1 (+0.80). LoF genes (blue) show no change. (C) Overall AUROC comparison: ESM2 alone (0.676), region membership alone (0.729), two-step combined (0.766). ΔAUROC = +0.089 overall; GoF-specific ΔAUROC = +0.374 [0.305, 0.441]. (D) Boundary sensitivity analysis. GoF non-amyloid AUROC across narrow (±10 aa contraction: 0.850), standard (0.869), and wide (±20 aa expansion: 0.857) region definitions. Range = 0.019, indicating that performance is robust to boundary placement.

**Figure 6. The conservation failure is mechanism-specific, not disorder-general.** (A) ESM2 AUROC by predicted disorder level (all genes pooled). Non-monotonic pattern: structured (< 0.3) = 0.63, boundary (0.3--0.5) = 0.80, moderate (0.5--0.8) = 0.75, strong IDR (> 0.8) = 0.56. Sample sizes shown per bin. (B) ESM2 AUROC cross-stratified by disorder level and mechanism group. Within strong IDRs (> 0.8): LoF structured retains AUROC = 0.68 (n = 270, P = 10); GoF non-amyloid drops to 0.40 (n = 410, P = 41). GoF non-amyloid pathogenic variants exist exclusively in the strong IDR bin. (C) Cohen's d effect sizes for five features in strong-IDR variants (disorder > 0.8): ESM2 LLR (+0.45), hydrophobicity change (+0.19), charge density (+0.13), aromatic density (+0.05), p_lock (−0.14). No feature achieves the effect size needed for practical classification.

---

### Supplementary Figures

**Figure S1. Masked-marginal ESM2 scoring does not rescue FUS prediction.** (A) Scatter plot of unmasked ESM2 LLR vs. masked-marginal LLR for all FUS variants (Pearson r = 0.990). Pathogenic variants (red) and benign/VUS (gray). (B) ROC curves for FUS: unmasked LLR (AUROC = 0.417) vs. masked-marginal LLR (0.449). Masked marginals provide negligible improvement; the NLS blind spot persists.

**Figure S2. Bootstrap confidence intervals for per-gene and group AUROCs.** Forest plot of ESM2 LLR AUROC with 95% bootstrap CIs for all 15 evaluable genes (top) and 5 mechanism groups (bottom). Genes with P < 5 pathogenic variants shown with open circles and italicized labels to indicate low statistical power. Group-level diamonds show pooled mechanism estimates. Dashed vertical line: chance (0.50).

**Figure S3. AlphaMissense vs. ESM2 performance comparison.** (A) Scatter plot of per-gene ESM2 AUROC (x-axis) vs. AlphaMissense AUROC (y-axis) for all evaluable genes. Genes colored by mechanism. Dashed line: parity. FUS (ESM2 = 0.42, AM = 0.87) and TARDBP (ESM2 = 0.41, AM = 0.31) are annotated. (B) Grouped bar chart comparing ESM2, AlphaMissense, and EVE AUROC for three GoF non-amyloid genes with sufficient data: FUS, TARDBP, HNRNPA1. AlphaMissense rescues FUS only; TARDBP is the only gene where AlphaMissense is worse than ESM2.

**Figure S4. ESM2 LLR vs. residue position for all evaluable genes.** Grid of scatter plots (one per gene) showing ESM2 LLR by residue position. Pathogenic variants (dark red circles) and benign/VUS (light gray). Critical functional regions shaded in pink where applicable. Gene name, AUROC, and number of pathogenic variants shown in each subplot title.

### Supplementary Tables

**Table S1.** Functional region annotations for all 22 genes, including region name, boundaries (start, end), whether designated as critical for the two-step predictor, and literature citation defining the region.

**Table S2.** Gene-level summary: gene name, UniProt accession, protein length, mechanism group, number of ClinVar pathogenic and benign/VUS variants, and ESM2 group AUROC.

**Table S3.** Per-variant feature table: gene, position, reference and alternate amino acids, ClinVar classification, ESM2 LLR, AlphaMissense score, disorder score, region membership, and biophysical features.

**Table S4.** VUS-excluded sensitivity analysis: per-gene and group AUROCs using confirmed P/LP vs. B/LB labels only.

**Table S5.** Bootstrap confidence intervals (2.5th and 97.5th percentiles) for per-gene and group AUROCs from 1,000 stratified resamples.

**Table S6.** EVE comparison: per-gene EVE AUROCs for genes with available EVE scores, alongside ESM2 and AlphaMissense AUROCs on the matched variant subset.

**Table S7.** AlphaMissense comparison: per-gene AlphaMissense AUROCs alongside ESM2 AUROCs.

### Supplementary Notes

**Supplementary Note S1. HNRNPA1: preliminary biophysical analysis of three pathogenic variants.**

HNRNPA1 has three ClinVar pathogenic variants---P340A, D314N, D314V---against 36 benign/VUS, making per-gene analysis statistically preliminary. Bootstrap 95% CIs for all HNRNPA1-specific AUROCs span 0.0--1.0 (Figure S2), and all results below should be treated as hypothesis-generating.

With that caveat, HNRNPA1 is the only GoF non-amyloid gene where an IDP-specific biophysical feature shows discriminative signal. Local sticker fraction (p_lock, computed in a ±15-residue window using the Wang et al.^24^ sticker--spacer framework where Y, F, W, R, K are stickers) achieves AUROC = 0.72 [0.29, 0.97] (Figure 4B). The three pathogenic variants occur at positions with higher sticker density (mean p_lock = 0.57) than benign/VUS variants (mean = 0.45), consistent with the sticker--spacer model of condensate maturation in prion-like domains.

**D314V: hydrophobic gain, not sticker conversion.** Kim et al.^8^ identified D262V as promoting amyloid-like aggregation of HNRNPA1. That variant was reported in the short isoform (320 residues, UniProt A0A2R8Y4L1); in the canonical isoform (372 residues, UniProt P09651), a 52-residue insert at positions 252--303 shifts numbering, making D262V equivalent to D314V in our dataset. D314V is the most physicochemically severe of the three pathogenic variants: Grantham distance = 152, hydrophobicity change = +7.7 (Kyte--Doolittle), and ESM2 LLR = 9.10---the only HNRNPA1 pathogenic variant where ESM2 assigns substantial conservation surprise. The sticker fraction change is zero (Δp_lock = 0.0): neither aspartate nor valine is a sticker residue. The mechanism is hydrophobic self-association---replacing a charged, soluble residue with a hydrophobic one in a sticker-rich micro-environment (local p_lock = 0.63)---not sticker gain per se (Figure 4C).

The three pathogenic variants show elevated hydrophobicity change (mean Δhydro = +3.7 vs. +1.4 for benign/VUS; AUROC = 0.70), suggesting that hydrophobic gain in the PrLD context is the relevant biophysical perturbation. Whether this pattern generalizes requires additional pathogenic variants; at n = 3, no quantitative conclusion is warranted.
