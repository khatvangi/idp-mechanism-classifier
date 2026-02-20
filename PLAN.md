# implementation plan — phase 2

## architecture: one script per axis, one integration script

```
idp-mechanism-classifier/
├── PROJECT.md                    # goals, proteins, hypotheses
├── LITERATURE_SYNTHESIS.md       # phase 1 findings
├── PLAN.md                       # this file
├── sequences/                    # FASTA files for all proteins + mutants
├── features/                     # computed features (CSV/NPY per axis)
│   ├── polymer_physics.csv       # kappa, SCD, NCPR, FCR, GRAVY, etc.
│   ├── maturation_grammar.csv    # p_lock, p_pi, p_polar per protein
│   ├── amyloid_propensity.csv    # APR count, beta propensity, etc.
│   └── esm2_embeddings.npy       # 1280-dim mean embeddings per protein
├── scripts/
│   ├── 01_prepare_sequences.py   # fetch/write all FASTA files
│   ├── 02_polymer_physics.py     # kappa, SCD, NCPR, FCR, Das-Pappu
│   ├── 03_maturation_grammar.py  # reuse condensate-maturation-theory
│   ├── 04_amyloid_propensity.py  # sequence-based APR scoring
│   ├── 05_esm2_embeddings.py     # extract 1280-dim embeddings
│   └── 06_integrate_landscape.py # combine axes, PCA/UMAP, figures
└── figures/                      # output visualizations
```

## step 1: prepare sequences (~30 min)

collect FASTA files for all 16 sequences (8 WT + 8 mutants):
- alpha-synuclein WT, A53T, E46K
- Abeta42 WT
- tau 2N4R WT, P301L
- TDP-43 WT, M337V, Q331K
- FUS WT, P525L
- hnRNPA1 WT, D262V
- PrP WT (mature, 23-230)
- Httex1 Q21 WT, Q46

source: UniProt sequences already fetched during AF3 input creation.

## step 2: polymer physics (~1 hr)

compute per protein using localCIDER + custom code:
- **kappa** (charge patterning, 0-1)
- **SCD** (sequence charge decoration)
- **NCPR** (net charge per residue)
- **FCR** (fraction charged residues)
- **mean hydropathy** (Kyte-Doolittle GRAVY)
- **Das-Pappu region** (R1-R5 classification from f+/f-)
- **aromatic fraction** (F+Y+W / length)
- **glycine fraction** (G / length)
- **proline fraction** (P / length)
- **Gln+Asn fraction** (Q+N / length) — solidification promoters
- **Ser fraction** (S / length) — solidification promoter

reuse: existing `02_disorder_analysis.py` from k-dense has kappa, SCD, NCPR, FCR already.
extend: add Das-Pappu region classification, Gln+Asn fraction, Ser fraction.

## step 3: maturation grammar (~2 hrs)

import grammar from condensate-maturation-theory project:
- compute p_lock, p_pi, p_polar matrices for each protein
- extract scalar summaries: mean p_lock, mean p_pi, mean p_polar, max p_lock
- compute Γ₀ (encounter rate proxy) from grammar

challenge: grammar was designed for LCDs (~150-200 aa). for full-length proteins (tau 441, FUS 526), need to either:
- (a) extract LCD region and score that, or
- (b) score full-length with appropriate windowing

decision: score BOTH full-length AND LCD region where applicable. the LCD score is the maturation-relevant one.

LCD regions (from literature):
- FUS: 1-163 (QGSY-rich LCD)
- TDP-43: 274-414 (glycine-rich LCD)
- hnRNPA1: 186-372 (PrLD)
- tau: no canonical LCD — use MTBR (244-368) as aggregation-prone region
- alpha-synuclein: NAC region (61-95) as aggregation-prone region
- PrP: octapeptide repeat (51-91) + hydrophobic core (112-145)
- Httex1: polyQ region
- Abeta42: full sequence (42 aa, already short)

## step 4: amyloid propensity (~2 hrs)

TANGO/WALTZ are web-only. alternatives:
- **sequence-based APR detection**: hydrophobic stretches of 5-7 residues with high beta-sheet propensity
- **Chou-Fasman beta-sheet propensity** (simple, well-established)
- **LARKS detection**: low-complexity amyloid-like reversible kinked segments
- **aggregation-prone region (APR) heuristic**: count residues in windows with high hydrophobicity + beta propensity + low charge

plan:
1. compute per-residue beta-sheet propensity (Chou-Fasman scale)
2. compute per-residue hydrophobicity (Kyte-Doolittle)
3. identify APRs: sliding window (7 aa), average hydrophobicity > 1.0 AND beta propensity > 1.0 AND |charge| < 0.3
4. count: number of APRs, total APR residues, max APR score, APR fraction
5. compute overall amyloid propensity score = mean(hydrophobicity × beta_propensity) over APR windows

known scales (no external tools needed):
- Chou-Fasman beta-sheet: V=1.70, I=1.60, Y=1.47, F=1.38, W=1.37, L=1.30, T=1.19, ...
- Kyte-Doolittle: I=4.5, V=4.2, L=3.8, F=2.8, ...

## step 5: ESM2 embeddings (~3 hrs)

use ESM2-650M (cached at ~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt):
- load via HuggingFace transformers (already installed)
- for each protein sequence:
  - tokenize
  - extract last hidden state (per-residue 1280-dim)
  - compute protein-level features:
    - mean embedding (1280-dim)
    - per-residue embeddings for LCD region
    - embedding PCA/variance (captures conformational diversity signal)
  - also compute masked marginal probabilities (LLR) at each position for comparison
- run on GPU 1 (CUDA_VISIBLE_DEVICES=1)

output: 16×1280 embedding matrix + 16-element LLR vectors

## step 6: integration + landscape visualization (~2 hrs)

1. combine all features into one matrix (16 proteins × N features)
2. standardize features (z-score)
3. PCA to find principal axes of variation
4. check if PC1-PC2 separates known mechanism classes
5. UMAP for nonlinear visualization
6. plot mechanism landscape with:
   - color by known mechanism (amyloid=red, condensate=blue, dual=purple, collapse=green, prion=orange)
   - arrows from WT → mutant showing mutation effects
   - confidence ellipses per mechanism class
7. compute inter-class distances
8. radar/spider plots per protein showing all 4 axes

## validation strategy

with only 8 proteins, formal train/test split is meaningless. instead:
1. **internal consistency**: do known amyloid formers (asyn, ab42) cluster? do condensate maturers (FUS, TDP-43, hnRNPA1) cluster?
2. **mutation effects**: does P301L tau shift toward amyloid? does D262V hnRNPA1 shift toward maturation?
3. **tau boundary**: does tau sit between amyloid and condensate clusters?
4. **leave-one-out**: remove each protein, check if remaining 7 predict its mechanism
5. **literature agreement**: compare feature rankings with experimental evidence

## self-critique: phase 2

### what's good about this plan?
- reuses existing code (grammar + polymer physics)
- no external web tools needed (APR scoring is self-contained)
- ESM2 is ready to run (cached weights, GPU available)
- honest about 8-protein limitation → landscape not classifier
- each axis is independently interpretable

### what's wrong?
1. **APR scoring is crude** — TANGO/WALTZ use sophisticated statistical mechanics, we're using a simple heuristic. BUT: for comparing 8 proteins against each other, relative ranking may be sufficient.
2. **ESM2 embeddings are 1280-dim for 16 proteins** — massive curse of dimensionality. need aggressive dimensionality reduction (PCA to 5-10 dims) before combining.
3. **maturation grammar for non-LCD proteins is unclear** — what does p_lock mean for alpha-synuclein (no canonical LCD)? may need to flag these as "not applicable" rather than forcing a score.
4. **no independent test set** — leave-one-out with 8 proteins is weak. could expand to ~20 proteins by adding IAPP, SOD1, TIA1, DDX4, LAF1, EWSR1, etc.
5. **sticker patterning (Martin 2020)** is not explicitly captured — should add aromatic clustering metric (variance of aromatic positions).

### adjustments needed
- add aromatic clustering metric to polymer physics axis
- flag maturation grammar scores as "LCD-applicable" vs "non-LCD"
- reduce ESM2 to top 10 PCA components before integration
- consider expanding protein panel in future iteration
