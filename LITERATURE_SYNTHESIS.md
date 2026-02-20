# literature synthesis — phase 1

## the landscape

### closest competitor: Frank et al. 2024 (PNAS)
- fine-tuned ESM2 for BOTH LLPS and amyloid prediction from sequence
- binary classification only (droplet vs amyloid)
- no condensate maturation pathway class
- no polymer physics features (kappa, SCD, NCPR)
- no mutation-level mechanism switching prediction
- **our gap: multi-class, multi-axis, mutation-aware**

### what exists (single-axis tools)
| tool | predicts | method |
|------|----------|--------|
| FuzDrop | LLPS + amyloid propensity | sequence features, per-residue |
| ALBATROSS | IDP conformational dimensions (Rg, Re, nu) | deep learning on CALVADOS |
| FINCHES | IDR-IDR intermolecular interactions | polymer physics (CALVADOS params) |
| CIDER/localCIDER | kappa, SCD, FCR, NCPR, Das-Pappu diagram | analytical |
| TANGO/WALTZ/CamSol | amyloid propensity, APR regions | sequence features |
| CANYA | amyloid nucleation | CNN-attention on 100K+ experiments |
| Seq2Phase/PSTP | LLPS prediction | ProtT5/ESM2 embeddings |
| AggNet/PALM | amyloid aggregation | ESM2 + structure features |

**nobody integrates these axes into a mechanism classifier.**

### ESM2 for IDPs — the key nuance
- vanilla LLR/pseudo-perplexity: **fails** in fully disordered regions
  - ESM-Effect: Spearman rho = -0.02 on SNCA C-terminal IDR
  - Luppino et al. 2025: up to 20% sensitivity gap for pathogenic variants in IDRs
  - 33 VEPs benchmarked: ALL show reduced sensitivity in IDRs
- ESM2 embeddings (1280-dim): **rescue performance**
  - Farquhar 2026: AUC 0.982 in IDRs by combining ESM2 embeddings + AlphaMissense + features
  - ESM2 embeddings contribute 83.7% of feature importance
  - Zhang et al. 2025 (eLife): ESM2 identifies conserved motifs in IDRs for phase separation
- **implication: use ESM2 embeddings as features, NOT just LLR scores**

### sticker-spacer framework (Wang 2018, Martin 2020)
- Tyr-Arg interactions drive liquid LLPS
- Gly spacers maintain fluidity
- Gln/Ser promote solidification
- aromatic valence + patterning determine liquid vs solid (Martin 2020, Science)
- uniformly distributed stickers → liquid; clustered → solid
- **this maps directly to our maturation grammar axis**

### LLPS and amyloid are separable (Boyko 2020, 2025)
- P301L tau: similar LLPS, much faster fibrillization
- FUS G156E: similar LLPS, accelerated solidification
- L-arginine inhibits condensate-to-fibril transition without perturbing LLPS
- surface nucleation at condensate interface (Gui et al. 2023)
- **different residues control each process — different feature axes needed**

### Das-Pappu diagram + charge patterning
- FCR, NCPR, kappa classify IDP conformational regimes (R1-R5)
- SCD superior to kappa for Rg prediction (accounts for contact order)
- kappa better for condensation behavior
- **need BOTH: SCD for conformation, kappa for condensation**

### thermodynamic framework (Zhang 2023, Michaels 2023)
- LLPS: low nucleation barrier (fast, reversible)
- amyloid: high nucleation barrier (slow, irreversible, thermodynamically favored)
- condensates create supersaturated microenvironment → accelerate amyloid nucleation
- competition ratio: LLPS barrier / fibril barrier determines pathway

---

## ralph-wiggin self-critique: phase 1

### what did we actually learn?
1. the gap is REAL — no multi-axis mechanism classifier exists
2. Frank et al. 2024 is the direct competitor but only binary
3. ESM2 embeddings (not LLR) are the right approach for IDPs
4. the molecular grammar (stickers/spacers) maps to maturation
5. LLPS and amyloid are thermodynamically separable → our multi-axis approach is well-motivated

### what's wrong with what we just did?
1. **training data problem**: we have 8 proteins with known mechanisms. Frank et al. trained on hundreds. 8 is not enough for ML. this is the elephant in the room.
2. **mechanism labels are fuzzy**: tau does BOTH amyloid and condensate. TDP-43 does BOTH pathways (Lin 2024). alpha-synuclein can also form condensates (Gracia 2022). the clean categories we proposed may not exist.
3. **ESM2 scanning was originally proposed as LLR — but LLR fails for IDRs**. we need to pivot to embeddings, which changes the feature extraction substantially.
4. **"classifier" with 8 training points is overfitting waiting to happen**. we need either more proteins OR a different approach (e.g., feature-space analysis rather than supervised classification).
5. **we haven't thought about what "validation" means**. with 8 proteins where we already know the answer, where's the test set?

### what would we do differently?
1. **reframe from "classifier" to "mechanism landscape"** — map proteins in multi-axis feature space, show clustering corresponds to known mechanisms, then use distances/positions to predict unknown proteins
2. **expand protein set** — add more IDPs with known mechanisms (IAPP, p53 TAD, SOD1, polyA-binding, etc.) to get to ~20-30 proteins
3. **use ESM2 embeddings not just LLR** — extract 1280-dim representations per residue, aggregate to protein-level features
4. **quantitative axes instead of labels** — instead of classifying "amyloid vs condensate", score each protein on continuous amyloid propensity + LLPS propensity + maturation risk. the combination tells the mechanism.
5. **leave-one-out validation** — for 8 known proteins, predict each using the other 7

### adjusted plan
the project should be:
1. compute four feature axes for each protein (polymer physics, maturation grammar, amyloid propensity, ESM2 embeddings)
2. visualize the mechanism landscape (PCA/UMAP of combined features)
3. show that known mechanisms cluster — amyloid proteins separate from condensate maturation proteins
4. show that mutations move proteins in the landscape (e.g., D262V hnRNPA1 shifts toward maturation)
5. validate predictions against experimental literature

this is more honest than a "classifier" with 8 training points.

---

## key references
- Frank et al. 2024, PNAS — ESM2 for LLPS + amyloid (direct competitor)
- Zhang et al. 2025, eLife — ESM2 conserved motifs in IDRs
- Farquhar 2026, Research Square — ESM2 embeddings rescue IDR prediction
- Martin et al. 2020, Science — aromatic patterning determines liquid vs solid
- Wang et al. 2018, Cell — molecular grammar (Tyr-Arg stickers)
- Das & Pappu 2013, PNAS — charge-hydropathy diagram
- Boyko et al. 2020, Nat Commun — P301L tau: separable LLPS and fibrillization
- Boyko et al. 2025, Nat Commun — LLPS and amyloid thermodynamically distinct
- Gui et al. 2023, Nat Chem — surface nucleation at condensate interface
- Horvath et al. 2022, Biochemistry — FuzDrop extension for condensate toxicity
- Lotthammer et al. 2024, Nat Methods — ALBATROSS
- Ginell et al. 2025, Science — FINCHES
- Pal et al. 2024, J Phys Chem Lett — SCD vs kappa comparison
