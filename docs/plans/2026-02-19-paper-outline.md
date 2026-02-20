# Paper Outline

## Working title

Evolutionary conservation is anti-predictive for gain-of-toxic-function mutations in intrinsically disordered protein regions

## The argument (one sentence)

Conservation-based pathogenicity predictors systematically invert for gain-of-toxic-function mutations in IDP motifs because pathogenicity in these genes depends on collective sequence properties of functional regions (NLS charge patterns, LCD composition, PrLD sticker density) that evolution does not conserve at the individual-residue level — and a two-step predictor combining functional region annotation with within-region conservation rescues performance.

## Target

JPC B (Article, not Letter). ~6000 words + figures. Combined Results and Discussion.

## Required new analyses (priority order)

1. Masked marginals on FUS (go/no-go gate) → script 09
2. Two-step predictor (determines publishability) → script 10
3. Bootstrap CIs for per-gene AUROCs → script 11
4. HTT exclusion + recompute → script 12
5. AlphaMissense comparison on FUS/TARDBP → script 13

## Risk assessment

| risk | probability | impact | mitigation |
|------|-------------|--------|------------|
| two-step predictor doesn't beat ESM2 alone | 20% | fatal | test quickly, abandon if <5% improvement |
| masked marginals fix FUS blind spot | 15% | fatal | run FUS first before two-step |
| reviewers say "just use AlphaMissense" | 60% | moderate | run AlphaMissense on FUS/TARDBP |
| HNRNPA1 P=3 criticized | 90% | low | present as hypothesis only |
| VUS-as-benign label noise | 70% | moderate | cite Pejaver 2022, note AUROC is lower bound |
