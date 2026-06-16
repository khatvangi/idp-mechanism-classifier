# Code / Science / Logic Audit (2026-02-20)

## Findings (ordered by severity)

1. High: Methods table does not match implemented features (reproducibility issue).
   - `scripts/mutation/03_disorder_and_features.py:13` uses `metapredict`, and `scripts/mutation/03_disorder_and_features.py:299` uses a `±15` window.
   - `scripts/mutation/16_supplementary_tables.py:291` states IUPred2A, and `scripts/mutation/16_supplementary_tables.py:293`-`scripts/mutation/16_supplementary_tables.py:302` describe `±10` windows and a different meaning for `local_p_lock`.
   - Paper-facing feature definitions are currently inaccurate.

2. High: Two-step headline AUROC is computed on a filtered subset, not all variants.
   - `scripts/mutation/10_two_step_predictor.py:246` skips held-out genes with single-class labels.
   - In current data this excludes 7 genes / 307 variants from scoring.
   - `data/variants/results_two_step.csv` does not report that denominator, so `AUROC=0.784` is valid for the evaluated subset only.

3. High: ESM2 baseline includes fabricated defaults for truncated HTT positions.
   - `scripts/mutation/06_approach_c_esm2.py:96`-`scripts/mutation/06_approach_c_esm2.py:100` truncates long proteins.
   - `scripts/mutation/06_approach_c_esm2.py:126`-`scripts/mutation/06_approach_c_esm2.py:136` fills out-of-range variants with zero/default features.
   - `scripts/mutation/06_approach_c_esm2.py:366`-`scripts/mutation/06_approach_c_esm2.py:382` still includes these rows in overall AUROC/AUPRC.

4. Medium: Mechanism-aware ensemble mixes incomparable score scales and uses oracle routing.
   - `scripts/mutation/08_mechanism_aware_model.py:255` uses known mechanism labels for routing.
   - `scripts/mutation/08_mechanism_aware_model.py:262`-`scripts/mutation/08_mechanism_aware_model.py:265` min-max scales LLR for some groups, while `scripts/mutation/08_mechanism_aware_model.py:302` uses model probabilities for others.
   - Cross-group ranking calibration is not principled.

5. Medium: Fold-skipping logic can bias overall metrics in approaches A/C.
   - Approach A initializes predictions to zero (`scripts/mutation/04_approach_a_xgboost.py:84`) and skips folds (`scripts/mutation/04_approach_a_xgboost.py:99`) but still computes overall AUROC on the full vector (`scripts/mutation/04_approach_a_xgboost.py:241`).
   - Similar pattern in approach C (`scripts/mutation/06_approach_c_esm2.py:189`, `scripts/mutation/06_approach_c_esm2.py:202`, `scripts/mutation/06_approach_c_esm2.py:366`).

6. Medium: Off-by-one ambiguity in FUS NLS boundaries across scripts.
   - `scripts/mutation/10_two_step_predictor.py:48` uses `(501, 526)` 0-indexed (positions 502-526, 25 aa).
   - `scripts/mutation/14_physics_calculations.py:69`-`scripts/mutation/14_physics_calculations.py:71` describes 501-526 and 26 aa.
   - `scripts/mutation/14_physics_calculations.py:97` filters from position 502, indicating inconsistency in prose/labels.

7. Medium: Narrative docs are stale vs current computed outputs.
   - `ANALYSIS.md:7` and `PROJECT.md:147` frame ESM2 LLR as the only useful signal.
   - Current outputs include stronger alternatives in-repo:
     - `data/variants/results_two_step.csv` (up to 0.784 AUROC on evaluated subset)
     - `data/variants/alphamissense_comparison.csv` (overall 0.768, FUS 0.870)

## Open Questions / Assumptions

1. Is the target claim "generalizable to unseen genes" or "best performance for known genes with curated region annotations"? Current two-step setup supports the latter.
2. Should supplementary tables describe exactly what current scripts compute, or intended future methods?

## Bottom Line

The pipeline runs and outputs are largely internally consistent, but key evaluation framing and method-description mismatches remain.  
As written, the project is not yet fully "doing what it says" at the manuscript/claim level until subset accounting, HTT handling, and method-table consistency are fixed.

