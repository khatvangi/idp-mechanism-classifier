# phase 5: validation against experimental literature

## summary

| prediction | verdict | confidence | key complication |
|------------|---------|------------|------------------|
| FUS matures, α-syn direct amyloid | PARTIALLY SUPPORTED | high for FUS; moderate for α-syn | α-syn can ALSO phase-separate (Ray 2020) |
| D262V accelerates maturation | STRONGLY SUPPORTED | high | steric zipper mechanism confirmed (Kim 2013) |
| P525L is localization-only | PARTIALLY SUPPORTED | moderate | also removes Transportin chaperoning (Hofweber 2018) |
| tau dual pathway | STRONGLY SUPPORTED | high | Wegmann 2018 + decades of direct amyloid work |
| TDP-43 M337V/Q331K | PARTIALLY SUPPORTED | moderate | both IMPAIR LLPS; M337V invisible to framework |
| E46K charge effect | STRONGLY SUPPORTED | high | charge-driven salt bridge disruption (Boyer 2020) |
| P301L shifts to amyloid | SUPPORTED | high | enhances fibrillization without altering LLPS |

**3 strong, 1 supported, 3 partial** = honest but not overwhelming validation.

---

## detailed validation

### 1. FUS LCD maturation vs α-synuclein amyloid

**our prediction**: FUS lcd_lockability_per_pair=0.49 >> α-syn NAC=0.15. FUS matures; α-syn doesn't.

**experimental evidence**:
- Patel et al. 2015 (*Cell* 162:1066-77): FUS forms liquid droplets that convert to solid over time. disease mutations accelerate this.
- Jawerth et al. 2020 (*Science* 370:1317-23): FUS condensates are aging Maxwell fluids with progressive viscoelastic maturation.
- Murray et al. 2017 (*Cell* 171:615-627): FUS LC droplets develop fibrils after ~144 hours.

**complication**: Ray et al. 2020 (*Nature Chemistry* 12:705-716) showed α-synuclein CAN undergo LLPS, and droplets mature into amyloid hydrogels. so α-syn isn't purely "direct amyloid" — it can also use a condensate pathway.

**verdict: PARTIALLY SUPPORTED**. FUS maturation is confirmed. α-syn's mechanism is more complex than our framework suggests. the 3.3× lockability difference is real but the binary framing is an overstatement.

### 2. D262V hnRNPA1 accelerates maturation

**our prediction**: +12.7% LCD lockability increase.

**experimental evidence**:
- Kim et al. 2013 (*Nature* 495:467-473): D262V creates a "steric zipper" motif (SYNVFG) that rapidly assembles into amyloid fibrils. WT peptide (SYNDFG) does not.
- Rhine et al. 2023 (*Nature Chemistry* 15:1513-21): D262V shows reduced FRAP recovery at condensate corona, indicating solid interfacial shell formation.
- Gui et al. 2019 (*Nature Communications* 10:2006): D262V forms irreversible fibrils; WT fibrils are reversible.

**verdict: STRONGLY SUPPORTED**. the grammar detects increased maturation potential (+12.7%) that matches the experimental finding of accelerated liquid-to-solid conversion. the actual mechanism (steric zipper) is more specific than the grammar models, but the direction is correct.

### 3. P525L FUS — localization vs maturation

**our prediction**: zero grammar change = localization mutation, not maturation.

**experimental evidence**:
- Dormann et al. 2010 (*EMBO Journal* 29:2841-57): P525L disrupts PY-NLS, reduces Transportin binding 10×, causes cytoplasmic mislocalization.
- Hofweber et al. 2018 (*Cell* 173:706-719): P525L also removes Transportin's chaperoning effect, which normally suppresses phase separation. so mislocalized FUS phase-separates MORE readily.

**verdict: PARTIALLY SUPPORTED**. our framework correctly identifies that P525L doesn't change sequence-intrinsic maturation potential (zero lockability change). the primary mechanism IS mislocalization. but P525L also indirectly enhances condensate formation by removing Transportin chaperoning — an interaction-dependent effect our sequence-only framework cannot capture.

### 4. tau dual pathway

**our prediction**: tau sits between amyloid and condensate clusters; amyloid propensity=1.10 (intermediate), LCD lockability=0.18 (intermediate).

**experimental evidence**:
- Wegmann et al. 2018 (*EMBO Journal* 37:e98049): tau undergoes LLPS, droplets become gel-like within minutes, thioflavin-S-positive aggregates form within days.
- Boyko et al. 2020 (*PNAS* 117:31882-90): LLPS accelerates fibrillar aggregate formation. pathogenic mutations enhance fibrillization but NOT droplet formation.
- Von Bergen et al. 2001 (*JBC* 276:48165-74): tau forms paired helical filaments directly from solution via heparin-induced nucleation-elongation.

**verdict: STRONGLY SUPPORTED**. tau genuinely has dual pathways — decades of direct amyloid work plus recent LLPS-to-amyloid evidence. our feature values correctly place tau at intermediate positions between amyloid and condensate clusters.

### 5. TDP-43 M337V and Q331K

**our prediction**: M337V changes maturation; Q331K changes charge. framework shows M337V Δ≈0, Q331K ΔSCD=-11.3%.

**experimental evidence**:
- Conicella et al. 2016 (*Structure* 24:1537-49): Q331K completely ABROGATES phase separation. M337V decreases turbidity ~50%. both disrupt a helical interface (residues 321-340) critical for LLPS.
- Conicella et al. 2020 (*PNAS* 117:5883-94): the alpha-helix in the LCD tunes LLPS; ALS mutations disrupt it.

**verdict: PARTIALLY SUPPORTED, with key limitation**. Q331K's charge effect (detected as -11.3% SCD change, -2.3% LCD lockability) aligns with the experimental finding that it disrupts LLPS via charge introduction. but M337V is invisible to our framework (Δ≈0 in all features) even though it experimentally reduces LLPS by 50%. the mechanism — local disruption of a helical interface — requires structural features we don't compute.

### 6. α-synuclein E46K charge effect

**our prediction**: E46K ΔSCD=-86.2% (largest feature change of any point mutation), +18.5% κ change.

**experimental evidence**:
- Boyer et al. 2020 (*PNAS* 117:3592-3602): cryo-EM structure shows E46K eliminates the E46-K80 salt bridge, avoiding a kinetic trap and reaching a deeper, more pathogenic fibril fold.
- Choi et al. 2004 (*FEBS Letters* 576:363-68): E46K increases liposome binding and filament assembly rate.

**verdict: STRONGLY SUPPORTED**. the massive SCD change captures a real electrostatic effect. E46K's pathogenesis is dominantly charge-driven — loss of the E46-K80 salt bridge reorganizes the fibril fold.

### 7. tau P301L shifts toward amyloid

**our prediction**: P301L shows +10% APR fraction increase, slight shift toward amyloid in landscape.

**experimental evidence**:
- Von Bergen et al. 2001 (*JBC*): P301L promotes aggregation by enhancing local beta-structure.
- Boyko et al. 2020 (*PNAS*): P301L increases aggregation RATE but does NOT affect LLPS propensity or droplet dynamics.
- Kanaan et al. 2020 (*Nature Communications* 11:2809): P301L expedites domain relaxation, exposing repeat domains for fibril formation.

**verdict: SUPPORTED**. P301L enhances fibrillization without affecting phase separation — shifting toward amyloid within the dual-pathway landscape. our +10% APR fraction change is directionally correct but modest. the real mechanism is local structural (proline removal enables beta-sheet stacking at PHF6) rather than compositional.

---

## critical assessment

### what the validation reveals about our framework

**strengths**:
1. correctly identifies maturation potential differences (FUS >> α-syn)
2. detects D262V's maturation-accelerating effect
3. captures tau's intermediate position
4. identifies E46K's charge disruption as the strongest point mutation effect

**weaknesses**:
1. **binary mechanism labels are wrong** — α-syn can phase-separate too; tau uses both pathways. the real biology is a continuum, not 5 discrete classes.
2. **local structural effects are invisible** — M337V disrupts a helical interface but produces Δ≈0 in our features. P301L removes a PPII-breaking proline but only changes APR by 10%. the framework misses structural mechanisms.
3. **interaction-dependent effects are undetectable** — P525L removes Transportin chaperoning; this can't be captured from sequence alone.
4. **the NAC region is too short** — scoring α-syn's 35-residue NAC with a 12-residue window grammar is at the edge of meaningful resolution.

### the honest conclusion

our framework captures COMPOSITIONAL differences between IDP disease mechanisms. proteins with different amino acid compositions aggregate through different pathways, and the features correctly reflect this. but the framework CANNOT capture:
- local structural effects of point mutations
- protein-protein interaction-dependent mechanisms
- environment-dependent pathway switching (same protein, different conditions → different mechanism)

this is a sequence composition landscape, not a mechanism predictor. the publication should frame it accordingly.
