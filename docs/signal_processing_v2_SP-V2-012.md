# SP-V2-012 — Scaling, Feature-Efficiency, and Harmonic-Modulation Mechanism Analysis

*This document was written in two passes — a daytime session (Motivation through the first pass of
Accepted/Rejected Interpretations) and an overnight continuation (the sample-scaling curve,
Artifact Verification §6–8, H1 re-verification §10–11, and the final Accepted/Rejected/Implications/
Final Feature Decisions sections). The overnight pass found that the daytime session's central
claim — that `inter_harmonic_modulation_coherence` is F010's standout feature — does not survive a
common-envelope-removal confound check, and revised the document's conclusions accordingly rather
than silently editing the daytime findings. See
[`SP-V2-012_overnight_summary.md`](SP-V2-012_overnight_summary.md) for a cold-read summary of the
full night's work, or read this document straight through — later sections supersede earlier
interpretations where they conflict, and say so explicitly each time.*

## Motivation

The real-corpus pilot (`signal_processing_v2_real_corpus_pilot.md`) found F010 (harmonic
modulation) individually the strongest of five feature families — the only medium effect size
(ε²=0.050) and the best classifier delta (+0.0136 macro F1, `situation`) in that pilot — but when
all five families' 92 features were combined, every family's marginal contribution turned
negative, most severely for F010 despite its individual strength. That result was left as an open
hypothesis (small-sample overfitting vs. genuine noise vs. redundancy vs. classifier-specific
artifact vs. sample-distribution artifact), not a conclusion. This phase decomposes that
hypothesis directly. **No new feature families are added. SP-V2-011 (adaptive harmonic model)
remains on hold.** F006/F007 stay frozen at REJECT.

## Current Hypotheses (RQ12-1 through RQ12-4)

- **RQ12-1**: Does F010's effect reproduce as sample size increases?
- **RQ12-2**: Is the 92-feature combination's performance drop a dimensionality/sample-size
  problem, or a noise/redundancy problem intrinsic to the new features?
- **RQ12-3**: What acoustic mechanism does F010 actually capture?
- **RQ12-4**: Are F008/F009 redundant with F010, or complementary?

## Dataset Availability

Full-corpus `situation` class counts (re-verified, audio files confirmed present):

| class | N |
|---|---|
| sadness | 9,126 |
| angry | 8,474 |
| happiness | 4,548 |
| disgust | 3,836 |
| neutral | 3,258 |
| fear | 2,967 |
| **surprise (minimum)** | **1,755** |

Maximum theoretically achievable *perfectly balanced* stratified sample: 1,755 × 7 = 12,285.
`majority_vote` (full corpus, 33,980 raw rows, 1,711 ambiguous excluded): sadness 12,532 (39%),
angry 5,978, happiness 4,329, neutral 3,767, disgust 2,382, fear 1,992, surprise 1,289 — the same
"surprise is the imbalanced-perceived-emotion minority" pattern the real-corpus pilot already
found in its own 1,400-file subsample, confirmed here at full-corpus scale. `gender`: female
23,846 / male 10,134 (70/30). `age`: 20–74, 32 distinct values, no corpus-batch variable beyond
the two source directories (4차년도/5차년도_2차) already accounted for in `similarity.py`. No
speaker ID exists anywhere in the metadata — unchanged from every prior audit in this project.

**Achievable sample-scaling levels, and the scope decision actually made**: per-file extraction
throughput for F008+F009+F010 together (excluding F006/F007, frozen this phase), measured directly
rather than assumed: **1.36 files/sec** with 6-way process parallelism on this machine. A full
3-seed × 8,400 (or higher) nested design, as the brief's initial target levels suggested, would
need on the order of 25,200+ new file extractions ≈ 5+ hours of background compute — judged not a
good use of a single session's compute budget relative to the marginal value of the highest N
levels. **Scope actually run: 3 seeds, nested N ∈ {1,400, 2,800, 5,600}** (800/class at the top
level) — a 4× range, sufficient to distinguish the brief's own Pattern A/B/C diagnostic (§6 of the
brief), at an estimated ~3.4 hours of background compute (3 × 5,600 ≈ 16,800 files). This is stated
explicitly as a compute-budget scoping decision, the same kind already made repeatedly earlier in
this project (280-file and 1,400-file pilots, not full-corpus runs) — not a claim that 5,600 is a
theoretical ceiling. The original pilot's existing 1,400-file, seed-42 extraction
(`real_corpus_features.csv`, non-nested — drawn by `RandomState.choice`, not the permutation-prefix
method this phase uses for genuine nesting) is kept and reused as an independent additional
reference point, clearly labeled as such wherever it appears below, not silently merged into the
new nested seeds.

**Nesting implementation note, recorded because it was a real bug caught mid-implementation**: a
first version generated each level's sample by calling `RandomState(seed).choice(pop, size=N)`
independently per N — this does **not** guarantee `N_1400 ⊂ N_2800`, because NumPy's `choice`
without replacement uses a size-dependent internal algorithm, so the same seed at different
requested sizes does not reliably produce nested output. Fixed by generating one random
**permutation** per class per seed once (`nested_permutation()`), then taking prefixes of
increasing length for each N level — a permutation's prefixes are nested by construction. A second
related issue: because extraction runs in parallel (`ProcessPoolExecutor` + `as_completed()`), rows
land in the output CSV in *completion* order, not *submission* order, so naively taking "the first
N rows per class" from the saved CSV would not reconstruct the correct nested subset either. Fixed
by re-deriving the canonical permutation at analysis time (deterministic given the same seed) and
selecting rows by `wav_id` membership rather than by file position (`sp012_scaling_common.
load_level_dataset`).

## Experimental Design

Every comparison below uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, applied
identically to baseline-alone and baseline+new conditions (paired folds). `situation` and
`majority_vote` are evaluated separately throughout, per this project's standing convention.

**M0–M7 model grid** (`signal_v2/evaluation/sp012_scaling_point.py`): baseline; baseline+F008;
+F009; +F010; +F008+F009; +F008+F010; +F009+F010; +F008+F009+F010. Run with Random Forest
(`n_estimators=200`, matching the real-corpus pilot's setting, held fixed rather than re-tuned to
this phase's results per the brief's own instruction) at every (N, seed, label) point — cheap
enough to run exhaustively.

**Classifier comparison** (`sp012_classifiers.py`): Random Forest (fixed hyperparameters, as
above), Elastic-Net multinomial logistic regression (`solver="saga"`, `StandardScaler` first,
`C`/`l1_ratio` chosen by `GridSearchCV` over `C∈{0.01,0.1,1,10}`, `l1_ratio∈{0.15,0.5,0.85}`, inner
`StratifiedKFold(3)` — strictly inside each outer training fold, never touching the outer test
fold), Linear SVM (`LinearSVC`, `C` tuned the same way). This is markedly more expensive than the
RF-only grid (nested CV means dozens of fits per outer fold) — **run only at the N range's
endpoints** (smallest and largest N, both extremes) rather than every point, an explicit
compute-scoping decision stated here rather than left implicit.

**Random-feature negative controls** (`sp012_negative_controls.py`): Gaussian noise matched to each
training fold's own per-column mean/std (never test-fold statistics), and label-permuted real
features (marginal distribution preserved, train/test each independently shuffled within itself) —
compared against `baseline`, `baseline+real_v2` on the same folds, for both F010-alone (33 dims)
and F008+F009+F010 (55 dims).

**Stability selection** (`sp012_classifiers.stability_selection_frequency`): across the Elastic-Net
fits' non-zero coefficients (any class), per new feature, tracked automatically as a byproduct of
the classifier-comparison runs.

## Feature-Efficiency Analysis (§9) — completed, N=1,400

Training-fold-only top-*k* selection (ranked by training-fold ε², re-ranked fresh in every outer
fold so no test-fold information enters the ranking) on the original 1,400-file pilot sample:

| k | combined macro F1 | Δ vs. baseline | folds improved |
|---|---|---|---|
| 1 | 0.2603 | **+0.0163** | 4/5 |
| 3 | 0.2334 | −0.0106 | 3/5 |
| 5 | 0.2530 | +0.0090 | 3/5 |
| 10 | 0.2460 | +0.0021 | 2/5 |
| 20 | 0.2478 | +0.0038 | 1/5 |
| all (33) | 0.2575 | +0.0136 | 3/5 |

**k=1 beats every other level, including all 33 features** — and the single feature selected is
`f010_inter_harmonic_modulation_coherence`, **in all 5 of 5 folds**, with no exceptions. This is
not a close or fold-dependent result: one specific feature is the load-bearing signal in this
family, non-monotonically surrounded by worse intermediate k values (k=3 is the single worst
result in the table, even worse than baseline) — consistent with **Pattern C** from the brief's own
diagnostic framework (§6): a sparse, strong signal diluted by a larger set of weaker/noisier
co-features, not a family that is uniformly strong or uniformly weak.

## F010 Mechanistic Analysis

All of this section uses the original 1,400-file pilot sample (`real_corpus_features.csv`,
already-extracted F008/F009/F010) merged with newly-extracted auxiliary quantities
(`aux_features_original_1400.csv`: duration/speaking-rate, F0 dynamics, energy dynamics, and a
generic whole-envelope modulation baseline) — chosen deliberately over waiting for the larger
nested extraction, since these questions don't need large N, only the confound variables, and
answering them doesn't block on the (much slower) scaling-curve extraction running in parallel.

### Duration / Speaking Rate (§12)

Spearman |ρ| between each of F010's 33 features and (`dur_total_sec`, `dur_voiced_sec`,
`dur_n_voiced_frames`, `dur_speaking_rate_proxy`): **0 of 33 features exceed |ρ|=0.7.** The
closest are the five `*_modulation_entropy` features (ρ=0.65–0.70, just under the strong-proxy
threshold) — elevated, and worth watching, but a meaningfully weaker duration dependency than
SP-V2-007's CWT entropy features showed on the same 1,400-file sample (ρ=0.86–0.96, clearly a
near-duplicate there). F010's entropy features are duration-*correlated* but not duration-
*redundant* in the way SP-V2-007's were.

### F0 Dynamics (§13)

0 of 33 features exceed |ρ|=0.7 against F0 mean/std/range/derivative-energy. The single strongest
correlation in this whole confound check belongs to the family's own top feature:
`inter_harmonic_modulation_coherence` ↔ F0 derivative energy, ρ=0.59 — a real, moderate
relationship (this feature is measuring something that co-varies with how much the pitch itself is
moving), not dominant enough to call it a redundant restatement of F0 dynamics, but large enough to
flag directly rather than omit.

### Energy Dynamics (§14) and Generic-Envelope Baseline

**This is the confound check with the clearest positive hit**: **10 of 33 F010 features exceed
|ρ|=0.7** against RMS-energy std/range — specifically, every `*_modulation_low_rate_energy` and
`*_modulation_high_rate_energy` sub-feature (ρ=0.85–0.90) across all 5 harmonics. Mechanistically
this is expected, not alarming on its own: these are *absolute-scale* modulation-energy
quantities computed from *absolute* (not H1-relative) per-harmonic amplitude trajectories, so of
course they scale with how much the utterance's overall loudness varies — an amplitude-modulation-
energy feature computed on an unnormalized signal will always partly re-encode the signal's own
energy dynamics. The `*_modulation_entropy`, `*_centroid`, and `inter_harmonic_modulation_coherence`
features do **not** show this same strong energy-confound pattern.

**The decisive comparison for the "harmonic-specific, not just any modulation" claim**
(dimension-matched, same classifier/fold setup as the pilot's own Evaluation E):

| condition | dims | Δ macro F1 (`situation`) | folds improved |
|---|---|---|---|
| Generic envelope modulation (whole-waveform RMS) | 6 | +0.0037 | 3/5 |
| F010, all harmonics | 33 | +0.0136 | 3/5 |
| **F010, H1 only (dimension-matched to generic envelope)** | **6** | **+0.0276** | **4/5** |

The dimension-matched H1-only subset beats *both* the generic-envelope baseline (by a wide margin
— 7.5×) *and* the full 33-feature F010 set. Two conclusions follow, and they point in different
directions on different questions: (1) **harmonic-specific modulation is doing something real that
whole-envelope modulation is not** — the generic-envelope control is the weakest performer here,
which is exactly what the brief's Case framework (§14) would call support for the
"harmonic-specific," not merely "modulation-spectrum," novelty claim; (2) **more harmonics is not
better** — going from H1-only (6 dims) to all 5 harmonics (33 dims) *loses* more than half the
improvement (+0.0276 → +0.0136), directly consistent with the top-*k* finding above (Pattern C,
sparse signal diluted by additional dimensions).

### Harmonic-Order Analysis (§15)

Per-harmonic top feature and its `situation` effect size (all five harmonics' top feature is
`*_modulation_entropy`, consistent with §14's finding that entropy — not the energy-confounded
sub-features — carries the real, less-confounded emotion signal within each harmonic):

| harmonic | top feature | ε² |
|---|---|---|
| H1 | modulation_entropy | 0.0409 |
| H2 | modulation_entropy | 0.0397 |
| H3 | modulation_entropy | 0.0278 |
| H4 | modulation_entropy | 0.0220 |
| H5 | modulation_entropy | 0.0333 |

A modest, not dramatic, low-harmonic advantage (H1/H2 strongest, H3/H4 weaker, H5 ticking back up
rather than continuing to decline) — consistent with, but not as sharp on its own as, the
classifier-level H1-only result above, which more decisively shows H1 carrying disproportionate
value once dimensionality (not just per-feature effect size) is accounted for.

### Modulation-Rate Analysis (§16) — a genuine, unanticipated correction to this family's own framing

The band boundaries in `harmonic_modulation.py`'s original design (`LOW_RATE_MAX_HZ=5.0`,
motivated at the time by a syllabic-rate rationale in the low single-digit-Hz-to-8Hz range commonly
cited in the speech-rhythm literature) were **not** derived from this corpus's actual observed
modulation rates — they were a plausible-sounding a priori choice. Checking the actual distribution
of every non-zero per-harmonic `dominant_freq` value across the 1,400-file sample: **median 0.54
Hz, p25=0.28 Hz, p75=1.12 Hz, max 9.49 Hz.** The real, dominant modulation rate in this corpus's
harmonic-amplitude trajectories is **far slower than syllabic rate** — closer to phrase/utterance-
level intonation drift (roughly one modulation cycle per 1–2 seconds) than to the ~4–8 Hz syllable-
rate range the original design's docstring assumed. This directly explains §14's energy/duration
confound findings: a modulation rate this slow is mechanistically close to "how the utterance's
overall loudness/pitch contour rises and falls once or twice across its length" — which is exactly
what utterance duration and energy-range also measure, from a different angle. Data-driven band
edges (33rd/67th percentile of the observed distribution, used descriptively here, not as a new
fixed design constant): slow < 0.35 Hz, fast > 0.87 Hz.

**This is reported as a correction, not hidden**: the original "harmonic modulation spectrum"
framing implicitly suggested capturing relatively fast temporal dynamics; what F010 actually
measures in this corpus is predominantly slow, phrase-scale amplitude drift. This does not
invalidate the feature (§18 below shows real signal survives controlling for the obvious slow-scale
confounds) but it does mean the feature should be *described* as phrase-level harmonic-amplitude
drift, not syllable-level modulation, in any future write-up.

### Harmonic Coherence — Exploratory Probe (§17)

Explicitly kept as a probe, not expanded into a new feature family this phase, per the brief's own
instruction. Extended `inter_harmonic_modulation_coherence` (a single all-pairs-average number) on
an independent 700-file subset (100/emotion, disjoint sampling from the main 1,400) into
adjacent-harmonic, distant-harmonic, and low-vs-high-harmonic-group coherence:

| measure | mean | ε² (`situation`) | direction |
|---|---|---|---|
| adjacent-harmonic coherence (H1-H2, H2-H3, ...) | 0.579 | 0.044 (medium) | `happiness` lowest ↔ `fear` highest |
| distant-harmonic coherence (\|i−j\|≥2) | 0.392 | 0.043 (medium) | `surprise` lowest ↔ `sadness` highest |
| low-group (H1-H2) vs. high-group (H4-H5) | 0.431 | 0.033 (small) | `surprise` lowest ↔ `sadness` highest |
| all-pairs mean (≈ F010's own feature, independent replication) | 0.466 | **0.047 (medium)** | `surprise` lowest ↔ `sadness` highest |
| all-pairs coherence *variance* within a file | 0.167 | 0.0006 (negligible) | no real emotion signal |

Two things worth noting: (1) the all-pairs-mean result (ε²=0.047) **independently replicates** the
main 1,400-file sample's `inter_harmonic_modulation_coherence` finding (ε²=0.050) on a disjoint
700-file subset — real cross-sample consistency, not a one-off; (2) it is the *level* of
cross-harmonic coherence that carries emotion signal, not how *variable* that coherence is across
different harmonic pairs within one file (the variance measure is flat). Adjacent harmonics are
consistently more coherent with each other than distant ones (0.579 vs. 0.392), a physically
sensible result (closer harmonics share more of the same underlying glottal-source amplitude
modulation), but the emotion effect doesn't concentrate more in one distance category than another.

### Partial Association (§18) — does the emotion effect survive confound control?

Residualized (OLS on `dur_total_sec + dur_speaking_rate_proxy + f0dyn_std + gender`, then
Kruskal-Wallis on the residual) for the five strongest F010 features:

| feature | raw ε² | controlled ε² | relative drop |
|---|---|---|---|
| `inter_harmonic_modulation_coherence` | 0.0501 | 0.0182 | **−64%** |
| `h1_modulation_entropy` | 0.0409 | 0.0313 | −23% |
| `h2_modulation_entropy` | 0.0397 | 0.0294 | −26% |
| `h5_modulation_entropy` | 0.0333 | 0.0295 | −12% |
| `h3_modulation_entropy` | 0.0278 | 0.0236 | −15% |

A genuinely mixed, not one-directional, result — and the direction is the opposite of what the
univariate confound checks (§12–14) alone would have predicted. `inter_harmonic_modulation_
coherence` had the *weakest* individual duration/F0/energy correlations of the two feature types
compared here, yet loses the *most* effect size (64%) under joint confound control; the entropy
features had *stronger* individual duration correlations (§12) yet retain 74–88% of their raw
effect size once duration, speaking rate, F0 variability, and gender are jointly controlled for.
Univariate redundancy and multivariate confound survival are measuring different things and do not
always point the same direction — a reminder not to over-interpret either check in isolation.
`inter_harmonic_modulation_coherence` still retains a real, non-trivial effect after control
(ε²=0.018, "small," not zero) — the emotion signal is reduced, not eliminated.

## H1-Specific Result — Overnight Re-Verification (§10–11)

Dimension-matched (6d each) per-harmonic and cumulative re-check, `situation` label, N=1,400:

| harmonic | Δ macro F1 (6d) | folds | max within-harmonic ε² |
|---|---|---|---|
| generic envelope (reference) | +0.0037 | 3/5 | — |
| **H1** | **+0.0276** | 4/5 | 0.0409 |
| H2 | −0.0094 | 2/5 | 0.0397 |
| H3 | +0.0121 | 4/5 | 0.0278 |
| H4 | +0.0004 | 2/5 | 0.0220 |
| **H5** | **+0.0143** | 3/5 | 0.0333 |

| cumulative | dims | Δ macro F1 | folds |
|---|---|---|---|
| H1 | 6 | **+0.0276** | 4/5 |
| H1+H2 | 12 | +0.0046 | 4/5 |
| H1-H3 | 18 | +0.0069 | 2/5 |
| H1-H4 | 24 | +0.0122 | 3/5 |
| H1-H5 (all) | 30 | +0.0097 | 3/5 |

Two findings, both already folded into Accepted Interpretations above: (1) H1 alone is the single
best individual harmonic and beats every cumulative addition on top of it — the dilution pattern
(Pattern C) holds at the per-harmonic level, not just at the whole-family level; (2) the per-
harmonic ranking (H1 > H5 > H3 > H4 > H2, distinctly non-monotonic) does not track a simple
"tracking reliability decreases with harmonic order" story, since H5 — presumably the least
reliably tracked of the five — outperforms H2, H3, and H4. This is evidence against, not for, the
"it's just tracking reliability" alternative explanation the brief asked to rule out (§11) — though
a full per-harmonic track-validity comparison (beyond H1/H5, which `deep_verification.csv` covers)
was not run, so this conclusion rests on the classifier-value ranking alone, not a direct
reliability measurement for H2–H4.

## F008/F009 Complementarity (§19–21)

Conditional value of adding F008/F009 on top of F010 alone, same paired-fold design, 1,400-file
sample:

| combination | `situation` Δ | folds | `majority_vote` Δ | folds | `majority_vote` marginal vs. F010 alone |
|---|---|---|---|---|---|
| F010 alone | +0.0136 | 3/5 | **−0.0204** | 2/5 | — |
| F010 + F008 | +0.0116 | 4/5 | **+0.0097** | 4/5 | **+0.0301** |
| F010 + F009 | +0.0062 | 3/5 | −0.0138 | 0/5 | +0.0067 |
| F010 + F008 + F009 | +0.0041 | 4/5 | −0.0304 | 1/5 | −0.0100 |

**F008 provides real, substantial complementary value to F010 specifically on `majority_vote`** —
the combination is the first clearly positive `majority_vote` classifier result found anywhere in
this whole SP-V2 line (every other family and combination tested across SP-V2-006 through 010's
real-corpus work has come back flat-to-negative on that label). For `situation`, every addition to
F010 alone is neutral-to-negative — F010 alone remains the best `situation` result. F009 does not
show the same complementary pattern on either label.

**Redundancy matrix** (representative features, Spearman ρ): within-family correlations are large
as expected (F008's three shown features are mutually |ρ|=0.87–0.97, all derived from the same
aperiodicity spectrum; F009's two are ρ=−0.78, near-tautological since one is 1 minus the other).
**Cross-family correlations are consistently weak**: F008↔F009 near zero (|ρ|≤0.04), F008↔F010
low-moderate (|ρ|=0.13–0.30), F009↔F010 low-moderate (|ρ|=0.07–0.42). The three families are not
restating the same underlying signal — supporting the F008+F010 complementarity result as a real
effect, not a correlation artifact.

**Cross-label consistency**: F010's top feature is `inter_harmonic_modulation_coherence` for
*both* labels, and its `majority_vote` effect size is larger (ε²=0.077) than its `situation` effect
size (0.050) — a good sign of genuine, label-scheme-independent acoustic signal. F009 is similarly
consistent (same feature type both labels, situation slightly stronger). **F008's top feature
changes between labels** — `mid_band_periodicity_std` for `situation` vs. `n_voiced_frames` (a
frame-count/duration proxy) for `majority_vote` — a flag that F008's `majority_vote` association
specifically may need the same duration-confound scrutiny given to F010, not yet performed for
F008 in this phase.

## Random-Feature Negative Controls, Classifier Comparison, and the Sample-Scaling Curve

*(Partially complete — both N=1,400/seed 42 endpoint points (`situation` and `majority_vote`) are
in, including the full M0–M7 grid, negative controls, and 3-classifier comparison. The N=5,600
endpoint and seeds 44/45 were still extracting in the background as this section was drafted; this
section will be extended, not rewritten, once those land — everything below is final for the
N=1,400 results it reports.)*

### N=1,400 endpoint results (seed 42) — both labels, full detail

**M0–M7 RF ablation grid**:

| model | `situation` Δ | folds | `majority_vote` Δ | folds |
|---|---|---|---|---|
| M1 (F008) | −0.0073 | 1/5 | −0.0048 | 2/5 |
| M2 (F009) | +0.0020 | 3/5 | −0.0137 | 1/5 |
| M3 (F010) | +0.0136 | 3/5 | −0.0204 | 2/5 |
| M4 (F008+F009) | +0.0018 | 3/5 | −0.0074 | 3/5 |
| M5 (F008+F010) | +0.0018 | 2/5 | +0.0014 | 2/5 |
| **M6 (F009+F010)** | **+0.0130** | **5/5** | +0.0008 | 3/5 |
| M7 (all three) | +0.0018 | 3/5 | −0.0153 | 0/5 |

A new result not visible in the real-corpus pilot's single-condition tests: **M6 (F009+F010)
matches F010-alone's `situation` benefit almost exactly (+0.0130 vs. +0.0136) while achieving
perfect 5/5-fold consistency**, where F010 alone only hit 3/5. F009 appears to *stabilize* F010's
benefit across folds even though F009 alone contributes little and its own marginal addition to
F010 was measured as slightly negative in the complementarity analysis above (§19, computed on RF
only) — a reminder that "marginal ΔF1" and "marginal fold-consistency" are different, both useful,
quantities that can point in different directions for the same feature pair.

**Random-feature negative controls (RF only, both probes)**:

| condition | `situation`: F010-alone (33d) | `situation`: F008+F009+F010 (55d) | `majority_vote`: F010-alone | `majority_vote`: F008+F009+F010 |
|---|---|---|---|---|
| baseline | 0.2440 | 0.2440 | 0.2943 | 0.2943 |
| real v2 | **0.2575** | 0.2458 | **0.2739** | 0.2790 |
| Gaussian noise | 0.2328 | 0.2322 | 0.2889 | 0.2608 |
| permuted v2 | 0.2371 | 0.2380 | 0.2880 | 0.2888 |

For `situation`: real features clearly beat baseline, and both noise controls fall clearly below
baseline — confirming a genuine "cost of adding dimensions" exists (Gaussian/permuted both hurt),
which F010's real signal overcomes with room to spare. This is the brief's own **Case 3** (§7):
real > baseline > random. For `majority_vote`, the picture inverts in a way that turns out to be
diagnostic rather than simply discouraging: **real F010 (0.2739) scores *below* both noise controls
and baseline** under RF — the worst of the four numbers in that column. Read alone, this would
suggest F010 actively hurts `majority_vote` more than random dimensions would. **It does not** —
see the classifier comparison immediately below, which shows this is specifically an RF limitation,
not a property of the feature set. This is an important methodological finding in its own right:
**a negative-control test run on only one classifier can be actively misleading about a feature
set's real value**, exactly the kind of failure mode multi-classifier comparison (§8 of the brief)
was designed to catch, and did catch here.

### Classifier comparison — the clearest single finding in this phase

| probe | classifier | `situation` Δ | folds | `majority_vote` Δ | folds |
|---|---|---|---|---|---|
| F010 alone | RF | **+0.0136** | 3/5 | **−0.0204** | 2/5 |
| F010 alone | Elastic-Net | −0.0004 | 2/5 | +0.0034 | 4/5 |
| F010 alone | Linear SVM | −0.0039 | 1/5 | **+0.0121** | **5/5** |
| F008+F009+F010 | RF | +0.0018 | 3/5 | −0.0153 | 0/5 |
| F008+F009+F010 | Elastic-Net | +0.0017 | 2/5 | **+0.0090** | 4/5 |
| F008+F009+F010 | Linear SVM | +0.0011 | 2/5 | −0.0003 | 3/5 |

**The RF/linear split is not just present, it *reverses* by label.** For `situation`, Random Forest
is the only classifier that finds real value in F010 alone (+0.0136); Elastic-Net and SVM are flat
to slightly negative. For `majority_vote`, that pattern **flips**: RF is clearly negative (−0.0204,
the worst number in the whole table) while **Linear SVM is positive in all 5 of 5 folds**
(+0.0121) and Elastic-Net is positive in 4 of 5 (+0.0034). This directly answers part of RQ12-2:
**a meaningful share of what looked like "F010 doesn't help `majority_vote`" in the real-corpus
pilot and in this phase's own RF-only ablation grid was a classifier-choice artifact, not a
property of the feature.** The right linear model, given the exact same 33 features and the exact
same folds, extracts real and unusually *consistent* signal from F010 for that specific label.
For the combined 55-feature set, Elastic-Net is the best-performing classifier on `majority_vote`
(+0.0090, 4/5) — consistent with the pattern that linear models, not RF, are where this label's
F010-derived signal shows up most reliably.

### Sample-scaling curve — RF-only M0–M7 grid, mean across seeds

All 3 nested seeds (43, 44, 45) extracted overnight (5,600 files each, 0 errors); combined with
the original seed 42 pilot at N=1,400, this gives up to 4 independent seeds at N=1,400 and 3 at
N=2,800/5,600 (`majority_vote`'s per-seed *n* varies slightly, 1,307–1,323 / 2,594–2,644 /
5,254–5,401, from ties dropping differently per random draw — bucketed by *nominal* N, i.e. by
`n_per_emotion×7`, not exact row count). Mean Δ macro F1 across available seeds:

**`situation`** (all three N levels now have 3+ independent seeds — N=5,600 updated after seed 43's
classifier-comparison job, which also produces the RF-only ablation grid, finished)

| model | N=1,400 (4 seeds) | N=2,800 (3 seeds) | N=5,600 (3 seeds) |
|---|---|---|---|
| M1 (F008) | +0.0044 | −0.0077 | +0.0053 |
| M2 (F009) | +0.0025 | −0.0037 | −0.0048 |
| M3 (F010) | +0.0028 | −0.0060 | +0.0039 |
| M4 (F008+F009) | +0.0050 | −0.0082 | −0.0003 |
| M5 (F008+F010) | +0.0038 | +0.0016 | +0.0014 |
| M6 (F009+F010) | +0.0061 | −0.0124 | −0.0009 |
| M7 (all three) | +0.0040 | −0.0046 | +0.0018 |

**`majority_vote`**

| model | N=1,400 (4 seeds) | N=2,800 (3 seeds) | N=5,600 (3 seeds) |
|---|---|---|---|
| M1 (F008) | +0.0009 | +0.0002 | +0.0035* |
| M2 (F009) | −0.0059 | −0.0105 | −0.0063* |
| M3 (F010) | +0.0001 | −0.0015 | +0.0008* |
| M4 (F008+F009) | −0.0008 | +0.0007 | −0.0025* |
| **M5 (F008+F010)** | **+0.0059** | **+0.0076** | **+0.0072*** |
| M6 (F009+F010) | −0.0017 | −0.0056 | −0.0023* |
| M7 (all three) | −0.0013 | +0.0027 | +0.0066* |

*(`majority_vote` N=5,600 marked `*` still reflects only seeds 44/45 (2 seeds) — seed 43's
`majority_vote` classifier-comparison point was launched immediately after its `situation`
counterpart finished and had not yet returned when this table was last updated; will be folded in
once available, but is not expected to change the qualitative picture given `situation`'s addition
moved every model by less than 0.005.)*

**This changes the picture substantially from the single-seed (seed-42-only) results reported
earlier in this document.** Findings, updated with the now-complete 3-seed `situation` picture:

1. **The single-seed N=1,400/seed-42 result for F010-alone (+0.0136) was partly a favorable-seed
   outlier.** Averaged across 4 seeds at the same N, F010-alone's true mean effect is only +0.0028
   — an order of magnitude smaller. With all three N levels now at 3+ seeds, the `situation`
   trajectory for F010 alone is +0.0028 → −0.0060 → +0.0039 — noisy, non-monotonic, and small in
   magnitude throughout, closer to **Pattern B** (flat/noisy) than Pattern A (clean recovery with
   N), though the N=5,600 value is not clearly worse than N=1,400's either. M7 (all three families
   combined) follows the same noisy-but-small pattern (+0.0040 → −0.0046 → +0.0018) — the honest
   summary remains **"no reliably detectable net effect at any tested N," not "hurts" or "helps."**
2. **M5 (F008+F010) is the one entry in either table with a consistent, same-sign effect across all
   three N-levels and both labels' worth of evidence** — for `majority_vote` specifically, +0.0059 /
   +0.0076 / +0.0072, remarkably stable in magnitude as N grows from 1,400 to 5,600, and positive in
   every seed contributing to each of those three averages. This is the single most replicated,
   least-seed-dependent positive finding in this entire scaling analysis — stronger evidence than
   the daytime session's single-seed +0.0301 marginal-contribution number, precisely because it is
   an *average over multiple independent draws*, not one lucky split.

### Negative-control scaling across N (§15) — a cleaner, more reassuring pattern than the raw deltas alone suggest

Mean absolute macro F1 (not delta) across available seeds, real F010/combined features vs. their
dimension-matched Gaussian-noise and label-permuted controls:

| N | probe | real | Gaussian noise | permuted | baseline |
|---|---|---|---|---|---|
| 1,400 | F010 alone | 0.2448 | 0.2372 | 0.2384 | 0.2419 |
| 2,800 | F010 alone | 0.2702 | 0.2569 | 0.2506 | 0.2762 |
| 5,600 | F010 alone | 0.2948 | 0.2829 | 0.2785 | 0.2938 |
| 1,400 | F008+F009+F010 | 0.2459 | 0.2339 | 0.2257 | 0.2419 |
| 2,800 | F008+F009+F010 | 0.2716 | 0.2493 | 0.2503 | 0.2762 |
| 5,600 | F008+F009+F010 | 0.2936 | 0.2689 | 0.2677 | 0.2938 |

(`majority_vote` shows the identical qualitative pattern — omitted for space, same conclusion.)

**This is the cleanest, most consistent result of the whole scaling analysis**: at *every* N level,
for *both* probes, on *both* labels (12 of 12 comparisons), **real features beat both noise
controls**, and the gap is not small (typically 1–3 points of macro F1). The dimensionality penalty
identified in the daytime session is confirmed here as a genuine, structural, *highly reproducible*
effect — Gaussian noise and permuted-real-features reliably score below baseline at every N tested,
not a one-off. Real F010/combined features land close to baseline (sometimes fractionally above,
sometimes fractionally below — never far in either direction) rather than clearly exceeding it. The
most defensible summary, reconciling this with the noisier raw-delta table above: **F010 does not
reliably improve macro F1 over the 117-feature baseline in this RF-only, seed-averaged view, but it
reliably does not act like noise either** — every equal-dimensional random control tested, at every
sample size tested, does measurably worse than either the real features or the baseline. This is a
narrower, more defensible claim than either "F010 helps" or "F010 is worthless," and it is the
claim the full weight of this section's evidence actually supports.

### N=5,600 classifier-comparison endpoint (seed 43) — `situation` complete

Took roughly 2 hours of background compute (vs. ~25–30 minutes at N=1,400 — Elastic-Net's `saga`
solver scales poorly with N); `majority_vote` was launched immediately after and may not be
reflected below depending on when this was last read.

| probe | classifier | N=1,400 Δ | N=5,600 Δ |
|---|---|---|---|
| F010 alone | RF | +0.0136 | +0.0098 |
| F010 alone | Elastic-Net | −0.0004 | **+0.0067** |
| F010 alone | Linear SVM | −0.0039 | **+0.0037** |
| F008+F009+F010 | RF | +0.0018 | +0.0059 |
| F008+F009+F010 | Elastic-Net | +0.0017 | +0.0078 (5/5 folds) |
| F008+F009+F010 | Linear SVM | +0.0011 | **+0.0100** (best of the six) |

**At larger N, the RF-only advantage for F010-alone that was so stark at N=1,400 mostly
disappears — all three classifiers agree it helps, modestly, at N=5,600** (+0.0098 / +0.0067 /
+0.0037 — RF still largest, but Elastic-Net and SVM are no longer flat-to-negative). The combined
55-feature set improves further and more consistently at N=5,600 across every classifier, with
Linear SVM now the single best performer of any cell in this table.

**Caution against over-reading this as confirmed Pattern-A recovery**: this is *one* seed's result
at the large-N endpoint, compared against a *different single seed* (42) at N=1,400 — exactly the
seed-to-seed variance that the multi-seed RF-only table above showed can be substantial (the
single-seed N=1,400/seed-42 F010-alone result of +0.0136 turned out to be a favorable outlier
against a 4-seed mean of +0.0028). This table is suggestive that classifier-choice sensitivity may
shrink with N, but it is not yet a multi-seed-confirmed claim — the classifier-comparison design
was deliberately scoped to 2 endpoints total (§ Experimental Design), and a proper answer would
need the same 3-seed replication the RF-only grid received, not done here given the per-point
compute cost (~25 min to ~2 hrs depending on N). Flagged as an open item, not a resolved one.

### `majority_vote` classifier-comparison endpoint (seed 43, N=5,600) — completes all 4 planned endpoints

| probe | classifier | N=1,400/seed 42 Δ | N=5,600/seed 43 Δ |
|---|---|---|---|
| F010 alone | RF | **−0.0204** | **+0.0046** |
| F010 alone | Elastic-Net | +0.0034 | −0.0064 |
| F010 alone | Linear SVM | +0.0121 (5/5 folds) | −0.0053 |
| F008+F009+F010 | RF | −0.0153 (0/5 folds) | +0.0087 |
| F008+F009+F010 | Elastic-Net | +0.0090 (4/5 folds) | −0.0019 |
| F008+F009+F010 | Linear SVM | −0.0003 | −0.0019 |

**The RF/linear-classifier split for `majority_vote` reverses *again* between N=1,400 and
N=5,600** — at N=1,400, RF was the worst performer and Linear SVM the best (5/5 folds); at
N=5,600, RF is now positive and both linear models are negative. Combined with `situation`'s own
partial convergence at N=5,600 (§ above), **the honest conclusion is that classifier-choice
sensitivity for these features is itself unstable across seeds and sample sizes, not a fixed,
generalizable property of any one classifier/label pairing.** This is a materially different (and
more cautious) conclusion than either the daytime session's "RF is where the signal shows up" or
the overnight session's earlier "the split reverses cleanly by label" framing — both were partial
readings of a noisier underlying pattern. **All four planned classifier-comparison endpoints
(seed 42/N=1,400 × 2 labels; seed 43/N=5,600 × 2 labels) are now complete.**

## Artifact Verification (§6–8) — 0.54 Hz check, common-envelope removal, tracker-quality confound

Overnight additions, run on the same 1,400-file sample plus a new `deep_verification.csv` pass
(peak-bin index, zero-padding sensitivity, Welch PSD cross-check, common-envelope-relative HMC,
and per-harmonic track-validity ratios — none of these were saved by the original F010 extraction,
so this required a fresh, targeted re-extraction, not a re-analysis of existing columns).

### §6: is the 0.54 Hz dominant frequency real, or a duration/FFT-resolution artifact?

Mixed, genuinely inconclusive-leaning-artifact evidence, reported in full rather than rounded to a
single verdict:

- **Correlation with 1/duration**: `dominant_freq` correlates with `1/utterance_duration` at
  Spearman ρ=0.23–0.26 (H1 and H3, both p≪0.001) — a real, moderate relationship. Not strong enough
  to say the frequency estimate *is* purely `1/duration` in disguise (that would need ρ near 1), but
  strong enough that duration is a genuine partial driver of where this "dominant frequency" lands.
- **Peak-bin distribution**: 37.6% (H1) / 27.3% (H3) of files have their dominant peak at the very
  first non-DC FFT bin; ~52%/42% at bin ≤2. A substantial minority-to-near-half of files are
  reporting "the slowest thing resolvable given this utterance's length," which is exactly the
  artifact pattern the brief was concerned about — but *most* files (48–58%) are not at the extreme
  edge, so it is not a universal artifact either.
- **Zero-padding stability**: median frequency shift under 8× zero-padding is only 0.02–0.03 Hz
  (small relative to the ~0.5 Hz typical dominant frequency) — the identified peak is a genuine
  local spectral maximum that persists as the frequency grid is refined, not an artifact of exactly
  where a coarse grid happens to land. This is evidence *against* pure discretization artifact.
- **FFT-peak vs. Welch PSD cross-check**: only moderate rank agreement (Spearman ρ=0.35–0.43)
  between the two independent spectral estimators, though their *central tendency* agrees well
  (median Welch/FFT ratio ≈1.00–1.09). One single-file spot-check during development showed a
  dramatic disagreement (0.30 Hz vs. 4.32 Hz for the same trajectory) that turned out to be an
  outlier, not representative of the full-sample pattern — a useful reminder not to generalize from
  one example, caught before it became a false claim in this document.
- **Duration-matched subset** (utterances 4.61–5.85 s, n=286): `dominant_freq`'s *own* emotion
  association is small-to-negligible in both the full sample and the duration-matched subset for H1
  (0.0043→0.0034) but *increases* for H3 (0.0015→0.0211) — inconsistent across harmonics and not
  large enough either way to be the basis of a strong claim. Worth noting: `dominant_freq` itself
  was never the carrier of F010's main emotion signal (§9's top-*k* analysis and §7 below both point
  at `inter_harmonic_modulation_coherence` and the entropy features instead), so this specific check
  bears less on the family's headline result than §7 does.

**Verdict: INCONCLUSIVE, artifact-leaning but not conclusively artifact.** The "~0.54 Hz, phrase-
scale" framing from the daytime mechanistic analysis should be treated as **not fully established**
— real duration-correlation and a large near-bin-1 minority both push toward "at least partly a
resolution artifact," while the zero-padding stability result pushes the other way. This section
does not resolve which explanation dominates; it establishes that the original framing was
under-qualified and should not be repeated as settled fact.

### §7: common-envelope removal — the most important result of the overnight session

**`inter_harmonic_modulation_coherence` substantially collapses once the shared/common amplitude
envelope is removed.** Computing a per-frame common envelope `E(t)` (the geometric mean, i.e.
mean-of-logs, across the 5 harmonics' amplitudes) and re-deriving HMC on `log A_k(t) − log E(t)`
(each harmonic's amplitude *relative to* the shared envelope, not its absolute value):

| quantity | raw HMC | envelope-removed (relative) HMC |
|---|---|---|
| correlation between the two versions | — | **ρ = −0.026** (essentially uncorrelated) |
| emotion `situation` ε² | 0.0501 (medium), p=2.6×10⁻¹⁴ | **0.0030 (negligible), p=0.118 — not even significant** |
| gender ε² | **0.2868 (very large)** | **0.0037 (negligible)** |
| correlation with `energy_std` | ρ=0.120 | ρ=0.094 |

The envelope-removed version is not merely a weaker copy of raw HMC — it is **statistically
independent of it** (ρ=−0.026) and carries essentially none of either the emotion signal or the
large gender confound that made raw HMC this pilot's standout feature. This is direct, strong
evidence that **raw HMC's apparent signal is substantially — perhaps almost entirely — a measure of
whether all harmonics share a common amplitude envelope (i.e., overall loudness/energy contour
shape), not of genuine, envelope-independent coordination between individual harmonics.**

**This requires walking back, not merely qualifying, the earlier "Accepted Interpretation" that
`inter_harmonic_modulation_coherence` is F010's standout, load-bearing, harmonic-specific feature.**
That characterization is not supported once the shared-envelope confound is properly controlled.
The finding is reported here exactly as it came out, per the brief's own explicit instruction not
to quietly protect an earlier conclusion — see the revised Accepted/Rejected Interpretations below.

Two things this does *not* invalidate, worth stating precisely rather than over-correcting: (1)
this is a check on **HMC specifically** (a *cross*-harmonic coherence measure) — it says nothing
directly about the separate, still-standing finding that **H1's own modulation-spectrum shape**
(entropy/rate/centroid — a *single*-harmonic measure, not a cross-harmonic one) beats a
generic-whole-envelope control (§10 below); those are different quantities probing different
questions, and only one of them just failed a confound check. (2) the entropy features (§10, §15 of
the daytime analysis) were not re-tested here and should not be assumed to survive or fail this same
check without their own verification — flagged as an open item, not resolved by this section.

### §8: harmonic-tracker-quality confound

`inter_harmonic_modulation_coherence` correlates with two track-quality proxies — the fraction of
frames where all 5 harmonics were simultaneously trackable (`track_full_continuity_ratio`, ρ=0.245)
and overall per-harmonic frame validity (`track_overall_valid_ratio`, ρ=0.246) — both moderate,
real, but not dominant relationships (p≪0.001, but ρ≈0.25 leaves most of the variance unexplained).
Controlling for these two proxies plus gender via residualization: HMC's `situation` ε² drops from
0.0501 to **0.0209** — retains about 42% of its raw effect size, a real but partial confound,
markedly less severe than §7's near-total collapse under envelope removal. **Tracker quality is a
real, secondary confound on HMC, not the primary explanation for its apparent signal** — the
common-envelope confound (§7) is by far the larger effect.

## situation vs. majority_vote

Covered inline above per-section (§21 for F008/F009/F010's cross-label top-feature comparison; the
complementarity table for F008+F010's asymmetric label-dependent value), plus the overnight scaling
curve, which adds one further point: **the two labels respond to these feature families in
opposite, not just different, ways.** On `situation`, no combination shows a reliable positive
effect once averaged across seeds (M7 hovers near zero at every N). On `majority_vote`, the one
combination that *is* reliably positive (M5, F008+F010) is specifically positive *there* and not
particularly on `situation` (situation M5: +0.0038 → +0.0016 → −0.0029, no consistent sign).
Combined with the N=1,400 classifier-comparison finding that RF and SVM disagree in *opposite
directions* by label for F010 alone, the picture that emerges is not "these features help emotion
recognition" in a general sense — it is "F008+F010 specifically helps recover *perceived* (rater-
majority) emotion, through a classifier/label-specific mechanism not yet explained, while showing
no reliable benefit for *scripted* (situation) emotion at all." This asymmetry is itself a finding,
not a nuisance to average away.

## Accepted Interpretations

- **H1's own modulation-spectrum shape is harmonic-specific, not a proxy for generic whole-envelope
  modulation** — a dimension-matched comparison shows H1-only (6 dims) beating a generic-envelope
  control by 7.5×, and this specific comparison has *not* been undermined by the overnight
  confound checks (§7 tested cross-harmonic coherence, a different quantity — see below). This is
  now the best-supported single "harmonic-specific" claim in the whole family.
- **More harmonics dilutes rather than helps** at this feature-to-sample ratio (Pattern C): H1-only
  beats every cumulative addition (H1+H2, H1-H3, ..., all-5) non-monotonically, confirmed again in
  the overnight H1-cumulative re-check (§10).
- **The "low harmonics are stronger because they're more reliably tracked" alternative explanation
  is not well supported**: H5 (presumably the least reliably tracked harmonic) scored the
  *second*-best individual classifier delta (+0.0143) of the five harmonics, ahead of H2, H3, and
  H4 — a monotonic tracking-reliability story would predict H5 last, not second.
- **F008 provides real complementary value to F010 specifically for `majority_vote`** — replicated
  across 2–4 seeds at all three tested N levels (§ Sample-scaling curve) — the single most
  consistently reproduced positive finding in this document.
- **Classifier choice matters, but not in a stable, predictable direction** — revised down from an
  earlier, cleaner-sounding claim. At N=1,400/seed 42, RF was the only classifier finding value in
  F010-alone for `situation`, while for `majority_vote` the pattern reversed (RF negative, SVM
  positive in 5/5 folds). At N=5,600/seed 43, that `majority_vote` pattern **reversed again** (RF
  now positive, both linear models negative), and `situation`'s RF-only advantage largely
  *disappeared* (Elastic-Net and SVM both turned positive too). **The right conclusion is that
  classifier sensitivity for these features is itself noisy and seed/N-dependent, not that any one
  classifier is "the" lens for this signal** — a more cautious, less quotable, but more honest
  reading of four data points than either single comparison alone would suggest.

## Rejected / Revised Interpretations

**Overnight reversal — the most important correction in this document.** The daytime session's
central claim, that `inter_harmonic_modulation_coherence` is "F010's single load-bearing,
harmonic-specific feature," is **not supported once the common-envelope confound is controlled**
(§7, overnight): envelope-removed HMC is statistically independent of raw HMC (ρ=−0.026) and
retains essentially none of raw HMC's emotion association (ε² 0.050→0.003, no longer significant)
or gender association (ε² 0.287→0.004). The convergent evidence that made HMC look like the
standout feature during the day — 5/5-fold top-*k* selection, the 700-file replication, the
cross-label consistency — all remains true as *observations about raw HMC*, but the mechanistic
conclusion drawn from them (that this reflects genuine harmonic-specific temporal coordination) was
premature. **The more defensible reading now is that raw HMC is substantially a proxy for shared
amplitude-envelope dynamics** — related to, and likely largely redundant with, ordinary energy/
loudness-contour information already available elsewhere in the feature set, not a novel
harmonic-organization signal. This is flagged as a correction, not a silent edit: the daytime
findings that produced the original claim are preserved above and in the daytime sections of this
document exactly as they were computed; only the *interpretation* built on top of them is revised,
per §19/§20 of the brief.

- The real-corpus pilot's framing of F010 as strong "on every criterion except small-sample
  classifier performance" undersold how *concentrated* that strength is — it is not that all 33
  features are individually weak-but-jointly-fine; one feature (HMC) does most of the work in the
  raw feature set, and that one feature is now itself in question per the paragraph above.
- The original modulation-rate framing ("harmonic amplitude modulation, syllable-rate and faster")
  is not what the data shows; revised (§6, overnight) to "possibly phrase-scale, but the evidence is
  mixed and partly consistent with a duration/FFT-resolution artifact — not established either way."
- **`inter_harmonic_modulation_coherence` is downgraded from "load-bearing standout feature" to
  "confounded, likely not primarily a harmonic-organization signal"** — see Final Feature Decisions.

## Implications for SP-V2-011

**Final call: HOLD.** Neither the daytime nor the overnight evidence points at harmonic-tracker
imprecision as a limiting factor. The one feature that survived every overnight confound check
(H1-specific modulation shape) is a single-harmonic measure, less exposed to multi-harmonic
tracking-continuity problems than a cross-harmonic quantity would be — and it is *already* the
best-performing candidate, not a candidate blocked by tracking noise. The feature that failed
overnight (HMC) failed because of a representation confound (shared amplitude envelope), which a
more precise adaptive harmonic tracker would not fix — tracking H1–H5 more accurately does not
change the fact that they mostly move together with overall loudness. Per the brief's own §29
criteria (adaptive harmonic modeling should be pursued only when tracking precision is
demonstrably the bottleneck), that condition is not met. **Do not start SP-V2-011.**

## Final Feature Decisions

| Feature | Decision | Basis |
|---|---|---|
| F008 (aperiodicity/MVF), alone | CONDITIONAL | Best standalone confound/redundancy profile (real-corpus pilot); RF-only scaling shows no clear net solo benefit; its value is clearest in combination (below) |
| F009 (harmonic phase), alone | CONDITIONAL → leaning REJECT | Weak and *worsening* with N on `situation` (+0.0025 → −0.0037 → −0.0081); no clear signal on `majority_vote` either |
| F010, all 33 features (as a block) | **REJECT** | Diluted by co-features at every N tested (Pattern C); negative-control scaling shows it only marginally beats equal-dimensional noise once seed-averaged |
| `inter_harmonic_modulation_coherence`, alone | **REJECT** (downgraded from a same-day provisional KEEP) | Collapses under common-envelope removal — ε² 0.050→0.003, no longer significant; substantially explained by shared loudness-contour dynamics, not harmonic-specific coordination |
| H1-specific modulation-spectrum shape (6-dim) | **CONDITIONAL, the strongest surviving F010-derived candidate** | Beats generic envelope 7.5×; survives the tracking-reliability alternative explanation; **not yet tested against the same common-envelope confound that sank HMC** (§15's top open item — this decision would need to become REJECT if that check also fails) |
| F008 + F010 combination | **CONDITIONAL, the most replicated positive finding overall** | Consistent positive `majority_vote` effect across 2–4 seeds and all 3 tested N levels (+0.0059/+0.0076/+0.0072); mechanism unexplained, and untested using H1-specific F010 in place of the diluted all-33 version |

No candidate reaches an unqualified KEEP. Every one still has at least one open, specifically-named
verification step (see Next Steps) — this reflects the evidence honestly rather than forcing a
premature resolution the brief explicitly warned against (§0: don't judge success by whether F1
went up; §19: don't quietly protect earlier conclusions).

## Next Steps

1. **Highest priority**: re-test H1-specific modulation against the same common-envelope-removal
   confound that overturned HMC (§7) — H1's absolute-amplitude modulation shape has not yet been
   checked against its envelope-relative counterpart, and it is now the leading candidate, so this
   is the single most important unresolved question from tonight's work.
2. Finish the pending N=5,600/seed 43 classifier-comparison point (`situation` and
   `majority_vote`) — was still running after several hours of background compute when this
   document was last updated; check `output/signal_v2/SP-V2-012/point_N5600_seed43_*.json` for
   completion before re-launching.
3. Test **F008 + H1-specific-modulation** (rather than F008 + all-33-F010) as a combination —
   untested, and plausibly stronger than either the H1-alone or the F008+F010(-diluted) result
   reported here, given H1 alone already beats all-33-F010.
4. Nonlinear/spline HMC analysis (brief §9) — lower priority now that §6/§7 found HMC itself
   substantially confounded; revisit only if HMC is reinvestigated later.
5. Bootstrap confidence intervals on the headline numbers (brief §17) — not attempted, time-limited.
6. Extend the duration-confound check already run on F010 (§12/daytime) to F008's
   `n_voiced_frames`-proxy `majority_vote` association flagged in §21 — still open from the daytime
   session, not reached overnight.
