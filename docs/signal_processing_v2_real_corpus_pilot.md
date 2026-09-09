# Signal Processing v2 — Real Corpus Pilot

Real-corpus evaluation of the five feature families that had passed synthetic validation
(SP-V2-006, 007, 008, 009, 010 — see `signal_processing_v2_log.md` for those results). Synthetic
validation shows a feature responds correctly to a *controlled, known* signal property; it says
nothing about whether that property varies meaningfully across *this corpus's* emotion labels,
survives a gender confound check, is already present in the existing 117-feature set, or helps a
classifier. This document answers those questions, family by family, with the same protocol
applied identically to all five — and ends each section with an explicit **KEEP / CONDITIONAL /
REJECT** decision, recorded even when the result is unfavorable.

## Dataset / Protocol

**Corpus facts re-verified before this pilot** (matches the audit exactly): 33,964 utterances have
complete baseline (117-feature) coverage; `situation` (scripted intent, 7 balanced-ish classes) and
`majority_vote` (5-rater intensity-weighted majority, ties excluded — 33,980 raw rows, 1,711
ambiguous/tied and excluded) are both available; `gender`/`age` are present in the raw metadata,
**no speaker ID** exists anywhere. Nothing here is described as "speaker-independent" — only
gender (and, not yet used, age/corpus-batch) confound control is possible.

**Pilot sample**: stratified, 200 files per `situation` class, seed 42 — **1,400 files**, all with
complete baseline-feature coverage (`output/signal_v2/real_corpus/real_corpus_features.csv`).
`majority_vote` labels are available for 1,307 of these 1,400 (93 ties excluded) but are **heavily
imbalanced** in this subsample despite `situation` being perfectly balanced: `sadness` 382 (29%),
`happiness` 200, `neutral` 189, `angry` 186, `fear` 125, `disgust` 122, `surprise` 103 (8%). This
imbalance — a real property of how raters' majority-vote labels distribute even under
situation-stratified sampling, not a sampling bug — means `majority_vote` classifier results below
carry substantially more per-fold variance than `situation` results (as few as ~20 test examples
for `surprise` per fold) and should be read with that caveat attached every time, not just once
here.

**Why 1,400, not the full 33,964**: SP-V2-006 needs real per-file pitch marking (cheap),
SP-V2-008 needs `pyworld` harvest+d4c (cheap), but SP-V2-009/010 reuse Phase B's pitch-synchronous
harmonic-trajectory extraction (NFFT=4096 per analysis frame, ~150–300 frames/utterance) — measured
throughput for all five families combined, extracted in one pass per file to share preprocessing,
was **1.27 files/sec** with 6-way process parallelism (`output/signal_v2/real_corpus/` extraction
log), i.e. ~25 minutes for 1,400 files and an estimated ~7.4 hours for the full corpus. A full-
corpus run is deferred until this pilot identifies which families are worth that cost.

**Baseline**: the existing `emotion_classifier.py` 117-feature set, loaded via
`load_combined_features()` — but for a *fair* ablation, restricted to the same 1,400 sampled
`wav_id`s used for the new features, not the full corpus. This means the baseline macro-F1 numbers
in this document (~0.24 on `situation`, ~0.29 on `majority_vote`) are **not comparable to the
README's full-corpus numbers** (0.324 / 0.349, trained on ~27k rows) — smaller training folds
(~1,120 rows) naturally score lower. Only the *baseline vs. baseline+family delta*, computed on
identical folds, is the meaningful quantity here.

**Classifier ablation protocol** (`signal_v2/evaluation/real_corpus_common.py`): `StratifiedKFold`
(5 splits, shuffled, `random_state=42`) — the exact same fold assignment used for baseline-alone
and baseline+family, every time, so the comparison is paired. `RandomForestClassifier(
n_estimators=200, class_weight="balanced", random_state=42)` — 200 rather than the base repo's 300
trees, for pilot-scale runtime, applied identically to every condition. No scaling/normalization
step exists in this pipeline for any new feature (raw per-file engineered values only), so there is
no cross-fold-leakage surface to audit here — confirmed directly rather than assumed only because
nothing needed fitting outside the fold in the first place.

**Confound classification** (Type A/B/C/D, `gender_confound_table()`): thresholds are explicit and
logged, not tuned to produce a nicer-looking table — `epsilon_sq >= 0.01` ("negligible" boundary,
same value `dissonance_tonality_stats.py` already uses) for a "meaningful" pooled effect,
`epsilon_sq >= 0.005` (half that, to allow for smaller within-gender N) for a "meaningful"
within-gender effect, and a male/female per-emotion mean-ranking Spearman correlation `>= 0.5` for
"the direction roughly agrees between genders." **Type A** (robust): pooled effect real, at least
one gender's within-group effect real, rankings agree. **Type B** (gender-dependent): a real
within-gender effect exists but rankings diverge — not a failure, a plausible gender-specific
expression pattern, recorded as such. **Type C** (confounded): pooled effect real but vanishes in
both genders separately — the pooled number was mostly gender artifact. **Type D**: no effect
anywhere. **Type E**: none of the above cleanly (ambiguous).

## Baseline

Already established (`emotion_classifier.py`, this session's audit): 117 features, Decision Tree +
Random Forest, `situation`/`majority_vote` evaluated in parallel, no speaker-independent split
possible. Full-corpus numbers (0.324/0.349 macro F1, RF) are the reference point for "does the
final pipeline improve" — not directly reproduced in this pilot's smaller-sample ablations, per the
caveat above.

## SP-V2-006 — Glide-aware jitter/shimmer

### Hypothesis
Conventional jitter/shimmer (this repo's own, and Praat's) partly measures intentional F0/amplitude
movement, not just true cycle-to-cycle perturbation (confirmed synthetically already). On real
speech, a glide-aware **residual** jitter/shimmer should depend less on F0 motion while retaining
whatever emotion signal jitter/shimmer carries.

### Method
6 features (`f006_conv_jitter/shimmer`, `f006_resid_jitter/shimmer`, `f006_f0_motion_energy`,
`f006_n_cycles`) extracted using the base repo's own real pitch-marking
(`voice_quality_features.mark_pitch_periods`, reused as-is — this isolates the *formula* difference
from any pitch-marking-quality difference) plus a residual-vs-conventional comparison on those same
marks. A **separate, targeted deep-dive** (`signal_v2/evaluation/f006_deep_dive.py` +
`f006_correlation_analysis.py`) additionally computes Praat (`parselmouth`) and the actual existing
`voice_quality_features.compute_voice_quality_features()` output (not a reimplementation) on the
identical 1,400 files, and tests the brief's core §11 question directly: does residual jitter
correlate less with F0 motion energy than legacy/Praat jitter does, while keeping the emotion
association?

*Bug found and fixed during extraction*: the first extraction run failed on all 1,400 files with
`zero-size array to reduction operation maximum` — real audio's pitch marks (unlike the always-
monotonic synthetic marks used in every prior validation) occasionally repeat or invert, producing
zero-length inter-mark segments that `np.ptp` cannot handle. Fixed by treating degenerate segments
as zero amplitude (`_segment_amplitudes` in `glide_aware_jitter.py`); re-verified the synthetic
SP-V2-006 benchmark still reproduces its exact previously-logged numbers after the fix (no
regression).

### Real-corpus statistics (Evaluation A)
All 6 features: 0% NaN, 0% constant, no quality flags.

### Emotion association (Evaluation B)
Weak across the board. `f006_n_cycles` is the strongest (ε²=0.025, "small", `sadness` lowest ↔
`surprise` highest — the same low/high-arousal axis the base repo finds almost everywhere).
`f006_resid_jitter`/`f006_conv_jitter` are both ε²≈0.0050 ("negligible", `angry` lowest ↔ `sadness`
highest) — **essentially identical** to each other. `f006_conv_shimmer`/`resid_shimmer`: ε²≈0.0046/
0.0023, negligible.

### Confound analysis (Evaluation C)
**All 6 features are Type B (gender-dependent)** — none pass the Type-A bar. Male/female rank
correlations are *negative* for every one of the 6 (−0.14 to −0.43): male and female emotion
rankings for jitter/shimmer don't just fail to agree, they trend in opposite directions — the same
qualitative pattern the base repo already found for `formant_roughness` (angry-highest for men,
happiness-highest for women), now recurring in an independently-built feature family.

### Redundancy (Evaluation D)
`f006_n_cycles` correlates at ρ=0.99999 with the existing `n_periods` — **not a new feature at
all**, the same count computed twice. `f006_conv_jitter`/`f006_resid_jitter` both correlate at
ρ≈0.83–0.84 with the existing `jitter_local` — high, expected (they're built on the same pitch
marks and a closely related formula), but not total-redundancy. `f006_f0_motion_energy` correlates
at ρ=0.894 with the existing `interval_abs_std` (melodic-interval-size feature) — also largely
redundant with information already in the pipeline.

### Classifier ablation (Evaluation E)
`situation`: baseline 0.2440 → combined 0.2451, Δ=+0.0011 (3/5 folds improved) — noise-level.
`majority_vote`: baseline 0.2943 → combined 0.2877, Δ=−0.0066 (3/5 folds improved) — noise-level,
negative.

### Deep-dive: does residual jitter actually reduce F0-motion dependence on real audio?
**No — this is the pilot's most important negative finding for this family, and it directly
contradicts what the synthetic F/G stress-test condition (SP-V2-006's own strongest positive
result) predicted.** Spearman correlation with `f0_motion_energy`: legacy (real
`voice_quality_features` output) r=0.258, this pilot's own conventional reimplementation r=0.222,
**residual r=0.231 — not meaningfully lower than conventional**, and both clearly higher than
Praat's r=0.056. Pooled emotion association for all four jitter variants (legacy/Praat/
conventional/residual) is statistically indistinguishable: ε²=0.0050–0.0065, all "negligible".
Partial association controlling for F0 motion + gender (residualized via OLS, since `statsmodels`
is not installed and this avoids adding it): ε²=0.0031–0.0048 for all four, still all negligible,
p-values 0.05–0.11 (not below 0.05 for conventional/residual once controlled).

**Interpretation**: the synthetic F/G condition (15 Hz, ±2-semitone modulation) showed a dramatic,
clean win for residual jitter because it used a deliberately extreme, fast pitch-accent stress
test. Real conversational speech in this corpus apparently does not move its F0 fast/far enough,
most of the time, for that specific failure mode to dominate — consistent with SP-V2-002's parallel
finding that its own tested glides (closer to typical prosody) were "too slow" to expose fixed-
window contamination either. **Praat's much lower F0-motion correlation (0.056 vs. everyone else's
0.22–0.26) is the more interesting real-corpus result**: Praat's periodic-cross-correlation pitch
marking is apparently far more robust to whatever is driving this correlation on real audio than
either this repo's own peak-tracking or the ground-truth-adjacent residual-formula fix — pointing
back at pitch-*marking* quality (as SP-V2-006's synthetic write-up already flagged as the base
repo's likely dominant real-world problem) rather than the jitter formula itself as the thing worth
fixing next, if this line is pursued further.

### Decision: **REJECT** (as currently implemented)
Weak-to-negligible standalone emotion signal, gender-divergent (not merely gender-modulated) for
every feature, near-total redundancy for 3 of 6 features, no classifier value on either label, and
— the specific methodological hypothesis this family was built to test — **no evidence that
residual jitter reduces F0-motion dependence on real speech**, unlike its clean synthetic result.
Not silently dropped: the Praat-vs-everyone-else pitch-marking-robustness gap is a genuine,
actionable finding worth carrying into any future revisit of this family, just not one this pilot's
own "residual formula" fix addresses.

## SP-V2-007 — CWT multi-scale prosody

### Hypothesis
Emotional prosody concentrates selectively at particular temporal scales (microprosody/syllable/
word/phrase/utterance), which a discrete note-level representation (kept, unchanged) cannot show.

### Method
31 features (5 bands × {energy, variance, entropy, peak_density, sign_change_rate} + energy_ratio
+ 1 dominant_scale/dominant_band pair) from `signal_v2/prosody/cwt_prosody.py`, computed on the same
pyin F0 track used by F006/F009/F010.

### Real-corpus statistics
0% NaN, 0% constant across all 31.

### Emotion association
Weak. Best: `f007_syllable_entropy` ε²=0.023 ("small"). The five `*_entropy` features (one per
band) are the only features reaching even "small" — `energy_ratio`/`peak_density`/
`sign_change_rate`/`dominant_scale_period_sec` are **uniformly negligible-to-zero** (most ε² at or
below 0, i.e. indistinguishable from noise) across every one of the 21 non-entropy features.

### Confound analysis
0/31 Type A (robust). 19 Type B (gender-dependent), 12 Type D (no effect anywhere — the majority of
the non-entropy features).

### Redundancy — the decisive finding for this family
**The only features with real (if small) emotion signal are almost exactly the features most
redundant with existing utterance-length proxies.** All 5 `*_entropy` features correlate at
ρ=0.86–0.96 with `n_frames`/`n_valid_formant_frames`/`n_notes`/`n_voiced_pitch_frames` — a
mechanical consequence of entropy estimation being sample-size-sensitive (a longer utterance feeds
more samples into the same band's entropy calculation, inflating the estimate independent of any
real prosodic content). Meanwhile the non-entropy features correlate more moderately (ρ=0.2–0.6)
with `notes_per_sec`, `total_pause_ratio`, `pause_dur_mean`, `pitch_std` — real but partial overlap
with existing rhythm/pause features — **and show no emotion signal at all**. This is exactly the
"CWT turns out to be almost entirely redundant with existing speaking-rate/duration features"
scenario the brief's §12 explicitly asked this pilot to check for, and it is what happened.

### Classifier ablation
`situation`: Δ=−0.0013 (2/5 folds improved). `majority_vote`: Δ=−0.0249, **0/5 folds improved** —
the only family in this pilot with a unanimous, consistent negative result across every fold, not
just a negative mean.

### Interpretation
This family's apparent "small effect size" signal is a duration/entropy-estimation artifact wearing
a prosody costume, not new emotional information — and the parts that aren't that artifact carry no
detectable emotion signal at all. The clean synthetic validation (SP-V2-007's own log entry: four
injected timescales correctly recovered) validated the *decomposition mechanism*, correctly, but
mechanism validity does not imply the *derived summary statistics* (especially entropy, computed
per-utterance without duration-normalization) are meaningful on real, variable-length utterances.

### Decision: **REJECT** (entropy sub-features specifically; energy_ratio/peak_density/
sign_change_rate are borderline REJECT — no signal, but also not clearly redundant, so a future
revisit isn't precluded)
The consistent (0/5 folds) `majority_vote` classifier harm is the strongest single piece of
evidence against keeping this family as currently summarized. If CWT prosody is revisited, the
entropy features would need duration-normalization (e.g. per-second rather than per-utterance) to
even be a fair test of whether they carry anything beyond utterance length.

## SP-V2-008 — Aperiodicity / MVF

### Hypothesis
Band-resolved periodicity (MVF, low/mid/high-band periodicity, aperiodicity spectral slope) should
distinguish breathiness/harshness in a way the existing single broadband `hnr_mean`/`std` cannot.

### Method
11 features from `signal_v2/voice_quality/aperiodicity.py` (`pyworld.harvest`+`d4c`, this phase's
own F0 re-estimate rather than the shared pyin track, since D4C needs WORLD's own F0).

### Real-corpus statistics
0% NaN, 0% constant across all 11.

### Emotion association
Small but real and broad-based: top feature `f008_mid_band_periodicity_std` ε²=0.033 ("small");
most of the 11 features cluster in the 0.008–0.033 range — **the most evenly-distributed,
consistently-nonzero effect-size profile of any family in this pilot** (no feature is exactly zero
or negative, unlike every other family here).

### Confound analysis — the best profile in this pilot
**7 of 11 features are Type A (robust)**: `aperiodicity_spectral_slope_mean`, both
`low_band_periodicity_{mean,std}`, `mid_band_periodicity_{mean,std}`, `mvf_std`,
`voiced_to_aperiodic_transition_rate`. Male/female rank correlations for these Type-A features are
notably high (0.54–0.89) — genuine cross-gender agreement, not a borderline pass. Only 4 features
(`high_band_periodicity_{mean,std}`, `mvf_mean`, `n_voiced_frames`) are Type B.

### Redundancy
**0/11 features exceed ρ=0.9 with any existing feature** — the cleanest redundancy profile of any
family. Correlations that do exist (ρ=0.39–0.72) are with spectral-shape features (MFCCs, ZCR,
spectral centroid/rolloff) and the wav2vec2 `arousal`/`dominance` VAD features — a sensible,
interpretable partial overlap (aperiodicity is a spectral-shape-adjacent quantity) rather than
outright duplication.

### Classifier ablation
`situation`: Δ=−0.0073 (1/5 folds improved). `majority_vote`: Δ=−0.0048 (2/5 folds improved). Both
negative despite the best standalone-effect/confound/redundancy profile in the pilot.

### Interpretation
A clean instance of the base repository's own repeated lesson (`arousal`/`dominance`/`valence`
rank #1–2 in the full classifier yet collapse through the Han & Cha ellipse method; `note_dur_mean`
has the largest standalone effect size in the whole project yet ranks 54th/117 in importance):
**standalone statistical quality and multivariate classifier contribution are different
questions**, and this family answers the first well and the second poorly, on this sample size.
Given 11 features add real capacity for a 200-tree forest to overfit on ~1,120 training rows per
fold, a negative classifier delta here is not strong evidence the family lacks value at full-corpus
scale — it's the single family in this pilot where that caveat most plausibly applies, precisely
because every earlier check (A/B/C/D) came back this favorably.

### Decision: **CONDITIONAL** — worth a full-corpus classifier check before a final call
This is the strongest candidate in this pilot on every criterion *except* the one this small-sample
classifier ablation measures directly, which is also the criterion most likely to be sample-size-
limited here. Recommend re-running Evaluation E alone (not the full extraction) at full-corpus
scale once a full-corpus F008 extraction exists.

## SP-V2-009 — Harmonic phase

### Hypothesis
Relative harmonic phase coherence (circular statistics) carries voice-quality information that
magnitude-based features (including SP-V2-008's aperiodicity) don't capture.

### Method
11 features from `signal_v2/harmonic/harmonic_phase.py`, computed on the same adaptive-window
harmonic trajectory (`c=3`, 5 harmonics) shared with F010.

### Real-corpus statistics
0% NaN, 0% constant across all 11.

### Emotion association
Small, concentrated in the lower harmonics: `f009_h3_phase_circular_variance`/
`h3_phase_resultant_length` ε²=0.014 (tied — they are the same quantity, `1-R` vs. `R`, and
necessarily produce identical Kruskal-Wallis statistics), `mean_phase_coherence` ε²=0.013,
`h4_*` ε²=0.013. `h2_*` and `h5_*` are weaker (0.005–0.007). `cross_harmonic_coherence_mean` and
`temporal_phase_instability` are both slightly *negative* (≈0, no signal). Direction is consistent
with the base repo's low/high-arousal axis: `sadness` lowest / `surprise` highest for circular
variance (i.e. surprise has *less* phase-locking — more phase dispersion — than sadness).

### Confound analysis
5 of 11 Type A (robust), 4 Type B, 2 Type D. A mixed but not unreasonable profile — roughly the
middle of this pilot's five families.

### Redundancy
Moderate, and mechanistically sensible rather than alarming: `h2_phase_circular_variance`/
`resultant_length`/`mean_phase_coherence` correlate at ρ=0.32–0.53 with the existing `jitter_local`
— phase-locking instability and cycle-to-cycle period jitter measure related physical phenomena
(both are disrupted by the same irregular glottal cycling), so partial overlap here is expected,
not a red flag on its own. No feature exceeds ρ=0.6.

### Classifier ablation
`situation`: Δ=+0.0020 (3/5 folds improved) — the **second-best** situation-label result in this
pilot, though still small in absolute terms. `majority_vote`: Δ=−0.0137 (1/5 folds improved) —
negative, consistent with every family's majority_vote result in this pilot (see the class-
imbalance caveat in Dataset/Protocol).

### Interpretation
The one family here with a positive, if modest, `situation` classifier delta *and* a coherent
mechanistic story (jitter-adjacent, not jitter-identical) *and* a reasonable confound profile. Not
a dramatic result on its own, but the least discouraging combination of the five families for
`situation`, and worth keeping alongside F008/F010 rather than in the same tier as F006/F007.

### Decision: **CONDITIONAL** — small positive `situation` signal, worth retaining pending the
adaptive-harmonic-model question (§ below) since phase estimation is the most precision-sensitive
of the harmonic-trajectory-based families (integer-bin phase reading, no parabolic refinement — a
documented limitation from the synthetic validation).

## SP-V2-010 — Harmonic modulation spectrum

### Hypothesis
The temporal modulation spectrum of each harmonic's amplitude trajectory carries emotion-relevant
information about *how fast* harmonic structure changes, not captured by any instantaneous
(magnitude/phase-at-a-point) feature.

### Method
33 features from `signal_v2/harmonic/harmonic_modulation.py`, same shared harmonic trajectory as
F009.

### Real-corpus statistics
0% NaN, 0% constant across all 33.

### Emotion association — the strongest in this pilot
`f010_inter_harmonic_modulation_coherence` ε²=0.050 (**"medium"** — the only medium-or-larger effect
size found in this entire pilot, across all 5 families, 92 features). `f010_h1_modulation_entropy`
ε²=0.041 (medium), `h2` ε²=0.040 (medium), `h3–h5_modulation_entropy` and
`modulation_low_rate_ratio_mean` all 0.020–0.033 (small). Direction: `sadness` lowest ↔ `surprise`
highest for `inter_harmonic_modulation_coherence` (surprise's harmonics modulate together *less*
coherently than sadness's) and the reverse for the entropy features (`sadness` lowest ↔ `surprise`
highest for entropy itself, meaning surprise's per-harmonic modulation spectrum is *more*
disordered/spread-out) — a coherent, physically-plausible joint story: surprise shows both less
cross-harmonic coordination and more spectral disorder in how each harmonic's amplitude moves,
consistent with a more agitated, less controlled phonation.

### Confound analysis
11 Type A, 18 Type B, 3 Type D, 1 Type E. `inter_harmonic_modulation_coherence` itself is Type A but
borderline: gender ε²=0.287 (a large gender effect co-exists with the emotion effect) and
male/female rank correlation exactly at the 0.5 threshold — a real effect, but flagged here as
**less unambiguously robust than the Type-A label alone suggests**; a stricter threshold would
likely reclassify it as Type B. The `h*_modulation_entropy` features are more clearly redundant
with sample-count proxies (see below) than genuinely gender-robust either way.

### Redundancy
The two strongest features tell different stories. `inter_harmonic_modulation_coherence` correlates
at ρ=−0.59 with `pitch_mean` (and moderately with several MFCCs) — related to, but not identical
with, existing pitch/timbre features; a real, distinct signal. **`h1`/`h2_modulation_entropy`
correlate at ρ=0.69–0.70 with `n_periods`/`n_frames`/`n_valid_formant_frames`** — the same
sample-size-sensitivity artifact identified in SP-V2-007's entropy features, recurring here because
both use the same entropy-of-a-variable-length-trajectory computation pattern. This means roughly
half of this family's medium-effect-size features are at least partly a duration artifact, not
purely a modulation-rate discovery.

### Classifier ablation
`situation`: Δ=**+0.0136** (3/5 folds improved) — the **best classifier result of any family in
this pilot**, on either label. `majority_vote`: Δ=−0.0204 (2/5 folds improved) — negative, again
consistent with this pilot's general `majority_vote` pattern.

### Interpretation
The strongest family in this pilot by two independent measures (standalone effect size and
classifier contribution) that don't reduce to the same underlying artifact — but with a real,
disclosed asterisk: part of its apparent strength (the `*_modulation_entropy` features specifically)
inherits the same duration-confound weakness that sank SP-V2-007 outright. The
non-entropy-dependent `inter_harmonic_modulation_coherence` result stands on its own (moderate,
not extreme, redundancy; real if borderline-robust confound profile) and is the more defensible
single feature to carry forward. Per §15 of the brief's own harmonic-tracker-instability check: the
underlying harmonic trajectory extraction (Phase B's simpler peak-picking, not a proper adaptive
harmonic model) is shared with F009, and this family's real-corpus success — not failure — argues
*against* an urgent need to replace it with SP-V2-011 for this specific family; instability was not
the limiting factor here (§ Implications for SP-V2-011 below).

### Decision: **CONDITIONAL** — `inter_harmonic_modulation_coherence` and the low-rate-energy
features are the strongest, least-artifact-driven candidates in this pilot on every individual
criterion (Evaluations A–E in isolation); drop or duration-normalize the `*_modulation_entropy`
sub-features before any further use, per the same fix SP-V2-007 would need. **Revised down from an
initial KEEP** after the Integrated Ablation (below) showed this family has the single worst
marginal contribution of all five once combined with the others on `majority_vote` (−0.0299) — its
individual strength does not yet mean it survives combination at this sample size, and that
question is unresolved, not answered, by this pilot.

## Integrated Ablation

**This section changes the overall conclusion, and is the single most important result in this
document — read it before the per-family KEEP/CONDITIONAL verdicts above, not after.**

Per the brief's §16 instruction, stepwise (cumulative) addition is reported but **not** treated as
the main comparison, because order changes the answer: for `situation`, the cumulative order
F006→F007→F008→F009→F010 produces combined macro F1 of 0.2451 → 0.2547 → 0.2317 → 0.2480 → 0.2411
— rising, then dropping sharply when F008 is added, then partially recovering, then dropping again.
A different addition order would trace a different, equally uninterpretable path. The **leave-one-
out** comparison (all 5 families' 92 features combined, then each family removed in turn) is the
real test:

| | `situation` combined F1 | `situation` marginal contribution | `majority_vote` combined F1 | `majority_vote` marginal contribution |
|---|---|---|---|---|
| Baseline alone | 0.2440 | — | 0.2943 | — |
| **All 5 families (92 feats)** | **0.2411 (Δ=−0.0029, 3/5 folds)** | — | **0.2585 (Δ=−0.0359, 0/5 folds)** | — |
| All − F006 | 0.2457 | −0.0046 | 0.2690 | −0.0106 |
| All − F007 | 0.2568 | −0.0156 | 0.2744 | −0.0159 |
| All − F008 | 0.2588 | **−0.0177** | 0.2790 | −0.0205 |
| All − F009 | 0.2553 | −0.0142 | 0.2859 | −0.0274 |
| All − F010 | 0.2480 | −0.0069 | 0.2883 | **−0.0299** |

Two things stand out. First, **the full 92-feature combination underperforms the 117-feature
baseline on both labels** — modestly for `situation` (−0.0029) but substantially and *unanimously*
for `majority_vote` (−0.0359, 0 of 5 folds improved — every single fold got worse). Second, and
more informative: **every family's marginal contribution to the combined set is negative on both
labels**, including F010, whose *individual* ablation (Evaluation E, its own section above) was
this pilot's best result on either label (+0.0136 situation). In combination, F010 is merely "the
family that hurts least" (situation) or "the family that hurts most" (majority_vote, −0.0299,
the single worst marginal number in this table) — a reversal that only shows up once all families
compete for the same ~1,120-training-row-per-fold budget alongside 92 new dimensions.

**Interpretation**: this is close to a textbook overfitting signature, not a clean "these features
don't help" result — but it is reported as what actually happened, not reframed into the more
convenient story. 92 new features against ~1,120 training rows per fold is a rich feature-to-sample
ratio for a Random Forest to search, and `majority_vote`'s additional class imbalance (§ Dataset/
Protocol) makes it the more exposed of the two labels, exactly matching the unanimous 0/5-fold
result there. **F008 is the clearest case that this reading, not a "the feature is bad" reading, is
correct**: it has this pilot's best standalone effect-size profile, best gender-confound profile
(7/11 Type A), and best redundancy profile (0/11 features over ρ=0.9 with anything existing) — yet
the single worst `situation` marginal contribution (−0.0177) when combined with the other four
families. A feature family that is this clean on every criterion *except* small-sample multivariate
performance is a much stronger candidate for "needs more training data to show its value" than for
"provides no value."

This does **not** overturn the individual per-family REJECT calls above (F006, F007 still show
weak-to-artifact-driven standalone signal *before* any combination effect even applies) — but it
does mean the CONDITIONAL calls for F008/F009/F010 should be read as genuinely open, not as
soft-KEEPs. The recommended full-corpus classifier re-check (Next Steps, below) is now the decisive
missing evidence for those three, more clearly than the per-family sections alone suggested.

## Candidate Features (scoring summary)

| Family | Physical validity (synthetic) | Emotion effect (real corpus) | Confound robustness | Novel info (low redundancy) | Δ macro F1 (situation) | Decision |
|---|---|---|---|---|---|---|
| F006 glide-aware jitter | Validated (synthetic F/G) | Negligible (ε²≈0.005) | Poor — all 6 Type B, gender-divergent | Poor — 3/6 near-total redundancy | +0.0011 (noise) | **REJECT** |
| F007 CWT prosody | Validated (4/4 scales recovered) | Small, artifact-driven (entropy only) | Poor — 0/31 Type A | Poor — signal-bearing features are the redundant ones | −0.0013 / **−0.0249 (0/5 folds)** | **REJECT** |
| F008 aperiodicity/MVF | Validated (after noise-generator bug fix) | Small, broad-based, evenly distributed | **Best — 7/11 Type A** | **Best — 0/11 >ρ0.9** | −0.0073 | **CONDITIONAL** (re-test E at scale) |
| F009 harmonic phase | Validated (clean vs. noisy separation) | Small, coherent low-harmonic pattern | Mixed — 5/11 Type A | Good — moderate, mechanistically sensible overlap only | **+0.0020** | **CONDITIONAL** |
| F010 harmonic modulation | Validated (after flatness-artifact bug fix) | **Best — one medium effect (ε²=0.050)** | Mixed — borderline Type A on the top feature | Mixed — top feature clean, entropy sub-features artifact-prone | **Best individually — +0.0136**, but **worst marginal contribution in combination (−0.0299, majority_vote)** | **CONDITIONAL** (revise from initial KEEP after integrated ablation — see below) |

**Individual-family scores above are necessary but not sufficient — see Integrated Ablation.** All
five families' marginal contributions turn negative once combined (92 features against ~1,120
training rows/fold), most severely for F010 despite its best individual result. This more plausibly
reflects a small-sample overfitting effect than a genuine lack of value, given F008's simultaneous
best-in-pilot standalone/confound/redundancy profile and worst `situation` marginal contribution —
but that is a hypothesis this pilot's sample size cannot itself resolve, not a conclusion.

## Failed / Rejected Features

Recorded here, not deleted from the codebase or hidden from this log, per the brief's explicit
instruction not to hide unfavorable results:

- **F006, all 6 features**: negligible standalone emotion signal on real speech; the specific
  methodological improvement this family was built to demonstrate (residual jitter reduces F0-
  motion dependence) **did not reproduce on real audio** despite a clean synthetic win — the
  synthetic stress condition (15 Hz/±2-semitone modulation) does not represent this corpus's typical
  prosodic movement. `f006_n_cycles` in particular is a near-exact duplicate of the existing
  `n_periods`.
- **F007, entropy sub-features (5 of 31)**: the only features with any real effect size are almost
  perfectly explained by utterance duration/frame count (ρ=0.86–0.96 with existing frame-count
  features) — an entropy-estimation sample-size artifact, not new prosodic information.
- **F007, non-entropy sub-features (26 of 31)**: no detectable emotion signal at all (most ε² at or
  below zero), and the family as a whole is the only one with a unanimous (0/5 folds) negative
  classifier result on `majority_vote`.
- **F010, `*_modulation_entropy` sub-features (5 of 33)**: carry the same duration-artifact
  weakness as F007's entropy features, despite otherwise being part of this pilot's
  strongest-performing family.

## Implications for SP-V2-011

The brief's own decision criteria (§22–23) turn on whether harmonic-trajectory *instability* (from
Phase B's simple peak-picking extraction, shared by F009 and F010) is the bottleneck limiting those
two families. **This pilot's evidence points away from an urgent need for SP-V2-011**: F010 — the
family most dependent on that shared trajectory extraction — produced this pilot's single best
standalone effect size and single best classifier result, and F009 produced a small but real
positive classifier delta. Neither family's real-corpus weaknesses trace to tracker noise as far as
this pilot can tell; F010's weakness is a duration-confound in its entropy sub-features (a feature-
*design* issue, unrelated to tracker precision) and F009's is simply a small effect size, not
evidence of instability. The features that failed clearly in this pilot (F006, F007) don't use the
shared harmonic-trajectory extraction at all, so a better tracker would not rescue them regardless.

**Recommendation: defer SP-V2-011.** Per the brief's own §23 framing, adaptive harmonic modeling
should be built when it demonstrably fixes a real bottleneck, not added to increase sophistication
on principle — and this pilot did not find that bottleneck. The Integrated Ablation's negative
marginal contributions (including F010's) are much better explained by small-sample overfitting
(92 new features against ~1,120 training rows/fold, worse for the more imbalanced `majority_vote`)
than by harmonic-tracker imprecision — a more precise tracker would not change the training-data-
to-feature-count ratio that this pilot's evidence points to as the more likely limiting factor. If
F008's full-corpus classifier re-check or a future duration-normalized re-test of F007/F010's
entropy features changes this picture, this recommendation should be revisited, not treated as
final.

## Next Steps

1. **Highest priority, motivated directly by the Integrated Ablation**: re-run Evaluation E
   (classifier ablation only — extraction already exists for this pilot's sample; a full-corpus
   extraction would additionally be needed to test at full scale) with a **larger sample** than
   this pilot's 1,400 files, to test whether the negative combined-family result is genuinely a
   small-sample/overfitting artifact (the more likely explanation given F008's profile) or holds up
   with more training data. This is now the single decisive open question in this whole pilot —
   more important than any individual family's per-criterion scores above.
2. Re-run F008's Evaluation E specifically (its own CONDITIONAL item, independent of the combined-
   family question) once a full-corpus F008 feature set exists.
3. Duration-normalize F007's and F010's entropy-based sub-features (e.g. per-second rather than
   per-utterance) and re-test whether any real signal survives once the duration confound is
   removed — currently unknown, not assumed to fail.
4. Gender/F0-register-stratified re-analysis remains open for SP-V2-004 (source-normalized
   roughness, from the prior pilot) — unrelated to this document's five families but still an open
   item from the broader plan.
5. Do not proceed with SP-V2-011 (adaptive harmonic model) unless one of the above changes the
   picture — see Implications above.
