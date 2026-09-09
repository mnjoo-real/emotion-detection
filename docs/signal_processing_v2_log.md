# Signal Processing v2 — Research Log

Companion to [`signal_processing_v2_audit.md`](signal_processing_v2_audit.md) (what exists) and
[`signal_processing_v2_plan.md`](signal_processing_v2_plan.md) (what's planned). This log records
each experiment's Hypothesis / Method / Result / Interpretation as it's completed — including
unfavorable results, per the project's own stated principle (carried over from the base
repository's README) that a rejected experiment is reported exactly as prominently as a
successful one, not quietly dropped.

All code lives under `signal_v2/`; all outputs (CSVs, plots, run_info.json with git commit +
command) live under `output/signal_v2/`.

## SP-V2-001 — F0 estimator benchmark

**Hypothesis**: the existing pipeline's sole F0 estimator (`librosa.pyin`, adopted once in
`harmony_features.py` and never validated) may have exploitable weaknesses — octave errors, poor
glide tracking, noise sensitivity — that a benchmark against WORLD Harvest, REAPER, and CREPE
would reveal, since none of these have ever been compared against each other or against known
ground truth in this project.

**Method**: `signal_v2/f0/synthetic_signals.py` generates 13 synthetic conditions with exact
known F0(t) — stationary sinusoid, harmonic stack (stationary and at two registers, 75/300 Hz),
linear glide, exponential glide, vibrato, abrupt pitch jump, additive white noise at 20/10/0 dB
SNR, a breathy/noisy harmonic mix, and a voiced-unvoiced-voiced transition — all sharing a common
16 kHz sample rate and a common 10 ms analysis hop so every estimator is compared on equal
footing (the base repo instead uses three different rates for different feature families; see
audit §2). `signal_v2/f0/estimators.py` wraps `librosa.pyin` (the existing baseline),
`pyworld.harvest`, `pyreaper.reaper`, and `torchcrepe.predict` (`model="tiny"`, CPU) behind one
interface. `signal_v2/evaluation/synthetic_metrics.py` computes RMSE-in-cents (on frames where
both ground truth and estimate agree on voicing), voiced-frame error rate, octave-error rate
(cents error near ±1200 with a ±150-cent tolerance), and trajectory smoothness (RMS of the
estimated contour's second difference, in cents — lower means less frame-to-frame jitter in the
*estimate itself*). Full results: `output/signal_v2/f0_benchmark/metrics.csv` and per-condition
overlay plots in `output/signal_v2/f0_benchmark/plots/`.

**Result** (means across all 13 conditions; `pyworld`/`pyreaper`/`torchcrepe` newly installed for
this benchmark — none were in the venv before):

| estimator | mean RMSE (cents) | mean voiced-frame error | mean octave-error rate | mean runtime (s/utterance) |
|---|---|---|---|---|
| `pyin` (existing baseline) | 6.5 | 0.031 | 0.000 | 0.78 |
| `world_harvest` | 20.5 | 0.077 | 0.001 | 0.27 |
| `reaper` | 9.6 | 0.085 | 0.000 | 0.03 |
| `crepe_tiny` | 10.8 | 0.006 | 0.000 | 0.35 |

*(Updated after a bug fix described below — the `breathy_noisy_harmonic` condition's noise
generator was corrected; the change to these means is small and does not affect any conclusion
in this section.)*

Per-condition detail worth calling out individually:

- **`pyin` wins outright on stationary/quasi-stationary harmonic signals** (RMSE 0.8–2.3 cents on
  `harmonic_stack`, `additive_noise_snr20/10db`, `breathy_noisy_harmonic`) — better than every
  other estimator on exactly the kind of signal most of a voiced phoneme actually looks like.
- **`world_harvest` tracks smooth glides/vibrato best** (RMSE 6–8 cents vs. pyin's 15–21 cents on
  `linear_glide`/`exponential_glide`/`vibrato`, and the lowest trajectory-smoothness numbers of
  any estimator there) — consistent with WORLD's design intent (continuous F0 tracking) — **but
  fails badly on the one genuinely discontinuous case**: `abrupt_pitch_jump` gives it 126.8 cents
  RMSE and the only octave-error hit in the whole benchmark, because Harvest's smoothing
  assumption actively fights a real discontinuity.
- **`world_harvest` cannot track a pure sinusoid at all** (`stationary_sinusoid`: 99% voiced-frame
  error) — it, like `reaper` (100% voiced-frame error, RMSE undefined), is built around detecting
  a glottal-pulse-train harmonic structure, not a single-partial tone. This is not a bug to fix;
  it's a reminder that "pure tone" is not a realistic proxy for voiced speech for these two
  algorithms specifically, and the benchmark should be read with that caveat for that one
  condition only (`pyin` and `crepe_tiny`, both spectral/perceptual rather than pulse-train-based,
  track it fine).
- **`crepe_tiny` has the best voicing decision by far** (0.006 mean voiced-frame error, the only
  estimator that stays composed at 0 dB SNR and through the voiced/unvoiced transition condition)
  at the cost of somewhat higher RMSE than pyin on clean signal and the second-slowest runtime.
- **`reaper` is the fastest by an order of magnitude** (0.03 s vs. 0.28–0.81 s for the others) and
  has competitive RMSE on most conditions, but its trajectory-smoothness numbers are consistently
  the worst among the estimators that don't outright fail (e.g. 148.9 cents on the pitch-jump
  condition, vs. 65.7 for Harvest and 1.5–8.3 for pyin/CREPE) — its frame-to-frame estimates are
  more jagged even when the mean error is acceptable.

**Bug found and fixed during SP-V2-008, retroactively affecting this section**:
`breathy_noisy_harmonic`'s noise generator originally used a leaky integrator (`b=0.02`) intended
to turn white noise into aspiration-like broadband noise — its actual −3dB cutoff, computed
after the fact, is only **≈51 Hz** (`f_c ≈ -sr·ln(1-b)/(2π)`), meaning that "breathy" condition's
added noise was almost entirely below 51 Hz — a slow near-DC drift, not audible broadband
aspiration noise at all, discovered when SP-V2-008's aperiodicity features came back completely
flat across `noise_mix ∈ {0.2, 0.4, 0.6}` while the pre-existing `hnr_mean` feature correctly
tracked the change. Fixed to a first-difference (simple +6 dB/oct high-pass) noise shaping, and
this benchmark re-run afterward — the table above and the `pyin`/`crepe_tiny` bullet's cited RMSE
already reflect the corrected condition (2.3→3.1 cents for pyin, a small increase consistent with
genuinely more perturbative noise now being present). Notably, `world_harvest`'s RMSE on this
condition **improved** with the fix (25.1→8.5 cents) — the old near-DC drift apparently disrupted
Harvest's estimator more than real broadband noise does, an interesting but secondary observation
not central to this phase's conclusions. No other condition in this benchmark used the buggy
function, so nothing else here needed re-running.

**Interpretation**: `pyin` was never validated before this benchmark, and the result is that it's
a genuinely reasonable choice for this dataset's dominant case (short utterances, most frames
either clearly stationary-voiced or clearly unvoiced) — but it is measurably worse than WORLD
Harvest specifically on smooth F0 movement (glides, vibrato), which is exactly the kind of
continuous prosodic dynamics the broader Harmonic-Prosodic v2 program (Phase D onward) wants to
model well. No single estimator dominates every condition, so **the recommendation is not to
replace `pyin`** in the existing pipeline (which the user's brief explicitly says not to do
"억지로" without cause) but to make `world_harvest` available as an alternate/ensemble F0 source
specifically for continuous-prosody-focused v2 modules (Phase D/J), while keeping `pyin` for
anything that leans on the existing pipeline's voicing decisions. `reaper`'s speed makes it a
candidate for large-scale rerun/ablation work where wall-clock matters more than the last few
cents of accuracy. `crepe_tiny` earns a place as the voicing-decision reference given how much
better it handles the noisy/transition conditions — worth revisiting with the full (non-`tiny`)
CREPE model if voicing-decision quality becomes a bottleneck later. **This benchmark should be
re-run** once Phase C (source-filter separation) exists, since IAIF's own pre-processing may
change which estimator is preferable on the corpus's real, non-synthetic recordings.

## SP-V2-002 — Pitch-synchronous window ablation

**Hypothesis**: the existing pipeline's fixed analysis windows (25 ms LPC / ~256 ms pyin / 128 ms
FFT-peak, none scaled to speaker F0 — audit §4) should produce less consistent per-harmonic
frequency/amplitude estimates than a pitch-synchronous window `L(t) = c·T0(t)` (`c ∈ {2,3,4}`),
because the same fixed window covers a very different number of pitch periods for a low-F0 male
voice vs. a high-F0 female voice.

**Method**: `signal_v2/pitch_sync/adaptive_window.py` extracts, at a 10 ms analysis hop, the
frequency and amplitude of harmonics 1–5 using either a fixed 40 ms Hamming window (representative
of the base repo's fixed-window style) or an adaptive window clamped to [15 ms, 100 ms]. Harmonic
peaks are located by a parabolic (quadratic) interpolation around the FFT bin nearest each
expected `k·F0`, and amplitude is normalized by the window's coherent gain (`sum(window)/2`) so
that windows of different lengths are comparable on an absolute amplitude scale, not just a
relative one. Ground-truth ﻿F0(t) is used directly (not an estimated contour) so this ablation
isolates the window-choice effect from Phase A's F0-estimation-error question. Test signals reuse
`synthetic_signals.py`'s harmonic-stack generators, whose constant per-harmonic amplitude rolloff
gives an exact, time-invariant ground truth for the H_k/H1 ratio even while F0 glides — so any
measured *variation* in the estimated ratio over time is pure measurement instability, not a real
signal. Conditions: a stationary stack, linear/exponential glides, vibrato, and — targeting the
user's specific register concern — the same stationary stack repeated at a low-F0 (75 Hz, ~3
cycles inside a fixed 40 ms window) and high-F0 (300 Hz, ~12 cycles) register. Full results:
`output/signal_v2/pitch_sync/metrics.csv` and per-condition H3/H1 trajectory overlay plots in
`output/signal_v2/pitch_sync/plots/`.

**Result** (means across harmonics 2–5, across all 6 conditions):

| window | mean harmonic-frequency RMSE (cents) | mean H_k/H1 ratio std (dB) |
|---|---|---|
| fixed 40 ms | **1.25** | 0.034 |
| adaptive c=2 (≈2 cycles) | 6.40 | 0.040 |
| adaptive c=3 (≈3 cycles) | 2.69 | 0.016 |
| adaptive c=4 (≈4 cycles) | 1.76 | **0.009** |

The register-targeted conditions are the clearest single result: at **75 Hz** (fixed window ≈3
cycles), fixed-window frequency RMSE is 3.20 cents — matched exactly by adaptive c=3 (3.20 cents,
because at this F0 a 3-cycle adaptive window and the fixed 40 ms window are nearly the same
length) and beaten only by adaptive c=4 (1.83 cents); adaptive c=2 is far worse (15.93 cents, the
single worst number in the whole ablation — a ~27 ms window at this F0 is too short for a clean
spectral estimate). At **300 Hz** (fixed window ≈12 cycles), every method is essentially
equivalent and near-perfect (0.36–0.74 cents) — there's no headroom left to improve on at that
register with either approach.

**Interpretation — this did not confirm the hypothesis as stated, and that's reported
directly rather than reframed after the fact.** In every noise-free synthetic condition tested,
the existing fixed 40 ms window matched or beat every pitch-synchronous variant except adaptive
c=4, which only reached rough parity (never a clear win). The clearest failure mode is the
opposite of what the hypothesis anticipated: a *short* adaptive window (c=2, ~2 cycles) hurts
frequency-estimation accuracy far more than a merely-suboptimal fixed window ever does, because
fewer cycles means a wider main lobe and materially worse parabolic-interpolation accuracy — a
basic time/frequency-resolution trade-off that a naive "make the window match the pitch period"
argument glosses over. The register-targeted test *does* show the predicted mechanism exists (at
75 Hz, a 3-4-cycle adaptive window is competitive with or better than the fixed window; at 300 Hz
there's no difference because the fixed window already captures enough cycles) — but the
practical benefit only shows up at exactly the low end of the speaker F0 range this corpus covers
(male voices near 100–130 Hz per the base repo's README), and only for `c=4`, not the smaller
multiples the literature more commonly recommends for genuinely nonstationary analysis. Two
follow-ups this suggests, not yet run: (1) repeat this ablation on a *fast*, large-excursion glide
(a full-octave sweep inside 100–150 ms, closer to a real emotional pitch accent) where a 40 ms
window would span enough F0 change to plausibly blur the harmonic peaks — the glides tested here
(100 Hz+ over a full second) may simply be too slow relative to 40 ms for the fixed window's
theoretical weakness to bite; (2) re-run this same ablation with `world_harvest`-estimated F0
(SP-V2-001's stronger glide tracker) instead of ground truth, since real deployment can't assume
known F0. **Given this result, Phase B's adaptive window is not adopted as a pipeline-wide
replacement** for the existing fixed-window harmonic features; it is kept as an available,
validated-on-synthetic-data alternative (`signal_v2/pitch_sync/adaptive_window.py`, `mode=
"adaptive"`) that Phase C/D can pull in specifically where per-cycle (pitch-synchronous) analysis
is independently motivated — e.g. glottal inverse filtering, which classically expects
pitch-synchronous frames for a different reason (isolating the closed phase) than harmonic
frequency-estimation accuracy alone.

## SP-V2-003 — Glottal inverse filtering (IAIF) validation

**Hypothesis**: every existing "harmonic" feature in the base repo (`formant_roughness`,
`melodic_dissonance`, `partial_roughness`) operates on the observed waveform, which conflates
glottal excitation and vocal-tract resonance (audit §13, weakness #1). Implementing Iterative
Adaptive Inverse Filtering (Alku, 1992) should let us separate the two, recovering a plausible
glottal source signal and a vocal-tract filter whose formants match a known ground truth on
synthetic material.

**Method**: `signal_v2/source_filter/iaif.py` implements the standard 4-step, 2-pass IAIF
procedure exactly as commonly described in the literature (a leaky integration to cancel lip
radiation, a coarse low-order LPC pass to grossly estimate the glottal tilt, a first vocal-tract
LPC pass and inverse filter to get a first glottal-flow-derivative estimate, then a second
refinement pass of both) — reusing the base repo's own `harmony_features.levinson_durbin` rather
than reimplementing LPC. This is explicitly **not** claimed as novel — it's an implementation of
published prior art; the potential contribution is what Phase F (SP-V2-004) builds on top of it.
`signal_v2/source_filter/synthetic_validation.py` builds a fully-known source-filter test signal:
a Rosenberg (1971) glottal-flow pulse train at a known F0, passed through a cascade of three
known 2-pole formant resonators, then a leaky-differentiator radiation filter — and checks (a)
correlation between the true and IAIF-recovered glottal flow (best-lag aligned, to allow for the
filter chain's group delay) and (b) how closely the vocal-tract LPC's root-derived formants match
the resonators' true frequencies. `signal_v2/source_filter/real_audio_demo.py` additionally
produces, per the brief's minimum requirement, a 3-panel visualization (raw waveform / estimated
glottal-flow derivative / estimated vocal-tract spectral envelope over the highest-energy frame)
for one real utterance per emotion — qualitative only, since real audio has no ground truth.
Full results: `output/signal_v2/source_filter/synthetic_validation_metrics.json` and plots in
`output/signal_v2/source_filter/plots/`.

**Result**: on the default synthetic case (F0=120 Hz, formants at 700/1220/2600 Hz), glottal-flow
correlation at best-lag alignment was **0.69**, with formant recovery excellent for F2/F3 (15–19
Hz error) but off by 111 Hz for F1. A follow-up robustness sweep across F0 and vowel (formant
set) combinations gave a much less uniform picture, reported here in full rather than cherry-
picked:

| condition | glottal flow correlation | F1 error (Hz) | F2 error (Hz) | F3 error (Hz) |
|---|---|---|---|---|
| F0=100 Hz, formants 700/1220/2600 | 0.76 | 39 | 11 | 23 |
| F0=120 Hz, formants 500/1800/2500 | **0.84** | 16 | 2 | 8 |
| F0=150 Hz, formants 700/1220/2600 | 0.76 | **510 (failed)** | 1339 (failed) | 1260 (failed) |
| F0=220 Hz, formants 700/1220/2600 | **0.54** | **432 (failed)** | 10 | 19 |

**Interpretation**: two distinct, honestly-reported limitations, neither hidden:

1. **Glottal-flow correlation degrades as F0 rises relative to F1** (0.84 at 120 Hz down to 0.54
   at 220 Hz on the same formant set) — consistent with the well-known theoretical limitation of
   LPC-based glottal inverse filtering in general: fewer harmonics fall below F1 as F0 increases,
   which makes the source/filter decomposition fundamentally more ambiguous, not a bug specific
   to this implementation. **This directly means IAIF-derived features in this v2 pipeline should
   be expected to work less reliably for higher-F0 (typically female) voices** — worth carrying
   forward as an explicit caveat into Phase F/E rather than assuming uniform quality across the
   corpus's gender-imbalanced F0 distribution (audit §4 already flagged this same register
   asymmetry for a different reason).
2. **Formant recovery failed outright in two of four conditions** (150 Hz and 220 Hz cases), and
   in both failures it was specifically **F1 that was misidentified**, with the error cascading
   into what should have been F2/F3 slots. Root cause: `lpc_to_formants` (adapted from the base
   repo's own `harmony_features.extract_formants_from_frame` root-picking convention) selects
   roots by ascending frequency with no cross-frame continuity or plausibility tracking — exactly
   the limitation the audit already flagged for the *existing* formant pipeline (audit §9, "no
   formant-tracking continuity... can misassign F1↔F2 if a formant briefly disappears from the
   LPC root set"). This is not a new problem introduced by IAIF; it's the same known weakness
   surfacing again on a second, independent piece of code that reuses the same root-picking
   pattern — a signal that a proper formant-continuity tracker (dynamic-programming path search
   across frames, or a bandwidth/prominence-based confidence filter) is worth building once, as a
   shared utility, rather than accepting ad hoc root-picking everywhere it recurs.

The real-audio qualitative check (7 files, one per emotion) produced plausible-looking
vocal-tract envelopes with 2–4 broad resonant bumps in speech-formant-typical frequency ranges
when the highest-energy frame was used (an early version that picked the geometric-center frame
instead sometimes landed in a low-energy inter-syllable dip and produced a less convincing
envelope — fixed before finalizing, and worth remembering as a frame-selection detail for any
future single-frame IAIF diagnostic). No quantitative claim is made from the real-audio check
since there is no ground truth to compare against — it confirms the pipeline runs and produces
non-degenerate output on real speech, nothing stronger.

**Given these results, IAIF is retained for Phase F (source-normalized roughness) with the above
two caveats carried forward explicitly** rather than silently assumed away: (a) expect weaker
separation quality at higher F0/female register, and (b) do not trust the per-formant frequency
identity from this simple root-picking scheme for anything that needs formant *labels*— Phase F
only needs harmonic amplitude/frequency values themselves (not formant identity), so this second
caveat does not block it, but would block any future phase that wants to reason about specific
formants (e.g. "does F1 bandwidth change with emotion") without first building the continuity
tracker noted above.

## SP-V2-004 — Raw vs. source-normalized roughness (pilot)

**Hypothesis**: since every existing "harmonic" feature conflates glottal source and
vocal-tract filter (audit §13; SP-V2-003's own motivation), computing the same Sethares
roughness model on the IAIF-estimated glottal source — instead of the raw observed spectrum —
should give an emotion-related roughness feature that is less confounded by vocal-tract/speaker
anatomy (and, per the base repository's sarcasm-project finding the user described, possibly more
consistent in effect direction across corpora, though that cross-corpus test is out of scope for
this pilot).

**Method**: `signal_v2/harmonic/source_roughness.py` computes three variants of the same Sethares
dissonance model (`harmony_features.sethares_dissonance`, unchanged) over harmonics extracted by
the *same* pitch-synchronous procedure (`signal_v2/pitch_sync/adaptive_window.py`, `c=3`) so that
only the **input signal** differs between arms, isolating the source-filter-separation variable
from any difference in extraction algorithm: (1) `raw` — harmonics located in the raw waveform,
absolute-amplitude weighted; (2) `source` — harmonics located in the IAIF-estimated glottal-flow-
derivative signal, absolute-amplitude weighted; (3) `source_norm` — same as `source` but using
each frame's H_k/H1 amplitude ratio (removing overall loudness/frame-energy scale). Since running
IAIF (iterative per-frame LPC) across the full 34k-file corpus was estimated, from this pilot's
own measured throughput (~0.9 files/sec with `MAX_WORKERS` processes), at roughly **10+ hours** —
squarely a background/overnight job, not something to run inside this session and call a pilot —
this experiment instead draws a **stratified sample of 280 files (40 per emotion)** and reports it
explicitly as a pilot, not a corpus-wide result. For each sampled file: F0 via `pyin` (matching
SP-V2-001's baseline), IAIF via `signal_v2/source_filter/iaif.py`, then the three roughness
variants. Effect size (Kruskal-Wallis + rank epsilon-squared) reuses
`dissonance_tonality_stats.py`'s exact functions rather than reimplementing them. Gender
sensitivity and redundancy (Pearson/Spearman correlation) against the two relevant pre-existing,
full-corpus features (`partial_roughness_mean`, `formant_roughness_mean`, looked up per sampled
file from the already-complete `output/partial_roughness/` and `output/harmony/` CSVs) round out
the pilot's checklist per the brief's Phase F requirements. One bug worth recording because it's
a repeat of a known pitfall: the first full run of this pilot failed on **all 280 files** with
`get_audio_path()` raising "audio folder unknown" — `similarity.py`'s path cache is populated by
`load_groups()` in the main process only, and `ProcessPoolExecutor` workers are separate
processes that never called it; fixed by resolving `audio_path` in the main process and passing
it through the task tuple, exactly the pattern the base repo's own `harmony_features.py` already
uses for the same reason. Full results: `output/signal_v2/source_roughness/pilot_per_file.csv`
and `pilot_analysis.json`.

**Result** (n=280, 40 files/emotion; effect sizes computed on this pilot sample only):

| variant | situation eps² | situation p | gender eps² | redundancy vs. existing partial_roughness (Pearson r) | redundancy vs. existing formant_roughness (Pearson r) |
|---|---|---|---|---|---|
| raw (this pilot's own extraction) | 0.011 (small, borderline) | 0.177 (n.s.) | 0.317 (large) | −0.086 | 0.036 |
| source (IAIF glottal, absolute amp) | **−0.009** (≈zero) | 0.742 | 0.176 (large) | 0.035 | −0.169 |
| source_norm (IAIF glottal, H_k/H1-normalized) | **−0.001** (≈zero) | 0.465 | 0.310 (large) | 0.036 | −0.162 |

For comparison, the two **already-existing, full-corpus** features re-checked on this exact same
280-file subsample (to separate a sample-size effect from a real method difference):
`existing_partial_roughness_mean`: eps²=**0.030** (small), p=0.027 — reaches the repo's own
significance convention even at n=280, and the low/high emotions (`sadness` lowest, `surprise`
highest) exactly match the direction the base repo's README reports on the full 34k-file corpus;
`existing_formant_roughness_mean`: eps²=0.020 (small), p=0.074 (borderline).

**Interpretation — the hypothesis is not supported by this pilot, and the null result is
reported as fully as a positive one would be.** Three things, not one:

1. **This is not simply a statistical-power artifact of the small sample.** The pre-existing
   `partial_roughness_mean` feature, evaluated on the *identical* 280-file subsample, still shows
   a small-but-real effect (p=0.027) — so 280 files is enough to detect a signal of that size when
   one exists. The source/source_norm variants' effect sizes (−0.009, −0.001) are not "a small
   effect that didn't reach significance at this sample size" — they are indistinguishable from
   zero outright, which more data narrowing the confidence interval is unlikely to rescue.
2. **This pilot's own `raw` arm is weaker than the pre-existing `partial_roughness_mean`** on the
   same 280 files (eps² 0.011 vs. 0.030) despite both operating on the unfiltered waveform — this
   is a real methodological difference, not the source-filter question. `partial_roughness.py`
   free-picks the top spectral peaks by amplitude, with no requirement that they sit near an
   integer multiple of F0; this pilot's harmonic extraction (reused from Phase B) deliberately
   searches only in a narrow tolerance band around each expected `k·F0`. That difference matters:
   if part of the emotion-relevant "roughness" signal in this corpus comes from energy that is
   *not* harmonically locked to F0 at all (subharmonics, broadband noise injected by breathy or
   harsh phonation, sidebands from fast F0 modulation), a harmonic-locked search structurally
   excludes exactly that energy by design — a plausible, testable explanation for why the
   free-peak-picking method retains more signal, worth checking directly in a follow-up rather
   than asserted here.
3. **The source/source_norm variants losing essentially all signal relative to even this pilot's
   own weaker `raw` arm is itself informative, and points toward — not away from — the vocal
   tract.** Read together with SP-V2-003's finding that this IAIF implementation's separation
   quality is uneven (good at some F0/vowel combinations, outright failing at others,
   particularly at higher F0), there are two live explanations that this pilot cannot yet
   distinguish between, and the brief's own instruction (§23) is to say so rather than pick one:
   (a) **the emotion-relevant roughness signal in this corpus lives substantially in the
   vocal-tract/formant structure itself, not the glottal source** — i.e. `formant_roughness` and
   `partial_roughness` were already measuring something real about vowel-articulation/spectral
   shape that a source-only feature cannot see by construction, regardless of separation quality;
   or (b) **this specific IAIF implementation's separation noise is large enough to wash out a
   real but smaller source-level effect**, especially given SP-V2-003's own finding that formant
   (and by extension, source-residual) quality varies a great deal by F0 register — and this
   pilot's 40-per-emotion sample was not stratified by F0/gender in a way that could separate
   "genuinely no source effect" from "source effect swamped by register-dependent separation
   noise." Distinguishing (a) from (b) is exactly the kind of follow-up the base repository's own
   `speaker_norm_features.py` precedent suggests (re-run split by gender/F0 register, the same way
   that script found `formant_roughness`'s gender-split effect tripled even though the pooled
   effect looked unremarkable) — **not yet done here**, flagged as the immediate next step before
   this line of work is either pursued further or set aside.

**Given this result, source-normalized roughness is not adopted, and a full-corpus run is not
launched** — the null result held up on the *same* sample where the existing raw features still
showed real signal, so scaling up the sample size alone is not the obvious next move. The
concrete next step, if this line is revisited, is the gender/F0-register-stratified re-analysis
described above, which would need to run before deciding whether (a) or (b) is the better
explanation — and only then would a full-corpus run (a background job on the order of ~10 hours
at this implementation's measured throughput) be worth committing to.

**Follow-up done same-day: gender-stratified re-analysis of this same 280-file pilot** (no new
extraction — a re-slice of the existing per-file CSV, split by `gender` before the same
Kruskal-Wallis/epsilon-squared test). The result reproduces the base repository's own
`formant_roughness` gender-confound story (README Phase 4: pooled effect near-zero, splitting by
gender roughly triples it for men, and men/women don't even agree on which emotion ranks highest)
almost exactly, on a completely different feature family:

| variant | male (n=72) eps² | male p | female (n=208) eps² | female p |
|---|---|---|---|---|
| raw | **0.097** (medium) | 0.056 (borderline) | −0.017 (≈zero) | 0.858 |
| source | 0.040 (small) | 0.199 | −0.023 (≈zero) | 0.972 |
| source_norm | **0.082** (medium) | 0.079 (borderline) | −0.025 (≈zero) | 0.989 |

The female subgroup shows literally no structure on any variant (p ≈ 0.86–0.99, effect sizes at or
below zero) — not "a small effect obscured by pooling," but no detectable relationship at all. The
male subgroup shows borderline-significant medium effects for `raw` and `source_norm` (both
p≈0.06–0.08 at n=72 — suggestive, not conclusive at this sample size) but **not** for the
un-normalized `source` arm, which stays weak even within-gender (eps²=0.040, p=0.199) — a hint
that the H1-normalization step (Phase F's `source_norm` motivation specifically) does earn its
keep *within* a register, even though it doesn't rescue the *pooled*, both-genders-together
result. This reframes the SP-V2-004 null result: it is very unlikely to be a simple case of "the
source signal has no emotion content" — it looks much more like the same gender/register confound
already documented for `formant_roughness`, now showing up in a feature family built through an
entirely different pipeline (IAIF + pitch-synchronous harmonic extraction vs. LPC formants). Given
this pilot's male subgroup is only 72 files, a properly powered gender-stratified (or F0-register-
stratified, which would generalize better than a binary gender split) re-run is the clear next
step before either committing to a full-corpus extraction or abandoning source-normalized
roughness — neither of which this pilot alone justifies.

## SP-V2-006 — Glide-aware (residual) jitter/shimmer

**Hypothesis**: the base repo's own peak-tracking jitter/shimmer (`voice_quality_features.py`)
predicts the next pitch period by linearly extrapolating the current period — an assumption a
real F0 glide/vibrato violates even with zero true perturbation, so a legitimate pitch movement
gets partly counted as jitter. Decomposing the observed period into a smooth expected trajectory
(from an independently-estimated F0 contour) plus a cycle-level residual, and defining jitter on
the residual only, should stay low under glide/vibrato with no true jitter and high only when true
perturbation is present.

**Method**: `signal_v2/voice_quality/synthetic_jitter_conditions.py` synthesizes period-
synchronously (each cycle's actual length and amplitude are set directly, cycle by cycle, with
phase kept continuous across cycle boundaries) so that "true jitter" and "true shimmer" can be
turned on or off independently of any smooth F0 trend — something the phase-integration approach
used for SP-V2-001/002's synthetic signals cannot do, since it has no cycle-level jitter at all by
construction. Conditions A–E follow the brief exactly (stable+jitter, glide+no-jitter,
glide+jitter, vibrato+no-jitter, vibrato+jitter); two more were added after the first pass (see
Result) — F/G, a much faster and larger-excursion pitch modulation (15 Hz, ±2 semitones, vs. D/E's
5.5 Hz/±0.5 semitone) — to stress-test the specific concern SP-V2-002 already flagged about its
own glide conditions possibly being too slow relative to the analysis window to expose the
theoretical weakness. `signal_v2/voice_quality/glide_aware_jitter.py` implements: (1)
`conventional_jitter_shimmer` — the base repo's own formula (mean absolute consecutive-period/
amplitude difference, normalized by the mean), reproduced independently rather than imported, run
here on **ground-truth cycle marks** so the formula itself can be evaluated separately from
pitch-marking error; (2) `residual_jitter_shimmer` — same formula, but applied to `epsilon_n = T_n
- T_hat_n`, where `T_hat_n` comes from an independently-estimated smooth F0 contour (interpolated
to each cycle's time) rather than the raw observed periods themselves; falls back to a moving-
average of the observed periods if no external F0 contour is supplied. Four variants of the
residual measure were compared: an **oracle** reference (the exact "intended" pre-jitter F0 used
by the synthesizer — an upper bound on how well this concept could work), a **pyin**-based
reference, a **WORLD Harvest**-based reference (added specifically to build on SP-V2-001's finding
that Harvest tracks smooth glides better than pyin), and a **fallback** (no external F0 estimator
at all, just smoothing the observed periods). All four are also checked against the base repo's
own `voice_quality_features.compute_voice_quality_features` (run unmodified, its own independent
peak-tracking) and against `parselmouth` (Praat, newly installed for this comparison — the base
repo's own README already states its jitter/shimmer is not Praat-equivalent, and this is the first
point in either codebase where an actual Praat value is computed for comparison). Full results:
`output/signal_v2/glide_jitter/metrics.csv` and a bar-chart comparison in
`output/signal_v2/glide_jitter/plots/`.

**Result**: the first pass (conditions A–E only) was inconclusive in an instructive way — with
ground-truth pitch marks, the *conventional* formula was only mildly affected by B/D's glide/
vibrato (0.003–0.006) versus A/C/E's true-jitter conditions (0.012–0.015), a clear ~2–4×
separation that does **not** show the dramatic contamination the hypothesis predicted. Adding the
much faster/larger F/G conditions changed the picture completely:

| condition | conventional (ground-truth marks) | residual, oracle F0 | residual, pyin F0 | residual, WORLD Harvest F0 | existing repo (own marking) | Praat |
|---|---|---|---|---|---|---|
| A (stable + jitter) | 0.0119 | 0.0119 | 0.0122 | 0.0114 | 0.1595 | 0.0108 |
| B (glide, no jitter) | 0.0061 | 0.0040 | 0.0045 | 0.0042 | 0.1435 | 0.0059 |
| D (vibrato, no jitter) | 0.0035 | 0.0044 | 0.0039 | 0.0042 | 0.1256 | 0.0035 |
| **F (fast accent, no jitter)** | **0.0385** | 0.0036 | 0.0190* | 0.0112 | 0.0918 | **0.0384** |
| G (fast accent + jitter) | 0.0394 | 0.0152 | 0.0231* | 0.0178 | 0.1390 | 0.0391 |

(*pyin's `residual_jitter_pyin` for F/G is flagged, not a genuine pyin-referenced value — see
below.)

**Interpretation — the hypothesis is confirmed, but only once the stress-test condition (F/G) is
included, and the mechanism turns out to be more specific than originally stated:**

1. **The conventional formula's contamination by smooth pitch motion is real but rate/depth-
   dependent, not universal.** B/D's slower, smaller glide/vibrato (matching SP-V2-002's own
   conditions) barely contaminates the conventional formula even with perfect pitch marks. Only
   F's much faster, larger-excursion modulation produces the dramatic, clearly-wrong result the
   hypothesis anticipated: **conventional jitter registers *higher* for F (0.0385, zero true
   jitter) than for every genuinely-jittery condition A/C/E (0.012–0.015)** — a clean false
   positive, and not a subtle one. This directly resolves the open question SP-V2-002 flagged
   about its own glide conditions possibly being too slow to expose the theoretical weakness: they
   were, and a faster/larger modulation was needed to see it.
2. **Praat is equally fooled** (0.0384 on F, essentially identical to the conventional formula on
   ground-truth marks) — this is not an artifact of a naive from-scratch implementation; it is a
   structural property of the local-jitter formula itself under fast pitch modulation, independent
   of how good the pitch-marking is. This is an important correction to the base repo's own
   framing (which attributed its jitter problems to not being "Praat-equivalent") — **the
   glide-contamination failure mode would survive switching to Praat**, whereas the base repo's
   *other*, separate problem (see point 4) would not.
3. **The residual concept works as designed when given a good smooth-F0 reference** — the oracle
   variant stays low on F (0.0036, matching the true no-jitter baseline) and separates F from G
   by roughly 4×. The *realistic* versions degrade this separation but do not eliminate it, and
   which realistic estimator is used matters a lot: **pyin failed to voice condition F/G at all**
   (0 of 101 frames — confirmed via a `pyin_n_voiced_frames` diagnostic added after noticing
   `residual_jitter_pyin` was numerically *identical* to the no-reference `fallback` value for F/G,
   which correctly traced back to the code's fallback branch silently triggering whenever the
   supplied F0 contour has fewer than 2 valid points), so the "pyin" column for F/G is actually the
   fallback estimate in disguise, not a real pyin-referenced result. **WORLD Harvest, tested here
   specifically because SP-V2-001 found it tracks smooth glides better than pyin, handled this
   condition without failing** (101/101 voiced) and gave a meaningfully better residual separation
   (0.0112 vs. 0.0178, ~1.6×) than the degraded pyin/fallback numbers (0.0190 vs. 0.0231, ~1.2×) —
   a concrete, practical payoff from SP-V2-001's earlier finding, and a specific reason to prefer
   Harvest over pyin as the smooth-F0 reference for this technique specifically, even though pyin
   remains the better general-purpose choice per SP-V2-001's broader results.
4. **The base repo's own implementation (`voice_quality_features.py`) has a different, arguably
   larger problem than glide contamination**: its jitter values are uniformly high across *every*
   condition (0.09–0.17) — roughly an order of magnitude above any ground-truth-marks-based
   measure — and barely separate true-jitter conditions (A=0.160, C=0.153, E=0.171) from
   no-jitter ones (B=0.144, D=0.126, F=0.092): a ~30–45% spread, far smaller than the 2–10×
   separations every other method achieves. This points at the pitch-*marking* step (its
   waveform-peak search, not the jitter formula) as the dominant noise source in the existing
   implementation — a related but mechanistically distinct problem from the one this phase set
   out to fix, and one that residual jitter alone would not solve if layered on top of the same
   marking algorithm.
5. **Shimmer's analogous "smooth-trend contamination" was not actually tested.** All of these
   synthetic conditions keep the underlying amplitude envelope flat except for the deliberately
   injected shimmer perturbation — there is no smooth intensity/loudness arc analogous to the F0
   glide, so nothing here validates or refutes a residual-shimmer benefit the way conditions F/G
   did for jitter. Worth flagging directly rather than letting the jitter result imply more than it
   does: `residual_shimmer_{pyin,world_harvest,fallback}` are in fact three prints of the *same*
   number in this implementation (amplitude smoothing uses only a moving average of observed
   amplitudes — no independent "smooth loudness estimator" analogous to pyin/Harvest for F0 exists
   yet), which is a real gap, not a redundant confirmation. The existing repo's own shimmer *does*
   show one suggestive false-positive on B (0.061, the highest shimmer value across all seven
   conditions despite zero true shimmer) — consistent with the same marking-noise explanation as
   point 4, but a proper smooth-amplitude-trend test condition would be needed to check this
   rigorously, and is a natural next addition to this synthetic suite.

**Given these results, residual jitter is adopted as a validated concept with two explicit
caveats carried forward**: (a) it needs a smooth F0 reference that can actually track fast/large
pitch modulation — WORLD Harvest, not pyin, per point 3 — and (b) it does not by itself fix the
base repo's larger pitch-marking-noise problem (point 4), which would need to be addressed
separately (a better cycle-marking algorithm) before residual jitter could meaningfully improve on
the existing feature in practice. A shimmer-specific smooth-trend test condition and an
independent smooth-amplitude estimator are flagged as unfinished, not silently assumed to work the
same way jitter did.

## SP-V2-008 — Band-wise periodicity / MVF (aperiodicity)

**Hypothesis**: the existing single broadband `hnr_mean`/`hnr_std` (audit §13, weakness #9)
cannot distinguish *where* in the spectrum a voice loses periodicity — breathiness (noise
concentrated at higher frequencies, low frequencies staying clean) and harshness/tenseness
(broadband instability) should look different under a band-resolved representation even if they
produce a similar single HNR number.

**Method**: `signal_v2/voice_quality/aperiodicity.py` uses `pyworld.harvest` (F0) +
`pyworld.d4c` (WORLD's D4C aperiodicity estimator — prior art, not claimed as novel; the
per-band/MVF feature design on top of it is this phase's contribution) to get, per frame, an
aperiodicity value at every frequency bin (0=periodic, 1=aperiodic). From this: **MVF** (the
lowest frequency at which aperiodicity crosses 0.5, per frame, then averaged), **band-wise
periodicity** (mean `1-aperiodicity` in 0–2 kHz / 2–4 kHz / 4–8 kHz), an **aperiodicity spectral
slope** (linear fit of aperiodicity vs. frequency per frame), and a **voiced-to-aperiodic
transition rate** (how often MVF jumps by more than 10% of Nyquist between adjacent frames).
Validated first on synthetic signals (per the brief's own physical-validation requirement) reusing
`signal_v2/f0/synthetic_signals.py`'s clean harmonic stack, additive-white-noise, and
breathy-noise conditions, checked against the pre-existing `hnr_mean` for directional agreement.
Full results: `output/signal_v2/aperiodicity/synthetic_validation.csv`.

**Result — and a second bug found by this validation, on top of SP-V2-001's**: the first pass
showed `low_band_periodicity`/`mid_band_periodicity`/`high_band_periodicity`/`mvf_mean` **all
completely flat** (to 3+ significant figures) across `breathy_noisy_harmonic` at `noise_mix ∈
{0.2, 0.4, 0.6}`, while the pre-existing `hnr_mean` correctly tracked the change (13.1→6.6→0.6 dB).
Investigating why led directly to the bug documented in SP-V2-001's write-up above (the noise
generator's ~51 Hz effective cutoff). After that fix, the same three `noise_mix` levels give a
properly graded, physically sensible result:

| condition | mvf_mean (Hz) | low-band periodicity | mid-band periodicity | high-band periodicity | existing hnr_mean (dB) |
|---|---|---|---|---|---|
| clean harmonic stack | 2789 | 0.981 | 0.422 | 0.077 | 16.6 |
| breathy, 20% mix | 2829 | 0.982 | 0.469 | 0.110 | 16.6 |
| breathy, 40% mix | 2717 | 0.943 | 0.450 | 0.105 | 13.1 |
| breathy, 60% mix | **0 (Harvest lost voicing entirely)** | 0 | 0 | 0 | 7.9 |

**Interpretation**: once the underlying noise-generation bug was fixed, `mvf_mean`/
`low_band_periodicity` and the pre-existing `hnr_mean` now agree directionally (both flat at 20%,
both drop at 40%) — the basic physical mechanism behaves as intended where WORLD's own F0 detector
(`harvest`) still finds voicing. The 60% condition surfaces a real, worth-noting limitation rather
than a clean success: **`pyworld.harvest` loses voicing entirely at a noise level where the
existing repo's `pyin`-based voice-quality pipeline still produces a (low but defined) HNR value**
— this aperiodicity feature family is only as robust as WORLD's own voicing decision, which this
one data point suggests may be *less* robust to this particular (high-passed, amplitude-mixed)
noise characteristic than pyin's voicing decision is, even though SP-V2-001 found the *opposite*
ordering for plain additive white noise at a controlled SNR (Harvest had zero voiced-frame error
at 0 dB SNR white noise, pyin had 25.7% error there) — a reminder that "which estimator is more
noise-robust" depends on the noise's specific spectral/statistical character, not just how much of
it there is, and shouldn't be generalized from one noise type to another. The bands' relative
sensitivities (low-band periodicity dropping proportionally at least as much as high-band between
20% and 40% mix) are not strongly interpreted here — a single synthetic run at three noise levels
is not enough to make a confident claim about which band is *most* sensitive to this specific
noise shape, only that the feature responds in the right direction overall.

**Given these results, the aperiodicity/MVF feature family is validated as physically sensible
on synthetic material and kept for future corpus-level evaluation, with two caveats carried
forward**: (a) its reliability is bounded by WORLD Harvest's own voicing robustness, which this
phase found to be noise-type-dependent rather than uniformly good or bad; (b) **no real-audio
pilot has been run yet** (unlike SP-V2-004's roughness pilot) — effect size against emotion labels,
gender sensitivity, and redundancy against the existing `hnr_mean`/`hnr_std` on real corpus data
are all open, and are the natural next step before this feature family could be added to the
classifier alongside the existing 117.

## SP-V2-007 — CWT multi-scale prosody

**Hypothesis**: the existing discrete note-segmentation representation (`melody_rhythm_features.py`,
kept unchanged and unreplaced per the brief) captures phoneme/syllable-level duration and pitch
interval structure, but cannot say *at which timescale* an emotion effect on the F0 contour shows
up — microprosody, syllable, word, phrase, or whole-utterance. A continuous wavelet decomposition
of the F0 contour, converted to a speaker-relative semitone scale first, should let that question
be asked directly.

**Method**: `signal_v2/prosody/cwt_prosody.py` converts a (possibly gappy, unvoiced-interrupted)
F0 contour to `12·log2(F0(t)/median(F0))`, linearly interpolates across unvoiced gaps and resamples
to a uniform 10 ms grid (both standard preprocessing steps in the CWT-prosody literature this
phase follows — e.g. the "Wavelet Prosody Analyzer" line of work by Suni, Aalto, Vainio et al. —
not claimed as novel here), then runs a Mexican-hat CWT (`PyWavelets`, newly available in this
venv) over 40 log-spaced pseudo-periods from 20 ms to the utterance length. Five bands
(microprosody 20–60 ms, syllable 60–250 ms, word 250–600 ms, phrase 0.6–1.5 s, utterance 1.5–4 s)
are then summarized by energy, variance, entropy, peak density, sign-change rate, and each band's
share of total cross-scale energy — the novel-if-anything part of this phase is this specific
derived-feature design on top of a standard CWT, not the CWT itself. Validated first on synthetic
semitone trajectories oscillating at exactly one known period each (per the brief's physical-
validation requirement), checking whether the dominant band matches the injected period. Full
results: `output/signal_v2/cwt_prosody/synthetic_validation.csv`.

**Result — a clean success, reported as straightforwardly as the less favorable results above**:
all four injected periods were correctly assigned to their expected band, with strong energy
concentration and physically sensible leakage into adjacent bands only:

| injected period | expected band | dominant band (recovered) | energy ratio in expected band |
|---|---|---|---|
| 0.04 s | microprosody | microprosody | 0.98 |
| 0.15 s | syllable | syllable | 0.94 |
| 0.40 s | word | word | 0.86 |
| 1.00 s | phrase | phrase | 0.66 (0.26 leaks into `utterance`, the adjacent band, on a 2 s test signal that only completes ~2 cycles — expected given the band edge sits at 1.5 s) |

**Interpretation**: the representation does what it's designed to do on synthetic material with a
single, unambiguous timescale present — a necessary but not sufficient condition (real emotional
speech will mix multiple timescales simultaneously, which this test does not probe). The phrase-
band result's leakage into the utterance band is not a failure; it is the expected consequence of
testing a slow oscillation against a fixed 2-second signal duration, and would be expected to
sharpen with longer utterances or would need a duration-normalized band-edge scheme to stay
comparable across this corpus's variable utterance lengths (unaddressed here — a real deployment
concern since utterance duration varies substantially across the corpus and this experiment always
used a fixed 2 s synthetic duration). **No real-audio pilot or corpus-level effect-size check has
been run for this feature family yet** (unlike SP-V2-004's roughness pilot) — that, plus deciding
how to handle the band-edge/utterance-duration interaction just noted, are the concrete next steps
before this joins the classifier.

## SP-V2-009 — Harmonic phase features

**Hypothesis**: every existing feature in the base repo is magnitude/F0-based (audit §13,
weakness #6); the relative phase between harmonics — whether they stay phase-locked to a
fixed relationship over time, or drift/decorrelate — is an unexplored dimension that a genuinely
periodic voiced signal and a degraded/noisy one should differ on.

**Method**: `signal_v2/pitch_sync/adaptive_window.py` was extended (backward-compatibly — the
existing `.freq`/`.amp`/`.amp_db_rel` fields and their values are unchanged, verified by re-running
SP-V2-002's own script after the change) to also record each harmonic's phase at the nearest FFT
bin to its refined-frequency estimate (an approximation — unlike frequency/amplitude, phase is not
parabolically interpolated here, since complex-valued sub-bin interpolation is more involved; this
is a documented limitation, not an oversight). `signal_v2/harmonic/harmonic_phase.py` computes the
relative phase `Δφ_k(t) = φ_k(t) - k·φ_1(t)` (wrapped to `[-π, π]`) and, since phase is a circular
quantity (audit-adjacent point from the brief: a plain mean/std would treat `-π` and `+π` as
opposite when they are nearly identical), summarizes it with circular statistics: the resultant
length `R_k = |mean_t(exp(i·Δφ_k(t)))|` (1 = perfectly phase-locked over time, 0 = fully
decorrelated) and circular variance `1-R_k`, per harmonic, plus a cross-harmonic, per-frame version
(`temporal_phase_instability` — the standard deviation, across time, of each frame's own
cross-harmonic resultant length). Validated on five synthetic conditions spanning clean, breathy
(the corrected noise from SP-V2-001/008's bug fix, at three mix levels), and heavy broadband noise
(0 dB SNR). Full results: `output/signal_v2/harmonic_phase/synthetic_validation.csv`.

**Result**:

| condition | mean phase coherence (R̄) | temporal phase instability |
|---|---|---|
| clean harmonic stack | 0.998 | 0.028 |
| breathy, 20% mix | 0.998 | 0.028 |
| breathy, 40% mix | 0.998 | 0.030 |
| breathy, 60% mix | 0.996 | 0.038 |
| additive white noise, 0 dB SNR | **0.632** | **0.138** |

**Interpretation**: the core mechanism validates cleanly at the extremes — a truly periodic
signal is (as it must be, by construction) almost perfectly phase-locked (0.998), and genuinely
overwhelming broadband noise (0 dB SNR, signal and noise power equal) collapses that coherence
sharply (0.632). The more interesting, unanticipated result is in between: **this phase-coherence
measure is far less sensitive to the same breathy-noise conditions that strongly perturbed the
SP-V2-008 aperiodicity/MVF feature** (which dropped from ~0.98 to 0 low-band periodicity across
the identical 20/40/60% breathy conditions) — here, even 60% breathy mix barely moves the needle
(0.998→0.996). A plausible mechanistic reason, not yet independently confirmed: phase coherence is
only ever evaluated **at the harmonic peaks that the extraction successfully locates** within a
narrow search tolerance around `k·F0`, so as long as a peak is findable there at all, its phase
reading can stay stable even while the surrounding noise floor (which is what aperiodicity/MVF is
actually measuring) rises substantially — a structural blind spot as much as a robustness property.
This suggests **phase coherence and band-wise aperiodicity are complementary, not redundant**:
they appear to have different sensitivity profiles across at least this one noise dimension, which
is itself a testable, useful property once both are run on real corpus data (an open item, along
with everything else in this line — no real-audio pilot has been run for this feature family yet).
The un-interpolated (integer-bin) phase reading is also worth flagging as a precision limitation
relative to the frequency/amplitude estimates in the same trajectory, which do get parabolic
refinement — not expected to change the qualitative synthetic result here, but worth keeping in
mind before this feature is used for anything requiring fine phase precision.

## SP-V2-010 — Harmonic modulation spectrum

**Hypothesis**: no existing feature models *how fast* harmonic structure changes over time — only
instantaneous values, averaged over the whole utterance. The temporal modulation spectrum of each
harmonic's amplitude trajectory `A_k(t)` (called `harmonic_modulation_features` in code per the
brief's own placeholder naming — final terminology deferred to a prior-art review, not decided
here) should recover a known modulation rate injected into a synthetic signal.

**Method**: `signal_v2/harmonic/harmonic_modulation.py` computes, per harmonic, the FFT-based
modulation spectrum of `A_k(t)` (the absolute, window-length-normalized amplitude trajectory from
`signal_v2/pitch_sync/adaptive_window.py` — deliberately *not* the H1-relative `amp_db_rel` used in
SP-V2-004/005, since a modulation applied equally to every harmonic would cancel out of a ratio to
H1 by construction, and this phase specifically wants to see shared as well as harmonic-specific
rate-of-change): dominant modulation frequency, low-rate (<5 Hz, syllabic-timescale) vs. high-rate
(5–50 Hz) energy, spectral centroid/bandwidth/entropy of the modulation spectrum, and a simple
inter-harmonic modulation coherence (mean pairwise Pearson correlation between harmonics' amplitude
trajectories). Validated on a synthetic signal with a known, controlled amplitude-modulation (AM)
rate applied uniformly across all harmonics (a scope limitation, disclosed directly: this does
**not** test harmonic-*specific* differential modulation, only shared/broadband AM — the
`inter_harmonic_modulation_coherence` field is designed to distinguish the two cases but was not
itself validated against a differential-modulation synthetic condition). Full results:
`output/signal_v2/harmonic_modulation/synthetic_validation.csv`.

**Result, including a bug found and fixed during validation**: the AM-rate recovery itself worked
cleanly on the first attempt — 2/4/8 Hz injected rates were recovered as 2.02/4.04/8.08 Hz (≤2%
error) across every harmonic. The **no-modulation control condition**, however, initially reported
a spurious "dominant modulation frequency" of exactly 50 Hz (the Nyquist edge of the 100 Hz
modulation-domain sampling rate implied by the 10 ms hop) — investigated because a flat trajectory
has no real modulation to detect. The actual amplitude trajectory for that condition had a
coefficient of variation of ~0.08% (std=0.0002 vs. mean=0.257) — pure extraction noise, not signal
— yet the noise happened to concentrate a locally prominent (relative to its immediate spectral
neighbors) artifact right at the Nyquist bin. **A first fix attempt — requiring the peak to exceed
4× the spectrum's own median magnitude — did not resolve it**: the artifact *is* locally prominent
by that relative-shape criterion, so the fix needed an absolute criterion instead. The working fix
checks the trajectory's own coefficient of variation *before* computing any spectrum at all
(`MIN_RELATIVE_VARIATION = 0.01`); trajectories flatter than that report `0.0` directly rather than
running an FFT whose peak location is meaningless. After this fix, the no-modulation condition
correctly reports `0.00 Hz` and the three real AM conditions are unaffected.

**Interpretation**: the core modulation-spectrum mechanism works as intended once the flatness
edge case is handled, and the debugging path here is worth keeping visible rather than collapsing
into just the final numbers — the first, plausible-looking fix (relative peak prominence) failed
silently on a case that mattered, and only an absolute-scale check on the input actually closed
the gap. This is a useful general lesson for every other "find the dominant X" style feature in
this codebase (harmonic phase's `dominant_scale_period_sec` in SP-V2-007's CWT work and the F0
estimators' own peak-picking are structurally similar and could in principle have the same class
of failure mode on a sufficiently flat/quiet input — not verified here, but worth keeping in mind).
As with every other phase in this line, **no real-audio pilot has been run** for harmonic
modulation features yet, and the harmonic-specific (as opposed to shared/broadband) modulation
case remains untested even synthetically — both are open next steps.

## Environment note

`pyworld==0.3.5`, `pyreaper==0.0.11`, `PyWavelets==1.8.0`, `torchcrepe==0.0.24`, and (later, for
SP-V2-006) `praat-parselmouth==0.4.7` were newly installed into the existing venv for this work
(none were present before — audit §15). All imported cleanly against the existing pinned
`numpy==1.26.4`/`torch==2.2.2` with no version conflicts reported by pip, consistent with what the
audit's PyPI metadata check predicted.

## Real-corpus pilot — SP-V2-006/007/008/009/010

Full write-up: [`signal_processing_v2_real_corpus_pilot.md`](signal_processing_v2_real_corpus_pilot.md).
Summary here for continuity with this log; that document has the complete per-family Evaluation
A–E tables, redundancy details, and decisions.

A stratified 1,400-file real-corpus sample (200/`situation` class) was extracted with all five
families' features computed together per file (`signal_v2/evaluation/extract_real_corpus_features.py`,
sharing F0/preprocessing cost across families — ~25 min at 1.27 files/sec with 6-way parallelism).
Two real bugs were found and fixed during this pass, both specific to *real* audio (neither showed
up in any synthetic test): (1) `ProcessPoolExecutor` workers calling `get_audio_path()` before
`load_groups()` populated their own process's path cache — fixed by resolving paths in the main
process and passing them through the task tuple, the same pattern the base repo's own extraction
scripts already use; (2) real (non-monotonic/duplicate) pitch marks from
`voice_quality_features.mark_pitch_periods` produced zero-length segments that crashed `np.ptp` —
never encountered by the always-monotonic synthetic marks used in every prior SP-V2-006 validation
— fixed by treating degenerate segments as zero amplitude, with a regression check confirming the
synthetic A–G benchmark's numbers were unaffected.

**Headline results** (full detail and decisions in the pilot doc): individually, **F010 (harmonic
modulation)** had this pilot's strongest standalone effect size (ε²=0.050, medium — the only
medium-or-larger effect across all 5 families/92 features) and best classifier delta
(+0.0136 macro F1, `situation`); **F008 (aperiodicity/MVF)** had the best gender-confound (7/11
Type A) and redundancy (0/11 features over ρ=0.9 with any existing feature) profile; **F006**
(glide-aware jitter) and **F007** (CWT prosody) both came back weak-to-artifact-driven and are
recommended for rejection — F007's only signal-bearing features (entropy) turned out to be
near-duplicates of existing frame-count/duration features (ρ up to 0.96), and F006's own core
methodological claim (residual jitter reduces F0-motion dependence) **did not reproduce on real
speech**, unlike its clean synthetic win, most plausibly because real conversational F0 movement in
this corpus doesn't reach the fast/large-excursion regime that made the synthetic stress condition
dramatic.

**The decisive finding, though, is the integrated (leave-one-out) ablation**: combining all five
families' 92 features together **underperforms the 117-feature baseline on both labels**
(`situation` Δ=−0.0029; `majority_vote` Δ=−0.0359, unanimously across all 5 folds), and *every*
family's marginal contribution to that combination is negative — including F010's, despite it
being individually this pilot's best result. Given F008 simultaneously has the best standalone/
confound/redundancy profile *and* the single worst individual marginal contribution
(−0.0177, `situation`), this pattern reads far more like small-sample overfitting (92 new features
against ~1,120 training rows/fold, worse for `majority_vote`'s class imbalance) than like a clean
"these features don't work" result — but that is this pilot's best-supported hypothesis, not a
resolved fact. **SP-V2-011 (adaptive harmonic model) is deferred**: none of the failure modes found
here trace to harmonic-tracker imprecision (F010, the family most dependent on the shared tracker,
was the strongest performer, not the weakest), so a more precise tracker would not address what
this pilot actually found limiting.

## SP-V2-012 — Scaling, Feature-Efficiency, and Harmonic-Modulation Mechanism Analysis

Full write-up: [`signal_processing_v2_SP-V2-012.md`](signal_processing_v2_SP-V2-012.md); overnight
continuation summary: [`SP-V2-012_overnight_summary.md`](SP-V2-012_overnight_summary.md). Follows up
on the real-corpus pilot's open question (does the negative combined-family result reflect
sample-size overfitting, intrinsic noise, redundancy, a classifier-specific artifact, or a sampling
artifact?). F006/F007 stayed frozen at REJECT throughout; SP-V2-011 concluded HOLD; no new feature
families added.

**Daytime phase** (feature-efficiency, mechanistic analysis, F008/F009 complementarity, N=1,400
classifier comparison) found `inter_harmonic_modulation_coherence` selected as training-fold top-1
in 5/5 folds and replicating on an independent 700-file subset (ε²=0.047 vs. 0.050); H1-only beating
a generic-envelope control by 7.5×; the family's dominant modulation rate far slower than the
syllabic rate originally assumed (median 0.54 Hz vs. an assumed ~4–8 Hz); and F008+F010 showing a
+0.0301 marginal `majority_vote` benefit in a single-seed check.

**Overnight continuation ran the full 3-seed nested sample-scaling curve (N∈{1,400, 2,800, 5,600})
plus three targeted confound checks (0.54 Hz artifact, common-envelope removal, tracker-quality),
and the results substantially revised the daytime conclusion about HMC:**

- **The single-seed N=1,400 result for F010-alone (+0.0136 Δmacro-F1) was largely a favorable-seed
  artifact** — averaged across 4 independent seeds at the same N, the true effect is +0.0028, and it
  does not grow with N (2,800: −0.0060; 5,600: +0.0010). Closer to Pattern B (flat/noisy) than
  Pattern A for F010 alone on `situation`.
- **Common-envelope removal collapses `inter_harmonic_modulation_coherence` almost entirely**:
  removing the shared/geometric-mean amplitude envelope before computing cross-harmonic coherence
  drops its emotion ε² from 0.0501 (medium, p=2.6×10⁻¹⁴) to **0.0030 (not significant, p=0.118)**,
  and its gender ε² from 0.287 to 0.004. The envelope-removed and raw versions are statistically
  independent (ρ=−0.026). **This overturns the daytime conclusion that HMC is F010's standout,
  harmonic-specific feature** — it is now read as substantially a proxy for shared loudness-contour
  dynamics, not independent harmonic coordination. `inter_harmonic_modulation_coherence` is
  downgraded to REJECT.
- **H1-specific modulation survives the equivalent scrutiny (so far) and is now the strongest F010-
  derived candidate**: the generic-envelope comparison replicated overnight, and the "it's just
  tracking reliability" alternative explanation is not supported (H5, presumably the least reliably
  tracked harmonic, scored second-best of the five, ahead of H2/H3/H4). H1 has **not yet** been
  tested against the same common-envelope confound that sank HMC — flagged as the single highest-
  priority open question.
- **F008+F010's `majority_vote` benefit is the most replicated finding of the whole investigation**:
  mean Δmacro-F1 = +0.0059 / +0.0076 / +0.0072 across 2–4 seeds at N=1,400 / 2,800 / 5,600 —
  consistent sign and magnitude at every tested sample size, not a single-seed fluke.
- **Negative-control scaling is clean and highly reproducible**: at every N and both labels (12/12
  comparisons), real F010/combined features beat dimension-matched Gaussian-noise and permuted
  controls — confirming a real, structural "cost of added dimensions" that real signal must
  overcome, even where the real-vs-baseline delta itself is close to zero.
- **A classifier-choice artifact explains part of the apparent RF-only pattern**: at N=1,400, RF
  is the only classifier finding value in F010-alone for `situation`; for `majority_vote` the
  pattern reverses outright (RF −0.0204, Linear SVM +0.0121 in 5/5 folds). "F010 doesn't help
  `majority_vote`" was partly an RF-specific artifact.
- **SP-V2-011 verdict: HOLD.** Neither daytime nor overnight evidence points at harmonic-tracker
  imprecision — the surviving candidate (H1, single-harmonic) is less exposed to multi-harmonic
  tracking issues, and the failed candidate (HMC) failed on a representation confound a better
  tracker would not fix.

Three real bugs were found and fixed, all specific to newly-added code, none affecting previously-
reported results from earlier phases: (1) a boolean-indexing bug in the new F0-dynamics probe (an
already-voiced-filtered array indexed by the full mask a second time); (2) a naive nested-sampling
design that didn't guarantee subset nesting across sample sizes, fixed by permutation-prefix
sampling plus canonical-permutation reconstruction at analysis time (parallel extraction doesn't
preserve submission order in the output file); (3) a `numpy` "mean of empty slice" warning (benign,
not a correctness bug) from frames where all 5 harmonics were simultaneously untracked, suppressed
for log cleanliness. One classifier-comparison point (N=5,600/seed 43) did not finish within the
overnight session — the RF-only scaling curve (17/18 points) does not depend on it.

**Final feature decisions**: F010 (all 33 features) REJECT; `inter_harmonic_modulation_coherence`
alone REJECT; H1-specific modulation (6-dim) CONDITIONAL (strongest surviving candidate, pending the
envelope-confound re-check); F008 alone CONDITIONAL; F009 alone CONDITIONAL leaning REJECT
(weakens with N); F008+F010 combination CONDITIONAL (most replicated positive result). No candidate
reached an unqualified KEEP — every one has a specifically-named open verification step remaining.

## Next steps

**Highest priority (from SP-V2-012's overnight conclusion)**: re-test H1-specific modulation
against the same common-envelope-removal confound that overturned `inter_harmonic_modulation_
coherence` — H1's absolute-amplitude modulation shape has not yet been checked against its
envelope-relative counterpart, and it is now the leading surviving F010-derived candidate, so this
is the single most important open question in the whole SP-V2 line as of this entry. Second
priority: finish the one incomplete scaling point (N=5,600/seed 43 classifier comparison, both
labels — was still running after several hours as SP-V2-012 concluded; check
`output/signal_v2/SP-V2-012/point_N5600_seed43_*.json` before re-launching). Third: test F008 +
H1-specific-modulation (rather than F008 + all-33-F010) as a combination, untested and plausibly
stronger than either component result reported so far.

Smaller flagged follow-ups, still open: (1) duration-normalize F007's entropy-based sub-features
(F010's own entropy features were checked overnight and found *less* duration-confounded than
F007's, ρ=0.65–0.70 vs. 0.86–0.96 — lower priority than previously stated) and re-test whether real
signal survives; (2) the gender/F0-register-stratified re-analysis of SP-V2-004's source-roughness
pilot (from the earlier, separate pilot); (3) a shimmer-specific smooth-amplitude-trend synthetic
condition for SP-V2-006, plus an independent smooth-loudness estimator; (4) SP-V2-009's
un-interpolated (integer-bin) phase precision; (5) nonlinear/spline HMC analysis and bootstrap
confidence intervals, both explicitly deferred overnight given time constraints and, for the
former, lower priority now that HMC itself is largely rejected; (6) speaker-relative normalization
discipline (§14 of the brief) for any of these new features has not been attempted at all yet,
since none of the current features use cross-file statistics — would only become relevant if a
future normalized variant is added.

**Do not proceed with SP-V2-011** — concluded HOLD as of SP-V2-012's overnight session. See
`signal_processing_v2_SP-V2-012.md`'s "Implications for SP-V2-011" section for the full reasoning.
