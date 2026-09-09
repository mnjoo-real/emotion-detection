# Signal Processing v2 — Implementation Plan

Companion to [`signal_processing_v2_audit.md`](signal_processing_v2_audit.md) (read that first —
this plan assumes its findings). Scope: turn the audit's findings into a concrete, phased build
plan for a Harmonic–Source–Prosodic speech representation, per the research brief. Nothing here
deletes or breaks any existing script; every new module is additive and lives under a new
`signal_v2/` package, evaluated against the existing pipeline as a baseline, never replacing it
outright.

## Current vs. proposed, component by component

| Component | Current method | Problem | Proposed v2 | Expected benefit | Validation |
|---|---|---|---|---|---|
| F0 estimation | `librosa.pyin`, 8 kHz, fixed fmin/fmax, never benchmarked | No ground-truth validation; unknown octave-error rate; unknown glide-tracking quality; is the single point of failure for every downstream harmonic feature | Benchmark pyin vs. WORLD Harvest vs. REAPER vs. CREPE/torchcrepe on synthetic signals with known F0; keep whichever wins per condition, or keep pyin if it's already competitive | Quantified confidence in the F0 substrate everything else depends on | SP-V2-001: synthetic ground-truth F0 tracking (RMSE-cents, octave-error rate, voiced-frame error, trajectory smoothness) |
| Analysis window | Three unrelated fixed windows (25 ms LPC, ~256 ms pyin, 128 ms FFT-peak), none pitch-adaptive | Same wall-clock window covers a very different number of pitch periods depending on speaker F0 — inconsistent harmonic resolution and inconsistent nonstationarity-within-frame across speakers | Pitch-synchronous window `L(t) = c·T0(t)` for `c∈{2,3,4}`, with fixed min/max clamps; compared against the existing fixed windows, not replacing them | More consistent per-cycle harmonic amplitude/frequency estimates across speakers of different pitch range | SP-V2-002: harmonic frequency/amplitude stability, dissonance/inharmonicity stability, emotion effect size, classifier contribution — fixed vs. 2T0/3T0/4T0 |
| Source–filter structure | None — every harmonic/roughness feature operates on the observed (source⊛filter) waveform/spectrum | Formant structure, glottal excitation, and channel effects are inseparably mixed in every existing "harmonic" feature | IAIF / GFM-IAIF glottal inverse filtering → separate glottal-source and vocal-tract branches | Enables source-normalized harmonic/roughness features that (hypothesis) travel better across speaker/domain than raw-spectrum ones | SP-V2-003: synthetic source-filter signal with known glottal source, reconstruction error; visual inspection (waveform / estimated source / estimated VT envelope) on real audio |
| Harmonic representation | Single-scalar-per-file (mean/std over whole utterance) of formant roughness, melodic dissonance, partial roughness | No per-harmonic time-varying amplitude/frequency/phase; can't see *when* or *how fast* harmonic structure changes | Adaptive harmonic model (aHM/QHM-inspired) giving `A_k(t)`, `f_k(t)`, `φ_k(t)` per harmonic | Foundation for every later phase (relative profile, roughness, phase, modulation) | Implemented method is prior art — validated by how well it reconstructs a known synthetic AM/FM harmonic signal, not claimed as novel itself |
| Roughness/dissonance basis | Always computed on the raw observed spectrum (`partial_roughness.py`) or on formants (`harmony_features.py`) | Raw dissonance is a source+filter+channel mix — a likely explanation for the sarcasm-project's cross-corpus sign flip the user described | Same Sethares model, computed on (a) raw spectrum [existing], (b) inverse-filtered glottal source, (c) amplitude-normalized glottal source | Tests directly whether source-normalized roughness is more emotion-consistent / less speaker-and-domain-sensitive than raw | SP-V2-004: effect size, gender sensitivity, corpus-direction-consistency (reusable module, portable to the sarcasm repo), classifier ablation |
| Harmonic amplitude scale | Absolute LPC/FFT-peak amplitudes only | Confounded with recording level, speaker vocal effort, distance to mic | Fundamental-relative profile `H_k = 20log10(A_k/A_1)`, H1-H2, H2-H4, spectral slope/centroid/entropy/flatness | Tests whether relative representation keeps emotion signal while shedding speaker/recording variance | SP-V2-005: absolute vs. relative correlation with gender/recording proxies, redundancy vs. existing formant/energy features |
| Jitter/shimmer | Waveform-peak period tracking, linear-extrapolation period prediction — conflates true perturbation with intentional F0 glide (confirmed in audit §8) | Exactly the glide-contamination failure the user already found empirically | Decompose observed period/amplitude into smooth trajectory + cycle-level residual; jitter/shimmer computed on the **residual** only | Should stay low under a smooth glide/vibrato with no true jitter, stay high only when true perturbation is present | SP-V2-006: synthetic A–E conditions (stable+jitter, glide+no-jitter, glide+jitter, vibrato+no-jitter, vibrato+jitter), compared against existing implementation and Praat (`parselmouth`) |
| Periodic/aperiodic structure | Single broadband HNR scalar | Can't distinguish breathy (high-freq noise) from harsh/tense (broadband instability) | Band-wise periodicity / MVF (WORLD aperiodicity as the concrete implementation) | Voice-quality feature that maps to breathiness/harshness/tenseness, not just "clean vs. noisy" | Effect size + generalization vs. existing hnr_mean/std |
| Phase | Not used anywhere | Unexplored dimension | Relative harmonic phase `Δφ_k = φ_k - k·φ_1`, circular statistics (mean resultant length, circular variance) | Tests whether phase coherence/distortion carries independent emotion signal | Circular-stats sanity checks on synthetic signals with known phase relationships, then effect size |
| Temporal multi-scale prosody | Discrete note segmentation only (kept) | Can't see which timescale (micro/syllable/word/phrase/utterance) an emotion effect lives at | CWT on speaker-relative semitone F0 (and, later, on roughness/H1-H2/slope/MVF/aperiodicity/energy trajectories) | Answers "at what timescale does emotion show up," complementing (not replacing) note-level features | Per-scale energy/entropy/peak-density effect size, alongside existing note_dur_mean etc. |
| Harmonic dynamics | None | No representation of *how fast* harmonic structure changes over time | Per-harmonic modulation spectrum of `A_k(t)` (FFT or CWT over time) | Second-order dynamics feature family, complementary to Phase D/E's per-instant profile | Effect size + inter-harmonic modulation coherence, generalization |
| Speaker/domain normalization | Post-hoc gender z-scoring, fit on the **full corpus** (leakage risk, audit §10) | Any new normalized v2 feature could repeat this leakage if built the same way | All new normalization statistics computed **inside** the CV loop (fit on train fold only); utterance-relative and gender-relative variants kept explicitly distinct; no speaker-relative (no speaker ID) | Removes a latent leakage risk while keeping the ratio/z-score normalization concept that already proved useful for formants | Explicit train/test boundary check in the ablation harness (SP-V2-007) |

## Proposed experiments

- **SP-V2-001** — F0 estimator benchmark (pyin vs. WORLD Harvest vs. REAPER vs. CREPE) on a
  synthetic ground-truth dataset. *Priority 1.*
- **SP-V2-002** — Pitch-synchronous window ablation (fixed 40 ms vs. 2T0/3T0/4T0) for harmonic
  frequency/amplitude stability. *Priority 1.*
- **SP-V2-003** — Glottal inverse filtering (IAIF/GFM-IAIF) validation on synthetic source-filter
  signals + qualitative inspection on real audio. *Priority 2.*
- **SP-V2-004** — Raw-spectrum vs. source-normalized roughness: effect size, gender sensitivity,
  cross-corpus direction consistency, classifier ablation; reusable module for the sarcasm repo.
  *Priority 3.*
- **SP-V2-005** — Relative harmonic profile (H1-H2, H2-H4, slope, centroid, entropy, flatness) vs.
  absolute: speaker/gender-variance reduction while preserving emotion signal. *Priority 3
  follow-on.*
- **SP-V2-006** — Glide-aware residual jitter/shimmer vs. conventional (this repo's own + Praat
  via `parselmouth`) on synthetic A–E conditions. *Priority 5.*
- **SP-V2-007** — CWT multi-scale prosody on speaker-relative semitone F0 (and secondarily on
  other trajectories). *Priority 4.*
- **SP-V2-008** — Band-wise periodicity / MVF / WORLD aperiodicity vs. broadband HNR. *Priority 6.*
- **SP-V2-009** — Harmonic phase features (circular statistics). *Priority 7.*
- **SP-V2-010** — Harmonic modulation spectrum. *Priority 8.*
- **SP-V2-011** — Adaptive harmonic model (aHM/QHM-style) implementation underlying SP-V2-004/005/
  009/010's per-harmonic `A_k(t)/f_k(t)/φ_k(t)` trajectories. *Priority 2, built alongside
  SP-V2-003.*

## Module layout (adapted to this repo's flat, single-purpose-script convention)

The existing repo has no `src/` — every feature family is one top-level `*_features.py` script
reusing `similarity.py`'s `load_groups()`/`get_audio_path()` and a common
CPU-process-pool-with-resume extraction pattern. v2 introduces one new top-level package,
`signal_v2/`, so new code is clearly separated but still reuses that pattern rather than
reinventing it:

```
signal_v2/
  __init__.py
  common/
    __init__.py
    audio_io.py           # shared waveform loading (thin wrapper reusing similarity.get_audio_path)
    extraction_runner.py  # shared process-pool + resume-on-rerun harness (factored out of the
                           # repeated _process_task/load_already_processed/build_tasks/extract_all
                           # boilerplate every existing *_features.py duplicates)
  f0/
    __init__.py
    synthetic_signals.py  # ground-truth synthetic F0 test signals (Phase A)
    estimators.py         # pyin / WORLD Harvest / REAPER / CREPE wrappers, one common interface
    benchmark.py          # SP-V2-001 driver
  pitch_sync/
    __init__.py
    adaptive_window.py    # pitch-synchronous window + per-cycle harmonic freq/amp extraction
    benchmark.py          # SP-V2-002 driver (fixed vs 2T0/3T0/4T0)
  source_filter/          # Priority 2
    __init__.py
    iaif.py
    synthetic_validation.py
  harmonic/               # Priority 2-3, 7-8
    __init__.py
    adaptive_harmonic_model.py
    harmonic_profile.py
    source_roughness.py
    harmonic_phase.py
    harmonic_modulation.py
  voice_quality/          # Priority 5-6
    __init__.py
    glide_aware_jitter.py
    aperiodicity.py
  prosody/                # Priority 4
    __init__.py
    cwt_prosody.py
  evaluation/
    __init__.py
    synthetic_metrics.py  # RMSE-cents, octave-error rate, voiced-frame error, trajectory smoothness
    effect_size.py        # thin wrapper reusing dissonance_tonality_stats.py's epsilon-squared code
    ablation.py            # baseline vs baseline+feature CV harness with per-fold normalization
    generalization.py

output/signal_v2/
  f0_benchmark/            # SP-V2-001: per-condition CSV + plots
  pitch_sync/               # SP-V2-002
  source_filter/            # SP-V2-003
  source_roughness/         # SP-V2-004
  harmonic_profile/         # SP-V2-005
  glide_jitter/              # SP-V2-006
  cwt_prosody/               # SP-V2-007
  aperiodicity/              # SP-V2-008
  harmonic_phase/            # SP-V2-009
  harmonic_modulation/       # SP-V2-010
```

Each experiment directory gets, at minimum: a config (JSON), the resulting feature/metric CSV,
any plots, and a short `run_info.json` (command, git commit hash, random seed) — per the user's
reproducibility requirement (§19). `evaluation/effect_size.py` and `ablation.py` are the shared
tools every later phase reuses, so effect-size-first, per-fold-normalization discipline is
enforced structurally rather than re-derived per phase.

## Dependencies needed (see audit §15 — already verified installable, no conflicts found)

Install now for Priority 1 (F0 benchmark): `pyworld`, `pyreaper`. Optional CREPE baseline:
`torchcrepe` (reuses the existing `torch==2.2.2`). Later priorities: `PyWavelets` (Phase J/CWT),
`praat-parselmouth` (independent jitter/shimmer/HNR reference for Phase G's validation).

## Conflicts with existing code

None expected at the file level — `signal_v2/` is a new top-level package, no existing script is
imported-from or modified. The one thing to watch: `signal_v2/common/audio_io.py` should import
`get_audio_path`/`load_groups` from the existing `similarity.py` rather than reimplementing WAV
path resolution, so the two datasets' (4차년도/5차년도_2차) file layout and situation-label
aliasing (`anger`→`angry`, `sad`→`sadness`) stays in exactly one place.

## Priority 1 implementation plan (this session)

1. `signal_v2/f0/synthetic_signals.py`: generator functions for stationary sinusoid+harmonics,
   harmonic stack, linear glide, exponential glide, vibrato, abrupt pitch jump, additive white
   noise, breathy/noisy harmonic signal — each returning `(waveform, sr, true_f0(t), true_voiced(t))`.
2. `signal_v2/f0/estimators.py`: one function per estimator (`estimate_pyin`,
   `estimate_world_harvest`, `estimate_reaper`, `estimate_crepe` if `torchcrepe` import succeeds,
   soft-skip otherwise) — common return shape `(times, f0, voiced)`.
3. `signal_v2/evaluation/synthetic_metrics.py`: RMSE-in-cents (voiced-and-both-tracked frames
   only), voiced/unvoiced frame accuracy, octave-error rate (freq ratio near 0.5/2.0), trajectory
   smoothness (second-difference roughness of the estimated contour).
4. `signal_v2/f0/benchmark.py` (SP-V2-001 driver): run every estimator over every synthetic
   condition, compute metrics, save `output/signal_v2/f0_benchmark/metrics.csv` +
   per-condition overlay plots (true vs. estimated F0).
5. `signal_v2/pitch_sync/adaptive_window.py`: given a waveform + F0 contour, extract per-cycle
   harmonic frequency/amplitude using `c*T0` windows (`c∈{2,3,4}`, clamped) vs. the existing fixed
   window, reusing `harmony_features.sethares_dissonance`-compatible outputs.
6. `signal_v2/pitch_sync/benchmark.py` (SP-V2-002 driver): stability comparison on synthetic
   signals first (do fixed vs adaptive windows give more/less stable harmonic amplitude estimates
   under a known glide?), then a small real-audio smoke test.
7. Update this plan's checklist and log results (hypothesis/method/result/interpretation) once
   Priority 1 lands — failures get written up exactly as prominently as successes, per the user's
   §20 (README/log discipline) and §15 (never hide failed experiments).

Everything past Priority 1 (source-filter separation, source-normalized roughness, CWT prosody,
glide-aware jitter/shimmer, aperiodicity, phase, modulation spectrum) is deliberately deferred
until Priority 1's benchmark results are in — per the user's own instruction not to implement
every phase at once, and to let Priority 1-3's findings inform whether Priority 4+ is worth
pursuing as specified or needs revision.
