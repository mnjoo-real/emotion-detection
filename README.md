# Emotion Recognition: An Acoustic / Music-Theoretic Exploration

This repository explores a Korean multimodal-adjacent speech corpus with a from-scratch question: **can
7-way emotion be recognized from the raw acoustics of speech, using features borrowed from music theory
(harmony, melody, rhythm) rather than a generic MFCC/deep-embedding pipeline** — and, separately, does a
literature method for representing emotion as *regions* rather than points in Valence-Arousal space actually
hold up once you leave the clean, self-reported data it was built on?

## Original dataset (condensed)

**AI Hub's "Dialogue Speech Dataset for Emotion Classification"** (4차년도 + 5차년도_2차 batches), 33,964
Korean utterances labeled two different ways: a **scripted `situation`** (the emotion the line was written to
convey — `angry/disgust/fear/happiness/neutral/sadness/surprise`) and **five raters' independent perceived
emotion + intensity** per utterance. Metadata also carries speaker `age`/`gender` but, critically, **no speaker
ID** — a limitation this exploration runs into directly (Phase 5) and tries, and fails, to work around (Phase
11). Raw audio (`4차년도/`, `5차년도_2차/`, `샘플/`, ~20GB+) and the raw metadata CSVs are not included in this
repository (size and redistribution terms); everything downstream of feature extraction is.

## Setup

- Python 3.11 venv. `torch==2.2.2` in this venv only supports `numpy<2` — the venv originally had `numpy==2.0.2`
  (for `librosa`/`scipy`) and adding `torch`/`transformers` broke silently in *both* directions (`tensor.numpy()`
  raised `Numpy is not available`; feeding a numpy array back into a `transformers` processor raised `Could not
  infer dtype of numpy.float32`). Fixed by downgrading to `numpy==1.26.4` in the same venv and re-verifying the
  existing librosa/scipy pipeline still worked, rather than keeping a second venv per model.
- `transformers` also needed pinning: the latest release (`5.15.1`) requires `torch>=2.5` and silently disables
  itself under `2.2.2`; pinned `transformers==4.44.2`.
- `librosa`, `scipy`, `scikit-learn`, `torch`/`torchaudio`/`transformers` (wav2vec2 VAD), `matplotlib`
  (ellipse plots). No Praat bindings — jitter/shimmer/HNR are a from-scratch, simplified approximation (Phase 7),
  not a Praat-equivalent implementation.
- Audio: 16-bit PCM WAV per utterance, read via `scipy.io.wavfile`/`librosa`, resampled per-feature as needed
  (8kHz for pitch tracking, 12kHz for formants, 16kHz for the wav2vec2 model).

## Methodology (the decisions that mattered most)

1. **Effect size, not p-value, is the real signal.** With 33,964 rows split across 7 unbalanced classes
   (1,755–9,126 each), a Kruskal-Wallis test on *any* acoustic feature returns p<0.001 — including features that
   turn out to carry essentially no usable signal. Every comparison in this repo is reported with
   **epsilon-squared** (rank-based effect size) alongside the p-value, and "negligible" (ε²<0.01) is treated as
   a real answer, not a rounding error (Phase 6).
2. **Two different labels give two different answers, on purpose.** `situation` (what the line was scripted to
   convey) and `majority_vote` (what 5 raters actually heard, intensity-weighted, ties dropped) are evaluated
   **in parallel throughout**, never as a single "the" label. `majority_vote` consistently shows larger effect
   sizes and better classifier performance — acted intent and perceived reality are correlated but not
   identical targets.
3. **No speaker ID means every accuracy number here carries an asterisk.** Random train/test splits can put the
   same speaker's other lines in both sides. This is named explicitly rather than hidden, an unsupervised
   fix is attempted and fails honestly (Phase 11), and the eventual "is this real" question is answered instead
   with what *is* available without speaker ID: permutation testing and repeated k-fold CV (Phase 9).

### Phase 1 — Harmony

Monophonic speech has no chords, so "harmony" is translated three ways (`harmony_features.py`,
`partial_roughness.py`): **formant roughness** (Sethares 1993 sensory-dissonance model applied to LPC-derived
formants F1–F4), **melodic dissonance** (the same dissonance model applied to consecutive voiced-frame pitch
jumps), **tonal stability** (entropy of the voiced-pitch distribution folded into 12 pitch-classes around the
utterance's own median), and **partial roughness** (FFT peak-picking on the raw spectrum → dissonance between
actual harmonics — a voice-*texture*/clarity-vs-harshness axis, distinct from the formant-structure axis above).

| feature | ε² (`situation`) | direction |
|---|---|---|
| `pitch_class_entropy` | 0.041 (medium) | sadness lowest ↔ surprise highest |
| `melodic_dissonance_mean` | 0.027 (small) | sadness ↔ surprise |
| `partial_roughness_mean`/`std` | 0.016 / 0.013 (small) | sadness ↔ surprise |
| `formant_roughness_mean` | 0.004 (**negligible**) | fear ↔ happiness |

Sadness is the lowest and surprise the highest on every one of these — a clean low-arousal/high-arousal axis,
found before any classifier was involved.

### Phase 2 — Melody

Melody = pitch organized in time, kept deliberately separate from Phase 1's simultaneous-pitch view and from
raw pitch statistics. `melodic_profile.py` measures interval size/direction in **semitones** (register-invariant
across speakers): step-vs-leap ratio, rising/falling/flat proportions, a 7-bin contour histogram.

`melody_rhythm_features.py` goes further and adds the piece Phase 2's semitone intervals leave out — **duration**.
Consecutive voiced frames within one semitone of each other are grouped into "notes"; each note's length and the
gap between note onsets (IOI) are measured directly, alongside the semitone-interval statistics between notes.

- **`note_dur_mean` turned out to be the single strongest feature found anywhere in this project** (ε²=0.051,
  medium — beating every harmony feature above): sadness draws notes out (0.35s) with wide gaps between them
  (IOI 0.41s, 2.37 notes/sec); surprise is short and rapid-fire (0.29s notes, IOI 0.37s, 2.58 notes/sec).
- `duration_pitch_corr` (does a higher-pitched note run longer or shorter?) splits by **valence**, not arousal:
  negative for fear/sadness/disgust/angry/neutral, positive for happiness/surprise — a smaller effect (ε²≈0.004)
  but a qualitatively different, valence-aligned pattern worth flagging on its own.
- Despite the huge standalone effect size, `note_dur_mean` ranks only **54th of 117** features in the final
  Random Forest's importance — its information overlaps with MFCC/pitch/energy statistics already in the model.
  Standalone statistical strength and multivariate classifier value are not the same question (see Phase 6,
  and again in Phase 10).

### Phase 3 — Tonality (major/minor key-fit) — rejected

`tonality_features.py` folds voiced F0 into a 12-pitch-class histogram and correlates it against all 12
rotations of the Krumhansl-Schmuckler major and minor key profiles — the same literal "is this passage in a
major or minor key" algorithm used on actual music. **Every effect size came back negligible** (ε²=0.0005–0.0026,
`situation` and `majority_vote` alike), and every emotion's mean `key_major_minor_diff` sits in the same narrow
negative band (–0.029 to –0.040) regardless of emotion — a structural property of how folded speech-pitch
histograms compare to a fixed template, not an emotion signal. The mechanism is the same reason melody-as-pitch
worked and melody-as-*scale* didn't: a discretized, quantized-pitch framework doesn't fit continuously-gliding
speech F0 the way it fits an actual melody.

### Phase 4 — Speaker/gender confound and normalization

No speaker ID exists, but `gender` does, and gender turns out to matter a lot: **gender explains 43% of the
variance in `pitch_mean`** (male 127.9Hz vs. female 212.8Hz) against only 3% for `situation`, and the gender
ratio itself varies 16.5%–41.7% across emotion categories — a real, uneven confound, not a hypothetical one
(`speaker_norm_features.py`).

- **Formant ratios** (`f2_f1_ratio`, `f3_f2_ratio`, `f4_f3_ratio` — standard phonetic speaker-normalization,
  replacing raw Hz with adjacent-formant ratios): a clear win. `f3_f2_ratio` alone (ε²=0.011) later outranks all
  four raw `f1–f4_mean` features in the final Random Forest.
- **Gender z-scoring pitch**: splitting `situation`'s effect on `pitch_mean` by gender (male ε²=0.027, female
  ε²=0.021) shows the sadness-lowest/happiness-highest ranking holds in *both* groups — the signal is real, not
  a gender artifact, so z-scoring mainly buys generalization to a differently-gender-balanced dataset. It costs
  a little raw predictive power on *this* dataset (the raw feature partly free-rides on the emotion/gender
  co-occurrence pattern here).
- **Gender z-scoring `formant_roughness`**: the opposite story. Splitting by gender roughly **triples** the
  effect size for men (ε²=0.004→0.011) — but re-pooling the z-scored values doesn't recover that gain
  (ε²=0.003, *worse* than the raw pooled value), because men and women don't even agree on which emotion scores
  highest (angry for men, happiness for women). This isn't gender noise masking a universal signal; it's a
  gender-specific pattern with no single normalized version to recover.

### Phase 5 — Pretrained audio embeddings (wav2vec2 V/A/D)

To test a genuine ceiling above hand-designed features, `vad_features.py` runs
`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (a wav2vec2-large model fine-tuned on MSP-Podcast to
regress arousal/dominance/valence directly from audio) over all 33,964 files.

- Hit the same checkpoint bug this model is known for: the saved positional-conv weights are `weight_norm`
  parametrized (`weight_g`/`weight_v`), but this torch/transformers combination's `from_pretrained` silently
  leaves that layer randomly initialized instead of reconstructing it. Fixed by loading the raw state dict and
  rebuilding the true weight via `torch._weight_norm(weight_v, weight_g, dim=2)` before inference.
- **MPS was slower than plain CPU** on this hardware (a 2017 MacBook Pro, Radeon Pro 560, 4GB): 0.33 files/sec
  on MPS vs. 0.74 files/sec on 8 CPU threads, batch size 1 — kernel-dispatch overhead on this older discrete GPU
  outweighing the actual compute for a batch-of-one forward pass. Switched to CPU; full extraction took ~13
  hours.
- **Result**: `arousal` and `dominance` land at **#1 and #2** in the final 117-feature Random Forest's importance
  ranking (`valence` #7, `situation`; #2 and #1, `majority_vote`) — the single most valuable feature family
  added, and it doesn't hold up alone (Phase 10).

### Phase 6 — Four more acoustic axes not covered above

Run as a chained batch once the VAD extraction freed the CPU: **spectral dynamics**
(`spectral_dynamics_features.py`: centroid, rolloff, flatness, flux, zero-crossing rate — the "brightness" and
rate-of-timbral-change MFCC doesn't directly capture), **pause structure** (`pause_features.py`: count,
duration, total ratio, longest gap — hesitation/breath pattern, distinct from Phase 2's *voiced*-region
rhythm), **formant bandwidth** (`formant_bandwidth_features.py`: recovered essentially for free — the LPC step
in `harmony_features.py` already computes bandwidth per formant and had been discarding it after a threshold
filter), and **voice quality** (`voice_quality_features.py`: jitter/shimmer via a simplified waveform-peak
pitch-marking pass, HNR via per-frame normalized autocorrelation — not a Praat-equivalent glottal-closure
detector, values run high relative to clinical norms, but relative differences across emotion are still usable).

`spectral_flux_mean` (#3) and `hnr_mean` (#10) both land in the final model's top-15 feature importances;
formant bandwidth and jitter/shimmer contribute more modestly (ranks 20–116 of 117) but are not dead weight.

### Phase 7 — Final integrated model

`emotion_classifier.py`, 117 features across all phases above (Decision Tree for interpretable rules, Random
Forest for the performance ceiling), evaluated on both label sources:

| Label | Model | accuracy | macro F1 |
|---|---|---|---|
| `situation` | Decision Tree (depth 6) | 0.219 | 0.192 |
| `situation` | Random Forest (300 trees) | 0.409 | 0.324 |
| `majority_vote` | Decision Tree (depth 6) | 0.230 | 0.211 |
| `majority_vote` | **Random Forest (300 trees)** | **0.477** | **0.349** |

Macro F1 climbed at every stage feature families were added with real evidence behind them: 0.295 (harmony +
melody + tonality + rhythm, 79 features) → 0.306 (+ speaker/gender normalization, 86) → 0.324 (+ VAD + 4 more
acoustic axes, 117), `situation`; 0.320 → 0.325 → 0.349, `majority_vote`. `sadness`/`neutral`/`happiness` are
classified well (F1 0.42–0.61); `fear`/`disgust`/`surprise` remain weak (F1 0.13–0.24) — smaller classes that
also overlap acoustically with their neighbors.

### Phase 8 — Is this performance actually meaningful?

"Macro F1 above 0.5 is chance level" is a rule for **balanced binary** classification and doesn't apply here —
computed the real chance level for this specific 7-class, imbalanced problem via three dummy strategies before
trusting anything:

| strategy | accuracy | macro F1 |
|---|---|---|
| uniform random (1-of-7) | 0.144 | 0.135 |
| stratified random | 0.187 | 0.144 |
| always predict the majority class | 0.269 (looks fine) | **0.061** (isn't) |

Then two stronger checks (`significance_validation.py`), both on `majority_vote`:

- **5-fold CV**: accuracy 0.473±0.006, macro F1 0.339±0.006 — the single-split number above wasn't a lucky
  split.
- **Permutation test**, 200 label-shuffles, identical train/test procedure each time: real macro F1 = 0.342;
  null distribution mean = 0.133 (max across all 200 shuffles = 0.142); **p<0.005**, real performance sits
  **55.8 standard deviations** above the null mean. Not one of 200 random-label runs came within striking
  distance of the real model.

### Phase 9 — Ellipse/QDA regions in V-A space (Han & Cha 2017) — rejected

한의환·차형태 (2017), *"A Novel Method for Modeling Emotional Dimensions using Expansion of Russell's Model"*
(감성과학 20(1), 75–82), represents emotion categories as bivariate-Gaussian confidence ellipses in
Valence-Arousal space (rotation from the V-A correlation, semi-axes from the standard deviations, k=1σ) with a
Bayesian decision rule between them — mathematically identical to fitting `QuadraticDiscriminantAnalysis` per
class. `emotion_region_vad.py` re-derives the paper's ellipse parameters directly (rotation/semi-axis formulas,
not just the equivalent QDA fit) from Phase 5's wav2vec2 V-A(-D) values, then evaluates classification with QDA
under the same train/test protocol used throughout.

**Total failure.** All 7 emotions' 1σ ellipses overlap almost completely — only `sadness` sits perceptibly
apart — and macro F1 (0.126–0.133 across the 2D/3D, `situation`/`majority_vote` combinations) comes in **below
even the dummy chance baseline** from Phase 8. The paper's own 92.86% figure was measured on human survey
*averages* for individual words (ANEW) — data that's already well-separated because people consciously rated it
that way. Real speech, run through a general-purpose wav2vec2 regressor, produces V-A values with far more
noise and far less between-class separation than self-reported word ratings, and valence in particular clusters
into a narrow 0.39–0.46 band regardless of emotion — consistent with the broader finding that valence is
difficult to read from acoustics alone. This is also a second, independent confirmation of Phase 2's "strong
standalone feature ≠ strong standalone classifier" lesson: arousal/dominance/valence are top-3 features in the
full 117-feature model (Phase 5) and yet collapse when run through this literature method with nothing else.

### Phase 10 — Attempting speaker separation via MFCC clustering — rejected

Revisits the asterisk named in the Methodology section: could unsupervised clustering on MFCC recover a
speaker grouping good enough for a speaker-independent split or per-speaker normalization?

Checked whether the metadata offered a shortcut first: utterance text is essentially unique per row (28,512
distinct scripts across 33,980 rows, mean repetition 1.19× — this is not "one speaker records the same line
once per emotion," it's many different speakers each recording a shared pool of template sentences), and
`age`/`gender` buckets are shared by hundreds to thousands of rows each (age 46 alone: 7,169 rows) — neither is
a usable speaker proxy.

Tested MFCC(13)+pitch-based `KMeans` on the easiest possible sub-case: recovering **gender** (k=2), a variable
with known ground truth and the largest verified effect size found anywhere in this project (ε²=0.43 on
`pitch_mean`). **It failed outright** — cluster membership tracked gender no better than chance (one cluster
88% female, the other a near-population-average 60/40 mix); repeating with *only* the three most
gender-discriminative pitch features still failed (one cluster near the population ratio, the other a small,
imperfect majority-female group). MFCC/pitch statistics conflate speaker identity with phonetic content
(28,512 distinct scripts) and emotional prosody, and unsupervised clustering optimizes for whichever source of
variance is largest — not necessarily the one being looked for. Proper speaker separation would need
purpose-built speaker-verification embeddings (ECAPA-TDNN, resemblyzer) trained to suppress exactly the content
and emotion variance that sank this attempt — and even then, there would be no ground-truth speaker ID to
validate the result against. Stopped here rather than build that pipeline on an unverifiable foundation.

### Lessons that generalized across the whole exploration

1. **Effect size, not p-value, is the real answer at this sample size.** Every feature tested came back
   p<0.001 somewhere; effect sizes ranged from negligible (ε²=0.0006, tonality) to medium (ε²=0.051, note
   duration) and only the effect size distinguishes a real discovery from statistical noise on a 34k-row corpus.
2. **The label you train on changes the answer.** `majority_vote` (what people actually heard) consistently
   showed larger effect sizes and higher classifier performance than `situation` (what the line was scripted to
   convey) — acted intent and perceived emotion correlate but aren't the same target, and reporting only one
   would have overstated or understated every result depending on which.
3. **A literal music-theory transplant was the weakest link.** Major/minor key-fit (Phase 3) assumes a
   discretized, quantized pitch system that continuous speech F0 doesn't have — every other harmony/melody/
   rhythm translation (dissonance, semitone intervals, note duration) worked to some degree; the literal
   scale-degree framework didn't work at all.
4. **A feature's standalone strength and its multivariate value are different questions, twice over.**
   `note_dur_mean` had the single largest effect size in the project (ε²=0.051) but ranks 54th of 117 in the
   Random Forest (Phase 2); arousal/dominance/valence rank #1–2 in that same model but collapse to worse-than-
   chance when run through the ellipse/QDA method alone (Phase 9) — a good ingredient still needs the rest of
   the recipe, in both directions.
5. **Unsupervised clustering doesn't rescue a missing label.** Gender has the largest verified effect size in
   the project (ε²=0.43) and is *still* not recoverable via naive MFCC/pitch clustering (Phase 10) — a variable
   being statistically detectable in aggregate says nothing about whether it's recoverable point-by-point.
6. **A confound-correction that helps one feature can be neutral or actively wrong for a sibling feature.**
   Gender z-scoring fixed a real problem for formant ratios and cost a little raw accuracy but improved
   generalization for pitch — while for `formant_roughness` it revealed the emotion ranking itself differs by
   gender, so no single normalized version was ever going to recover a universal signal that doesn't exist
   (Phase 4).
7. **"F1=0.5 is chance" is a rule for balanced binary classification, not a universal threshold.** The actual
   chance level for this 7-class, imbalanced problem is macro F1 0.06–0.14 depending on strategy — a naive
   "always predict the majority class" baseline reaches 27% *accuracy* while scoring 0.061 macro F1, exactly the
   failure mode that makes accuracy alone misleading here.
8. **Clearing one significance bar isn't enough when you can clear three.** Dummy-baseline comparison, 5-fold
   CV, and a 200-shuffle permutation test all had to agree before treating ~0.32–0.35 macro F1 as a genuine,
   non-random signal (55.8 SD above the permutation null) rather than a single flattering train/test split.
9. **A numpy/torch ABI mismatch is worth fixing at the root once, not routing around per-model.** Downgrading
   `numpy` to a `torch==2.2.2`-compatible 1.x build in the shared venv — and re-verifying the existing
   librosa/scipy pipeline still worked — fixed the wav2vec2 integration outright, instead of maintaining a
   second venv for the one model that needed it.
10. **Long unattended background jobs need their own babysitting, separate from the model.** A ~13-hour wav2vec2
    extraction pass surfaced an unrelated risk: this machine's system-sleep timer was set to 1 minute, which
    would have silently paused the entire pipeline overnight — caught and fixed with `caffeinate` before it
    mattered, not after.

## Repository layout

```
├── similarity.py, similarity_vectorized.py       # waveform-similarity exploration; load_groups()/get_audio_path()
│                                                  # reused by every extraction script below
├── extract_features.py                           # MFCC / pitch / energy baseline
├── harmony_features.py                           # Phase 1 — formant roughness, melodic dissonance, tonal stability
├── partial_roughness.py                          # Phase 1 — FFT-partial dissonance (voice texture)
├── melodic_profile.py                            # Phase 2 — semitone interval profile
├── melody_rhythm_features.py                     # Phase 2 — note segmentation, duration, IOI
├── tonality_features.py                          # Phase 3 — Krumhansl-Schmuckler major/minor key-fit (rejected)
├── dissonance_tonality_stats.py                  # Kruskal-Wallis + epsilon-squared, situation & majority_vote
├── speaker_norm_features.py                      # Phase 4 — formant ratios, gender z-scoring
├── vad_features.py                               # Phase 5 — wav2vec2 arousal/dominance/valence
├── spectral_dynamics_features.py                 # Phase 6 — centroid/rolloff/flatness/flux/ZCR
├── pause_features.py                             # Phase 6 — silence/pause statistics
├── formant_bandwidth_features.py                 # Phase 6 — formant bandwidth (reuses harmony_features.py's LPC)
├── voice_quality_features.py                     # Phase 6 — jitter/shimmer/HNR
├── emotion_classifier.py                         # Phase 7 — combines every feature set, Decision Tree + Random Forest
├── significance_validation.py                    # Phase 8 — 5-fold CV + 200-shuffle permutation test
├── emotion_region_vad.py                         # Phase 9 — Han & Cha (2017) ellipse/QDA regions (rejected)
├── text_features.py                              # TF-IDF+SVD on transcript text; corpus-specific, unused in the final model
├── output/
│   ├── features/, harmony/, melodic_profile/, partial_roughness/, tonality/,
│   │   melody_rhythm/, speaker_norm/, vad/, pause/, spectral_dynamics/,
│   │   formant_bandwidth/, voice_quality/, text_features/
│   │       ├── per_file_*.csv          # one row per utterance
│   │       └── summary_by_emotion.csv  # per-emotion mean/std
│   ├── vad_region/                     # Phase 9 ellipse plots (PNG)
│   └── model/
│       ├── decision_tree*.joblib, feature_names*.json
│       └── random_forest*.joblib       # Git LFS
```

`output/similarity/` (Phase-0-equivalent waveform-comparison output, up to 2.18GB per file) and the raw
audio/metadata are not included — see `.gitignore`.
