# SP-V2-012 Overnight Summary

Read this first. Full technical detail lives in `signal_processing_v2_SP-V2-012.md` (updated
throughout the night — every table below has a corresponding section there with the full method).
This summary is written to be read cold, first thing in the morning.

## 1. What completed — everything originally planned for tonight finished

- Three independent nested-seed extractions (seeds 43/44/45, 5,600 files each, 0 errors) — the
  full sample-scaling dataset (N∈{1,400, 2,800, 5,600} × 3 seeds, plus the original seed-42 pilot
  as a 4th independent point at N=1,400).
- Full RF-only M0–M7 ablation grid + Gaussian/permuted negative controls at **all 18** planned
  (N, seed, label) points.
- **All 4 planned classifier-comparison endpoints** (RF/Elastic-Net/Linear SVM): N=1,400/seed 42
  and N=5,600/seed 43, both labels. The N=5,600 points took far longer than expected (~2 hours each
  vs. ~25–30 min at N=1,400 — Elastic-Net's `saga` solver scales poorly with N) but all four
  completed before this summary was finalized.
- A new, targeted `deep_verification.csv` extraction (1,400 files) built specifically to test the
  0.54 Hz artifact question, common-envelope removal, and tracker-quality confounds — none of these
  were answerable from previously-saved columns, so this required fresh instrumentation, not just
  re-analysis.
- H1-specific re-verification: per-harmonic (H1–H5) and cumulative (H1, H1+H2, ..., all-5)
  dimension-matched classifier comparisons.
- F008/F009/F010 complementarity, redundancy matrix, cross-label consistency (completed earlier in
  the day, unchanged).

## 2. What failed / did not complete

Nothing from the originally-scoped Tier-1 plan failed outright. Everything took longer than
estimated (the N=5,600 classifier-comparison endpoints in particular, ~2 hours each vs. an
estimated 25–30 minutes), but all completed. The one genuine surprise from the *last* completed
piece (N=5,600/seed 43 `majority_vote` classifier comparison): the RF/linear-classifier split for
`majority_vote` **reversed a second time** relative to N=1,400 (at N=1,400 RF was negative and SVM
strongly positive; at N=5,600 RF turned positive and both linear models turned negative). The
overall, more cautious conclusion this forces — classifier sensitivity for these features is itself
noisy and seed/N-dependent, not a stable property of any one classifier — replaces an earlier,
cleaner-sounding "RF-specific artifact" framing that turned out to be premature after only 2 of the
4 endpoints were in.

Tier 2/3 items explicitly not attempted, by design, given time: nonlinear/spline (GAM-style) HMC
analysis (§8 below), bootstrap confidence intervals, Elastic-Net/SVM comparison at every N/seed
(only the 4 planned endpoints were run), full per-harmonic (H2–H4) track-quality measurement (only
H1/H5 were instrumented), duration-matched analysis extended beyond `dominant_freq`, and
modulation-rate band analysis (deferred entirely, correctly, since §5 found the 0.54 Hz figure
itself questionable).
- **Tier 2/3 items explicitly not attempted, by design, given time**: nonlinear/spline (GAM-style)
  HMC analysis (§9 of the brief), bootstrap confidence intervals (§17), Elastic-Net/SVM classifier
  comparison at every N/seed (only run at the 2 completed endpoints), full per-harmonic (H2–H4)
  track-quality/reliability measurement (only H1/H5 were instrumented in `deep_verification.csv`),
  duration-matched analysis extended to the entropy features specifically (only run for
  `dominant_freq`), modulation-rate band analysis using literature-justified boundaries (deferred
  entirely, correctly, since §5 below found the 0.54 Hz figure itself questionable — analyzing bands
  built on a number that might be an artifact would compound the problem, not fix it).

## 3. Scaling results (RQ12-1 / RQ12-2)

Full tables in the main doc. Headline: **the single-seed N=1,400 result for F010-alone
(+0.0136, `situation`) was substantially a favorable-seed artifact.** Averaged across 4 independent
seeds at the same N, the true mean effect is +0.0028 — an order of magnitude smaller — and it does
not grow cleanly with N (N=2,800: −0.0060; N=5,600: +0.0010). This is closer to **Pattern B**
(flat/noisy) than Pattern A (recovers with N) for F010 alone on `situation`.

The one clear exception, and the strongest positive finding of the whole night: **M5 (F008+F010)
on `majority_vote` is positive and remarkably stable across all three N levels** (+0.0059 / +0.0076
/ +0.0072) — the single most-replicated result in this analysis, not a one-seed fluke.

Negative-control scaling (12 of 12 comparisons across N/probe/label) consistently shows real
features beating both Gaussian-noise and permuted controls, even where the real-vs-baseline delta
itself is close to zero — i.e., **the dimensionality penalty is real and reproducible, and F010
reliably avoids acting like noise even when it isn't clearly beating baseline either.**

## 4. HMC replication

`inter_harmonic_modulation_coherence`'s raw emotion effect replicated cleanly across an independent
700-file subset during the day (ε²=0.047 vs. 0.050) and was the consistent top-1 training-fold
selection in every fold tested. **This replication is not in question. What is in question, per §5
below, is whether raw HMC means what it was initially interpreted to mean.**

## 5. 0.54 Hz artifact verdict: **INCONCLUSIVE, artifact-leaning**

- Moderate correlation with 1/duration (ρ=0.23–0.26) — a real but partial relationship.
- 38–53% of files have their dominant peak at the first 1–2 FFT bins — concerning, not universal.
- Zero-padding sensitivity is small (peak location stable under 8× zero-padding) — evidence *for* a
  genuine local spectral feature, not pure discretization.
- FFT-peak vs. Welch PSD agree only moderately in rank (ρ=0.35–0.43) though their central tendency
  matches (median ratio ≈1.0–1.09).
- **Net verdict: the original "phrase-scale, ~0.54 Hz" framing is not established and should not be
  repeated as settled fact.** Evidence points in both directions; this section resolves the question
  as genuinely open, not as confirmed.

## 6. Global-envelope artifact verdict: **CONFIRMED — this is the central finding of the night**

Removing the shared/common amplitude envelope from the per-harmonic trajectories before computing
inter-harmonic coherence **collapses the signal almost entirely**: envelope-removed HMC is
uncorrelated with raw HMC (ρ=−0.026), and its emotion association drops from ε²=0.0501 (medium,
p=2.6×10⁻¹⁴) to **ε²=0.0030 (negligible, p=0.118 — not significant)**. Its gender association drops
from ε²=0.287 (very large) to 0.004. **Raw HMC's apparent signal is substantially a measure of
shared amplitude-envelope dynamics (overall loudness contour), not independent harmonic-specific
coordination.** This directly overturns the daytime conclusion that HMC is F010's standout,
harmonic-specific feature — see §12/§13 below for what survives this correction.

## 7. Tracker-quality artifact verdict: **PARTIAL CONFOUND, secondary to §6**

HMC correlates moderately with two track-quality proxies (ρ≈0.245–0.246, both p≪0.001).
Controlling for them plus gender reduces HMC's emotion ε² from 0.0501 to 0.0209 — retains ~42% of
its raw effect, a real but partial confound. Much less severe than the common-envelope collapse —
tracker quality is a secondary, not primary, explanation for HMC's apparent signal.

## 8. Nonlinearity verdict: **NOT TESTED — time constraint**

The spline/GAM-style nonlinear-response analysis (§9 of the brief) to test whether RF's advantage
over Elastic-Net/SVM at N=1,400 reflects a genuinely nonlinear HMC-emotion relationship was not
run. Given §6's finding that raw HMC is substantially envelope-confounded, this question is now
lower-priority than it seemed during the day — worth revisiting only after deciding whether raw HMC
is worth further investment at all.

## 9. H1 vs. generic envelope: **CONFIRMED, and this claim survives the night's confound checks**

H1-only (6 dims) beats a dimension-matched generic-whole-envelope-modulation control by 7.5× (Δ
macro F1 +0.0276 vs. +0.0037) — re-verified overnight with the same result. Per-harmonic
comparison: H1 best (+0.0276), H5 second (+0.0143), H2/H3/H4 weaker and non-monotonic — **evidence
against** the "it's just tracking reliability" alternative (H5, presumably least reliable, beats
three middle harmonics). Cumulative addition of more harmonics on top of H1 never recovers H1's
solo performance (Pattern C, dilution). **This is now the single best-supported "harmonic-specific"
claim in the family** — importantly, a *different quantity* (H1's own modulation-spectrum shape)
than HMC (cross-harmonic coherence, §6), which is why one survived the night's scrutiny and the
other did not.

## 10. F008+F010 majority_vote replication: **CONFIRMED — most robust finding of the analysis**

+0.0301 marginal contribution (single seed, daytime) → **+0.0059 / +0.0076 / +0.0072 mean across
2–4 seeds at N=1,400/2,800/5,600** (§3 above). Smaller in magnitude than the original single-seed
number, but consistent in sign and magnitude across every seed and every N level tested — the most
reproducible positive result in this entire investigation.

## 11. Sparse-feature stability: **CONFIRMED**

`inter_harmonic_modulation_coherence` was the training-fold top-1 selection in 5/5 folds during the
day, and H1-alone beating all-33-features replicated overnight (§9 above). The *sparsity* finding
(one or a few features carry the signal, more features dilute it) is robust; what changed overnight
is *which* feature should be trusted as meaningful (H1's shape, not HMC).

## 11b. Classifier-choice sensitivity: all 4 planned endpoints complete — unstable, not a fixed rule

RF/Elastic-Net/Linear SVM were compared at both N=1,400 (seed 42) and N=5,600 (seed 43), both
labels. Result: the classifier that "wins" for a given feature set/label pairing **is not stable
across N or seed**. At N=1,400, RF was uniquely positive for F010-alone on `situation`, and SVM was
uniquely (5/5-fold) positive for F010-alone on `majority_vote`. At N=5,600, `situation`'s RF
advantage largely disappeared (all three classifiers converged to modest positive values), while
`majority_vote`'s pattern flipped outright (RF turned positive, both linear models turned
negative). **No single classifier reliably "unlocks" these features' value** — this replaces an
earlier, more quotable but premature framing (based on only 2 of the 4 endpoints) that RF was
uniquely well-suited to `situation` and SVM to `majority_vote`. The instability itself is the
finding.

## 12. Final current interpretation

Two candidate statements from the brief's own §29, evaluated against tonight's evidence:

> "Inter-harmonic modulation coherence captures emotion-related temporal organization of vocal
> harmonics that cannot be fully explained by utterance duration, conventional F0/energy dynamics,
> or generic amplitude-envelope modulation."

**NOT SUPPORTED as originally stated.** HMC's signal is substantially explained by generic
amplitude-envelope dynamics specifically (§6) — the one confound category the hypothesis explicitly
claimed to survive. The daytime evidence for this claim (replication, top-*k* selection, cross-label
consistency) was real and is not retracted, but the *interpretation* was wrong: those checks never
tested envelope-relativity, and the one that did (overnight) failed decisively.

> "Aperiodicity/MVF may provide complementary information to harmonic temporal coordination,
> particularly for perceived emotion labels."

**SUPPORTED**, more strongly than during the day. F008+F010's `majority_vote` benefit replicated
across seeds and N levels (§10) — the most solid finding of the whole investigation.

A third statement, not in the original brief but earned by tonight's work, is now the most
defensible F010-related claim overall:

> H1's individual modulation-spectrum shape (not the cross-harmonic coherence measure) carries
> emotion-relevant information beyond both a generic whole-envelope control and beyond what
> additional harmonics add — and this specific result survived every confound check applied to it.

## 13. KEEP / CONDITIONAL / REJECT

| Candidate | Status | Basis |
|---|---|---|
| F008 (aperiodicity/MVF) alone | CONDITIONAL | Best standalone confound/redundancy profile in the earlier pilot; RF-only scaling shows no clear net benefit alone, needs the full-corpus check already flagged as outstanding |
| F009 (harmonic phase) alone | CONDITIONAL, leaning REJECT | Weak and inconsistent across N in tonight's scaling curve (situation: +0.0025→−0.0037→−0.0081, trending worse with N) |
| F010, all 33 features | REJECT (as a block) | Diluted by co-features; negative-control scaling shows it barely edges out baseline once averaged across seeds |
| `inter_harmonic_modulation_coherence` alone | **REJECT — downgraded from the daytime's provisional KEEP** | Collapses under common-envelope removal (§6); likely a loudness-contour proxy, not a novel harmonic-organization signal |
| H1-specific modulation (6-dim, non-HMC) | **CONDITIONAL, upgraded — the strongest surviving F010-derived candidate** | Beats generic envelope 7.5×, survives the tracking-reliability alternative explanation; not yet tested against the common-envelope confound specifically (open item) |
| F008 + F010 combination | **CONDITIONAL, upgraded — most replicated positive finding overall** | Consistent positive effect on `majority_vote` across 2–4 seeds and all 3 N levels; not yet tested with H1-specific-F010 in place of all-33-F010 |

None reach an unqualified KEEP tonight — every candidate has at least one open verification step
remaining (see §15).

## 14. SP-V2-011 decision: **HOLD, unchanged**

No finding tonight points at harmonic-tracker imprecision as a limiting factor. The feature that
survived scrutiny best (H1-specific modulation) is a single-harmonic measure less exposed to
multi-harmonic tracking-continuity issues in the first place, and the feature that failed (HMC) failed
because of a *representation* confound (shared envelope), not a *tracking-precision* one. Building a
more precise adaptive harmonic tracker would not address either finding. Recommendation stands:
**do not start SP-V2-011.**

## 15. Highest-priority next experiments

1. **Re-test H1-specific modulation against the common-envelope confound specifically** — §6 only
   tested HMC (a cross-harmonic quantity); H1's own modulation shape has not yet been checked
   against a comparably rigorous confound (does H1's entropy/rate signature also collapse if
   computed on envelope-relative rather than absolute amplitude?). This is the most important open
   question given H1 is now the leading candidate.
2. Finish the pending N=5,600/seed 43 classifier-comparison point (already running; check for
   completion before re-launching).
3. Test F008 + H1-specific-modulation (rather than F008 + all-33-F010) as the combination — given
   H1 alone outperforms all-33-F010 and F008+F010 is the most replicated positive result, their
   combination with H1 specifically (not the diluted 33-feature F010) is untested and plausibly
   stronger than either component result reported tonight.
4. Nonlinear/spline HMC analysis (§9 of the brief) — lower priority now given §6, but still open if
   HMC is revisited later.
5. Bootstrap CIs on the handful of headline numbers (§17 of the brief) — not done, time-limited.

## Abstract-ready quantitative findings (2026-09-22 제출 고려)

Only results actually verified tonight or during the day are listed. Each includes its replication
status so nothing here is presented as more certain than it is.

**Finding 1**
- Exact result: `inter_harmonic_modulation_coherence`'s apparent emotion association (ε²=0.050,
  medium) collapses to ε²=0.003 (not significant) when the shared amplitude envelope is removed
  before computing cross-harmonic coherence.
- Evaluation protocol: Kruskal-Wallis + rank epsilon-squared, `situation` label, N=1,400,
  common-envelope defined as the geometric mean of 5 per-harmonic amplitude trajectories.
- Replication status: single-sample (N=1,400), not yet replicated on an independent subset —
  unlike the *raw* HMC finding, which did replicate on a disjoint 700-file sample.
- Main limitation: envelope-removed HMC itself has not been separately validated as measuring a
  coherent physical quantity beyond "not the envelope" — it may simply be noise once the dominant
  shared component is removed. Not yet distinguished from that possibility.

**Finding 2**
- Exact result: a dimension-matched (6 features each) comparison shows H1's own modulation-spectrum
  features outperforming a generic whole-envelope-modulation control by 7.5× (Δ macro F1 +0.0276
  vs. +0.0037, `situation`, N=1,400, Random Forest, 5-fold CV).
- Evaluation protocol: paired StratifiedKFold, identical folds for both conditions, baseline =
  existing 117-feature set.
- Replication status: re-verified overnight with an independent code path re-run (same result to
  4 decimal places, as expected from a deterministic pipeline on the same data) — not yet tested at
  larger N or on a different seed's sample.
- Main limitation: not yet checked against the same common-envelope confound that sank HMC (§15,
  priority 1 above) — this result should be treated as provisional until that check is done.

**Finding 3**
- Exact result: F008 (aperiodicity/MVF) added on top of F010 shows a consistent positive marginal
  effect on `majority_vote` classification: mean Δ macro F1 across seeds = +0.0059 (N=1,400,
  4 seeds), +0.0076 (N=2,800, 3 seeds), +0.0072 (N=5,600, 3 seeds).
- Evaluation protocol: Random Forest, paired 5-fold StratifiedKFold, mean taken across independently
  seeded nested samples at each N.
- Replication status: **the most replicated finding in this document** — consistent sign and
  magnitude across 2–4 independent seeds at 3 different sample sizes.
- Main limitation: absolute effect size is modest (macro F1 +0.006 to +0.008); mechanism for *why*
  F008 and F010 are complementary specifically for `majority_vote` and not `situation` is not
  established, only observed.
