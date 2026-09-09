**FULLY COMPLETE (2026-09-10, ~04:00 KST).** All originally-planned overnight work finished: all 3
nested seeds extracted, the full RF-only scaling curve (18/18 points), all 4 classifier-comparison
endpoints, and all §6–8 artifact-verification checks. **This checkpoint file is no longer needed
for resuming anything** — read `SP-V2-012_overnight_summary.md` instead (written to be read cold)
or `signal_processing_v2_SP-V2-012.md` for full technical detail. Kept below for the historical
record of the mid-session interruption only; caffeinate was left running for the user to stop by
hand in the morning (`ps aux | grep caffeinate`).

---

# SP-V2-012 — Checkpoint (paused 2026-09-09 17:37 KST, resumed 22:45 KST)

**Resume update (22:45)**: the two jobs below were confirmed dead when the session was checked
after the pause (no process, no output file — killed along with the previous Claude Code process,
not merely paused by minimizing). Re-launched fresh at 22:45: seed-44 extraction (task
`b0ufltvdv`) and the N=1,400/seed 42/`majority_vote` classifier-comparison point (task
`bncsu0gh4`), running in parallel this time (user confirmed staying online, so parallel wall-clock
savings outweigh the contention slowdown observed earlier). If this session also gets interrupted,
check these two task outputs / the corresponding files
(`sp012_seed44_n5600.csv`, `point_N1400_seed42_majority_vote.json`) before re-launching again.

Work was paused here at the user's request. This file is the resume point — read it first before
continuing SP-V2-012. It supplements, not replaces, `signal_processing_v2_SP-V2-012.md` (the
actual findings write-up, already substantially complete for everything finished so far).

## What's safely done (on disk, in `output/signal_v2/SP-V2-012/`)

- `mechanistic_analysis.json`, `complementarity.json`, `f010_top_k_efficiency.json`,
  `coherence_probe.csv`, `aux_features_original_1400.csv` — all §11–21 mechanistic/complementarity
  analysis, complete, already written up in the main doc.
- `sp012_seed43_n5600.csv` — seed 43's full nested extraction (5,600 files, 0 errors). Covers
  N=1400/2800/5600 for seed 43 via prefix slicing (`load_level_dataset`).
- `point_N1400_seed42_situation.json` — full M0–M7 grid + negative controls + 3-classifier
  comparison for N=1,400/seed 42/`situation`. **Key result already captured**: RF shows a clear
  positive F010 effect (+0.0136) that ElasticNet (−0.0004) and SVM (−0.0039) do not reproduce —
  first evidence of a possible classifier-specific component. Negative controls (Gaussian/permuted)
  both underperform baseline, confirming a real dimensionality penalty exists that F010's real
  signal overcomes.

## What was mid-flight when paused (NOT saved — must be re-run)

Two background jobs were running and had **not yet produced output** when paused:

1. **Seed-44 nested extraction** (task id `bd78de4z5`, command below) — was ~56+ min into an
   estimated ~70 min run. If the process was killed by closing the terminal/machine, this needs to
   be re-run from scratch (no partial-save capability in the current extraction script).
2. **N=1,400/seed 42/`majority_vote` classifier-comparison point** (task id `bgtjme1d9`) — was a
   few minutes in. Also needs a clean re-run if killed.

**Check first whether either survived** (they may still be running if the machine never actually
shut down):
```bash
ls -la "output/signal_v2/SP-V2-012/sp012_seed44_n5600.csv"        # exists? seed 44 finished
ls -la "output/signal_v2/SP-V2-012/point_N1400_seed42_majority_vote.json"  # exists? that point finished
ps aux | grep -E "extract_sp012_features|sp012_scaling_point" | grep -v grep   # still running?
```

If neither file exists and no matching process is running, re-launch with:
```bash
source .venv/bin/activate
cd "감정 분류를 위한 대화 음성 데이터셋"   # repo root

# seed 44 extraction (~70 min alone, no other background jobs running at the same time)
python3 -m signal_v2.evaluation.extract_sp012_features --n-per-emotion 800 --seed 44 --out-name sp012_seed44_n5600.csv

# N=1400 majority_vote classifier comparison (~25-30 min, run after or alongside seed 44 —
# stacking roughly doubled seed 43's wall time earlier this session, so prefer sequential if time allows)
python3 -m signal_v2.evaluation.sp012_scaling_point --csv-path output/signal_v2/real_corpus/real_corpus_features.csv --seed 42 --n-per-emotion 200 --label majority_vote --out-name point_N1400_seed42_majority_vote.json
```

## Remaining steps, in order (per the original plan)

1. Seed 44 extraction (above, if not already done).
2. Seed 45 extraction: `python3 -m signal_v2.evaluation.extract_sp012_features --n-per-emotion 800 --seed 45 --out-name sp012_seed45_n5600.csv`
3. N=5,600/seed 43 classifier-comparison, both labels (seed 43's CSV already exists):
   ```bash
   python3 -m signal_v2.evaluation.sp012_scaling_point --csv-path output/signal_v2/SP-V2-012/sp012_seed43_n5600.csv --seed 43 --n-per-emotion 800 --label situation --out-name point_N5600_seed43_situation.json
   python3 -m signal_v2.evaluation.sp012_scaling_point --csv-path output/signal_v2/SP-V2-012/sp012_seed43_n5600.csv --seed 43 --n-per-emotion 800 --label majority_vote --out-name point_N5600_seed43_majority_vote.json
   ```
4. Once all 3 seeds' CSVs exist, RF-only (cheap, `run_classifier_comparison=False`) sweep for the
   remaining (N, seed, label) points not yet covered by steps above — i.e. seed 43/44/45 ×
   N∈{1400,2800} + seed 44/45 × N=5600, × both labels. `sp012_run_scaling_curve.py` was written for
   this but its `run_classifier_comparison` gating currently defaults to "endpoints only" rather
   than the specific 2-point plan actually used (seed 42 N=1400 + seed 43 N=5600) — check/adjust
   before running it wholesale, or just call `run_one_point(..., run_classifier_comparison=False)`
   directly per point as the wakeup prompts were doing.
5. Compile every `point_*.json` into the scaling-curve table, determine Pattern A (improves with N)
   vs. B (stays negative) vs. C (sparse subset — already the leading hypothesis from the
   feature-efficiency/mechanistic sections) per the brief's own diagnostic framework.
6. Finish `signal_processing_v2_SP-V2-012.md`'s pending section ("Random-Feature Negative
   Controls, Classifier Comparison, and the Sample-Scaling Curve" — currently a placeholder) plus
   Accepted/Rejected Interpretations, Implications for SP-V2-011, and Final Feature Decisions
   (KEEP/CONDITIONAL/REJECT for F008/F009/F010, with the specific recommendation — already well
   evidenced — to prefer `inter_harmonic_modulation_coherence` alone over all 33 F010 features).
7. Update `signal_processing_v2_log.md`'s SP-V2-012 entry from "in progress" to final.

## What to tell a future session to resume

Something like: *"Resume SP-V2-012 from
docs/signal_processing_v2_SP-V2-012_CHECKPOINT.md — check whether the two in-flight background
jobs finished or need re-running, then continue the remaining steps listed there."*

## Compute-time reality check, for planning the next session

Each nested-seed extraction (5,600 files) has taken 55–90 minutes depending on contention with
other concurrent jobs; each classifier-comparison point (RF+ElasticNet+SVM with inner-CV tuning)
takes ~25–30 minutes. Remaining work is roughly: 1 more extraction (seed 45, ~70 min) + 3 more
classifier-comparison points (~75–90 min) + a handful of cheap RF-only points (~10 min total) +
write-up (fast). **Realistically another 2–3 hours of mostly-background compute** to fully close
out SP-V2-012 as scoped. If that's more than is wanted, the already-completed mechanistic/
feature-efficiency evidence (Pattern C leaning, RF-specificity at N=1,400) is already strong enough
to draft provisional Final Feature Decisions without waiting for the full 3-seed curve — worth
considering as a scope cut if time is tight next time, rather than defaulting to running everything
originally planned.
