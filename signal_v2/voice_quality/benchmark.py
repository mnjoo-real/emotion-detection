"""SP-V2-006 driver: conditions A-E에서 conventional vs residual(glide-aware)
jitter/shimmer, 기존 저장소 구현(voice_quality_features.py), Praat(parselmouth)을
모두 비교한다.

좋은(glide-aware) 지표라면: B/D(매끄러운 glide/vibrato, 진짜 jitter 없음)에서는
낮고, A/C/E(진짜 jitter 있음)에서는 높아야 한다 - glide/vibrato 자체가 있는지
없는지와 무관하게 "진짜 jitter의 유무"만 반영해야 한다는 뜻.
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.f0.estimators import estimate_pyin, estimate_world_harvest
from signal_v2.voice_quality.glide_aware_jitter import (
    conventional_jitter_shimmer, residual_jitter_shimmer, praat_jitter_shimmer,
)
from signal_v2.voice_quality.synthetic_jitter_conditions import all_conditions
from voice_quality_features import compute_voice_quality_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "glide_jitter")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

FIELDNAMES = [
    "condition", "true_jitter_std", "true_shimmer_std",
    "conventional_jitter_gtmarks", "conventional_shimmer_gtmarks",
    "residual_jitter_oracle", "residual_shimmer_oracle",
    "residual_jitter_pyin", "residual_shimmer_pyin", "pyin_n_voiced_frames",
    "residual_jitter_world_harvest", "residual_shimmer_world_harvest",
    "residual_jitter_fallback", "residual_shimmer_fallback",
    "existing_repo_jitter", "existing_repo_shimmer",
    "praat_jitter", "praat_shimmer",
]


def evaluate_condition(cond) -> dict:
    marks = cond.cycle_start_samples
    mark_times = marks[:-1] / cond.sr

    conv = conventional_jitter_shimmer(marks, cond.waveform, cond.sr)

    # oracle: 실제 "의도된" F0 궤적 자체를 기대 주기의 근거로 사용(상한 참고용)
    oracle = residual_jitter_shimmer(
        marks, cond.waveform, cond.sr,
        f0_smooth_times=mark_times, f0_smooth_values=cond.true_f0_intended[:-1],
    )

    # 실전 조건: pyin으로 추정한 F0 궤적을 기대 주기의 근거로 사용
    pyin_result = estimate_pyin(cond.waveform, cond.sr)
    pyin_n_voiced = int(np.sum(~np.isnan(pyin_result.f0)))
    realistic_pyin = residual_jitter_shimmer(
        marks, cond.waveform, cond.sr,
        f0_smooth_times=pyin_result.times, f0_smooth_values=pyin_result.f0,
    )

    # SP-V2-001에서 WORLD Harvest가 pyin보다 매끄러운 glide/vibrato를 더 잘 따라간다는
    # 결과를 이어받아, 대안적인 "실전" 기대-주기 근거로 함께 비교한다.
    world_result = estimate_world_harvest(cond.waveform, cond.sr)
    realistic_world = residual_jitter_shimmer(
        marks, cond.waveform, cond.sr,
        f0_smooth_times=world_result.times, f0_smooth_values=world_result.f0,
    )

    # fallback: 별도 F0 추정 없이 관측 주기 자체의 이동평균만 사용
    fallback = residual_jitter_shimmer(marks, cond.waveform, cond.sr)

    existing = compute_voice_quality_features(cond.waveform, cond.sr)

    try:
        praat = praat_jitter_shimmer(cond.waveform, cond.sr)
    except Exception as exc:  # noqa: BLE001
        praat = {"praat_jitter": np.nan, "praat_shimmer": np.nan, "error": str(exc)}

    return {
        "condition": cond.name,
        "true_jitter_std": cond.true_jitter_std, "true_shimmer_std": cond.true_shimmer_std,
        "conventional_jitter_gtmarks": conv["jitter"], "conventional_shimmer_gtmarks": conv["shimmer"],
        "residual_jitter_oracle": oracle["residual_jitter"], "residual_shimmer_oracle": oracle["residual_shimmer"],
        "residual_jitter_pyin": realistic_pyin["residual_jitter"], "residual_shimmer_pyin": realistic_pyin["residual_shimmer"],
        "pyin_n_voiced_frames": pyin_n_voiced,
        "residual_jitter_world_harvest": realistic_world["residual_jitter"],
        "residual_shimmer_world_harvest": realistic_world["residual_shimmer"],
        "residual_jitter_fallback": fallback["residual_jitter"], "residual_shimmer_fallback": fallback["residual_shimmer"],
        "existing_repo_jitter": existing["jitter_local"], "existing_repo_shimmer": existing["shimmer_local"],
        "praat_jitter": praat.get("praat_jitter", np.nan), "praat_shimmer": praat.get("praat_shimmer", np.nan),
    }


def plot_summary(rows: list) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    conditions = [r["condition"] for r in rows]
    metrics_jitter = ["conventional_jitter_gtmarks", "residual_jitter_oracle", "residual_jitter_pyin",
                       "residual_jitter_world_harvest", "residual_jitter_fallback",
                       "existing_repo_jitter", "praat_jitter"]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(conditions))
    width = 0.13
    for i, metric in enumerate(metrics_jitter):
        vals = [r[metric] for r in rows]
        ax.bar(x + (i - len(metrics_jitter) / 2) * width, vals, width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("jitter measure")
    ax.set_title("Conventional vs. glide-aware residual jitter across conditions A-E\n"
                 "(good metric: low for B/D, high for A/C/E)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "jitter_comparison.png"), dpi=110)
    plt.close(fig)


def run_benchmark() -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for cond in all_conditions():
        print(f"\n=== {cond.name} ({cond.description}) ===")
        row = evaluate_condition(cond)
        rows.append(row)
        for key in FIELDNAMES[3:]:
            print(f"  {key:<28}{row[key]:.5f}" if isinstance(row[key], float) else f"  {key:<28}{row[key]}")

    per_file_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    with open(per_file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n저장 완료: {per_file_path}")

    plot_summary(rows)
    return rows


if __name__ == "__main__":
    run_benchmark()
