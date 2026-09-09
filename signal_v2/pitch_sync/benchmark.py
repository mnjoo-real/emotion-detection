"""SP-V2-002 driver: 고정 윈도우 vs pitch-synchronous(c*T0) 윈도우 배음 추출 안정성 비교.

synthetic_signals.py의 harmonic 구조가 있는 glide/vibrato 조건(rolloff로 만든
배음 진폭비가 시간에 따라 '일정해야 하는' ground truth를 준다는 점을 이용)에서:
  - freq_rmse_cents: 추정 배음 주파수 vs 참값(k*F0(t))의 cents 오차 RMS
  - amp_ratio_std_db: 시간에 따른 20log10(A_k/A_1) 추정치의 표준편차(작을수록 안정)
  - amp_ratio_bias_db: 추정 평균과 참값(상수) 사이 편향

을 고정(40ms) vs 적응형(c=2,3,4 * T0) 윈도우 각각에 대해 계산한다. Ground-truth
F0(t)를 그대로 사용해 F0 추정 오차와 윈도우 선택 효과를 분리한다(Phase A와 독립적인
ablation).
"""

import csv
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.f0 import synthetic_signals as synth
from signal_v2.pitch_sync.adaptive_window import extract_harmonic_trajectory, FIXED_WINDOW_SEC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "pitch_sync")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

N_HARMONICS = 5
ROLLOFF = 0.88
MODES = [("fixed", None), ("adaptive_c2", 2.0), ("adaptive_c3", 3.0), ("adaptive_c4", 4.0)]

FIELDNAMES = [
    "condition", "mode", "harmonic", "freq_rmse_cents", "amp_ratio_std_db",
    "amp_ratio_bias_db", "true_amp_ratio_db", "n_valid_frames",
]


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _conditions_with_known_harmonics() -> list:
    conditions = [
        synth.harmonic_stack(f0=150.0, n_harmonics=N_HARMONICS, rolloff=ROLLOFF),
        synth.linear_glide(f0_start=100.0, f0_end=250.0, n_harmonics=N_HARMONICS, rolloff=ROLLOFF),
        synth.exponential_glide(f0_start=100.0, f0_end=300.0, n_harmonics=N_HARMONICS, rolloff=ROLLOFF),
        synth.vibrato(f0_center=180.0, n_harmonics=N_HARMONICS, rolloff=ROLLOFF),
        # 화자 F0 레지스터 효과를 직접 겨냥: 고정 40ms 윈도우가 담는 주기 수가
        # 저음역(75Hz -> 3주기)과 고음역(300Hz -> 12주기)에서 4배 차이 나는데,
        # 이게 배음 추정 안정성에 실제로 영향을 주는지 확인한다.
        synth.harmonic_stack(f0=75.0, n_harmonics=N_HARMONICS, rolloff=ROLLOFF),
        synth.harmonic_stack(f0=300.0, n_harmonics=N_HARMONICS, rolloff=ROLLOFF),
    ]
    # harmonic_stack이 f0만 다르게 세 번 나오므로 이름이 겹친다 - CSV/플롯에서
    # 구분되도록 f0를 이름에 반영한다.
    seen = {}
    for cond in conditions:
        seen[cond.name] = seen.get(cond.name, 0) + 1
        if seen[cond.name] > 1 or cond.name == "harmonic_stack":
            cond.name = f"{cond.name}_f0_{int(round(cond.tags.get('f0', 0)))}"
    return conditions


def evaluate_condition(gt: synth.SyntheticF0Signal) -> list:
    rows = []
    trajectories = {}
    for mode_name, c in MODES:
        mode = "fixed" if c is None else "adaptive"
        traj = extract_harmonic_trajectory(
            gt.waveform, gt.sr, gt.times, gt.f0_true,
            mode=mode, c=c if c is not None else 3.0, n_harmonics=N_HARMONICS,
        )
        trajectories[mode_name] = traj

        f0_at_traj = np.interp(traj.times, gt.times, gt.f0_true)
        for k in range(1, N_HARMONICS + 1):
            true_freq_k = k * f0_at_traj
            est_freq_k = traj.freq[:, k - 1]
            valid = ~np.isnan(est_freq_k)
            n_valid = int(valid.sum())

            if n_valid > 0:
                cents_err = 1200.0 * np.log2(est_freq_k[valid] / true_freq_k[valid])
                freq_rmse = float(np.sqrt(np.mean(cents_err ** 2)))
            else:
                freq_rmse = np.nan

            true_amp_ratio_db = 20 * np.log10(ROLLOFF ** (k - 1)) if k > 1 else 0.0
            est_amp_ratio = traj.amp_db_rel[:, k - 1]
            valid_amp = ~np.isnan(est_amp_ratio)
            if valid_amp.sum() > 0 and k > 1:
                amp_std = float(np.std(est_amp_ratio[valid_amp]))
                amp_bias = float(np.mean(est_amp_ratio[valid_amp]) - true_amp_ratio_db)
            else:
                amp_std = np.nan
                amp_bias = np.nan

            rows.append({
                "condition": gt.name, "mode": mode_name, "harmonic": k,
                "freq_rmse_cents": freq_rmse, "amp_ratio_std_db": amp_std,
                "amp_ratio_bias_db": amp_bias, "true_amp_ratio_db": true_amp_ratio_db,
                "n_valid_frames": n_valid,
            })

    plot_amp_trajectories(gt, trajectories, harmonic_idx=2)  # 3번째 배음(k=3)을 대표로 플롯
    return rows


def plot_amp_trajectories(gt: synth.SyntheticF0Signal, trajectories: dict, harmonic_idx: int) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    k = harmonic_idx + 1
    true_db = 20 * np.log10(ROLLOFF ** (k - 1))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(true_db, color="k", linestyle="--", linewidth=2, label=f"true H{k}/H1 ratio (const)")
    for mode_name, traj in trajectories.items():
        ax.plot(traj.times, traj.amp_db_rel[:, harmonic_idx], linewidth=1, alpha=0.8, label=mode_name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"H{k}/H1 amplitude ratio (dB)")
    ax.set_title(f"Fixed vs pitch-synchronous window — {gt.name}, harmonic {k}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{gt.name}_h{k}_amp_ratio.png"), dpi=110)
    plt.close(fig)


def summarize(rows: list) -> None:
    by_mode: dict = {}
    for row in rows:
        if row["harmonic"] == 1 or np.isnan(row["amp_ratio_std_db"]):
            continue
        by_mode.setdefault(row["mode"], {"freq_rmse": [], "amp_std": []})
        by_mode[row["mode"]]["freq_rmse"].append(row["freq_rmse_cents"])
        by_mode[row["mode"]]["amp_std"].append(row["amp_ratio_std_db"])

    print("\n" + "=" * 70)
    print(f"{'mode':<16}{'mean freq RMSE (cents)':>26}{'mean amp ratio std (dB)':>26}")
    print("=" * 70)
    for mode_name, vals in by_mode.items():
        print(f"{mode_name:<16}{np.nanmean(vals['freq_rmse']):>26.2f}{np.nanmean(vals['amp_std']):>26.3f}")


def run_benchmark() -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_rows = []
    for gt in _conditions_with_known_harmonics():
        print(f"\n=== condition: {gt.name} ===")
        rows = evaluate_condition(gt)
        all_rows.extend(rows)
        for mode_name, _ in MODES:
            mode_rows = [r for r in rows if r["mode"] == mode_name and r["harmonic"] > 1]
            mean_rmse = np.nanmean([r["freq_rmse_cents"] for r in mode_rows])
            mean_std = np.nanmean([r["amp_ratio_std_db"] for r in mode_rows])
            print(f"  [{mode_name:<12}] mean freq RMSE={mean_rmse:.2f} cents, mean amp ratio std={mean_std:.3f} dB")

    per_file_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    with open(per_file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    run_info = {
        "command": " ".join(sys.argv),
        "git_commit": _git_commit_hash(),
        "modes": [m[0] for m in MODES],
        "fixed_window_sec": FIXED_WINDOW_SEC,
        "n_harmonics": N_HARMONICS,
        "rolloff": ROLLOFF,
    }
    with open(os.path.join(OUTPUT_DIR, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {per_file_path}")
    summarize(all_rows)
    return all_rows


if __name__ == "__main__":
    run_benchmark()
