"""SP-V2-001: F0 추정기 벤치마크 driver.

synthetic_signals.all_conditions()의 각 조건에 대해 estimators.ESTIMATORS의 모든
추정기를 돌리고, synthetic_metrics.compute_metrics()로 RMSE(cents)/voiced frame
error/octave error rate/trajectory smoothness/실행 시간을 계산해
output/signal_v2/f0_benchmark/metrics.csv에 저장한다. 조건별로 true F0 vs 추정
F0 궤적을 겹쳐 그린 플롯도 저장한다(재현성: 사용한 명령/시드/커밋 해시도 함께 기록).
"""

import csv
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.f0 import synthetic_signals as synth
from signal_v2.f0.estimators import ESTIMATORS, run_estimator
from signal_v2.evaluation.synthetic_metrics import compute_metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "f0_benchmark")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

FIELDNAMES = [
    "condition", "estimator", "rmse_cents", "voiced_frame_error", "octave_error_rate",
    "trajectory_smoothness_cents", "n_frames_compared", "n_voiced_matched",
    "runtime_sec", "error",
]


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def plot_condition(condition_name: str, gt: synth.SyntheticF0Signal, results: dict) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gt.times, gt.f0_true, "k--", linewidth=2, label="ground truth", zorder=10)
    for name, est in results.items():
        if est.error or est.times.size == 0:
            continue
        ax.plot(est.times, est.f0, linewidth=1, alpha=0.85, label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title(f"F0 estimator comparison — {condition_name}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{condition_name}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run_benchmark(estimator_names: list = None) -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if estimator_names is None:
        estimator_names = list(ESTIMATORS.keys())

    conditions = synth.all_conditions()
    rows = []

    for gt in conditions:
        print(f"\n=== condition: {gt.name} ({gt.description}) ===")
        results = {}
        for est_name in estimator_names:
            start = time.perf_counter()
            est = run_estimator(est_name, gt.waveform, gt.sr)
            runtime = time.perf_counter() - start
            results[est_name] = est

            if est.error:
                print(f"  [{est_name}] 실패: {est.error}")
                rows.append({
                    "condition": gt.name, "estimator": est_name,
                    "rmse_cents": "", "voiced_frame_error": "", "octave_error_rate": "",
                    "trajectory_smoothness_cents": "", "n_frames_compared": 0,
                    "n_voiced_matched": 0, "runtime_sec": f"{runtime:.4f}", "error": est.error,
                })
                continue

            metrics = compute_metrics(
                est.times, est.f0, est.voiced, gt.times, gt.f0_true, gt.voiced_true,
            )
            print(
                f"  [{est_name:>14}] RMSE={metrics['rmse_cents']:.1f}cents  "
                f"voiced_err={metrics['voiced_frame_error']:.3f}  "
                f"octave_err={metrics['octave_error_rate']:.3f}  "
                f"smooth={metrics['trajectory_smoothness_cents']:.1f}  "
                f"({runtime:.3f}s)"
            )
            row = {"condition": gt.name, "estimator": est_name, "runtime_sec": f"{runtime:.4f}", "error": ""}
            row.update(metrics)
            rows.append(row)

        plot_path = plot_condition(gt.name, gt, results)
        print(f"  plot: {plot_path}")

    per_file_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    with open(per_file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    run_info = {
        "command": " ".join(sys.argv),
        "git_commit": _git_commit_hash(),
        "estimators": estimator_names,
        "n_conditions": len(conditions),
        "frame_period_ms": __import__("signal_v2.f0.estimators", fromlist=["FRAME_PERIOD_MS"]).FRAME_PERIOD_MS,
    }
    with open(os.path.join(OUTPUT_DIR, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {per_file_path}")
    return rows


def summarize(rows: list) -> None:
    """추정기별 평균 지표(모든 조건에 걸쳐)를 콘솔에 요약 출력."""
    by_estimator: dict = {}
    for row in rows:
        if row["error"]:
            continue
        by_estimator.setdefault(row["estimator"], {"rmse": [], "voiced_err": [], "octave_err": [], "smooth": [], "runtime": []})
        by_estimator[row["estimator"]]["rmse"].append(float(row["rmse_cents"]))
        by_estimator[row["estimator"]]["voiced_err"].append(float(row["voiced_frame_error"]))
        by_estimator[row["estimator"]]["octave_err"].append(float(row["octave_error_rate"]))
        by_estimator[row["estimator"]]["smooth"].append(float(row["trajectory_smoothness_cents"]))
        by_estimator[row["estimator"]]["runtime"].append(float(row["runtime_sec"]))

    print("\n" + "=" * 90)
    print(f"{'estimator':<16}{'mean RMSE(cents)':>18}{'mean voiced_err':>18}{'mean octave_err':>18}{'mean runtime(s)':>18}")
    print("=" * 90)
    for name, vals in by_estimator.items():
        print(
            f"{name:<16}{np.nanmean(vals['rmse']):>18.1f}{np.nanmean(vals['voiced_err']):>18.3f}"
            f"{np.nanmean(vals['octave_err']):>18.3f}{np.nanmean(vals['runtime']):>18.4f}"
        )


def main() -> None:
    rows = run_benchmark()
    summarize(rows)


if __name__ == "__main__":
    main()
