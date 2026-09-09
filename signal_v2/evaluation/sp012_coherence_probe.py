"""SP-V2-012 §17: harmonic modulation coherence exploratory probe.

명시적으로 exploratory probe다 - 즉시 새 main feature family로 확대하지 않는다.
기존 F010의 inter_harmonic_modulation_coherence는 모든 배음 쌍의 평균 상관
하나로만 요약했다. 여기서는 그걸 더 세분화한다:
  - adjacent-harmonic coherence: 바로 인접한 배음 쌍(H1-H2, H2-H3, ...)의 평균 상관
  - distant-harmonic coherence: 2 이상 떨어진 배음 쌍의 평균 상관
  - low-vs-high coherence: 낮은 배음군(H1-H2) 대표값과 높은 배음군(H4-H5) 대표값 사이 상관
  - coherence variance: 모든 쌍별 상관의 표준편차(coherence 자체가 얼마나 균일한지)

비용을 통제하기 위해 전체 1,400개가 아니라 감정별 100개(700개) 부분표본에만 적용한다.
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harmony_features import FMAX, FMIN, PITCH_SR
from signal_v2.common.audio_io import load_waveform_float
from signal_v2.pitch_sync.adaptive_window import extract_harmonic_trajectory
from similarity import get_audio_path, load_groups

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")
ANALYSIS_SR = 16000
N_HARMONICS = 5
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)


def pairwise_correlations(amp: np.ndarray) -> dict:
    n = amp.shape[1]
    adjacent, distant, all_pairs = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = amp[:, i], amp[:, j]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if valid.sum() < 4 or a[valid].std() == 0 or b[valid].std() == 0:
                continue
            r = np.corrcoef(a[valid], b[valid])[0, 1]
            all_pairs.append(r)
            if j - i == 1:
                adjacent.append(r)
            else:
                distant.append(r)

    low_high = np.nan
    if n >= 5:
        low = np.nanmean(amp[:, 0:2], axis=1)  # H1-H2 대표값
        high = np.nanmean(amp[:, 3:5], axis=1)  # H4-H5 대표값
        valid = ~np.isnan(low) & ~np.isnan(high)
        if valid.sum() >= 4 and low[valid].std() > 0 and high[valid].std() > 0:
            low_high = float(np.corrcoef(low[valid], high[valid])[0, 1])

    return {
        "coh_adjacent_mean": float(np.mean(adjacent)) if adjacent else np.nan,
        "coh_distant_mean": float(np.mean(distant)) if distant else np.nan,
        "coh_low_vs_high": low_high,
        "coh_all_pairs_std": float(np.std(all_pairs)) if all_pairs else np.nan,
        "coh_all_pairs_mean": float(np.mean(all_pairs)) if all_pairs else np.nan,
    }


def extract_one(audio_path: str) -> dict:
    waveform, sr = load_waveform_float(audio_path)
    y16k = librosa.resample(waveform, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else waveform
    y8k = librosa.resample(waveform, orig_sr=sr, target_sr=PITCH_SR) if sr != PITCH_SR else waveform

    f0, _, _ = librosa.pyin(y8k, fmin=FMIN, fmax=FMAX, sr=PITCH_SR, frame_length=2048, hop_length=512)
    frame_times = np.arange(len(f0)) * 512 / PITCH_SR

    traj = extract_harmonic_trajectory(y16k, ANALYSIS_SR, frame_times, f0, mode="adaptive", c=3, n_harmonics=N_HARMONICS)
    return pairwise_correlations(traj.amp)


def _process_task(task: tuple) -> dict:
    wav_id, situation, audio_path = task
    try:
        row = extract_one(audio_path)
        row["error"] = ""
    except Exception as exc:  # noqa: BLE001
        row = {"error": str(exc)}
    row["wav_id"] = wav_id
    row["situation"] = situation
    return row


def run(n_per_emotion: int = 100, seed: int = 42) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = load_groups()
    rng = np.random.RandomState(seed)
    sample = []
    for situation, wav_ids in groups.items():
        existing = [w for w in wav_ids if os.path.exists(get_audio_path(w))]
        chosen = rng.choice(existing, size=min(n_per_emotion, len(existing)), replace=False)
        sample.extend((wid, situation, get_audio_path(wid)) for wid in chosen)
    print(f"표본: {len(sample)}개 파일 (coherence exploratory probe)")

    rows = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_process_task, task) for task in sample]
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(sample)} 완료 ({done / (time.perf_counter() - start):.2f}개/초)")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    out_path = os.path.join(OUTPUT_DIR, "coherence_probe.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    n_errors = sum(1 for r in rows if r.get("error"))
    print(f"\n저장 완료: {out_path} (오류 {n_errors}/{len(rows)}개)")
    return out_path


if __name__ == "__main__":
    run()
