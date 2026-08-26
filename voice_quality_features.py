"""
성대 진동 자체의 미세한 불안정성(jitter/shimmer)과 목소리의 맑음/쉰소리 정도(HNR)를 뽑는다.

지금까지의 화성/선율/리듬/음계 feature는 전부 "주파수들 사이의 관계"나 "시간에 따른
피치/음의 흐름"을 다뤘는데, 여기서는 그보다 더 미시적인 축 - 성대가 매 주기마다
얼마나 일정하게 떨리는지 - 를 본다. 감정에 따라 후두 긴장도가 달라지면(예: 공포로
인한 긴장, 슬픔으로 인한 이완) 이 미세한 주기성 자체가 흔들린다.

- Jitter(local, relative): 인접한 두 성대 진동 주기 길이 차이 / 평균 주기 길이.
- Shimmer(local, relative): 인접한 두 주기의 진폭(peak-to-peak) 차이 / 평균 진폭.
- HNR(Harmonics-to-Noise Ratio): 신호가 얼마나 주기적인지(배음 성분)와 얼마나
  비주기적인지(잡음 성분)의 비율을 자기상관으로 추정(Boersma 1993 방식의 근사).

Praat처럼 정밀한 성문 폐쇄점 검출은 아니고, pyin의 프레임별 F0 추정치를 이용해
파형에서 국소 최댓값을 따라가며 성대 주기를 근사적으로 추적하는 단순화된 구현이다
(harmony_features.py/melodic_profile.py와 같은 근사 노선).

similarity.py의 load_groups()/get_audio_path()를 재사용해서 4차년도+5차년도
통합 데이터셋, 7개 감정 전부를 대상으로 하고, 다른 추출 스크립트들과 동일한 방식
(CPU 프로세스 풀 병렬 + 재실행 시 이어서 진행)으로 처리한다.
"""

import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
from scipy.io import wavfile

from harmony_features import FMAX, FMIN, PITCH_SR
from similarity import get_audio_path, load_groups

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "voice_quality")

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PROGRESS_LOG_EVERY = 200

# librosa.pyin 기본값(frame_length=2048, hop_length=512)과 동일하게 맞춰서
# 프레임 인덱스 <-> 샘플 위치 변환이 일관되게 한다.
FRAME_LENGTH = 2048
HOP_LENGTH = 512

PERIOD_SEARCH_RATIO = 0.3    # 다음 주기 예측 위치 주변 +-30% 범위에서 실제 파형 피크를 재탐색
MIN_PERIODS_PER_RUN = 3      # 최소 이만큼 연속 주기가 잡혀야 그 구간을 jitter/shimmer 계산에 포함

HNR_SEARCH_RATIO = 0.2       # 자기상관에서 예측 주기 +-20% 범위 안에서 최대값 탐색

FIELDNAMES = [
    "wav_id", "situation",
    "jitter_local", "shimmer_local",
    "hnr_mean", "hnr_std",
    "n_periods", "n_hnr_frames",
]


def find_nearest_peak(y: np.ndarray, center_idx: int, radius: int) -> int:
    lo = max(0, center_idx - radius)
    hi = min(len(y), center_idx + radius + 1)
    if hi <= lo:
        return int(np.clip(center_idx, 0, len(y) - 1))
    local = y[lo:hi]
    return lo + int(np.argmax(np.abs(local)))


def mark_pitch_periods(y: np.ndarray, f0: np.ndarray, voiced_idx: np.ndarray) -> list:
    """연속된(사이에 무성음이 끼지 않은) 유성음 프레임 구간 하나에서 파형 피크를 따라가며
    성대 주기 경계(샘플 인덱스)를 근사적으로 추적한다."""
    start_frame, end_frame = voiced_idx[0], voiced_idx[-1]
    run_end_sample = (end_frame + 1) * HOP_LENGTH

    pos = start_frame * HOP_LENGTH
    f0_start = f0[start_frame]
    if f0_start <= 0 or np.isnan(f0_start):
        return []
    pos = find_nearest_peak(y, pos, radius=int(PITCH_SR / f0_start / 2))

    marks = [pos]
    while True:
        frame_idx = int(round(pos / HOP_LENGTH))
        frame_idx = min(max(frame_idx, start_frame), end_frame)
        f0_val = f0[frame_idx]
        if f0_val <= 0 or np.isnan(f0_val):
            break
        period = PITCH_SR / f0_val
        next_pos_est = pos + period
        if next_pos_est >= run_end_sample:
            break
        next_pos = find_nearest_peak(y, int(round(next_pos_est)), radius=int(period * PERIOD_SEARCH_RATIO))
        if next_pos <= pos:
            next_pos = pos + max(1, int(period * 0.5))
        marks.append(next_pos)
        pos = next_pos

    return marks


def compute_jitter_shimmer(y: np.ndarray, f0: np.ndarray) -> dict:
    voiced_idx = np.where(~np.isnan(f0) & (f0 > 0))[0]
    result = {"jitter_local": 0.0, "shimmer_local": 0.0, "n_periods": 0}
    if voiced_idx.size == 0:
        return result

    runs = np.split(voiced_idx, np.where(np.diff(voiced_idx) != 1)[0] + 1)

    period_pool, jitter_diffs = [], []
    amp_pool, shimmer_diffs = [], []

    for run in runs:
        if run.size < 2:
            continue
        marks = mark_pitch_periods(y, f0, run)
        if len(marks) < MIN_PERIODS_PER_RUN + 1:
            continue

        periods = np.diff(marks) / PITCH_SR
        amps = np.array([
            np.ptp(seg) if (seg := y[marks[i]:marks[i + 1]]).size > 0 else 0.0
            for i in range(len(marks) - 1)
        ])

        period_pool.extend(periods.tolist())
        jitter_diffs.extend(np.abs(np.diff(periods)).tolist())
        amp_pool.extend(amps.tolist())
        shimmer_diffs.extend(np.abs(np.diff(amps)).tolist())

    result["n_periods"] = len(period_pool)
    if jitter_diffs and np.mean(period_pool) > 0:
        result["jitter_local"] = float(np.mean(jitter_diffs) / np.mean(period_pool))
    if shimmer_diffs and np.mean(amp_pool) > 0:
        result["shimmer_local"] = float(np.mean(shimmer_diffs) / np.mean(amp_pool))

    return result


def compute_hnr(y: np.ndarray, f0: np.ndarray) -> dict:
    voiced_idx = np.where(~np.isnan(f0) & (f0 > 0))[0]
    result = {"hnr_mean": 0.0, "hnr_std": 0.0, "n_hnr_frames": 0}
    if voiced_idx.size == 0:
        return result

    pad = FRAME_LENGTH // 2
    y_padded = np.pad(y, pad, mode="reflect")

    hnr_values = []
    for idx in voiced_idx:
        center = idx * HOP_LENGTH + pad
        frame = y_padded[center - pad: center + pad]
        if len(frame) < FRAME_LENGTH:
            continue
        frame = frame * np.hamming(len(frame))
        energy = np.sum(frame ** 2)
        if energy == 0:
            continue

        period = PITCH_SR / f0[idx]
        lag_center = int(round(period))
        radius = max(1, int(period * HNR_SEARCH_RATIO))
        lo, hi = max(1, lag_center - radius), min(len(frame) - 1, lag_center + radius)
        if hi <= lo:
            continue

        best_r = 0.0
        for lag in range(lo, hi + 1):
            a, b = frame[:-lag], frame[lag:]
            denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
            if denom == 0:
                continue
            r = np.sum(a * b) / denom
            if r > best_r:
                best_r = r

        best_r = np.clip(best_r, 1e-6, 0.999999)
        hnr_db = 10 * np.log10(best_r / (1 - best_r))
        hnr_values.append(hnr_db)

    if hnr_values:
        result["hnr_mean"] = float(np.mean(hnr_values))
        result["hnr_std"] = float(np.std(hnr_values))
        result["n_hnr_frames"] = len(hnr_values)

    return result


def compute_voice_quality_features(waveform: np.ndarray, orig_sr: int) -> dict:
    y = librosa.resample(waveform, orig_sr=orig_sr, target_sr=PITCH_SR) if orig_sr != PITCH_SR else waveform
    f0, _, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=PITCH_SR, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)

    result = {}
    result.update(compute_jitter_shimmer(y, f0))
    result.update(compute_hnr(y, f0))
    return result


def extract_voice_quality_features_from_path(audio_path: str) -> dict:
    sr, waveform = wavfile.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float64)
    if waveform.size and np.abs(waveform).max() > 1.0:
        waveform = waveform / 32768.0

    return compute_voice_quality_features(waveform, sr)


def extract_voice_quality_features(wav_id: str) -> dict:
    return extract_voice_quality_features_from_path(get_audio_path(wav_id))


def _process_task(task: tuple) -> tuple:
    wav_id, situation, audio_path = task
    try:
        features = extract_voice_quality_features_from_path(audio_path)
    except Exception as exc:  # noqa: BLE001
        return wav_id, situation, None, str(exc)
    return wav_id, situation, features, None


def load_already_processed(per_file_path: str) -> set:
    processed = set()
    if os.path.exists(per_file_path):
        with open(per_file_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                processed.add(row["wav_id"])
    return processed


def build_tasks(groups: dict, processed: set) -> list:
    tasks = []
    for situation, wav_ids in groups.items():
        for wav_id in wav_ids:
            if wav_id in processed:
                continue
            audio_path = get_audio_path(wav_id)
            if not os.path.exists(audio_path):
                continue
            tasks.append((wav_id, situation, audio_path))
    return tasks


def extract_all(groups: dict, per_file_path: str) -> None:
    processed = load_already_processed(per_file_path)
    tasks = build_tasks(groups, processed)

    total = len(processed) + len(tasks)
    done = len(processed)
    if not tasks:
        print(f"모든 파일이 이미 처리되어 있습니다 ({done}/{total}).")
        return

    print(f"{len(tasks)}개 파일 처리 시작 (이미 완료 {done}개, 전체 {total}개, 워커 {MAX_WORKERS}개)")

    mode = "a" if processed else "w"
    start = time.perf_counter()

    with open(per_file_path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not processed:
            writer.writeheader()

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(_process_task, task) for task in tasks]
            for future in as_completed(futures):
                wav_id, situation, features, error = future.result()
                if error is not None:
                    print(f"경고: '{wav_id}' 처리 실패, 건너뜀 ({error})")
                    continue

                row = {"wav_id": wav_id, "situation": situation}
                row.update(features)
                writer.writerow(row)

                done += 1
                if done % PROGRESS_LOG_EVERY == 0:
                    f.flush()
                    elapsed = time.perf_counter() - start
                    rate = (done - len(processed)) / elapsed if elapsed > 0 else 0
                    remaining = (total - done) / rate if rate > 0 else float("inf")
                    print(f"{done}/{total} 완료 ({rate:.1f}개/초, 남은 시간 {remaining / 60:.1f}분)")


def save_summary_by_emotion(per_file_path: str, summary_path: str) -> None:
    numeric_cols = [c for c in FIELDNAMES if c not in ("wav_id", "situation")]
    per_situation: dict = {}
    with open(per_file_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            situation = row["situation"]
            per_situation.setdefault(situation, {col: [] for col in numeric_cols})
            for col in numeric_cols:
                per_situation[situation][col].append(float(row[col]))

    fieldnames = ["situation", "n_files"]
    for col in numeric_cols:
        fieldnames += [f"{col}_avg", f"{col}_std"]

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for situation, columns in per_situation.items():
            row = {"situation": situation, "n_files": len(next(iter(columns.values())))}
            for col in numeric_cols:
                values = np.array(columns[col])
                row[f"{col}_avg"] = float(values.mean())
                row[f"{col}_std"] = float(values.std())
            writer.writerow(row)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = load_groups()

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_voice_quality.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_emotion.csv")

    extract_all(groups, per_file_path)
    save_summary_by_emotion(per_file_path, summary_path)
    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
