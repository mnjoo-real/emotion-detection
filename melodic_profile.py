"""
음 자체의 절대적 높낮이(피치)가 아니라, 연속된 음들 '사이의 관계' — 선율적 음정(interval) —
을 화자의 음역대(성별/개인차)와 무관하게 분석한다.

harmony_features.py의 melodic_dissonance가 "인접 음정이 감각적으로 얼마나 거친가"를
하나의 점수로 요약했다면, 여기서는 그 음정 자체의 크기/방향 분포를 화성학/선율분석의
고전적 개념으로 뜯어본다:

- 반음(semitone) 단위 음정 = 12*log2(f2/f1) — 절대 주파수가 아니라 '비율'이라
  화자의 음역대(성별, 개인 음높이)가 달라도 값이 그대로 비교 가능하다.
- 순차진행(step, 온음/반음 정도의 작은 이동) vs 도약(leap, 3반음 초과의 큰 이동)
  — 선율분석에서 쓰는 표준 이분법.
- 상행/하행/유지 방향 비율, 그리고 방향+크기별 7단계 히스토그램(선율 윤곽 프로파일).

similarity.py의 load_groups()/get_audio_path()를 재사용해서 4차년도+5차년도
통합 데이터셋, 7개 감정 전부를 대상으로 한다. extract_features.py/harmony_features.py와
동일한 방식(CPU 프로세스 풀 병렬 + 재실행 시 이어서 진행)으로 처리한다.
"""

import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
from scipy.io import wavfile

from similarity import get_audio_path, load_groups

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "melodic_profile")

PITCH_SR = 8000
FMIN = 65.0    # ~C2
FMAX = 2093.0  # ~C7

STEP_LEAP_THRESHOLD = 2.5   # 이 값(반음) 이하면 순차진행, 초과하면 도약
FLAT_THRESHOLD = 0.5        # 이 값(반음) 이내면 '유지'로 간주

# 방향+크기별 7단계 구간 경계(반음). (-inf,-7], (-7,-3], (-3,-0.5], (-0.5,0.5], (0.5,3], (3,7], (7,inf)
HIST_EDGES = [-7, -3, -FLAT_THRESHOLD, FLAT_THRESHOLD, 3, 7]
HIST_LABELS = [
    "hist_large_fall", "hist_mid_fall", "hist_small_fall",
    "hist_flat",
    "hist_small_rise", "hist_mid_rise", "hist_large_rise",
]

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PROGRESS_LOG_EVERY = 200

FIELDNAMES = (
    ["wav_id", "situation"]
    + ["interval_abs_mean", "interval_abs_std"]
    + ["pct_rising", "pct_falling", "pct_flat"]
    + ["pct_step", "pct_leap"]
    + HIST_LABELS
    + ["n_intervals"]
)


def semitone_intervals(waveform: np.ndarray, orig_sr: int) -> np.ndarray:
    """연속된(사이에 무성음이 끼지 않은) 유성음 프레임 쌍 사이의 반음 음정 배열을 반환."""
    y_pitch = librosa.resample(waveform, orig_sr=orig_sr, target_sr=PITCH_SR) if orig_sr != PITCH_SR else waveform
    f0, _, _ = librosa.pyin(y_pitch, fmin=FMIN, fmax=FMAX, sr=PITCH_SR)

    voiced_idx = np.where(~np.isnan(f0))[0]
    if voiced_idx.size < 2:
        return np.array([])

    consecutive = voiced_idx[1:] == voiced_idx[:-1] + 1
    f1 = f0[voiced_idx[:-1][consecutive]]
    f2 = f0[voiced_idx[1:][consecutive]]
    if f1.size == 0:
        return np.array([])

    return 12 * np.log2(f2 / f1)


def compute_melodic_profile(waveform: np.ndarray, orig_sr: int) -> dict:
    intervals = semitone_intervals(waveform, orig_sr)

    result = {
        "interval_abs_mean": 0.0, "interval_abs_std": 0.0,
        "pct_rising": 0.0, "pct_falling": 0.0, "pct_flat": 0.0,
        "pct_step": 0.0, "pct_leap": 0.0,
        "n_intervals": int(intervals.size),
    }
    for label in HIST_LABELS:
        result[label] = 0.0

    if intervals.size == 0:
        return result

    abs_iv = np.abs(intervals)
    result["interval_abs_mean"] = float(abs_iv.mean())
    result["interval_abs_std"] = float(abs_iv.std())

    n = intervals.size
    result["pct_rising"] = float(np.mean(intervals > FLAT_THRESHOLD))
    result["pct_falling"] = float(np.mean(intervals < -FLAT_THRESHOLD))
    result["pct_flat"] = float(np.mean(abs_iv <= FLAT_THRESHOLD))

    result["pct_step"] = float(np.mean((abs_iv > FLAT_THRESHOLD) & (abs_iv <= STEP_LEAP_THRESHOLD)))
    result["pct_leap"] = float(np.mean(abs_iv > STEP_LEAP_THRESHOLD))

    bin_idx = np.digitize(intervals, HIST_EDGES)  # 0..len(HIST_LABELS)-1
    counts = np.bincount(bin_idx, minlength=len(HIST_LABELS))
    for i, label in enumerate(HIST_LABELS):
        result[label] = float(counts[i] / n)

    return result


def extract_melodic_profile_from_path(audio_path: str) -> dict:
    sr, waveform = wavfile.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float64)
    if waveform.size and np.abs(waveform).max() > 1.0:
        waveform = waveform / 32768.0

    return compute_melodic_profile(waveform, sr)


def extract_melodic_profile(wav_id: str) -> dict:
    return extract_melodic_profile_from_path(get_audio_path(wav_id))


def _process_task(task: tuple) -> tuple:
    wav_id, situation, audio_path = task
    try:
        features = extract_melodic_profile_from_path(audio_path)
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

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_melodic_profile.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_emotion.csv")

    extract_all(groups, per_file_path)
    save_summary_by_emotion(per_file_path, summary_path)
    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
