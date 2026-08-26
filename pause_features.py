"""
발화 중간에 끊기는 '휴지(pause)' 패턴을 뽑는다.

melody_rhythm_features.py의 리듬 feature들이 "소리가 나는 구간"만 다뤘다면,
여기서는 정반대로 "소리가 끊기는 구간"을 본다 - 망설임, 숨고르기, 감정적 정지 등이
반영될 수 있는 축이다. 프레임 RMS 에너지가 파일 자신의 최대 RMS 대비 낮은 구간을
무음으로 보고, 그런 구간이 일정 길이 이상 지속될 때만 '휴지'로 센다(너무 짧은
무음은 자음 사이 자연스러운 끊김이라 휴지로 보지 않음).

similarity.py의 load_groups()/get_audio_path()를 재사용해서 4차년도+5차년도
통합 데이터셋, 7개 감정 전부를 대상으로 하고, 다른 추출 스크립트들과 동일한 방식
(CPU 프로세스 풀 병렬 + 재실행 시 이어서 진행)으로 처리한다.
"""

import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.io import wavfile

from similarity import get_audio_path, load_groups

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "pause")

FRAME_LEN_SEC = 0.025
HOP_SEC = 0.010
SILENCE_REL_THRESHOLD = 0.1   # 파일 최대 RMS 대비 이 비율 미만이면 무음 프레임
MIN_PAUSE_SEC = 0.15          # 이보다 짧은 무음은 자음 사이 끊김으로 보고 휴지로 안 셈

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PROGRESS_LOG_EVERY = 200

FIELDNAMES = [
    "wav_id", "situation",
    "n_pauses", "pause_dur_mean", "pause_dur_std",
    "total_pause_ratio", "longest_pause_dur",
    "n_frames",
]


def compute_pause_features(waveform: np.ndarray, sr: int) -> dict:
    frame_len = int(FRAME_LEN_SEC * sr)
    hop = int(HOP_SEC * sr)
    n_frames = 1 + max(0, (len(waveform) - frame_len) // hop)

    result = {
        "n_pauses": 0, "pause_dur_mean": 0.0, "pause_dur_std": 0.0,
        "total_pause_ratio": 0.0, "longest_pause_dur": 0.0, "n_frames": n_frames,
    }
    if n_frames == 0:
        return result

    rms = np.array([
        np.sqrt(np.mean(waveform[i * hop: i * hop + frame_len] ** 2))
        for i in range(n_frames)
    ])
    max_rms = rms.max()
    if max_rms == 0:
        return result

    is_silent = rms < max_rms * SILENCE_REL_THRESHOLD

    pause_durations = []
    run_len = 0
    for silent in is_silent:
        if silent:
            run_len += 1
        else:
            if run_len > 0:
                dur = run_len * HOP_SEC
                if dur >= MIN_PAUSE_SEC:
                    pause_durations.append(dur)
            run_len = 0
    if run_len > 0:
        dur = run_len * HOP_SEC
        if dur >= MIN_PAUSE_SEC:
            pause_durations.append(dur)

    if pause_durations:
        arr = np.array(pause_durations)
        total_duration = n_frames * HOP_SEC
        result["n_pauses"] = int(arr.size)
        result["pause_dur_mean"] = float(arr.mean())
        result["pause_dur_std"] = float(arr.std())
        result["total_pause_ratio"] = float(arr.sum() / total_duration)
        result["longest_pause_dur"] = float(arr.max())

    return result


def extract_pause_features_from_path(audio_path: str) -> dict:
    sr, waveform = wavfile.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float64)
    if waveform.size and np.abs(waveform).max() > 1.0:
        waveform = waveform / 32768.0

    return compute_pause_features(waveform, sr)


def extract_pause_features(wav_id: str) -> dict:
    return extract_pause_features_from_path(get_audio_path(wav_id))


def _process_task(task: tuple) -> tuple:
    wav_id, situation, audio_path = task
    try:
        features = extract_pause_features_from_path(audio_path)
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

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_pause.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_emotion.csv")

    extract_all(groups, per_file_path)
    save_summary_by_emotion(per_file_path, summary_path)
    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
