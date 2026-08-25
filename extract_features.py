"""
MFCC / 피치(F0) / 에너지(RMS) 통계를 뽑아 감정별 음향 특징을 추출한다.

similarity.py의 load_groups()/get_audio_path()를 그대로 재사용해서
4차년도 + 5차년도_2차 통합 데이터셋, 7개 감정(happiness, angry, disgust,
fear, neutral, sadness, surprise) 전부를 처리한다.

- 피치 추적(librosa.pyin)은 표본 수가 많을수록 급격히 느려지므로, 사람 목소리
  피치 대역(~C7 = 2093Hz)을 충분히 커버하는 PITCH_SR(8000Hz)로 다운샘플링한
  신호에서만 계산한다. MFCC/에너지는 원본 샘플레이트를 그대로 사용한다.
- 파일 수가 34,000개 규모라 프로세스 풀로 병렬 처리한다.
- 파일 단위 특징(per_file_features.csv)은 이미 처리된 wav_id를 건너뛰는
  방식으로 재실행 시 이어서 진행한다.
"""

import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np

from similarity import get_audio_path, load_groups

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "features")

N_MFCC = 13
PITCH_SR = 8000
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PROGRESS_LOG_EVERY = 200

MFCC_MEAN_COLS = [f"mfcc_{i + 1}_mean" for i in range(N_MFCC)]
MFCC_STD_COLS = [f"mfcc_{i + 1}_std" for i in range(N_MFCC)]
PITCH_ENERGY_COLS = ["pitch_mean", "pitch_std", "pitch_median", "voiced_ratio", "energy_mean", "energy_std"]
FEATURE_COLS = MFCC_MEAN_COLS + MFCC_STD_COLS + PITCH_ENERGY_COLS
FIELDNAMES = ["wav_id", "situation"] + FEATURE_COLS


def extract_features_from_path(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    y_pitch = librosa.resample(y, orig_sr=sr, target_sr=PITCH_SR) if sr != PITCH_SR else y
    f0, voiced_flag, _ = librosa.pyin(
        y_pitch, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=PITCH_SR
    )
    voiced_f0 = f0[~np.isnan(f0)]
    if voiced_f0.size > 0:
        pitch_mean = float(np.mean(voiced_f0))
        pitch_std = float(np.std(voiced_f0))
        pitch_median = float(np.median(voiced_f0))
    else:
        pitch_mean = pitch_std = pitch_median = 0.0
    voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag.size > 0 else 0.0

    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(rms.mean())
    energy_std = float(rms.std())

    row = {}
    row.update({col: float(mfcc_mean[i]) for i, col in enumerate(MFCC_MEAN_COLS)})
    row.update({col: float(mfcc_std[i]) for i, col in enumerate(MFCC_STD_COLS)})
    row.update(
        {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_median": pitch_median,
            "voiced_ratio": voiced_ratio,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
        }
    )
    return row


def extract_features(wav_id: str) -> dict:
    return extract_features_from_path(get_audio_path(wav_id))


def _process_task(task: tuple) -> tuple:
    wav_id, situation, audio_path = task
    try:
        features = extract_features_from_path(audio_path)
    except Exception as exc:  # noqa: BLE001 - 워커 프로세스 예외를 메인으로 전달
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
    per_situation: dict = {}
    with open(per_file_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            situation = row["situation"]
            per_situation.setdefault(situation, {col: [] for col in FEATURE_COLS})
            for col in FEATURE_COLS:
                per_situation[situation][col].append(float(row[col]))

    fieldnames = ["situation", "n_files"]
    for col in FEATURE_COLS:
        fieldnames += [f"{col}_avg", f"{col}_std"]

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for situation, columns in per_situation.items():
            row = {"situation": situation, "n_files": len(next(iter(columns.values())))}
            for col in FEATURE_COLS:
                values = np.array(columns[col])
                row[f"{col}_avg"] = float(values.mean())
                row[f"{col}_std"] = float(values.std())
            writer.writerow(row)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = load_groups()

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_features.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_emotion.csv")

    extract_all(groups, per_file_path)
    save_summary_by_emotion(per_file_path, summary_path)
    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
