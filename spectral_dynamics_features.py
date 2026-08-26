"""
MFCC가 놓치는 스펙트럼의 "모양"과 "시간에 따른 변화"를 추가로 뽑는다.

- 스펙트럼 중심(centroid): 에너지가 저주파/고주파 중 어디에 쏠려있는지 - "음색의 밝기".
- 스펙트럼 롤오프(rolloff): 전체 에너지의 85%가 어느 주파수 아래에 들어가는지.
- 스펙트럼 평탄도(flatness): 스펙트럼이 잡음처럼 평평한지(1에 가까움) 또는 배음처럼
  뾰족한 피크가 있는지(0에 가까움) - 목소리의 "맑음/거침"과 관련.
- 영교차율(zero-crossing rate): 파형이 0을 넘나드는 빈도 - 고주파 성분/마찰음 비중의
  거친 지표.
- 스펙트럼 플럭스(flux): 인접 프레임 사이 정규화된 스펙트럼의 변화량 - 음색이 프레임마다
  얼마나 급격히 바뀌는지(조음의 역동성).

전부 librosa로 프레임 단위 계산 후 평균/표준편차로 요약한다. similarity.py의
load_groups()/get_audio_path()를 재사용해서 4차년도+5차년도 통합 데이터셋, 7개 감정
전부를 대상으로 하고, 다른 추출 스크립트들과 동일한 방식(CPU 프로세스 풀 병렬 +
재실행 시 이어서 진행)으로 처리한다.
"""

import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np

from similarity import get_audio_path, load_groups

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "spectral_dynamics")

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PROGRESS_LOG_EVERY = 200

FIELDNAMES = [
    "wav_id", "situation",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "zcr_mean", "zcr_std",
    "spectral_flux_mean", "spectral_flux_std",
]


def compute_spectral_dynamics(y: np.ndarray, sr: int) -> dict:
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    stft = np.abs(librosa.stft(y))
    norms = np.linalg.norm(stft, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = stft / norms
    flux = np.linalg.norm(np.diff(normalized, axis=1), axis=0) if normalized.shape[1] > 1 else np.array([0.0])

    return {
        "spectral_centroid_mean": float(centroid.mean()), "spectral_centroid_std": float(centroid.std()),
        "spectral_rolloff_mean": float(rolloff.mean()), "spectral_rolloff_std": float(rolloff.std()),
        "spectral_flatness_mean": float(flatness.mean()), "spectral_flatness_std": float(flatness.std()),
        "zcr_mean": float(zcr.mean()), "zcr_std": float(zcr.std()),
        "spectral_flux_mean": float(flux.mean()), "spectral_flux_std": float(flux.std()),
    }


def extract_spectral_dynamics_from_path(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    return compute_spectral_dynamics(y, sr)


def extract_spectral_dynamics(wav_id: str) -> dict:
    return extract_spectral_dynamics_from_path(get_audio_path(wav_id))


def _process_task(task: tuple) -> tuple:
    wav_id, situation, audio_path = task
    try:
        features = extract_spectral_dynamics_from_path(audio_path)
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

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_spectral_dynamics.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_emotion.csv")

    extract_all(groups, per_file_path)
    save_summary_by_emotion(per_file_path, summary_path)
    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
