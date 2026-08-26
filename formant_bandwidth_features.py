"""
harmony_features.py의 LPC 포먼트 검출 과정에서 이미 계산되지만 버려지고 있던
포먼트 대역폭(bandwidth)을 저장한다.

대역폭이 넓다는 건 그 공진이 빨리 감쇠한다는 뜻이고, 이는 성대 개방 시간이 길거나
성도 벽의 흡수가 큰 발성(breathy, 긴장 이완)과 관련이 있다 - 협화도(주파수들 사이의
관계)와는 다른, 포먼트 하나하나의 '선명도' 축이다.

harmony_features.py의 extract_formants_from_frame(..., return_bandwidth=True)와
FORMANT_SR/LPC_ORDER 등 동일 설정을 그대로 재사용해서 포먼트 검출 결과 자체가
harmony_features.py의 f1~f4_mean과 일관되게 나오도록 한다. 오디오를 새로 읽어
LPC를 다시 돌려야 해서 harmony_features.py만큼 시간이 걸린다.

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
from scipy.signal import resample_poly

from harmony_features import (
    FORMANT_FRAME_LEN, FORMANT_HOP, FORMANT_SR, MAX_FORMANTS,
    extract_formants_from_frame,
)
from similarity import get_audio_path, load_groups

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "formant_bandwidth")

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PROGRESS_LOG_EVERY = 200

BW_COLS = [f"bw{i + 1}_mean" for i in range(MAX_FORMANTS)]
FIELDNAMES = ["wav_id", "situation"] + BW_COLS + ["bw_overall_mean", "bw_overall_std", "n_valid_frames"]


def compute_formant_bandwidth_features(waveform: np.ndarray, orig_sr: int) -> dict:
    g = np.gcd(FORMANT_SR, orig_sr)
    resampled = resample_poly(waveform, FORMANT_SR // g, orig_sr // g)

    n_frames = 1 + max(0, (len(resampled) - FORMANT_FRAME_LEN) // FORMANT_HOP)

    per_formant_bw = [[] for _ in range(MAX_FORMANTS)]
    all_bw = []
    n_valid = 0
    for i in range(n_frames):
        start = i * FORMANT_HOP
        frame = resampled[start: start + FORMANT_FRAME_LEN]
        if len(frame) < FORMANT_FRAME_LEN:
            continue
        formants = extract_formants_from_frame(frame, FORMANT_SR, return_bandwidth=True)
        if len(formants) < 2:
            continue
        n_valid += 1
        for idx, (_, bw) in enumerate(formants):
            if idx < MAX_FORMANTS:
                per_formant_bw[idx].append(bw)
            all_bw.append(bw)

    result = {col: 0.0 for col in BW_COLS}
    result["bw_overall_mean"] = 0.0
    result["bw_overall_std"] = 0.0
    result["n_valid_frames"] = n_valid

    for idx, col in enumerate(BW_COLS):
        if per_formant_bw[idx]:
            result[col] = float(np.mean(per_formant_bw[idx]))
    if all_bw:
        result["bw_overall_mean"] = float(np.mean(all_bw))
        result["bw_overall_std"] = float(np.std(all_bw))

    return result


def extract_formant_bandwidth_features_from_path(audio_path: str) -> dict:
    sr, waveform = wavfile.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float64)
    if waveform.size and np.abs(waveform).max() > 1.0:
        waveform = waveform / 32768.0

    return compute_formant_bandwidth_features(waveform, sr)


def extract_formant_bandwidth_features(wav_id: str) -> dict:
    return extract_formant_bandwidth_features_from_path(get_audio_path(wav_id))


def _process_task(task: tuple) -> tuple:
    wav_id, situation, audio_path = task
    try:
        features = extract_formant_bandwidth_features_from_path(audio_path)
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

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_formant_bandwidth.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_emotion.csv")

    extract_all(groups, per_file_path)
    save_summary_by_emotion(per_file_path, summary_path)
    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
