"""
발화의 유성음 피치(F0)를 12음계(pitch class) 분포로 접은 뒤, Krumhansl-Schmuckler
조성 판별 알고리즘으로 장조(major)/단조(minor) 중 어느 쪽 음계 프로파일에 더 가까운지를
feature로 뽑는다.

harmony_features.py의 pitch_class_entropy가 "한 음역에 머무르는지"만 봤다면, 여기서는
그 분포의 '모양'을 서양 음악 이론의 장조/단조 프로파일(Krumhansl & Kessler, 1982)과
비교한다:

- 파일 자신의 중앙값 피치를 기준(0 cents)으로 유성음 프레임을 12개 pitch class
  구간에 접어 히스토그램을 만든다 (harmony_features.py와 동일 방식).
- 그 히스토그램을 12개 장조 프로파일(모든 조옮김)과 12개 단조 프로파일 각각과
  피어슨 상관계수로 비교해, 가장 잘 맞는 장조 상관값/단조 상관값을 구한다.
- 장조 상관 - 단조 상관 값이 클수록 "장조에 가까운"(밝은) 음높이 분포,
  작을수록(음수) "단조에 가까운"(어두운) 음높이 분포로 해석한다.

말소리의 F0 윤곽은 화성이 아니라 억양이라 음악적 조성 개념을 문자 그대로 가진 건
아니지만, harmony_features.py와 같은 방식으로 화성학 개념을 억양 분석에 번역해
적용한 것이다.

similarity.py의 load_groups()/get_audio_path()를 재사용해서 4차년도+5차년도
통합 데이터셋, 7개 감정 전부를 대상으로 한다. 다른 추출 스크립트들과 동일한 방식
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

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "tonality")

# Krumhansl & Kessler (1982) 조성 프로파일 - 각 pitch class(0=으뜸음)가 해당 조성에서
# 얼마나 안정적/중심적으로 느껴지는지를 나타내는 경험적 가중치.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PROGRESS_LOG_EVERY = 200

FIELDNAMES = [
    "wav_id", "situation",
    "key_major_corr", "key_minor_corr", "key_major_minor_diff", "key_mode",
    "n_voiced_pitch_frames",
]


def best_key_correlation(hist: np.ndarray, profile: np.ndarray) -> float:
    """hist(12,)를 profile의 12가지 조옮김(순환 이동) 각각과 피어슨 상관 비교해 최댓값을 반환."""
    best = -1.0
    for shift in range(12):
        rotated = np.roll(profile, shift)
        if hist.std() == 0 or rotated.std() == 0:
            continue
        corr = float(np.corrcoef(hist, rotated)[0, 1])
        if not np.isnan(corr) and corr > best:
            best = corr
    return best


def compute_tonality_features(waveform: np.ndarray, orig_sr: int) -> dict:
    y_pitch = librosa.resample(waveform, orig_sr=orig_sr, target_sr=PITCH_SR) if orig_sr != PITCH_SR else waveform
    f0, _, _ = librosa.pyin(y_pitch, fmin=FMIN, fmax=FMAX, sr=PITCH_SR)

    voiced_f0 = f0[~np.isnan(f0)]
    result = {
        "key_major_corr": 0.0, "key_minor_corr": 0.0, "key_major_minor_diff": 0.0,
        "key_mode": "unknown", "n_voiced_pitch_frames": int(voiced_f0.size),
    }
    # 12개 구간 히스토그램이 의미 있으려면 최소한의 유성음 프레임 수가 필요하다.
    if voiced_f0.size < 12:
        return result

    ref = np.median(voiced_f0)
    cents = 1200 * np.log2(voiced_f0 / ref)
    pitch_class = np.mod(cents, 1200)
    hist, _ = np.histogram(pitch_class, bins=12, range=(0, 1200))
    hist = hist.astype(np.float64)

    major_corr = best_key_correlation(hist, MAJOR_PROFILE)
    minor_corr = best_key_correlation(hist, MINOR_PROFILE)

    result["key_major_corr"] = major_corr
    result["key_minor_corr"] = minor_corr
    result["key_major_minor_diff"] = major_corr - minor_corr
    result["key_mode"] = "major" if major_corr >= minor_corr else "minor"
    return result


def extract_tonality_features_from_path(audio_path: str) -> dict:
    sr, waveform = wavfile.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float64)
    if waveform.size and np.abs(waveform).max() > 1.0:
        waveform = waveform / 32768.0

    return compute_tonality_features(waveform, sr)


def extract_tonality_features(wav_id: str) -> dict:
    return extract_tonality_features_from_path(get_audio_path(wav_id))


def _process_task(task: tuple) -> tuple:
    wav_id, situation, audio_path = task
    try:
        features = extract_tonality_features_from_path(audio_path)
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
    numeric_cols = ["key_major_corr", "key_minor_corr", "key_major_minor_diff", "n_voiced_pitch_frames"]
    per_situation: dict = {}
    with open(per_file_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            situation = row["situation"]
            per_situation.setdefault(situation, {"mode_counts": {"major": 0, "minor": 0}, **{col: [] for col in numeric_cols}})
            for col in numeric_cols:
                per_situation[situation][col].append(float(row[col]))
            mode = row["key_mode"]
            if mode in ("major", "minor"):
                per_situation[situation]["mode_counts"][mode] += 1

    fieldnames = ["situation", "n_files", "pct_major"]
    for col in numeric_cols:
        fieldnames += [f"{col}_avg", f"{col}_std"]

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for situation, columns in per_situation.items():
            n_files = len(columns[numeric_cols[0]])
            mode_counts = columns["mode_counts"]
            n_mode = mode_counts["major"] + mode_counts["minor"]
            row = {
                "situation": situation,
                "n_files": n_files,
                "pct_major": (mode_counts["major"] / n_mode) if n_mode else 0.0,
            }
            for col in numeric_cols:
                values = np.array(columns[col])
                row[f"{col}_avg"] = float(values.mean())
                row[f"{col}_std"] = float(values.std())
            writer.writerow(row)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = load_groups()

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_tonality.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_emotion.csv")

    extract_all(groups, per_file_path)
    save_summary_by_emotion(per_file_path, summary_path)
    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
