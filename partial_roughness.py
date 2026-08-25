"""
오디오 파형을 FFT로 분리해서 실제로 존재하는 배음(부분음, partial) 각각의
주파수/세기를 뽑고, 그 부분음들 사이의 Sethares 감각적 불협화도를 계산한다.

harmony_features.py의 두 지표와 다른 세 번째 축이다:
- 포먼트 러프니스: 발음(모음)에 따른 성도 공진 구조
- 선율적 불협화도: 시간에 따른 피치의 움직임
- 부분음 러프니스(이 스크립트): 순간순간 목소리 질감(맑음 vs 거침/beating)

깨끗한 목소리는 배음이 F0의 정수배(2:1, 3:2 같은 단순 비율)라 불협화도가 낮게 나오고,
거칠거나(harsh) 쉰 소리(creaky/breathy)면 배음이 정수배에서 어긋나거나 잡음이
섞여 불협화도가 올라간다 — 사실상 화성학적 불협화도로 잰 음질(voice quality) 지표다.

similarity.py의 load_groups()/get_audio_path()를 재사용해서 4차년도+5차년도
통합 데이터셋, 7개 감정 전부를 대상으로 한다. 다른 추출 스크립트들과 동일한 방식
(CPU 프로세스 풀 병렬 + 재실행 시 이어서 진행)으로 처리한다.
"""

import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from harmony_features import sethares_dissonance
from similarity import get_audio_path, load_groups

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "partial_roughness")

SPECTRAL_SR = 16000
FRAME_LEN = 2048   # 128ms - 배음을 촘촘히 구분할 만큼 충분한 주파수 해상도(~7.8Hz/bin)
HOP = 512           # 32ms
FREQ_MIN = 80.0
FREQ_MAX = 5000.0
MAX_PEAKS = 12
PEAK_REL_THRESHOLD = 0.1   # 가장 센 피크 대비 이 비율(10%) 미만이면 스펙트럼 누설/잡음으로 보고 버림
SILENCE_REL_THRESHOLD = 0.05  # 파일 최대 RMS 대비 이 비율 미만인 프레임은 무음으로 보고 건너뜀

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PROGRESS_LOG_EVERY = 200

FIELDNAMES = [
    "wav_id", "situation",
    "partial_roughness_mean", "partial_roughness_std",
    "avg_n_peaks", "n_valid_frames",
]


def pick_peaks(frame: np.ndarray, sr: int):
    window = np.hamming(len(frame))
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(len(frame), d=1 / sr)

    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    spec, fr = spectrum[mask], freqs[mask]
    if spec.size < 3:
        return np.array([]), np.array([])

    idx = np.where((spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:]))[0] + 1
    if idx.size == 0:
        return np.array([]), np.array([])

    peak_freqs, peak_amps = fr[idx], spec[idx]
    thresh = peak_amps.max() * PEAK_REL_THRESHOLD
    keep = peak_amps >= thresh
    peak_freqs, peak_amps = peak_freqs[keep], peak_amps[keep]

    if peak_freqs.size > MAX_PEAKS:
        top = np.argsort(-peak_amps)[:MAX_PEAKS]
        peak_freqs, peak_amps = peak_freqs[top], peak_amps[top]

    if peak_amps.size:
        peak_amps = peak_amps / peak_amps.max()
    return peak_freqs, peak_amps


def frame_roughness(frame: np.ndarray, sr: int):
    freqs, amps = pick_peaks(frame, sr)
    if freqs.size < 2:
        return None
    scores = []
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            scores.append(sethares_dissonance(freqs[i], freqs[j], amps[i], amps[j]))
    return float(np.mean(scores)), freqs.size


def compute_partial_roughness(waveform: np.ndarray, orig_sr: int) -> dict:
    g = np.gcd(SPECTRAL_SR, orig_sr)
    resampled = resample_poly(waveform, SPECTRAL_SR // g, orig_sr // g)

    n_frames = 1 + max(0, (len(resampled) - FRAME_LEN) // HOP)

    frame_rms = []
    for i in range(n_frames):
        start = i * HOP
        frame = resampled[start : start + FRAME_LEN]
        if len(frame) < FRAME_LEN:
            continue
        frame_rms.append(np.sqrt(np.mean(frame**2)))
    max_rms = max(frame_rms) if frame_rms else 0.0

    roughness_scores = []
    peak_counts = []
    for i in range(n_frames):
        start = i * HOP
        frame = resampled[start : start + FRAME_LEN]
        if len(frame) < FRAME_LEN:
            continue
        rms = np.sqrt(np.mean(frame**2))
        if max_rms == 0 or rms < max_rms * SILENCE_REL_THRESHOLD:
            continue

        result = frame_roughness(frame, SPECTRAL_SR)
        if result is None:
            continue
        score, n_peaks = result
        roughness_scores.append(score)
        peak_counts.append(n_peaks)

    if not roughness_scores:
        return {
            "partial_roughness_mean": 0.0, "partial_roughness_std": 0.0,
            "avg_n_peaks": 0.0, "n_valid_frames": 0,
        }

    return {
        "partial_roughness_mean": float(np.mean(roughness_scores)),
        "partial_roughness_std": float(np.std(roughness_scores)),
        "avg_n_peaks": float(np.mean(peak_counts)),
        "n_valid_frames": len(roughness_scores),
    }


def extract_partial_roughness_from_path(audio_path: str) -> dict:
    sr, waveform = wavfile.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float64)
    if waveform.size and np.abs(waveform).max() > 1.0:
        waveform = waveform / 32768.0

    return compute_partial_roughness(waveform, sr)


def extract_partial_roughness(wav_id: str) -> dict:
    return extract_partial_roughness_from_path(get_audio_path(wav_id))


def _process_task(task: tuple) -> tuple:
    wav_id, situation, audio_path = task
    try:
        features = extract_partial_roughness_from_path(audio_path)
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

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_partial_roughness.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_emotion.csv")

    extract_all(groups, per_file_path)
    save_summary_by_emotion(per_file_path, summary_path)
    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
