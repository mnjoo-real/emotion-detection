"""SP-V2-012: 기존 1,400개 pilot 표본(real_corpus_features.csv, 이미 F008/F009/
F010을 갖고 있음)에 새로 필요한 보조 quantity만 가볍게 추가 추출한다 - F008/F009/
F010을 다시 계산하지 않아서 훨씬 빠르다(harmonic trajectory FFT, d4c 재계산 없음).

추가하는 것: generic envelope modulation(§14 대조군), duration/speaking-rate(§12),
F0 dynamics(§13), energy dynamics(§14) 원시 quantity.
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
from signal_v2.evaluation.extract_sp012_features import (
    ANALYSIS_SR, duration_speaking_rate_features, energy_dynamics_features,
    f0_dynamics_features, generic_envelope_modulation_features,
)
from similarity import get_audio_path, load_groups

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLD_PILOT_CSV = os.path.join(BASE_DIR, "output", "signal_v2", "real_corpus", "real_corpus_features.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)


def extract_light(audio_path: str) -> dict:
    waveform, sr = load_waveform_float(audio_path)
    y16k = librosa.resample(waveform, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else waveform
    y8k = librosa.resample(waveform, orig_sr=sr, target_sr=PITCH_SR) if sr != PITCH_SR else waveform

    f0, _, _ = librosa.pyin(y8k, fmin=FMIN, fmax=FMAX, sr=PITCH_SR, frame_length=2048, hop_length=512)
    frame_times = np.arange(len(f0)) * 512 / PITCH_SR

    row = {}
    row.update(generic_envelope_modulation_features(y16k, ANALYSIS_SR))
    row.update(duration_speaking_rate_features(y16k, ANALYSIS_SR, f0, frame_times))
    row.update(f0_dynamics_features(f0, frame_times))
    row.update(energy_dynamics_features(y16k, ANALYSIS_SR))
    return row


def _process_task(task: tuple) -> dict:
    wav_id, situation, audio_path = task
    try:
        row = extract_light(audio_path)
        row["error"] = ""
    except Exception as exc:  # noqa: BLE001
        row = {"error": str(exc)}
    row["wav_id"] = wav_id
    row["situation"] = situation
    return row


def run() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    load_groups()

    with open(OLD_PILOT_CSV, encoding="utf-8-sig", newline="") as f:
        old_rows = list(csv.DictReader(f))
    tasks = [(r["wav_id"], r["situation"], get_audio_path(r["wav_id"])) for r in old_rows]
    print(f"표본: {len(tasks)}개 파일 (기존 1,400개 pilot과 동일)")

    rows = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_process_task, task) for task in tasks]
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 200 == 0:
                elapsed = time.perf_counter() - start
                print(f"  {done}/{len(tasks)} 완료 ({done / elapsed:.2f}개/초)")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    out_path = os.path.join(OUTPUT_DIR, "aux_features_original_1400.csv")
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
