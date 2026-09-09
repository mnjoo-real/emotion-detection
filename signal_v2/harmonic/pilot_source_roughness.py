"""SP-V2-004 pilot driver: raw vs source(-normalized) roughness on a stratified sample.

전체 34k 코퍼스에 IAIF를 다 돌리는 건 이번 세션 범위를 넘는 다중 시간대 작업이라
(harmony_features.py의 LPC formant 추출도 원래 몇 시간 걸렸다는 README 기록과
비슷한 자릿수), 여기서는 감정별 균형 표집한 파일 표본(기본 40개/감정, 총 280개)에
대해 먼저 raw/source/source_norm roughness를 계산하고 effect size, 성별 민감도,
기존 partial_roughness_mean/formant_roughness_mean과의 상관(redundancy)을 확인하는
파일럿을 수행한다. 결과가 유망하면 전체 코퍼스 재실행을 별도 백그라운드 작업으로
고려한다 - 이 파일럿 자체를 전체 코퍼스 결과인 것처럼 보고하지 않는다.
"""

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.common.audio_io import load_waveform_float
from signal_v2.f0.estimators import estimate_pyin
from signal_v2.harmonic.source_roughness import compute_roughness_variants
from signal_v2.source_filter.iaif import iaif_windowed
from similarity import get_audio_path, load_groups

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "source_roughness")
ANALYSIS_SR = 16000
N_PER_EMOTION = 40
SEED = 42
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)

FIELDNAMES = [
    "wav_id", "situation", "gender",
    "raw_roughness_mean", "raw_roughness_std",
    "source_roughness_mean", "source_roughness_std",
    "source_roughness_norm_mean", "source_roughness_norm_std",
    "n_valid_frames_raw", "n_valid_frames_source",
    "existing_partial_roughness_mean", "existing_formant_roughness_mean",
    "error",
]


def load_gender_map() -> dict:
    gender_map = {}
    for path, encoding in [
        (os.path.join(BASE_DIR, "4차년도.csv"), "cp949"),
        (os.path.join(BASE_DIR, "5차년도_2차.csv"), "cp949"),
    ]:
        with open(path, encoding=encoding, newline="") as f:
            for row in csv.DictReader(f):
                gender_map[row["wav_id"].strip()] = row["성별"].strip()
    return gender_map


def load_existing_csv(path: str) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["wav_id"]: row for row in csv.DictReader(f)}


def stratified_sample(groups: dict, n_per_emotion: int, seed: int) -> list:
    """(wav_id, situation, audio_path) 튜플 목록을 반환한다. audio_path를 메인
    프로세스에서 미리 계산해두는 이유: get_audio_path()는 similarity.py의
    module-level 캐시(_WAV_ID_TO_AUDIO_DIR, load_groups()가 채움)에 의존하는데,
    ProcessPoolExecutor의 워커 프로세스는 그 캐시가 채워지지 않은 새 프로세스라
    워커 안에서 get_audio_path()를 호출하면 실패한다 - 기존 harmony_features.py
    등이 task 튜플에 audio_path를 미리 담아 넘기는 것과 동일한 이유다."""
    rng = np.random.RandomState(seed)
    sample = []
    for situation, wav_ids in groups.items():
        existing = [w for w in wav_ids if os.path.exists(get_audio_path(w))]
        if not existing:
            continue
        chosen = rng.choice(existing, size=min(n_per_emotion, len(existing)), replace=False)
        sample.extend((wid, situation, get_audio_path(wid)) for wid in chosen)
    return sample


def process_one(audio_path: str) -> dict:
    waveform, sr = load_waveform_float(audio_path)
    y = librosa.resample(waveform, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else waveform

    pyin_result = estimate_pyin(y, ANALYSIS_SR)
    iaif_result = iaif_windowed(y, ANALYSIS_SR, frame_sec=0.032, hop_sec=0.016)

    variants = compute_roughness_variants(
        y, iaif_result["glottal_flow_derivative"], ANALYSIS_SR,
        pyin_result.times, pyin_result.f0,
    )
    return variants


def _process_task(task: tuple) -> dict:
    wav_id, situation, audio_path = task
    try:
        variants = process_one(audio_path)
        variants["wav_id"] = wav_id
        variants["situation"] = situation
        variants["error"] = ""
    except Exception as exc:  # noqa: BLE001 - 워커 실패로 전체 파일럿이 중단되지 않게
        variants = {"wav_id": wav_id, "situation": situation, "error": str(exc)}
    return variants


def run_pilot(n_per_emotion: int = N_PER_EMOTION) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    groups = load_groups()
    sample = stratified_sample(groups, n_per_emotion, SEED)
    print(f"표본: {len(sample)}개 파일 ({n_per_emotion}개/감정 x {len(groups)}개 감정)")

    gender_map = load_gender_map()
    existing_partial = load_existing_csv(os.path.join(BASE_DIR, "output", "partial_roughness", "per_file_partial_roughness.csv"))
    existing_harmony = load_existing_csv(os.path.join(BASE_DIR, "output", "harmony", "per_file_harmony.csv"))

    per_file_path = os.path.join(OUTPUT_DIR, "pilot_per_file.csv")
    rows = []
    start = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_process_task, task) for task in sample]
        for future in as_completed(futures):
            variants = future.result()
            variants["gender"] = gender_map.get(variants["wav_id"], "")
            variants["existing_partial_roughness_mean"] = existing_partial.get(variants["wav_id"], {}).get("partial_roughness_mean", "")
            variants["existing_formant_roughness_mean"] = existing_harmony.get(variants["wav_id"], {}).get("formant_roughness_mean", "")
            rows.append(variants)

            done += 1
            if done % 20 == 0:
                elapsed = time.perf_counter() - start
                rate = done / elapsed
                print(f"  {done}/{len(sample)} 완료 ({rate:.1f}개/초, 남은 시간 {(len(sample) - done) / rate:.0f}초)")

    with open(per_file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})

    n_errors = sum(1 for r in rows if r.get("error"))
    print(f"\n저장 완료: {per_file_path} (오류 {n_errors}/{len(rows)}개)")
    return per_file_path


if __name__ == "__main__":
    run_pilot()
