"""SP-V2-006 심층 분석(브리핑 §11): conventional/legacy/Praat jitter가 F0 움직임
(glide/vibrato 등 의도된 움직임)에 얼마나 의존하는지, residual jitter가 그
의존성을 얼마나 줄이면서 emotion 신호는 유지하는지 확인한다.

extract_real_corpus_features.py와 동일한 시드/표집으로 같은 파일 집합을 골라서
Praat(parselmouth)와 기존 저장소 구현(voice_quality_features.py)의 jitter/shimmer/HNR을
추가로 계산한다 - 메인 추출 파이프라인과 분리한 이유는 이미 실행 중인 메인
추출과 CPU를 다투지 않기 위함(둘 다 CPU-bound 병렬 작업).
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harmony_features import FMAX, FMIN, PITCH_SR
from signal_v2.common.audio_io import load_waveform_float
from signal_v2.evaluation.extract_real_corpus_features import stratified_sample, MAX_WORKERS
from signal_v2.voice_quality.glide_aware_jitter import praat_jitter_shimmer
from voice_quality_features import compute_voice_quality_features
from similarity import load_groups

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "real_corpus")


def process_one(audio_path: str) -> dict:
    waveform, sr = load_waveform_float(audio_path)
    existing = compute_voice_quality_features(waveform, sr)
    try:
        praat = praat_jitter_shimmer(waveform, sr, f0min=FMIN, f0max=min(FMAX, sr / 2 - 100))
    except Exception as exc:  # noqa: BLE001
        praat = {"praat_jitter": float("nan"), "praat_shimmer": float("nan")}
    return {
        "existing_repo_jitter": existing["jitter_local"], "existing_repo_shimmer": existing["shimmer_local"],
        "existing_repo_hnr_mean": existing["hnr_mean"],
        "praat_jitter": praat.get("praat_jitter", float("nan")), "praat_shimmer": praat.get("praat_shimmer", float("nan")),
    }


def _process_task(task: tuple) -> dict:
    wav_id, situation, audio_path = task
    try:
        row = process_one(audio_path)
        row["error"] = ""
    except Exception as exc:  # noqa: BLE001
        row = {"error": str(exc)}
    row["wav_id"] = wav_id
    row["situation"] = situation
    return row


def run(n_per_emotion: int = 200, seed: int = 42) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = load_groups()
    sample = stratified_sample(groups, n_per_emotion, seed)
    print(f"표본: {len(sample)}개 파일 (extract_real_corpus_features.py와 동일 시드)")

    rows = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_process_task, task) for task in sample]
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 100 == 0:
                elapsed = time.perf_counter() - start
                print(f"  {done}/{len(sample)} 완료 ({done / elapsed:.1f}개/초)")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    out_path = os.path.join(OUTPUT_DIR, "f006_praat_legacy.csv")
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
