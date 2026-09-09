"""Stage 1: F006-F010 다섯 feature family를 실제 코퍼스 표본에 대해 한 번에
추출한다(전처리 공유로 중복 계산을 줄인다 - signal_processing_v2_log.md의
"Next steps"에 이미 남긴 계획).

파일당 공유되는 전처리:
  - waveform 로드 + 16kHz 리샘플(F008/F009/F010용)
  - pyin F0 궤적, 8kHz/frame_length=2048/hop=512 (voice_quality_features.py와
    동일 설정 - F006의 실제 pitch marking이 이 관례에 의존하기 때문) - F006/F007/
    F009/F010이 이 F0 궤적을 공유해서 쓴다.
  - F008만 독립적으로 pyworld.harvest를 다시 돌린다(WORLD 자체 F0가 필요).
"""

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harmony_features import FMAX, FMIN, PITCH_SR
from signal_v2.common.audio_io import load_waveform_float
from signal_v2.harmonic.harmonic_modulation import harmonic_modulation_features
from signal_v2.harmonic.harmonic_phase import harmonic_phase_features
from signal_v2.pitch_sync.adaptive_window import extract_harmonic_trajectory
from signal_v2.prosody.cwt_prosody import extract_cwt_prosody_features
from signal_v2.voice_quality.aperiodicity import extract_aperiodicity_features
from signal_v2.voice_quality.glide_aware_jitter import conventional_jitter_shimmer, residual_jitter_shimmer
from voice_quality_features import FRAME_LENGTH, HOP_LENGTH, mark_pitch_periods
from similarity import get_audio_path, load_groups

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "real_corpus")
ANALYSIS_SR = 16000
N_HARMONICS = 5
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)


def f0_motion_energy(f0_times: np.ndarray, f0_values: np.ndarray) -> float:
    """F0 궤적(반음 환산)의 1차 미분 RMS - "pitch-motion energy"의 단순하고
    해석 가능한 대응 지표(브리핑 §11)."""
    voiced = ~np.isnan(f0_values) & (f0_values > 0)
    if voiced.sum() < 4:
        return 0.0
    ref = np.median(f0_values[voiced])
    semitone = 12 * np.log2(np.where(voiced, f0_values, ref) / ref)
    semitone_interp = np.interp(f0_times, f0_times[voiced], semitone[voiced])
    dt = np.diff(f0_times)
    dt[dt == 0] = np.nan
    slope = np.diff(semitone_interp) / dt
    slope = slope[np.isfinite(slope)]
    return float(np.sqrt(np.mean(slope ** 2))) if slope.size else 0.0


def extract_f006(y8k: np.ndarray, f0: np.ndarray) -> dict:
    """실제 오디오에서 conventional vs residual jitter/shimmer를 계산한다.
    voice_quality_features.mark_pitch_periods를 그대로 재사용해서 기존
    파이프라인과 동일한(불완전할 수 있는) pitch marking을 공유하고, 그 위에
    conventional/residual 두 공식만 비교한다 - 마킹 품질 차이가 아니라 공식
    자체의 차이만 isolate하기 위함."""
    voiced_idx = np.where(~np.isnan(f0) & (f0 > 0))[0]
    if voiced_idx.size == 0:
        return {"conv_jitter": np.nan, "conv_shimmer": np.nan, "resid_jitter": np.nan, "resid_shimmer": np.nan, "n_cycles": 0}

    runs = np.split(voiced_idx, np.where(np.diff(voiced_idx) != 1)[0] + 1)
    frame_times = np.arange(len(f0)) * HOP_LENGTH / PITCH_SR

    conv_j, conv_s, resid_j, resid_s, n_cycles = [], [], [], [], []
    for run in runs:
        if run.size < 3:
            continue
        marks = np.array(mark_pitch_periods(y8k, f0, run))
        if len(marks) < 4:
            continue
        conv = conventional_jitter_shimmer(marks, y8k, PITCH_SR)
        resid = residual_jitter_shimmer(marks, y8k, PITCH_SR, f0_smooth_times=frame_times, f0_smooth_values=f0)
        if not np.isnan(conv["jitter"]):
            conv_j.append(conv["jitter"]); conv_s.append(conv["shimmer"])
        if not np.isnan(resid["residual_jitter"]):
            resid_j.append(resid["residual_jitter"]); resid_s.append(resid["residual_shimmer"])
        n_cycles.append(conv["n_cycles"])

    return {
        "conv_jitter": float(np.mean(conv_j)) if conv_j else np.nan,
        "conv_shimmer": float(np.mean(conv_s)) if conv_s else np.nan,
        "resid_jitter": float(np.mean(resid_j)) if resid_j else np.nan,
        "resid_shimmer": float(np.mean(resid_s)) if resid_s else np.nan,
        "n_cycles": int(sum(n_cycles)),
    }


def extract_all_families(audio_path: str) -> dict:
    waveform, sr = load_waveform_float(audio_path)
    y16k = librosa.resample(waveform, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else waveform
    y8k = librosa.resample(waveform, orig_sr=sr, target_sr=PITCH_SR) if sr != PITCH_SR else waveform

    f0, _, _ = librosa.pyin(y8k, fmin=FMIN, fmax=FMAX, sr=PITCH_SR, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    frame_times = np.arange(len(f0)) * HOP_LENGTH / PITCH_SR

    row = {}

    # F006
    row.update({f"f006_{k}": v for k, v in extract_f006(y8k, f0).items()})
    row["f006_f0_motion_energy"] = f0_motion_energy(frame_times, f0)

    # F007 (CWT prosody) - 같은 pyin f0 궤적 재사용
    cwt_feats = extract_cwt_prosody_features(frame_times, f0)
    row.update({f"f007_{k}": v for k, v in cwt_feats.items() if k != "dominant_band"})
    row["f007_dominant_band"] = cwt_feats.get("dominant_band", "")

    # F008 (aperiodicity/MVF) - WORLD 자체 F0 필요, 독립적으로 재추정
    ap_feats = extract_aperiodicity_features(y16k, ANALYSIS_SR)
    row.update({f"f008_{k}": v for k, v in ap_feats.items()})

    # F009/F010 - 같은 pyin f0 궤적으로 harmonic trajectory를 한 번만 뽑아 공유
    traj = extract_harmonic_trajectory(y16k, ANALYSIS_SR, frame_times, f0, mode="adaptive", c=3, n_harmonics=N_HARMONICS)
    phase_feats = harmonic_phase_features(traj)
    mod_feats = harmonic_modulation_features(traj)
    row.update({f"f009_{k}": v for k, v in phase_feats.items()})
    row.update({f"f010_{k}": v for k, v in mod_feats.items()})

    return row


def _process_task(task: tuple) -> dict:
    wav_id, situation, audio_path = task
    try:
        row = extract_all_families(audio_path)
        row["error"] = ""
    except Exception as exc:  # noqa: BLE001
        row = {"error": str(exc)}
    row["wav_id"] = wav_id
    row["situation"] = situation
    return row


def stratified_sample(groups: dict, n_per_emotion: int, seed: int) -> list:
    rng = np.random.RandomState(seed)
    sample = []
    for situation, wav_ids in groups.items():
        existing = [w for w in wav_ids if os.path.exists(get_audio_path(w))]
        if not existing:
            continue
        chosen = rng.choice(existing, size=min(n_per_emotion, len(existing)), replace=False)
        sample.extend((wid, situation, get_audio_path(wid)) for wid in chosen)
    return sample


def run_extraction(n_per_emotion: int, seed: int = 42, out_name: str = "real_corpus_features.csv") -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = load_groups()
    sample = stratified_sample(groups, n_per_emotion, seed)
    print(f"표본: {len(sample)}개 파일 ({n_per_emotion}개/감정 x {len(groups)}개 감정)")

    rows = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_process_task, task) for task in sample]
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 50 == 0:
                elapsed = time.perf_counter() - start
                rate = done / elapsed
                print(f"  {done}/{len(sample)} 완료 ({rate:.2f}개/초, 남은 시간 {(len(sample) - done) / rate:.0f}초)")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    per_file_path = os.path.join(OUTPUT_DIR, out_name)
    with open(per_file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    n_errors = sum(1 for r in rows if r.get("error"))
    print(f"\n저장 완료: {per_file_path} (오류 {n_errors}/{len(rows)}개)")
    if n_errors:
        for r in rows:
            if r.get("error"):
                print(f"  {r['wav_id']}: {r['error']}")
    return per_file_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-emotion", type=int, default=150)
    parser.add_argument("--out-name", type=str, default="real_corpus_features.csv")
    args = parser.parse_args()
    run_extraction(args.n_per_emotion, out_name=args.out_name)
