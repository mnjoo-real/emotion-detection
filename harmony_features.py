"""
화성학적(harmony theory) 관점에서 감정별 음성 특징을 추출한다.

단성(monophonic) 음성에는 화음이 없지만, 아래 세 가지 방식으로 화성학 개념을
번역해서 적용한다:

1. 포먼트 러프니스(formant roughness): 한 프레임 안에 실제로 동시에 존재하는
   주파수들(포먼트 F1~F4, LPC로 추출)이 서로 단순 정수비(협화)에 가까운지
   복잡한 비율(불협화)인지를 Plomp-Levelt/Sethares 감각적 불협화도 모델로 계산.
   이게 한 목소리 안에서 실제로 "화음"에 가장 가까운 실체다.
2. 선율적 불협화도(melodic dissonance): 시간에 따른 피치(F0) 변화를 선율의
   음정 진행으로 보고, 인접한 유성음 프레임 사이의 음정을 같은 불협화도
   모델로 평가. 억양이 매끄럽게 움직이는지 거칠게 튀는지를 정량화.
3. 조성 안정성(tonal stability): 발화 전체의 유성음 피치를 파일 자신의
   중앙값 기준 12음계(pitch class)로 접어서 분포를 만들고, 그 분포의 엔트로피로
   "한 음역에 머무르는지 여러 음역을 오가는지"를 측정 (낮을수록 조성적으로 안정).

similarity.py의 load_groups()/get_audio_path()를 재사용해서 4차년도+5차년도
통합 데이터셋, 7개 감정 전부를 대상으로 한다. 파일 수가 많아 extract_features.py와
같은 방식(CPU 프로세스 풀 병렬 + 재실행 시 이어서 진행)으로 처리한다.
"""

import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from similarity import get_audio_path, load_groups

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "harmony")

# --- 포먼트(LPC) 설정 ---
FORMANT_SR = 12000          # 포먼트는 대부분 5kHz 이내라 Nyquist 6kHz면 충분
LPC_ORDER = 16
FORMANT_FRAME_LEN = int(0.025 * FORMANT_SR)  # 25ms
FORMANT_HOP = int(0.010 * FORMANT_SR)         # 10ms
MAX_FORMANTS = 4
FORMANT_FREQ_MIN = 90.0
FORMANT_BANDWIDTH_MAX = 400.0                  # 이보다 넓은 대역(=불명확한 공진)은 포먼트로 인정 안 함

# --- 피치(F0) 설정 - extract_features.py와 동일 파라미터로 일관성 유지 ---
PITCH_SR = 8000
FMIN = 65.0   # ~C2
FMAX = 2093.0  # ~C7

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PROGRESS_LOG_EVERY = 200

FIELDNAMES = [
    "wav_id",
    "situation",
    "f1_mean", "f2_mean", "f3_mean", "f4_mean",
    "formant_roughness_mean", "formant_roughness_std",
    "melodic_dissonance_mean", "melodic_dissonance_std",
    "pitch_class_entropy",
    "n_valid_formant_frames", "n_voiced_pitch_frames",
]


def sethares_dissonance(f1: np.ndarray, f2: np.ndarray, a1: float = 1.0, a2: float = 1.0) -> np.ndarray:
    """Plomp-Levelt 임계대역 러프니스에 기반한 Sethares(1993) 감각적 불협화도.
    0=매끄러움(협화), 클수록 거침(불협화/beating). 넓게 벌어지면 다시 0으로 수렴한다
    (러프니스는 임계대역 안에서만 발생하는 감각 현상이라 절대적 "음정 불협화"와는 다르다).
    """
    b1, b2 = 3.5, 5.75
    s1, s2 = 0.0207, 18.96
    dstar = 0.24
    fmin = np.minimum(f1, f2)
    fmax = np.maximum(f1, f2)
    df = fmax - fmin
    s = dstar / (s1 * fmin + s2)
    amin = min(a1, a2)
    return amin * (np.exp(-b1 * s * df) - np.exp(-b2 * s * df))


def levinson_durbin(r: np.ndarray, order: int):
    """자기상관 r(0..order)로부터 LPC 계수를 구한다. 반환: a (order+1,, a[0]=1), 예측오차 e."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    if e <= 0:
        return a, 0.0
    for i in range(1, order + 1):
        acc = r[i] + np.sum(a[1:i] * r[i - 1 : 0 : -1])
        k = -acc / e
        new_a = a.copy()
        new_a[i] = k
        new_a[1:i] += k * a[i - 1 : 0 : -1]
        a = new_a
        e *= 1 - k * k
        if e <= 0:
            break
    return a, e


def extract_formants_from_frame(frame: np.ndarray, sr: int, return_bandwidth: bool = False) -> list:
    windowed = frame * np.hamming(len(frame))
    pre = np.append(windowed[0], windowed[1:] - 0.97 * windowed[:-1])

    r = np.correlate(pre, pre, mode="full")[len(pre) - 1 :][: LPC_ORDER + 1]
    if r[0] <= 0:
        return []

    a, err = levinson_durbin(r, LPC_ORDER)
    if err <= 0 or np.any(np.isnan(a)):
        return []

    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]
    freqs = np.angle(roots) * (sr / (2 * np.pi))
    bandwidths = -0.5 * (sr / np.pi) * np.log(np.abs(roots).clip(1e-6, None))

    formants = [
        (f, bw) for f, bw in zip(freqs, bandwidths)
        if FORMANT_FREQ_MIN < f < sr / 2 - 100 and bw < FORMANT_BANDWIDTH_MAX
    ]
    formants.sort(key=lambda fb: fb[0])
    formants = formants[:MAX_FORMANTS]
    if return_bandwidth:
        return formants
    return [f for f, _ in formants]


def compute_formant_features(waveform: np.ndarray, orig_sr: int) -> dict:
    g = np.gcd(FORMANT_SR, orig_sr)
    resampled = resample_poly(waveform, FORMANT_SR // g, orig_sr // g)

    n_frames = 1 + max(0, (len(resampled) - FORMANT_FRAME_LEN) // FORMANT_HOP)

    per_frame_formants = []
    roughness_scores = []
    for i in range(n_frames):
        start = i * FORMANT_HOP
        frame = resampled[start : start + FORMANT_FRAME_LEN]
        if len(frame) < FORMANT_FRAME_LEN:
            continue
        formants = extract_formants_from_frame(frame, FORMANT_SR)
        if len(formants) < 2:
            continue
        per_frame_formants.append(formants)

        pair_scores = []
        for i1 in range(len(formants)):
            for i2 in range(i1 + 1, len(formants)):
                pair_scores.append(sethares_dissonance(formants[i1], formants[i2]))
        roughness_scores.append(float(np.mean(pair_scores)))

    result = {
        "f1_mean": 0.0, "f2_mean": 0.0, "f3_mean": 0.0, "f4_mean": 0.0,
        "formant_roughness_mean": 0.0, "formant_roughness_std": 0.0,
        "n_valid_formant_frames": len(per_frame_formants),
    }
    if per_frame_formants:
        for idx, key in enumerate(["f1_mean", "f2_mean", "f3_mean", "f4_mean"]):
            vals = [f[idx] for f in per_frame_formants if len(f) > idx]
            if vals:
                result[key] = float(np.mean(vals))
    if roughness_scores:
        result["formant_roughness_mean"] = float(np.mean(roughness_scores))
        result["formant_roughness_std"] = float(np.std(roughness_scores))

    return result


def compute_pitch_harmony_features(waveform: np.ndarray, orig_sr: int) -> dict:
    y_pitch = librosa.resample(waveform, orig_sr=orig_sr, target_sr=PITCH_SR) if orig_sr != PITCH_SR else waveform
    f0, voiced_flag, _ = librosa.pyin(y_pitch, fmin=FMIN, fmax=FMAX, sr=PITCH_SR)

    voiced_idx = np.where(~np.isnan(f0))[0]
    result = {
        "melodic_dissonance_mean": 0.0,
        "melodic_dissonance_std": 0.0,
        "pitch_class_entropy": 0.0,
        "n_voiced_pitch_frames": int(voiced_idx.size),
    }
    if voiced_idx.size == 0:
        return result

    voiced_f0 = f0[voiced_idx]

    # 선율적 불협화도: '연속된' 유성음 프레임 사이에서만 계산 (프레임 인덱스가
    # 바로 이어질 때만 - 중간에 무성음 구간이 끼면 그 도약은 선율적 진행이 아니므로 제외)
    consecutive_pairs = []
    for k in range(len(voiced_idx) - 1):
        if voiced_idx[k + 1] == voiced_idx[k] + 1:
            consecutive_pairs.append((f0[voiced_idx[k]], f0[voiced_idx[k + 1]]))
    if consecutive_pairs:
        pairs = np.array(consecutive_pairs)
        scores = sethares_dissonance(pairs[:, 0], pairs[:, 1])
        result["melodic_dissonance_mean"] = float(np.mean(scores))
        result["melodic_dissonance_std"] = float(np.std(scores))

    # 조성 안정성: 파일 자신의 중앙값을 기준(0 cents)으로 삼아 옥타브를 접어
    # 12개 음계 구간(pitch class)에 대한 분포 엔트로피를 구한다.
    ref = np.median(voiced_f0)
    cents = 1200 * np.log2(voiced_f0 / ref)
    pitch_class = np.mod(cents, 1200)
    hist, _ = np.histogram(pitch_class, bins=12, range=(0, 1200))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    result["pitch_class_entropy"] = float(-np.sum(probs * np.log2(probs)))

    return result


def extract_harmony_features_from_path(audio_path: str) -> dict:
    sr, waveform = wavfile.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float64)
    if waveform.size and np.abs(waveform).max() > 1.0:
        waveform = waveform / 32768.0

    row = {}
    row.update(compute_formant_features(waveform, sr))
    row.update(compute_pitch_harmony_features(waveform, sr))
    return row


def extract_harmony_features(wav_id: str) -> dict:
    return extract_harmony_features_from_path(get_audio_path(wav_id))


def _process_task(task: tuple) -> tuple:
    wav_id, situation, audio_path = task
    try:
        features = extract_harmony_features_from_path(audio_path)
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

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_harmony.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_emotion.csv")

    extract_all(groups, per_file_path)
    save_summary_by_emotion(per_file_path, summary_path)
    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
