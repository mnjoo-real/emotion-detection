"""
음악의 3요소 중 '선율(melody)' - 음높이(pitch)가 리듬(rhythm, 길이/간격)을 타고
연속적으로 이어지는 흐름 - 을 feature로 추출한다.

harmony_features.py가 화성(동시에 존재하는 주파수들의 협화/불협화), melodic_profile.py가
프레임 단위 음정(pitch interval)의 크기/방향만 봤다면, 여기서는 유성음 F0를 먼저
'음(note)' 단위로 묶어서 각 음의 길이(duration)와 음 사이 시간 간격(IOI, inter-onset
interval)이라는 리듬 정보를 선율 분석에 도입한다:

1. 음 분리(note segmentation): 연속된 유성음 프레임 중 피치가 일정 범위(반음 1개)
   안에서 유지되는 구간을 하나의 '음'으로 묶는다. 무성음(자음/무음)으로 끊기거나
   피치가 크게 튀면 새 음이 시작된 것으로 본다 - 단성 멜로디 채보(monophonic note
   transcription)의 표준적인 단순화 방식.
2. 리듬: 음 하나의 길이(note duration), 그리고 음이 시작되는 시점들 사이의 간격
   (IOI) - 무성음 구간(쉼표에 해당)까지 포함해서 실제 "말의 리듬"을 반영한다.
   길이가 일정하면(변동계수 낮음) 규칙적인 리듬, 들쭉날쭉하면 불규칙한 리듬.
3. 선율: 연속된 음들 사이의 반음 음정(멜로디 진행) 통계.
4. 리듬-선율 결합: 음 높이와 음 길이 사이의 상관 - 높은 음이 유독 길게/짧게
   끄는 경향이 있는지(일종의 억양의 아고긱 강세) - '선율' 정의 그대로
   음높이와 리듬이 함께 나타나는 패턴을 직접 겨냥한 지표.

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

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "melody_rhythm")

# librosa.pyin 기본값(frame_length=2048, hop_length=frame_length//4)과 동일하게 맞춰서
# 다른 스크립트들과 프레임 시간 해상도를 일치시킨다.
HOP_LENGTH = 512
FRAME_TIME = HOP_LENGTH / PITCH_SR  # 64ms

NOTE_SEMITONE_THRESHOLD = 1.0   # 이 값(반음) 넘게 벗어나면 새 음으로 분리
MIN_NOTE_FRAMES = 2             # 이보다 짧으면 노이즈성 떨림으로 보고 음으로 인정 안 함
FLAT_THRESHOLD = 0.5            # 음정 크기가 이 값(반음) 이내면 '반복(같은 음)'으로 간주

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PROGRESS_LOG_EVERY = 200

FIELDNAMES = [
    "wav_id", "situation",
    "n_notes", "notes_per_sec",
    "note_dur_mean", "note_dur_std", "note_dur_cv",
    "ioi_mean", "ioi_std", "ioi_cv",
    "note_interval_semitone_mean", "note_interval_semitone_std",
    "pct_notes_rising", "pct_notes_falling", "pct_notes_repeated",
    "duration_pitch_corr",
    "n_voiced_pitch_frames",
]


def segment_notes(f0: np.ndarray, voiced_idx: np.ndarray) -> list:
    """연속된 유성음 프레임을 피치 안정 구간(음) 단위로 묶는다.
    반환: [{"start": frame_idx, "pitches": [...]}]"""
    notes = []
    current = None
    prev_idx = None

    for idx in voiced_idx:
        pitch = f0[idx]
        is_continuation = (
            current is not None
            and prev_idx is not None
            and idx == prev_idx + 1
            and abs(12 * np.log2(pitch / np.mean(current["pitches"]))) <= NOTE_SEMITONE_THRESHOLD
        )
        if is_continuation:
            current["pitches"].append(pitch)
        else:
            if current is not None:
                notes.append(current)
            current = {"start": idx, "pitches": [pitch]}
        prev_idx = idx

    if current is not None:
        notes.append(current)

    return [n for n in notes if len(n["pitches"]) >= MIN_NOTE_FRAMES]


def compute_melody_rhythm_features(waveform: np.ndarray, orig_sr: int) -> dict:
    y_pitch = librosa.resample(waveform, orig_sr=orig_sr, target_sr=PITCH_SR) if orig_sr != PITCH_SR else waveform
    f0, _, _ = librosa.pyin(y_pitch, fmin=FMIN, fmax=FMAX, sr=PITCH_SR)

    voiced_idx = np.where(~np.isnan(f0))[0]
    result = {
        "n_notes": 0, "notes_per_sec": 0.0,
        "note_dur_mean": 0.0, "note_dur_std": 0.0, "note_dur_cv": 0.0,
        "ioi_mean": 0.0, "ioi_std": 0.0, "ioi_cv": 0.0,
        "note_interval_semitone_mean": 0.0, "note_interval_semitone_std": 0.0,
        "pct_notes_rising": 0.0, "pct_notes_falling": 0.0, "pct_notes_repeated": 0.0,
        "duration_pitch_corr": 0.0,
        "n_voiced_pitch_frames": int(voiced_idx.size),
    }
    if f0.size == 0:
        return result

    notes = segment_notes(f0, voiced_idx)
    n_notes = len(notes)
    result["n_notes"] = n_notes
    total_duration_sec = f0.size * FRAME_TIME
    if total_duration_sec > 0:
        result["notes_per_sec"] = n_notes / total_duration_sec
    if n_notes == 0:
        return result

    onsets = np.array([n["start"] * FRAME_TIME for n in notes])
    durations = np.array([len(n["pitches"]) * FRAME_TIME for n in notes])
    mean_pitches = np.array([np.mean(n["pitches"]) for n in notes])

    result["note_dur_mean"] = float(durations.mean())
    result["note_dur_std"] = float(durations.std())
    if result["note_dur_mean"] > 0:
        result["note_dur_cv"] = result["note_dur_std"] / result["note_dur_mean"]

    if n_notes >= 2:
        iois = np.diff(onsets)
        result["ioi_mean"] = float(iois.mean())
        result["ioi_std"] = float(iois.std())
        if result["ioi_mean"] > 0:
            result["ioi_cv"] = result["ioi_std"] / result["ioi_mean"]

        note_intervals = 12 * np.log2(mean_pitches[1:] / mean_pitches[:-1])
        result["note_interval_semitone_mean"] = float(np.abs(note_intervals).mean())
        result["note_interval_semitone_std"] = float(np.abs(note_intervals).std())
        result["pct_notes_rising"] = float(np.mean(note_intervals > FLAT_THRESHOLD))
        result["pct_notes_falling"] = float(np.mean(note_intervals < -FLAT_THRESHOLD))
        result["pct_notes_repeated"] = float(np.mean(np.abs(note_intervals) <= FLAT_THRESHOLD))

    if n_notes >= 3 and durations.std() > 0 and mean_pitches.std() > 0:
        corr = np.corrcoef(mean_pitches, durations)[0, 1]
        if not np.isnan(corr):
            result["duration_pitch_corr"] = float(corr)

    return result


def extract_melody_rhythm_features_from_path(audio_path: str) -> dict:
    sr, waveform = wavfile.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float64)
    if waveform.size and np.abs(waveform).max() > 1.0:
        waveform = waveform / 32768.0

    return compute_melody_rhythm_features(waveform, sr)


def extract_melody_rhythm_features(wav_id: str) -> dict:
    return extract_melody_rhythm_features_from_path(get_audio_path(wav_id))


def _process_task(task: tuple) -> tuple:
    wav_id, situation, audio_path = task
    try:
        features = extract_melody_rhythm_features_from_path(audio_path)
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

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_melody_rhythm.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_emotion.csv")

    extract_all(groups, per_file_path)
    save_summary_by_emotion(per_file_path, summary_path)
    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
