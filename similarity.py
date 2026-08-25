import os
import csv
from itertools import combinations

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, resample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 4차년도/5차년도 데이터셋을 통합해서 처리한다. 각 데이터셋은 오디오 폴더와
# 매칭되는 CSV(wav_id -> 상황)를 갖는다.
DATASETS = [
    {
        "audio_dir": os.path.join(BASE_DIR, "4차년도"),
        "sample_csv": os.path.join(BASE_DIR, "4차년도.csv"),
        "encoding": "cp949",
    },
    {
        "audio_dir": os.path.join(BASE_DIR, "5차년도_2차"),
        "sample_csv": os.path.join(BASE_DIR, "5차년도_2차.csv"),
        "encoding": "cp949",
    },
]

# 데이터셋마다 상황 라벨 표기가 달라서(예: 4차년도 "anger"/"sad" vs
# 5차년도 "angry"/"sadness") 하나로 맞춰준다.
SITUATION_ALIASES = {
    "anger": "angry",
    "sad": "sadness",
}

OUTPUT_DIR = os.path.join(BASE_DIR, "output", "similarity")

RESAMPLE_LENGTH = 1000

# load_groups()가 채우는, wav_id -> 해당 오디오가 들어있는 데이터셋의 audio_dir.
_WAV_ID_TO_AUDIO_DIR: dict = {}


def load_groups() -> dict:
    groups: dict = {}
    for dataset in DATASETS:
        with open(dataset["sample_csv"], encoding=dataset["encoding"], newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                situation = SITUATION_ALIASES.get(row["상황"].strip(), row["상황"].strip())
                wav_id = row["wav_id"].strip()
                groups.setdefault(situation, []).append(wav_id)
                _WAV_ID_TO_AUDIO_DIR[wav_id] = dataset["audio_dir"]
    return groups


def get_audio_path(wav_id: str) -> str:
    audio_dir = _WAV_ID_TO_AUDIO_DIR.get(wav_id)
    if audio_dir is None:
        raise KeyError(f"'{wav_id}'의 오디오 폴더를 알 수 없습니다. load_groups()를 먼저 호출했는지 확인하세요.")
    return os.path.join(audio_dir, f"{wav_id}.wav")


def load_waveform(wav_id: str) -> np.ndarray:
    wav_path = get_audio_path(wav_id)
    _, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float64)


def preprocess(data: np.ndarray, target_len: int = RESAMPLE_LENGTH) -> np.ndarray:
    # 파일마다 길이가 달라서 동일한 길이로 리샘플링하고,
    # 절대 음량 차이를 제거하기 위해 z-score 정규화(평균 0, 표준편차 1)한다.
    resampled = resample(data, target_len)
    std = resampled.std()
    if std == 0:
        return resampled - resampled.mean()
    return (resampled - resampled.mean()) / std


def cross_correlation_similarity(x: np.ndarray, y: np.ndarray) -> float:
    corr = correlate(x, y, mode="full")
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denom == 0:
        return 0.0
    return float(np.max(corr) / denom)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def mean_squared_error(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x - y) ** 2))


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            cost = abs(xi - y[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # 경로 길이로 나눠 신호 길이에 따른 편향을 줄인다.
    return float(dtw[n, m] / (n + m))


def compare_pair(wav_id_1: str, wav_id_2: str) -> dict:
    x = preprocess(load_waveform(wav_id_1))
    y = preprocess(load_waveform(wav_id_2))

    return {
        "file_1": wav_id_1,
        "file_2": wav_id_2,
        "cross_correlation": cross_correlation_similarity(x, y),
        "cosine_similarity": cosine_similarity(x, y),
        "mse": mean_squared_error(x, y),
        "dtw_distance": dtw_distance(x, y),
    }


def save_group_results(wav_ids: list, output_path: str) -> None:
    fieldnames = ["file_1", "file_2", "cross_correlation", "cosine_similarity", "mse", "dtw_distance"]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for wav_id_1, wav_id_2 in combinations(wav_ids, 2):
            writer.writerow(compare_pair(wav_id_1, wav_id_2))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = load_groups()

    for situation, wav_ids in groups.items():
        existing = [wid for wid in wav_ids if os.path.exists(get_audio_path(wid))]
        skipped = len(wav_ids) - len(existing)
        if skipped:
            print(f"경고: '{situation}' 상황에서 오디오 파일이 없는 {skipped}개 항목을 건너뜀.")
        wav_ids = existing

        if len(wav_ids) < 2:
            print(f"경고: '{situation}' 상황에 해당하는 파일이 2개 미만이라 건너뜀.")
            continue

        file_name = f"{situation}.csv"
        output_path = os.path.join(OUTPUT_DIR, file_name)
        save_group_results(wav_ids, output_path)
        print(f"저장 완료: {file_name} ({len(wav_ids)}개 파일, {len(list(combinations(wav_ids, 2)))}개 쌍)")


if __name__ == "__main__":
    main()
