"""
Cross-Correlation / Cosine / MSE를 행렬 연산으로 완전히 벡터화해서
그룹 내 전체 쌍(pair-wise) 유사도를 빠르게 계산한다.

더 큰 규모로 확장할 때는 similarity.py의 DATASETS 리스트에 오디오 폴더/CSV
경로만 추가해주면 이 스크립트는 그대로 재사용할 수 있다.
DTW는 O(n^2 * L^2)라 이 방식으로 벡터화하기 어려워 제외했다 (DTW_colab.ipynb에서
GPU 배치 처리로 별도 수행).
"""

import os
import csv
import time

import numpy as np

from similarity import RESAMPLE_LENGTH, get_audio_path, load_groups, load_waveform, preprocess

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "similarity", "vectorized")

# 정규화 교차상관을 탐색할 최대 지연(lag). 전체 길이(2*L-1)를 다 훑지 않고
# 리샘플링 길이의 ±5% 범위만 탐색해서 (2*MAX_LAG+1)번의 행렬곱으로 끝낸다.
MAX_LAG = max(1, RESAMPLE_LENGTH // 20)


def build_matrix(wav_ids: list) -> np.ndarray:
    vectors = [preprocess(load_waveform(wav_id)) for wav_id in wav_ids]
    return np.vstack(vectors)


def compute_cosine_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    x_norm = x / norms
    return x_norm @ x_norm.T


def compute_mse_matrix(x: np.ndarray) -> np.ndarray:
    length = x.shape[1]
    sq_norms = np.sum(x**2, axis=1)
    dot = x @ x.T
    mse = (sq_norms[:, None] + sq_norms[None, :] - 2 * dot) / length
    return np.clip(mse, 0, None)


def compute_cross_correlation_matrix(x: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    n, length = x.shape
    sq_norms = np.sum(x**2, axis=1)
    denom = np.sqrt(np.outer(sq_norms, sq_norms))
    denom[denom == 0] = 1e-12

    best = np.full((n, n), -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = x[:, : length - lag] if lag > 0 else x
            b = x[:, lag:]
        else:
            k = -lag
            a = x[:, k:]
            b = x[:, : length - k]

        normalized = (a @ b.T) / denom
        best = np.maximum(best, normalized)

    return best


def save_group_results(wav_ids: list, output_path: str) -> None:
    x = build_matrix(wav_ids)

    cosine_matrix = compute_cosine_matrix(x)
    mse_matrix = compute_mse_matrix(x)
    cross_corr_matrix = compute_cross_correlation_matrix(x)

    n = len(wav_ids)
    rows_i, rows_j = np.triu_indices(n, k=1)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file_1", "file_2", "cross_correlation", "cosine_similarity", "mse"])
        for i, j in zip(rows_i, rows_j):
            writer.writerow(
                [
                    wav_ids[i],
                    wav_ids[j],
                    float(cross_corr_matrix[i, j]),
                    float(cosine_matrix[i, j]),
                    float(mse_matrix[i, j]),
                ]
            )


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

        start = time.perf_counter()
        output_path = os.path.join(OUTPUT_DIR, f"{situation}.csv")
        save_group_results(wav_ids, output_path)
        elapsed = time.perf_counter() - start

        n = len(wav_ids)
        pair_count = n * (n - 1) // 2
        print(f"저장 완료: {situation}.csv ({n}개 파일, {pair_count}개 쌍, {elapsed:.2f}초)")


if __name__ == "__main__":
    main()
