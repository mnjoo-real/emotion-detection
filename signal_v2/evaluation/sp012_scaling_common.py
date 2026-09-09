"""SP-V2-012: 여러 N(sample size)에서 재사용할 데이터 로딩/결합 유틸리티."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from emotion_classifier import load_combined_features, load_majority_vote_labels
from signal_v2.evaluation.run_family_evaluation import load_gender_map

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SP012_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")

FAMILY_PREFIXES = {"F008": "f008_", "F009": "f009_", "F010": "f010_"}


def numeric_cols(df: pd.DataFrame, prefix: str) -> list:
    return [c for c in df.columns if c.startswith(prefix) and pd.api.types.is_numeric_dtype(df[c])]


_BASELINE_CACHE = None


def _baseline():
    global _BASELINE_CACHE
    if _BASELINE_CACHE is None:
        wav_ids, X, y_situation, feature_names = load_combined_features()
        mv = load_majority_vote_labels()
        gender_map = load_gender_map()
        _BASELINE_CACHE = {
            "base_lookup": {w: X[i] for i, w in enumerate(wav_ids)},
            "situation_lookup": {w: y_situation[i] for i, w in enumerate(wav_ids)},
            "mv_lookup": mv, "gender_map": gender_map, "feature_names": feature_names,
        }
    return _BASELINE_CACHE


def load_level_dataset(csv_path: str, n_per_emotion: int = None, seed: int = None) -> dict:
    """csv_path(어떤 seed로 뽑은 feature CSV)를 읽고 baseline 117-feature/
    situation/majority_vote/gender와 합친다.

    n_per_emotion이 주어지면 더 작은 nested N level로 자른다 - 단, CSV 파일 안의
    행 순서는 ProcessPoolExecutor의 완료 순서라 permutation 순서와 다르다(병렬
    처리라 제출 순서 != 완료 순서). 그래서 단순히 "앞 N개"를 자르면 nested subset이
    보장되지 않는다. 대신 extract_sp012_features.nested_permutation(seed)을 다시
    생성해서(같은 seed면 결정적으로 동일한 순열이 나옴) 그 순열의 앞 N개 wav_id
    집합을 얻고, CSV에서 그 wav_id들만 골라낸다 - 이렇게 하면 CSV 저장 순서와
    무관하게 진짜 nested subset이 된다."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "error" in df.columns:
        df = df[df["error"].fillna("") == ""]

    if n_per_emotion is not None:
        if seed is None:
            raise ValueError("n_per_emotion으로 자르려면 seed가 필요합니다 (nested permutation 재현용)")
        from signal_v2.evaluation.extract_sp012_features import nested_permutation
        from similarity import load_groups
        permuted = nested_permutation(load_groups(), seed)
        keep_ids = set()
        for situation, ids in permuted.items():
            keep_ids.update(ids[:n_per_emotion])
        df = df[df["wav_id"].isin(keep_ids)]

    cache = _baseline()
    common = [w for w in df["wav_id"] if w in cache["base_lookup"]]
    df = df[df["wav_id"].isin(common)].set_index("wav_id").loc[common].reset_index()

    df["gender"] = df["wav_id"].map(cache["gender_map"])
    df["majority_vote"] = df["wav_id"].map(cache["mv_lookup"])
    X_baseline = np.array([cache["base_lookup"][w] for w in df["wav_id"]])
    y_situation = np.array([cache["situation_lookup"][w] for w in df["wav_id"]])

    return {"df": df, "X_baseline": X_baseline, "y_situation": y_situation,
            "feature_names": cache["feature_names"]}
