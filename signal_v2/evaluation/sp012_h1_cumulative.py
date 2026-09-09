"""SP-V2-012 overnight §10-11: H1-specific 결과를 더 엄밀히 검증 + harmonic-order별
reliability와 signal을 분리한다. 이미 추출된 real_corpus_features.csv(F010 h1-h5
전부 있음) + aux_features_original_1400.csv(generic envelope)만 있으면 되므로 새
오디오 처리 없이 classifier 비교만 수행한다 - 가볍다.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dissonance_tonality_stats import epsilon_squared
from signal_v2.evaluation.real_corpus_common import classifier_ablation
from signal_v2.evaluation.sp012_scaling_common import load_level_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_CORPUS_CSV = os.path.join(BASE_DIR, "output", "signal_v2", "real_corpus", "real_corpus_features.csv")
AUX_CSV = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012", "aux_features_original_1400.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")

ENVMOD_COLS = ["envmod_dominant_freq", "envmod_low_rate_energy", "envmod_high_rate_energy",
               "envmod_centroid", "envmod_bandwidth", "envmod_entropy"]


def emotion_eps2(values: np.ndarray, labels: np.ndarray) -> float:
    valid = np.isfinite(values)
    v, l = values[valid], labels[valid]
    groups = [v[l == e] for e in np.unique(l)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan
    n = sum(len(g) for g in groups)
    k = len(groups)
    h, _ = stats.kruskal(*groups)
    return epsilon_squared(h, n, k)


def run() -> dict:
    data = load_level_dataset(REAL_CORPUS_CSV)
    df = data["df"]
    aux = pd.read_csv(AUX_CSV, encoding="utf-8-sig")
    if "error" in aux.columns:
        aux = aux[aux["error"].fillna("") == ""]
    df = df.merge(aux[["wav_id"] + ENVMOD_COLS], on="wav_id", how="inner")

    from signal_v2.evaluation.sp012_scaling_common import _baseline
    cache = _baseline()
    X_baseline = np.array([cache["base_lookup"][w] for w in df["wav_id"]])
    y = df["situation"].values

    print(f"분석 대상: {len(df)}개 파일")

    # 각 harmonic order별 개별(6-dim) 비교 + generic envelope baseline
    results = {"per_harmonic": {}, "cumulative": {}}

    print("\n=== §10 harmonic order별 6-dim 비교 (dimension-matched) ===")
    X_env = df[ENVMOD_COLS].fillna(0.0).values
    env_result = classifier_ablation(X_baseline, X_env, y)
    print(f"  generic_envelope (6d): delta={env_result['delta_mean']:+.4f} ({env_result['n_folds_improved']}/5)")
    results["per_harmonic"]["generic_envelope"] = env_result

    h_cols_all = {}
    for h in range(1, 6):
        h_cols = [c for c in df.columns if c.startswith(f"f010_h{h}_") and pd.api.types.is_numeric_dtype(df[c])]
        h_cols_all[h] = h_cols
        X_h = df[h_cols].fillna(0.0).values
        res = classifier_ablation(X_baseline, X_h, y)
        top_eps = max((emotion_eps2(df[c].values, y) for c in h_cols), default=np.nan)
        print(f"  H{h} ({len(h_cols)}d): delta={res['delta_mean']:+.4f} ({res['n_folds_improved']}/5), max_eps2={top_eps:.4f}")
        results["per_harmonic"][f"H{h}"] = {"ablation": res, "max_epsilon_sq": top_eps, "n_dims": len(h_cols)}

    print("\n=== §10 cumulative ablation (H1, H1+H2, ..., all) ===")
    cumulative_cols = []
    for h in range(1, 6):
        cumulative_cols += h_cols_all[h]
        X_cum = df[cumulative_cols].fillna(0.0).values
        res = classifier_ablation(X_baseline, X_cum, y)
        label = f"H1-H{h}" if h > 1 else "H1"
        print(f"  {label} ({len(cumulative_cols)}d): delta={res['delta_mean']:+.4f} ({res['n_folds_improved']}/5)")
        results["cumulative"][label] = res

    with open(os.path.join(OUTPUT_DIR, "h1_cumulative_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n저장 완료: {os.path.join(OUTPUT_DIR, 'h1_cumulative_analysis.json')}")
    return results


if __name__ == "__main__":
    run()
