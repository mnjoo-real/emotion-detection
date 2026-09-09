"""SP-V2-012 §19-21: F008/F009가 F010 위에 conditional value를 주는지, family간
redundancy matrix, situation vs majority_vote cross-label consistency.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signal_v2.evaluation.real_corpus_common import classifier_ablation
from signal_v2.evaluation.sp012_scaling_common import load_level_dataset, numeric_cols

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_CORPUS_CSV = os.path.join(BASE_DIR, "output", "signal_v2", "real_corpus", "real_corpus_features.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "SP-V2-012")

COMBOS = {
    "F010_alone": ["F010"],
    "F010_F008": ["F010", "F008"],
    "F010_F009": ["F010", "F009"],
    "F010_F008_F009": ["F010", "F008", "F009"],
}


def run_conditional_value(label: str = "situation") -> dict:
    data = load_level_dataset(REAL_CORPUS_CSV)
    df = data["df"]
    if label == "majority_vote":
        df = df.dropna(subset=["majority_vote"])
    y = df[label].values

    family_cols = {"F008": numeric_cols(df, "f008_"), "F009": numeric_cols(df, "f009_"), "F010": numeric_cols(df, "f010_")}

    # majority_vote일 때 df가 dropna로 줄어들 수 있으므로 wav_id로 baseline을 다시 정렬
    from signal_v2.evaluation.sp012_scaling_common import _baseline
    cache = _baseline()
    X_baseline = np.array([cache["base_lookup"][w] for w in df["wav_id"]])

    print(f"\n=== §19 conditional value of F008/F009 on top of F010 ({label}, n={len(df)}) ===")
    results = {}
    for combo_name, fams in COMBOS.items():
        cols = [c for fam in fams for c in family_cols[fam]]
        X_new = df[cols].fillna(0.0).values
        res = classifier_ablation(X_baseline, X_new, y)
        results[combo_name] = res
        print(f"  {combo_name:<20} delta={res['delta_mean']:+.4f} ({res['n_folds_improved']}/{res['n_folds']} folds)")

    f010_alone_delta = results["F010_alone"]["delta_mean"]
    for combo_name in ["F010_F008", "F010_F009", "F010_F008_F009"]:
        marginal = results[combo_name]["delta_mean"] - f010_alone_delta
        results[combo_name]["marginal_over_f010_alone"] = marginal
        print(f"  {combo_name} vs F010 alone: marginal={marginal:+.4f}")

    return results


def redundancy_matrix() -> pd.DataFrame:
    """family 대표 feature들 사이 Spearman correlation matrix (§20)."""
    data = load_level_dataset(REAL_CORPUS_CSV)
    df = data["df"]

    representative = {
        "F008_low_band_periodicity_mean": "f008_low_band_periodicity_mean",
        "F008_mid_band_periodicity_std": "f008_mid_band_periodicity_std",
        "F008_aperiodicity_slope": "f008_aperiodicity_spectral_slope_mean",
        "F009_mean_phase_coherence": "f009_mean_phase_coherence",
        "F009_h3_circular_variance": "f009_h3_phase_circular_variance",
        "F010_inter_harmonic_coherence": "f010_inter_harmonic_modulation_coherence",
        "F010_h1_entropy": "f010_h1_modulation_entropy",
    }
    cols = list(representative.values())
    labels = list(representative.keys())
    sub = df[cols].astype(float)

    n = len(cols)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            valid = np.isfinite(sub.iloc[:, i]) & np.isfinite(sub.iloc[:, j])
            if valid.sum() < 5:
                mat[i, j] = np.nan
                continue
            r, _ = stats.spearmanr(sub.iloc[:, i][valid], sub.iloc[:, j][valid])
            mat[i, j] = r

    return pd.DataFrame(mat, index=labels, columns=labels)


def cross_label_consistency() -> pd.DataFrame:
    """§21: situation vs majority_vote에서 family별 top feature effect size 비교."""
    data = load_level_dataset(REAL_CORPUS_CSV)
    df = data["df"]
    df_mv = df.dropna(subset=["majority_vote"])

    from signal_v2.evaluation.real_corpus_common import emotion_association_table
    rows = []
    for fam, prefix in [("F008", "f008_"), ("F009", "f009_"), ("F010", "f010_")]:
        cols = numeric_cols(df, prefix)
        sit_table = emotion_association_table(df, cols, "situation")
        mv_table = emotion_association_table(df_mv, cols, "majority_vote")
        top_sit = sit_table.iloc[0]
        top_mv = mv_table.iloc[0]
        rows.append({
            "family": fam,
            "top_situation_feature": top_sit["feature"], "situation_eps2": top_sit["epsilon_sq"],
            "top_majority_vote_feature": top_mv["feature"], "majority_vote_eps2": top_mv["epsilon_sq"],
        })
    return pd.DataFrame(rows)


def run() -> dict:
    results = {}
    results["conditional_value_situation"] = run_conditional_value("situation")
    results["conditional_value_majority_vote"] = run_conditional_value("majority_vote")

    print("\n=== §20 redundancy matrix (representative features) ===")
    redund = redundancy_matrix()
    print(redund.round(3).to_string())
    results["redundancy_matrix"] = redund.round(3).to_dict()

    print("\n=== §21 cross-label consistency ===")
    cross = cross_label_consistency()
    print(cross.to_string(index=False))
    results["cross_label_consistency"] = cross.to_dict("records")

    with open(os.path.join(OUTPUT_DIR, "complementarity.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n저장 완료: {os.path.join(OUTPUT_DIR, 'complementarity.json')}")
    return results


if __name__ == "__main__":
    run()
