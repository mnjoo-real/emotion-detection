"""Stage 2-6: F006-F010 각 family에 대해 A-E evaluation을 전부 수행하고 결과를
output/signal_v2/real_corpus/<family>/에 저장한다.

각 family를 정확히 같은 절차로 평가한다(브리핑 §4의 요구):
  A. feature quality (NaN/Inf/zero/constant, 분포 요약)
  B. emotion association (situation/majority_vote 각각, effect size 우선)
  C. gender confound (Type A/B/C 분류)
  D. redundancy vs 기존 117 feature
  E. classifier ablation (baseline vs baseline+family, 동일 StratifiedKFold split)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from emotion_classifier import load_combined_features, load_majority_vote_labels
from signal_v2.evaluation.real_corpus_common import (
    classifier_ablation, emotion_association_table, feature_quality_report,
    gender_confound_table, redundancy_table, save_table,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_CORPUS_DIR = os.path.join(BASE_DIR, "output", "signal_v2", "real_corpus")
FEATURES_CSV = os.path.join(REAL_CORPUS_DIR, "real_corpus_features.csv")

FAMILY_PREFIXES = {
    "F006": "f006_", "F007": "f007_", "F008": "f008_", "F009": "f009_", "F010": "f010_",
}


def load_gender_map() -> dict:
    import csv
    gender_map = {}
    for path, encoding in [
        (os.path.join(BASE_DIR, "4차년도.csv"), "cp949"),
        (os.path.join(BASE_DIR, "5차년도_2차.csv"), "cp949"),
    ]:
        with open(path, encoding=encoding, newline="") as f:
            for row in csv.DictReader(f):
                gender_map[row["wav_id"].strip()] = row["성별"].strip()
    return gender_map


def numeric_feature_cols(df: pd.DataFrame, prefix: str) -> list:
    cols = [c for c in df.columns if c.startswith(prefix)]
    numeric = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
    return numeric


def build_merged_dataset() -> tuple:
    """real_corpus_features.csv + gender + majority_vote + baseline 117-feature를
    wav_id로 합친다. 반환: (merged_df, baseline_feature_names, baseline_X_full_dict)."""
    real_df = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    # 빈 문자열로 저장된 "error" 칼럼을 pandas가 읽으면 NaN이 된다(따옴표 없는 빈 필드는
    # 기본적으로 결측으로 해석됨) - "" 비교만으로는 전부 걸러지므로 fillna로 보정한다.
    if "error" in real_df.columns:
        real_df = real_df[real_df["error"].fillna("") == ""].copy()
    real_df = real_df.set_index("wav_id")

    gender_map = load_gender_map()
    mv_labels = load_majority_vote_labels()

    wav_ids, X_base, y_situation, feature_names = load_combined_features()
    baseline_lookup = {wid: X_base[i] for i, wid in enumerate(wav_ids)}
    situation_lookup = {wid: y_situation[i] for i, wid in enumerate(wav_ids)}

    common_ids = [wid for wid in real_df.index if wid in baseline_lookup]
    print(f"real_corpus 표본 {len(real_df)}개 중 baseline과 공통: {len(common_ids)}개")

    real_df = real_df.loc[common_ids]
    real_df["gender"] = [gender_map.get(wid, "") for wid in common_ids]
    real_df["situation"] = [situation_lookup[wid] for wid in common_ids]
    real_df["majority_vote"] = [mv_labels.get(wid) for wid in common_ids]

    baseline_X = np.array([baseline_lookup[wid] for wid in common_ids])

    return real_df, feature_names, baseline_X, common_ids


def evaluate_family(family: str, real_df: pd.DataFrame, feature_names: list, baseline_X: np.ndarray,
                     common_ids: list) -> dict:
    prefix = FAMILY_PREFIXES[family]
    feature_cols = numeric_feature_cols(real_df, prefix)
    print(f"\n{'=' * 70}\n{family} ({prefix}) - {len(feature_cols)}개 feature\n{'=' * 70}")

    out_dir = os.path.join(REAL_CORPUS_DIR, family)
    os.makedirs(out_dir, exist_ok=True)

    # A. feature quality
    quality = feature_quality_report(real_df, feature_cols)
    save_table(quality, os.path.join(out_dir, "A_feature_quality.csv"))
    n_constant = int(quality["is_constant"].sum())
    n_high_nan = int((quality["nan_pct"] > 20).sum())
    print(f"[A] constant features: {n_constant}/{len(feature_cols)}, NaN>20%: {n_high_nan}/{len(feature_cols)}")

    # B. emotion association (situation + majority_vote)
    b_situation = emotion_association_table(real_df, feature_cols, "situation")
    real_df_mv = real_df.dropna(subset=["majority_vote"])
    b_mv = emotion_association_table(real_df_mv, feature_cols, "majority_vote") if len(real_df_mv) > 0 else pd.DataFrame()
    save_table(b_situation, os.path.join(out_dir, "B_emotion_association_situation.csv"))
    save_table(b_mv, os.path.join(out_dir, "B_emotion_association_majority_vote.csv"))
    top_situation = b_situation.iloc[0] if len(b_situation) else None
    if top_situation is not None:
        print(f"[B] top feature (situation): {top_situation['feature']} eps^2={top_situation['epsilon_sq']:.4f} ({top_situation['effect_label']})")

    # C. gender confound
    c_confound = gender_confound_table(real_df, feature_cols, emotion_col="situation")
    save_table(c_confound, os.path.join(out_dir, "C_gender_confound.csv"))
    type_counts = c_confound["type"].value_counts().to_dict()
    print(f"[C] type distribution: {type_counts}")

    # D. redundancy vs existing 117
    d_redundancy = redundancy_table(real_df[feature_cols], pd.DataFrame(baseline_X, columns=feature_names, index=real_df.index),
                                      feature_cols, feature_names)
    save_table(d_redundancy, os.path.join(out_dir, "D_redundancy.csv"))
    high_redundancy = int((d_redundancy["max_abs_rho"] > 0.9).sum())
    print(f"[D] features with max|rho|>0.9 vs existing: {high_redundancy}/{len(feature_cols)}")

    # E. classifier ablation (situation + majority_vote)
    X_new = real_df[feature_cols].fillna(0.0).values
    y_situation = real_df["situation"].values

    e_situation = classifier_ablation(baseline_X, X_new, y_situation)
    print(f"[E-situation] baseline={e_situation['baseline_f1_mean']:.4f} "
          f"combined={e_situation['combined_f1_mean']:.4f} "
          f"delta={e_situation['delta_mean']:+.4f} ({e_situation['n_folds_improved']}/{e_situation['n_folds']} folds improved)")

    e_mv = None
    if len(real_df_mv) >= 50:
        idx_mv = [i for i, wid in enumerate(common_ids) if wid in real_df_mv.index]
        X_base_mv = baseline_X[idx_mv]
        X_new_mv = real_df_mv[feature_cols].fillna(0.0).values
        y_mv = real_df_mv["majority_vote"].values
        e_mv = classifier_ablation(X_base_mv, X_new_mv, y_mv)
        print(f"[E-majority_vote] baseline={e_mv['baseline_f1_mean']:.4f} "
              f"combined={e_mv['combined_f1_mean']:.4f} "
              f"delta={e_mv['delta_mean']:+.4f} ({e_mv['n_folds_improved']}/{e_mv['n_folds']} folds improved)")

    result = {
        "family": family, "n_features": len(feature_cols), "feature_cols": feature_cols,
        "n_constant": n_constant, "n_high_nan": n_high_nan,
        "top_situation_feature": top_situation["feature"] if top_situation is not None else None,
        "top_situation_eps2": float(top_situation["epsilon_sq"]) if top_situation is not None else None,
        "type_counts": type_counts, "n_high_redundancy": high_redundancy,
        "classifier_situation": e_situation, "classifier_majority_vote": e_mv,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return result


def main() -> None:
    real_df, feature_names, baseline_X, common_ids = build_merged_dataset()
    all_results = {}
    for family in FAMILY_PREFIXES:
        all_results[family] = evaluate_family(family, real_df, feature_names, baseline_X, common_ids)

    with open(os.path.join(REAL_CORPUS_DIR, "all_families_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n저장 완료: {os.path.join(REAL_CORPUS_DIR, 'all_families_summary.json')}")


if __name__ == "__main__":
    main()
