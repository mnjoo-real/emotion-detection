"""Real-corpus pilot 공통 evaluation framework (Stage 1).

F006(glide-aware jitter)/F007(CWT prosody)/F008(aperiodicity)/F009(harmonic phase)/
F010(harmonic modulation) 다섯 feature family 모두에 동일한 protocol(A-E)을
적용하기 위한 공유 유틸리티. dissonance_tonality_stats.py의 epsilon_squared를
그대로 재사용해서 "p-value보다 effect size" 원칙을 새 feature에도 동일하게 적용한다.

leakage 방지 원칙: 이 모듈이 계산하는 quantity(effect size, redundancy correlation,
classifier CV)는 전부 새 feature 자체의 값(파일 단위로 독립적으로 계산됨, 코퍼스
통계에 의존하지 않음)에 대해서만 동작한다. classifier_ablation()의 CV는
StratifiedKFold로 매 fold 학습을 독립적으로 수행하며, 어떤 정규화도 fold 밖
통계를 사용하지 않는다(애초에 이 pilot은 새 feature에 대한 정규화를 추가하지
않는다 - 원 raw feature 값을 그대로 baseline에 추가해서 test).
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from dissonance_tonality_stats import epsilon_squared, effect_label

N_SPLITS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 200  # emotion_classifier.py의 300보다 조금 줄임(파일럿 표본 크기가 작아 속도 우선, 실제/baseline 모두 동일 조건)


# ---------------------------------------------------------------------------
# Evaluation A — feature sanity check
# ---------------------------------------------------------------------------

def feature_quality_report(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        vals = df[col].astype(float)
        n = len(vals)
        nan_pct = float(vals.isna().mean() * 100)
        finite = vals[np.isfinite(vals)]
        inf_pct = float((~np.isfinite(vals) & ~vals.isna()).mean() * 100)
        zero_pct = float((finite == 0).mean() * 100) if len(finite) else np.nan
        is_constant = bool(finite.nunique() <= 1) if len(finite) else True

        if len(finite) >= 4:
            q1, q3 = finite.quantile(0.25), finite.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_pct = float(((finite < lo) | (finite > hi)).mean() * 100)
        else:
            iqr = np.nan
            outlier_pct = np.nan

        rows.append({
            "feature": col, "n": n, "nan_pct": nan_pct, "inf_pct": inf_pct,
            "zero_pct": zero_pct, "is_constant": is_constant,
            "mean": float(finite.mean()) if len(finite) else np.nan,
            "std": float(finite.std()) if len(finite) else np.nan,
            "median": float(finite.median()) if len(finite) else np.nan,
            "iqr": float(iqr) if not pd.isna(iqr) else np.nan,
            "min": float(finite.min()) if len(finite) else np.nan,
            "max": float(finite.max()) if len(finite) else np.nan,
            "outlier_pct_iqr1.5": outlier_pct,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation B — emotion association (effect size first)
# ---------------------------------------------------------------------------

def kruskal_effect_size(groups: dict) -> dict:
    arrays = [np.asarray(v, dtype=float) for v in groups.values() if len(v) > 0]
    arrays = [a[np.isfinite(a)] for a in arrays]
    arrays = [a for a in arrays if len(a) > 0]
    if len(arrays) < 2:
        return {"h_stat": np.nan, "p_value": np.nan, "epsilon_sq": np.nan}
    n = sum(len(a) for a in arrays)
    k = len(arrays)
    h_stat, p_value = stats.kruskal(*arrays)
    eps = epsilon_squared(h_stat, n, k)
    means = {g: float(a.mean()) for g, a in zip(groups.keys(), arrays)}
    ranked = sorted(means.items(), key=lambda kv: kv[1])
    return {
        "h_stat": float(h_stat), "p_value": float(p_value), "epsilon_sq": eps,
        "effect_label": effect_label(eps) if not np.isnan(eps) else "n/a",
        "n": n, "k": k, "means": means, "lowest": ranked[0], "highest": ranked[-1],
    }


def emotion_association_table(df: pd.DataFrame, feature_cols: list, label_col: str) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        sub = df[[col, label_col]].dropna()
        groups = {label: sub.loc[sub[label_col] == label, col].values for label in sub[label_col].unique()}
        result = kruskal_effect_size(groups)
        rows.append({
            "feature": col, "label": label_col,
            "epsilon_sq": result["epsilon_sq"], "p_value": result["p_value"],
            "h_stat": result["h_stat"], "effect_label": result.get("effect_label", "n/a"),
            "lowest_group": result["lowest"][0] if "lowest" in result else None,
            "highest_group": result["highest"][0] if "highest" in result else None,
        })
    return pd.DataFrame(rows).sort_values("epsilon_sq", ascending=False)


# ---------------------------------------------------------------------------
# Evaluation C — confound sensitivity (gender)
# ---------------------------------------------------------------------------

def gender_confound_table(df: pd.DataFrame, feature_cols: list, emotion_col: str = "situation",
                           gender_col: str = "gender",
                           min_eps_meaningful: float = 0.01, min_eps_within_gender: float = 0.005,
                           rank_corr_threshold: float = 0.5) -> pd.DataFrame:
    """각 feature를 Type A(robust)/B(gender-dependent)/C(confounded)로 분류한다.
    임계값(min_eps_meaningful=0.01, dissonance_tonality_stats.py가 이미 쓰는
    "negligible" 경계와 동일; min_eps_within_gender=0.005는 성별 분리 시 표본이
    작아지는 것을 감안해 절반으로 완화; rank_corr_threshold=0.5는 남/녀 감정별
    평균 순위의 Spearman 상관이 이 값 이상이면 "방향이 대체로 일치"로 본다)는
    전부 이 함수 인자로 노출되어 있고, 로그에 그대로 기록한다."""
    rows = []
    for col in feature_cols:
        sub = df[[col, emotion_col, gender_col]].dropna()
        sub = sub[sub[gender_col].isin(["male", "female"])]

        pooled_groups = {e: sub.loc[sub[emotion_col] == e, col].values for e in sub[emotion_col].unique()}
        pooled = kruskal_effect_size(pooled_groups)

        gender_groups = {g: sub.loc[sub[gender_col] == g, col].values for g in ["male", "female"] if (sub[gender_col] == g).any()}
        gender_effect = kruskal_effect_size(gender_groups)

        within = {}
        rank_series = {}
        for g in ["male", "female"]:
            gsub = sub[sub[gender_col] == g]
            groups = {e: gsub.loc[gsub[emotion_col] == e, col].values for e in gsub[emotion_col].unique()}
            result = kruskal_effect_size(groups)
            within[g] = result
            if "means" in result:
                rank_series[g] = pd.Series(result["means"])

        rank_corr = np.nan
        if "male" in rank_series and "female" in rank_series:
            common = rank_series["male"].index.intersection(rank_series["female"].index)
            if len(common) >= 3:
                rank_corr = float(stats.spearmanr(rank_series["male"][common], rank_series["female"][common]).correlation)

        male_eps = within.get("male", {}).get("epsilon_sq", np.nan)
        female_eps = within.get("female", {}).get("epsilon_sq", np.nan)
        pooled_eps = pooled.get("epsilon_sq", np.nan)

        male_ok = not np.isnan(male_eps) and male_eps >= min_eps_within_gender
        female_ok = not np.isnan(female_eps) and female_eps >= min_eps_within_gender
        pooled_ok = not np.isnan(pooled_eps) and pooled_eps >= min_eps_meaningful
        rank_agrees = not np.isnan(rank_corr) and rank_corr >= rank_corr_threshold

        if pooled_ok and (male_ok or female_ok) and rank_agrees:
            feature_type = "A_robust"
        elif (male_ok or female_ok) and not rank_agrees:
            feature_type = "B_gender_dependent"
        elif pooled_ok and not male_ok and not female_ok:
            feature_type = "C_confounded"
        elif not pooled_ok and not male_ok and not female_ok:
            feature_type = "D_no_effect_anywhere"
        else:
            feature_type = "E_ambiguous"

        rows.append({
            "feature": col,
            "pooled_epsilon_sq": pooled_eps,
            "gender_epsilon_sq": gender_effect.get("epsilon_sq", np.nan),
            "male_within_epsilon_sq": male_eps, "female_within_epsilon_sq": female_eps,
            "male_female_rank_corr": rank_corr,
            "type": feature_type,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation D — redundancy with existing 117 features
# ---------------------------------------------------------------------------

def redundancy_table(new_df: pd.DataFrame, existing_df: pd.DataFrame, new_cols: list, existing_cols: list) -> pd.DataFrame:
    rows = []
    for new_col in new_cols:
        new_vals = new_df[new_col].astype(float)
        correlations = {}
        for exist_col in existing_cols:
            exist_vals = existing_df[exist_col].astype(float)
            valid = np.isfinite(new_vals) & np.isfinite(exist_vals)
            if valid.sum() < 5 or new_vals[valid].std() == 0 or exist_vals[valid].std() == 0:
                continue
            rho = stats.spearmanr(new_vals[valid], exist_vals[valid]).correlation
            if not np.isnan(rho):
                correlations[exist_col] = rho

        if not correlations:
            rows.append({"new_feature": new_col, "max_abs_rho": np.nan, "most_correlated_existing": None,
                         "top5_correlated_existing": None})
            continue

        ranked = sorted(correlations.items(), key=lambda kv: -abs(kv[1]))
        top5 = "; ".join(f"{k}={v:.3f}" for k, v in ranked[:5])
        rows.append({
            "new_feature": new_col,
            "max_abs_rho": abs(ranked[0][1]),
            "most_correlated_existing": ranked[0][0],
            "most_correlated_existing_rho": ranked[0][1],
            "top5_correlated_existing": top5,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation E — incremental classifier value
# ---------------------------------------------------------------------------

def _fit_eval_fold(X_train, y_train, X_test, y_test) -> float:
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return float(f1_score(y_test, pred, average="macro", zero_division=0))


def classifier_ablation(X_baseline: np.ndarray, X_new: np.ndarray, y: np.ndarray,
                         n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> dict:
    """baseline vs baseline+new를 정확히 같은 StratifiedKFold split으로 비교한다.
    fold별 macro F1과 fold별 delta를 모두 보존해서 평균만 보고하지 않는다."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_combined = np.hstack([X_baseline, X_new])

    baseline_f1s, combined_f1s = [], []
    for train_idx, test_idx in skf.split(X_baseline, y):
        baseline_f1s.append(_fit_eval_fold(X_baseline[train_idx], y[train_idx], X_baseline[test_idx], y[test_idx]))
        combined_f1s.append(_fit_eval_fold(X_combined[train_idx], y[train_idx], X_combined[test_idx], y[test_idx]))

    baseline_f1s = np.array(baseline_f1s)
    combined_f1s = np.array(combined_f1s)
    deltas = combined_f1s - baseline_f1s

    return {
        "baseline_f1_per_fold": baseline_f1s.tolist(), "combined_f1_per_fold": combined_f1s.tolist(),
        "delta_per_fold": deltas.tolist(),
        "baseline_f1_mean": float(baseline_f1s.mean()), "baseline_f1_std": float(baseline_f1s.std()),
        "combined_f1_mean": float(combined_f1s.mean()), "combined_f1_std": float(combined_f1s.std()),
        "delta_mean": float(deltas.mean()), "delta_std": float(deltas.std()), "delta_median": float(np.median(deltas)),
        "n_folds_improved": int(np.sum(deltas > 0)), "n_folds": n_splits,
    }


def save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
