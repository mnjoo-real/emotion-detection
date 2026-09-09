"""SP-V2-012 §9: F010 top-k feature efficiency.

F010 33개 feature 전체가 다 좋은가, 아니면 소수가 signal 대부분을 담당하는가?
top-k selection은 반드시 training fold 내부에서만 수행한다(outer test fold의
ranking 정보가 선택 과정에 들어가지 않도록) - StratifiedKFold의 각 outer training
fold 안에서 순위를 다시 매기고, 그 순위로 고른 top-k만 같은 fold의 test에 적용한다.
"""

import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dissonance_tonality_stats import epsilon_squared
from signal_v2.evaluation.sp012_classifiers import N_OUTER_SPLITS, RANDOM_STATE, make_rf

K_VALUES = [1, 3, 5, 10, 20, "all"]


def rank_features_by_training_fold_eps2(X_train: np.ndarray, y_train: np.ndarray, feature_names: list) -> list:
    """training fold 안에서만 epsilon^2로 feature를 순위 매긴다."""
    scores = []
    for j, name in enumerate(feature_names):
        col = X_train[:, j]
        groups = [col[y_train == c] for c in np.unique(y_train)]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            scores.append((name, j, -999.0))
            continue
        n = sum(len(g) for g in groups)
        k = len(groups)
        h, _ = stats.kruskal(*groups)
        eps = epsilon_squared(h, n, k)
        scores.append((name, j, eps if not np.isnan(eps) else -999.0))
    scores.sort(key=lambda t: -t[2])
    return scores


def top_k_efficiency(X_baseline: np.ndarray, X_f010: np.ndarray, y: np.ndarray, feature_names: list,
                      k_values: list = K_VALUES, n_splits: int = N_OUTER_SPLITS,
                      random_state: int = RANDOM_STATE) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {str(k): {"combined_f1": [], "top_features_per_fold": []} for k in k_values}
    baseline_f1s = []

    for train_idx, test_idx in skf.split(X_baseline, y):
        y_train, y_test = y[train_idx], y[test_idx]

        clf_base = make_rf()
        clf_base.fit(X_baseline[train_idx], y_train)
        baseline_f1s.append(f1_score(y_test, clf_base.predict(X_baseline[test_idx]), average="macro", zero_division=0))

        ranked = rank_features_by_training_fold_eps2(X_f010[train_idx], y_train, feature_names)

        for k in k_values:
            if k == "all":
                cols = list(range(len(feature_names)))
                top_names = feature_names
            else:
                top = ranked[:k]
                cols = [j for _, j, _ in top]
                top_names = [name for name, _, _ in top]

            X_train_combined = np.hstack([X_baseline[train_idx], X_f010[train_idx][:, cols]])
            X_test_combined = np.hstack([X_baseline[test_idx], X_f010[test_idx][:, cols]])
            clf = make_rf()
            clf.fit(X_train_combined, y_train)
            f1 = f1_score(y_test, clf.predict(X_test_combined), average="macro", zero_division=0)
            results[str(k)]["combined_f1"].append(f1)
            results[str(k)]["top_features_per_fold"].append(top_names)

    baseline_f1s = np.array(baseline_f1s)
    summary = {"baseline_f1_mean": float(baseline_f1s.mean()), "baseline_f1_std": float(baseline_f1s.std())}
    for k in k_values:
        f1s = np.array(results[str(k)]["combined_f1"])
        delta = f1s - baseline_f1s
        summary[str(k)] = {
            "combined_f1_mean": float(f1s.mean()), "combined_f1_std": float(f1s.std()),
            "delta_mean": float(delta.mean()), "n_folds_improved": int(np.sum(delta > 0)),
            "top_features_per_fold": results[str(k)]["top_features_per_fold"],
        }
    return summary


if __name__ == "__main__":
    from signal_v2.evaluation.sp012_scaling_common import load_level_dataset, numeric_cols

    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "output", "signal_v2", "real_corpus", "real_corpus_features.csv")
    data = load_level_dataset(csv_path)
    df = data["df"]
    f010_features = numeric_cols(df, "f010_")
    X_f010 = df[f010_features].fillna(0.0).values

    summary = top_k_efficiency(data["X_baseline"], X_f010, data["y_situation"], f010_features)
    for k in K_VALUES:
        s = summary[str(k)]
        print(f"k={k!s:<5} combined_f1={s['combined_f1_mean']:.4f} delta={s['delta_mean']:+.4f} "
              f"({s['n_folds_improved']}/5 folds)")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "output", "signal_v2", "SP-V2-012")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "f010_top_k_efficiency.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n저장 완료: {os.path.join(out_dir, 'f010_top_k_efficiency.json')}")
