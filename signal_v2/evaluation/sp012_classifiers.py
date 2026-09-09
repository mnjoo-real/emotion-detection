"""SP-V2-012 §8-10: classifier-specific hypothesis 검증 + stability selection.

Random Forest만으로 "high-dimensionality 문제"라고 결론 내리지 않기 위해 Elastic-Net
multinomial logistic regression과 Linear SVM을 추가한다. 하이퍼파라미터는 반드시
outer test fold 밖(training fold 내부 inner CV)에서만 고른다 - 이 파일의 모든
tuning은 GridSearchCV(cv=inner_splits)를 training fold에만 적용하는 구조로
강제한다.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
RF_N_ESTIMATORS = 200

ELASTICNET_GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__l1_ratio": [0.15, 0.5, 0.85]}
SVM_GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0]}


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1)


def make_elasticnet_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=3000,
                                    class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def make_svm_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000)),
    ])


def _fit_predict(name: str, X_train, y_train, X_test, inner_splits: int = N_INNER_SPLITS):
    """name에 맞는 모델을 (필요하면 training-fold-only inner CV로 튜닝해서) 학습하고
    X_test에 대한 예측을 반환한다. RF는 하이퍼파라미터를 이 파일의 기본값(RF_N_ESTIMATORS)
    으로 고정한다 - 브리핑이 명시한 대로 "기존 hyperparameter를 baseline으로 유지"."""
    if name == "rf":
        clf = make_rf()
        clf.fit(X_train, y_train)
        return clf.predict(X_test)

    if name == "elasticnet":
        pipeline = make_elasticnet_pipeline()
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(pipeline, ELASTICNET_GRID, cv=inner_cv, scoring="f1_macro", n_jobs=-1)
        search.fit(X_train, y_train)
        return search.predict(X_test), search.best_estimator_

    if name == "svm":
        pipeline = make_svm_pipeline()
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(pipeline, SVM_GRID, cv=inner_cv, scoring="f1_macro", n_jobs=-1)
        search.fit(X_train, y_train)
        return search.predict(X_test), search.best_estimator_

    raise ValueError(f"unknown classifier: {name}")


def classifier_comparison(X_baseline: np.ndarray, X_new: np.ndarray, y: np.ndarray,
                           classifiers: list = ("rf", "elasticnet", "svm"),
                           n_outer_splits: int = N_OUTER_SPLITS, random_state: int = RANDOM_STATE) -> dict:
    """baseline vs baseline+new를 여러 classifier로 비교한다. 모든 classifier가
    정확히 동일한 outer StratifiedKFold split을 쓴다."""
    skf = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
    X_combined = np.hstack([X_baseline, X_new])

    results = {name: {"baseline_f1": [], "combined_f1": []} for name in classifiers}
    elasticnet_coefs = []  # stability selection용 (§10)

    for train_idx, test_idx in skf.split(X_baseline, y):
        y_train, y_test = y[train_idx], y[test_idx]
        for name in classifiers:
            if name == "rf":
                pred_base = _fit_predict(name, X_baseline[train_idx], y_train, X_baseline[test_idx])
                pred_comb = _fit_predict(name, X_combined[train_idx], y_train, X_combined[test_idx])
            else:
                pred_base, _ = _fit_predict(name, X_baseline[train_idx], y_train, X_baseline[test_idx])
                pred_comb, best_est = _fit_predict(name, X_combined[train_idx], y_train, X_combined[test_idx])
                if name == "elasticnet":
                    coef = best_est.named_steps["clf"].coef_  # (n_classes, n_features)
                    elasticnet_coefs.append(coef[:, X_baseline.shape[1]:])  # new-feature 부분만

            results[name]["baseline_f1"].append(f1_score(y_test, pred_base, average="macro", zero_division=0))
            results[name]["combined_f1"].append(f1_score(y_test, pred_comb, average="macro", zero_division=0))

    summary = {}
    for name in classifiers:
        base = np.array(results[name]["baseline_f1"])
        comb = np.array(results[name]["combined_f1"])
        delta = comb - base
        summary[name] = {
            "baseline_f1_mean": float(base.mean()), "baseline_f1_std": float(base.std()),
            "combined_f1_mean": float(comb.mean()), "combined_f1_std": float(comb.std()),
            "delta_mean": float(delta.mean()), "delta_std": float(delta.std()),
            "delta_per_fold": delta.tolist(), "n_folds_improved": int(np.sum(delta > 0)),
            "n_folds": n_outer_splits,
        }

    return {"summary": summary, "elasticnet_new_feature_coefs": elasticnet_coefs}


def stability_selection_frequency(elasticnet_coefs: list, feature_names: list, zero_tol: float = 1e-6) -> dict:
    """fold마다 non-zero로 살아남은 feature의 비율(selection frequency)을 계산한다.
    다중 클래스이므로 클래스 중 하나라도 non-zero면 "선택됨"으로 본다."""
    if not elasticnet_coefs:
        return {}
    n_runs = len(elasticnet_coefs)
    freq = {name: 0 for name in feature_names}
    for coef in elasticnet_coefs:  # coef: (n_classes, n_new_features)
        nonzero_any_class = np.any(np.abs(coef) > zero_tol, axis=0)
        for name, is_selected in zip(feature_names, nonzero_any_class):
            if is_selected:
                freq[name] += 1
    return {name: count / n_runs for name, count in freq.items()}
