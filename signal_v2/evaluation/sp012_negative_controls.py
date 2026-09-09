"""SP-V2-012 §7: random-feature negative control.

목적: "92개(혹은 F010 33개) 차원을 추가했다는 사실 자체"가 성능을 얼마나
깎아먹는지 분리해서 본다. 두 종류:

  A. Gaussian noise - 실제 feature와 같은 차원 수, 각 컬럼의 mean/std를 모방하되
     반드시 training fold 통계로만 생성한다(leakage 방지 - test fold 값은
     generation에 전혀 관여하지 않음).
  B. Permuted real features - 실제 v2 feature 값의 분포(marginal)는 그대로
     유지한 채 label과의 연관성만 제거한다. training fold 내부에서 행을
     shuffle하고, test fold는 "훈련 시 학습된 절차와 일관되게" 별도로
     독립 shuffle한 값을 쓴다(즉 test fold 고유의 real 값이 새어 들어가지
     않게, test fold 자체도 자기 자신 안에서만 shuffle).
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from signal_v2.evaluation.sp012_classifiers import N_OUTER_SPLITS, RANDOM_STATE, make_rf
from sklearn.metrics import f1_score


def gaussian_noise_like(X_train: np.ndarray, n_test: int, rng: np.random.RandomState) -> tuple:
    """X_train의 컬럼별 mean/std를 training fold에서만 추정해서 (train_noise, test_noise)를
    생성한다 - 두 세트 모두 이 training-fold 통계에서 뽑히므로 test 쪽에 test 자신의
    정보가 들어가지 않는다."""
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1.0
    train_noise = rng.normal(means, stds, size=X_train.shape)
    test_noise = rng.normal(means, stds, size=(n_test, X_train.shape[1]))
    return train_noise, test_noise


def permuted_like(X_train: np.ndarray, X_test: np.ndarray, rng: np.random.RandomState) -> tuple:
    """실제 feature 값의 marginal distribution은 유지하되, 행 순서를 독립적으로
    섞어서 label과의 연관을 끊는다. train/test 각각 자기 자신의 값만 섞는다
    (test fold의 real 값이 shuffle 소스로만 쓰이고 label과의 원래 대응은 깨짐)."""
    train_perm = X_train[rng.permutation(len(X_train))]
    test_perm = X_test[rng.permutation(len(X_test))]
    return train_perm, test_perm


def negative_control_comparison(X_baseline: np.ndarray, X_new: np.ndarray, y: np.ndarray,
                                 n_splits: int = N_OUTER_SPLITS, random_state: int = RANDOM_STATE,
                                 seed_for_controls: int = 0) -> dict:
    """baseline / baseline+real / baseline+gaussian / baseline+permuted를 정확히
    같은 outer split에서 비교한다(RF만 사용 - 이 probe의 목적은 classifier
    비교가 아니라 "차원 추가 자체의 비용" 측정이므로 sp012_classifiers의
    3-classifier 비교와는 별개 축)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.RandomState(seed_for_controls)

    conditions = {"baseline": [], "real_v2": [], "gaussian_noise": [], "permuted_v2": []}

    for train_idx, test_idx in skf.split(X_baseline, y):
        y_train, y_test = y[train_idx], y[test_idx]
        Xb_train, Xb_test = X_baseline[train_idx], X_baseline[test_idx]
        Xn_train, Xn_test = X_new[train_idx], X_new[test_idx]

        gauss_train, gauss_test = gaussian_noise_like(Xn_train, len(test_idx), rng)
        perm_train, perm_test = permuted_like(Xn_train, Xn_test, rng)

        for cond_name, (tr_extra, te_extra) in [
            ("baseline", (np.zeros((len(train_idx), 0)), np.zeros((len(test_idx), 0)))),
            ("real_v2", (Xn_train, Xn_test)),
            ("gaussian_noise", (gauss_train, gauss_test)),
            ("permuted_v2", (perm_train, perm_test)),
        ]:
            X_train = np.hstack([Xb_train, tr_extra])
            X_test = np.hstack([Xb_test, te_extra])
            clf = make_rf()
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            f1 = f1_score(y_test, pred, average="macro", zero_division=0)
            conditions[cond_name].append(f1)

    summary = {}
    for name, scores in conditions.items():
        arr = np.array(scores)
        summary[name] = {"mean": float(arr.mean()), "std": float(arr.std()), "per_fold": arr.tolist()}
    return summary
