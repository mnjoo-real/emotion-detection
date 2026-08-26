"""
지금까지 만든 감정 분류기(117개 feature, Random Forest)의 성능이 우연이 아닌
진짜 신호인지 두 가지 방법으로 검증한다.

1. 순열 검정(permutation test): 라벨을 무작위로 섞은 뒤 완전히 동일한 절차로
   다시 학습·평가하는 걸 여러 번(N회) 반복해서 "우연히 이 정도 성능이 나올 확률"의
   분포(귀무분포)를 직접 만든다. 실제 라벨로 학습한 성능이 이 분포의 상위 몇 %에
   드는지가 p-value다. 단순히 더미 분류기 하나와 비교하는 것보다 훨씬 엄밀하다 -
   더미 비교는 "이 전략이 우리 모델보다 나쁘다"만 보여주지만, 순열 검정은 "우리
   모델이 만든 X-y 관계 자체가 우연히 생길 수 있는 수준을 실제로 넘는지"를 직접 잰다.
2. 5-fold 교차검증: 지금까지 본 성능 수치가 random_state=42라는 하나의 train/test
   분할에서 우연히 잘 나온 게 아닌지, 분할을 5번 바꿔가며 평균/표준편차로 확인한다.

화자 단위 분리는 여전히 안 되어 있다는 기존 한계는 그대로 남아있다 - 이 검증은
"신호가 존재하는지"를 확인하는 것이지 "새 화자에게 일반화되는지"를 확인하는 게
아니다.
"""

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from emotion_classifier import load_combined_features, load_majority_vote_labels

N_PERMUTATIONS = 200
N_ESTIMATORS = 100  # 순열 검정은 속도를 위해 300 대신 100그루로 통일(실제/순열 모두 동일 조건)


def run_permutation_test(X, y, label_name: str) -> None:
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    actual_acc = clf.score(X_test, y_test)
    actual_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"[{label_name}] 실제 성능: accuracy={actual_acc:.4f}, macro F1={actual_f1:.4f}")

    rng = np.random.RandomState(0)
    perm_accs, perm_f1s = [], []
    start = time.perf_counter()
    for i in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)
        clf_p = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=42, n_jobs=-1)
        clf_p.fit(X[train_idx], y_perm[train_idx])
        pred_p = clf_p.predict(X[test_idx])
        perm_accs.append((pred_p == y_perm[test_idx]).mean())
        perm_f1s.append(f1_score(y_perm[test_idx], pred_p, average="macro", zero_division=0))
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - start
            print(f"  {i + 1}/{N_PERMUTATIONS} 순열 완료 ({elapsed:.0f}초 경과)")

    perm_accs = np.array(perm_accs)
    perm_f1s = np.array(perm_f1s)

    p_acc = (1 + np.sum(perm_accs >= actual_acc)) / (N_PERMUTATIONS + 1)
    p_f1 = (1 + np.sum(perm_f1s >= actual_f1)) / (N_PERMUTATIONS + 1)

    print(f"\n[{label_name}] 순열(귀무분포) 통계:")
    print(f"  accuracy: mean={perm_accs.mean():.4f}, std={perm_accs.std():.4f}, max={perm_accs.max():.4f}")
    print(f"  macro F1: mean={perm_f1s.mean():.4f}, std={perm_f1s.std():.4f}, max={perm_f1s.max():.4f}")
    print(f"  p-value(accuracy) = {p_acc:.4f} ({'<0.005' if p_acc < 0.005 else p_acc})")
    print(f"  p-value(macro F1) = {p_f1:.4f} ({'<0.005' if p_f1 < 0.005 else p_f1})")

    z_f1 = (actual_f1 - perm_f1s.mean()) / perm_f1s.std() if perm_f1s.std() > 0 else float("inf")
    print(f"  실제 macro F1은 귀무분포 평균보다 {z_f1:.1f} 표준편차 위에 있음 (z-score 개념)")


def run_kfold_cv(X, y, label_name: str) -> None:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        acc = (pred == y[test_idx]).mean()
        f1 = f1_score(y[test_idx], pred, average="macro", zero_division=0)
        accs.append(acc)
        f1s.append(f1)
        print(f"  fold {fold}: accuracy={acc:.4f}, macro F1={f1:.4f}")

    accs, f1s = np.array(accs), np.array(f1s)
    print(f"[{label_name}] 5-fold CV: accuracy={accs.mean():.4f}±{accs.std():.4f}, macro F1={f1s.mean():.4f}±{f1s.std():.4f}")


def main() -> None:
    print("=" * 70)
    print("majority_vote 라벨 기준 검증")
    print("=" * 70)
    label_override = load_majority_vote_labels()
    wav_ids, X, y, feature_names = load_combined_features(label_override=label_override)
    print(f"샘플 수: {len(wav_ids)}, feature 수: {len(feature_names)}\n")

    print("--- 5-fold 교차검증 ---")
    run_kfold_cv(X, y, "majority_vote")

    print("\n--- 순열 검정 ---")
    run_permutation_test(X, y, "majority_vote")


if __name__ == "__main__":
    main()
