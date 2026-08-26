"""
한의환·차형태(2017) "러셀 모델의 확장을 통한 감정차원 모델링 방법 연구"의 방법을
우리 데이터에 적용한다.

논문 요지: Russell의 Circumplex Model(Valence-Arousal 2차원)에서 감정을 하나의
점(평균)이 아니라, 데이터 분포를 정규분포로 가정한 타원 영역으로 표현한다.
- 타원의 중심 = 감정 그룹의 (Valence, Arousal) 평균
- 회전각 θ = 두 축의 상관계수(ρ)로부터 유도 (논문 Eq. 3)
- 장/단축 길이 = 표준편차 × k (논문은 k=1, 즉 1σ 타원 사용, Eq. 4)
- 여러 감정의 타원 영역을 베이지안 결정 규칙(사전확률 × 우도)으로 비교해서
  분류 경계(결정 곡선)를 얻는다 (논문 Fig. 13, Eq. 1).

논문은 ANEW 83개 단어의 설문 평균/분산(즉 '단어에 대한 사람들의 평가')에 이 방법을
적용해 92.86% 정확도를 얻었다 - 이건 사람이 이미 잘 구분해서 응답한 단어 자체를
재분류하는 것에 가깝다. 우리는 이걸 훨씬 어려운 조건에 적용한다: 실제 발화 음성에서
wav2vec2로 예측한 (arousal, valence, dominance) 값(잡음이 많고 화자·문맥에 따라
크게 흔들림)을 가지고 7개 감정을 구분해야 한다.

수학적으로 "감정별 평균+공분산으로 타원을 만들고 베이지안 결정 규칙으로 분류"는
scikit-learn의 QuadraticDiscriminantAnalysis(QDA)와 완전히 동일하다 - 클래스별
다변량 정규분포를 가정하고 사전확률까지 곱해서 사후확률이 가장 큰 클래스로
분류하는 것이 정확히 QDA의 정의이기 때문이다. 그래서 타원 파라미터는 논문 방식
그대로(Eq. 3, 4) 직접 유도해서 표/그림으로 보여주고, 실제 분류 정확도는 QDA로
검증한다.
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from emotion_classifier import load_majority_vote_labels

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAD_CSV = os.path.join(BASE_DIR, "output", "vad", "per_file_vad.csv")

K_SCALE = 1.0  # 논문과 동일하게 1시그마 타원


def load_vad() -> dict:
    """wav_id -> {valence, arousal, dominance, situation}"""
    rows = {}
    with open(VAD_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["wav_id"]] = {
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
                "dominance": float(row["dominance"]),
                "situation": row["situation"],
            }
    return rows


def derive_ellipse_params(valence: np.ndarray, arousal: np.ndarray) -> dict:
    """논문 Eq. 3(회전각), Eq. 4(장단축)를 그대로 구현."""
    m1, m2 = float(valence.mean()), float(arousal.mean())
    s1, s2 = float(valence.std()), float(arousal.std())
    rho = float(np.corrcoef(valence, arousal)[0, 1])

    # Eq. 3: theta = 1/2 * atan(2*rho*s1*s2 / (s1^2 - s2^2))
    denom = s1 ** 2 - s2 ** 2
    theta = 0.5 * np.arctan2(2 * rho * s1 * s2, denom) if denom != 0 else np.pi / 4

    a = K_SCALE * s1  # valence 방향 반축
    b = K_SCALE * s2  # arousal 방향 반축

    return {
        "mean_valence": m1, "mean_arousal": m2,
        "std_valence": s1, "std_arousal": s2,
        "correlation": rho, "theta_deg": float(np.degrees(theta)),
        "semi_axis_valence": a, "semi_axis_arousal": b,
    }


def build_ellipse_table(vad_by_id: dict, label_map: dict = None) -> dict:
    groups: dict = {}
    for wav_id, row in vad_by_id.items():
        label = row["situation"] if label_map is None else label_map.get(wav_id)
        if label is None:
            continue
        groups.setdefault(label, {"valence": [], "arousal": []})
        groups[label]["valence"].append(row["valence"])
        groups[label]["arousal"].append(row["arousal"])

    params = {}
    for label, vals in groups.items():
        v, a = np.array(vals["valence"]), np.array(vals["arousal"])
        params[label] = derive_ellipse_params(v, a)
        params[label]["n"] = len(v)
    return params


def print_ellipse_table(params: dict, title: str) -> None:
    print(f"\n=== {title} : 감정별 (Valence, Arousal) 타원 파라미터 (k=1) ===")
    print(f"{'감정':<12}{'n':>6}{'mean_V':>9}{'mean_A':>9}{'std_V':>8}{'std_A':>8}{'corr':>8}{'θ(deg)':>9}")
    for label, p in sorted(params.items(), key=lambda kv: kv[1]["mean_arousal"]):
        print(
            f"{label:<12}{p['n']:>6}{p['mean_valence']:>9.4f}{p['mean_arousal']:>9.4f}"
            f"{p['std_valence']:>8.4f}{p['std_arousal']:>8.4f}{p['correlation']:>8.3f}{p['theta_deg']:>9.1f}"
        )


def run_qda(X: np.ndarray, y: np.ndarray, label_name: str, feature_name: str) -> None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    priors = None  # 클래스 비율을 그대로 사전확률로 사용 (논문처럼 균등 사전확률을 쓰려면 priors=uniform)
    qda = QuadraticDiscriminantAnalysis(priors=priors)
    qda.fit(X_train, y_train)
    pred = qda.predict(X_test)

    acc = qda.score(X_test, y_test)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"\n[{label_name} / {feature_name}] QDA(베이지안, 타원 영역) 분류 결과")
    print(f"accuracy={acc:.4f}, macro F1={macro_f1:.4f}")
    print(classification_report(y_test, pred, zero_division=0))


def plot_ellipses(vad_by_id: dict, label_map: dict, title: str, out_path: str) -> None:
    groups: dict = {}
    for wav_id, row in vad_by_id.items():
        label = row["situation"] if label_map is None else label_map.get(wav_id)
        if label is None:
            continue
        groups.setdefault(label, {"valence": [], "arousal": []})
        groups[label]["valence"].append(row["valence"])
        groups[label]["arousal"].append(row["arousal"])

    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    fig, ax = plt.subplots(figsize=(8, 7))
    for (label, vals), color in zip(sorted(groups.items()), colors):
        v, a = np.array(vals["valence"]), np.array(vals["arousal"])
        p = derive_ellipse_params(v, a)

        # 산점도는 너무 많으니 일부만 표시
        idx = np.random.RandomState(0).choice(len(v), size=min(300, len(v)), replace=False)
        ax.scatter(v[idx], a[idx], s=4, alpha=0.15, color=color)

        ellipse = Ellipse(
            (p["mean_valence"], p["mean_arousal"]),
            width=2 * p["semi_axis_valence"], height=2 * p["semi_axis_arousal"],
            angle=p["theta_deg"], edgecolor=color, facecolor="none", linewidth=2.2,
        )
        ax.add_patch(ellipse)
        ax.annotate(label, (p["mean_valence"], p["mean_arousal"]), color=color,
                     fontsize=11, fontweight="bold", ha="center")

    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n저장 완료: {out_path}")


def main() -> None:
    vad_by_id = load_vad()
    majority_vote = load_majority_vote_labels()

    # --- 1. 타원 파라미터 표 (situation / majority_vote 둘 다) ---
    params_situation = build_ellipse_table(vad_by_id, label_map=None)
    print_ellipse_table(params_situation, "situation")

    params_mv = build_ellipse_table(vad_by_id, label_map=majority_vote)
    print_ellipse_table(params_mv, "majority_vote")

    # --- 2. QDA(베이지안, 타원 영역 등가) 분류 정확도 ---
    common_ids_sit = sorted(vad_by_id)
    X_va = np.array([[vad_by_id[w]["valence"], vad_by_id[w]["arousal"]] for w in common_ids_sit])
    X_vad = np.array([[vad_by_id[w]["valence"], vad_by_id[w]["arousal"], vad_by_id[w]["dominance"]] for w in common_ids_sit])
    y_sit = np.array([vad_by_id[w]["situation"] for w in common_ids_sit])

    run_qda(X_va, y_sit, "situation", "V-A (2D, 논문과 동일)")
    run_qda(X_vad, y_sit, "situation", "V-A-D (3D, 확장)")

    common_ids_mv = [w for w in common_ids_sit if majority_vote.get(w) is not None]
    X_va_mv = np.array([[vad_by_id[w]["valence"], vad_by_id[w]["arousal"]] for w in common_ids_mv])
    X_vad_mv = np.array([[vad_by_id[w]["valence"], vad_by_id[w]["arousal"], vad_by_id[w]["dominance"]] for w in common_ids_mv])
    y_mv = np.array([majority_vote[w] for w in common_ids_mv])

    run_qda(X_va_mv, y_mv, "majority_vote", "V-A (2D, 논문과 동일)")
    run_qda(X_vad_mv, y_mv, "majority_vote", "V-A-D (3D, 확장)")

    # --- 3. 시각화 ---
    out_dir = os.path.join(BASE_DIR, "output", "vad_region")
    os.makedirs(out_dir, exist_ok=True)
    plot_ellipses(vad_by_id, None, "Emotional Region (situation labels, k=1σ)",
                  os.path.join(out_dir, "emotion_region_situation.png"))
    plot_ellipses(vad_by_id, majority_vote, "Emotional Region (majority_vote labels, k=1σ)",
                  os.path.join(out_dir, "emotion_region_majority_vote.png"))


if __name__ == "__main__":
    main()
