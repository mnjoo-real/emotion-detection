"""
지금까지 추출한 여섯 세트의 음향 특징(MFCC/피치/에너지, 화성학적 특징, 선율 음정 관계,
부분음 러프니스, 음계(장조/단조) 적합도, 선율/리듬 - 음 길이·간격)을 합쳐서 감정 분류
규칙(1차 모델)을 만든다.

- Decision Tree: 사람이 읽을 수 있는 if-then 규칙을 직접 뽑기 위한 주 모델.
- Random Forest: 정확도 상한(이 피처 세트로 얼마나 분류 가능한지)을 가늠하는 비교용 모델.

나중에 별도 데이터셋(현재 다운로드 중)으로 검증할 때는, 그 데이터셋에도
extract_features.py / harmony_features.py / melodic_profile.py를 똑같이 돌려
같은 스키마의 세 CSV를 만든 뒤, load_combined_features()로 합치고
output/model/에 저장된 모델로 predict()하면 된다.

주의: 원본 메타데이터(4차년도.csv/5차년도_2차.csv)에 화자를 구분할 고유 ID가 없어서
train/test를 화자 단위로 분리하지 못했다. 같은 화자의 다른 발화가 train/test 양쪽에
섞여 있을 수 있어 정확도가 실제보다 낙관적으로 나올 수 있다 — 별도 데이터셋 검증이
중요한 이유 중 하나다.
"""

import csv
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "output", "features", "per_file_features.csv")
HARMONY_CSV = os.path.join(BASE_DIR, "output", "harmony", "per_file_harmony.csv")
MELODIC_CSV = os.path.join(BASE_DIR, "output", "melodic_profile", "per_file_melodic_profile.csv")
ROUGHNESS_CSV = os.path.join(BASE_DIR, "output", "partial_roughness", "per_file_partial_roughness.csv")
TONALITY_CSV = os.path.join(BASE_DIR, "output", "tonality", "per_file_tonality.csv")
MELODY_RHYTHM_CSV = os.path.join(BASE_DIR, "output", "melody_rhythm", "per_file_melody_rhythm.csv")
SPEAKER_NORM_CSV = os.path.join(BASE_DIR, "output", "speaker_norm", "per_file_speaker_norm.csv")
VAD_CSV = os.path.join(BASE_DIR, "output", "vad", "per_file_vad.csv")
PAUSE_CSV = os.path.join(BASE_DIR, "output", "pause", "per_file_pause.csv")
SPECTRAL_DYNAMICS_CSV = os.path.join(BASE_DIR, "output", "spectral_dynamics", "per_file_spectral_dynamics.csv")
FORMANT_BANDWIDTH_CSV = os.path.join(BASE_DIR, "output", "formant_bandwidth", "per_file_formant_bandwidth.csv")
VOICE_QUALITY_CSV = os.path.join(BASE_DIR, "output", "voice_quality", "per_file_voice_quality.csv")
TEXT_CSV = os.path.join(BASE_DIR, "output", "text_features", "per_file_text_features.csv")
RAW_CSVS = [
    (os.path.join(BASE_DIR, "4차년도.csv"), "cp949"),
    (os.path.join(BASE_DIR, "5차년도_2차.csv"), "cp949"),
]

MODEL_DIR = os.path.join(BASE_DIR, "output", "model")

# key_mode(major/minor)는 key_major_corr/key_minor_corr/key_major_minor_diff로,
# gender는 각 gender_z 피처로 이미 수치화되어 있어서 범주형 원본은 제외한다.
NON_FEATURE_COLS = {"wav_id", "situation", "key_mode", "gender"}

# 상황(시나리오 라벨) 표기를 평가자 감정 라벨 표기와 통일하기 위한 별칭.
LABEL_ALIASES = {"anger": "angry", "sad": "sadness"}


def _load_csv(path: str) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["wav_id"]: row for row in csv.DictReader(f)}


def load_majority_vote_labels() -> dict:
    """4차년도.csv/5차년도_2차.csv의 1~5번 평가자 감정+세기로 다수결(강도 가중) 라벨을 만든다.

    상황(시나리오 의도 라벨)과 달리, 실제로 사람 5명이 들었을 때 어떤 감정으로
    인지했는지를 반영한다. 세기(강도)까지 가중해서 합산 후 최댓값을 고르고,
    그래도 동률이면 라벨이 애매하다고 보고 제외한다(None).
    """
    labels: dict = {}
    for path, encoding in RAW_CSVS:
        with open(path, encoding=encoding, newline="") as f:
            for row in csv.DictReader(f):
                wav_id = row["wav_id"].strip()
                weights: dict = {}
                for n in ["1", "2", "3", "4", "5"]:
                    emo = row.get(f"{n}번 감정", "").strip().lower()
                    emo = LABEL_ALIASES.get(emo, emo)
                    intensity_key = f"{n}번 감정세기" if f"{n}번 감정세기" in row else f"{n}번감정세기"
                    intensity = float(row.get(intensity_key, "0") or 0)
                    # 세기 0(무감정 취급)도 "이 라벨에 투표했다"는 사실 자체는 반영하기 위해 +1
                    weights[emo] = weights.get(emo, 0.0) + intensity + 1

                ranked = sorted(weights.items(), key=lambda kv: -kv[1])
                if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                    labels[wav_id] = None  # 동률 - 애매한 라벨
                else:
                    labels[wav_id] = ranked[0][0]
    return labels


def load_combined_features(
    features_csv: str = FEATURES_CSV,
    harmony_csv: str = HARMONY_CSV,
    melodic_csv: str = MELODIC_CSV,
    roughness_csv: str = ROUGHNESS_CSV,
    tonality_csv: str = TONALITY_CSV,
    melody_rhythm_csv: str = MELODY_RHYTHM_CSV,
    speaker_norm_csv: str = SPEAKER_NORM_CSV,
    vad_csv: str = VAD_CSV,
    pause_csv: str = PAUSE_CSV,
    spectral_dynamics_csv: str = SPECTRAL_DYNAMICS_CSV,
    formant_bandwidth_csv: str = FORMANT_BANDWIDTH_CSV,
    voice_quality_csv: str = VOICE_QUALITY_CSV,
    text_csv: str = None,
    label_override: dict = None,
):
    """음향 피처 CSV들(+선택적으로 텍스트 피처 CSV)을 wav_id로 합쳐서
    (wav_ids, X, y, feature_names)를 반환한다.

    text_csv를 지정하면(예: TEXT_CSV) 발화문 기반 텍스트 피처도 포함한다 — 단, 이건
    이 코퍼스의 대본 어휘에 종속적이라 새 데이터셋에 같은 문장이 없으면 의미가 없다.
    label_override가 주어지면 situation 컬럼 대신 그 dict(wav_id -> label)를 라벨로 쓴다.
    값이 None인 wav_id(애매한 라벨)는 제외한다.
    """
    sources = {
        "a": _load_csv(features_csv),
        "b": _load_csv(harmony_csv),
        "c": _load_csv(melodic_csv),
        "d": _load_csv(roughness_csv),
        "f": _load_csv(tonality_csv),
        "g": _load_csv(melody_rhythm_csv),
        "h": _load_csv(speaker_norm_csv),
        "i": _load_csv(vad_csv),
        "j": _load_csv(pause_csv),
        "k": _load_csv(spectral_dynamics_csv),
        "l": _load_csv(formant_bandwidth_csv),
        "m": _load_csv(voice_quality_csv),
    }
    if text_csv is not None:
        sources["e"] = _load_csv(text_csv)

    common_ids = set.intersection(*(set(d) for d in sources.values()))
    common_ids = sorted(common_ids)
    if label_override is not None:
        common_ids = [wid for wid in common_ids if label_override.get(wid) is not None]
    if not common_ids:
        raise ValueError("사용 가능한 wav_id가 없습니다.")

    # 여러 소스에 동일한 이름의 컬럼(예: n_voiced_pitch_frames - harmony/tonality/
    # melody_rhythm 세 스크립트 모두 같은 pyin 파라미터로 계산해 값도 동일함)이 있으면
    # 중복 컬럼이 생기므로 처음 등장한 것만 남긴다.
    feature_names = []
    seen = set()
    for d in sources.values():
        for k in next(iter(d.values())):
            if k in NON_FEATURE_COLS or k in seen:
                continue
            feature_names.append(k)
            seen.add(k)

    X = np.zeros((len(common_ids), len(feature_names)), dtype=np.float64)
    y = []
    for i, wid in enumerate(common_ids):
        row = {}
        for d in sources.values():
            row.update(d[wid])
        label = label_override[wid] if label_override is not None else row["situation"]
        y.append(label)
        for j, col in enumerate(feature_names):
            X[i, j] = float(row[col])

    return common_ids, X, np.array(y), feature_names


def train_and_evaluate(label_override: dict = None, model_suffix: str = "", include_text: bool = False) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    text_csv = TEXT_CSV if include_text else None
    wav_ids, X, y, feature_names = load_combined_features(text_csv=text_csv, label_override=label_override)
    print(f"전체 샘플: {len(wav_ids)}개, 피처 수: {len(feature_names)}개")
    classes, counts = np.unique(y, return_counts=True)
    print("클래스 분포:", dict(zip(classes, counts)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\ntrain: {len(X_train)}개, test: {len(X_test)}개 (파일 단위 랜덤 분할 — 화자 ID가 없어 화자 단위 분할 불가)")

    # --- Decision Tree: 규칙 추출용 ---
    tree = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)

    print("\n" + "=" * 60)
    print("Decision Tree (max_depth=6) 결과")
    print("=" * 60)
    print(f"accuracy: {tree.score(X_test, y_test):.3f}, macro F1: {f1_score(y_test, tree_pred, average='macro'):.3f}")
    print(classification_report(y_test, tree_pred, zero_division=0))

    print("--- 혼동 행렬 (행=실제, 열=예측) ---")
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, tree_pred, labels=labels)
    print(f"{'':<12}" + "".join(f"{l[:8]:>10}" for l in labels))
    for label, row in zip(labels, cm):
        print(f"{label:<12}" + "".join(f"{v:>10}" for v in row))

    print("\n--- 추출된 규칙 (상위 depth) ---")
    print(export_text(tree, feature_names=feature_names, max_depth=4))

    # --- Random Forest: 정확도 상한 + 피처 중요도 ---
    forest = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1
    )
    forest.fit(X_train, y_train)
    forest_pred = forest.predict(X_test)

    print("\n" + "=" * 60)
    print("Random Forest (300 trees) 결과 — 이 피처 세트의 정확도 상한 참고용")
    print("=" * 60)
    print(f"accuracy: {forest.score(X_test, y_test):.3f}, macro F1: {f1_score(y_test, forest_pred, average='macro'):.3f}")
    print(classification_report(y_test, forest_pred, zero_division=0))

    importances = sorted(zip(feature_names, forest.feature_importances_), key=lambda kv: -kv[1])
    print("--- 피처 중요도 상위 15개 ---")
    for name, imp in importances[:15]:
        print(f"  {name:<28}{imp:.4f}")

    # --- 저장: 나중에 새 데이터셋에 그대로 적용하기 위해 ---
    tree_path = os.path.join(MODEL_DIR, f"decision_tree{model_suffix}.joblib")
    forest_path = os.path.join(MODEL_DIR, f"random_forest{model_suffix}.joblib")
    joblib.dump(tree, tree_path)
    joblib.dump(forest, forest_path)
    with open(os.path.join(MODEL_DIR, f"feature_names{model_suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    print(f"\n모델 저장 완료: {tree_path}, {forest_path}")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    use_majority = "majority_vote" in args
    use_text = "with_text" in args

    label_override = None
    suffix = ""
    if use_majority:
        print("### 다수결(평가자 5명, 강도 가중) 라벨로 학습 ###\n")
        label_override = load_majority_vote_labels()
        n_ambiguous = sum(1 for v in label_override.values() if v is None)
        print(f"전체 {len(label_override)}개 중 동률(애매한 라벨) {n_ambiguous}개 제외\n")
        suffix += "_majority_vote"
    else:
        print("### 상황(시나리오 의도) 라벨로 학습 ###\n")

    if use_text:
        print("### 텍스트(발화문 TF-IDF+SVD) 피처 포함 ###\n")
        suffix += "_with_text"

    train_and_evaluate(label_override=label_override, model_suffix=suffix, include_text=use_text)
