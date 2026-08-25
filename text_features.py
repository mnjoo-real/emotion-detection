"""
발화문(대사 텍스트)을 TF-IDF(문자 n-gram) + SVD로 압축해서, 다른 음향 피처들과
같은 형태(wav_id별 고정 길이 수치 벡터)로 저장한다.

주의(중요): 이 코퍼스는 감정 시나리오에 맞춰 미리 써놓은 대본이라, 텍스트 피처는
"감정이 담긴 말투"보다 "이 시나리오에서만 쓰인 어휘"를 학습할 가능성이 높다.
그래서 emotion_classifier.py에서는 음향 전용 모델을 그대로 두고, 이 피처를 추가한
버전을 별도로 만들어 비교한다 — 새 데이터셋에 이 문장들이 없으면 이 피처는 의미가 없다.

한국어 형태소 분석기가 없어서 단어 단위가 아니라 문자 n-gram(2~4글자) 단위로
TF-IDF를 계산한다. SVD 차원은 N_COMPONENTS로 조절한다.
"""

import csv
import os

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from similarity import SITUATION_ALIASES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSVS = [
    (os.path.join(BASE_DIR, "4차년도.csv"), "cp949"),
    (os.path.join(BASE_DIR, "5차년도_2차.csv"), "cp949"),
]
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "text_features")

N_COMPONENTS = 50
FEATURE_COLS = [f"text_svd_{i + 1}" for i in range(N_COMPONENTS)]


def load_texts() -> tuple:
    wav_ids, situations, texts = [], [], []
    for path, encoding in RAW_CSVS:
        with open(path, encoding=encoding, newline="") as f:
            for row in csv.DictReader(f):
                sit = SITUATION_ALIASES.get(row["상황"].strip(), row["상황"].strip())
                wav_ids.append(row["wav_id"].strip())
                situations.append(sit)
                texts.append(row["발화문"].strip())
    return wav_ids, situations, texts


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_ids, situations, texts = load_texts()
    print(f"발화문 {len(texts)}개 로드")

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
    tfidf = vec.fit_transform(texts)
    print(f"TF-IDF shape: {tfidf.shape}")

    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    reduced = svd.fit_transform(tfidf)
    print(f"SVD로 {N_COMPONENTS}차원 압축, 설명된 분산 비율: {svd.explained_variance_ratio_.sum():.3f}")

    per_file_path = os.path.join(OUTPUT_DIR, "per_file_text_features.csv")
    with open(per_file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_id", "situation"] + FEATURE_COLS)
        for wid, sit, vec_row in zip(wav_ids, situations, reduced):
            writer.writerow([wid, sit] + [float(v) for v in vec_row])

    joblib.dump(vec, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(svd, os.path.join(OUTPUT_DIR, "svd.joblib"))

    print(f"저장 완료: {per_file_path}")
    print(f"저장 완료: {OUTPUT_DIR}/tfidf_vectorizer.joblib, svd.joblib (새 데이터셋에도 같은 벡터라이저 재사용 시 필요)")


if __name__ == "__main__":
    main()
