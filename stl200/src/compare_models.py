"""
compare_models.py — 군집 모델 비교 (구간 3)

목적: "k=3이 적절한가? K-means가 최선인가?"를 실루엣 점수로 객관 비교.
입력: data/processed/features.csv (156개 아이템 × 7개 지표)
출력: 화면에 비교표 + 추천, data/processed/model_comparison.csv

세 가지를 비교:
  ① K-means      : 중심점 k개로 묶기 (k를 직접 지정)
  ② 계층적 군집화 : 가까운 것부터 합치기 (k를 직접 지정)
  ③ HDBSCAN      : 밀도로 묶고 outlier는 '잡음'으로 분리 (k 자동)

채점 도구: 실루엣 점수(높을수록 군집이 잘 분리됨, 0~1)
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "processed" / "features.csv"
OUT = ROOT / "data" / "processed" / "model_comparison.csv"

# 군집화에 쓰는 7개 지표 (analyze.py와 동일)
CLUSTER_FEATURES = ["F_T", "F_S", "slope_norm", "duration_ratio",
                    "cv_resid", "peak_sharpness", "time_to_peak"]


def main() -> None:
    df = pd.read_csv(FEATURES)
    # 1) 7개 지표만 뽑아서 표준화 (평균0·표준편차1 — 단위 영향 제거)
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES].to_numpy())
    print(f"분석 대상: {len(df)}개 아이템 × {len(CLUSTER_FEATURES)}개 지표\n")

    rows = []

    # 2) K-means: k=2~8까지 돌려 실루엣 점수 측정
    print("=" * 56)
    print("① K-means (k를 2~8로 바꿔가며)")
    print("=" * 56)
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append(("K-means", k, round(sil, 4)))
        bar = "█" * int(sil * 50)
        print(f"  k={k}: 실루엣 {sil:.4f}  {bar}")

    # 3) 계층적 군집화: 동일하게 k=2~8
    print("\n" + "=" * 56)
    print("② 계층적 군집화 (k를 2~8로)")
    print("=" * 56)
    for k in range(2, 9):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append(("계층", k, round(sil, 4)))
        bar = "█" * int(sil * 50)
        print(f"  k={k}: 실루엣 {sil:.4f}  {bar}")

    # 4) HDBSCAN: k를 안 정함. 밀도로 자동 군집 + outlier(잡음, 라벨 -1) 분리
    print("\n" + "=" * 56)
    print("③ HDBSCAN (k 자동, outlier는 잡음으로 분리)")
    print("=" * 56)
    for mcs in (5, 8, 10):  # min_cluster_size: 군집으로 인정할 최소 묶음 크기
        labels = HDBSCAN(min_cluster_size=mcs).fit_predict(X)
        n_clusters = len(set(labels) - {-1})       # -1은 잡음(outlier)
        n_noise = int((labels == -1).sum())
        # 잡음을 뺀 점들로만 실루엣 계산 (군집이 2개 이상일 때만)
        mask = labels != -1
        sil = (silhouette_score(X[mask], labels[mask])
               if n_clusters >= 2 and mask.sum() > n_clusters else float("nan"))
        rows.append((f"HDBSCAN(mcs={mcs})", n_clusters, round(sil, 4) if sil == sil else None))
        print(f"  min_cluster_size={mcs}: 군집 {n_clusters}개, 잡음 {n_noise}개, "
              f"실루엣 {sil:.4f}" if sil == sil else
              f"  min_cluster_size={mcs}: 군집 {n_clusters}개, 잡음 {n_noise}개, 실루엣 측정불가")

    # 5) 결과 저장 + 최고 점수 추천
    res = pd.DataFrame(rows, columns=["model", "k_or_clusters", "silhouette"])
    res.to_csv(OUT, index=False)
    best = res.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False).iloc[0]

    print("\n" + "=" * 56)
    print("📊 종합")
    print("=" * 56)
    print(f"최고 실루엣: {best['model']} (군집수 {int(best['k_or_clusters'])}) "
          f"= {best['silhouette']:.4f}")
    # K-means k=3 (현재 우리 설정)과 비교
    cur = res[(res.model == "K-means") & (res.k_or_clusters == 3)]["silhouette"]
    if len(cur):
        print(f"현재 설정(K-means k=3): {cur.iloc[0]:.4f}")
    print(f"\n→ 결과표 저장: {OUT}")


if __name__ == "__main__":
    main()
