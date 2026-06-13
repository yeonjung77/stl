"""
save_model.py — 학습된 군집 모델을 파일로 저장 (구간 4)

왜? analyze.py는 매번 모델(scaler+KMeans)을 새로 만들어 쓰고 버린다.
실시간 앱은 '미리 학습된 모델'을 불러와 새 키워드만 분류해야 하므로,
그 모델(쿠키 틀)을 파일로 얼려둔다.

저장 내용 (model/cluster_model.joblib 하나에 묶음):
  - scaler        : 표준화 기준(156개의 평균·표준편차)
  - kmeans        : 군집 중심점 3개
  - cluster_names : 군집번호(0/1/2) → 메가/마이크로/스테디 이름
  - features      : 사용한 7개 지표 순서

analyze.py의 cluster_and_label()과 동일한 절차·시드를 사용해
기존 분류 결과(features.csv)와 정확히 일치하도록 한다.
"""
from pathlib import Path
import warnings

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "cluster_model.joblib"

CLUSTER_FEATURES = ["F_T", "F_S", "slope_norm", "duration_ratio",
                    "cv_resid", "peak_sharpness", "time_to_peak"]


def train_and_name(df: pd.DataFrame):
    """analyze.py와 동일: 표준화 → KMeans(k=3) → 센트로이드로 이름 매핑."""
    scaler = StandardScaler().fit(df[CLUSTER_FEATURES].to_numpy())
    X = scaler.transform(df[CLUSTER_FEATURES].to_numpy())

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
    clusters = kmeans.predict(X)

    # 군집별 지표 평균으로 이름 매핑 (analyze.py와 동일 규칙)
    prof = pd.DataFrame(df[CLUSTER_FEATURES].to_numpy(), columns=CLUSTER_FEATURES)
    prof["cluster"] = clusters
    prof = prof.groupby("cluster")[CLUSTER_FEATURES].mean()

    steady_c = prof["F_S"].idxmax()                      # 계절성 최강 = 스테디
    remaining = [c for c in prof.index if c != steady_c]

    def micro_score(c):
        return (prof.loc[c, "cv_resid"] + prof.loc[c, "peak_sharpness"]
                - prof.loc[c, "duration_ratio"])
    remaining.sort(key=micro_score, reverse=True)
    micro_c, mega_c = remaining[0], remaining[1]
    cluster_names = {int(steady_c): "스테디", int(micro_c): "마이크로", int(mega_c): "메가"}
    return scaler, kmeans, cluster_names


def main() -> None:
    df = pd.read_csv(FEATURES)
    scaler, kmeans, cluster_names = train_and_name(df)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "scaler": scaler,
        "kmeans": kmeans,
        "cluster_names": cluster_names,
        "features": CLUSTER_FEATURES,
    }, MODEL_PATH)
    print(f"✅ 모델 저장 완료 → {MODEL_PATH}")
    print(f"   군집번호 → 이름: {cluster_names}")

    # --- 검증: 저장한 모델로 156개 다시 분류 → 기존 features.csv와 일치하나? ---
    m = joblib.load(MODEL_PATH)
    X = m["scaler"].transform(df[m["features"]].to_numpy())
    pred = [m["cluster_names"][c] for c in m["kmeans"].predict(X)]
    match = (pd.Series(pred) == df["cluster_label"].reset_index(drop=True)).mean()
    print(f"   검증: 저장 모델 재분류가 기존 결과와 {match:.0%} 일치 "
          f"({'✅ 완벽' if match == 1 else '⚠️ 불일치'})")


if __name__ == "__main__":
    main()
