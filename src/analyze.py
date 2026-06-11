"""
analyze.py — 패션 트렌드 분석 파이프라인 (1단계)

흐름:
  data/raw/<slug>.csv (구글 트렌드 원본)
    → 로딩/정제
    → STL 분해 (트렌드/계절성/잔차)
    → 정량 지표(feature) 산출
    → 군집화 + 규칙기반 분류 (메가/마이크로/스테디)
    → 수명주기 단계 판정
    → data/processed/features.csv (지표·분류 표)
    → data/processed/components/<slug>.csv (STL 성분, 시각화용)

실행: /opt/anaconda3/envs/env_stl/bin/python src/analyze.py
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
EPS = 1e-9
PERIODS_PER_YEAR = 52  # 주별 데이터 기준


# ---------------------------------------------------------------------------
# 1. 데이터 로딩 / 정제
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> pd.DataFrame:
    """data/keywords.csv (slug, category, keyword, display_name)"""
    return pd.read_csv(path)


def find_timeseries_csv(raw_dir: Path, slug: str) -> Path | None:
    """
    아이템의 '관심도 시계열' CSV를 찾는다.
    구글 트렌드 다운로드 시 파일명이 multiTimeline 형태라 여러 이름을 허용:
      <slug>.csv  /  <slug>_multiTimeline.csv  /  <slug>_multiTimeline (1).csv
    (relatedQueries/relatedEntities 파일은 시계열이 아니므로 제외)
    """
    exact = raw_dir / f"{slug}.csv"
    if exact.exists():
        return exact
    cands = sorted(raw_dir.glob(f"{slug}_multiTimeline*.csv"))
    cands = [c for c in cands if "related" not in c.name.lower()]
    return cands[0] if cands else None


def load_trends_csv(path: Path) -> pd.Series | None:
    """
    구글 트렌드 multiTimeline.csv를 (날짜 index, 값) Series로 로딩.
    파일 앞부분의 'Category' / 빈줄 / 헤더 줄 형식이 제각각이라
    '첫 칼럼이 날짜로 파싱되는 줄'만 골라 쓰는 방식으로 견고하게 처리.
    값의 '<1' 표기는 0.5로 치환.
    """
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    dates, values = [], []
    for line in text:
        if "," not in line:
            continue
        first, _, rest = line.partition(",")
        ts = pd.to_datetime(first.strip(), errors="coerce")
        if pd.isna(ts):
            continue  # 프리앰블/헤더 줄 건너뜀
        val = rest.strip().replace("<1", "0.5")
        try:
            values.append(float(val))
            dates.append(ts)
        except ValueError:
            continue
    if not dates:
        return None
    s = pd.Series(values, index=pd.DatetimeIndex(dates), name="value").sort_index()
    s = s[~s.index.duplicated(keep="first")]
    # 주별 규칙 정렬(빠진 주가 있으면 선형보간)
    s = s.asfreq("W-SUN") if s.index.freq is None else s
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="W-SUN"))
    s = s.interpolate(limit_direction="both")
    return s


# ---------------------------------------------------------------------------
# 2. STL 분해 + 지표 산출
# ---------------------------------------------------------------------------
def decompose(series: pd.Series, period: int = PERIODS_PER_YEAR):
    """STL 분해. 데이터가 2주기 미만이면 period를 줄여 재시도."""
    n = len(series)
    p = period
    while p >= 4 and n < 2 * p + 1:
        p //= 2
    stl = STL(series, period=p, robust=True)
    res = stl.fit()
    comp = pd.DataFrame(
        {
            "observed": series.values,
            "trend": res.trend.values,
            "seasonal": res.seasonal.values,
            "resid": res.resid.values,
        },
        index=series.index,
    )
    return comp, p


def _safe_strength(numer_var: float, denom_var: float) -> float:
    return float(max(0.0, 1.0 - numer_var / (denom_var + EPS)))


def extract_features(comp: pd.DataFrame) -> dict:
    """STL 성분으로부터 분류용 정량 지표 산출."""
    obs = comp["observed"].to_numpy()
    T = comp["trend"].to_numpy()
    S = comp["seasonal"].to_numpy()
    R = comp["resid"].to_numpy()
    n = len(obs)
    mean_obs = float(np.mean(obs)) + EPS
    x = np.arange(n)

    # 트렌드/계절성 강도 (Hyndman feature)
    var_R = np.var(R)
    F_T = _safe_strength(var_R, np.var(T + R))
    F_S = _safe_strength(var_R, np.var(S + R))

    # 전체 추세 기울기 → 연간 변화량을 평균 대비 비율로 정규화
    slope_step = np.polyfit(x, T, 1)[0]
    slope_norm = float(slope_step * PERIODS_PER_YEAR / mean_obs)

    # 최근 1년 추세 변화(수명주기 판정용)
    w = min(PERIODS_PER_YEAR, n - 1)
    recent_change = float((T[-1] - T[-1 - w]) / (np.max(T) + EPS))

    # 지속기간: 피크의 50% 이상을 유지한 비율
    peak = float(np.max(obs))
    duration_ratio = float(np.mean(obs >= 0.5 * peak))

    # 잔차 변동성(CV)
    cv_resid = float(np.std(R) / mean_obs)

    # 피크 첨예도(스파이크성) = 피크 / 평소 수준
    # 검색량이 절반 이상 0인 아이템은 median(obs)=0 → 0으로 나눠 폭발하므로,
    # '검색이 있었던 주들의 중앙값'을 평소 수준으로 사용(없으면 평균)한다.
    nonzero = obs[obs > 0]
    typical = float(np.median(nonzero)) if nonzero.size else mean_obs
    peak_sharpness = float(peak / (typical + EPS))

    # 피크 시점 비율(앞쪽일수록 과거에 떴다가 식은 마이크로 성향)
    time_to_peak = float(np.argmax(T) / max(n - 1, 1))

    return {
        "F_T": round(F_T, 4),
        "F_S": round(F_S, 4),
        "slope_norm": round(slope_norm, 4),
        "recent_change": round(recent_change, 4),
        "duration_ratio": round(duration_ratio, 4),
        "cv_resid": round(cv_resid, 4),
        "peak_sharpness": round(peak_sharpness, 4),
        "time_to_peak": round(time_to_peak, 4),
        "peak_value": round(peak, 2),
        "current_value": round(float(obs[-1]), 2),
        "n_weeks": n,
    }


# ---------------------------------------------------------------------------
# 3. 수명주기 단계 판정 (규칙 기반)
# ---------------------------------------------------------------------------
def lifecycle_stage(feat: dict) -> str:
    cur_ratio = feat["current_value"] / (feat["peak_value"] + EPS)
    change = feat["recent_change"]
    if change > 0.10:  # 최근 상승
        return "성장" if cur_ratio >= 0.4 else "도입"
    if change < -0.10:  # 최근 하락
        return "쇠퇴"
    # 정체
    return "정점/성숙" if cur_ratio >= 0.7 else "쇠퇴/잔존"


# ---------------------------------------------------------------------------
# 4. 분류: 군집화 + 규칙 기반 교차검증
# ---------------------------------------------------------------------------
def rule_based_label(f: pd.Series) -> str:
    """간단한 규칙으로 1차 분류(군집 라벨과 대조용)."""
    if f["F_S"] >= 0.55 and f["F_S"] >= f["F_T"]:
        return "스테디"
    if f["peak_sharpness"] >= 3.0 and f["duration_ratio"] <= 0.35:
        return "마이크로"
    if f["duration_ratio"] >= 0.5 and f["cv_resid"] <= 0.25:
        return "메가"
    # 경계: 변동성/지속성으로 결정
    return "마이크로" if f["cv_resid"] > 0.25 else "메가"


CLUSTER_FEATURES = [
    "F_T", "F_S", "slope_norm", "duration_ratio",
    "cv_resid", "peak_sharpness", "time_to_peak",
]


def cluster_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """KMeans(k=3)로 군집 → 센트로이드 특성으로 메가/마이크로/스테디 이름 부여."""
    n = len(df)
    if n < 3:
        df["cluster"] = -1
        df["cluster_label"] = df.apply(rule_based_label, axis=1)
        return df

    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES].to_numpy())
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    df["cluster"] = km.fit_predict(X)

    # 군집별 평균으로 이름 매핑
    prof = df.groupby("cluster")[CLUSTER_FEATURES].mean()
    # 1) 계절성 가장 강한 군집 = 스테디
    steady_c = prof["F_S"].idxmax()
    remaining = [c for c in prof.index if c != steady_c]
    # 2) 남은 둘 중 '마이크로 점수' 높은 쪽 = 마이크로
    def micro_score(c):
        return (prof.loc[c, "cv_resid"] + prof.loc[c, "peak_sharpness"]
                - prof.loc[c, "duration_ratio"])
    remaining.sort(key=micro_score, reverse=True)
    micro_c, mega_c = remaining[0], remaining[1]
    name = {steady_c: "스테디", micro_c: "마이크로", mega_c: "메가"}
    df["cluster_label"] = df["cluster"].map(name)
    return df


# ---------------------------------------------------------------------------
# 5. 사분면 좌표 (지속성 X, 변동성 Y) — 시각화용
# ---------------------------------------------------------------------------
def quadrant_coords(df: pd.DataFrame) -> pd.DataFrame:
    def z(col):
        v = df[col]
        return (v - v.mean()) / (v.std(ddof=0) + EPS)
    df["persistence"] = (z("F_T") + z("duration_ratio")).round(4)      # X축
    df["volatility"] = (z("cv_resid") + z("peak_sharpness")).round(4)  # Y축
    return df


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "keywords.csv"))
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    comp_dir = out_dir / "components"
    comp_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.manifest))

    rows = []
    found, missing = [], []
    for _, m in manifest.iterrows():
        slug = m["slug"]
        csv_path = find_timeseries_csv(raw_dir, slug)
        if csv_path is None:
            missing.append(slug)
            continue
        series = load_trends_csv(csv_path)
        if series is None or len(series) < 10:
            print(f"  ⚠️  {slug}: 데이터가 비었거나 너무 짧음 → 건너뜀")
            missing.append(slug)
            continue

        comp, used_period = decompose(series)
        comp.to_csv(comp_dir / f"{slug}.csv", index_label="date")

        feat = extract_features(comp)
        feat["stage"] = lifecycle_stage(feat)
        feat.update(
            slug=slug,
            category=m["category"],          # 사전 가정(정답 아님, 검증용)
            display_name=m["display_name"],
            used_period=used_period,
        )
        rows.append(feat)
        found.append(slug)

    if not rows:
        print("\n❌ data/raw/ 에 분석할 CSV가 없습니다. DOWNLOAD_GUIDE.md를 보고 받아주세요.")
        return

    df = pd.DataFrame(rows).set_index("slug")
    df = cluster_and_label(df)
    df["rule_label"] = df.apply(rule_based_label, axis=1)
    df = quadrant_coords(df)

    # 보기 좋은 칼럼 순서
    front = ["display_name", "category", "cluster_label", "rule_label", "stage",
             "persistence", "volatility"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]

    out_csv = out_dir / "features.csv"
    df.to_csv(out_csv, encoding="utf-8")

    # 콘솔 요약
    print(f"\n✅ 분석 완료: {len(found)}개 / 누락 {len(missing)}개")
    if missing:
        print(f"   (누락: {', '.join(missing)})")
    print(f"   → {out_csv}")
    print(f"   → {comp_dir}/<slug>.csv (STL 성분)\n")
    show = df[["display_name", "category", "cluster_label", "stage",
               "F_T", "F_S", "cv_resid", "peak_sharpness", "duration_ratio"]]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(show.to_string())

    # 사전가정 vs 군집 일치율(참고용)
    cat_map = {"mega": "메가", "micro": "마이크로", "steady": "스테디"}
    agree = (df["category"].map(cat_map) == df["cluster_label"]).mean()
    print(f"\n사전가정 대비 군집 일치율: {agree:.0%}  (낮아도 정상 — 데이터 기반 재분류 결과)")


if __name__ == "__main__":
    main()
