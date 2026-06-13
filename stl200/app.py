"""
app.py — 패션 트렌드 실시간 분류기 (Streamlit, 구간 6)

사용자가 패션 키워드를 입력하면:
  1) pytrends로 구글 트렌드 데이터를 즉석 수집
  2) analyze.py의 STL 분해 + 지표 산출 함수 재사용
  3) save_model.py가 저장한 모델(scaler+KMeans+이름)로 분류
  4) 규칙으로 수명주기 단계 판정
  5) 결과(분류·단계·전략·STL 그래프)를 화면에 표시

핵심: 분석(analyze.py)·모델(model/)을 그대로 재사용 → 웹 결과와 동일 기준.
실행: streamlit run app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pytrends.request import TrendReq

# analyze.py의 분석 함수 재사용 (src/ 를 import 경로에 추가)
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from analyze import decompose, extract_features, lifecycle_stage  # noqa: E402

MODEL_PATH = ROOT / "model" / "cluster_model.joblib"
COLOR = {"메가": "#2563eb", "마이크로": "#dc2626", "스테디": "#16a34a"}
STAGE_BADGE = {"도입": "#64748b", "성장": "#16a34a", "정점/성숙": "#2563eb",
               "쇠퇴": "#dc2626", "쇠퇴/잔존": "#f59e0b"}
STRATEGY = {
    ("스테디", "도입"): "시즌 진입 초기 — 시즌 캘린더 기준 사전 발주 준비, 초기 물량은 소량.",
    ("스테디", "성장"): "수요 상승 중인 계절 핵심 품목 — 물량 확대, 사이즈·컬러 확장.",
    ("스테디", "정점/성숙"): "계절 피크 — 캐리오버 스테디셀러로 운영, 원가·효율 관리.",
    ("스테디", "쇠퇴/잔존"): "피크 지난 계절품 — 시즌오프 마크다운 소진, 검증된 코어만 재발주.",
    ("스테디", "쇠퇴"): "수요 식은 계절품 — 재고 최소화, 다음 시즌 재발주 재검토.",
    ("메가", "도입"): "부상 초기 장기 후보 — 핵심 라인 편입 검토, 물량 점진 확대.",
    ("메가", "성장"): "성장 중인 장기 트렌드 — 코어 상품화, 장기 발주·시그니처 라인 투자.",
    ("메가", "정점/성숙"): "고점 안정 스테디셀러 — 캐리오버 운영, 원가·효율 중심.",
    ("메가", "쇠퇴/잔존"): "관심 둔화 — 캐리오버로 효율 운영, 신규 투자 자제·베이직만 유지.",
    ("메가", "쇠퇴"): "하락 전환 — 라인 축소, 기본형 위주, 발주량 단계 축소.",
    ("마이크로", "도입"): "신규 부상 트렌드 — 소량·퀵리스폰스 테스트, 반응 보고 즉시 리오더(한정).",
    ("마이크로", "성장"): "빠르게 확산 — test-and-repeat 소량 반복 발주, SNS 집중.",
    ("마이크로", "정점/성숙"): "정점 도달 — 추가 발주 보수적, 빠른 소진 준비.",
    ("마이크로", "쇠퇴/잔존"): "열기 식는 중 — 추가 발주 중단, 마크다운 정리.",
    ("마이크로", "쇠퇴"): "열기 식음 — 추가 발주 중단, 신속 철수·마크다운으로 재고 최소화.",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trend(keyword: str) -> pd.Series | None:
    """구글 트렌드(today 5-y · geo=KR · 주별)를 수집해 주별 Series로 반환."""
    pt = TrendReq(hl="ko-KR", tz=540)
    pt.build_payload([keyword], timeframe="today 5-y", geo="KR")
    df = pt.interest_over_time()
    if df.empty or keyword not in df.columns:
        return None
    s = df[keyword].astype(float)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="W")
    return s.reindex(grid).interpolate(limit_direction="both")


def stl_chart(comp: pd.DataFrame) -> go.Figure:
    titles = ("원본(관심도)", "트렌드 T", "계절성 S", "잔차 R")
    colors = ["#0f172a", "#2563eb", "#16a34a", "#dc2626"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=titles, vertical_spacing=0.06)
    for r, (col, c) in enumerate(zip(["observed", "trend", "seasonal", "resid"], colors), 1):
        fig.add_trace(go.Scatter(x=comp.index, y=comp[col], mode="lines",
                                 line=dict(color=c, width=1.5), showlegend=False), row=r, col=1)
    for ann, c in zip(fig.layout.annotations, colors):
        ann.font.color = c
    fig.update_layout(height=620, template="plotly_white", margin=dict(t=40))
    return fig


# ---------------------------------------------------------------------------
st.set_page_config(page_title="패션 트렌드 실시간 분류기", page_icon="👗")
st.title("👗 패션 트렌드 실시간 분류기")
st.caption("키워드를 입력하면 구글 트렌드 데이터를 즉석 분석해 "
           "**메가 · 마이크로 · 스테디** 분류와 **수명주기 단계**를 알려줍니다.")

model = load_model()
keyword = st.text_input("패션 아이템 키워드", placeholder="예: 버킷햇, 카고팬츠, 발레코어 …")

if st.button("분석하기", type="primary") and keyword.strip():
    kw = keyword.strip()
    try:
        with st.spinner(f"'{kw}' 구글 트렌드 데이터 수집 중…"):
            s = fetch_trend(kw)
    except Exception as e:
        st.error(f"데이터 수집 실패 (구글 트렌드 차단 가능): {type(e).__name__}. 잠시 후 다시 시도해 주세요.")
        st.stop()

    if s is None or len(s) < 110:
        st.error("데이터가 부족해 분석할 수 없어요. (검색량이 너무 적거나 수집 실패)")
        st.stop()

    nonzero = int((s > 0).sum())
    if nonzero < 20:
        st.warning(f"⚠️ 검색량이 매우 적어요(검색 있던 주 {nonzero}/{len(s)}). 결과 신뢰도가 낮을 수 있습니다.")

    # STL 분해 → 지표 → 분류 → 단계
    comp, _ = decompose(s)
    feat = extract_features(comp)
    X = model["scaler"].transform([[feat[f] for f in model["features"]]])
    cluster = model["cluster_names"][int(model["kmeans"].predict(X)[0])]
    stage = lifecycle_stage(feat)
    strat = STRATEGY.get((cluster, stage), "—")

    # --- 결과 표시 ---
    st.divider()
    c1, c2 = st.columns(2)
    c1.markdown(f"### 분류 · <span style='color:{COLOR[cluster]}'>{cluster}</span>",
                unsafe_allow_html=True)
    c2.markdown(f"### 단계 · <span style='color:{STAGE_BADGE[stage]}'>{stage}</span>",
                unsafe_allow_html=True)
    st.info(f"**실무 전략:** {strat}")

    with st.expander("📊 산출된 지표 보기"):
        st.table(pd.DataFrame({
            "지표": ["트렌드강도(F_T)", "계절성(F_S)", "기울기", "지속기간",
                    "잔차변동", "첨예도", "피크시점"],
            "값": [round(feat[f], 3) for f in model["features"]],
        }))

    st.subheader("STL 분해")
    st.plotly_chart(stl_chart(comp), use_container_width=True)

st.divider()
st.caption("※ 검색 관심도는 구글 트렌드의 검색량(정규화 상대값)입니다. · "
           "200개 학습 모델 기준 분류 · 분석: STL + K-means(k=3)")
