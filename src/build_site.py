"""
build_site.py — 분석 결과를 정적 HTML 웹페이지로 빌드 (Plotly)

입력:  data/processed/features.csv, data/processed/components/<slug>.csv
출력:  docs/index.html  (GitHub Pages 배포용, 단일 파일)

구성: 개요 → 사분면 지도 → STL 분해 뷰어 → 수명주기/분류 표 → 전략 매트릭스 → 한계
실행: /opt/anaconda3/envs/env_stl/bin/python src/build_site.py
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
COMP = PROC / "components"
DOCS = ROOT / "docs"

FONT_FAMILY = "Pretendard, -apple-system, BlinkMacSystemFont, sans-serif"
COLOR = {"메가": "#2563eb", "마이크로": "#dc2626", "스테디": "#16a34a"}
STAGE_BADGE = {
    "도입": "#64748b", "성장": "#16a34a",
    "정점/성숙": "#2563eb", "쇠퇴": "#dc2626", "쇠퇴/잔존": "#f59e0b",
}
# 수명주기 흐름 순서 + 한 줄 의미 (단계 범례용)
STAGE_LEGEND = [
    ("도입", "관심도 상승 초기 단계로, 절대 수준은 낮음"),
    ("성장", "관심도가 빠르게 확대되는 국면"),
    ("정점/성숙", "최고 수준에서 안정적으로 유지"),
    ("쇠퇴/잔존", "정점 대비 하락 후 낮은 수준에서 정체"),
    ("쇠퇴", "관심도가 지속적으로 감소하는 국면"),
]


def stage_legend_html() -> str:
    """표와 동일한 컬러 뱃지를 수명주기 순서대로 나열한 범례."""
    items = []
    for name, desc in STAGE_LEGEND:
        c = STAGE_BADGE.get(name, "#888")
        items.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0">'
            f'<span style="background:{c};color:#fff;padding:2px 10px;border-radius:10px;'
            f'font-size:12px">{name}</span>'
            f'<span class="note" style="font-size:12px">{desc}</span></span>'
        )
    return ('<div style="margin:10px 0 6px">'
            '<b style="font-size:13px">수명주기 단계</b><br>'
            + "".join(items) + '</div>')


def fig_quadrant(df: pd.DataFrame) -> go.Figure:
    """사분면 지도: X=지속성, Y=변동성, 색=분류, hover=이름·단계."""
    fig = go.Figure()
    for label, g in df.groupby("cluster_label"):
        fig.add_trace(go.Scatter(
            x=g["persistence"], y=g["volatility"],
            mode="markers+text",
            text=g["display_name"], textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=14, color=COLOR.get(label, "#888"),
                        line=dict(width=1, color="white")),
            name=label,
            customdata=g[["stage", "F_T", "F_S", "cv_resid"]].to_numpy(),
            hovertemplate=("<b>%{text}</b><br>분류: " + label +
                           "<br>단계: %{customdata[0]}"
                           "<br>지속성: %{x:.2f} / 변동성: %{y:.2f}"
                           "<br>F_T=%{customdata[1]:.2f}, F_S=%{customdata[2]:.2f}"
                           "<extra></extra>"),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
    fig.add_vline(x=0, line_dash="dot", line_color="#cbd5e1")
    # 사분면 주석
    ann = [(-0.7, 1.6, "마이크로형<br>(단기·스파이크)"),
           (0.9, 1.6, "버즈/변동 메가"),
           (-0.7, -1.4, "스테디/저변동"),
           (0.9, -1.4, "메가형<br>(장기·안정)")]
    for ax, ay, t in ann:
        fig.add_annotation(x=ax, y=ay, text=t, showarrow=False,
                           font=dict(size=11, color="#94a3b8"))
    fig.update_layout(
        title="트렌드 라이프사이클 사분면 지도",
        xaxis_title="지속성 (오른쪽일수록 오래감)",
        yaxis_title="변동성 (위일수록 들쭉날쭉)",
        height=620, template="plotly_white",
        legend=dict(orientation="h", y=1.05, x=0),
        font=dict(family=FONT_FAMILY),
    )
    return fig


def fig_stl_viewer(df: pd.DataFrame) -> go.Figure | None:
    """드롭다운으로 아이템 선택 → 원본/트렌드/계절성/잔차 4단 그래프."""
    slugs = [s for s in df.index if (COMP / f"{s}.csv").exists()]
    if not slugs:
        return None
    # 분류 표와 동일 순서: 스테디 → 메가 → 마이크로 (그룹 내 순서는 유지)
    slugs = sorted(slugs, key=lambda s: CLUSTER_ORDER.get(df.loc[s, "cluster_label"], 99))
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("원본(관심도)", "트렌드 T", "계절성 S", "잔차 R"),
                        vertical_spacing=0.06)
    series_names = ["observed", "trend", "seasonal", "resid"]
    colors = ["#0f172a", "#2563eb", "#16a34a", "#dc2626"]
    traces_per_item = len(series_names)
    for i, slug in enumerate(slugs):
        comp = pd.read_csv(COMP / f"{slug}.csv", parse_dates=["date"])
        for r, (col, c) in enumerate(zip(series_names, colors), start=1):
            fig.add_trace(go.Scatter(
                x=comp["date"], y=comp[col], mode="lines",
                line=dict(color=c, width=1.5), showlegend=False,
                visible=(i == 0),
            ), row=r, col=1)

    # 드롭다운 버튼
    total = len(slugs) * traces_per_item
    buttons = []
    for i, slug in enumerate(slugs):
        vis = [False] * total
        for k in range(traces_per_item):
            vis[i * traces_per_item + k] = True
        name = df.loc[slug, "display_name"]
        label = df.loc[slug, "cluster_label"]
        stage = df.loc[slug, "stage"]
        buttons.append(dict(
            label=f"{name} ({label}·{stage})",
            method="update",
            args=[{"visible": vis}],
        ))
    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons, x=0, y=1.16, xanchor="left", yanchor="bottom",
            showactive=True, direction="down",
            bgcolor="#ffffff", bordercolor="#cbd5e1", borderwidth=1,
            font=dict(family=FONT_FAMILY, size=14, color="#1e293b"),
            pad={"l": 8, "r": 8, "t": 6, "b": 6},
        )],
        height=740, template="plotly_white", margin=dict(t=150),
        font=dict(family=FONT_FAMILY),
    )
    # 드롭다운 위 안내 라벨
    fig.add_annotation(x=0, y=1.28, xref="paper", yref="paper",
                       xanchor="left", showarrow=False,
                       text="▼ 아이템 선택",
                       font=dict(family=FONT_FAMILY, size=12, color="#64748b"))
    return fig


CLUSTER_ORDER = {"스테디": 0, "메가": 1, "마이크로": 2}


def html_table(df: pd.DataFrame) -> str:
    show = df.reset_index()[[
        "display_name", "cluster_label", "rule_label", "stage",
        "F_T", "F_S", "slope_norm", "duration_ratio", "cv_resid", "peak_sharpness",
    ]].copy()
    # 분류(군집) 스테디 → 메가 → 마이크로 순 정렬
    show = show.sort_values(
        by="cluster_label",
        key=lambda s: s.map(CLUSTER_ORDER).fillna(99),
        kind="stable",
    )
    show.columns = ["아이템", "분류(군집)", "분류(규칙)", "단계",
                    "F_T", "F_S", "기울기", "지속기간", "잔차변동", "첨예도"]
    rows = []
    for _, r in show.iterrows():
        badge_c = COLOR.get(r["분류(군집)"], "#888")
        stage_c = STAGE_BADGE.get(r["단계"], "#888")
        tds = "".join(
            f"<td>{v}</td>" for v in [
                r["아이템"],
                f'<b style="color:{badge_c}">{r["분류(군집)"]}</b>',
                r["분류(규칙)"],
                f'<span style="background:{stage_c};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">{r["단계"]}</span>',
                r["F_T"], r["F_S"], r["기울기"], r["지속기간"], r["잔차변동"], r["첨예도"],
            ])
        rows.append(f"<tr>{tds}</tr>")
    head = "".join(f"<th>{c}</th>" for c in show.columns)
    return f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


STRATEGY_HTML = """
<table class="data">
<thead><tr><th>분류 · 단계</th><th>실무 전략</th></tr></thead>
<tbody>
<tr><td><b style="color:#2563eb">메가 · 성장기</b></td><td>코어 상품화, 장기 리드타임 발주 OK, 시그니처 라인 투자</td></tr>
<tr><td><b style="color:#2563eb">메가 · 정점/성숙</b></td><td>캐리오버 스테디셀러 운영, 원가·효율 중심</td></tr>
<tr><td><b style="color:#dc2626">마이크로 · 성장기</b></td><td>소량·퀵리스폰스, test-and-repeat, 한정수량, SNS 집중</td></tr>
<tr><td><b style="color:#dc2626">마이크로 · 쇠퇴기</b></td><td>빠른 철수·과감한 마크다운, 재고 최소화</td></tr>
<tr><td><b style="color:#16a34a">스테디</b></td><td>계절 피크 역산 발주 캘린더, 사전 생산</td></tr>
</tbody></table>
"""

PAGE_CSS = """
<link rel="stylesheet" as="style" crossorigin
 href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<style>
 body{font-family:'Pretendard',-apple-system,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;
      max-width:1040px;margin:0 auto;padding:24px;color:#0f172a;line-height:1.6}
 h1{font-size:28px;margin-bottom:4px} h2{margin-top:48px;border-bottom:2px solid #e2e8f0;padding-bottom:6px}
 .sub{color:#64748b;margin-top:0}
 .meta{margin:14px 0 4px;border-top:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;
       padding:10px 0;font-size:12px;color:#475569}
 .meta dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:0}
 .meta dt{font-weight:600;color:#334155;white-space:nowrap}
 .meta dt::after{content:' :'}
 .meta dd{margin:0;color:#475569}
 .card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;margin:16px 0}
 table.data{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}
 table.data th,table.data td{border:1px solid #e2e8f0;padding:6px 10px;text-align:center}
 table.data th{background:#f1f5f9}
 .note{font-size:13px;color:#64748b}
 footer{margin-top:48px;color:#94a3b8;font-size:12px;border-top:1px solid #e2e8f0;padding-top:16px}
</style>
"""


def build():
    df = pd.read_csv(PROC / "features.csv").set_index("slug")
    DOCS.mkdir(exist_ok=True)

    q = to_html(fig_quadrant(df), full_html=False, include_plotlyjs="cdn")
    stl_fig = fig_stl_viewer(df)
    stl = to_html(stl_fig, full_html=False, include_plotlyjs=False) if stl_fig else ""

    n = len(df)
    counts = df["cluster_label"].value_counts().to_dict()
    summary = " · ".join(f"{k} {v}개" for k, v in counts.items())

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>패션 트렌드 수명주기 분류</title>{PAGE_CSS}</head><body>

<h1>패션 아이템 수명주기 기반 트렌드 분류</h1>
<div class="meta"><dl>
 <dt>데이터 출처</dt><dd>Google Trends (검색 관심도)</dd>
 <dt>분석 방법</dt><dd>STL 시계열 분해 · 군집화(K-means) + 규칙기반 분류</dd>
 <dt>대상 지역</dt><dd>대한민국 (geo=KR)</dd>
 <dt>분석 기간</dt><dd>최근 5년 · 주별(weekly)</dd>
 <dt>분석 대상</dt><dd>패션 아이템 {n}개 (메가·마이크로·스테디)</dd>
 <dt>작성일</dt><dd>2026-06-11</dd>
</dl></div>

<div class="card">
 <b>한눈에:</b> 검색 관심도의 <b>형태</b>를 STL로 분해해 아이템을 <b>{summary}</b>로 분류하고,
 각 아이템의 <b>수명주기 단계</b>와 <b>실무 전략</b>을 도출했습니다.
 <span class="note">※ 수치는 구글 트렌드의 검색량(정규화 상대값)입니다.</span>
</div>

<h2>① 트렌드 라이프사이클 사분면 지도</h2>
<p class="note">오른쪽으로 갈수록 오래 지속(메가형), 위로 갈수록 변동이 큼(마이크로형). 색은 군집 분류 결과.</p>
{q}

<h2>② STL 분해 뷰어</h2>
<p class="note">드롭다운에서 아이템을 선택하면 원본·트렌드·계절성·잔차로 분해된 그래프를 볼 수 있습니다.</p>
{stl}
<div class="card note" style="margin-top:8px">
 STL 분해는 하나의 검색 곡선을 <b>3가지로 쪼갠 것</b>입니다. (원본 = 트렌드 + 계절성 + 잔차)
 <ul style="margin:8px 0 0;padding-left:20px">
  <li><b style="color:#2563eb">트렌드 (T)</b> — 계절 영향을 걷어낸 <b>장기 흐름</b>. 전반적으로 뜨는지/지는지.</li>
  <li><b style="color:#16a34a">계절성 (S)</b> — 매년 <b>같은 시기에 반복</b>되는 패턴(예: 패딩은 겨울마다 ↑).</li>
  <li><b style="color:#dc2626">잔차 (R)</b> — 트렌드·계절성으로 <b>설명 안 되는 나머지</b> 흔들림(불규칙·노이즈).</li>
 </ul>
</div>

<h2>③ 분류 · 수명주기 단계</h2>
<p class="note">‘분류(군집)’은 데이터가 스스로 묶어준 결과, ‘분류(규칙)’은 간단한 규칙으로 매긴 결과입니다. 분류(군집) 기준 스테디·메가·마이크로 순으로 정렬했습니다.</p>
{stage_legend_html()}
{html_table(df)}

<h2>④ 분면 × 단계 비즈니스 전략</h2>
{STRATEGY_HTML}

<h2>지표 설명 & 한계</h2>
<p class="note">각 수치가 어떻게 나오는지 — 수식과 쉬운 말 풀이입니다. (T=트렌드, S=계절성, R=잔차, Y=원본)</p>
<table class="data">
<thead><tr><th>지표</th><th>계산식</th><th>쉽게 말하면</th></tr></thead>
<tbody>
<tr><td><b>F_T</b> 트렌드 강도</td><td>max(0, 1 − Var(R) / Var(T+R))</td>
    <td>잔차(설명 안 되는 흔들림)가 작을수록 1에 가까움 → <b>추세가 또렷</b></td></tr>
<tr><td><b>F_S</b> 계절성 강도</td><td>max(0, 1 − Var(R) / Var(S+R))</td>
    <td>1에 가까울수록 <b>매년 반복되는 계절 패턴</b>이 뚜렷</td></tr>
<tr><td><b>기울기</b> slope</td><td>트렌드 T에 직선을 맞춘 기울기(정규화)</td>
    <td>양수면 <b>상승세</b>, 음수면 <b>하락세</b> (장기 방향)</td></tr>
<tr><td><b>지속기간</b></td><td>(Y ≥ 피크의 50%)인 주의 비율</td>
    <td>인기가 <b>오래 유지</b>될수록 큼 (반짝 유행은 작음)</td></tr>
<tr><td><b>잔차변동</b> CV</td><td>std(R) / mean(Y)</td>
    <td>평소 대비 <b>들쭉날쭉한 정도</b> (불안정성)</td></tr>
<tr><td><b>첨예도</b></td><td>피크 / (검색 있던 주들의 중앙값)</td>
    <td>평소보다 <b>확 튀어오른 정도</b> (스파이크성)</td></tr>
<tr><td><b>단계</b> 수명주기</td>
    <td>현재비율 = 현재값 / 역대피크 &nbsp;+&nbsp; 최근 1년 트렌드 방향</td>
    <td>오름+40%↑=<b>성장</b> / 오름+40%↓=<b>도입</b> / 내림=<b>쇠퇴</b> / 정체+70%↑=<b>정점·성숙</b> / 정체+70%↓=<b>쇠퇴·잔존</b></td></tr>
</tbody></table>
<div class="card note" style="margin-top:16px">
 <b>한계</b> ① 구글 트렌드 검색량은 정규화 상대값이라 절대량 비교 불가
 ② 마이너 신조어는 표본 부족 노이즈 ③ 메가/마이크로 라벨은 사전 가정→군집으로 사후 검증.
</div>

<footer>패션 트렌드 수명주기 분류 프로젝트 · Google Trends 데이터 기반 · 분석/시각화: Python(statsmodels STL, scikit-learn, Plotly)</footer>
</body></html>"""

    out = DOCS / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ 웹페이지 빌드 완료 → {out}")
    print(f"   브라우저에서 열기: file://{out}")


if __name__ == "__main__":
    build()
