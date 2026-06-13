"""
build_site.py — 분석 결과를 정적 HTML 웹페이지로 빌드 (Plotly)

입력:  data/processed/features.csv, data/processed/components/<slug>.csv
출력:  docs/index.html  (GitHub Pages 배포용, 단일 파일)

구성: 개요 → 사분면 지도 → STL 분해 뷰어 → 수명주기/분류 표 → 전략 매트릭스 → 한계
실행: /opt/anaconda3/envs/env_stl/bin/python src/build_site.py
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
COMP = PROC / "components"
# 배포: GitHub Pages가 저장소 루트 /docs 를 서빙 → 거기로 빌드
DOCS = ROOT.parent / "docs"

FONT_FAMILY = "Pretendard, -apple-system, BlinkMacSystemFont, sans-serif"
# 실시간 분류기(Streamlit Cloud) 임베드 URL
STREAMLIT_URL = "https://gqfhgm689dykskx6puypjk.streamlit.app"
COLOR = {"메가": "#2563eb", "마이크로": "#dc2626", "스테디": "#16a34a"}
STAGE_BADGE = {
    "도입": "#64748b", "성장": "#16a34a",
    "정점/성숙": "#2563eb", "쇠퇴": "#dc2626", "쇠퇴/잔존": "#f59e0b",
}
# 수명주기 흐름 순서 + 한 줄 의미 (단계 범례용)
STAGE_LEGEND = [
    ("도입", "상승 초기 단계"),
    ("성장", "빠르게 상승"),
    ("정점/성숙", "최고 수준에서 유지"),
    ("쇠퇴/잔존", "정점 대비 하락 후 정체"),
    ("쇠퇴", "지속적으로 감소"),
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


CLUSTER_FEATURES = ["F_T", "F_S", "slope_norm", "duration_ratio",
                    "cv_resid", "peak_sharpness", "time_to_peak"]


def _highlight_trace() -> go.Scatter:
    """선택한 아이템을 감싸는 강조용 빈 trace(링 + 이름)."""
    return go.Scatter(
        x=[None], y=[None], mode="markers+text", name="선택",
        text=[""], textposition="top center",
        textfont=dict(family=FONT_FAMILY, size=14, color="#0f172a"),
        marker=dict(size=22, symbol="circle-open", color="#0f172a",
                    line=dict(width=3, color="#0f172a")),
        showlegend=False, hoverinfo="skip",
    )


def fig_clusters(df: pd.DataFrame):
    """군집화 결과 시각화: 7개 지표를 PCA로 2D 투영, 색=군집.
    반환: (figure, name→(pc1,pc2) 맵, 강조 trace 인덱스)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES].to_numpy())
    pc = PCA(n_components=2).fit_transform(X)
    d = df.copy()
    d["pc1"], d["pc2"] = pc[:, 0], pc[:, 1]
    fig = go.Figure()
    for label, g in d.groupby("cluster_label"):
        fig.add_trace(go.Scatter(
            x=g["pc1"], y=g["pc2"], mode="markers",
            marker=dict(size=11, color=COLOR.get(label, "#888"),
                        line=dict(width=1, color="white")),
            name=label, text=g["display_name"],
            hovertemplate="<b>%{text}</b><br>군집: " + label + "<extra></extra>",
        ))
    hl_index = len(fig.data)
    fig.add_trace(_highlight_trace())
    fig.update_layout(
        xaxis_title="주성분 1 (PC1)", yaxis_title="주성분 2 (PC2)",
        height=560, template="plotly_white",
        legend=dict(orientation="h", y=1.06, x=0),
        font=dict(family=FONT_FAMILY),
    )
    name2pc = {r.display_name: (round(float(r.pc1), 4), round(float(r.pc2), 4))
               for r in d.itertuples()}
    return fig, name2pc, hl_index


def fig_quadrant(df: pd.DataFrame):
    """사분면 지도: X=지속성, Y=변동성, 색=분류, hover=이름·단계.
    반환: (figure, 강조 trace 인덱스)."""
    fig = go.Figure()
    for label, g in df.groupby("cluster_label"):
        fig.add_trace(go.Scatter(
            x=g["persistence"], y=g["volatility"],
            mode="markers",
            text=g["display_name"],
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
    hl_index = len(fig.data)
    fig.add_trace(_highlight_trace())
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
    fig.add_vline(x=0, line_dash="dot", line_color="#cbd5e1")
    fig.update_layout(
        xaxis_title="지속성 = 트렌드강도 + 지속기간",
        yaxis_title="변동성 = 잔차변동 + 첨예도",
        height=620, template="plotly_white",
        legend=dict(orientation="h", y=1.05, x=0),
        font=dict(family=FONT_FAMILY),
    )
    return fig, hl_index


STL_DIV_ID = "stl-viewer"


def fig_stl_viewer(df: pd.DataFrame):
    """아이템 선택 → 원본/트렌드/계절성/잔차 4단 그래프.
    선택 UI는 HTML <select>(검색 가능)로 분리하고, 여기선 그래프(전 아이템 trace)만 만든다.
    반환: (figure, items)  items=순서대로 {name,cluster,stage}."""
    slugs = [s for s in df.index if (COMP / f"{s}.csv").exists()]
    if not slugs:
        return None, []
    # 아이템 이름 가나다 순
    slugs = sorted(slugs, key=lambda s: str(df.loc[s, "display_name"]))
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("원본(관심도)", "트렌드 T", "계절성 S", "잔차 R"),
                        vertical_spacing=0.06)
    series_names = ["observed", "trend", "seasonal", "resid"]
    colors = ["#0f172a", "#2563eb", "#16a34a", "#dc2626"]
    for i, slug in enumerate(slugs):
        comp = pd.read_csv(COMP / f"{slug}.csv", parse_dates=["date"])
        for r, (col, c) in enumerate(zip(series_names, colors), start=1):
            fig.add_trace(go.Scatter(
                x=comp["date"], y=comp[col], mode="lines",
                line=dict(color=c, width=1.5), showlegend=False,
                visible=False,  # 선택 전엔 모두 숨김(빈 차트)
            ), row=r, col=1)
    fig.update_layout(height=720, template="plotly_white",
                      margin=dict(t=40), font=dict(family=FONT_FAMILY))
    # 소제목을 각 선 색과 일치시킴 (원본·트렌드·계절성·잔차)
    for ann, c in zip(fig.layout.annotations, colors):
        ann.font.color = c
        ann.font.size = 14
    items = [{"name": df.loc[s, "display_name"],
              "cluster": df.loc[s, "cluster_label"],
              "stage": df.loc[s, "stage"]} for s in slugs]
    return fig, items


def stl_select_html(items: list, sel_id: str) -> str:
    """가나다 순 HTML <select>. 선택 시 selectItem(이름)으로 ①②③ 연동(JS는 build에서).
    같은 class(item-select)로 ①·③ 드롭다운을 동기화한다."""
    opts = ['<option value="" data-name="" selected>— 아이템을 선택하세요 —</option>']
    for i, it in enumerate(items):
        opts.append(f'<option value="{i}" data-name="{it["name"]}">'
                    f'{it["name"]}</option>')
    return f'''<div style="margin:8px 0">
 <label style="font-size:13px;color:#475569;margin-right:8px">아이템 선택</label>
 <select id="{sel_id}" class="item-select"
   onchange="selectItem(this.options[this.selectedIndex].dataset.name)"
   style="font-family:{FONT_FAMILY};font-size:14px;padding:6px 10px;border:1px solid #cbd5e1;border-radius:6px;min-width:280px">
{"".join(opts)}
 </select>
</div>'''


CLUSTER_ORDER = {"스테디": 0, "메가": 1, "마이크로": 2}


def html_table(df: pd.DataFrame) -> str:
    show = df.reset_index()[[
        "display_name", "cluster_label", "stage",
        "F_T", "F_S", "slope_norm", "duration_ratio", "cv_resid", "peak_sharpness",
    ]].copy()
    # 분류 스테디 → 메가 → 마이크로 순, 그 안에서는 아이템 가나다 순
    show["_corder"] = show["cluster_label"].map(CLUSTER_ORDER).fillna(99)
    show = show.sort_values(by=["_corder", "display_name"], kind="stable")
    show = show.drop(columns="_corder")
    show.columns = ["아이템", "분류", "단계",
                    "F_T", "F_S", "기울기", "지속기간", "잔차변동", "첨예도"]
    rows = []
    for _, r in show.iterrows():
        badge_c = COLOR.get(r["분류"], "#888")
        stage_c = STAGE_BADGE.get(r["단계"], "#888")
        tds = "".join(
            f"<td>{v}</td>" for v in [
                r["아이템"],
                f'<b style="color:{badge_c}">{r["분류"]}</b>',
                f'<span style="background:{stage_c};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">{r["단계"]}</span>',
                r["F_T"], r["F_S"], r["기울기"], r["지속기간"], r["잔차변동"], r["첨예도"],
            ])
        rows.append(f"<tr>{tds}</tr>")
    head = "".join(f"<th>{c}</th>" for c in show.columns)
    return f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


# 분류 × 단계별 실무 전략 (보고서 말투)
STRATEGY_BY = {
    ("스테디", "도입"): "시즌 진입 초기 단계로, 시즌 캘린더에 따라 사전 발주를 준비하고 초기 물량은 소량으로 운영한다.",
    ("스테디", "성장"): "수요가 상승 중인 계절 핵심 품목으로, 물량을 확대하고 사이즈·컬러를 확장하며 주력 매대에 배치한다.",
    ("스테디", "정점/성숙"): "계절 피크에 도달한 핵심 품목으로, 캐리오버 스테디셀러로 운영하며 원가·효율을 관리한다.",
    ("스테디", "쇠퇴/잔존"): "계절 피크가 지난 품목으로, 시즌오프 마크다운으로 재고를 소진하고 다음 시즌에는 검증된 핵심 품목만 재발주한다.",
    ("스테디", "쇠퇴"): "수요가 식은 계절 품목으로, 재고를 최소화하고 다음 시즌 재발주 여부를 재검토한다.",
    ("메가", "도입"): "부상 초기의 장기 후보 품목으로, 핵심 라인 편입을 검토하며 물량을 점진적으로 확대한다.",
    ("메가", "성장"): "성장 중인 장기 트렌드로, 코어 상품화하고 장기 리드타임 발주와 시그니처 라인에 투자한다.",
    ("메가", "정점/성숙"): "고점에서 안정적인 스테디셀러로, 캐리오버 운영하며 원가·효율 중심으로 관리한다.",
    ("메가", "쇠퇴/잔존"): "한때 강세였으나 관심도가 둔화된 품목으로, 캐리오버 스테디셀러로 효율 중심 운영하며 신규 투자는 자제하고 베이직 라인만 유지한다.",
    ("메가", "쇠퇴"): "하락세로 전환한 품목으로, 라인을 축소하고 기본형 위주로 운영하며 신상 비중과 발주량을 단계적으로 축소한다.",
    ("마이크로", "도입"): "신규 부상 트렌드로, 소량·퀵리스폰스 방식으로 운영하고 반응에 따라 즉시 추가 발주하되 수량을 한정한다.",
    ("마이크로", "성장"): "빠르게 확산 중인 단기 트렌드로, test-and-repeat으로 소량 반복 발주하고 SNS 마케팅에 집중한다.",
    ("마이크로", "정점/성숙"): "정점에 도달한 단기 트렌드로, 추가 발주를 보수적으로 가져가며 빠른 소진을 준비한다.",
    ("마이크로", "쇠퇴/잔존"): "열기가 식어가는 단기 트렌드로, 추가 발주를 중단하고 마크다운으로 재고를 정리한다.",
    ("마이크로", "쇠퇴"): "열기가 식은 품목으로, 추가 발주를 중단하고 신속히 철수하며 마크다운으로 재고를 최소화한다.",
}


def strategy_html(df: pd.DataFrame) -> str:
    """실제 (분류 × 단계) 그룹별로 해당 아이템 + 실무 전략을 표로 생성."""
    oc = {"스테디": 0, "메가": 1, "마이크로": 2}
    os_ = {"도입": 0, "성장": 1, "정점/성숙": 2, "쇠퇴/잔존": 3, "쇠퇴": 4}
    g = (df.reset_index().groupby(["cluster_label", "stage"])["display_name"]
         .apply(list).reset_index())
    g["co"] = g.cluster_label.map(oc).fillna(9)
    g["so"] = g.stage.map(os_).fillna(9)
    g = g.sort_values(["co", "so"])
    rows = []
    for _, r in g.iterrows():
        cl, st = r.cluster_label, r.stage
        cc, sc = COLOR.get(cl, "#888"), STAGE_BADGE.get(st, "#888")
        items = "、".join(sorted(r.display_name))
        strat = STRATEGY_BY.get((cl, st), "—")
        badge = (f'<span style="background:{sc};color:#fff;padding:2px 8px;'
                 f'border-radius:10px;font-size:12px">{st}</span>')
        rows.append(
            f'<tr><td style="white-space:nowrap"><b style="color:{cc}">{cl}</b></td>'
            f'<td style="white-space:nowrap">{badge}</td>'
            f'<td style="text-align:left">{items}</td>'
            f'<td style="text-align:left">{strat}</td></tr>')
    return ('<table class="data" style="table-layout:fixed">\n'
            '<colgroup><col style="width:60px"><col style="width:96px">'
            '<col style="width:26%"><col></colgroup>\n'
            '<thead><tr><th style="white-space:nowrap">분류</th>'
            '<th style="white-space:nowrap">단계</th>'
            '<th>해당 아이템</th><th>실무 전략</th></tr></thead>\n'
            '<tbody>\n' + "\n".join(rows) + '\n</tbody></table>')

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

    cl_fig, name2pc, cl_hl = fig_clusters(df)
    q_fig, q_hl = fig_quadrant(df)
    cl = to_html(cl_fig, full_html=False, include_plotlyjs="cdn", div_id="cluster-chart")
    q = to_html(q_fig, full_html=False, include_plotlyjs=False, div_id="quadrant-chart")
    stl_fig, stl_items = fig_stl_viewer(df)
    if stl_fig is not None:
        stl_select_top = stl_select_html(stl_items, "item-select-1")
        stl_select_btm = stl_select_html(stl_items, "item-select-3")
        stl = to_html(stl_fig, full_html=False, include_plotlyjs=False,
                      div_id=STL_DIV_ID)
    else:
        stl_select_top, stl_select_btm, stl = "", "", ""

    # ①②③ 연동: 아이템명 → 좌표·STL인덱스 맵 + 클릭/선택 스크립트
    stl_index = {it["name"]: i for i, it in enumerate(stl_items)}
    items_map = {}
    for r in df.reset_index().itertuples():
        nm = r.display_name
        if nm in stl_index:
            items_map[nm] = {"c": list(name2pc[nm]),
                             "q": [round(float(r.persistence), 4),
                                   round(float(r.volatility), 4)],
                             "s": stl_index[nm],
                             "cl": r.cluster_label, "st": r.stage}
    default_name = stl_items[0]["name"] if stl_items else ""
    link_js = f"""<script>
var ITEMS = {json.dumps(items_map, ensure_ascii=False)};
var STL_N = {len(stl_items)}, CL_HL = {cl_hl}, Q_HL = {q_hl};
function stlShow(i){{
  var vis = [];
  for (var k = 0; k < STL_N * 4; k++) vis.push(Math.floor(k / 4) === i);
  Plotly.restyle('{STL_DIV_ID}', {{visible: vis}});
}}
function selectItem(name){{
  var it = ITEMS[name];
  if (!it) return;
  Plotly.restyle('cluster-chart', {{x: [[it.c[0]]], y: [[it.c[1]]], text: [[name]]}}, [CL_HL]);
  Plotly.restyle('quadrant-chart', {{x: [[it.q[0]]], y: [[it.q[1]]], text: [[name]]}}, [Q_HL]);
  stlShow(it.s);
  document.querySelectorAll('select.item-select').forEach(function(s){{ s.value = it.s; }});
  var t = document.getElementById('stl-title');
  if (t) t.innerHTML = name +
    '<span style="display:block;font-size:13px;font-weight:400;color:#64748b">'
    + it.cl + ' · ' + it.st + '</span>';
}}
window.addEventListener('load', function(){{
  var cd = document.getElementById('cluster-chart');
  var qd = document.getElementById('quadrant-chart');
  if (cd && cd.on) cd.on('plotly_click', function(e){{ selectItem(e.points[0].text); }});
  if (qd && qd.on) qd.on('plotly_click', function(e){{ selectItem(e.points[0].text); }});
}});
</script>"""

    n = len(df)
    counts = df["cluster_label"].value_counts().to_dict()
    summary = " · ".join(f"{k} {v}개" for k, v in counts.items())
    n_clusters = len(counts)

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>패션 트렌드 수명주기 분류</title>{PAGE_CSS}</head><body>

<h1>패션 아이템 수명주기 기반 트렌드 분류</h1>
<div class="meta"><dl>
 <dt>데이터 출처</dt><dd>Google Trends (검색 관심도)</dd>
 <dt>분석 방법</dt><dd>STL 시계열 분해 · 군집화(K-means) + 규칙기반 분류</dd>
 <dt>대상 지역</dt><dd>대한민국 (geo=KR)</dd>
 <dt>분석 기간</dt><dd>최근 5년 · 주별(weekly)</dd>
 <dt>분석 대상</dt><dd>패션 아이템 {n}개</dd>
 <dt>작성일</dt><dd>2026-06-12</dd>
</dl></div>

<div class="card">
 패션 아이템의 검색 관심도를 <b>STL로 트렌드·계절성·잔차로 분해한 지표를 활용</b>하여,
 <b>메가·마이크로·스테디 3개 군집</b>으로 분류하고,
 각 아이템의 <b>수명주기 단계</b>와 결합해 <b>실무 전략</b>을 도출했습니다.
 <br><span class="note">※ 검색 관심도는 구글 트렌드의 검색량(정규화 상대값)입니다.</span>
</div>

<h2>🔍 패션 트렌드 실시간 분류기</h2>
<p class="note">패션 아이템 키워드를 입력하면 구글 트렌드 데이터를 실시간으로 수집하고, 학습된 모델로 분류·수명주기 단계를 판정해 알려드립니다. (아래 156개 분석과 동일한 모델을 사용합니다.)</p>
<iframe src="{STREAMLIT_URL}/?embed=true" width="100%" height="860"
        style="border:1px solid #e2e8f0;border-radius:12px" loading="lazy"></iframe>

<div class="card" style="margin-top:44px">
 <b>아래 ①~⑤</b>는 위 실시간 분류기가 사용하는 <b>모델을 학습시킨 패션 아이템 156개</b>의 분석 결과입니다.
 군집화·사분면 지도·STL 분해·수명주기 단계·비즈니스 전략 순으로 살펴봅니다.
</div>

<h2>① 군집화 결과</h2>
<p class="note">STL 분해 성분과 원본 검색량에서 산출한 7개 지표(트렌드강도·계절성·기울기·지속기간·잔차변동·첨예도·피크시점)를 바탕으로 머신러닝(K-means 군집화)을 적용해 아이템을 메가·마이크로·스테디 {n_clusters}개 군집으로 분류했습니다. 아래 그림은 이 지표를 PCA로 2D에 나타낸 것으로, 색은 군집이며 가까이 모인 점일수록 성격이 비슷합니다.</p>
<p class="note" style="margin-bottom:4px">아이템을 선택하면 ①·② 차트에서 그 아이템의 위치가 테두리로 강조되고, ③ STL 분해 뷰어에서 해당 아이템의 분해 그래프를 볼 수 있습니다.</p>
{stl_select_top}
{cl}
<div style="font-size:12px;color:#94a3b8;line-height:1.9;margin:8px 0">
주성분 1 : 7개 지표를 종합해 아이템 간 차이가 가장 크게 나타나는 대표 축 (정보량 1순위)<br>
주성분 2 : 주성분 1과 직각을 이루며, 그다음으로 차이가 크게 나타나는 축 (정보량 2순위)<br>
두 축 모두 여러 지표가 섞인 값이므로, 축 숫자보다 색깔별로 점이 모여 있는지를 봅니다.
</div>

<h2>② 트렌드 라이프사이클 사분면 지도</h2>
<p class="note">①의 군집을 <b>해석 가능한 두 축(지속성·변동성)</b> 위에 다시 배치한 지도입니다.</p>
{q}
<div style="font-size:12px;color:#94a3b8;line-height:1.9;margin:8px 0">
트렌드강도 : 장기적으로 오르거나 내리는 추세가 또렷한 정도<br>
지속기간 : 인기가 오래 유지된 정도 (반짝 유행일수록 짧음)<br>
잔차변동 : 설명되지 않는 들쭉날쭉함 (불안정한 정도)<br>
첨예도 : 평소보다 갑자기 확 튀어오른 정도 (스파이크성)
</div>

<h2>③ STL 분해 뷰어</h2>
<p class="note">선택한 아이템의 원본·트렌드·계절성·잔차로 분해된 그래프를 볼 수 있습니다.</p>
{stl_select_btm}
<div id="stl-title" style="text-align:center;font-size:20px;font-weight:700;margin:16px 0 6px;color:#334155">아이템 미선택</div>
{stl}
<div class="card note" style="margin-top:8px">
 STL 분해는 하나의 검색 곡선을 <b>3가지로 쪼갠 것</b>입니다. (원본 = 트렌드 + 계절성 + 잔차)
 <ul style="margin:8px 0 0;padding-left:20px">
  <li><b style="color:#2563eb">트렌드 (T)</b> — 계절 영향을 걷어낸 <b>장기 흐름</b>. 전반적으로 뜨는지/지는지.</li>
  <li><b style="color:#16a34a">계절성 (S)</b> — 매년 <b>같은 시기에 반복</b>되는 패턴(예: 패딩은 겨울마다 ↑).</li>
  <li><b style="color:#dc2626">잔차 (R)</b> — 트렌드·계절성으로 <b>설명 안 되는 나머지</b> 흔들림(불규칙·노이즈).</li>
 </ul>
</div>

<h2>④ 분류 · 수명주기 단계</h2>
{stage_legend_html()}
{html_table(df)}

<h2>⑤ 분면 × 단계 비즈니스 전략</h2>
{strategy_html(df)}

<h2>지표 설명 & 한계</h2>
<p class="note">T=트렌드, S=계절성, R=잔차, Y=원본</p>
<table class="data">
<thead><tr><th>지표</th><th>계산식</th><th>설명</th></tr></thead>
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
    <td>오름+40%↑=<b>성장</b> / 오름+40%↓=<b>도입</b> / 내림=<b>쇠퇴</b> / 정체+70%↑=<b>정점·성숙</b> / 정체+70%↓=<b>쇠퇴·잔존</b>
    <div style="margin-top:8px;font-size:12px;color:#64748b">※ 오름 : 1년새 역대최고치의 10% 이상 상승 · 내림 : 1년새 10% 이상 하락 · 정체 : 1년새 변화가 ±10% 이내(거의 변동 없음)</div></td></tr>
</tbody></table>
<div class="card note" style="margin-top:16px">
 <b>한계</b> ① 구글 트렌드 검색량은 정규화 상대값이라 절대량 비교 불가
 ② 마이너 신조어는 표본 부족 노이즈 ③ 메가/마이크로 라벨은 사전 가정→군집으로 사후 검증.
</div>

<footer>패션 트렌드 수명주기 분류 프로젝트 · Google Trends 데이터 기반 · 분석/시각화: Python(statsmodels STL, scikit-learn, Plotly)</footer>
{link_js}
</body></html>"""

    out = DOCS / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ 웹페이지 빌드 완료 → {out}")
    print(f"   브라우저에서 열기: file://{out}")


if __name__ == "__main__":
    build()
