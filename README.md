# 패션 아이템 수명주기 기반 트렌드 분류

> Google Trends 검색 관심도를 **STL 시계열 분해**로 분석해, 패션 아이템 30개를 **메가 · 마이크로 · 스테디** 트렌드로 분류하고 각 아이템의 **수명주기 단계(도입·성장·정점·쇠퇴)**와 분류·단계별 **실무 전략**을 도출한 데이터 분석 프로젝트.

**🔗 결과 웹페이지:** [https://yeonjung77.github.io/stl/](https://yeonjung77.github.io/stl/) *(GitHub Pages 활성화 후 접속 가능)*

---

## 개요

패션 시장의 트렌드는 장기간 지속되는 **메가 트렌드**, 짧고 급격한 **마이크로 트렌드**, 매년 반복되는 **스테디(시즌) 트렌드**로 나뉜다. 본 프로젝트는 "이 아이템이 어떤 유형의 트렌드인가"를 주관적 판단이 아닌 **데이터 기반의 객관적 지표**로 판별하고, 실무(발주·재고·마케팅) 의사결정에 활용할 수 있는 전략을 제시한다.

- **데이터:** Google Trends 검색 관심도 (대한민국 · 최근 5년 · 주별)
- **방법:** STL 분해 → 정량 지표 6종 산출 → 군집화(K-means) + 규칙기반 분류 → 수명주기 단계 판정
- **산출물:** Plotly 기반 인터랙티브 정적 웹페이지 (GitHub Pages 배포)

---

## 주요 결과

| 분류 | 아이템 수 | 특징 |
|---|---|---|
| 🟩 스테디 | 14개 | 계절성이 강하게 반복되는 품목 |
| 🔵 메가 | 9개 | 추세가 또렷하고 변동이 안정적인 품목 |
| 🔴 마이크로 | 7개 | 변동성이 크고 스파이크성이 강한 품목 |

- 분석 대상 **30개 아이템**, 각 **262주(5년) 주별 데이터**
- 사전 가정(수작업 분류) 대비 데이터 기반 군집 **일치율 67%** — 나머지 33%는 데이터가 밝혀낸 재분류로, 분석의 핵심 인사이트

---

## 분석 방법

### 1. STL 시계열 분해
검색 곡선 `Y_t`를 트렌드·계절성·잔차로 분해한다. (`Y_t = T_t + S_t + R_t`, 주별이므로 `period=52`)

### 2. 정량 지표(feature) 6종

| 지표 | 계산식 | 의미 |
|---|---|---|
| `F_T` 트렌드 강도 | `max(0, 1 − Var(R)/Var(T+R))` | 추세의 또렷함 |
| `F_S` 계절성 강도 | `max(0, 1 − Var(R)/Var(S+R))` | 계절 반복의 또렷함 |
| 기울기 | 트렌드 성분의 선형회귀 계수 | 장기 상승/하락 방향 |
| 지속기간 | 피크 50% 이상 유지 비율 | 인기 지속성 |
| 잔차변동 (CV) | `std(R) / mean(Y)` | 불안정성 |
| 첨예도 | `peak / median(검색이 있던 주)` | 스파이크성 |

### 3. 군집화 + 규칙기반 분류
정답 라벨이 없는 소표본 환경이므로 지도학습 대신 **비지도 군집화**를 사용한다. 6개 지표를 표준화한 뒤 **K-means(k=3)**로 자연 그룹을 도출하고, 군집 중심 특성을 해석해 메가/마이크로/스테디 라벨을 부여한 후 규칙으로 교차검증한다.

### 4. 수명주기 단계 판정
현재값/역대피크 비율과 최근 1년 트렌드 방향으로 **도입·성장·정점/성숙·쇠퇴·쇠퇴/잔존**을 판정한다.

---

## 폴더 구조

```
stl/
├─ data/
│  ├─ keywords.csv              # 분석 아이템 30개 목록 (slug, 분류, 검색어)
│  ├─ raw/                      # Google Trends 원본 CSV (30개)
│  └─ processed/
│     ├─ features.csv           # 산출 지표 · 분류 · 단계 결과표
│     └─ components/            # 아이템별 STL 성분 (시각화용)
├─ src/
│  ├─ download_trends.py        # Google Trends 자동 수집 (pytrends)
│  ├─ analyze.py                # STL 분해 → 지표 → 군집·분류 → 단계
│  └─ build_site.py             # 분석 결과 → 정적 HTML 빌드 (Plotly)
├─ docs/
│  └─ index.html                # 배포용 웹페이지 (GitHub Pages 서빙 경로)
├─ requirements.txt
└─ README.md
```

---

## 재현 방법

### 1. 환경 설정
```bash
git clone https://github.com/yeonjung77/stl.git
cd stl
pip install -r requirements.txt
```

### 2. 데이터 준비
원본 CSV 30개는 `data/raw/`에 포함되어 있다. 새로 수집하거나 키워드를 교체하려면:

- **자동 수집:** `python src/download_trends.py` — `data/keywords.csv` 기준으로 Google Trends에서 자동 다운로드 (pytrends, 비공식 API라 차단 가능성 있음)
- **수동 수집:** Google Trends 웹에서 각 키워드를 `대한민국 · 최근 5년` 조건으로 CSV 다운로드 후 `data/raw/<slug>.csv`로 저장

### 3. 분석 실행
```bash
python src/analyze.py        # → data/processed/features.csv 생성
python src/build_site.py     # → docs/index.html 생성
```

### 4. 웹페이지 확인 / 배포
```bash
open docs/index.html         # 로컬 확인 (macOS)
```
GitHub Pages 배포: 저장소 **Settings → Pages → Source**를 `main` 브랜치의 `/docs` 폴더로 지정하면 위 결과 URL로 공개된다.

---

## 기술 스택

- **언어:** Python 3.11+
- **분석:** pandas, numpy, statsmodels(STL), scikit-learn(표준화·K-means)
- **시각화:** Plotly (인터랙티브 차트)
- **데이터 수집:** Google Trends (CSV) · pytrends(선택)
- **배포:** 정적 HTML → GitHub Pages

---

## 한계

1. 검색 관심도는 실제 판매가 아니라 관심·인지의 간접 지표(프록시)다.
2. Google Trends 수치는 정규화된 상대값(0–100)이라 키워드 간 절대량 비교는 불가하며, 곡선의 형태로만 해석한다.
3. 마이너 신조어 키워드는 표본 부족으로 노이즈가 클 수 있다 (데이터 품질 컷으로 일부 키워드 교체).
4. 메가/마이크로 라벨은 사전 가정이며, 군집·규칙으로 사후 검증하는 구조다.
5. 키워드 선정에 주관이 개입되므로 선정 근거를 문서로 공개해 재현성을 확보한다.
