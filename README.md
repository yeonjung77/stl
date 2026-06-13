# 패션 트렌드 수명주기 분류 (STL)

> Google Trends 검색 관심도를 **STL 시계열 분해**로 분석해, 패션 아이템을 **메가 · 마이크로 · 스테디** 트렌드로 분류하고 **수명주기 단계·실무 전략**을 도출하는 프로젝트.

🔗 **웹페이지:** https://yeonjung77.github.io/stl/ · **실시간 분류기:** https://gqfhgm689dykskx6puypjk.streamlit.app/

---

## 무엇을 하나

- 패션 아이템 **156개**(한국·최근 5년·주별)를 분석
- **정적 웹페이지**: 군집·사분면 지도·STL 분해·전략을 시각화
- **실시간 앱**: 아무 키워드나 입력 → 학습된 모델로 즉석 분류

## 방법 (4단계)

1. **STL 분해** — 검색 곡선을 `트렌드 + 계절성 + 잔차`로 분리
2. **지표 산출** — 분해 성분에서 7개 정량 지표 계산 (트렌드강도·계절성·기울기·지속기간·잔차변동·첨예도·피크시점)
3. **군집화** — 표준화 후 **K-means(k=3)** → 메가/마이크로/스테디 분류 *(여러 모델·k 비교로 검증, [MODEL_COMPARISON.md](stl200/MODEL_COMPARISON.md))*
4. **수명주기 단계** — 트렌드 방향 + 현재/피크 비율로 도입~쇠퇴 판정

## 폴더

```
stl200/          # 본 프로젝트
├─ src/          # analyze(분석) · build_site(웹빌드) · compare_models · save_model · download_trends
├─ app.py        # Streamlit 실시간 분류기
├─ model/        # 저장된 학습 모델 (scaler + KMeans)
└─ data/         # 원본 CSV · 분석 결과(features.csv)
docs/            # 배포용 웹페이지 (GitHub Pages)
```

## 재현

```bash
cd stl200
pip install -r requirements.txt
python src/analyze.py        # STL → 지표 → 분류 → features.csv
python src/save_model.py     # 학습 모델 저장
python src/build_site.py     # 웹페이지 빌드(→ ../docs)
streamlit run app.py         # 실시간 앱 로컬 실행
```

## 기술 스택

Python · statsmodels(STL) · scikit-learn(K-means) · Plotly · Streamlit · pytrends · GitHub Pages

## 한계

검색 관심도는 실제 판매가 아닌 관심의 프록시이며, Google Trends 수치는 정규화 상대값이라 곡선의 **형태**로만 해석한다.
