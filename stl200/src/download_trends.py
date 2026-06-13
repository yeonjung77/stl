"""
구글 트렌드 자동 다운로드 (pytrends).
data/keywords.csv를 읽어 각 아이템을 today 5-y · geo=KR · 주별로 받아
data/raw/<slug>.csv 로 저장한다. (수동 다운로드와 동일 형식)

- 이미 받은 파일(<slug>.csv 또는 <slug>_multiTimeline*.csv)은 건너뜀
- 차단(429) 회피용 요청 간격 + 실패 시 지수 백오프 재시도
- 끝에 성공/빈데이터/실패 요약 출력 (노이즈 키워드 확인용)
"""
from pathlib import Path
import sys
import time
import random
import pandas as pd
from pytrends.request import TrendReq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "keywords.csv"


def already_have(slug: str) -> bool:
    if (RAW / f"{slug}.csv").exists():
        return True
    return bool(list(RAW.glob(f"{slug}_multiTimeline*.csv")))


def to_trends_csv(series: pd.Series, keyword: str, path: Path) -> None:
    """수동 다운로드와 동일한 multiTimeline 형식으로 저장."""
    lines = ["카테고리: 모든 카테고리", "", f"주,{keyword}: (대한민국)"]
    for ts, val in series.items():
        lines.append(f"{ts.strftime('%Y-%m-%d')},{int(round(val))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch(pt: TrendReq, keyword: str, retries: int = 5) -> pd.Series | None:
    for attempt in range(retries):
        try:
            pt.build_payload([keyword], timeframe="today 5-y", geo="KR")
            df = pt.interest_over_time()
            if df.empty or keyword not in df.columns:
                return None
            return df[keyword]
        except Exception as e:
            # 200개 대량 수집: 차단(429)을 식히기 위해 대기를 길게 (60→300s)
            wait = (attempt + 1) * 60 + random.uniform(0, 20)
            print(f"    ⚠️ 실패({type(e).__name__}) → {wait:.0f}s 후 재시도 [{attempt+1}/{retries}]")
            time.sleep(wait)
    return None


def main() -> None:
    import sys
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else MANIFEST
    RAW.mkdir(parents=True, exist_ok=True)
    items = pd.read_csv(manifest)
    print(f"manifest: {manifest}")
    pt = TrendReq(hl="ko-KR", tz=540)

    done, empty, failed, skipped = [], [], [], []
    todo = [r for _, r in items.iterrows() if not already_have(r["slug"])]
    print(f"총 {len(items)}개 중 {len(todo)}개 다운로드 (이미 받은 {len(items)-len(todo)}개 건너뜀)\n")

    for i, row in enumerate(todo, 1):
        slug, kw = row["slug"], row["keyword"]
        print(f"[{i}/{len(todo)}] {slug}  ('{kw}')")
        s = fetch(pt, kw)
        if s is None:
            failed.append(slug)
            print("    ❌ 다운로드 실패")
        elif s.max() <= 1 or (s > 0).sum() < 10:
            to_trends_csv(s, kw, RAW / f"{slug}.csv")
            empty.append(slug)
            print(f"    ⚠️ 저장했지만 검색량 거의 없음 (max={s.max()}) → 노이즈 가능")
        else:
            to_trends_csv(s, kw, RAW / f"{slug}.csv")
            done.append(slug)
            print(f"    ✅ 저장 (rows={len(s)}, max={s.max()})")
        # 차단 회피: 요청 사이 간격 (200개 대량 → 넉넉하게)
        if i < len(todo):
            time.sleep(random.uniform(18, 32))

    print("\n" + "=" * 50)
    print(f"✅ 정상: {len(done)}개")
    if empty:
        print(f"⚠️  저검색(노이즈 가능, 키워드 교체 검토): {len(empty)}개 → {empty}")
    if failed:
        print(f"❌ 실패(재실행 필요): {len(failed)}개 → {failed}")
    print("=" * 50)


if __name__ == "__main__":
    main()
