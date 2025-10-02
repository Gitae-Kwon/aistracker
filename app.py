#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 도구 시점별 트래킹 스크립트
- Google Trends: pytrends (정규화 지수, 지역/기간 비교)
- Wikimedia Pageviews API: 각 도구 위키 문서 월별 조회수
- 결과물: ./data/ 디렉토리에 CSV 저장, ./figures/ 에 PNG 그래프 저장
참고:
- pytrends 문서: https://pypi.org/project/pytrends/
- Google Trends 수치는 상대 지표(0~100 정규화)임. 동일 요청 내 비교에 적합. (isPartial 주의) 
- Wikimedia Pageviews API: https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/concepts/page-views.html
"""

import os
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pytrends.request import TrendReq

# -------------------------------
# 0) 설정
# -------------------------------
KEYWORDS = ["ChatGPT", "Midjourney", "GitHub Copilot", "Google Gemini"]  # 필요 시 자유롭게 추가
REGION = "US"           # 'US', 'KR', 'BR', ''(글로벌) 등
START = "2022-01-01"    # 시작일
TODAY = date.today().strftime("%Y-%m-%d")

DATA_DIR = "./data"
FIG_DIR = "./figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 위키문서 매핑 (필요 시 로컬라이즈 가능: 'en.wikipedia' 기준 제목)
WIKI_PAGES = {
    "ChatGPT": "ChatGPT",
    "Midjourney": "Midjourney",
    "GitHub Copilot": "GitHub_Copilot",
    "Google Gemini": "Google_Gemini"
}

# -------------------------------
# 1) Google Trends 수집
# -------------------------------
def fetch_google_trends(keywords, start, end, region):
    """
    Google Trends 관심도(정규화 지수, 주/일 단위 → 월집계)
    region: ''이면 글로벌, 'US' 등 국가코드 가능
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    timeframe = f"{start} {end}"
    pytrends.build_payload(keywords, timeframe=timeframe, geo=region)
    df = pytrends.interest_over_time()
    if df.empty:
        return pd.DataFrame()
    # isPartial 컬럼 제거 및 월 집계(평균)
    if 'isPartial' in df.columns:
        df = df.drop(columns=['isPartial'])
    df = df.resample('MS').mean().round(2)  # 월초 기준 평균
    df.index.name = "month"
    return df

# -------------------------------
# 2) Wikimedia Pageviews 수집
# -------------------------------
def fetch_wiki_pageviews(page_title, start, end, project="en.wikipedia", access="all-access", agent="user"):
    """
    Wikimedia Pageviews 월별 조회수
    API: /metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/monthly/{start}/{end}
    start/end: YYYYMMDD 형식, 월별이면 YYYYMM01 권장
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    start_str = start_dt.strftime("%Y%m01")
    end_last = (end_dt + relativedelta(day=31)).strftime("%Y%m%d")

    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{project}/{access}/{agent}/{page_title}/monthly/{start_str}/{end_last}"
    )
    
    try:
        # API 요청 보내기
        r = requests.get(url, timeout=30)
        r.raise_for_status()  # HTTP 오류 발생 시 예외 발생
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # 상세 에러 메시지 출력
        return pd.Series(dtype="float64")  # 에러 발생 시 빈 Series 반환
    except Exception as err:
        print(f"Other error occurred: {err}")  # 다른 예외 처리
        return pd.Series(dtype="float64")
    
    items = r.json().get("items", [])
    if not items:
        return pd.Series(dtype="float64")

    # 시리즈로 변환
    data = {}
    for it in items:
        ts = str(it["timestamp"])  # e.g., 2025010100
        y = int(ts[:4]); m = int(ts[4:6])
        month = pd.Timestamp(year=y, month=m, day=1)
        data[month] = it["views"]
    s = pd.Series(data).sort_index()
    s.name = page_title
    return s

def fetch_all_wiki_pageviews(page_map, start, end):
    series_list = []
    for k, title in page_map.items():
        s = fetch_wiki_pageviews(title, start, end)
        s.name = k
        series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1)
    df.index.name = "month"
    return df

# -------------------------------
# 3) 실행
# -------------------------------
if __name__ == "__main__":
    end = TODAY

    # 3-1) Google Trends
    trends_df = fetch_google_trends(KEYWORDS, START, end, REGION)
    if not trends_df.empty:
        trends_path = os.path.join(DATA_DIR, f"google_trends_{REGION or 'GLOBAL'}_{START}_{end}.csv")
        trends_df.to_csv(trends_path, encoding="utf-8")
        # 그래프
        plt.figure(figsize=(12,6))
        for col in trends_df.columns:
            plt.plot(trends_df.index, trends_df[col], label=col)
        plt.title(f"Google Trends (region={REGION or 'GLOBAL'}) — monthly mean", fontsize=13)
        plt.xlabel("Month"); plt.ylabel("Relative interest (0–100)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"google_trends_{REGION or 'GLOBAL'}.png"), dpi=150)
        plt.close()

    # 3-2) Wikimedia Pageviews
    wiki_df = fetch_all_wiki_pageviews(WIKI_PAGES, START, end)
    if not wiki_df.empty:
        wiki_path = os.path.join(DATA_DIR, f"wikipedia_pageviews_{START}_{end}.csv")
        wiki_df.to_csv(wiki_path, encoding="utf-8")
        # 그래프
        plt.figure(figsize=(12,6))
        for col in wiki_df.columns:
            plt.plot(wiki_df.index, wiki_df[col], label=col)
        plt.title("Wikipedia Pageviews — monthly", fontsize=13)
        plt.xlabel("Month"); plt.ylabel("Views")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "wikipedia_pageviews.png"), dpi=150)
        plt.close()

    # 3-3) 통합 저장(outer join)
    if not trends_df.empty or not wiki_df.empty:
        combined = trends_df.join(wiki_df, how="outer", lsuffix="_trends", rsuffix="_pageviews")
        combined_path = os.path.join(DATA_DIR, f"ai_tools_combined_{START}_{end}.csv")
        combined.to_csv(combined_path, encoding="utf-8")
        print(f"[Saved]\n- {trends_path if not trends_df.empty else '(no trends)'}\n- {wiki_path if not wiki_df.empty else '(no wiki)'}\n- {combined_path}")
    else:
        print("No data fetched. Check network or parameters.")
