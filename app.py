# app.py
# -*- coding: utf-8 -*-
import os
from datetime import date
from dateutil.relativedelta import relativedelta
from urllib.parse import quote

import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from pytrends.request import TrendReq

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="AI Tools Popularity Tracker", page_icon="📈", layout="wide")

DEFAULT_KEYWORDS = ["ChatGPT", "Midjourney", "GitHub Copilot", "Google Gemini"]
DEFAULT_REGION = "US"  # '', 'US', 'KR' 등
DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date.today()

# 위키 문서 매핑(영어 위키 기준). 필요시 사이드바에서 수정 가능
DEFAULT_WIKI_PAGES = {
    "ChatGPT": "ChatGPT",
    "Midjourney": "Midjourney",
    "GitHub Copilot": "GitHub_Copilot",
    "Google Gemini": "Google_Gemini",
}

# Wikimedia 요청 세션 (User-Agent 필수)
SESSION = requests.Session()
SESSION.headers.update({
    # 연락처/프로젝트 링크 포함 권장(본인 이메일/깃헙 이슈 URL 등으로 교체하세요)
    "User-Agent": "AI-Tools-Tracker/1.0 (contact: your_email@example.com)"
})
retries = Retry(
    total=5,
    backoff_factor=1.5,
    status_forcelist=[403, 429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
SESSION.mount("https://", HTTPAdapter(max_retries=retries))

# =========================
# 함수: Google Trends
# =========================
def _build_pytrends():
    # tz: 360 = GMT+6가 아니라, pytrends 내부 표준. 크게 영향 없음
    return TrendReq(hl="en-US", tz=360)

def _chunks(lst, n):
    """lst를 길이 n로 쪼개는 제너레이터"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@st.cache_data(ttl=60*60)  # 1시간 캐시
def fetch_google_trends_monthly_mean(keywords, start: date, end: date, region: str = "") -> pd.DataFrame:
    """
    Google Trends 관심도(0-100 상대지표) 월 평균으로 집계해서 반환.
    - 요청을 4~5개 키워드 단위 배치로 나눠서 안정성 확보
    - 부분 실패 시 해당 배치는 건너뛰고 나머지 병합
    """
    if not keywords:
        return pd.DataFrame()

    time_window = f"{start.isoformat()} {end.isoformat()}"
    pytrends = _build_pytrends()
    frames = []
    for batch in _chunks(keywords, 5):
        try:
            pytrends.build_payload(batch, timeframe=time_window, geo=region)
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                continue
            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])
            # 월 평균
            df = df.resample("MS").mean().round(2)
            frames.append(df)
        except Exception as e:
            # pytrends가 가끔 막히는 경우가 있음 (IP/빈도/세션 이슈)
            st.warning(f"⚠️ Google Trends 요청 일부 실패: {batch} | {e}")
            continue

    if not frames:
        return pd.DataFrame()
    # 동일 날짜 인덱스 기준 병합
    base = frames[0]
    for f in frames[1:]:
        base = base.join(f, how="outer")
    base.index.name = "month"
    # 일부 키워드만 성공했을 수 있으므로 컬럼 순서 정리
    cols = [k for k in keywords if k in base.columns]
    return base[cols].sort_index()

# =========================
# 함수: Wikimedia Pageviews
# =========================
def _wiki_month_bounds(start: date, end: date):
    start_str = pd.Timestamp(start).strftime("%Y%m01")
    end_last = (pd.Timestamp(end) + relativedelta(day=31)).strftime("%Y%m%d")
    return start_str, end_last

def _wiki_url(project, access, agent, article, start_str, end_last):
    encoded_title = quote(article, safe="")
    return (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{project}/{access}/{agent}/{encoded_title}/monthly/{start_str}/{end_last}"
    )

def fetch_wiki_pageviews_one(title: str, start: date, end: date,
                             project="en.wikipedia", access="all-access", agent="user") -> pd.Series:
    start_str, end_last = _wiki_month_bounds(start, end)
    url = _wiki_url(project, access, agent, title, start_str, end_last)
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return pd.Series(dtype="float64")

        data = {}
        for it in items:
            ts = str(it["timestamp"])  # e.g., 2025010100
            y, m = int(ts[:4]), int(ts[4:6])
            data[pd.Timestamp(year=y, month=m, day=1)] = it["views"]
        s = pd.Series(data).sort_index()
        return s
    except Exception as e:
        # 403/429 등 발생 시 빈 시리즈 반환(앱이 죽지 않도록)
        st.info(f"ℹ️ 위키 조회 실패: {title} | {e}")
        return pd.Series(dtype="float64")

@st.cache_data(ttl=60*60)
def fetch_wiki_pageviews_map(page_map: dict, start: date, end: date) -> pd.DataFrame:
    if not page_map:
        return pd.DataFrame()
    series = []
    for key, title in page_map.items():
        s = fetch_wiki_pageviews_one(title, start, end)
        s.name = key
        series.append(s)
    if not series:
        return pd.DataFrame()
    df = pd.concat(series, axis=1)
    df.index.name = "month"
    return df.sort_index()

# =========================
# UI
# =========================
st.title("📈 AI 도구 시점별 인기/인지도 트래커")
st.caption("Google Trends(상대지표) + Wikipedia Pageviews(절대조회)를 월 단위로 트래킹합니다.")

with st.sidebar:
    st.header("⚙️ 설정")
    region = st.selectbox("Google Trends 지역 (빈값=글로벌)", options=["", "US", "KR", "BR", "JP", "GB", "DE", "FR"], index=1)
    start = st.date_input("시작일", value=DEFAULT_START, min_value=date(2019,1,1), max_value=DEFAULT_END)
    end = st.date_input("종료일", value=DEFAULT_END, min_value=DEFAULT_START, max_value=DEFAULT_END)

    kw_text = st.text_area("키워드(쉼표로 구분)", value=", ".join(DEFAULT_KEYWORDS))
    keywords = [k.strip() for k in kw_text.split(",") if k.strip()]

    st.markdown("---")
    st.subheader("Wikipedia 문서 매핑")
    st.caption("필요 시 문서 제목을 수정하세요. (영문 위키 기준)")
    wiki_map = {}
    for k in keywords:
        default_title = DEFAULT_WIKI_PAGES.get(k, k.replace(" ", "_"))
        wiki_map[k] = st.text_input(f"{k} →", value=default_title, key=f"wiki_{k}")

    use_wiki = st.checkbox("Wikipedia Pageviews 사용", value=True)
    st.markdown("---")
    st.caption("Tip: 403/429가 뜨면 잠시 후 다시 시도하거나 키워드를 줄여보세요.")

# =========================
# 데이터 수집 & 표시
# =========================
col1, col2 = st.columns(2)

with st.spinner("Google Trends 수집 중..."):
    trends_df = fetch_google_trends_monthly_mean(keywords, start, end, region)

if not trends_df.empty:
    with col1:
        st.subheader("Google Trends (월 평균, 0–100 상대지표)")
        st.line_chart(trends_df, height=340, use_container_width=True)
        st.dataframe(trends_df.tail(12), use_container_width=True)
        st.download_button(
            "⬇️ Trends CSV 다운로드",
            trends_df.to_csv(index=True).encode("utf-8"),
            file_name=f"google_trends_{region or 'GLOBAL'}_{start}_{end}.csv",
            mime="text/csv"
        )
else:
    with col1:
        st.warning("Trends 데이터를 가져오지 못했습니다. 키워드/기간/지역을 조정하거나 잠시 후 재시도하세요.")

if use_wiki:
    with st.spinner("Wikipedia Pageviews 수집 중..."):
        wiki_df = fetch_wiki_pageviews_map(wiki_map, start, end)
else:
    wiki_df = pd.DataFrame()

if use_wiki:
    if not wiki_df.empty:
        with col2:
            st.subheader("Wikipedia Pageviews (월별 절대 조회수)")
            st.line_chart(wiki_df, height=340, use_container_width=True)
            st.dataframe(wiki_df.tail(12), use_container_width=True)
            st.download_button(
                "⬇️ Wiki CSV 다운로드",
                wiki_df.to_csv(index=True).encode("utf-8"),
                file_name=f"wikipedia_pageviews_{start}_{end}.csv",
                mime="text/csv"
            )
    else:
        with col2:
            st.info("Wikipedia Pageviews를 불러오지 못했습니다. 문서 제목/네트워크/User-Agent를 확인하세요.")

# =========================
# 통합 테이블 (선택)
# =========================
if not trends_df.empty or (use_wiki and not wiki_df.empty):
    combined = trends_df.join(wiki_df, how="outer", lsuffix="_trends", rsuffix="_pageviews")
    st.markdown("### 🔗 통합 뷰 (Trends + Wiki)")
    st.dataframe(combined.tail(12), use_container_width=True)
    st.download_button(
        "⬇️ Combined CSV 다운로드",
        combined.to_csv(index=True).encode("utf-8"),
        file_name=f"ai_tools_combined_{start}_{end}.csv",
        mime="text/csv"
    )

# =========================
# 푸터: 참고/주의
# =========================
with st.expander("ℹ️ 참고 및 주의사항"):
    st.markdown("""
- **Google Trends는 상대지표(0–100)** 입니다. 동일 요청(키워드·기간·지역 조합) 내에서의 비교에 적합하며, 다른 요청과는 스케일이 다를 수 있습니다.
- **Wikipedia Pageviews는 절대 조회수**로 관심의 규모감을 보완합니다. 페이지 제목 변경/리다이렉트가 있는지 가끔 확인하세요.
- API 호출은 빈도 제한/차단(403/429)에 민감할 수 있으니, **키워드를 나눠 요청**하고 **캐시**를 활용하세요.
- User-Agent에는 연락처(이메일/깃헙 URL)를 명시하는 것이 **Wikimedia 권고**입니다.
""")
