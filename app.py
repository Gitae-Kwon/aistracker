# app.py
# -*- coding: utf-8 -*-
import os
from datetime import date
from dateutil.relativedelta import relativedelta
from urllib.parse import quote

import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
from requests.adapters import HTTPAdapter, Retry
from pytrends.request import TrendReq

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="AI Tools Tracker (Persistent)", page_icon="📊", layout="wide")

DATA_DIR = "./data"
FIG_DIR = "./figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 추적 기본값
DEFAULT_REGION = "US"     # '', 'US', 'KR', ...
DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date.today()

# 6개 분야와 키워드(원하면 자유롭게 수정/추가)
CATEGORIES = {
    "1. 생산성/업무보조": ["ChatGPT", "Microsoft Copilot", "Google Gemini", "Notion AI", "Grammarly"],
    "2. 마케팅": ["Jasper", "Copy.ai", "Anyword", "HubSpot AI", "Grammarly Business"],
    "3. 디자인/영상/이미지": ["Canva", "Midjourney", "Adobe Firefly", "Runway", "DALL·E"],
    "4. 개발/코딩": ["GitHub Copilot", "ChatGPT", "Cursor", "Codeium", "Claude"],
    "5. 고객서비스/챗봇": ["Zendesk AI", "Intercom", "Salesforce Einstein", "Dialogflow", "ChatGPT"],
    "6. 운영/자동화": ["Zapier", "Make", "UiPath", "Power Automate", "n8n"]
}
# 위키 문서 매핑(영문 위키)
WIKI_PAGES_DEFAULT = {
    "ChatGPT": "ChatGPT",
    "Microsoft Copilot": "Microsoft_Copilot",
    "Google Gemini": "Google_Gemini",
    "Notion AI": "Notion_(product)",  # 통합 문서로 대략 추적
    "Grammarly": "Grammarly",

    "Jasper": "Jasper_(software)",
    "Copy.ai": "Copy.ai",
    "Anyword": "Anyword",
    "HubSpot AI": "HubSpot",
    "Grammarly Business": "Grammarly",

    "Canva": "Canva",
    "Midjourney": "Midjourney",
    "Adobe Firefly": "Adobe_Firefly",
    "Runway": "Runway_(company)",
    "DALL·E": "DALL-E",

    "GitHub Copilot": "GitHub_Copilot",
    "Cursor": "Cursor_(software)",
    "Codeium": "Codeium",
    "Claude": "Claude_(language_model)",

    "Zendesk AI": "Zendesk",
    "Intercom": "Intercom_(company)",
    "Salesforce Einstein": "Salesforce_Einstein",
    "Dialogflow": "Dialogflow",

    "Zapier": "Zapier",
    "Make": "Integromat",
    "UiPath": "UiPath",
    "Power Automate": "Power_Automate",
    "n8n": "N8n"
}

# Wikimedia 요청 세션 (User-Agent 필수)
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "AI-Tools-Tracker/1.0 (contact: your_email@example.com)"  # 본인 이메일/깃헙 URL 등으로 교체
})
retries = Retry(
    total=5, backoff_factor=1.5,
    status_forcelist=[403, 429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
SESSION.mount("https://", HTTPAdapter(max_retries=retries))

# =========================
# 유틸
# =========================
def ensure_month(dt: date) -> pd.Timestamp:
    return pd.Timestamp(year=dt.year, month=dt.month, day=1)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# =========================
# Google Trends
# =========================
def build_pytrends():
    return TrendReq(hl="en-US", tz=360)

@st.cache_data(ttl=60*60)
def fetch_google_trends_monthly_mean(keywords, start: date, end: date, region: str = "") -> pd.DataFrame:
    """키워드를 5개 이하 배치로 나눠 안정적으로 수집 → 월 평균으로 리샘플"""
    if not keywords:
        return pd.DataFrame()
    time_window = f"{start.isoformat()} {end.isoformat()}"
    pytrends = build_pytrends()
    frames = []
    for batch in chunks(keywords, 5):
        try:
            pytrends.build_payload(batch, timeframe=time_window, geo=region)
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                continue
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            df = df.resample("MS").mean().round(2)
            frames.append(df)
        except Exception as e:
            st.warning(f"⚠️ Google Trends 일부 실패: {batch} | {e}")
            continue
    if not frames:
        return pd.DataFrame()
    base = frames[0]
    for f in frames[1:]:
        base = base.join(f, how="outer")
    base.index.name = "month"
    cols = [k for k in keywords if k in base.columns]
    return base[cols].sort_index()

# =========================
# Wikimedia Pageviews
# =========================
def wiki_month_bounds(start: date, end: date):
    start_str = pd.Timestamp(start).strftime("%Y%m01")
    end_last = (pd.Timestamp(end) + relativedelta(day=31)).strftime("%Y%m%d")
    return start_str, end_last

def wiki_url(project, access, agent, article, start_str, end_last):
    encoded_title = quote(article, safe="")
    return (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"{project}/{access}/{agent}/{encoded_title}/monthly/{start_str}/{end_last}")

def fetch_wiki_one(title: str, start: date, end: date,
                   project="en.wikipedia", access="all-access", agent="user") -> pd.Series:
    start_str, end_last = wiki_month_bounds(start, end)
    url = wiki_url(project, access, agent, title, start_str, end_last)
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return pd.Series(dtype="float64")
        data = {}
        for it in items:
            ts = str(it["timestamp"])
            y, m = int(ts[:4]), int(ts[4:6])
            data[pd.Timestamp(year=y, month=m, day=1)] = it["views"]
        return pd.Series(data).sort_index()
    except Exception as e:
        st.info(f"ℹ️ 위키 조회 실패: {title} | {e}")
        return pd.Series(dtype="float64")

@st.cache_data(ttl=60*60)
def fetch_wiki_map(page_map: dict, start: date, end: date) -> pd.DataFrame:
    if not page_map:
        return pd.DataFrame()
    series = []
    for key, title in page_map.items():
        s = fetch_wiki_one(title, start, end)
        s.name = key
        series.append(s)
    if not series:
        return pd.DataFrame()
    df = pd.concat(series, axis=1)
    df.index.name = "month"
    return df.sort_index()

# =========================
# 지속 추적(히스토리 누적)
# =========================
def load_history(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["month"])
            df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
            df = df.set_index("month").sort_index()
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_history(path: str, df: pd.DataFrame):
    if df.empty:
        return
    out = df.copy()
    out = out.sort_index()
    out.index.name = "month"
    out.to_csv(path, index=True)

def merge_history(history: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return new_df
    merged = history.combine_first(new_df).copy()
    for col in new_df.columns:
        merged[col].update(new_df[col])
    return merged

# =========================
# 점수 산출(정규화 + 가중합)
# =========================
def minmax_norm(s: pd.Series) -> pd.Series:
    if s.dropna().empty:
        return s
    mn, mx = s.min(), s.max()
    if mx == mn:
        return s * 0
    return (s - mn) / (mx - mn)

def zscore_norm(s: pd.Series) -> pd.Series:
    if s.dropna().empty:
        return s
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0:
        return s * 0
    return (s - mu) / sd

def compute_scores(trends_df: pd.DataFrame, wiki_df: pd.DataFrame,
                   w_trends: float = 0.6, w_wiki: float = 0.4) -> pd.DataFrame:
    """
    월별/도구별 점수:
    - Trends: min-max 정규화
    - Wiki: z-score → min-max
    - 복합 = 0.6*Trends_norm + 0.4*Wiki_norm
    """
    all_tools = sorted(list(set(trends_df.columns.tolist()) | set(wiki_df.columns.tolist())))
    idx = trends_df.index.union(wiki_df.index).sort_values()
    trends_norm = pd.DataFrame(index=idx, columns=all_tools, dtype=float)
    wiki_norm = pd.DataFrame(index=idx, columns=all_tools, dtype=float)

    for t in all_tools:
        if t in trends_df.columns:
            trends_norm[t] = minmax_norm(trends_df[t])
        if t in wiki_df.columns:
            wiki_norm[t] = minmax_norm(zscore_norm(wiki_df[t]))

    score = (w_trends * trends_norm.fillna(0) + w_wiki * wiki_norm.fillna(0))
    score.index.name = "month"
    return score

# =========================
# 순위 히스토리 생성
# =========================
def build_rank_history(score_hist: pd.DataFrame, categories: dict, method: str = "dense") -> dict:
    """
    월별 점수(score_hist)를 카테고리별 '순위(1=최상)' 히스토리로 변환
    return: {category: DataFrame(index=month, columns=tools) with integer ranks}
    """
    if score_hist.empty:
        return {}
    rank_hists = {}
    for cat, tools in categories.items():
        cols = [t for t in tools if t in score_hist.columns]
        if not cols:
            rank_hists[cat] = pd.DataFrame()
            continue
        sub = score_hist[cols].copy()
        ranks = []
        for _, row in sub.iterrows():
            r = (-row).rank(method=method)  # 점수 높을수록 1위에 가깝게
            ranks.append(r)
        ranks_df = pd.DataFrame(ranks, index=sub.index)
        ranks_df = ranks_df.round(0).astype("Int64")
        ranks_df.index.name = "month"
        rank_hists[cat] = ranks_df
    return rank_hists

# =========================
# UI - 사이드바
# =========================
st.title("📊 AI 도구 지속 추적 & 분야별 리더보드 + 순위 히스토리")
with st.sidebar:
    st.header("⚙️ 설정")
    region = st.selectbox("Google Trends 지역 (빈값=글로벌)", ["", "US", "KR", "BR", "JP", "GB", "DE", "FR"], index=1)
    start = st.date_input("시작일", value=DEFAULT_START, min_value=date(2019,1,1), max_value=DEFAULT_END)
    end = st.date_input("종료일", value=DEFAULT_END, min_value=DEFAULT_START, max_value=DEFAULT_END)
    use_wiki = st.checkbox("Wikipedia Pageviews 사용", value=True)
    st.markdown("---")
    st.caption("Tip: 먼저 기본값으로 실행 후 필요 시 조정하세요.")

# 추적 대상 키워드/위키 매핑 생성
ALL_TOOLS = sorted({tool for v in CATEGORIES.values() for tool in v})
WIKI_MAP = {k: WIKI_PAGES_DEFAULT.get(k, k.replace(" ", "_")) for k in ALL_TOOLS}

# =========================
# 데이터 수집 (이번 실행분)
# =========================
with st.spinner("Google Trends 수집 중..."):
    trends_now = fetch_google_trends_monthly_mean(ALL_TOOLS, start, end, region)

if use_wiki:
    with st.spinner("Wikipedia Pageviews 수집 중..."):
        wiki_now = fetch_wiki_map(WIKI_MAP, start, end)
else:
    wiki_now = pd.DataFrame()

# =========================
# 히스토리 로드 & 병합 & 저장
# =========================
hist_trends_path = os.path.join(DATA_DIR, f"history_trends_{region or 'GLOBAL'}.csv")
hist_wiki_path = os.path.join(DATA_DIR, "history_wiki.csv")
hist_score_path = os.path.join(DATA_DIR, f"history_scores_{region or 'GLOBAL'}.csv")

hist_trends = load_history(hist_trends_path)
hist_wiki = load_history(hist_wiki_path)

trends_hist_new = merge_history(hist_trends, trends_now) if not trends_now.empty else hist_trends
wiki_hist_new = merge_history(hist_wiki, wiki_now) if not wiki_now.empty else hist_wiki

save_history(hist_trends_path, trends_hist_new)
if use_wiki:
    save_history(hist_wiki_path, wiki_hist_new)

# 점수 계산 & 저장
score_hist = compute_scores(trends_hist_new if not trends_hist_new.empty else pd.DataFrame(),
                            wiki_hist_new if not wiki_hist_new.empty else pd.DataFrame(),
                            w_trends=0.6, w_wiki=0.4)
save_history(hist_score_path, score_hist)

# 순위 히스토리 생성
rank_histories = build_rank_history(score_hist, CATEGORIES, method="dense")

# =========================
# 화면 표시: 히스토리/리더보드
# =========================
c1, c2 = st.columns(2)
with c1:
    st.subheader("Google Trends (월 평균, 상대지표 0–100)")
    if not trends_hist_new.empty:
        st.line_chart(trends_hist_new[ALL_TOOLS].tail(36), height=320, use_container_width=True)
        st.dataframe(trends_hist_new.tail(12), use_container_width=True)
        st.download_button("⬇️ Trends 히스토리 CSV", trends_hist_new.to_csv().encode("utf-8"),
                           file_name=os.path.basename(hist_trends_path), mime="text/csv")
    else:
        st.info("Trends 데이터가 없습니다.")

with c2:
    st.subheader("Wikipedia Pageviews (월별 절대 조회수)")
    if use_wiki and not wiki_hist_new.empty:
        st.line_chart(wiki_hist_new[ALL_TOOLS].tail(36), height=320, use_container_width=True)
        st.dataframe(wiki_hist_new.tail(12), use_container_width=True)
        st.download_button("⬇️ Wiki 히스토리 CSV", wiki_hist_new.to_csv().encode("utf-8"),
                           file_name=os.path.basename(hist_wiki_path), mime="text/csv")
    else:
        st.info("Wiki 데이터를 사용하지 않거나 비어 있습니다.")

st.markdown("---")
st.subheader("🧮 복합 점수(정규화 결합) 히스토리")
if not score_hist.empty:
    st.line_chart(score_hist[ALL_TOOLS].tail(36), height=320, use_container_width=True)
    st.dataframe(score_hist.tail(12), use_container_width=True)
    st.download_button("⬇️ Score 히스토리 CSV", score_hist.to_csv().encode("utf-8"),
                       file_name=os.path.basename(hist_score_path), mime="text/csv")
else:
    st.info("점수 히스토리가 비어 있습니다.")

# 분야별 리더보드 (최근 월 Top N)
st.markdown("---")
st.header("🏆 분야별 리더보드 (최근 월 기준)")
top_n = st.slider("Top N", min_value=3, max_value=10, value=5, step=1)

if not score_hist.empty:
    latest_month = score_hist.index.max()
    st.caption(f"최근 월: **{latest_month.strftime('%Y-%m')}**")
    lb_cols = st.columns(3)
    i = 0
    for cat, tools in CATEGORIES.items():
        sub = score_hist.loc[latest_month, score_hist.columns.intersection(tools)].sort_values(ascending=False)
        df_show = pd.DataFrame({"Rank": range(1, len(sub)+1), "Tool": sub.index, "Score": sub.values}).head(top_n)
        with lb_cols[i % 3]:
            st.markdown(f"**{cat}**")
            st.dataframe(df_show, use_container_width=True, hide_index=True)
        i += 1
else:
    st.info("점수 데이터가 없어 리더보드를 만들 수 없습니다.")

# 분야별 순위 히스토리
st.markdown("---")
st.header("📉 분야별 순위 히스토리 (시간 흐름)")

if score_hist.empty or not rank_histories:
    st.info("점수 또는 랭크 데이터가 없어 히스토리를 표시할 수 없습니다.")
else:
    cat = st.selectbox("분야 선택", list(CATEGORIES.keys()))
    lookback_months = st.slider("최근 N개월 보기", min_value=6, max_value=48, value=24, step=3)
    smooth = st.checkbox("이동평균(3개월)로 부드럽게 보기", value=False)
    show_top_k = st.slider("상위 K개만 표시 (현재 월 기준)", min_value=3,
                           max_value=min(10, len(CATEGORIES[cat])), value=5, step=1)

    rdf = rank_histories.get(cat, pd.DataFrame())
    if rdf is None or rdf.empty:
        st.warning("해당 분야에 순위 데이터가 없습니다.")
    else:
        rdf = rdf.sort_index()
        if len(rdf) > lookback_months:
            rdf = rdf.iloc[-lookback_months:]

        latest = rdf.iloc[-1].dropna().sort_values()  # 낮을수록 상위
        topk_tools = latest.index[:show_top_k].tolist()
        rdf = rdf[topk_tools]

        if smooth and len(rdf) >= 3:
            rdf = rdf.rolling(3, min_periods=1).mean()

        plot_df = rdf.reset_index().melt(id_vars="month", var_name="tool", value_name="rank")
        plot_df["month"] = pd.to_datetime(plot_df["month"])

        chart = alt.Chart(plot_df).mark_line(point=True).encode(
            x=alt.X('month:T', title='Month'),
            y=alt.Y('rank:Q', title='Rank (1=Best)', scale=alt.Scale(reverse=True)),
            color=alt.Color('tool:N', title='Tool'),
            tooltip=['month:T', 'tool:N', alt.Tooltip('rank:Q', format='.0f')]
        ).properties(height=360, width='container')

        st.altair_chart(chart, use_container_width=True)

        st.subheader("현재 월 순위")
        current_rank_tbl = latest.reset_index()
        current_rank_tbl.columns = ["Tool", "Rank"]
        st.dataframe(current_rank_tbl, use_container_width=True)

        st.download_button(
            "⬇️ 이 분야의 순위 히스토리 CSV",
            rdf.to_csv(index=True).encode("utf-8"),
            file_name=f"rank_history_{cat.replace(' ', '_')}.csv",
            mime="text/csv"
        )

# 참고/주의
with st.expander("ℹ️ 참고 및 주의사항"):
    st.markdown("""
- **지속 추적 방식**: 실행 시 새 월 데이터가 있으면 history CSV에 병합/갱신합니다.
- **복합 점수**: Trends(min-max) 0.6 + Wiki(zscore→min-max) 0.4 가중합(코드에서 조정 가능).
- **Google Trends는 상대지표**입니다. 동일 키워드 세트를 유지하면 비교 일관성이 좋아집니다.
- **Wikipedia Pageviews**는 절대 조회수이며 일부 도구는 정확 위키 문서가 없을 수 있습니다(그 경우 빈값 처리).
- **403/429** 방지를 위해 User-Agent를 **본인 연락처 포함** 값으로 변경하세요.
""")
