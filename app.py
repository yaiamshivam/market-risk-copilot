from __future__ import annotations

from datetime import datetime
from pathlib import Path


import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.market_copilot.anomaly import add_anomaly_scores
from src.market_copilot.data_loader import get_tickers, load_market_data
from src.market_copilot.events import classify_event, event_severity
from src.market_copilot.explain import build_explanation
from src.market_copilot.features import engineer_features, latest_ticker_frame
from src.market_copilot.news import fetch_google_news, ticker_query
from src.market_copilot.risk import calculate_risk_score
from src.market_copilot.sentiment import SentimentAnalyzer, sentiment_negativity


DEFAULT_CSV = Path("data/raw/Trading View (Mission 300cr) - 2025.csv")
GOOGLE_SHEETS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTBmX3WtsWtqU5mDtb2rudMx_PwRCbrSpALMzyqnjxzDRRhzIPC7ZlMjf7o7kPKVwMGMySMxLQecvPt/pub?output=csv"

def ticker_from_sheet_label(label: object) -> str:
    text = str(label).strip().upper()
    if "RELIANCE" in text:
        return "RELIANCE"
    if "TATA" in text or text == "TCS":
        return "TCS"
    if "STATE BANK" in text or text == "SBIN":
        return "SBIN"
    if "SENSEX" in text:
        return "SENSEX"
    if "NIFTY" in text:
        return "NIFTY"
    if "NVIDIA" in text or "NVDA" in text:
        return "NVIDIA"
    if "APPLE" in text or "AAPL" in text:
        return "AAPL"
    if "NETFLIX" in text or "NFLX" in text:
        return "NFLX"
    return text.split("(")[0].strip().replace(" ", "_")


def normalize_google_finance_sheet(sheet_df: pd.DataFrame) -> pd.DataFrame:
    header_matches = sheet_df.apply(
        lambda row: row.astype(str).str.strip().str.lower().eq("date").sum(), axis=1
    )
    header_row = int(header_matches.idxmax())
    if header_matches.loc[header_row] == 0:
        raise ValueError("Could not find the Google Finance header row in the published sheet.")

    asset_row = max(0, header_row - 1)
    date_columns = [
        int(column_index)
        for column_index, value in sheet_df.iloc[header_row].items()
        if str(value).strip().lower() == "date"
    ]

    frames = []
    for start in date_columns:
        block = sheet_df.iloc[header_row + 1 :, start : start + 6].copy()
        if block.shape[1] < 6:
            continue
        block.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        block = block.dropna(how="all")
        block = block[block["Date"].notna()].copy()
        if block.empty:
            continue

        ticker = ticker_from_sheet_label(sheet_df.iloc[asset_row, start])
        block["Ticker"] = ticker
        block["Date"] = pd.to_datetime(block["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
        frames.append(block)

    if not frames:
        raise ValueError("No ticker blocks were found in the published Google Sheet.")
    return pd.concat(frames, ignore_index=True)



st.set_page_config(
    page_title="AI Market Intelligence Copilot",
    layout="wide",
)


@st.cache_data(ttl=1800, show_spinner=False)
def load_live_market_data() -> tuple[pd.DataFrame, str, datetime, str | None]:
    try:
        live_raw = pd.read_csv(GOOGLE_SHEETS_CSV_URL, header=None)
        normalized_live = normalize_google_finance_sheet(live_raw)
        temp_path = Path("data/raw/live_google_sheet_cache.csv")
        normalized_live.to_csv(temp_path, index=False)
        raw = load_market_data(temp_path)
        source = "Google Sheets (Google Finance)"
        warning = None
    except Exception as exc:
        raw = load_market_data(DEFAULT_CSV)
        source = "Bundled CSV fallback"
        warning = f"Google Sheets is temporarily unavailable, so the bundled CSV is being used. Details: {exc}"

    featured = engineer_features(raw)
    scored = add_anomaly_scores(featured)
    return scored, source, datetime.now(), warning

@st.cache_resource(show_spinner=False)
def load_sentiment_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


@st.cache_data(ttl=900, show_spinner=False)
def load_news(ticker: str) -> list[dict[str, str]]:
    return [item.__dict__ for item in fetch_google_news(ticker)]


def make_candlestick_chart(ticker_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=ticker_df["Date"],
            open=ticker_df["Open"],
            high=ticker_df["High"],
            low=ticker_df["Low"],
            close=ticker_df["Close"],
            name="OHLC",
        )
    )
    fig.add_trace(go.Scatter(x=ticker_df["Date"], y=ticker_df["EMA20"], name="EMA20", line=dict(width=1.6)))
    fig.add_trace(go.Scatter(x=ticker_df["Date"], y=ticker_df["EMA50"], name="EMA50", line=dict(width=1.6)))
    fig.add_trace(
        go.Scatter(
            x=ticker_df["Date"],
            y=ticker_df["BB_High"],
            name="Bollinger High",
            line=dict(width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ticker_df["Date"],
            y=ticker_df["BB_Low"],
            name="Bollinger Low",
            line=dict(width=1, dash="dot"),
        )
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
    return fig


def make_anomaly_chart(ticker_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ticker_df["Date"], y=ticker_df["Close"], name="Close", line=dict(width=1.8)))
    anomalies = ticker_df[ticker_df["Is_Anomaly"] == 1]
    fig.add_trace(
        go.Scatter(
            x=anomalies["Date"],
            y=anomalies["Close"],
            name="Anomaly",
            mode="markers",
            marker=dict(size=9, color="#d62728", symbol="x"),
        )
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def make_risk_gauge(score: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 30], "color": "#d8f3dc"},
                    {"range": [31, 60], "color": "#fff3b0"},
                    {"range": [61, 80], "color": "#ffd6a5"},
                    {"range": [81, 100], "color": "#ffadad"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def analyze_news(ticker: str) -> pd.DataFrame:
    analyzer = load_sentiment_analyzer()
    rows = []
    for item in load_news(ticker):
        sentiment = analyzer.analyze(item["title"])
        event = classify_event(item["title"])
        rows.append(
            {
                "headline": item["title"],
                "source": item["source"],
                "published": item["published"],
                "sentiment": sentiment.label,
                "confidence": round(sentiment.confidence, 3),
                "event": event,
                "link": item["link"],
            }
        )
    return pd.DataFrame(rows)


st.title("AI Market Intelligence Copilot")
st.caption("Explainable technical, anomaly, news sentiment, event, and risk intelligence for mixed India and US market data.")

try:
    data, data_source, last_updated, data_warning = load_live_market_data()
except Exception as exc:
    st.error(f"Could not load market data from Google Sheets or the bundled fallback CSV: {exc}")
    st.stop()

if data_warning:
    st.warning(data_warning)

status_color = "#16a34a" if data_source.startswith("Google Sheets") else "#d97706"
status_label = "LIVE MARKET DATA" if data_source.startswith("Google Sheets") else "FALLBACK DATA"
assets_count = data["Ticker"].nunique() if "Ticker" in data.columns else 0
max_date = data["Date"].max() if "Date" in data.columns else None
range_end = "Present" if data_source.startswith("Google Sheets") else max_date.strftime("%d %b %Y")

st.markdown(
    f"""
    <div style="border:1px solid #dbe7dd;border-radius:14px;padding:16px 18px;margin:12px 0 22px 0;background:linear-gradient(90deg,#f0fdf4,#ffffff);box-shadow:0 1px 3px rgba(15,23,42,0.06);">
        <div style="display:flex;align-items:center;gap:10px;font-weight:800;font-size:18px;color:{status_color};">
            <span style="height:11px;width:11px;border-radius:50%;background:{status_color};display:inline-block;"></span>
            🟢 {status_label}
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:12px;color:#334155;font-size:14px;">
            <div><strong>Source:</strong><br>{data_source}</div>
            <div><strong>Last Updated:</strong><br>{last_updated.strftime('%d %b %Y, %I:%M %p')}</div>
            <div><strong>Assets Monitored:</strong><br>{assets_count}</div>
            <div><strong>Date Range:</strong><br>01 Jan 2024 &rarr; {range_end}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    st.caption("Historical and newly updated market data load from Google Sheets automatically.")

tickers = get_tickers(data)
with st.sidebar:
    selected_ticker = st.selectbox("Ticker", tickers, index=0)
    lookback = st.slider("Chart lookback days", min_value=60, max_value=600, value=250, step=10)

ticker_df = latest_ticker_frame(data, selected_ticker)
latest = ticker_df.iloc[-1]
news_df = analyze_news(selected_ticker)
sentiment_results = []
if not news_df.empty:
    analyzer = load_sentiment_analyzer()
    sentiment_results = [analyzer.analyze(headline) for headline in news_df["headline"].tolist()]
event_labels = news_df["event"].tolist() if not news_df.empty else []
risk = calculate_risk_score(latest, sentiment_negativity(sentiment_results), event_severity(event_labels))
explanation = build_explanation(latest, risk, event_labels)

top_metrics = st.columns(5)
top_metrics[0].metric("Ticker", selected_ticker)
top_metrics[1].metric("Latest Close", f"{latest['Close']:.2f}")
top_metrics[2].metric("RSI", f"{latest['RSI']:.1f}")
top_metrics[3].metric("Anomaly Score", f"{latest['Anomaly_Score']:.2f}")
top_metrics[4].metric("Risk", f"{risk.score}/100", risk.band)

chart_df = ticker_df.tail(lookback)
left, right = st.columns([2.1, 1])
with left:
    st.subheader(f"{selected_ticker} price action")
    st.plotly_chart(make_candlestick_chart(chart_df), use_container_width=True)
with right:
    st.subheader("Explainable risk")
    st.plotly_chart(make_risk_gauge(risk.score), use_container_width=True)
    st.info(explanation)

indicator_cols = st.columns(4)
indicator_cols[0].metric("EMA20", f"{latest['EMA20']:.2f}")
indicator_cols[1].metric("EMA50", f"{latest['EMA50']:.2f}")
indicator_cols[2].metric("ATR", f"{latest['ATR']:.2f}")
indicator_cols[3].metric("Drawdown", f"{latest['Drawdown']:.2%}")

st.subheader("Anomaly detection")
st.plotly_chart(make_anomaly_chart(chart_df), use_container_width=True)

st.subheader("Technical snapshot")
technical_columns = [
    "Date",
    "Close",
    "Daily_Return",
    "Rolling_Return_5D",
    "Rolling_Volatility_20D",
    "RSI",
    "MACD",
    "Volume_ZScore",
    "Price_ZScore",
    "Anomaly_Score",
    "Is_Anomaly",
]
st.dataframe(ticker_df[technical_columns].tail(20).sort_values("Date", ascending=False), use_container_width=True)

st.subheader(f"Latest news for {ticker_query(selected_ticker)}")
if news_df.empty:
    st.warning("No live news was returned. Check internet access or try again later.")
else:
    st.dataframe(
        news_df[["headline", "source", "published", "sentiment", "confidence", "event"]],
        use_container_width=True,
        hide_index=True,
    )
    for _, row in news_df.iterrows():
        st.markdown(f"- [{row['headline']}]({row['link']})")

with st.expander("Risk components"):
    component_df = pd.DataFrame(
        [{"component": key, "normalized_value": round(value, 3)} for key, value in risk.components.items()]
    )
    st.dataframe(component_df, use_container_width=True, hide_index=True)









