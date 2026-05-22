from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
INDIA_MARKET_KEYWORDS = ["NIFTY", "SENSEX", "RELIANCE", "TCS", "SBIN", ".NS", ".BO"]


@dataclass(frozen=True)
class MarketAudit:
    rows: int
    columns: int
    date_start: pd.Timestamp | None
    date_end: pd.Timestamp | None
    missing_values: dict[str, int]
    exact_duplicates: int
    ticker_date_duplicates: int | None
    ticker_counts: dict[str, int]
    market_counts: dict[str, int]
    invalid_ohlc_rows: int
    max_gap_by_ticker: dict[str, int | None]


def classify_market(ticker: object) -> str:
    ticker_text = str(ticker).upper()
    if any(keyword in ticker_text for keyword in INDIA_MARKET_KEYWORDS):
        return "India"
    return "US/Other"


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    if "tick" in cleaned.columns and "Ticker" not in cleaned.columns:
        cleaned = cleaned.rename(columns={"tick": "Ticker"})

    cleaned = cleaned.dropna(how="all")

    if "Date" in cleaned.columns:
        cleaned["Date"] = pd.to_datetime(
            cleaned["Date"].astype(int),
            unit="D",
            origin="1899-12-30",
        )

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "Ticker" in cleaned.columns:
        cleaned["Market"] = cleaned["Ticker"].apply(classify_market)

    return cleaned


def count_invalid_ohlc_rows(df: pd.DataFrame) -> int:
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        return 0

    invalid_rows = df[
        (df["High"] < df["Low"])
        | (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["Low"] > df["Open"])
        | (df["Low"] > df["Close"])
    ]
    return int(len(invalid_rows))


def calculate_max_gaps(df: pd.DataFrame) -> dict[str, int | None]:
    if not {"Ticker", "Date"}.issubset(df.columns):
        return {}

    gaps: dict[str, int | None] = {}
    for ticker, ticker_df in df.sort_values("Date").groupby("Ticker"):
        gap_series = ticker_df["Date"].diff().dt.days.dropna()
        gaps[str(ticker)] = int(gap_series.max()) if not gap_series.empty else None
    return gaps


def audit_market_data(df: pd.DataFrame) -> MarketAudit:
    cleaned = clean_market_data(df)

    date_start = cleaned["Date"].min() if "Date" in cleaned.columns else None
    date_end = cleaned["Date"].max() if "Date" in cleaned.columns else None

    ticker_date_duplicates = None
    if {"Ticker", "Date"}.issubset(cleaned.columns):
        ticker_date_duplicates = int(cleaned.duplicated(subset=["Ticker", "Date"]).sum())

    ticker_counts = {}
    if "Ticker" in cleaned.columns:
        ticker_counts = {
            str(ticker): int(count)
            for ticker, count in cleaned["Ticker"].value_counts().items()
        }

    market_counts = {}
    if "Market" in cleaned.columns:
        market_counts = {
            str(market): int(count)
            for market, count in cleaned["Market"].value_counts().items()
        }

    return MarketAudit(
        rows=int(cleaned.shape[0]),
        columns=int(cleaned.shape[1]),
        date_start=date_start,
        date_end=date_end,
        missing_values={str(column): int(count) for column, count in cleaned.isnull().sum().items()},
        exact_duplicates=int(cleaned.duplicated().sum()),
        ticker_date_duplicates=ticker_date_duplicates,
        ticker_counts=ticker_counts,
        market_counts=market_counts,
        invalid_ohlc_rows=count_invalid_ohlc_rows(cleaned),
        max_gap_by_ticker=calculate_max_gaps(cleaned),
    )

