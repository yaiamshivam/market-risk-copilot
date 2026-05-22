from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]


def parse_market_dates(series: pd.Series) -> pd.Series:
    """Parse TradingView/Excel serial dates and normal date strings."""
    numeric_dates = pd.to_numeric(series, errors="coerce")
    parsed_numeric = pd.to_datetime(
        numeric_dates.dropna().astype(int),
        unit="D",
        origin="1899-12-30",
        errors="coerce",
    )

    parsed = pd.to_datetime(series, errors="coerce")
    if not parsed_numeric.empty:
        parsed.loc[parsed_numeric.index] = parsed_numeric
    return parsed


def load_market_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "tick" in df.columns and "Ticker" not in df.columns:
        df = df.rename(columns={"tick": "Ticker"})

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(how="all")
    df["Date"] = parse_market_dates(df["Date"])
    df = df.dropna(subset=["Date", "Ticker", "Open", "High", "Low", "Close"])

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Volume"] = df["Volume"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df.loc[df["Volume"] < 0, "Volume"] = 0

    valid_ohlc = (
        (df["High"] >= df["Low"])
        & (df["High"] >= df["Open"])
        & (df["High"] >= df["Close"])
        & (df["Low"] <= df["Open"])
        & (df["Low"] <= df["Close"])
    )
    df = df.loc[valid_ohlc].copy()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"])
    return df.reset_index(drop=True)


def get_tickers(df: pd.DataFrame) -> list[str]:
    return sorted(df["Ticker"].dropna().unique().tolist())

