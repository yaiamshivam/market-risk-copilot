from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator, MACD
    from ta.volatility import AverageTrueRange, BollingerBands
except Exception:  # pragma: no cover - fallback helps Colab/local partial installs
    RSIIndicator = None
    EMAIndicator = None
    MACD = None
    AverageTrueRange = None
    BollingerBands = None


def _zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window, min_periods=5).mean()
    std = series.rolling(window, min_periods=5).std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0)


def _fallback_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker, part in df.groupby("Ticker", sort=False):
        ticker_df = part.sort_values("Date").copy()
        close = ticker_df["Close"]
        high = ticker_df["High"]
        low = ticker_df["Low"]
        volume = ticker_df["Volume"].replace(0, np.nan)

        ticker_df["Daily_Return"] = close.pct_change().fillna(0)
        ticker_df["Rolling_Return_5D"] = close.pct_change(5).fillna(0)
        ticker_df["Rolling_Return_20D"] = close.pct_change(20).fillna(0)
        ticker_df["Rolling_Volatility_20D"] = ticker_df["Daily_Return"].rolling(20, min_periods=5).std().fillna(0)

        if RSIIndicator:
            ticker_df["RSI"] = RSIIndicator(close=close, window=14).rsi().fillna(50)
            macd = MACD(close=close)
            ticker_df["MACD"] = macd.macd().fillna(0)
            ticker_df["MACD_Signal"] = macd.macd_signal().fillna(0)
            ticker_df["EMA20"] = EMAIndicator(close=close, window=20).ema_indicator().bfill().ffill()
            ticker_df["EMA50"] = EMAIndicator(close=close, window=50).ema_indicator().bfill().ffill()
            bands = BollingerBands(close=close, window=20, window_dev=2)
            ticker_df["BB_High"] = bands.bollinger_hband().bfill().ffill()
            ticker_df["BB_Low"] = bands.bollinger_lband().bfill().ffill()
            ticker_df["ATR"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().fillna(0)
        else:
            ticker_df["RSI"] = _fallback_rsi(close)
            ticker_df["EMA20"] = close.ewm(span=20, adjust=False).mean()
            ticker_df["EMA50"] = close.ewm(span=50, adjust=False).mean()
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            ticker_df["MACD"] = ema12 - ema26
            ticker_df["MACD_Signal"] = ticker_df["MACD"].ewm(span=9, adjust=False).mean()
            rolling_mean = close.rolling(20, min_periods=5).mean()
            rolling_std = close.rolling(20, min_periods=5).std().fillna(0)
            ticker_df["BB_High"] = (rolling_mean + 2 * rolling_std).bfill().ffill()
            ticker_df["BB_Low"] = (rolling_mean - 2 * rolling_std).bfill().ffill()
            prev_close = close.shift(1)
            true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            ticker_df["ATR"] = true_range.rolling(14, min_periods=5).mean().fillna(0)

        ticker_df["Momentum_10D"] = close - close.shift(10)
        ticker_df["Drawdown"] = (close / close.cummax() - 1).fillna(0)
        ticker_df["Volume_ZScore"] = _zscore(volume.fillna(0))
        ticker_df["Price_ZScore"] = _zscore(close)
        ticker_df["Bearish_Trend"] = (ticker_df["EMA20"] < ticker_df["EMA50"]).astype(int)
        frames.append(ticker_df)

    featured = pd.concat(frames, ignore_index=True)
    return featured.replace([np.inf, -np.inf], np.nan).fillna(0)


def latest_ticker_frame(featured: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return featured[featured["Ticker"] == ticker].sort_values("Date").copy()

