from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


ANOMALY_FEATURES = [
    "Daily_Return",
    "Rolling_Return_5D",
    "Rolling_Volatility_20D",
    "RSI",
    "MACD",
    "ATR",
    "Momentum_10D",
    "Drawdown",
    "Volume_ZScore",
    "Price_ZScore",
]


def add_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    output = []
    for ticker, part in df.groupby("Ticker", sort=False):
        ticker_df = part.sort_values("Date").copy()
        available = [column for column in ANOMALY_FEATURES if column in ticker_df.columns]
        if len(ticker_df) < 20 or not available:
            ticker_df["Anomaly_Score"] = 0.0
            ticker_df["Is_Anomaly"] = 0
            output.append(ticker_df)
            continue

        x = ticker_df[available].fillna(0)
        scaled = StandardScaler().fit_transform(x)
        model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
        labels = model.fit_predict(scaled)
        raw_scores = -model.decision_function(scaled)
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        normalized = (raw_scores - min_score) / (max_score - min_score) if max_score > min_score else raw_scores * 0
        ticker_df["Anomaly_Score"] = normalized
        ticker_df["Is_Anomaly"] = (labels == -1).astype(int)
        output.append(ticker_df)

    return pd.concat(output, ignore_index=True)

