from __future__ import annotations

import pandas as pd

from .risk import RiskResult


def build_explanation(latest_row: pd.Series, risk: RiskResult, event_labels: list[str]) -> str:
    reasons: list[str] = []
    if risk.components.get("anomaly", 0) > 0.65:
        reasons.append("anomalous market behavior")
    if abs(latest_row.get("Volume_ZScore", 0)) > 2:
        reasons.append("abnormal volume spike")
    if abs(latest_row.get("Price_ZScore", 0)) > 2:
        reasons.append("unusual price movement")
    if risk.components.get("bearish_trend", 0) > 0:
        reasons.append("bearish EMA20 versus EMA50 trend")
    if risk.components.get("sentiment_negativity", 0) > 0.25:
        reasons.append("negative news sentiment")

    severe_events = [label for label in event_labels if label not in {"general market news", "product launch", "AI announcement"}]
    if severe_events:
        reasons.append(f"{severe_events[0]} event risk")

    if not reasons:
        reasons.append("stable technical signals and limited negative news pressure")

    return f"Risk is {risk.band} ({risk.score}/100) due to " + ", ".join(reasons) + "."

