from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskResult:
    score: int
    band: str
    components: dict[str, float]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def risk_band(score: int) -> str:
    if score <= 30:
        return "low risk"
    if score <= 60:
        return "moderate"
    if score <= 80:
        return "elevated"
    return "high risk"


def calculate_risk_score(latest_row: pd.Series, sentiment_negativity: float, event_severity: float) -> RiskResult:
    volatility = _clip01(latest_row.get("Rolling_Volatility_20D", 0) / 0.05)
    anomaly = _clip01(latest_row.get("Anomaly_Score", 0))
    bearish_trend = 1.0 if bool(latest_row.get("Bearish_Trend", 0)) else 0.0
    price_stress = _clip01(abs(latest_row.get("Price_ZScore", 0)) / 3)
    volume_stress = _clip01(abs(latest_row.get("Volume_ZScore", 0)) / 3)
    technical_stress = max(price_stress, volume_stress)

    components = {
        "volatility": volatility,
        "anomaly": anomaly,
        "sentiment_negativity": _clip01(sentiment_negativity),
        "bearish_trend": bearish_trend,
        "event_severity": _clip01(event_severity),
        "technical_stress": technical_stress,
    }
    weighted = (
        0.22 * components["volatility"]
        + 0.22 * components["anomaly"]
        + 0.20 * components["sentiment_negativity"]
        + 0.16 * components["bearish_trend"]
        + 0.14 * components["event_severity"]
        + 0.06 * components["technical_stress"]
    )
    score = int(round(_clip01(weighted) * 100))
    return RiskResult(score=score, band=risk_band(score), components=components)

