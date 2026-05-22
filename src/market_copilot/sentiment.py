from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentResult:
    label: str
    confidence: float


POSITIVE_WORDS = {
    "beat",
    "beats",
    "growth",
    "surge",
    "rally",
    "gain",
    "record",
    "profit",
    "upgrade",
    "launch",
    "partnership",
    "expands",
}
NEGATIVE_WORDS = {
    "miss",
    "falls",
    "fall",
    "drop",
    "crash",
    "lawsuit",
    "probe",
    "layoffs",
    "cut",
    "downgrade",
    "loss",
    "regulatory",
    "risk",
    "panic",
}


class SentimentAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        self.model_name = model_name
        self.pipeline = None
        self.error: str | None = None
        try:
            from transformers import pipeline

            self.pipeline = pipeline("sentiment-analysis", model=model_name, truncation=True)
        except Exception as exc:
            self.error = str(exc)

    def analyze(self, text: str) -> SentimentResult:
        if self.pipeline is not None:
            result = self.pipeline(text[:512])[0]
            return SentimentResult(label=str(result["label"]).lower(), confidence=float(result["score"]))
        return heuristic_sentiment(text)


def heuristic_sentiment(text: str) -> SentimentResult:
    words = {word.strip(".,:;!?()[]{}'\"").lower() for word in text.split()}
    positive_hits = len(words & POSITIVE_WORDS)
    negative_hits = len(words & NEGATIVE_WORDS)
    if negative_hits > positive_hits:
        return SentimentResult("negative", min(0.95, 0.55 + 0.1 * negative_hits))
    if positive_hits > negative_hits:
        return SentimentResult("positive", min(0.95, 0.55 + 0.1 * positive_hits))
    return SentimentResult("neutral", 0.55)


def sentiment_negativity(results: list[SentimentResult]) -> float:
    if not results:
        return 0.0
    weighted = []
    for result in results:
        label = result.label.lower()
        if "negative" in label:
            weighted.append(result.confidence)
        elif "positive" in label:
            weighted.append(-0.5 * result.confidence)
        else:
            weighted.append(0.0)
    return max(0.0, min(1.0, sum(weighted) / len(results)))

