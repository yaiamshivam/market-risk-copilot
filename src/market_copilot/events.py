from __future__ import annotations


EVENT_RULES = {
    "earnings": ["earnings", "quarterly", "results", "profit", "revenue", "guidance"],
    "regulation": ["regulator", "regulatory", "sec", "rbi", "sebi", "antitrust", "compliance"],
    "lawsuit": ["lawsuit", "sues", "sued", "court", "legal", "settlement"],
    "acquisition": ["acquire", "acquisition", "merger", "buyout", "stake"],
    "layoffs": ["layoff", "layoffs", "job cuts", "workforce", "headcount"],
    "product launch": ["launch", "unveils", "introduces", "product", "release"],
    "macro policy": ["fed", "rate cut", "interest rate", "inflation", "budget", "policy", "tariff"],
    "geopolitical risk": ["war", "sanction", "geopolitical", "conflict", "border"],
    "AI announcement": ["ai", "artificial intelligence", "gpu", "chip", "model"],
    "market crash/panic": ["crash", "panic", "selloff", "plunge", "bear market"],
}

EVENT_SEVERITY = {
    "earnings": 0.35,
    "regulation": 0.70,
    "lawsuit": 0.75,
    "acquisition": 0.45,
    "layoffs": 0.55,
    "product launch": 0.25,
    "macro policy": 0.60,
    "geopolitical risk": 0.80,
    "AI announcement": 0.30,
    "market crash/panic": 0.90,
    "general market news": 0.20,
}


def classify_event(text: str) -> str:
    lowered = text.lower()
    for event, keywords in EVENT_RULES.items():
        if any(keyword in lowered for keyword in keywords):
            return event
    return "general market news"


def event_severity(event_labels: list[str]) -> float:
    if not event_labels:
        return 0.0
    return max(EVENT_SEVERITY.get(label, 0.2) for label in event_labels)

