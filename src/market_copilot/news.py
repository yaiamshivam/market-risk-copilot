from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus

import feedparser


TICKER_ALIASES = {
    "AAPL": "Apple",
    "NFLX": "Netflix",
    "NVIDIA": "Nvidia",
    "NVDA": "Nvidia",
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "SBIN": "State Bank of India",
    "NIFTY": "Nifty 50",
    "SENSEX": "BSE Sensex",
}


@dataclass(frozen=True)
class NewsItem:
    title: str
    link: str
    published: str
    source: str


def ticker_query(ticker: str) -> str:
    return TICKER_ALIASES.get(ticker.upper(), ticker.upper())


def fetch_google_news(ticker: str, limit: int = 8) -> list[NewsItem]:
    query = quote_plus(f"{ticker_query(ticker)} stock market")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    items: list[NewsItem] = []
    for entry in feed.entries[:limit]:
        items.append(
            NewsItem(
                title=getattr(entry, "title", ""),
                link=getattr(entry, "link", ""),
                published=getattr(entry, "published", ""),
                source=getattr(getattr(entry, "source", None), "title", "Google News"),
            )
        )
    return items

