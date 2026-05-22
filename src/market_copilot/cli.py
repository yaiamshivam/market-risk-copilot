from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .audit import audit_market_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit market OHLCV CSV data.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to the TradingView CSV export.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.csv)
    audit = audit_market_data(df)

    print("AI Market Intelligence Copilot - Data Audit")
    print("=" * 48)
    print(f"Rows: {audit.rows}")
    print(f"Columns: {audit.columns}")
    print(f"Date range: {audit.date_start} to {audit.date_end}")
    print(f"Exact duplicate rows: {audit.exact_duplicates}")
    print(f"Duplicate Ticker+Date rows: {audit.ticker_date_duplicates}")
    print(f"Invalid OHLC rows: {audit.invalid_ohlc_rows}")

    print("\nMarket split")
    for market, count in audit.market_counts.items():
        print(f"- {market}: {count}")

    print("\nRows per ticker")
    for ticker, count in audit.ticker_counts.items():
        print(f"- {ticker}: {count}")

    print("\nMax date gap by ticker")
    for ticker, gap in audit.max_gap_by_ticker.items():
        print(f"- {ticker}: {gap}")


if __name__ == "__main__":
    main()

