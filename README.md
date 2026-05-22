# AI Market Intelligence Copilot

An end-to-end Python MVP for explainable market intelligence using historical OHLCV data and live Google News RSS sentiment.

## What It Does

Select a ticker and the app returns:

- technical analysis summary
- Isolation Forest anomaly detection
- latest relevant news headlines
- headline sentiment with FinBERT when available
- rule-based event classification
- explainable 0-100 risk score
- human-readable explanation

## Dataset

The default dataset is stored at:

```text
data/raw/Trading View (Mission 300cr) - 2025.csv
```

Expected columns:

```text
Date, Open, High, Low, Close, Volume, Ticker
```

The loader also supports the original `tick` column name and renames it to `Ticker`.

## Project Structure

```text
.
├── app.py
├── data/
│   └── raw/
├── notebooks/
├── reports/
├── src/
│   └── market_copilot/
│       ├── anomaly.py
│       ├── data_loader.py
│       ├── events.py
│       ├── explain.py
│       ├── features.py
│       ├── news.py
│       ├── risk.py
│       └── sentiment.py
├── requirements.txt
└── README.md
```

## Local Setup

```powershell
cd "D:\AI Market Intelligence Copilot"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Google Colab Setup

```python
!pip install -r requirements.txt
```

Then run the modules or launch the Streamlit app through a tunnel such as `pyngrok` if needed.

## Notes

- News uses Google News RSS, a free source.
- Sentiment prefers `ProsusAI/finbert`. If the model cannot download or load, the app falls back to a simple keyword heuristic so the MVP remains usable.
- The risk score is explainable and weighted from volatility, anomaly score, negative sentiment, bearish trend, event severity, and technical stress.

