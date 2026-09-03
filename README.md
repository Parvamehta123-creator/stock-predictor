# Stock movement predictor

An end-to-end ML system that predicts next-day stock price *direction* (up/down),
with a FastAPI backend and Streamlit dashboard.

**This predicts direction, not price.** Stock prices are close to a random walk;
claiming to predict exact prices is not a credible claim. What's testable and
useful is whether a model can identify a statistical edge in direction, sized
and evaluated honestly against a buy-and-hold baseline.

## Status

- [x] Data pipeline (real + synthetic sources, same schema)
- [x] Feature engineering (technical indicators, no lookahead bias -- tested)
- [x] Sentiment pipeline (commentary notes -> per-ticker sentiment features, tested)
- [ ] Model training + walk-forward validation
- [ ] Backtesting framework
- [ ] FastAPI backend
- [ ] Streamlit dashboard
- [ ] Docker + deployment

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Get data

Real data (requires internet):
```bash
python pipeline/fetch_data.py AAPL --start 2018-01-01
```

Synthetic data (for development/testing without hitting an API):
```bash
python pipeline/generate_sample_data.py DEMO --days 750
```

Both write to `data/raw/{TICKER}.csv` with identical schema
(`Date, Open, High, Low, Close, Volume`), so everything downstream works
with either source unmodified.

### 2. Build features

```bash
python pipeline/features.py AAPL   # or DEMO
```

Writes `data/processed/{TICKER}_features.csv`. This step:
- Computes technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, volume ratios)
- Labels each row with `target`: 1 if next-day close is higher, else 0
- Drops rows with unresolved indicators (rolling-window warm-up) or unresolved
  targets (final row(s), since there's no "next day" for them yet)

### 3. Run tests

```bash
pytest tests/ -v
```

The most important test (`test_indicators_do_not_use_future_data`) proves
that truncating the dataset to an earlier date doesn't change any
already-computed feature value -- i.e., no feature is secretly using future
information. This is the #1 bug in stock-prediction projects and the #1
thing to be able to explain if asked about it.

### 4. Add sentiment features (optional, per ticker)

Drop commentary notes (`.docx`) into `data/notes/raw/`. Each note must contain
a `DD/MM/YYYY` date somewhere in the text -- that's how it gets placed on the
timeline. Then:

```bash
python -m pipeline.merge_sentiment CYIENT
```

This extracts company mentions and sentiment from every note (fuzzy-matched
against `WATCHLIST` in `pipeline/sentiment.py` -- typos like "Glaxosmithcline"
for "GlaxoSmithKline" are tolerated), rolls note dates forward to the next
actual trading day, and merges the result onto `{TICKER}_features.csv`,
producing `{TICKER}_features_with_sentiment.csv` with `mentioned`,
`mention_sentiment`, and `macro_sentiment` columns.

**Extend `WATCHLIST` and `POSITIVE_WORDS`/`NEGATIVE_WORDS` in
`pipeline/sentiment.py`** as you feed it more notes -- the lexicon only knows
what you put in it. The sentiment scoring is a rule-based v1 by design (see
the module docstring for the upgrade path to a real model like FinBERT).

## Design notes

- **Data source is swappable.** `fetch_data.py` (real) and `generate_sample_data.py`
  (synthetic) both write the same schema. Feature engineering, modeling, and serving
  code never know or care which one produced the data.
- **Lookahead bias is tested, not just avoided.** See `tests/test_features.py`.
- **Validation must be walk-forward, not random k-fold**, once we get to modeling --
  shuffling time-series data leaks future information into training folds and
  silently inflates reported accuracy.
- **A note dated on a non-trading day doesn't just vanish.** `align_to_trading_days`
  in `pipeline/merge_sentiment.py` rolls it forward to the next real trading day.
  Without this, a plain merge on Date silently drops the row -- no error, no
  warning, just a sentiment column that's suspiciously all-zero. Found this by
  noticing the output looked "too clean," which is itself worth remembering:
  a data pipeline that runs without errors has told you nothing about whether
  it's correct.
