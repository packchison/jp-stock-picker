"""yfinance data access with disk caching and threaded fetches.

Two kinds of data are cached separately, both keyed by trading date so a
same-day re-run is instant and later runs auto-refresh:
  - price history (OHLCV) per ticker, fetched in one batched yf.download call
  - fundamentals snapshot (.info dict) per ticker, fetched with a thread pool
    since yfinance has no bulk endpoint for it
"""
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import config

config.CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_stamp() -> str:
    """A cache key that rolls over once per CACHE_TTL_HOURS window."""
    bucket = int(time.time() // (config.CACHE_TTL_HOURS * 3600))
    return str(bucket)


def _history_cache_path() -> Path:
    return config.CACHE_DIR / f"history_{_cache_stamp()}.pkl"


def _fundamentals_cache_path() -> Path:
    return config.CACHE_DIR / f"fundamentals_{_cache_stamp()}.json"


def fetch_price_history(tickers: list[str], period: str = None) -> dict[str, pd.DataFrame]:
    """Batch-download daily OHLCV history for all tickers, with disk caching.

    Returns {ticker: DataFrame} — missing/failed tickers are omitted.
    """
    period = period or config.HISTORY_PERIOD
    cache_path = _history_cache_path()
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if set(tickers) <= set(cached.keys()):
            return {t: cached[t] for t in tickers if t in cached}

    raw = yf.download(
        tickers,
        period=period,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    result: dict[str, pd.DataFrame] = {}
    if len(tickers) == 1:
        t = tickers[0]
        df = raw.dropna(how="all")
        if not df.empty:
            result[t] = df
    else:
        for t in tickers:
            try:
                df = raw[t].dropna(how="all")
            except (KeyError, TypeError):
                continue
            if not df.empty:
                result[t] = df

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def _fetch_one_info(ticker: str) -> tuple[str, dict]:
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    return ticker, info


def fetch_fundamentals(tickers: list[str], max_workers: int = None) -> dict[str, dict]:
    """Fetch .info snapshots for all tickers in parallel, with disk caching."""
    max_workers = max_workers or config.MAX_WORKERS
    cache_path = _fundamentals_cache_path()
    cached: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)

    missing = [t for t in tickers if t not in cached]
    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_fetch_one_info, t) for t in missing]
            for fut in as_completed(futures):
                t, info = fut.result()
                cached[t] = info

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cached, f)

    return {t: cached[t] for t in tickers if t in cached and cached[t]}


def clear_cache() -> None:
    for p in config.CACHE_DIR.glob("*"):
        p.unlink()
