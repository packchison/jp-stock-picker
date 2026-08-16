"""Mid-term (weeks-to-a-few-months) swing/trend-following screener for JP equities.

This horizon is normally driven by institutional/foreign-investor flow trends and
Japan's quarterly earnings-guidance-revision cycle, but a true TSE foreign-flow
series and an earnings-revision index are not obtainable from yfinance. Instead
this approximates "quality momentum" with what is actually available: a
price-based relative-strength-vs-Nikkei-225 momentum signal, a Minervini-style
trend template as the core healthy-uptrend filter, and an ROE-based quality
overlay on top of fundamentals (bonus only — never disqualifying).
"""
import numpy as np
import pandas as pd

from .. import config, indicators, scoring

MIN_HISTORY_ROWS = 210
RS_LOOKBACK = 126
RS_NEAR_HIGH_THRESHOLD = 0.98

OUTPUT_COLUMNS = [
    "code", "name", "sector", "price", "score",
    "rs_near_high", "trend_template", "golden_cross", "roe", "reason",
]


def _reason(row: pd.Series) -> str:
    parts = []
    if row["trend_template"]:
        parts.append("トレンドテンプレート合格(株価が上昇中の75日線・200日線の上に位置する健全な上昇トレンド)")
    if row["rs_near_high"]:
        parts.append("対日経平均の相対力(RS)が過去半年来高値圏で推移し先導株の値動き")
    if row["golden_cross"]:
        parts.append("25日移動平均線が75日移動平均線を直近ゴールデンクロス")
    if row["roe"] is not None:
        if row["roe"] >= config.JP_STRONG_ROE:
            parts.append(f"ROE{row['roe'] * 100:.1f}%と資本コストを大きく上回る高い収益性")
        elif row["roe"] >= config.JP_COST_OF_CAPITAL_ROE:
            parts.append(f"ROE{row['roe'] * 100:.1f}%で資本コスト(8%)を上回る")
    if not parts:
        parts.append("価格モメンタムとトレンド指標を総合したスコアが相対的に高い")
    return "、".join(parts[:3]) + "。"


def screen(
    universe: list[dict],
    price_history: dict,
    benchmark_history: pd.DataFrame,
    fundamentals: dict = None,
    top_n: int = None,
) -> pd.DataFrame:
    top_n = top_n or config.DEFAULT_TOP_N
    fundamentals = fundamentals or {}
    benchmark_close = benchmark_history["Close"]

    rows = []
    for stock in universe:
        ticker = stock["ticker"]
        df = price_history.get(ticker)
        if df is None or len(df) < MIN_HISTORY_ROWS:
            continue

        close, volume = df["Close"], df["Volume"]
        latest_price = float(close.iloc[-1])
        if latest_price > config.MAX_PRICE_JPY:
            continue
        if indicators.avg_turnover_jpy(close, volume) < config.MIN_AVG_TURNOVER_JPY:
            continue

        ma25 = indicators.sma(close, 25)
        ma75 = indicators.sma(close, 75)
        ma200 = indicators.sma(close, 200)

        trend_template = indicators.trend_template_pass(close, ma75, ma200)
        golden_cross = indicators.golden_cross_recent(ma25, ma75, lookback=5)

        rs_line = indicators.relative_strength_line(close, benchmark_close)
        rs_roll_max = rs_line.rolling(RS_LOOKBACK, min_periods=40).max()
        rs_current = rs_line.iloc[-1] if len(rs_line) else np.nan
        rs_high = rs_roll_max.iloc[-1] if len(rs_roll_max) else np.nan
        rs_pct_of_high = (
            rs_current / rs_high
            if pd.notna(rs_current) and pd.notna(rs_high) and rs_high != 0
            else np.nan
        )
        rs_near_high = bool(pd.notna(rs_pct_of_high) and rs_pct_of_high >= RS_NEAR_HIGH_THRESHOLD)

        ma75_valid = ma75.dropna()
        ma75_slope = (
            (ma75.iloc[-1] - ma75.iloc[-22]) / ma75.iloc[-22] * 100
            if len(ma75_valid) > 22 and ma75.iloc[-22] != 0
            else 0.0
        )
        trend_quality_raw = (
            (2.0 if trend_template else 0.0)
            + (1.0 if golden_cross else 0.0)
            + ma75_slope
        )

        info = fundamentals.get(ticker, {}) or {}
        roe = info.get("returnOnEquity")
        try:
            roe = float(roe) if roe is not None else None
        except (TypeError, ValueError):
            roe = None

        if roe is not None:
            if roe >= config.JP_STRONG_ROE:
                quality_raw = 2.0 + (roe - config.JP_STRONG_ROE)
            elif roe >= config.JP_COST_OF_CAPITAL_ROE:
                quality_raw = 1.0 + (roe - config.JP_COST_OF_CAPITAL_ROE)
            else:
                quality_raw = roe
        else:
            quality_raw = np.nan

        rows.append({
            "ticker": ticker,
            "code": stock["code"],
            "name": stock["name"],
            "sector": stock["sector"],
            "price": latest_price,
            "rs_pct_of_high": rs_pct_of_high,
            "rs_near_high": rs_near_high,
            "trend_template": bool(trend_template),
            "trend_quality_raw": trend_quality_raw,
            "golden_cross": bool(golden_cross),
            "roe": roe,
            "quality_raw": quality_raw,
        })

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    data = pd.DataFrame(rows).set_index("ticker")

    rs_score = scoring.percentile_score(data["rs_pct_of_high"], higher_is_better=True)
    trend_score = scoring.percentile_score(data["trend_quality_raw"], higher_is_better=True)

    quality_valid = data["quality_raw"].dropna()
    if len(quality_valid):
        quality_score = scoring.percentile_score(quality_valid, higher_is_better=True)
        quality_score = quality_score.reindex(data.index).fillna(50.0)
    else:
        quality_score = pd.Series(50.0, index=data.index)

    data["score"] = scoring.weighted_composite(
        {
            "relative_strength": rs_score,
            "trend_quality": trend_score,
            "fundamental_quality": quality_score,
        },
        {
            "relative_strength": 0.4,
            "trend_quality": 0.4,
            "fundamental_quality": 0.2,
        },
    )
    data["reason"] = data.apply(_reason, axis=1)

    result = data.reset_index(drop=True)[OUTPUT_COLUMNS]
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result.head(top_n)
