"""Long-term (1+ year) fundamentals-driven screener.

Ranks the universe on value (PER/PBR), quality (ROE, leverage), dividend
(yield + payout sustainability) and growth (revenue/earnings growth) factors,
combining them into one composite score. Also flags TSE reform-catalyst
candidates: PBR < 1.0 with ROE below the TSE's own cost-of-capital benchmark,
the classic low-valuation / sub-cost-of-capital profile the exchange's 2023+
governance push is aimed at re-rating.
"""
import pandas as pd

from .. import config, indicators, scoring


def _get_price(ticker: str, price_history: dict, funda: dict):
    """Latest close from price_history if usable, else a fundamentals fallback.

    Returns (price, price_df) where price_df is only set when it came from
    price_history (used later for the liquidity check) — the fundamentals
    fallback carries no history to check liquidity against.
    """
    df = price_history.get(ticker) if price_history else None
    if df is not None and len(df) > 0 and "Close" in df.columns:
        close = df["Close"].dropna()
        if len(close):
            return float(close.iloc[-1]), df

    for key in ("currentPrice", "regularMarketPrice"):
        val = funda.get(key)
        if val is not None:
            try:
                return float(val), None
            except (TypeError, ValueError):
                continue
    return None, None


def _liquidity_ok(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 20:
        return True
    turnover = indicators.avg_turnover_jpy(df["Close"], df["Volume"])
    if turnover <= 0:
        return True
    return turnover >= config.MIN_AVG_TURNOVER_JPY


def _safe_percentile(raw: pd.Series, higher_is_better: bool) -> pd.Series:
    """percentile_score, but forces missing inputs back to NaN.

    scoring.percentile_score's na_option="bottom" ranking otherwise scores a
    missing higher_is_better=True metric as the *best* in the universe, which
    is wrong for missing fundamentals data; masking by raw.notna() restores
    "missing = no credit" and lets weighted_composite's fillna(0) treat it as
    the worst outcome for that factor instead.
    """
    score = scoring.percentile_score(raw, higher_is_better=higher_is_better)
    return score.where(raw.notna())


def _notna(x) -> bool:
    return pd.notna(x)


def _build_reason(row: dict) -> str:
    per, pbr, roe = row.get("per"), row.get("pbr"), row.get("roe")
    div, debt = row.get("dividend_yield"), row.get("debt_to_equity")
    earnings_growth = row.get("earnings_growth")
    reform = row.get("reform_candidate")

    bits = []

    val_txt = []
    if _notna(per):
        val_txt.append(f"PER{per:.1f}倍")
    if _notna(pbr):
        val_txt.append(f"PBR{pbr:.2f}倍")
    if val_txt:
        deep = _notna(per) and per > 0 and per < config.JP_DEEP_VALUE_PER
        cheap = deep or (_notna(pbr) and 0 < pbr < config.JP_VALUE_PBR) or \
            (_notna(per) and 0 < per < config.JP_CHEAP_PER)
        tag = "の超割安" if deep else ("の割安" if cheap else "")
        bits.append("・".join(val_txt) + tag)

    if _notna(roe):
        tier = "高ROE" if roe >= config.JP_STRONG_ROE else "ROE"
        bits.append(f"{tier}{roe * 100:.1f}%")

    if _notna(div):
        tag = "・高配当" if div >= config.JP_HIGH_DIVIDEND_YIELD else ""
        bits.append(f"配当利回り{div * 100:.1f}%{tag}")

    if reform:
        bits.append("東証PBR改革の再評価候補(低PBR×ROE未達)")

    if _notna(debt) and debt < 50:
        bits.append("負債少なく財務健全")

    if _notna(earnings_growth) and earnings_growth > 0:
        bits.append("増益基調")

    if not bits:
        return "主要指標のデータが限定的だが相対評価で上位選出"
    return "、".join(bits)


def screen(universe: list[dict], fundamentals: dict, price_history: dict = None, top_n: int = None) -> pd.DataFrame:
    top_n = top_n or config.DEFAULT_TOP_N
    price_history = price_history or {}
    fundamentals = fundamentals or {}

    output_cols = ["code", "name", "sector", "price", "score", "per", "pbr", "roe",
                   "dividend_yield", "debt_to_equity", "reform_candidate", "reason"]

    rows = []
    for stock in universe:
        ticker = stock["ticker"]
        funda = fundamentals.get(ticker, {}) or {}

        price, price_df = _get_price(ticker, price_history, funda)
        if price is None or price <= 0 or price > config.MAX_PRICE_JPY:
            continue
        if not _liquidity_ok(price_df):
            continue

        per = funda.get("trailingPE")
        if per is None:
            per = funda.get("forwardPE")
        pbr = funda.get("priceToBook")
        if per is None and pbr is None:
            continue

        # This yfinance version reports dividendYield as a percentage number
        # (e.g. 3.31 meaning 3.31%), not a fraction — normalize to a fraction
        # so it's directly comparable to config.JP_HIGH_DIVIDEND_YIELD (0.03)
        # and to payoutRatio, which IS already a fraction.
        raw_div_yield = funda.get("dividendYield")
        dividend_yield = raw_div_yield / 100 if raw_div_yield is not None else None

        rows.append({
            "code": stock["code"], "name": stock["name"], "sector": stock["sector"],
            "price": price,
            "per": per, "pbr": pbr,
            "roe": funda.get("returnOnEquity"),
            "dividend_yield": dividend_yield,
            "payout_ratio": funda.get("payoutRatio"),
            "debt_to_equity": funda.get("debtToEquity"),
            "revenue_growth": funda.get("revenueGrowth"),
            "earnings_growth": funda.get("earningsGrowth"),
        })

    if not rows:
        return pd.DataFrame(columns=output_cols)

    data = pd.DataFrame(rows).set_index("code")

    pbr_valid = data["pbr"].where(data["pbr"] > 0)
    data["reform_candidate"] = (
        pbr_valid.notna() & (pbr_valid < config.JP_VALUE_PBR) &
        data["roe"].notna() & (data["roe"] < config.JP_COST_OF_CAPITAL_ROE)
    )

    per_valid = data["per"].where(data["per"] > 0)
    per_score = _safe_percentile(per_valid, higher_is_better=False)
    pbr_score = _safe_percentile(pbr_valid, higher_is_better=False)
    value_score = pd.concat([per_score, pbr_score], axis=1).mean(axis=1, skipna=True)

    debt_valid = data["debt_to_equity"].where(data["debt_to_equity"] >= 0)
    roe_score = _safe_percentile(data["roe"], higher_is_better=True)
    debt_score = _safe_percentile(debt_valid, higher_is_better=False)
    quality_score = roe_score.fillna(0) * 0.7 + debt_score.fillna(0) * 0.3

    div_score = _safe_percentile(data["dividend_yield"], higher_is_better=True)
    payout = data["payout_ratio"]
    penalty = pd.Series(1.0, index=data.index)
    penalty = penalty.mask(payout > 1.0, 0.5)
    penalty = penalty.mask((payout > 0.8) & (payout <= 1.0), 0.8)
    dividend_score = div_score * penalty

    rev_score = _safe_percentile(data["revenue_growth"], higher_is_better=True)
    earn_score = _safe_percentile(data["earnings_growth"], higher_is_better=True)
    growth_score = pd.concat([rev_score, earn_score], axis=1).mean(axis=1, skipna=True)

    data["score"] = scoring.weighted_composite(
        {"value": value_score, "quality": quality_score, "dividend": dividend_score, "growth": growth_score},
        {"value": 0.3, "quality": 0.3, "dividend": 0.3, "growth": 0.1},
    )

    data = data.reset_index()
    data["reason"] = [_build_reason(r) for r in data.to_dict("records")]
    data = data.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    return data[output_cols]
