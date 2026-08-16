"""Short-term (~1 week horizon) technical screener for JP equities.

At this horizon returns are dominated by price/volume mechanics rather than
fundamentals, so stocks are ranked on three factors: trend/momentum health
(golden cross, MACD, RSI sweet-spot, moderate 25-day deviation), volume
confirmation of the latest move, and breakout/reversal positioning within
the 52-week range and Bollinger Bands.
"""
import pandas as pd

from .. import config, indicators, scoring


def _clip100(x: float) -> float:
    return max(0.0, min(100.0, x))


def _build_reason(r: dict) -> str:
    reasons = []
    if r["golden_cross"]:
        reasons.append("ゴールデンクロス直後")
    if r["macd_cross"]:
        reasons.append("MACDが強気転換")
    if r["volume_ratio"] >= 2.0 and r["up_day"]:
        reasons.append(f"出来高急増({r['volume_ratio']:.1f}倍)")
    elif r["volume_ratio"] >= 1.5 and r["up_day"]:
        reasons.append(f"出来高増加({r['volume_ratio']:.1f}倍)")
    if r["pctb"] >= 0.9:
        reasons.append("ボリンジャーバンド上限で推移しブレイクアウト継続")
    elif r["pctb"] <= 0.1 and r["turning_up"]:
        reasons.append("ボリンジャーバンド下限からの反発サイン")
    if r["pct_off_52w_high"] >= -3:
        reasons.append("52週高値圏で推移")
    if 50 <= r["rsi"] <= 70:
        reasons.append(f"RSI{r['rsi']:.0f}で健全な上昇トレンド")
    elif r["rsi"] > 80:
        reasons.append(f"RSI{r['rsi']:.0f}で過熱感はあるが勢い強い")
    elif r["rsi"] < 20 and r["turning_up"]:
        reasons.append(f"RSI{r['rsi']:.0f}の売られ過ぎからの反発")
    if 2 <= r["dev_rate_25"] <= 8:
        reasons.append(f"25日線から+{r['dev_rate_25']:.1f}%の適度な乖離で上昇継続")
    elif r["dev_rate_25"] <= -8 and r["turning_up"]:
        reasons.append("急落後の自律反発局面")

    if not reasons:
        reasons.append(f"総合スコア{r['score']:.0f}点、モメンタムと出来高のバランスが良好")

    return "、".join(reasons[:2])


def screen(universe: list[dict], price_history: dict, top_n: int = None) -> pd.DataFrame:
    top_n = top_n or config.DEFAULT_TOP_N

    out_cols = [
        "code", "name", "sector", "price", "score",
        "rsi", "macd_hist", "dev_rate_25", "volume_ratio", "golden_cross", "reason",
    ]

    rows = []
    for stock in universe:
        ticker = stock["ticker"]
        df = price_history.get(ticker)
        if df is None or len(df) < 30:
            continue

        close, volume = df["Close"], df["Volume"]
        latest_price = float(close.iloc[-1])
        if pd.isna(latest_price) or latest_price > config.MAX_PRICE_JPY:
            continue

        turnover = indicators.avg_turnover_jpy(close, volume)
        if turnover < config.MIN_AVG_TURNOVER_JPY:
            continue

        sma5 = indicators.sma(close, 5)
        sma25 = indicators.sma(close, 25)
        dev25 = indicators.deviation_rate(close, 25)
        rsi14 = indicators.rsi(close, 14)
        macd_df = indicators.macd(close)
        pctb = indicators.bollinger_percent_b(close, 20)
        volratio = indicators.volume_ratio(volume, 20)
        off_high = indicators.pct_off_52w_high(close)

        rsi_val = float(rsi14.iloc[-1])
        macd_hist = float(macd_df["hist"].iloc[-1])
        dev_val = float(dev25.iloc[-1]) if pd.notna(dev25.iloc[-1]) else 0.0
        pctb_val = float(pctb.iloc[-1]) if pd.notna(pctb.iloc[-1]) else 0.5
        volratio_val = float(volratio.iloc[-1]) if pd.notna(volratio.iloc[-1]) else 1.0
        off_high_val = float(off_high.iloc[-1]) if pd.notna(off_high.iloc[-1]) else -50.0
        gcross = indicators.golden_cross_recent(sma5, sma25, lookback=5)
        macd_cross = indicators.golden_cross_recent(macd_df["macd"], macd_df["signal"], lookback=5)

        up_day = len(close) > 1 and close.iloc[-1] >= close.iloc[-2]
        turning_up = len(close) > 4 and close.iloc[-1] > close.iloc[-4]

        # --- raw, monotonic ("higher is better") feature transforms, to be
        # percentile-ranked across the filtered universe below ---
        trend_cross_raw = (1.0 if gcross else 0.0) + (0.5 if macd_cross else 0.0)
        macd_pct_raw = (macd_hist / latest_price) * 100 if latest_price else 0.0
        rsi_bell_raw = _clip100(100 - abs(rsi_val - 60) * 2)  # peaks at RSI 60, decays both ways
        dev_bell_raw = _clip100(100 - abs(dev_val - 4) * 4)  # peaks near +4% (healthy continuation)

        vol_raw = volratio_val if up_day else volratio_val * 0.3  # dampen high volume on down days

        if pctb_val >= 0.7:
            bb_raw = pctb_val * 100  # riding the upper band = breakout strength
        elif pctb_val <= 0.3 and turning_up:
            bb_raw = (1 - pctb_val) * 100 * 0.8  # lower band + bounce = reversal setup
        elif pctb_val <= 0.3:
            bb_raw = (1 - pctb_val) * 100 * 0.3  # lower band, no bounce yet = low confidence
        else:
            bb_raw = 50.0  # mid-band, no positional edge

        rows.append({
            "code": stock["code"], "name": stock["name"], "sector": stock["sector"],
            "ticker": ticker, "price": round(latest_price, 1),
            "rsi": round(rsi_val, 1), "macd_hist": round(macd_hist, 2),
            "dev_rate_25": round(dev_val, 2), "volume_ratio": round(volratio_val, 2),
            "golden_cross": bool(gcross), "macd_cross": bool(macd_cross),
            "pct_off_52w_high": round(off_high_val, 2), "pctb": round(pctb_val, 3),
            "turning_up": bool(turning_up), "up_day": bool(up_day),
            "trend_cross_raw": trend_cross_raw, "macd_pct_raw": macd_pct_raw,
            "rsi_bell_raw": rsi_bell_raw, "dev_bell_raw": dev_bell_raw,
            "vol_raw": vol_raw, "bb_raw": bb_raw, "off_high_raw": off_high_val,
        })

    if not rows:
        return pd.DataFrame(columns=out_cols)

    data = pd.DataFrame(rows).set_index("ticker")

    momentum_trend = pd.concat([
        scoring.percentile_score(data["trend_cross_raw"], higher_is_better=True),
        scoring.percentile_score(data["macd_pct_raw"], higher_is_better=True),
        scoring.percentile_score(data["rsi_bell_raw"], higher_is_better=True),
        scoring.percentile_score(data["dev_bell_raw"], higher_is_better=True),
    ], axis=1).mean(axis=1)

    volume_confirmation = scoring.percentile_score(data["vol_raw"], higher_is_better=True)

    breakout_position = pd.concat([
        scoring.percentile_score(data["off_high_raw"], higher_is_better=True),
        scoring.percentile_score(data["bb_raw"], higher_is_better=True),
    ], axis=1).mean(axis=1)

    factor_scores = {
        "momentum_trend": momentum_trend,
        "volume_confirmation": volume_confirmation,
        "breakout_position": breakout_position,
    }
    weights = {"momentum_trend": 0.40, "volume_confirmation": 0.35, "breakout_position": 0.25}
    data["score"] = scoring.weighted_composite(factor_scores, weights)

    data["reason"] = [_build_reason(r) for r in data.to_dict("records")]

    out = data[out_cols].sort_values("score", ascending=False).reset_index(drop=True)
    return out.head(top_n)
