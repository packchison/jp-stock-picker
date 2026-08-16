"""Pure-pandas technical indicator functions operating on an OHLCV DataFrame.

Every function takes a DataFrame with columns [Open, High, Low, Close, Volume]
(as returned by yfinance) and returns a pandas Series aligned to the same index,
unless noted otherwise.
"""
import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def deviation_rate(close: pd.Series, window: int = 25) -> pd.Series:
    """% distance of price from its N-day moving average (乖離率)."""
    ma = sma(close, window)
    return (close - ma) / ma * 100


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger_percent_b(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = sma(close, window)
    std = close.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (close - lower) / (upper - lower).replace(0, np.nan)


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Today's volume vs its trailing N-day average."""
    avg = volume.rolling(window).mean()
    return volume / avg.replace(0, np.nan)


def avg_turnover_jpy(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    """Average daily trading value (JPY) over the trailing window — liquidity proxy."""
    turnover = (close * volume).rolling(window).mean()
    return float(turnover.iloc[-1]) if len(turnover.dropna()) else 0.0


def pct_off_52w_high(close: pd.Series) -> pd.Series:
    high_52w = close.rolling(252, min_periods=20).max()
    return (close - high_52w) / high_52w * 100


def golden_cross_recent(fast_ma: pd.Series, slow_ma: pd.Series, lookback: int = 5) -> bool:
    """True if fast MA crossed above slow MA within the last `lookback` bars."""
    diff = (fast_ma - slow_ma).dropna()
    if len(diff) < lookback + 1:
        return False
    recent = diff.iloc[-(lookback + 1):]
    crossed_up = (recent.shift(1) < 0) & (recent >= 0)
    return bool(crossed_up.any())


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Ichimoku Kinko Hyo components (tenkan/kijun/senkou spans A&B)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return pd.DataFrame({
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": senkou_a, "senkou_b": senkou_b,
    }, index=close.index)


def relative_strength_line(close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    """Stock price / benchmark index price, aligned on common dates."""
    aligned = pd.concat([close, benchmark_close], axis=1, join="inner")
    aligned.columns = ["stock", "bench"]
    return aligned["stock"] / aligned["bench"]


def trend_template_pass(close: pd.Series, ma75: pd.Series, ma200: pd.Series) -> bool:
    """Simplified Minervini-style trend template: price > MA75 > MA200, both rising."""
    if len(close.dropna()) < 210 or ma200.dropna().empty:
        return False
    price_ok = close.iloc[-1] > ma75.iloc[-1] > ma200.iloc[-1]
    ma200_rising = ma200.iloc[-1] > ma200.iloc[-21] if len(ma200.dropna()) > 21 else False
    return bool(price_ok and ma200_rising)
