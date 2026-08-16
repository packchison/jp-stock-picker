"""Stock universe definition and price-based filtering."""
from ._nikkei225_data import NIKKEI225


def to_ticker(code: str) -> str:
    """Convert a 4-char JPX securities code to a yfinance ticker symbol."""
    return f"{code}.T"


def get_universe() -> list[dict]:
    """Return the full Nikkei 225 universe as a list of {code, name, sector, ticker}."""
    return [
        {**s, "ticker": to_ticker(s["code"])}
        for s in NIKKEI225
    ]
