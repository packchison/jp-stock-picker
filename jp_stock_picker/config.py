"""Central configuration for the JP stock screening system."""
from pathlib import Path

# --- User constraints ---
MAX_PRICE_JPY = 10_000  # only consider stocks trading at or below this per-share price

# --- Universe / data ---
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_TTL_HOURS = 20  # ~once/trading-day refresh

BENCHMARK_TICKER = "^N225"  # Nikkei 225 index, used for relative-strength calcs

# --- Fetching ---
MAX_WORKERS = 12  # thread pool size for per-ticker fundamentals/history fetches
HISTORY_PERIOD = "2y"  # daily bars needed for 200-day trend template etc.

# --- Liquidity floor (applies to all horizons) ---
# Nikkei 225 members are all reasonably liquid; this mainly screens out
# thinly-traded edge cases rather than being a binding constraint.
MIN_AVG_TURNOVER_JPY = 1e8  # ~100 million JPY/day average trading value

# --- Screener output ---
DEFAULT_TOP_N = 15

# --- Long-term fundamentals reference points (Japan-specific norms) ---
# TSE's own "cost of capital" benchmark used in its 2023 governance reform push.
JP_COST_OF_CAPITAL_ROE = 0.08
JP_STRONG_ROE = 0.12
JP_CHEAP_PER = 15.0
JP_DEEP_VALUE_PER = 10.0
JP_VALUE_PBR = 1.0
JP_HIGH_DIVIDEND_YIELD = 0.03  # 3%
