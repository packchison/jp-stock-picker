"""Persist every screening run's picks so they can be graded later by evaluate.py."""
from datetime import date

import pandas as pd

from . import config

HISTORY_PATH = config.OUTPUT_DIR / "picks_history.csv"
HISTORY_COLUMNS = ["run_date", "horizon", "code", "name", "sector", "price_at_pick", "score", "reason"]


def append_picks(results: dict[str, pd.DataFrame], run_date: str = None) -> int:
    """Append this run's picks to the persistent history CSV.

    Re-running on the same day for the same horizon/code overwrites the prior
    same-day row rather than duplicating it (keeps the history one row per
    run_date+horizon+code even across --refresh-cache re-runs).

    Returns the number of rows now in the combined history.
    """
    run_date = run_date or date.today().isoformat()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for horizon, df in results.items():
        for _, r in df.iterrows():
            rows.append({
                "run_date": run_date, "horizon": horizon, "code": r["code"], "name": r["name"],
                "sector": r.get("sector", ""), "price_at_pick": r["price"],
                "score": r["score"], "reason": r["reason"],
            })

    if not rows:
        return _row_count()

    new_df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    if HISTORY_PATH.exists():
        existing = pd.read_csv(HISTORY_PATH, dtype={"code": str})
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["run_date", "horizon", "code"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(["run_date", "horizon", "code"]).reset_index(drop=True)
    combined.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
    return len(combined)


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.read_csv(HISTORY_PATH, dtype={"code": str})


def _row_count() -> int:
    return len(load_history())
