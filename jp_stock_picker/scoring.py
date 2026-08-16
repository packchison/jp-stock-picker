"""Shared scoring/ranking helpers used by all screeners.

Each screener computes several raw factors per stock, then uses these helpers
to normalize them onto a comparable 0-100 scale and combine them into one
composite score with configurable weights.
"""
import pandas as pd


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Rank a factor across the universe into a 0-100 percentile score."""
    ranks = series.rank(pct=True, na_option="bottom")
    if not higher_is_better:
        ranks = 1 - ranks
    return (ranks * 100).round(1)


def weighted_composite(scores: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Combine several 0-100 factor-score Series into one weighted composite."""
    total_weight = sum(weights.values())
    combined = None
    for key, series in scores.items():
        w = weights.get(key, 0) / total_weight
        term = series.fillna(0) * w
        combined = term if combined is None else combined + term
    return combined.round(1)
