"""Grade past screener picks against what actually happened to their price.

Usage (run from the C:\\Users\\hyuga\\Python\\main directory):
    python -m jp_stock_picker.evaluate
    python -m jp_stock_picker.evaluate --horizon short --min-days 5

Reads jp_stock_picker/output/picks_history.csv (written by main.py on every
run), fetches each ticker's current price plus the Nikkei 225 benchmark, and
reports the realized return since the pick date — both raw and relative to
the benchmark over the same window (excess return), since a pick that merely
tracked a rising market isn't evidence the screener added value.
"""
import argparse
import sys
from datetime import date, datetime

import pandas as pd

from . import config, data_fetcher, tracker, universe

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Rough "by when should this pick have played out" reference points, used only
# to label whether a pick's typical holding window has fully elapsed yet —
# picks are still shown (and worth looking at) before that point.
HORIZON_TARGET_DAYS = {"short": 7, "mid": 60, "long": 365}


def _benchmark_price_on_or_before(bench_close: pd.Series, target_date: pd.Timestamp):
    idx = bench_close.index
    naive_idx = idx.tz_localize(None) if idx.tz is not None else idx
    mask = naive_idx <= target_date
    if not mask.any():
        return None
    pos = mask.nonzero()[0][-1]
    return float(bench_close.iloc[pos])


def build_evaluation(min_days: int = 0) -> pd.DataFrame:
    history = tracker.load_history()
    if history.empty:
        return history

    history = history.copy()
    history["run_date"] = pd.to_datetime(history["run_date"])
    today = pd.Timestamp(date.today())
    history["elapsed_days"] = (today - history["run_date"]).dt.days
    history = history[history["elapsed_days"] >= min_days]
    if history.empty:
        return history

    codes = sorted(history["code"].unique().tolist())
    tickers = [universe.to_ticker(c) for c in codes]
    price_history = data_fetcher.fetch_price_history(tickers + [config.BENCHMARK_TICKER])

    bench_df = price_history.get(config.BENCHMARK_TICKER)
    bench_close = bench_df["Close"] if bench_df is not None and len(bench_df) else None
    bench_now = float(bench_close.iloc[-1]) if bench_close is not None else None

    current_price = {}
    for code, ticker in zip(codes, tickers):
        df = price_history.get(ticker)
        if df is not None and len(df) and pd.notna(df["Close"].iloc[-1]):
            current_price[code] = float(df["Close"].iloc[-1])

    rows = []
    for _, r in history.iterrows():
        code = r["code"]
        cur = current_price.get(code)
        if cur is None:
            continue

        ret_pct = (cur / r["price_at_pick"] - 1) * 100

        bench_ret_pct = None
        if bench_close is not None and bench_now is not None:
            bench_then = _benchmark_price_on_or_before(bench_close, r["run_date"])
            if bench_then:
                bench_ret_pct = (bench_now / bench_then - 1) * 100

        excess_pct = (ret_pct - bench_ret_pct) if bench_ret_pct is not None else None
        target_days = HORIZON_TARGET_DAYS.get(r["horizon"], 0)

        rows.append({
            "run_date": r["run_date"].date().isoformat(),
            "horizon": r["horizon"],
            "code": code,
            "name": r["name"],
            "price_at_pick": r["price_at_pick"],
            "current_price": cur,
            "elapsed_days": int(r["elapsed_days"]),
            "target_days": target_days,
            "window_complete": bool(r["elapsed_days"] >= target_days),
            "return_pct": round(ret_pct, 2),
            "nikkei_return_pct": round(bench_ret_pct, 2) if bench_ret_pct is not None else None,
            "excess_return_pct": round(excess_pct, 2) if excess_pct is not None else None,
            "score_at_pick": r["score"],
        })

    return pd.DataFrame(rows)


def summarize(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return eval_df
    grouped = eval_df.groupby("horizon")
    summary = grouped.agg(
        n_picks=("code", "count"),
        avg_return_pct=("return_pct", "mean"),
        median_return_pct=("return_pct", "median"),
        win_rate_pct=("return_pct", lambda s: round((s > 0).mean() * 100, 1)),
        avg_excess_vs_nikkei_pct=("excess_return_pct", "mean"),
        n_window_complete=("window_complete", "sum"),
    ).round(2)
    return summary.reset_index()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="過去の選定結果を実際の値動きで答え合わせする")
    p.add_argument("--horizon", choices=["short", "mid", "long", "all"], default="all",
                    help="評価対象の投資期間 (デフォルト: all)")
    p.add_argument("--min-days", type=int, default=1,
                    help="この日数以上経過した選定のみ評価する (デフォルト: 1)")
    p.add_argument("--no-save", action="store_true", help="CSVへの出力を行わない")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    eval_df = build_evaluation(min_days=args.min_days)
    if eval_df.empty:
        print("評価できる履歴がまだありません。")
        print("`python -m jp_stock_picker.main` を数回(数週間)実行して選定結果を蓄積してから、")
        print("もう一度 `python -m jp_stock_picker.evaluate` を実行してください。")
        return 0

    if args.horizon != "all":
        eval_df = eval_df[eval_df["horizon"] == args.horizon].reset_index(drop=True)

    if eval_df.empty:
        print(f"指定期間(--horizon {args.horizon})の評価対象がありません。")
        return 0

    print(f"評価対象: {len(eval_df)}件 (経過日数 >= {args.min_days}日)")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(eval_df.sort_values(["horizon", "run_date", "code"]).to_string(index=False))

    print("\n--- 期間別サマリー ---")
    summary = summarize(eval_df)
    print(summary.to_string(index=False))
    print(
        "\n※ window_complete=False の銘柄は、その期間の想定保有期間"
        "(短期7日/中期60日/長期365日)がまだ経過していない途中経過です。"
    )
    print("※ excess_return_pct は日経平均の同期間リターンとの差分(超過リターン)です。")

    if not args.no_save:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = config.OUTPUT_DIR / f"evaluation_{stamp}.csv"
        eval_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n保存: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
