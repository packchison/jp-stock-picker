"""CLI entry point for the JP stock screening system.

Usage (run from the C:\\Users\\hyuga\\Python\\main directory):
    python -m jp_stock_picker.main --horizon all
    python -m jp_stock_picker.main --horizon short --top 10
    python -m jp_stock_picker.main --refresh-cache
"""
import argparse
import sys
import time

from . import config, data_fetcher, report, tracker, universe
from .screeners import long_term, mid_term, short_term

# Windows consoles often default to a non-UTF-8 codepage (e.g. cp1252), which
# cannot encode Japanese output; force UTF-8 on stdout/stderr where supported.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

HORIZON_CHOICES = ["short", "mid", "long", "all"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="日本株(日経225)銘柄選定システム")
    p.add_argument("--horizon", choices=HORIZON_CHOICES, default="all",
                    help="対象の投資期間 (デフォルト: all)")
    p.add_argument("--top", type=int, default=config.DEFAULT_TOP_N,
                    help=f"各期間の表示件数 (デフォルト: {config.DEFAULT_TOP_N})")
    p.add_argument("--refresh-cache", action="store_true",
                    help="キャッシュを無視してデータを再取得する")
    p.add_argument("--no-save", action="store_true",
                    help="CSV/Markdownへの出力を行わない")
    p.add_argument("--no-track", action="store_true",
                    help="picks_history.csvへの記録を行わない(答え合わせ機能への蓄積をスキップ)")
    return p.parse_args(argv)


def resolve_horizons(choice: str) -> list[str]:
    return ["short", "mid", "long"] if choice == "all" else [choice]


def main(argv=None) -> int:
    args = parse_args(argv)
    horizons = resolve_horizons(args.horizon)

    if args.refresh_cache:
        print("キャッシュを削除しています...")
        data_fetcher.clear_cache()

    uni = universe.get_universe()
    tickers = [u["ticker"] for u in uni]
    all_tickers = tickers + [config.BENCHMARK_TICKER]

    print(f"株価データ取得中... ({len(all_tickers)}銘柄 + ベンチマーク)")
    t0 = time.time()
    price_history = data_fetcher.fetch_price_history(all_tickers)
    print(f"  -> {len(price_history)}銘柄分を取得 ({time.time() - t0:.1f}秒)")

    benchmark_history = price_history.get(config.BENCHMARK_TICKER)
    if benchmark_history is None:
        print(f"警告: ベンチマーク({config.BENCHMARK_TICKER})の取得に失敗しました。中期スクリーニングの精度が落ちます。", file=sys.stderr)

    fundamentals = {}
    if "mid" in horizons or "long" in horizons:
        price_filtered = [
            t for t in tickers
            if t in price_history
            and not price_history[t]["Close"].empty
            and price_history[t]["Close"].iloc[-1] <= config.MAX_PRICE_JPY
        ]
        print(f"ファンダメンタルズ取得中... ({len(price_filtered)}銘柄, 価格フィルタ後)")
        t0 = time.time()
        fundamentals = data_fetcher.fetch_fundamentals(price_filtered)
        print(f"  -> {len(fundamentals)}銘柄分を取得 ({time.time() - t0:.1f}秒)")

    results = {}
    if "short" in horizons:
        results["short"] = short_term.screen(uni, price_history, top_n=args.top)
    if "mid" in horizons:
        results["mid"] = mid_term.screen(uni, price_history, benchmark_history, fundamentals, top_n=args.top)
    if "long" in horizons:
        results["long"] = long_term.screen(uni, fundamentals, price_history, top_n=args.top)

    for horizon, df in results.items():
        report.print_table(df, horizon)

    if not args.no_save:
        paths = report.save_outputs(results)
        print("\n出力ファイル:")
        for p in paths:
            print(f"  {p}")

    if not args.no_track:
        total = tracker.append_picks(results)
        print(f"\n選定履歴に記録しました (累計{total}件, {tracker.HISTORY_PATH})")
        print("しばらく経ったら `python -m jp_stock_picker.evaluate` で答え合わせできます。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
