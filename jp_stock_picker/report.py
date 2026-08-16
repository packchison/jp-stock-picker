"""Format screener results for console display and export to CSV/Markdown."""
from datetime import datetime

import pandas as pd

from . import config

HORIZON_LABELS = {
    "short": "短期 (~1週間) — テクニカル",
    "mid": "中期 (数週間~数ヶ月) — トレンド+ファンダ",
    "long": "長期 (1年以上) — バリュー+クオリティ+配当",
}


def print_table(df: pd.DataFrame, horizon: str) -> None:
    label = HORIZON_LABELS.get(horizon, horizon)
    print(f"\n{'=' * 70}")
    print(f" {label}  (該当 {len(df)} 銘柄, 価格 <= {config.MAX_PRICE_JPY:,}円)")
    print(f"{'=' * 70}")
    if df.empty:
        print(" 条件に合う銘柄が見つかりませんでした。")
        return

    display_cols = [c for c in df.columns if c != "reason"]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df[display_cols].to_string(index=False))
    print("\n--- 選定理由 ---")
    for _, row in df.iterrows():
        print(f" [{row['code']}] {row['name']}: {row['reason']}")


def save_outputs(results: dict[str, pd.DataFrame], stamp: str = None) -> list[str]:
    """Write each horizon's results to CSV + a combined Markdown summary. Returns file paths written."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []

    md_lines = [f"# 日本株スクリーニング結果 ({stamp})\n"]
    for horizon, df in results.items():
        csv_path = config.OUTPUT_DIR / f"{horizon}_{stamp}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        paths.append(str(csv_path))

        md_lines.append(f"\n## {HORIZON_LABELS.get(horizon, horizon)}\n")
        if df.empty:
            md_lines.append("該当銘柄なし\n")
            continue
        cols = [c for c in df.columns if c != "reason"]
        md_lines.append("| " + " | ".join(cols) + " |")
        md_lines.append("|" + "---|" * len(cols))
        for _, row in df.iterrows():
            md_lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        md_lines.append("\n**選定理由:**\n")
        for _, row in df.iterrows():
            md_lines.append(f"- [{row['code']}] {row['name']}: {row['reason']}")

    md_path = config.OUTPUT_DIR / f"summary_{stamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    paths.append(str(md_path))

    return paths
