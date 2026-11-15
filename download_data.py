"""Seed the bundled index and Treasury warehouse, and print the link grid."""
from __future__ import annotations

import pandas as pd

from market_data import HISTORY_START, default_window_end, fetch_live_bundle, write_warehouse
from yield_engine import ERAS, INDICES, TENOR_LABELS, TENORS, changes, daily_panel, link_grid, slice_panel


def _print_grid(title: str, grid) -> None:
    print(f"\n{title}")
    for tenor in TENORS:
        bits = []
        block = grid[grid["tenor"] == tenor]
        for index in INDICES:
            row = block[block["index"] == index]
            if row.empty:
                bits.append(f"{index} n/a")
                continue
            rec = row.iloc[0]
            bits.append(
                f"{index} {rec['beta_per_10bp']:+.2f}% "
                f"ρ={rec['corr']:+.2f} R²={rec['r2']:.2f} t={rec['tstat']:+.1f} n={int(rec['n'])}"
            )
        print(f"  {TENOR_LABELS[tenor]}: " + " | ".join(bits))


def main() -> None:
    end = default_window_end()
    bundle = fetch_live_bundle(HISTORY_START, end)
    meta = write_warehouse(bundle, source="Yahoo chart + US Treasury daily par yield curve")
    panel = daily_panel(bundle["prices"], bundle["yields"])
    print(
        f"Prices {meta['price_start']} → {meta['price_end']} · "
        f"yields {meta['yield_start']} → {meta['yield_end']} · "
        f"aligned days {len(panel)}"
    )
    weekly = changes(panel, weekly=True)
    _print_grid(f"Full sample weekly ({weekly.index.min().date()} → {weekly.index.max().date()})", link_grid(weekly))

    last = panel.index.max()
    for label, start, era_end in ERAS:
        end_ts = last if era_end is None else min(pd.Timestamp(era_end), last)
        view = slice_panel(panel, start, end_ts)
        frame = changes(view, weekly=True)
        _print_grid(f"{label} weekly", link_grid(frame))


if __name__ == "__main__":
    main()
