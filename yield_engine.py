"""Index return vs Treasury yield-change link, by tenor.

Weekly (or daily) log return is regressed on the same-horizon change in a
constant-maturity yield. The object is the co-movement, not a trading rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

INDICES: tuple[str, ...] = ("SPX", "NDX", "DJI", "RUT")
INDEX_LABELS: dict[str, str] = {
    "SPX": "S&P 500",
    "NDX": "Nasdaq-100",
    "DJI": "Dow",
    "RUT": "Russell 2000",
}
TENORS: tuple[str, ...] = ("y3m", "y2", "y5", "y10", "y30")
TENOR_LABELS: dict[str, str] = {
    "y3m": "3-month",
    "y2": "2-year",
    "y5": "5-year",
    "y10": "10-year",
    "y30": "30-year",
}

# Named windows the article quotes. The last one runs through the sample end.
ERAS: tuple[tuple[str, str, str | None], ...] = (
    ("2010–2019", "2010-01-01", "2019-12-31"),
    ("2020–2021", "2020-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2022–2023", "2022-01-01", "2023-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", None),
)

MIN_OBS = 26


@dataclass(frozen=True)
class LinkFit:
    index: str
    tenor: str
    n: int
    beta_per_10bp: float
    corr: float
    r2: float
    tstat: float
    alpha_pct: float


def _wide_prices(prices: pd.DataFrame) -> pd.DataFrame:
    wide = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    missing = [name for name in INDICES if name not in wide.columns]
    if missing:
        raise ValueError(f"Price panel is missing {missing}.")
    return wide.loc[:, list(INDICES)]


def _wide_yields(yields: pd.DataFrame) -> pd.DataFrame:
    frame = yields.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.drop_duplicates("date").set_index("date").sort_index()
    missing = [name for name in TENORS if name not in frame.columns]
    if missing:
        raise ValueError(f"Yield panel is missing {missing}.")
    return frame.loc[:, list(TENORS)].apply(pd.to_numeric, errors="coerce")


def daily_panel(prices: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    """Inner join of index closes and yield levels. Rows with any gap are dropped."""
    px = _wide_prices(prices)
    yld = _wide_yields(yields)
    panel = px.join(yld, how="inner").dropna(how="any")
    if panel.empty:
        raise ValueError("No overlapping index and yield dates.")
    panel.index = pd.to_datetime(panel.index).tz_localize(None).normalize()
    return panel.sort_index()


def slice_panel(panel: pd.DataFrame, start, end) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    view = panel.loc[(panel.index >= start_ts) & (panel.index <= end_ts)]
    if view.empty:
        raise ValueError("No rows in that date window.")
    return view


def changes(panel: pd.DataFrame, *, weekly: bool) -> pd.DataFrame:
    """Log index return in percent, and yield change in basis points, same horizon."""
    levels = panel.resample("W-FRI").last().dropna(how="any") if weekly else panel
    out = pd.DataFrame(index=levels.index)
    for name in INDICES:
        out[f"{name}_ret"] = np.log(levels[name]).diff() * 100.0
    for name in TENORS:
        out[f"{name}_bp"] = levels[name].diff() * 100.0
    out = out.dropna(how="any")
    if out.empty:
        raise ValueError("Not enough observations to difference the window.")
    return out


def fit_link(frame: pd.DataFrame, index: str, tenor: str) -> LinkFit | None:
    """OLS of percent log return on the yield change in basis points."""
    y = frame[f"{index}_ret"].to_numpy(dtype=float)
    x = frame[f"{tenor}_bp"].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    n = int(y.size)
    if n < MIN_OBS:
        return None
    x_center = x - x.mean()
    y_center = y - y.mean()
    ss_x = float(x_center @ x_center)
    ss_y = float(y_center @ y_center)
    if ss_x <= 0 or ss_y <= 0:
        return None
    design = np.column_stack([np.ones(n), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    sigma2 = float(resid @ resid) / (n - 2)
    xtx_inv = np.linalg.inv(design.T @ design)
    se = float(np.sqrt(xtx_inv[1, 1] * sigma2))
    beta = float(coef[1])
    tstat = beta / se if se > 0 else float("nan")
    corr = float(x_center @ y_center) / np.sqrt(ss_x * ss_y)
    r2 = float(corr * corr)
    return LinkFit(
        index=index,
        tenor=tenor,
        n=n,
        beta_per_10bp=beta * 10.0,
        corr=corr,
        r2=r2,
        tstat=float(tstat),
        alpha_pct=float(coef[0]),
    )


def link_grid(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in INDICES:
        for tenor in TENORS:
            fit = fit_link(frame, index, tenor)
            if fit is None:
                continue
            rows.append(
                {
                    "index": fit.index,
                    "index_label": INDEX_LABELS[fit.index],
                    "tenor": fit.tenor,
                    "tenor_label": TENOR_LABELS[fit.tenor],
                    "n": fit.n,
                    "beta_per_10bp": fit.beta_per_10bp,
                    "corr": fit.corr,
                    "r2": fit.r2,
                    "tstat": fit.tstat,
                    "alpha_pct": fit.alpha_pct,
                }
            )
    if not rows:
        raise ValueError(f"Need at least {MIN_OBS} overlapping changes in the window.")
    return pd.DataFrame(rows)


def rolling_corr(frame: pd.DataFrame, index: str, tenor: str, window: int) -> pd.Series:
    if window < 8:
        raise ValueError("Rolling window must be at least 8 observations.")
    series = frame[f"{index}_ret"].rolling(window, min_periods=window).corr(frame[f"{tenor}_bp"])
    series.name = f"{index}_{tenor}"
    return series.dropna()


def rebase(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rebase each column to 100 at the first row of the slice."""
    block = panel.loc[:, columns].dropna(how="any")
    if block.empty:
        raise ValueError("Nothing to rebase in this window.")
    base = block.iloc[0]
    return block.divide(base).mul(100.0)


def latest_yields(panel: pd.DataFrame) -> pd.Series:
    return panel.iloc[-1].loc[list(TENORS)]


def yield_change(panel: pd.DataFrame) -> pd.Series:
    """End minus start, in percentage points, on the level panel."""
    return panel.iloc[-1].loc[list(TENORS)] - panel.iloc[0].loc[list(TENORS)]


def annual_grid(panel: pd.DataFrame, *, weekly: bool) -> pd.DataFrame:
    """One fit per calendar year. A full-sample beta blends years that do not agree."""
    frame = changes(panel, weekly=weekly)
    rows: list[pd.DataFrame] = []
    for year, block in frame.groupby(frame.index.year):
        try:
            grid = link_grid(block)
        except ValueError:
            continue
        grid = grid.copy()
        grid.insert(0, "year", int(year))
        rows.append(grid)
    if not rows:
        raise ValueError("No calendar year has enough observations.")
    return pd.concat(rows, ignore_index=True)


def week_split(frame: pd.DataFrame, tenor: str) -> pd.DataFrame:
    """Average index return in weeks the tenor rose, and in weeks it fell."""
    move = frame[f"{tenor}_bp"]
    sides = (("Yield up", frame.loc[move > 0]), ("Yield down", frame.loc[move < 0]))
    rows: list[dict[str, object]] = []
    for label, block in sides:
        if block.empty:
            continue
        for index in INDICES:
            rows.append(
                {
                    "side": label,
                    "index": index,
                    "index_label": INDEX_LABELS[index],
                    "n": int(len(block)),
                    "mean_bp": float(block[f"{tenor}_bp"].mean()),
                    "mean_ret": float(block[f"{index}_ret"].mean()),
                }
            )
    if not rows:
        raise ValueError("No weeks with a yield change in this window.")
    return pd.DataFrame(rows)


def study_verdict(grid: pd.DataFrame, tenor: str = "y10") -> tuple[str, str]:
    """Which index the selected tenor lines up with in this window, and how tightly."""
    label = TENOR_LABELS[tenor]
    ten = grid[grid["tenor"] == tenor].sort_values("beta_per_10bp")
    if len(ten) < 2:
        return "Window too short", "Widen the dates. A fit needs at least 26 changes."
    hard = ten.iloc[0]
    soft = ten.iloc[-1]
    gap = float(soft["beta_per_10bp"] - hard["beta_per_10bp"])
    hard_b = float(hard["beta_per_10bp"])
    soft_b = float(soft["beta_per_10bp"])
    if hard_b < 0 < soft_b and gap >= 0.25:
        title = (
            f"A higher {label} lines up with a weaker {hard['index_label']}, "
            f"not with the {soft['index_label']}"
        )
    elif hard_b > 0.15 and soft_b > 0.15:
        title = f"The {label} and the major indices moved together in this window"
    elif soft_b < -0.15:
        title = (
            f"A higher {label} lines up with lower indices, "
            f"and the {hard['index_label']} moves more than the {soft['index_label']}"
        )
    else:
        title = f"The {label} link is weak in this window"
    n = int(hard["n"])
    body = (
        f"A +10 bp move in the {label} lines up with {hard_b:+.2f}% "
        f"on the {hard['index_label']} and {soft_b:+.2f}% on the {soft['index_label']} "
        f"({n} observations). "
        f"Correlations are {float(hard['corr']):+.2f} and {float(soft['corr']):+.2f}. "
        f"That yield explains about {100 * float(hard['r2']):.0f}% of "
        f"{hard['index_label']} variation here, so most of the index is still something else."
    )
    return title, body
