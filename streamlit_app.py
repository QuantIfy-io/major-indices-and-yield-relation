"""Streamlit: major US indices against Treasury yields, by tenor."""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_data import FRED_SERIES, load_bundle
from yield_engine import (
    INDEX_LABELS,
    INDICES,
    TENOR_LABELS,
    TENORS,
    annual_grid,
    changes,
    daily_panel,
    latest_yields,
    link_grid,
    rebase,
    rolling_corr,
    slice_panel,
    study_verdict,
    week_split,
    yield_change,
)

st.set_page_config(
    page_title="Indices and Yields",
    page_icon=":material/monitoring:",
    layout="wide",
)

AMBER = "#e07a3d"
BLUE = "#4c8bf5"
TEAL = "#38a8a4"
RED = "#d45d79"
GOLD = "#c9a227"
PURPLE = "#9b7ed9"
MUTED = "#8a93a3"

INDEX_COLORS = {"SPX": AMBER, "NDX": BLUE, "DJI": TEAL, "RUT": GOLD}
TENOR_COLORS = {"y3m": MUTED, "y2": RED, "y5": PURPLE, "y10": AMBER, "y30": BLUE}

PLOTLY_LEGEND = dict(
    orientation="h",
    yanchor="top",
    y=-0.28,
    xanchor="center",
    x=0.5,
    bgcolor="rgba(0,0,0,0)",
)


def _style(
    fig: go.Figure,
    *,
    title: str,
    height: int,
    ytitle: str,
    xtitle: str = "",
    showlegend: bool = True,
    hovermode: str = "x unified",
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15)),
        showlegend=showlegend,
        legend=PLOTLY_LEGEND if showlegend else dict(),
        margin=dict(l=56, r=16, t=52, b=110 if showlegend else 48),
        height=height,
        hovermode=hovermode,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        yaxis_title=ytitle,
        xaxis_title=xtitle,
        barmode="group",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=True, zerolinecolor="rgba(128,128,128,0.45)")
    return fig


def _formulas() -> None:
    with st.expander("The fit", expanded=False):
        st.markdown(
            "Each index is a weekly (or daily) log return, in percent. "
            "Each yield change is the move in that constant-maturity par yield, in basis points, "
            "over the same horizon. The slope is an ordinary least-squares fit with an intercept. "
            "It is a co-movement, not a hedge ratio and not a forecast."
        )
        st.latex(r"r_{i,t} = 100 \times \ln(P_{i,t} / P_{i,t-1})")
        st.latex(r"\Delta y_{k,t} = 100 \times (y_{k,t} - y_{k,t-1})")
        st.latex(r"r_{i,t} = \alpha_{i,k} + \beta_{i,k}\,\Delta y_{k,t} + \varepsilon_{i,t}")
        st.markdown(
            r"The bars are $10 \times \beta_{i,k}$: the percent index move lined up with a **+10 bp** rise "
            r"in yield $k$. A positive bar means the index rose in weeks that yield rose. "
            r"The $t$-statistic on the chart table is classical OLS, not autocorrelation-robust."
        )


def _line_chart(
    frame: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    *,
    title: str,
    ytitle: str,
) -> go.Figure:
    fig = go.Figure()
    for column in columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                name=labels[column],
                mode="lines",
                line=dict(color=colors[column], width=2),
                hovertemplate="%{y:.2f}<extra>" + labels[column] + "</extra>",
            )
        )
    return _style(fig, title=title, height=420, ytitle=ytitle, xtitle="Date")


def _beta_bars(grid: pd.DataFrame, *, x_col: str, x_order: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for index in INDICES:
        block = grid[grid["index"] == index].set_index(x_col).reindex(x_order)
        custom = block[["corr", "r2", "tstat", "n"]].to_numpy() if "corr" in block.columns else None
        fig.add_trace(
            go.Bar(
                x=[str(v) for v in x_order],
                y=block["beta_per_10bp"],
                name=INDEX_LABELS[index],
                marker_color=INDEX_COLORS[index],
                customdata=custom,
                hovertemplate=(
                    "%{x}<br>%{y:+.2f}% per 10 bp"
                    "<br>corr %{customdata[0]:+.2f}"
                    "<br>R-squared %{customdata[1]:.2f}"
                    "<br>t %{customdata[2]:+.1f}"
                    "<br>n %{customdata[3]:.0f}<extra>" + INDEX_LABELS[index] + "</extra>"
                ),
            )
        )
    return _style(
        fig,
        title=title,
        height=460,
        ytitle="Percent per 10 bp",
        hovermode="x unified",
    )


@st.cache_data(ttl=3600, show_spinner="Loading index closes and Treasury yields…")
def _load(refresh: bool):
    bundle, meta = load_bundle(refresh=refresh)
    panel = daily_panel(bundle["prices"], bundle["yields"])
    return panel, meta


st.title("Indices and yields")
st.markdown(
    "S&P 500, Nasdaq-100, Dow, and Russell 2000 against the Treasury par curve: "
    "3-month, 2-year, 5-year, 10-year, and 30-year. "
    "The question is which index moves with which tenor, and whether that link keeps its sign."
)
_formulas()

refresh = st.toggle("Refresh live Yahoo and Treasury", value=False)
try:
    panel, meta = _load(refresh)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if meta.get("live_error"):
    st.caption(
        "Live refresh was blocked. Using the bundled warehouse "
        f"through **{meta.get('price_end', 'the last saved session')}**."
    )

d_min = pd.Timestamp(panel.index.min()).date()
d_max = pd.Timestamp(panel.index.max()).date()

PRESETS: dict[str, tuple[date, date]] = {
    "2026": (date(2026, 1, 1), d_max),
    "2025": (date(2025, 1, 1), date(2025, 12, 31)),
    "2022–2023": (date(2022, 1, 1), date(2023, 12, 31)),
    "2010–2019": (date(2010, 1, 1), date(2019, 12, 31)),
    "Full sample": (d_min, d_max),
}


def _clamp(day: date) -> date:
    return min(max(day, d_min), d_max)


def _apply_preset() -> None:
    label = st.session_state.preset
    if label not in PRESETS:
        return
    start, end = PRESETS[label]
    st.session_state.date_from = _clamp(start)
    st.session_state.date_to = _clamp(end)


def _mark_custom() -> None:
    st.session_state.preset = "Custom"


if "date_from" not in st.session_state:
    st.session_state.preset = "2026"
    st.session_state.date_from = _clamp(date(2026, 1, 1))
    st.session_state.date_to = d_max

c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
c1.selectbox(
    "Window",
    list(PRESETS) + ["Custom"],
    key="preset",
    on_change=_apply_preset,
)
c2.date_input(
    "Date from",
    min_value=d_min,
    max_value=d_max,
    key="date_from",
    on_change=_mark_custom,
)
c3.date_input(
    "Date to",
    min_value=d_min,
    max_value=d_max,
    key="date_to",
    on_change=_mark_custom,
)
horizon = c4.radio("Horizon", ["Weekly", "Daily"], horizontal=True, index=0)
weekly = horizon == "Weekly"

tenor_label = st.selectbox("Tenor", [TENOR_LABELS[name] for name in TENORS], index=3)
tenor = next(name for name, label in TENOR_LABELS.items() if label == tenor_label)
roll_window = st.selectbox(
    "Rolling window",
    [13, 26, 52],
    index=1,
    format_func=lambda n: f"{n} {'weeks' if weekly else 'sessions'}",
)

if st.session_state.date_from > st.session_state.date_to:
    st.error("Date from must be on or before date to.")
    st.stop()

try:
    view = slice_panel(panel, st.session_state.date_from, st.session_state.date_to)
    frame = changes(view, weekly=weekly)
    grid = link_grid(frame)
    split = week_split(frame, tenor)
    years = annual_grid(panel, weekly=weekly)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

verdict_title, verdict_body = study_verdict(grid, tenor)
source = "live Yahoo + Treasury" if meta.get("live") else meta.get("source", "bundled")
st.caption(
    f"{source} · **{view.index.min().date()}** → **{view.index.max().date()}** · "
    f"{len(frame)} {horizon.lower()} changes · fit needs 26"
)
st.info(f"**{verdict_title}.** {verdict_body}")

levels = latest_yields(view)
change_pp = yield_change(view)
metric_row = st.columns(5)
for column, name in zip(metric_row, TENORS):
    column.metric(
        f"Yield, {TENOR_LABELS[name]}",
        f"{float(levels[name]):.2f}%",
        f"{float(change_pp[name]):+.2f} pp in window",
        delta_color="off",
    )

focus = grid[grid["tenor"] == tenor].set_index("index")
st.caption(f"Percent return lined up with a +10 bp rise in the {tenor_label}. The line under each figure is the correlation.")
beta_row = st.columns(4)
for column, index in zip(beta_row, INDICES):
    if index not in focus.index:
        column.metric(INDEX_LABELS[index], "—")
        continue
    rec = focus.loc[index]
    column.metric(INDEX_LABELS[index], f"{float(rec['beta_per_10bp']):+.2f}%")
    column.caption(f"corr {float(rec['corr']):+.2f}")

st.markdown(f"**When the {tenor_label} rose, and when it fell**")
if weekly:
    st.caption(
        "Mean weekly log return, in percent. "
        "Weeks with no yield change are in neither group. Same return the fit uses."
    )
else:
    st.caption(
        "Mean daily log return, in percent. "
        "Days with no yield change are in neither group. Same return the fit uses."
    )
split_fig = go.Figure()
for side, color in (("Yield up", RED), ("Yield down", TEAL)):
    block = split[split["side"] == side].set_index("index").reindex(list(INDICES))
    split_fig.add_trace(
        go.Bar(
            x=[INDEX_LABELS[index] for index in INDICES],
            y=block["mean_ret"],
            name=side,
            marker_color=color,
            customdata=block[["n", "mean_bp"]].to_numpy(),
            hovertemplate=(
                "%{x}<br>%{y:+.2f}%"
                "<br>n %{customdata[0]:.0f}"
                "<br>avg yield move %{customdata[1]:+.1f} bp<extra>" + side + "</extra>"
            ),
        )
    )
st.plotly_chart(
    _style(
        split_fig,
        title=f"Average index return on {tenor_label} up-moves and down-moves",
        height=440,
        ytitle="Average return, percent",
        hovermode="x unified",
    ),
    use_container_width=True,
)
up_n = int(split.loc[split["side"] == "Yield up", "n"].iloc[0]) if (split["side"] == "Yield up").any() else 0
down_n = int(split.loc[split["side"] == "Yield down", "n"].iloc[0]) if (split["side"] == "Yield down").any() else 0
st.caption(f"{tenor_label} rose in **{up_n}** periods and fell in **{down_n}**.")

st.markdown("**Every year on its own**")
st.caption(
    "Full file, not the date filter. Each bar is that calendar year's fit on the selected tenor. "
    "A single beta over 2010–2026 blends the years above zero with the years below it."
)
year_block = years[years["tenor"] == tenor]
year_order = [str(year) for year in sorted(year_block["year"].unique())]
year_plot = year_block.copy()
year_plot["year"] = year_plot["year"].astype(str)
st.plotly_chart(
    _beta_bars(
        year_plot,
        x_col="year",
        x_order=year_order,
        title=f"Percent return per 10 bp rise in the {tenor_label}, by year",
    ),
    use_container_width=True,
)

st.markdown("**This window, every tenor**")
st.caption("Same dates and horizon as the metrics above. One slope per index and tenor.")
st.plotly_chart(
    _beta_bars(
        grid,
        x_col="tenor_label",
        x_order=[TENOR_LABELS[name] for name in TENORS],
        title="Percent return per 10 bp, by tenor",
    ),
    use_container_width=True,
)

left, right = st.columns(2)
equity = rebase(view, list(INDICES))
with left:
    st.plotly_chart(
        _line_chart(
            equity,
            list(INDICES),
            INDEX_LABELS,
            INDEX_COLORS,
            title="Indices, rebased to 100",
            ytitle="Level, start = 100",
        ),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        _line_chart(
            view,
            list(TENORS),
            TENOR_LABELS,
            TENOR_COLORS,
            title="Treasury par yield",
            ytitle="Yield, percent",
        ),
        use_container_width=True,
    )

st.markdown("**Rolling correlation**")
st.caption("Each chart has its own legend. Correlation is between the index return and that tenor's yield change.")
roll_index = st.selectbox("Index", [INDEX_LABELS[name] for name in INDICES], index=3)
index_code = next(name for name, label in INDEX_LABELS.items() if label == roll_index)
tenor_series = [rolling_corr(frame, index_code, name, int(roll_window)) for name in TENORS]
if all(series.empty for series in tenor_series):
    st.caption(
        f"This window has {len(frame)} changes. "
        f"The rolling window is {int(roll_window)}, so there is no correlation path yet."
    )
roll_by_tenor = go.Figure()
for name, series in zip(TENORS, tenor_series):
    roll_by_tenor.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            name=TENOR_LABELS[name],
            mode="lines",
            line=dict(color=TENOR_COLORS[name], width=2),
        )
    )
st.plotly_chart(
    _style(
        roll_by_tenor,
        title=f"{roll_index} correlation with each tenor",
        height=420,
        ytitle="Correlation",
        xtitle="Date",
    ),
    use_container_width=True,
)

roll_by_index = go.Figure()
for name in INDICES:
    series = rolling_corr(frame, name, tenor, int(roll_window))
    roll_by_index.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            name=INDEX_LABELS[name],
            mode="lines",
            line=dict(color=INDEX_COLORS[name], width=2),
        )
    )
st.plotly_chart(
    _style(
        roll_by_index,
        title=f"Each index against the {tenor_label}",
        height=420,
        ytitle="Correlation",
        xtitle="Date",
    ),
    use_container_width=True,
)

st.markdown("**Fit table**")
table = grid.loc[:, ["index_label", "tenor_label", "beta_per_10bp", "corr", "r2", "tstat", "n"]].copy()
table.columns = ["Index", "Tenor", "Per 10 bp", "Correlation", "R-squared", "t-stat", "N"]
table = table.sort_values(["Tenor", "Index"]).reset_index(drop=True)
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Per 10 bp": st.column_config.NumberColumn(format="%+.2f"),
        "Correlation": st.column_config.NumberColumn(format="%+.2f"),
        "R-squared": st.column_config.NumberColumn(format="%.2f"),
        "t-stat": st.column_config.NumberColumn(format="%+.1f"),
        "N": st.column_config.NumberColumn(format="%d"),
    },
)

fred_ids = ", ".join(FRED_SERIES[name] for name in TENORS)
st.caption(
    "Index closes: Yahoo Finance chart, ^GSPC, ^NDX, ^DJI, ^RUT. "
    "Yields: US Treasury daily par yield curve "
    f"(the H.15 quotes FRED publishes as {fred_ids}). "
    "A week is the last print on or before Friday. "
    "Days missing any series are dropped before the change is taken. "
    f"Bundled through {meta.get('price_end', d_max)}."
)
