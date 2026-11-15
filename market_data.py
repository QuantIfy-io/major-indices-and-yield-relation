"""Yahoo index closes and FRED constant-maturity yields, with a parquet fallback."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
PRICES_PATH = DATA_DIR / "prices.parquet"
YIELDS_PATH = DATA_DIR / "yields.parquet"
METADATA_PATH = DATA_DIR / "warehouse_metadata.json"

YAHOO_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) QuantIfy/1.0"}
TREASURY_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) QuantIfy/1.0"}

# Treasury.gov column headers for the daily par yield curve. These are the H.15
# constant-maturity quotes FRED redistributes as DGS3MO, DGS2, DGS5, DGS10, DGS30.
TREASURY_COLUMNS: dict[str, str] = {
    "y3m": "3 Mo",
    "y2": "2 Yr",
    "y5": "5 Yr",
    "y10": "10 Yr",
    "y30": "30 Yr",
}

# Cash indices. Nasdaq-100, not the Composite: one growth book, not a second broad market.
YAHOO_SYMBOLS: dict[str, str] = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "DJI": "^DJI",
    "RUT": "^RUT",
}

# FRED H.15 constant-maturity Treasury yields, percent per annum.
FRED_SERIES: dict[str, str] = {
    "y3m": "DGS3MO",
    "y2": "DGS2",
    "y5": "DGS5",
    "y10": "DGS10",
    "y30": "DGS30",
}

HISTORY_START = "2010-01-01"


def default_window_end() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _yahoo_chart_url(symbol: str, start: str, end: str) -> str:
    start_ts = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(
        (datetime.fromisoformat(end) + timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp()
    )
    return (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(symbol, safe='')}?period1={start_ts}&period2={end_ts}&interval=1d"
    )


def fetch_yahoo_close(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Daily closes. Session dates are New York dates."""
    resp = requests.get(_yahoo_chart_url(symbol, start, end), headers=YAHOO_UA, timeout=45)
    resp.raise_for_status()
    payload = resp.json()
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        error = (payload.get("chart") or {}).get("error") or payload
        raise RuntimeError(f"Yahoo chart empty for {symbol}: {error}")
    block = results[0]
    stamps = block.get("timestamp") or []
    quote = ((block.get("indicators") or {}).get("quote") or [{}])[0]
    dates = (
        pd.to_datetime(stamps, unit="s", utc=True)
        .tz_convert("America/New_York")
        .tz_localize(None)
        .normalize()
    )
    frame = pd.DataFrame({"date": dates, "close": pd.to_numeric(quote.get("close"), errors="coerce")})
    frame = frame.dropna(subset=["close"])
    frame = frame[frame["close"] > 0]
    if frame.empty:
        raise RuntimeError(f"Yahoo chart had no closes for {symbol}.")
    return frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def _treasury_year_url(year: int) -> str:
    return (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        f"daily-treasury-rates.csv/{year}/all"
        f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
    )


def fetch_treasury_year(year: int) -> pd.DataFrame:
    resp = requests.get(_treasury_year_url(year), headers=TREASURY_UA, timeout=30)
    resp.raise_for_status()
    if not resp.text.strip():
        raise RuntimeError(f"Treasury yield curve for {year} was empty.")
    raw = pd.read_csv(StringIO(resp.text))
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [label for label in TREASURY_COLUMNS.values() if label not in raw.columns]
    if missing or "Date" not in raw.columns:
        found = ", ".join(raw.columns.astype(str))
        raise RuntimeError(f"Treasury {year} is missing {missing or 'Date'}. Columns: {found}")
    frame = pd.DataFrame({"date": pd.to_datetime(raw["Date"], format="%m/%d/%Y")})
    for name, label in TREASURY_COLUMNS.items():
        frame[name] = pd.to_numeric(raw[label], errors="coerce")
    return frame.dropna(subset=["date"]).drop_duplicates("date")


def fetch_prices(start: str, end: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for name, symbol in YAHOO_SYMBOLS.items():
        try:
            ohlc = fetch_yahoo_close(symbol, start, end)
            ohlc.insert(1, "symbol", name)
            frames.append(ohlc)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
    missing = sorted(set(YAHOO_SYMBOLS) - {frame["symbol"].iloc[0] for frame in frames})
    if missing:
        detail = "; ".join(errors) or "no series returned"
        raise RuntimeError(f"Could not load prices for {missing}. {detail}")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def fetch_yields(start: str) -> pd.DataFrame:
    """Daily par yields from Treasury.gov, one calendar year per request."""
    start_year = datetime.fromisoformat(start).year
    end_year = datetime.now(timezone.utc).year
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for year in range(start_year, end_year + 1):
        try:
            frames.append(fetch_treasury_year(year))
        except Exception as exc:
            errors.append(f"{year}: {exc}")
    if not frames:
        detail = "; ".join(errors) or "no years returned"
        raise RuntimeError(f"Could not load the Treasury yield curve. {detail}")
    yields = pd.concat(frames, ignore_index=True)
    yields["date"] = pd.to_datetime(yields["date"]).dt.normalize()
    yields = yields[yields["date"] >= pd.Timestamp(start)]
    yields = yields.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if yields.empty:
        raise RuntimeError("Treasury yield curve had no rows on or after the start date.")
    if errors:
        yields.attrs["errors"] = errors
    return yields


def fetch_live_bundle(start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    end = end or default_window_end()
    start = start or HISTORY_START
    return {"prices": fetch_prices(start, end), "yields": fetch_yields(start)}


def write_warehouse(bundle: dict[str, pd.DataFrame], *, source: str) -> dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices = bundle["prices"].copy()
    yields = bundle["yields"].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    yields["date"] = pd.to_datetime(yields["date"])
    prices.to_parquet(PRICES_PATH, index=False)
    yields.to_parquet(YIELDS_PATH, index=False)
    meta: dict[str, object] = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "price_start": pd.Timestamp(prices["date"].min()).strftime("%Y-%m-%d"),
        "price_end": pd.Timestamp(prices["date"].max()).strftime("%Y-%m-%d"),
        "yield_start": pd.Timestamp(yields["date"].min()).strftime("%Y-%m-%d"),
        "yield_end": pd.Timestamp(yields["date"].max()).strftime("%Y-%m-%d"),
        "n_prices": int(len(prices)),
        "n_yield_rows": int(len(yields)),
        "symbols": sorted(prices["symbol"].unique().tolist()),
        "tenors": [c for c in yields.columns if c != "date"],
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _normalize_dates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    out = frame.copy()
    out[column] = pd.to_datetime(out[column]).dt.tz_localize(None).dt.normalize()
    return out


def load_warehouse() -> dict[str, pd.DataFrame]:
    missing = [path.name for path in (PRICES_PATH, YIELDS_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"No bundled warehouse ({', '.join(missing)}). Run python3 download_data.py first."
        )
    return {
        "prices": _normalize_dates(pd.read_parquet(PRICES_PATH), "date"),
        "yields": _normalize_dates(pd.read_parquet(YIELDS_PATH), "date"),
    }


def load_warehouse_metadata() -> dict[str, object]:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_bundle(*, refresh: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Prefer the bundled parquet. A live refresh falls back to it if the network fails."""
    bundled_meta = load_warehouse_metadata()
    if refresh:
        try:
            live = fetch_live_bundle()
            meta = {
                **bundled_meta,
                "source": "live Yahoo + FRED",
                "price_end": pd.Timestamp(live["prices"]["date"].max()).strftime("%Y-%m-%d"),
                "yield_end": pd.Timestamp(live["yields"]["date"].max()).strftime("%Y-%m-%d"),
                "live": True,
            }
            return live, meta
        except Exception as exc:
            if PRICES_PATH.exists() and YIELDS_PATH.exists():
                return load_warehouse(), {**bundled_meta, "live": False, "live_error": str(exc)}
            raise
    return load_warehouse(), {**bundled_meta, "live": False}
