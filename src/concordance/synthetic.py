"""Synthetic CRSP-shaped markets with the awkward cases engines must agree on: halts,
delistings (some to zero), splits, dividends, missing opens, inconsistent or missing
high/low, zero or missing volume. Used by the tests, the demo and the strategy validator.
"""

import numpy as np
import pandas as pd

from concordance.calendar import calendar


def random_long_frame(seed: int, n_assets: int | None = None, n_sessions: int | None = None,
                      halt_rate: float = 0.06, missing_open_rate: float = 0.03):
    """`halt_rate` / `missing_open_rate`: chance per day of no row / no opening print. The
    defaults are hostile on purpose (engine tests); real S&P 500 data is near 0.1%."""
    rng = np.random.default_rng(seed)
    N = n_assets or int(rng.integers(2, 9))
    T = n_sessions or int(rng.integers(15, 60))
    # Real XNYS sessions (not every weekday), so calendar helpers such as
    # is_last_session_of_month behave as they do on CRSP data.
    dates = pd.DatetimeIndex(calendar().sessions_window("2015-01-02", T)).tz_localize(None)
    rows = []
    for j in range(N):
        start = int(rng.integers(0, T // 3))
        end = T - 1 if rng.random() < 0.6 else int(rng.integers(start + 2, T))
        raw = float(rng.uniform(3, 300))
        prev_close = None
        for i in range(start, end + 1):
            if prev_close is not None and rng.random() < halt_rate and i != end:
                continue  # halt: no row today; next return is measured from the last close
            gap = rng.normal(0, 0.01)
            intraday = rng.normal(0, 0.015)
            split = 1.0
            div = 0.0
            if prev_close is not None and rng.random() < 0.03:
                split = float(rng.choice([2.0, 3.0, 0.5]))
            if prev_close is not None and rng.random() < 0.05:
                div = prev_close * float(rng.uniform(0.002, 0.03))
            base = raw if prev_close is None else prev_close
            o = base * np.exp(gap) / split - (div / split if div else 0.0)
            o = max(o, 0.01)
            c = max(o * np.exp(intraday), 0.01)
            if prev_close is None:
                ret = np.nan
            else:
                ret = (c * split + div) / prev_close - 1.0
            if i == end and end < T - 1 and rng.random() < 0.5:
                ret = -1.0 if rng.random() < 0.3 else float(rng.uniform(-0.6, 0.1))
            row = {"date": dates[i], "asset": f"S{j}", "open": round(o, 4), "close": round(c, 4),
                   "ret": ret}
            if rng.random() < 0.8:
                half = c * float(rng.uniform(0.0002, 0.01))
                row["bid"], row["ask"] = c - half, c + half
            if rng.random() < missing_open_rate:
                row["open"] = np.nan  # no open print: untradable today
            rows.append(row)
            prev_close = c
    df = pd.DataFrame(rows)
    # Rounding prices after computing ret mimics CRSP: ret is authoritative, prices rounded.
    for col in ("bid", "ask"):
        if col not in df:
            df[col] = np.nan
    _add_range_and_volume(df, np.random.default_rng(seed + 777))
    _add_reference_fields(df, np.random.default_rng(seed + 888))
    return df


def _add_range_and_volume(df: pd.DataFrame, rng) -> None:
    """High/low around open and close (sometimes missing or inconsistent, as in CRSP) and
    share volume (sometimes zero or missing). A separate generator keeps the price paths
    of existing seeds unchanged."""
    n = len(df)
    top = np.fmax(df["open"].to_numpy(), df["close"].to_numpy())
    bot = np.fmin(df["open"].to_numpy(), df["close"].to_numpy())
    df["high"] = np.round(top * (1 + rng.exponential(0.01, n)), 4)
    df["low"] = np.round(bot * (1 - rng.exponential(0.01, n)), 4)
    odd = rng.random(n) < 0.02  # vendor range that excludes the open or close
    df.loc[odd, "high"] = np.round(bot[odd] * 0.999, 4)
    gone = rng.random(n) < 0.02
    df.loc[gone, ["high", "low"]] = np.nan
    df["volume"] = np.round(rng.lognormal(8, 1.5, n))
    df.loc[rng.random(n) < 0.02, "volume"] = 0.0
    df.loc[rng.random(n) < 0.02, "volume"] = np.nan


def _add_reference_fields(df: pd.DataFrame, rng) -> None:
    """retx, shares outstanding (thousands) and market cap ($ thousands), as in CRSP."""
    shares = {a: float(rng.integers(50_000, 5_000_000)) for a in df["asset"].unique()}
    df["shrout"] = df["asset"].map(shares)
    df["mktcap_k"] = df["close"] * df["shrout"]
    df["retx"] = df["ret"]
