"""Canonical market panel: the single data object every engine consumes.

A panel is built from a long frame (one row per session x asset) and holds dense
(T x N) arrays. Raw fields are for strategy logic; the derived total-return prices,
execution prices and valuation prices are what engines use for P&L. See docs/SEMANTICS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRICE_DECIMALS = 9
QTY_DECIMALS = 6

# Fields a long frame may carry; only date/asset/open/close/ret are required.
REQUIRED = ("date", "asset", "open", "close", "ret")
OPTIONAL_FLOAT = ("high", "low", "volume", "bid", "ask", "retx", "mktcap_k", "shrout")


class PanelError(ValueError):
    """Raised when input data violates the panel contract. Never silently repaired."""


@dataclass(frozen=True)
class MarketPanel:
    sessions: pd.DatetimeIndex
    assets: pd.Index
    fields: dict[str, np.ndarray]  # raw (T, N) float arrays visible to strategies
    member: np.ndarray  # (T, N) bool universe membership (e.g. S&P 500 on that session)
    tr_close: np.ndarray  # (T, N) total-return close, NaN where no observation
    exec_price: np.ndarray  # (T, N) price an order at the open of session i fills at
    liquidate: np.ndarray  # (T, N) bool forced full liquidation at the open of session i
    val_price: np.ndarray  # (T, N) forward-filled tr_close for valuation (NaN before first obs)
    meta: dict = field(default_factory=dict)
    # Point-in-time supplier -> customer links (data/customers.py), when attached.
    links: object = None
    # Point-in-time quarterly fundamentals (data/fundamentals.py), when attached.
    fundamentals: object = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.tr_close.shape

    def __getitem__(self, name: str) -> np.ndarray:
        return self.fields[name]

    def frame(self, name: str) -> pd.DataFrame:
        arr = {"tr_close": self.tr_close, "exec_price": self.exec_price,
               "member": self.member}.get(name)
        if arr is None:
            arr = self.fields[name]
        return pd.DataFrame(arr, index=self.sessions, columns=self.assets)


def _round_price(x: np.ndarray) -> np.ndarray:
    return np.round(x, PRICE_DECIMALS)


def build_panel(long: pd.DataFrame, sessions: pd.DatetimeIndex | None = None,
                member_col: str | None = "member") -> MarketPanel:
    """Build a panel from a long frame. Validates the contract and raises on violations.

    `ret` must be the CRSP total return including any delisting return, measured from the
    previous available close. `open`/`close` are raw (unadjusted) prices, positive.
    """
    missing = [c for c in REQUIRED if c not in long.columns]
    if missing:
        raise PanelError(f"missing required columns: {missing}")
    df = long.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    dup = df.duplicated(["date", "asset"], keep=False)
    if dup.any():
        raise PanelError(f"{int(dup.sum())} duplicate (date, asset) rows, e.g.\n"
                         f"{df.loc[dup].head()}")

    if sessions is None:
        sessions = pd.DatetimeIndex(sorted(df["date"].unique()))
    else:
        sessions = pd.DatetimeIndex(sessions).tz_localize(None).normalize()
        extra = ~df["date"].isin(sessions)
        if extra.any():
            raise PanelError(f"{int(extra.sum())} rows on non-session dates, "
                             f"e.g. {df.loc[extra, 'date'].iloc[0].date()}")
    assets = pd.Index(sorted(df["asset"].unique()), name="asset")

    def dense(col: str, fill=np.nan, dtype=float) -> np.ndarray:
        wide = df.pivot(index="date", columns="asset", values=col)
        wide = wide.reindex(index=sessions, columns=assets)
        return wide.to_numpy(dtype=dtype, na_value=fill)

    open_ = dense("open")
    close = dense("close")
    ret = dense("ret")
    fields = {"open": open_, "close": close, "ret": ret}
    for c in OPTIONAL_FLOAT:
        if c in df.columns:
            fields[c] = dense(c)

    if member_col and member_col in df.columns:
        member = dense(member_col, fill=0.0).astype(bool)
    else:
        member = np.isfinite(close)

    _validate(sessions, assets, open_, close, ret)
    tr_close, exec_price, liquidate, val_price = derive_prices(open_, close, ret)
    return MarketPanel(sessions=sessions, assets=assets, fields=fields, member=member,
                       tr_close=tr_close, exec_price=exec_price, liquidate=liquidate,
                       val_price=val_price)


def _validate(sessions, assets, open_, close, ret) -> None:
    def where(mask):
        i, j = np.argwhere(mask)[0]
        return f"{sessions[i].date()} asset={assets[j]}"

    bad = np.isfinite(close) & (close <= 0)
    if bad.any():
        raise PanelError(f"non-positive close at {where(bad)} (take abs() of CRSP prc first)")
    bad = np.isfinite(open_) & (open_ <= 0)
    if bad.any():
        raise PanelError(f"non-positive open at {where(bad)}")
    bad = np.isfinite(open_) & ~np.isfinite(close)
    if bad.any():
        raise PanelError(f"open without close at {where(bad)}")
    has = np.isfinite(close)
    first = np.where(has.any(0), has.argmax(0), -1)
    later = has.copy()
    for j, f in enumerate(first):
        if f >= 0:
            later[f, j] = False
    bad = later & ~np.isfinite(ret)
    if bad.any():
        raise PanelError(f"missing total return mid-history at {where(bad)}; "
                         "refusing to treat it as 0")
    bad = np.isfinite(ret) & (ret <= -1.0) & later
    # A -100% return is legitimate only as the final observation (delisting to zero).
    if bad.any():
        last = np.where(has.any(0), has.shape[0] - 1 - has[::-1].argmax(0), -1)
        for i, j in np.argwhere(bad):
            if i != last[j]:
                raise PanelError(f"return <= -100% before the last observation at "
                                 f"{sessions[i].date()} asset={assets[j]}")


def derive_prices(open_: np.ndarray, close: np.ndarray, ret: np.ndarray):
    """Causal total-return prices plus execution/valuation arrays (see SEMANTICS.md)."""
    T, N = close.shape
    has = np.isfinite(close)
    any_obs = has.any(0)
    first = np.where(any_obs, has.argmax(0), 0)
    # Growth factors: 1 on rows without an observation, anchor = first close (known then).
    growth = np.where(has, 1.0 + np.nan_to_num(ret), 1.0)
    cols = np.flatnonzero(any_obs)
    growth[first[cols], cols] = close[first[cols], cols]
    tr_close = _round_price(np.where(has, np.cumprod(growth, axis=0), np.nan))

    with np.errstate(invalid="ignore", divide="ignore"):
        tr_open = _round_price(tr_close * open_ / close)
    exec_price = np.where(np.isfinite(open_) & (tr_open > 0), tr_open, np.nan)

    liquidate = np.zeros((T, N), dtype=bool)
    last = np.where(any_obs, T - 1 - has[::-1].argmax(0), -1)
    for j in range(N):
        if any_obs[j] and last[j] < T - 1:
            liquidate[last[j] + 1, j] = True
            # Sold at the final total-return value; a -100% delisting gives price 0,
            # which is represented by a tiny positive price so engines accept it.
            exec_price[last[j] + 1, j] = max(tr_close[last[j], j], 10.0 ** -PRICE_DECIMALS)

    val_price = pd.DataFrame(tr_close).ffill().to_numpy()
    return tr_close, exec_price, liquidate, val_price
