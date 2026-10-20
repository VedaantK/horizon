"""Futures market panel: one asset per market (ES, ZN, CL, ...), not per contract.

Built from each market's Panama continuous series (`continuous.build_continuous`). The
engines use only *differences* of adjusted prices, which equal the held contract's own
price changes, across rolls too (see `continuous.py`), so P&L is causal. Strategies never
see adjusted levels (they carry future roll gaps):

    open, close  unadjusted price of the contract held that session
    ret          excess return of a fully collateralized position: the held contract's
                 point change divided by its own previous settle (no cash interest)

`meta["futures"]` holds what the margin engine needs (adjusted open/close for P&L, the
unadjusted open for sizing, roll flags, contract specs) and `meta["rf"]` the daily
risk-free rate earned on the account's cash.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from concordance.panel import MarketPanel

# Futures are margined: weights are notional exposure as a fraction of equity, and the
# account may hold several times its equity in notional (bounded by the margin cap).
WEIGHT_LIMITS = {"max_net": 8.0, "max_gross": 8.0}


@dataclass(frozen=True)
class Spec:
    root: str
    multiplier: float  # USD per 1.0 price point, per contract
    tick: float  # minimum price increment, in price points
    margin_rate: float  # initial margin as a share of notional (exchange-level, approximate)
    commission: float = 2.0  # USD per contract per side (broker + exchange fees)
    slippage_ticks: float = 1.0  # ticks per side, at every trade and roll leg


@dataclass(frozen=True)
class FuturesData:
    adj_open: np.ndarray  # (T, N) Panama-adjusted open of the held contract
    adj_close: np.ndarray  # (T, N) Panama-adjusted settle of the held contract
    open_unadj: np.ndarray  # (T, N) the held contract's own open (sizing, fills)
    close_unadj: np.ndarray  # (T, N) the held contract's own settle
    roll: np.ndarray  # (T, N) bool: the position moves to the next contract at this close
    contract: np.ndarray  # (T, N) object: held contract id
    specs: tuple[Spec, ...]

    @property
    def multiplier(self) -> np.ndarray:
        return np.array([s.multiplier for s in self.specs])


def build_futures_panel(series: dict[str, pd.DataFrame], specs: dict[str, Spec],
                        rf: pd.Series, sessions: pd.DatetimeIndex | None = None
                        ) -> MarketPanel:
    """`series[root]` is `build_continuous(...)[0]` (adjusted OHLC, `close_unadj`,
    `contract`). `rf` is the daily risk-free return indexed by session."""
    roots = sorted(series)
    if sessions is None:
        sessions = pd.DatetimeIndex(sorted(set().union(*(s.index for s in series.values()))))
    T, N = len(sessions), len(roots)
    arr = {k: np.full((T, N), np.nan) for k in ("ao", "ac", "ou", "cu")}
    roll = np.zeros((T, N), bool)
    contract = np.full((T, N), None, dtype=object)
    for j, root in enumerate(roots):
        s = series[root].reindex(sessions)
        offset = s["close"] - s["close_unadj"]
        arr["ac"][:, j] = s["close"]
        arr["ao"][:, j] = s["open"]
        arr["cu"][:, j] = s["close_unadj"]
        arr["ou"][:, j] = s["open"] - offset
        held = s["contract"]
        nxt = held.shift(-1)
        roll[:, j] = (held.notna() & nxt.notna() & (held != nxt)).to_numpy()
        contract[:, j] = held.to_numpy()

    # Excess return: today's point change over the held contract's own previous settle.
    # After a roll, "previous settle" is the new contract's settle at the roll close.
    prev_own = np.vstack([np.full((1, N), np.nan), arr["ac"][:-1]]) - (arr["ac"] - arr["cu"])
    ret = (arr["ac"] - np.vstack([np.full((1, N), np.nan), arr["ac"][:-1]])) / prev_own
    active = np.isfinite(arr["cu"])
    tr = np.where(active, arr["ac"], np.nan)  # valuation marks for reconciliation only
    panel = MarketPanel(
        sessions=sessions,
        assets=pd.Index(roots, name="market"),
        fields={"open": arr["ou"], "close": arr["cu"], "ret": ret},
        member=active,
        tr_close=tr,
        exec_price=arr["ou"],
        liquidate=np.zeros((T, N), bool),
        val_price=pd.DataFrame(tr).ffill().to_numpy(),
        meta={
            "source": "futures",
            "account": "margin",
            "weight_limits": WEIGHT_LIMITS,
            "futures": FuturesData(arr["ao"], arr["ac"], arr["ou"], arr["cu"], roll,
                                   contract, tuple(specs[r] for r in roots)),
            "rf": rf.reindex(sessions).fillna(0.0).to_numpy(float),
        },
    )
    return panel
