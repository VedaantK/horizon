"""Reference margin-account engine for futures: a plain loop, written to be audited.

Session i, in order (docs/SEMANTICS.md, "Futures"):

1. **Interest.** Cash from the previous close earns that session's risk-free return:
   ``cash *= 1 + rf[i]``. All equity is collateral in bills; futures cost no cash.
2. **Open.**
   - *Mark the overnight change:* if a target was decided at the previous close, held
     contracts first mark from the previous adjusted settle to today's adjusted open:
     ``pos * mult * (adj_open - adj_close_prev)``.
   - *Size the target:* ``round(w * E / (mult * open_unadj))`` contracts, where E is
     equity at the previous close.
   - *Margin cap:* if the targets need more initial margin than
     ``max_margin * E`` (margin = ``|q| * mult * open_unadj * margin_rate``), every target
     is scaled down by the same factor and rounded toward zero.
   - *Trade:* each changed position trades at ``open_unadj``. The fee per contract is
     ``commission + slippage_ticks * tick * mult``.
3. **Close.** Positions mark to the adjusted settle: from the open on a rebalance session,
   else from the previous settle.
4. **Roll.** Where the series switches contract after this close, the position moves to
   the next contract at the settle. That is two legs, so the fee is
   ``2 * |pos| * per-contract fee``. The position size in contracts is unchanged.

Equity is cash: P&L settles daily (variation margin). A session with no bar for a held
market marks nothing; the next bar marks from the last known settle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from concordance.futures.panel import FuturesData
from concordance.panel import MarketPanel
from concordance.result import FILL_COLUMNS, AccountWipedOut, BacktestResult
from concordance.strategy import Schedule

ROLL_PHASE = 2  # fills phase for roll legs (0 = open, 1 = intraday)


@dataclass(frozen=True)
class MarginRules:
    max_margin: float = 0.5  # initial margin may use at most this share of equity


def round_half_away(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def target_contracts(w, equity, mult, price, margin_rate, rules: MarginRules):
    """Integer contract targets for weights `w` (NaN price: no target for that market)."""
    ok = np.isfinite(price) & (price > 0)
    raw = np.zeros_like(w)
    raw[ok] = w[ok] * equity / (mult[ok] * price[ok])
    q = round_half_away(raw)
    need = float(np.sum(np.abs(q[ok]) * mult[ok] * price[ok] * margin_rate[ok]))
    cap = rules.max_margin * equity
    scaled = need > cap > 0
    if scaled:
        q = np.sign(raw) * np.floor(np.abs(raw) * cap / need)
    return q, ok, scaled


DEFAULT_RULES = MarginRules()


def run(panel: MarketPanel, sched: Schedule, init_cash: float,
        rules: MarginRules = DEFAULT_RULES) -> BacktestResult:
    fd: FuturesData = panel.meta["futures"]
    rf = panel.meta["rf"]
    T, N = panel.shape
    mult = fd.multiplier
    fee_per = np.array([s.commission + s.slippage_ticks * s.tick * s.multiplier
                        for s in fd.specs])
    margin_rate = np.array([s.margin_rate for s in fd.specs])

    cash = float(init_cash)
    pos = np.zeros(N)
    last_close = np.full(N, np.nan)  # last adjusted settle seen per market
    equity = np.empty(T)
    cash_hist = np.empty(T)
    pos_hist = np.empty((T, N))
    fills: list[tuple] = []
    scaled: list[int] = []

    for i in range(T):
        e_prev = cash  # equity at the previous close (cash is equity)
        cash *= 1.0 + rf[i]
        ao, ac = fd.adj_open[i], fd.adj_close[i]
        if sched.has[i]:
            # Overnight mark of the pre-trade book, then rebalance at the open.
            moved = (pos != 0) & np.isfinite(ao) & np.isfinite(last_close)
            cash += float(np.sum(pos[moved] * mult[moved] * (ao[moved] - last_close[moved])))
            # From here every market with an open marks from it, old and new contracts alike.
            has_open = np.isfinite(ao)
            last_close[has_open] = ao[has_open]
            q, ok, was_scaled = target_contracts(sched.weights[i], e_prev, mult,
                                                 fd.open_unadj[i], margin_rate, rules)
            if was_scaled:
                scaled.append(i)
            for j in np.flatnonzero(ok & (q != pos)):
                delta = q[j] - pos[j]
                fee = abs(delta) * fee_per[j]
                cash -= fee
                pos[j] = q[j]
                fills.append((i, 0, j, delta, float(fd.open_unadj[i, j]), fee))
        # Close: mark to the adjusted settle.
        moved = (pos != 0) & np.isfinite(ac) & np.isfinite(last_close)
        cash += float(np.sum(pos[moved] * mult[moved] * (ac[moved] - last_close[moved])))
        seen = np.isfinite(ac)
        last_close[seen] = ac[seen]
        # Roll the held contracts that expire into the next one at this settle.
        for j in np.flatnonzero(fd.roll[i] & (pos != 0)):
            fee = 2.0 * abs(pos[j]) * fee_per[j]
            cash -= fee
            fills.append((i, ROLL_PHASE, j, 0.0, float(fd.close_unadj[i, j]), fee))
        equity[i] = cash
        if not cash > 0:
            raise AccountWipedOut(f"equity {cash:.2f} <= 0 at session {i}")
        cash_hist[i] = cash
        pos_hist[i] = pos

    fills_df = pd.DataFrame(fills, columns=FILL_COLUMNS).astype(
        {"session": np.int64, "phase": np.int64, "asset": np.int64})
    return BacktestResult("reference", panel.sessions, panel.assets, equity, cash_hist,
                          pos_hist, fills_df, scaled, np.zeros(T),
                          meta={"account": "margin"})
