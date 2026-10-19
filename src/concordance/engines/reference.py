"""Reference engine: a plain-Python loop written to be read and audited, not to be fast.

Its design follows QuantMega's eventlab backtester (the one QuantMega component whose
accounting reconciled), with its known gaps closed: no missing return is treated as 0,
delisted positions are liquidated at their final total-return value, and positions open at
the end of the window are reported rather than dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from concordance.costs import ExecTables
from concordance.panel import MarketPanel
from concordance.planner import plan_session
from concordance.result import FILL_COLUMNS, AccountWipedOut, BacktestResult
from concordance.strategy import Schedule

_plan = plan_session.py_func  # pure Python: independent of the numba compilation


def run(panel: MarketPanel, sched: Schedule, tables: ExecTables, init_cash: float,
        compiled: bool = False) -> BacktestResult:
    """`compiled=True` uses the numba planner (same rule, much faster) for statistical
    re-runs such as the placebo. Reconciliation always uses the pure-Python planner."""
    plan = plan_session if compiled else _plan
    T, N = panel.shape
    cash = float(init_cash)
    pos = np.zeros(N)
    pending = np.zeros(N)
    last_val = np.full(N, np.nan)  # last known valuation price per asset

    equity = np.empty(T)
    cash_hist = np.empty(T)
    pos_hist = np.empty((T, N))
    fills: list[tuple] = []
    scaled: list[int] = []
    borrow_hist = np.zeros(T)

    def trade(i, phase, j, qty, price, rate):
        nonlocal cash
        notional = abs(qty) * price
        fee = notional * rate
        cash += -(notional + fee) if qty > 0 else notional - fee
        # Only a buy may never overdraw. Borrow fees can leave a fully invested long/short
        # book a little below zero; the planner then scales buys to nothing until sells
        # restore cash.
        if qty > 0 and cash < 0:
            raise AssertionError(f"a buy overdrew cash at session {i}: {cash}")
        pos[j] += qty
        fills.append((i, phase, j, qty, price, fee))

    for i in range(T):
        # --- open: plan and execute -------------------------------------------------
        exit_active = np.isfinite(sched.stop[i]) | np.isfinite(sched.take_profit[i])
        (delta, rate, order, m, was_scaled, limit_qty, limit_rate, exit_qty, exit_rate,
         pending) = plan(
            cash, pos, pending, panel.exec_price[i], last_val, tables.base_fee[i],
            sched.weights[i], bool(sched.has[i]), sched.hold[i], panel.liquidate[i],
            sched.gap_exit[i],
            sched.entry_limit[i], exit_active, tables.cap_buy[i], tables.cap_sell[i],
            tables.adv[i], tables.sigma[i], tables.impact_coef, tables.max_impact,
            tables.min_trade)
        if was_scaled:
            scaled.append(i)
        for k in range(m):
            j = order[k]
            trade(i, 0, j, delta[j], panel.exec_price[i, j], rate[j])

        # --- intraday: resting exits, then resting limit buys (cash already reserved) --
        for j in np.flatnonzero(np.isfinite(sched.exit_price[i])):
            pending[j] = 0.0  # the schedule closed this position
            if exit_qty[j] > 0:
                trade(i, 1, j, -exit_qty[j], sched.exit_price[i, j], exit_rate[j])
        for j in np.flatnonzero(sched.limit_filled[i] & (limit_qty > 0)):
            trade(i, 1, j, limit_qty[j], sched.entry_limit[i, j], limit_rate[j])

        # --- close: mark to market ---------------------------------------------------
        close_px = panel.tr_close[i]
        seen = np.isfinite(close_px)
        last_val[seen] = close_px[seen]
        held = pos != 0
        if np.isnan(last_val[held]).any():
            raise AssertionError(f"holding an asset with no valuation price at session {i}")
        short = pos < 0
        if short.any() and tables.borrow_rate > 0:
            borrow_hist[i] = tables.borrow_rate * float(np.dot(-pos[short], last_val[short]))
            cash -= borrow_hist[i]
        equity[i] = cash + float(np.dot(pos[held], last_val[held]))
        if short.any() and not equity[i] > 0:
            raise AccountWipedOut(f"equity {equity[i]:.2f} <= 0 at session {i} with "
                                  "short positions open")
        cash_hist[i] = cash
        pos_hist[i] = pos

    fills_df = pd.DataFrame(fills, columns=FILL_COLUMNS).astype(
        {"session": np.int64, "phase": np.int64, "asset": np.int64})
    return BacktestResult("reference", panel.sessions, panel.assets, equity, cash_hist,
                          pos_hist, fills_df, scaled, borrow_hist)
