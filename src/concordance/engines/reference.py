"""Reference engine: a plain-Python loop written to be read and audited, not to be fast.

Its design follows QuantMega's eventlab backtester (the one QuantMega component whose
accounting reconciled), with its known gaps closed: no missing return is treated as 0,
delisted positions are liquidated at their final total-return value, and positions open at
the end of the window are reported rather than dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from concordance.panel import MarketPanel
from concordance.planner import plan_orders
from concordance.result import FILL_COLUMNS, BacktestResult

_plan = plan_orders.py_func  # pure Python: independent of the numba compilation


def run(panel: MarketPanel, weights: np.ndarray, has_weights: np.ndarray,
        fee_rate: np.ndarray, init_cash: float, compiled: bool = False) -> BacktestResult:
    """`compiled=True` uses the numba planner (same rule, much faster) for statistical
    re-runs such as the placebo. Reconciliation always uses the pure-Python planner."""
    plan = plan_orders if compiled else _plan
    T, N = panel.shape
    cash = float(init_cash)
    pos = np.zeros(N)
    last_val = np.full(N, np.nan)  # last known valuation price per asset

    equity = np.empty(T)
    cash_hist = np.empty(T)
    pos_hist = np.empty((T, N))
    fills: list[tuple] = []
    scaled: list[int] = []

    for i in range(T):
        # --- open: plan and execute -------------------------------------------------
        delta, order, m, was_scaled = plan(
            cash, pos, panel.exec_price[i], last_val, fee_rate[i],
            weights[i], bool(has_weights[i]), panel.liquidate[i])
        if was_scaled:
            scaled.append(i)
        for k in range(m):
            j = order[k]
            qty = delta[j]
            price = panel.exec_price[i, j]
            notional = abs(qty) * price
            fee = notional * fee_rate[i, j]
            if qty > 0:
                cash -= notional + fee
            else:
                cash += notional - fee
            if cash < 0:
                raise AssertionError(f"cash went negative at session {i}: {cash}")
            pos[j] += qty
            fills.append((i, j, qty, price, fee))

        # --- close: mark to market ---------------------------------------------------
        close_px = panel.tr_close[i]
        seen = np.isfinite(close_px)
        last_val[seen] = close_px[seen]
        held = pos != 0
        if np.isnan(last_val[held]).any():
            raise AssertionError(f"holding an asset with no valuation price at session {i}")
        equity[i] = cash + float(np.dot(pos[held], last_val[held]))
        cash_hist[i] = cash
        pos_hist[i] = pos

    fills_df = pd.DataFrame(fills, columns=FILL_COLUMNS).astype(
        {"session": np.int64, "asset": np.int64})
    return BacktestResult("reference", panel.sessions, panel.assets, equity, cash_hist,
                          pos_hist, fills_df, scaled)
