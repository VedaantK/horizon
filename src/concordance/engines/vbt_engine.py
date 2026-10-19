"""vectorbt engine: vectorbt's own simulator does the fills, cash, positions and valuation.

Sizing uses the shared planner, called inside vectorbt's pre-segment callback from
vectorbt's own state (its cash, positions and forward-filled valuation prices). vectorbt's
flexible mode then executes the open orders and, after them, the intraday exits and limit
buys the schedule says were reached. Orders are sent with `allow_partial=False,
raise_reject=True`, so if vectorbt ever wanted to shrink or reject an order the run fails
loudly instead of drifting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from vectorbt.portfolio.enums import Direction, NoOrder, SizeType
from vectorbt.portfolio.nb import order_nb

from concordance.costs import ExecTables
from concordance.panel import MarketPanel
from concordance.planner import plan_session
from concordance.result import FILL_COLUMNS, AccountWipedOut, BacktestResult
from concordance.strategy import Schedule


@njit
def _pre_segment(c, exec_price, base_fee, weights, has_weights, hold, liquidate, gap_exit,
                 stop, tp, entry_limit, exit_price, limit_filled, cap_buy, cap_sell, adv, sigma,
                 impact_coef, max_impact, min_trade, pending, scaled_out, log, log_n, charged):
    i = c.i
    n = c.to_col - c.from_col
    pos = c.last_position[c.from_col:c.to_col].copy()
    mark_prev = c.last_val_price[c.from_col:c.to_col].copy()
    exit_active = np.isfinite(stop[i]) | np.isfinite(tp[i])
    (delta, rate, order, m, scaled, limit_qty, limit_rate, exit_qty, exit_rate,
     new_pending) = plan_session(
        c.last_cash[c.group] - charged[0], pos, pending, exec_price[i], mark_prev, base_fee[i],
        weights[i], has_weights[i], hold[i], liquidate[i], gap_exit[i], entry_limit[i],
        exit_active,
        cap_buy[i], cap_sell[i], adv[i], sigma[i], impact_coef, max_impact, min_trade)
    scaled_out[i] = scaled
    pending[:] = new_pending

    # Orders for this session in execution order: open (sells, buys), exits, limit buys.
    cols = np.empty(3 * n, dtype=np.int64)
    size = np.empty(3 * n)
    price = np.empty(3 * n)
    fee = np.empty(3 * n)
    k = 0
    for q in range(m):
        j = order[q]
        cols[k], size[k], price[k], fee[k] = j, delta[j], exec_price[i, j], rate[j]
        log[log_n[0]] = 0
        log_n[0] += 1
        k += 1
    for j in range(n):
        if np.isfinite(exit_price[i, j]):
            pending[j] = 0.0  # the schedule closed this position
            if exit_qty[j] > 0.0:
                cols[k], size[k], price[k], fee[k] = j, -exit_qty[j], exit_price[i, j], \
                    exit_rate[j]
                log[log_n[0]] = 1
                log_n[0] += 1
                k += 1
    for j in range(n):
        if limit_filled[i, j] and limit_qty[j] > 0.0:
            cols[k], size[k], price[k], fee[k] = j, limit_qty[j], entry_limit[i, j], \
                limit_rate[j]
            log[log_n[0]] = 1
            log_n[0] += 1
            k += 1
    return (cols, size, price, fee, k)


@njit
def _post_segment(c, borrow_rate, borrow_out, charged):
    """At the close: stock borrow on vectorbt's short positions, valued at vectorbt's own
    (forward-filled) close. vectorbt has no cash-flow input for this (and refuses any order
    once its cash is <= 0), so the fee is kept outside it: `charged` is the running total
    the planner subtracts from vectorbt's cash, and the cash and value series vectorbt
    rebuilds from its order records are reduced by the cumulative fee afterwards."""
    if borrow_rate == 0.0:
        return
    short_value = 0.0
    for col in range(c.from_col, c.to_col):
        if c.last_position[col] < 0.0:
            short_value -= c.last_position[col] * c.last_val_price[col]
    if short_value > 0.0:
        fee = borrow_rate * short_value
        borrow_out[c.i] = fee
        charged[0] += fee


@njit
def _flex_order(c, cols, size, price, fee, k):
    if c.call_idx >= k:
        return -1, NoOrder
    return c.from_col + cols[c.call_idx], order_nb(
        size=size[c.call_idx],
        price=price[c.call_idx],
        size_type=SizeType.Amount,
        direction=Direction.Both,
        fees=fee[c.call_idx],
        allow_partial=False,
        raise_reject=True,
    )


def run(panel: MarketPanel, sched: Schedule, tables: ExecTables,
        init_cash: float) -> BacktestResult:
    T, N = panel.shape
    close = pd.DataFrame(panel.tr_close, index=panel.sessions, columns=panel.assets)

    def f64(a):
        return np.ascontiguousarray(a, dtype=np.float64)

    def b(a):
        return np.ascontiguousarray(a, dtype=np.bool_)

    scaled_out = np.zeros(T, dtype=np.bool_)
    borrow_out = np.zeros(T)
    charged = np.zeros(1)
    # At most one open order per (session, asset), plus the intraday fills the schedule has.
    max_orders = T * N + int(np.isfinite(sched.exit_price).sum() + sched.limit_filled.sum())
    log = np.full(max_orders, -1, dtype=np.int64)
    log_n = np.zeros(1, dtype=np.int64)
    args = (f64(panel.exec_price), f64(tables.base_fee), f64(sched.weights), b(sched.has),
            b(sched.hold), b(panel.liquidate), b(sched.gap_exit), f64(sched.stop),
            f64(sched.take_profit), f64(sched.entry_limit), f64(sched.exit_price),
            b(sched.limit_filled),
            f64(tables.cap_buy), f64(tables.cap_sell), f64(tables.adv), f64(tables.sigma),
            float(tables.impact_coef), float(tables.max_impact), float(tables.min_trade),
            np.zeros(N), scaled_out,
            log, log_n, charged)

    pf = vbt.Portfolio.from_order_func(
        close,
        _flex_order,
        flexible=True,
        pre_segment_func_nb=_pre_segment,
        pre_segment_args=args,
        post_segment_func_nb=_post_segment,
        post_segment_args=(float(tables.borrow_rate), borrow_out, charged),
        call_post_segment=True,
        segment_mask=True,
        call_pre_segment=True,
        cash_sharing=True,
        group_by=True,
        init_cash=init_cash,
        ffill_val_price=True,
        update_value=True,
        max_orders=max_orders,
        freq="1D",
    )

    charged = np.cumsum(borrow_out)
    equity = pf.value().to_numpy(dtype=float) - charged
    cash = pf.cash().to_numpy(dtype=float) - charged
    positions = pf.assets().to_numpy(dtype=float)
    wiped = (positions < 0).any(axis=1) & ~(equity > 0)
    if wiped.any():
        i = int(np.argmax(wiped))
        raise AccountWipedOut(f"vectorbt: equity {equity[i]:.2f} <= 0 at session {i}")
    rec = pf.orders.values
    if len(rec) != log_n[0]:
        raise RuntimeError(f"vectorbt recorded {len(rec)} orders, {log_n[0]} were sent")
    # Records are in creation order within a session, so open fills precede intraday ones.
    order = np.lexsort((np.arange(len(rec)), rec["idx"]))
    rec = rec[order]
    side = np.where(rec["side"] == 0, 1.0, -1.0)
    fills = pd.DataFrame({
        "session": rec["idx"].astype(np.int64),
        "phase": log[: log_n[0]][order],
        "asset": rec["col"].astype(np.int64),
        "qty": side * rec["size"],
        "price": rec["price"],
        "fee": rec["fees"],
    })[FILL_COLUMNS]
    return BacktestResult("vectorbt", panel.sessions, panel.assets, equity, cash, positions,
                          fills, [int(i) for i in np.flatnonzero(scaled_out)], borrow_out)
