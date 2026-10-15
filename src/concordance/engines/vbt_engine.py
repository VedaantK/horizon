"""vectorbt engine: vectorbt's own simulator does the fills, cash, positions and valuation.

Sizing uses the shared planner, called inside vectorbt's pre-segment callback from
vectorbt's own state (its cash, positions and forward-filled valuation prices). Orders are
sent with `allow_partial=False, raise_reject=True`, so if vectorbt ever wanted to shrink or
reject an order the run fails loudly instead of drifting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from vectorbt.portfolio.enums import Direction, NoOrder, SizeType
from vectorbt.portfolio.nb import order_nb

from concordance.panel import MarketPanel
from concordance.planner import plan_orders
from concordance.result import FILL_COLUMNS, BacktestResult


@njit
def _pre_segment(c, exec_price, fee_rate, weights, has_weights, liquidate, scaled_out):
    i = c.i
    n = c.to_col - c.from_col
    pos = c.last_position[c.from_col:c.to_col].copy()
    mark_prev = c.last_val_price[c.from_col:c.to_col].copy()
    delta, order, m, scaled = plan_orders(
        c.last_cash[c.group], pos, exec_price[i], mark_prev, fee_rate[i],
        weights[i], has_weights[i], liquidate[i])
    scaled_out[i] = scaled
    # Call sequence: planned orders first (sells, then buys), then everything else.
    used = np.zeros(n, dtype=np.bool_)
    for k in range(m):
        c.call_seq_now[k] = order[k]
        used[order[k]] = True
    k = m
    for j in range(n):
        if not used[j]:
            c.call_seq_now[k] = j
            k += 1
    return (delta,)


@njit
def _order(c, delta, exec_price, fee_rate):
    j = c.col - c.from_col
    d = delta[j]
    if d == 0.0:
        return NoOrder
    return order_nb(
        size=d,
        price=exec_price[c.i, c.col],
        size_type=SizeType.Amount,
        direction=Direction.LongOnly,
        fees=fee_rate[c.i, c.col],
        allow_partial=False,
        raise_reject=True,
    )


def run(panel: MarketPanel, weights: np.ndarray, has_weights: np.ndarray,
        fee_rate: np.ndarray, init_cash: float) -> BacktestResult:
    T, N = panel.shape
    close = pd.DataFrame(panel.tr_close, index=panel.sessions, columns=panel.assets)
    exec_price = np.ascontiguousarray(panel.exec_price, dtype=np.float64)
    fee = np.ascontiguousarray(fee_rate, dtype=np.float64)
    w = np.ascontiguousarray(weights, dtype=np.float64)
    has = np.ascontiguousarray(has_weights, dtype=np.bool_)
    liq = np.ascontiguousarray(panel.liquidate, dtype=np.bool_)
    scaled_out = np.zeros(T, dtype=np.bool_)

    pf = vbt.Portfolio.from_order_func(
        close,
        _order,
        exec_price, fee,
        pre_segment_func_nb=_pre_segment,
        pre_segment_args=(exec_price, fee, w, has, liq, scaled_out),
        segment_mask=True,
        call_pre_segment=True,
        cash_sharing=True,
        group_by=True,
        init_cash=init_cash,
        ffill_val_price=True,
        update_value=True,
        freq="1D",
    )

    equity = pf.value().to_numpy(dtype=float)
    cash = pf.cash().to_numpy(dtype=float)
    positions = pf.assets().to_numpy(dtype=float)
    rec = pf.orders.values
    side = np.where(rec["side"] == 0, 1.0, -1.0)
    fills = pd.DataFrame({
        "session": rec["idx"].astype(np.int64),
        "asset": rec["col"].astype(np.int64),
        "qty": side * rec["size"],
        "price": rec["price"],
        "fee": rec["fees"],
    })[FILL_COLUMNS]
    return BacktestResult("vectorbt", panel.sessions, panel.assets, equity, cash, positions,
                          fills, [int(i) for i in np.flatnonzero(scaled_out)])
