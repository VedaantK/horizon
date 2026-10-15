"""Order planning at a session open — the single sizing rule all engines share.

Numba-compiled so vectorbt can call it from inside its simulation loop; the reference
engine calls the pure-Python original (`plan_orders.py_func`). See docs/SEMANTICS.md.
"""

from __future__ import annotations

import numpy as np
from numba import njit

QTY_SCALE = 1e6  # quantities live on a 1e-6 grid (Nautilus size_precision=6)
SAFETY = 1e-9  # keeps scaled buys strictly inside available cash under float rounding


@njit(cache=True)
def _round_qty(x):
    return np.round(x * QTY_SCALE) / QTY_SCALE


@njit(cache=True)
def plan_orders(cash, pos, exec_price, mark_prev, fee_rate, weights, has_weights, liquidate):
    """Return (delta, order, n_orders, scaled).

    delta[j]  signed quantity to trade in asset j (0 = no order)
    order     asset indices to execute, sells first then buys, each in ascending index
    scaled    True if buys were scaled down to fit available cash
    """
    n = pos.shape[0]
    tradable = np.isfinite(exec_price)

    value = cash
    for j in range(n):
        if pos[j] != 0.0:
            mark = exec_price[j] if tradable[j] else mark_prev[j]
            value += pos[j] * mark

    delta = np.zeros(n)
    for j in range(n):
        if not tradable[j]:
            continue
        if liquidate[j]:
            target = 0.0
        elif has_weights:
            target = _round_qty(weights[j] * value / exec_price[j])
        else:
            continue
        if target == 0.0:
            delta[j] = -pos[j]  # a full exit sells exactly what is held: no float residue
        else:
            delta[j] = _round_qty(target - pos[j])

    # Cash after all sells, then check buys fit; scale them uniformly if not.
    cash_after = cash
    buy_cost = 0.0
    for j in range(n):
        if delta[j] < 0.0:
            notional = -delta[j] * exec_price[j]
            cash_after += notional - notional * fee_rate[j]
        elif delta[j] > 0.0:
            notional = delta[j] * exec_price[j]
            buy_cost += notional + notional * fee_rate[j]
    scaled = False
    if buy_cost > cash_after:
        scaled = True
        k = max(cash_after, 0.0) / buy_cost * (1.0 - SAFETY)
        for j in range(n):
            if delta[j] > 0.0:
                delta[j] = np.floor(delta[j] * k * QTY_SCALE) / QTY_SCALE

    order = np.empty(n, dtype=np.int64)
    m = 0
    for j in range(n):
        if delta[j] < 0.0:
            order[m] = j
            m += 1
    for j in range(n):
        if delta[j] > 0.0:
            order[m] = j
            m += 1
    return delta, order, m, scaled
