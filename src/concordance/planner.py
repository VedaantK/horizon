"""Order planning at a session open: the single sizing rule all engines share.

Numba-compiled so vectorbt can call it from inside its simulation loop; the reference
engine calls the pure-Python original (`plan_session.py_func`). See docs/SEMANTICS.md.
"""

from __future__ import annotations

import numpy as np
from numba import njit

QTY_SCALE = 1e6  # quantities live on a 1e-6 grid (Nautilus size_precision=6)
HALF_TICK = 0.5 / QTY_SCALE
SAFETY = 1e-9  # keeps scaled buys strictly inside available cash under float rounding


@njit(cache=True)
def _round_qty(x):
    return np.round(x * QTY_SCALE) / QTY_SCALE


@njit(cache=True)
def _floor_qty(x):
    return np.floor(x * QTY_SCALE) / QTY_SCALE


@njit(cache=True)
def impact_rate(qty, adv, sigma, coef, max_impact):
    """Square-root market impact as a rate on notional (0 when the model is off)."""
    if coef == 0.0 and max_impact == 0.0:
        return 0.0
    if not (adv > 0.0) or not np.isfinite(sigma):
        return max_impact
    r = coef * sigma * np.sqrt(qty / adv)
    return r if r < max_impact else max_impact


@njit(cache=True)
def _fees(rate, limit_rate, delta, limit_qty, liquidate, base_fee, adv, sigma, impact_coef,
          max_impact):
    for j in range(delta.shape[0]):
        rate[j] = 0.0
        if delta[j] != 0.0 and not liquidate[j]:
            rate[j] = base_fee[j] + impact_rate(abs(delta[j]), adv[j], sigma[j],
                                                impact_coef, max_impact)
        limit_rate[j] = 0.0
        if limit_qty[j] > 0.0:
            limit_rate[j] = base_fee[j] + impact_rate(limit_qty[j], adv[j], sigma[j],
                                                      impact_coef, max_impact)


@njit(cache=True)
def plan_session(cash, pos, pending, exec_price, mark_prev, base_fee, weights, has_weights,
                 hold, liquidate, gap_exit, limit_px, exit_active, cap_buy, cap_sell, adv,
                 sigma, impact_coef, max_impact):
    """Plan session i from the state at its open.

    Returns (delta, rate, order, m, scaled, limit_qty, limit_rate, exit_qty, exit_rate,
    new_pending):
      delta[j]      signed quantity traded at the open (0 = no order); `rate` its fee rate
      order[:m]     open orders to execute: sells first, then buys, each in ascending index
      scaled        True if buys (open and limit) were scaled down to fit available cash
      limit_qty[j]  quantity of the resting limit buy (fills only if the level is reached)
      exit_qty[j]   quantity a resting stop / take-profit sells if it triggers
      new_pending   rebalance quantity the volume cap held back, carried to the next open
    """
    n = pos.shape[0]
    tradable = np.isfinite(exec_price)

    value = cash
    for j in range(n):
        if pos[j] != 0.0:
            mark = exec_price[j] if tradable[j] else mark_prev[j]
            value += pos[j] * mark

    delta = np.zeros(n)
    limit_qty = np.zeros(n)
    new_pending = pending.copy()
    for j in range(n):
        fresh = has_weights and not hold[j]  # a new target replaces carried quantity
        if fresh:
            new_pending[j] = 0.0
        if not tradable[j]:
            continue
        if liquidate[j] or gap_exit[j]:
            delta[j] = -pos[j]  # never capped
            new_pending[j] = 0.0
            continue
        if np.isfinite(limit_px[j]):  # resting limit buy; has_weights is True here
            want = _round_qty(_round_qty(weights[j] * value / limit_px[j]) - pos[j])
            if want > 0.0:
                cap = _floor_qty(cap_buy[j])
                limit_qty[j] = want if want <= cap else cap
            continue
        if fresh:
            target = _round_qty(weights[j] * value / exec_price[j])
            # a full exit sells exactly what is held: no float residue
            desired = -pos[j] if target == 0.0 else _round_qty(target - pos[j])
        else:
            # a carried sell within half a grid step of the position is a full exit
            desired = -pos[j] if pending[j] <= -pos[j] + HALF_TICK else pending[j]
        done = 0.0
        if desired > 0.0:
            cap = _floor_qty(cap_buy[j])
            done = desired if desired <= cap else cap
        elif desired < 0.0:
            cap = _floor_qty(cap_sell[j])
            done = desired if -desired <= cap else -cap
        delta[j] = done
        new_pending[j] = _round_qty(desired - done)

    rate = np.zeros(n)
    limit_rate = np.zeros(n)
    _fees(rate, limit_rate, delta, limit_qty, liquidate, base_fee, adv, sigma, impact_coef,
          max_impact)
    # Cash after all sells, then check every buy (open and limit) fits; scale uniformly if
    # not. Limit buys reserve cash at the open, so intraday fill order never matters.
    cash_after = cash
    buy_cost = 0.0
    for j in range(n):
        if delta[j] < 0.0:
            notional = -delta[j] * exec_price[j]
            cash_after += notional - notional * rate[j]
        elif delta[j] > 0.0:
            notional = delta[j] * exec_price[j]
            buy_cost += notional + notional * rate[j]
        if limit_qty[j] > 0.0:
            notional = limit_qty[j] * limit_px[j]
            buy_cost += notional + notional * limit_rate[j]
    scaled = False
    if buy_cost > cash_after:
        scaled = True
        k = max(cash_after, 0.0) / buy_cost * (1.0 - SAFETY)
        for j in range(n):
            if delta[j] > 0.0:
                delta[j] = _floor_qty(delta[j] * k)
            if limit_qty[j] > 0.0:
                limit_qty[j] = _floor_qty(limit_qty[j] * k)
        # impact falls with size, so the scaled buys still fit
        _fees(rate, limit_rate, delta, limit_qty, liquidate, base_fee, adv, sigma,
              impact_coef, max_impact)

    exit_qty = np.zeros(n)
    exit_rate = np.zeros(n)
    for j in range(n):
        if exit_active[j] and tradable[j]:
            held = pos[j] + delta[j]  # exactly the position after the open fills
            if held > 0.0:
                exit_qty[j] = held
                exit_rate[j] = base_fee[j] + impact_rate(held, adv[j], sigma[j],
                                                         impact_coef, max_impact)

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
    return (delta, rate, order, m, scaled, limit_qty, limit_rate, exit_qty, exit_rate,
            new_pending)
