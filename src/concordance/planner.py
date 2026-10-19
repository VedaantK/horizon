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
def _capped(desired, pos, cap_buy, cap_sell):
    """The part of `desired` the volume cap allows this session.

    Shares that reduce the position (an exit or a cover) use `cap_sell`, which is unlimited
    when volume history is unknown; shares that open or add to a position (long or short)
    use `cap_buy`, which is 0 then. With a known cap both are equal, so the total traded is
    capped once. For long-only trading this is exactly the earlier buy/sell rule.
    """
    if desired == 0.0:
        return 0.0
    size = abs(desired)
    reducing = (desired > 0.0) != (pos > 0.0) and pos != 0.0
    reduce = min(size, abs(pos)) if reducing else 0.0
    enter = size - reduce if reducing else size
    cs = _floor_qty(cap_sell)
    cb = _floor_qty(cap_buy)
    out = reduce if reduce <= cs else cs
    room = cb - out
    if enter > 0.0 and room > 0.0:
        out = _round_qty(out + (enter if enter <= room else _floor_qty(room)))
    return out if desired > 0.0 else -out


@njit(cache=True)
def plan_session(cash, pos, pending, exec_price, mark_prev, base_fee, weights, has_weights,
                 hold, liquidate, gap_exit, limit_px, exit_active, cap_buy, cap_sell, adv,
                 sigma, impact_coef, max_impact, min_trade=0.0):
    """Plan session i from the state at its open. `min_trade` is a fraction of the
    pre-trade value: a rebalance trade worth less is skipped (full exits always trade).

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
            # a full exit trades exactly what is held: no float residue
            desired = -pos[j] if target == 0.0 else _round_qty(target - pos[j])
            if (min_trade > 0.0 and target != 0.0
                    and abs(desired) * exec_price[j] < min_trade * value):
                desired = 0.0  # too small to be worth a trade; the position stays as it is
        elif pending[j] != 0.0 and abs(pending[j] + pos[j]) <= HALF_TICK:
            desired = -pos[j]  # a carried trade within half a grid step of flat is a full exit
        else:
            desired = pending[j]
        delta[j] = _capped(desired, pos[j], cap_buy[j], cap_sell[j])
        new_pending[j] = _round_qty(desired - delta[j])

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
    if buy_cost > cash_after and buy_cost > 0.0:  # cash can dip below 0 from borrow fees
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
