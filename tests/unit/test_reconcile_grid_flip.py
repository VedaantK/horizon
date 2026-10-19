"""A sub-1e-9 money difference can tip an order size across a half step of the 1e-6 grid.
Reconciliation accepts exactly that one step and only the money it explains."""

import numpy as np
import pandas as pd

from concordance.reconcile import reconcile
from concordance.result import FILL_COLUMNS, BacktestResult

SESSIONS = pd.bdate_range("2002-10-14", periods=3)
ASSETS = pd.Index([88587])
PRICE, RATE, CASH0 = 100.0, 0.0002, 100_000.0


def result(engine, qty, cash_error=0.0, sell=True, marks=(PRICE, PRICE, PRICE)):
    """Buy `qty` on day 0 at PRICE, optionally sell it on day 1; valued at `marks`."""
    fee = abs(qty) * PRICE * RATE
    rows = [(0, 0, 0, qty, PRICE, fee)]
    cash = CASH0 - qty * PRICE - fee
    cash_path, pos_path = [cash], [qty]
    if sell:
        rows.append((1, 0, 0, -qty, PRICE, fee))
        cash += qty * PRICE - fee
        cash_path += [cash, cash]
        pos_path += [0.0, 0.0]
    else:
        cash_path += [cash, cash]
        pos_path += [qty, qty]
    cash_arr = np.array(cash_path) + cash_error
    positions = np.array(pos_path).reshape(-1, 1)
    equity = cash_arr + positions[:, 0] * np.array(marks)
    fills = pd.DataFrame(rows, columns=FILL_COLUMNS)
    return BacktestResult(engine, SESSIONS, ASSETS, equity, cash_arr, positions, fills)


MARKS = np.full((3, 1), PRICE)


def test_one_grid_step_with_its_own_money_passes():
    out = reconcile({"reference": result("reference", 298.147109),
                     "nautilus": result("nautilus", 298.147108)}, MARKS)
    assert out.ok, out.summary()


def test_an_open_one_step_position_is_valued_with_marks():
    # The price rises 100x while the one-step difference is held: equity differs by
    # 1e-6 x 9,900 = 0.0099, far above the 1e-9 money allowance unless marks explain it.
    rally = (PRICE, 10_000.0, 10_000.0)
    a = result("reference", 298.147109, sell=False, marks=rally)
    b = result("nautilus", 298.147108, sell=False, marks=rally)
    marks = np.array(rally).reshape(-1, 1)
    assert reconcile({"reference": a, "nautilus": b}, marks).ok
    assert not reconcile({"reference": a, "nautilus": b}).ok  # no prices: cannot explain it


def test_two_grid_steps_fail():
    out = reconcile({"reference": result("reference", 298.147110),
                     "nautilus": result("nautilus", 298.147108)}, MARKS)
    assert not out.ok
    assert "qty differs" in out.summary()


def test_a_flip_does_not_excuse_unexplained_cash():
    out = reconcile({"reference": result("reference", 298.147109),
                     "nautilus": result("nautilus", 298.147108, cash_error=0.01)}, MARKS)
    assert not out.ok
    assert "cash diverges" in out.summary()


def scaled_pair(a_steps, scaled=True):
    """s0: X flips by one step at $200, leaving a $0.0002 cash gap. s1: both engines buy A
    at $100 with buys scaled to fit cash; the reference buys `a_steps` grid steps more.
    Scale factor k = cash / buy cost B, so a gap G moves A by |qty| x G / B. G = the carried
    cash gap 0.0002 + one step of the session's one buy at $100 (B itself may differ by
    that) = 0.0003, so 300 x 0.0003 / 30,000 = 3e-6 (three steps); plus one step for a
    flip and one for the rounding-down of the scaled size -> up to five steps."""
    sessions = pd.bdate_range("2008-11-10", periods=3)
    assets = pd.Index(["X", "A"])
    out = {}
    for engine, x, a in (("reference", 10.000001, 300.0 + a_steps * 1e-6),
                         ("nautilus", 10.0, 300.0)):
        fills = pd.DataFrame([(0, 0, 0, x, 200.0, 0.0), (1, 0, 1, a, 100.0, 0.0)],
                             columns=FILL_COLUMNS)
        cash = np.array([CASH0 - 200 * x, CASH0 - 200 * x - 100 * a,
                         CASH0 - 200 * x - 100 * a])
        pos = np.array([[x, 0.0], [x, a], [x, a]])
        marks = np.array([[200.0, 100.0]] * 3)
        equity = cash + (pos * marks).sum(axis=1)
        out[engine] = BacktestResult(engine, sessions, assets, equity, cash, pos, fills,
                                     [1] if scaled else [])
    return out, np.array([[200.0, 100.0]] * 3)


def test_a_carried_cash_gap_moves_scaled_buys_in_proportion():
    results, marks = scaled_pair(5)
    assert reconcile(results, marks).ok, reconcile(results, marks).summary()


def test_more_than_the_carried_gap_explains_fails():
    results, marks = scaled_pair(6)
    out = reconcile(results, marks)
    assert not out.ok and "qty differs" in out.summary()


def test_without_scaling_a_cash_gap_explains_nothing():
    results, marks = scaled_pair(3, scaled=False)
    out = reconcile(results, marks)
    assert not out.ok and "qty differs" in out.summary()
