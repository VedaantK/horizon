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
