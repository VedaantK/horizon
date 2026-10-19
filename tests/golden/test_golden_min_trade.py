"""Hand-computed cases for the minimum trade size (`CostModel.min_trade_bps`).

A rebalance trade worth less than min_trade_bps of the pre-trade value V_pre is skipped;
a full exit always trades. Expected values are worked out in the comments.
"""

from dataclasses import replace

import numpy as np
import pytest
from conftest import Fixed, long_frame, path

from concordance.costs import ZERO_COST, exec_tables
from concordance.panel import build_panel
from concordance.runner import ALL_ENGINES, run_engine
from concordance.strategy import build_schedule

ENGINES = list(ALL_ENGINES)


def go(engine, rows, schedule, min_trade_bps, init_cash=1000.0):
    panel = build_panel(long_frame(rows))
    sched = build_schedule(Fixed(schedule), panel)
    tables = exec_tables(panel, replace(ZERO_COST, min_trade_bps=min_trade_bps))
    return run_engine(engine, panel, sched, tables, init_cash)


def fills(r):
    return [[int(s), int(a), round(q, 6), p]
            for s, a, q, p in r.fills[["session", "asset", "qty", "price"]].values.tolist()]


# A rises 100 -> 102 at the s2 open; B stays at 100. Both are 50/50 again from s1.
DRIFT = path("A", closes=[100, 100, 102, 102]) + path("B", closes=[100] * 4)


@pytest.mark.parametrize("engine", ENGINES)
def test_small_rebalances_are_skipped(engine):
    # s1: V = 1000 -> 5 A @100 and 5 B @100. s2 open: V = 5*102 + 5*100 = 1010.
    # A: target round6(505/102) = 4.950980, trade -0.049020 sh = $5.00 = 0.495% of V.
    # B: target 505/100 = 5.05, trade +0.05 sh = $5.00. Both below 1% (100 bps): no fills.
    r = go(engine, DRIFT, {0: {"A": 0.5, "B": 0.5}, 1: {"A": 0.5, "B": 0.5}}, 100)
    assert fills(r) == [[1, 0, 5.0, 100.0], [1, 1, 5.0, 100.0]]
    np.testing.assert_allclose(r.equity, [1000, 1000, 1010, 1010], rtol=0, atol=1e-9)
    np.testing.assert_allclose(r.positions[-1], [5.0, 5.0], rtol=0, atol=0)


@pytest.mark.parametrize("engine", ENGINES)
def test_trades_above_the_threshold_still_happen(engine):
    # Same path at 10 bps ($1.01): both trades go through. Sells first: A -0.04902 @102
    # = +5.00004 cash, then B +0.05 @100 = -5. Cash 0.00004; equity 0.00004 +
    # 4.950980*102 (504.99996) + 5.05*100 (505) = 1010: a rebalance moves no value.
    r = go(engine, DRIFT, {0: {"A": 0.5, "B": 0.5}, 1: {"A": 0.5, "B": 0.5}}, 10)
    assert fills(r) == [[1, 0, 5.0, 100.0], [1, 1, 5.0, 100.0],
                        [2, 0, -0.04902, 102.0], [2, 1, 0.05, 100.0]]
    np.testing.assert_allclose(r.cash[-1], 0.00004, rtol=0, atol=1e-9)
    np.testing.assert_allclose(r.equity[-1], 1010.0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_tiny_new_position_is_not_opened(engine):
    # s1: V = 1000. A: 995/100 = 9.95 sh ($995). B: 0.05 sh = $5 < $10 (100 bps): skipped.
    rows = path("A", closes=[100] * 3) + path("B", closes=[100] * 3)
    r = go(engine, rows, {0: {"A": 0.995, "B": 0.005}}, 100)
    assert fills(r) == [[1, 0, 9.95, 100.0]]
    np.testing.assert_allclose(r.cash[-1], 5.0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_full_exit_trades_however_small(engine):
    # s1: 5 A @100, 5 B @100. B opens and closes at 1 from s2: equity 500 + 5 = 505.
    # Decision at s2: A 50%, B out. s3 open: V = 505, threshold 1% = $5.05.
    # B: full exit -5 sh @1 = $5 < $5.05, but a full exit always trades.
    # A: target 252.5/100 = 2.525, trade -2.475 sh ($247.50). Cash 0 + 5 + 247.5 = 252.5.
    rows = path("A", closes=[100] * 4) + path("B", closes=[100, 100, 1, 1])
    r = go(engine, rows, {0: {"A": 0.5, "B": 0.5}, 2: {"A": 0.5}}, 100)
    assert fills(r) == [[1, 0, 5.0, 100.0], [1, 1, 5.0, 100.0],
                        [3, 0, -2.475, 100.0], [3, 1, -5.0, 1.0]]
    np.testing.assert_allclose(r.cash[-1], 252.5, rtol=0, atol=1e-9)
    np.testing.assert_allclose(r.positions[-1], [2.525, 0.0], rtol=0, atol=0)
