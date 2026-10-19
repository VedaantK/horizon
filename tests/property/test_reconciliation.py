"""All engines must agree on randomized markets (fills, positions, cash, equity)."""

import numpy as np
import pytest
from random_market import RandomLongShort, RandomOrders, RandomRebalancer, random_long_frame

from concordance.costs import ZERO_COST, CostModel, LiquidityModel
from concordance.panel import build_panel
from concordance.runner import run

COSTS = [ZERO_COST, CostModel(commission_bps=2.0), CostModel(commission_bps=25.0,
                                                               default_half_spread_bps=40)]


@pytest.mark.parametrize("seed", range(40))
def test_engines_agree_on_random_markets(seed):
    panel = build_panel(random_long_frame(seed))
    costs = COSTS[seed % len(COSTS)]
    out = run(panel, RandomRebalancer(seed), costs=costs, init_cash=100_000.0, strict=False)
    assert out.report.ok, out.report.summary()
    ref = out.results["reference"]
    assert np.isfinite(ref.equity).all()
    assert (ref.cash >= -1e-9).all()


@pytest.mark.parametrize("seed", range(40))
def test_engines_agree_with_orders_and_liquidity(seed):
    """Stops, take-profits (absolute and %), limit entries, and on odd seeds the volume cap
    with carry-over and market impact. Nautilus decides triggers with its own matching."""
    panel = build_panel(random_long_frame(seed))
    liq = LiquidityModel(participation=0.05, window=5, min_obs=3) if seed % 2 else None
    out = run(panel, RandomOrders(seed), costs=COSTS[seed % len(COSTS)], init_cash=1e6,
              strict=False, liquidity=liq)
    assert out.report.ok, out.report.summary()
    ref = out.results["reference"]
    assert (ref.cash >= -1e-9).all()
    # An engine never holds what the schedule says it closed.
    f = ref.fills
    exits = f[f["phase"] == 1]
    s = out.schedule
    for i, j in zip(exits["session"], exits["asset"], strict=True):
        if np.isfinite(s.exit_price[i, j]):
            assert ref.positions[i, j] == 0.0


SHORT_COSTS = [CostModel(borrow_bps=300.0), ZERO_COST,
               CostModel(commission_bps=25.0, default_half_spread_bps=40, borrow_bps=50.0)]


@pytest.mark.parametrize("seed", range(40))
def test_engines_agree_long_short(seed):
    """Shorts, flips through zero and long/short books, with stock borrow; on odd seeds the
    volume cap with carry-over and market impact."""
    panel = build_panel(random_long_frame(seed))
    liq = LiquidityModel(participation=0.05, window=5, min_obs=3) if seed % 2 else None
    out = run(panel, RandomLongShort(seed), costs=SHORT_COSTS[seed % 3], init_cash=1e6,
              strict=False, liquidity=liq)
    assert out.report.ok, out.report.summary()
    ref = out.results["reference"]
    assert np.isfinite(ref.equity).all() and (ref.equity > 0).all()
    assert (ref.positions < 0).any()
    if seed % 3 != 1:
        assert ref.borrow.sum() > 0


def test_order_markets_exercise_the_new_cases():
    """Guard against a generator that stops producing gaps, triggers, limits and caps."""
    from concordance.strategy import build_schedule

    gaps = exits = limits = carried = 0
    for seed in range(40):
        panel = build_panel(random_long_frame(seed))
        s = build_schedule(RandomOrders(seed), panel)
        gaps += int(s.gap_exit.sum())
        exits += int(np.isfinite(s.exit_price).sum())
        limits += int(s.limit_filled.sum())
        if seed % 2:
            liq = LiquidityModel(participation=0.05, window=5, min_obs=3)
            r = run(panel, RandomOrders(seed), init_cash=1e6, engines=("reference",),
                    liquidity=liq).results["reference"]
            # carried quantity shows up as an open fill on a session without new targets
            f = r.fills[r.fills["phase"] == 0]
            carried += int((~s.has[f["session"].to_numpy()]
                            & ~panel.liquidate[f["session"], f["asset"]]
                            & ~s.gap_exit[f["session"], f["asset"]]).sum())
    assert gaps > 20 and exits > 100 and limits > 10 and carried > 10


def test_random_markets_exercise_the_hard_cases():
    """Guard against a generator that silently stops producing the edge cases."""
    liq = halts = no_open = full_loss = 0
    for seed in range(40):
        df = random_long_frame(seed)
        p = build_panel(df)
        liq += int(p.liquidate.sum())
        no_open += int((df["open"].isna()).sum())
        full_loss += int((df["ret"] == -1.0).sum())
        has = np.isfinite(p.tr_close)
        for j in range(has.shape[1]):
            idx = np.flatnonzero(has[:, j])
            halts += int((np.diff(idx) > 1).sum())
    assert liq > 10 and halts > 10 and no_open > 5 and full_loss >= 2


@pytest.mark.parametrize("seed", range(10))
def test_compiled_reference_matches_pure_python(seed):
    """The placebo's fast path (numba planner) is the same accounting as the audited loop."""
    from concordance.costs import exec_tables
    from concordance.engines import reference
    from concordance.strategy import build_schedule

    panel = build_panel(random_long_frame(seed))
    sched = build_schedule(RandomRebalancer(seed), panel)
    tables = exec_tables(panel, COSTS[seed % len(COSTS)])
    a = reference.run(panel, sched, tables, 100_000.0)
    b = reference.run(panel, sched, tables, 100_000.0, compiled=True)
    assert np.array_equal(a.equity, b.equity) and a.fills.equals(b.fills)


@pytest.mark.parametrize("seed", range(0, 40, 4))
def test_leg_attribution_adds_up_to_equity(seed):
    from concordance.reconcile import valuation_marks
    from concordance.report import leg_attribution

    panel = build_panel(random_long_frame(seed))
    r = run(panel, RandomLongShort(seed), costs=SHORT_COSTS[seed % 3], init_cash=1e6,
            engines=("reference",)).results["reference"]
    for start in (0, 5):
        legs = leg_attribution(r, valuation_marks(panel.tr_close), start)
        total = legs["long_pnl"] + legs["short_pnl"] - legs["fees_after_start"] \
            - legs["total_borrow"]
        assert total == pytest.approx(r.equity[-1] - r.equity[start], abs=1e-6)
