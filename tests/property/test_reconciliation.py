"""All engines must agree on randomized markets (fills, positions, cash, equity)."""

import numpy as np
import pytest
from random_market import RandomRebalancer, random_long_frame

from concordance.costs import ZERO_COST, CostModel
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
    from concordance.engines import reference
    from concordance.strategy import weight_schedule

    panel = build_panel(random_long_frame(seed))
    w, has = weight_schedule(RandomRebalancer(seed), panel)
    fee = COSTS[seed % len(COSTS)].fee_rates(panel)
    a = reference.run(panel, w, has, fee, 100_000.0)
    b = reference.run(panel, w, has, fee, 100_000.0, compiled=True)
    assert np.array_equal(a.equity, b.equity) and a.fills.equals(b.fills)
