"""Hand-computed accounting cases. Every engine must hit these numbers exactly.

Expected values are worked out in the comments from first principles, independent of the
engine code. If an engine disagrees with a comment, the engine is wrong.
"""

import numpy as np
import pytest
from conftest import Fixed, long_frame, path

from concordance.costs import ZERO_COST, CostModel
from concordance.panel import PanelError, build_panel
from concordance.runner import ALL_ENGINES, run_engine
from concordance.strategy import weight_schedule

ENGINES = list(ALL_ENGINES)


def go(engine, rows, schedule, init_cash=1000.0, costs=ZERO_COST):
    panel = build_panel(long_frame(rows))
    w, has = weight_schedule(Fixed(schedule), panel)
    return panel, run_engine(engine, panel, (w, has), costs.fee_rates(panel), init_cash)


@pytest.mark.parametrize("engine", ENGINES)
def test_buy_and_hold_next_open(engine):
    # Decide at close of s0 -> buy at open of s1 at 10 -> 100 shares.
    rows = path("A", closes=[10, 10, 11, 12.1], opens=[10, 10, 11.5, 12])
    _, r = go(engine, rows, {0: {"A": 1.0}})
    assert r.fills[["session", "asset", "qty", "price", "fee"]].values.tolist() == [
        [1, 0, 100.0, 10.0, 0.0]]
    np.testing.assert_allclose(r.equity, [1000, 1000, 1100, 1210], rtol=0, atol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_dividend_goes_to_holder_not_ex_date_buyer(engine):
    # s2 is the ex-date: raw 100 -> 99 with a $2 dividend, ret = 101/100 - 1 = 1%.
    rows = path("A", closes=[100, 100, 99, 108.9], opens=[100, 100, 99, 99],
                rets=[np.nan, 0.0, 0.01, 0.1])
    # Holder bought at s1 open (10 sh @ 100): s2 = 10*99 + 10*$2 = 1010; s3 = 1010*1.1.
    _, r = go(engine, rows, {0: {"A": 1.0}})
    np.testing.assert_allclose(r.equity, [1000, 1000, 1010, 1111], atol=1e-9)
    # Buyer at the ex-date open (tr_open = 101*99/99 = 101): no dividend, flat on s2.
    _, r = go(engine, rows, {1: {"A": 1.0}}, init_cash=1010.0)
    assert r.fills["price"].tolist() == [101.0]
    np.testing.assert_allclose(r.equity, [1010, 1010, 1010, 1111], atol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_split_is_not_a_loss(engine):
    # 2:1 split at s2: raw 100 -> 50, ret = 50*2/100 - 1 = 0.
    rows = path("A", closes=[100, 100, 50, 55], opens=[100, 100, 50, 50],
                rets=[np.nan, 0.0, 0.0, 0.1])
    _, r = go(engine, rows, {0: {"A": 1.0}})
    np.testing.assert_allclose(r.equity, [1000, 1000, 1000, 1100], atol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_delisting_to_zero_is_kept_and_liquidated(engine):
    # A delists at s2 with a -100% return; B stays flat at 100 through s4.
    rows = (path("A", closes=[100, 100, 1], rets=[np.nan, 0.0, -1.0])
            + path("B", closes=[100] * 5))
    panel, r = go(engine, rows, {0: {"A": 0.5, "B": 0.5}})
    # s1: buy 5 A @100, 5 B @100. s2: A worth 0 -> 500. s3: A force-sold at the 1e-9 floor.
    assert panel.liquidate[3, 0]
    np.testing.assert_allclose(r.equity, [1000, 1000, 500, 500 + 5e-9, 500 + 5e-9], atol=1e-9)
    liq = r.fills[(r.fills.session == 3) & (r.fills.asset == 0)]
    assert liq["qty"].tolist() == [-5.0] and liq["fee"].tolist() == [0.0]
    assert r.positions[-1, 0] == 0.0


@pytest.mark.parametrize("engine", ENGINES)
def test_delisting_proceeds_reinvested_same_open(engine):
    # A delists at s2 with -30%; decision at s2: all into B.
    rows = (path("A", closes=[100, 100, 70], rets=[np.nan, 0.0, -0.3])
            + path("B", closes=[100] * 5))
    _, r = go(engine, rows, {0: {"A": 0.5, "B": 0.5}, 2: {"B": 1.0}},
              costs=CostModel(commission_bps=10, use_prior_quoted_spread=False,
                              default_half_spread_bps=0))
    # s1 buys are scaled by fees (2*5*100*1.001 > 1000), so hold whatever was bought.
    held_a, held_b = r.positions[2]
    # s3 open: A force-sold in full at its final value 70 with zero fee.
    s3 = r.fills[r.fills.session == 3].sort_values("asset")
    assert s3["qty"].tolist()[0] == -held_a and s3["price"].tolist()[0] == 70.0
    assert s3["fee"].tolist()[0] == 0.0
    # V_pre = cash + 70*held_a + 100*held_b; proceeds reinvested in B the same open.
    v_pre = r.cash[2] + 70 * held_a + 100 * held_b
    buy_b = s3["qty"].tolist()[1]
    assert buy_b > 0 and r.cash[3] >= 0
    # Everything that could be bought was bought: less than one fee-unit of cash remains.
    assert r.cash[3] < 0.002 * v_pre


@pytest.mark.parametrize("engine", ENGINES)
def test_fees_scale_buys_to_available_cash(engine):
    # 10 bps commission, weight 1 at 100: 10 sh would cost 1001 > 1000, so buys scale by
    # k = 1000/1001*(1-1e-9): qty = floor6(9.990009980...) = 9.990009.
    rows = path("A", closes=[100, 100, 100])
    _, r = go(engine, rows, {0: {"A": 1.0}},
              costs=CostModel(commission_bps=10, use_prior_quoted_spread=False,
                              default_half_spread_bps=0))
    assert r.fills["qty"].tolist() == [9.990009]
    fee = 9.990009 * 100 * 0.001
    assert r.fills["fee"].iloc[0] == pytest.approx(fee, abs=1e-12)
    cash = 1000 - 999.0009 - fee  # = 0.0000991
    np.testing.assert_allclose(r.cash[1], cash, atol=1e-9)
    np.testing.assert_allclose(r.equity[1], 999.0009 + cash, atol=1e-9)
    assert r.scaled_sessions == [1]


@pytest.mark.parametrize("engine", ENGINES)
def test_spread_cost_uses_prior_close_quote(engine):
    # Quote at s0 close: bid 99.9 / ask 100.1 -> half-spread 0.1/100 = 10 bps for the s1 buy.
    rows = path("A", closes=[100, 100, 100])
    for k, r in enumerate(rows):
        rows[k] = (*r, {"bid": 99.9 if k == 0 else 50.0, "ask": 100.1 if k == 0 else 150.0})
    panel = build_panel(long_frame(rows))
    rates = CostModel(commission_bps=0).fee_rates(panel)
    assert rates[1, 0] == pytest.approx(0.001)
    w, has = weight_schedule(Fixed({0: {"A": 0.5}}), panel)
    r = run_engine(engine, panel, (w, has), rates, 1000.0)
    assert r.fills["fee"].iloc[0] == pytest.approx(5 * 100 * 0.001, abs=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_halted_asset_keeps_position(engine):
    # A has no row on s2 (halt) and resumes s3 with ret measured from the s1 close.
    rows = ([(0, "A", 100, 100, np.nan), (1, "A", 100, 100, 0.0), (3, "A", 110, 110, 0.1)]
            + path("B", closes=[100] * 4))
    _, r = go(engine, rows, {0: {"A": 1.0}, 1: {"B": 1.0}})
    # s2: A untradable -> kept; buying B needs cash we don't have -> scaled to zero.
    assert r.positions[2, 0] == 10.0 and r.positions[2, 1] == 0.0
    np.testing.assert_allclose(r.equity, [1000, 1000, 1000, 1100], atol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_open_position_at_window_end_is_reported(engine):
    rows = path("A", closes=[10, 10, 10])
    _, r = go(engine, rows, {0: {"A": 1.0}})
    assert r.open_positions().to_dict() == {"A": 100.0}


def test_missing_return_mid_history_is_an_error():
    rows = path("A", closes=[10, 10, 10], rets=[np.nan, 0.0, np.nan])
    with pytest.raises(PanelError, match="missing total return"):
        build_panel(long_frame(rows))


def test_duplicates_and_bad_prices_are_errors():
    with pytest.raises(PanelError, match="duplicate"):
        build_panel(long_frame(path("A", [10, 10]) + path("A", [10, 10])))
    with pytest.raises(PanelError, match="non-positive close"):
        build_panel(long_frame([(0, "A", 10, -10, np.nan)]))
    with pytest.raises(PanelError, match="-100%"):
        build_panel(long_frame(path("A", [10, 1, 1, 1], rets=[np.nan, -1.0, 0, 0])))


def test_view_cannot_see_the_future():
    rows = path("A", closes=[10, 11, 12, 13])
    panel = build_panel(long_frame(rows))
    seen = []

    class Probe(Fixed):
        def decide(self, view):
            c = view["close"]
            seen.append((view.i, c.shape[0], c[-1, 0]))
            with pytest.raises(ValueError):
                c[0, 0] = 0.0  # read-only
            return None

    weight_schedule(Probe({}), panel)
    assert seen == [(0, 1, 10.0), (1, 2, 11.0), (2, 3, 12.0)]  # never sees s3


@pytest.mark.parametrize("engine", ENGINES)
def test_full_exit_leaves_exactly_zero(engine):
    # Price 29 everywhere, cash 1000, so value stays 1000. Targets on the 1e-6 grid:
    #   s1: 0.29*1000/29 = 10 sh; s2: 0.09*1000/29 = 3.103448 sh (sell 6.896552);
    #   s3: 0.06*1000/29 = 2.068966 sh (sell 1.034482).
    # In floats 10 - 6.896552 - 1.034482 = 2.0689660000000005, one ulp off the grid, so
    # rounding the exit (round(0 - pos)) would leave a 4e-16 crumb. s4 exits fully: the
    # position must be exactly 0, as it would be at a broker.
    rows = path("A", closes=[29.0] * 6, opens=[29.0] * 6)
    sched = {0: {"A": 0.29}, 1: {"A": 0.09}, 2: {"A": 0.06}, 3: {}}
    _, r = go(engine, rows, sched)
    assert r.fills["qty"].round(6).tolist() == [10.0, -6.896552, -1.034482, -2.068966]
    assert r.positions[4, 0] == 0.0 and r.positions[-1, 0] == 0.0
    np.testing.assert_allclose(r.equity, 1000.0, atol=1e-9)
