"""Hand-computed short-selling cases. Every engine must hit these numbers exactly.

Conventions (docs/SEMANTICS.md, "Short positions"): a short sale credits its proceeds to
cash; equity = cash + signed quantity x valuation price, so a short loses what the stock
gains including dividends; stock borrow is charged at each close on the short market value.
"""

import numpy as np
import pytest
from conftest import Fixed, long_frame, path

from concordance.costs import ZERO_COST, CostModel, exec_tables
from concordance.panel import build_panel
from concordance.planner import _capped
from concordance.result import AccountWipedOut
from concordance.runner import ALL_ENGINES, run_engine
from concordance.strategy import Orders, weight_schedule

ENGINES = list(ALL_ENGINES)
# borrow_bps giving exactly 0.001 per session: 0.001 * 252 * 1e4
BORROW_0P1PCT = CostModel(commission_bps=0.0, use_prior_quoted_spread=False,
                          default_half_spread_bps=0.0, borrow_bps=0.001 * 252 * 1e4)


def go(engine, rows, schedule, init_cash=1000.0, costs=ZERO_COST):
    panel = build_panel(long_frame(rows))
    w, has = weight_schedule(Fixed(schedule), panel)
    return panel, run_engine(engine, panel, (w, has), exec_tables(panel, costs), init_cash)


@pytest.mark.parametrize("engine", ENGINES)
def test_short_profits_from_a_fall(engine):
    # s1 open: sell 100 @ 10 -> cash 2000, equity 1000. s2: 2000 - 100*8 = 1200.
    # Decide flat at s2 -> s3 open: buy back 100 @ 8 -> cash 1200, equity 1200.
    rows = path("A", closes=[10, 10, 8, 8], opens=[10, 10, 9, 8])
    _, r = go(engine, rows, {0: {"A": -1.0}, 2: {"A": 0.0}})
    assert r.fills[["session", "qty", "price"]].values.tolist() == [[1, -100.0, 10.0],
                                                                    [3, 100.0, 8.0]]
    np.testing.assert_allclose(r.cash, [1000, 2000, 2000, 1200], atol=1e-9)
    np.testing.assert_allclose(r.equity, [1000, 1000, 1200, 1200], atol=1e-9)
    assert r.positions[:, 0].tolist() == [0.0, -100.0, -100.0, 0.0]


@pytest.mark.parametrize("engine", ENGINES)
def test_borrow_is_charged_on_short_value_at_each_close(engine):
    # Same trades at 0.1% per session: s1 0.001 * 100 * 10 = 1.0, s2 0.001 * 100 * 8 = 0.8.
    # s3 covers at the open, so nothing is short at its close and no fee is due.
    rows = path("A", closes=[10, 10, 8, 8], opens=[10, 10, 9, 8])
    _, r = go(engine, rows, {0: {"A": -1.0}, 2: {"A": 0.0}}, costs=BORROW_0P1PCT)
    np.testing.assert_allclose(r.borrow, [0, 1.0, 0.8, 0], atol=1e-9)
    np.testing.assert_allclose(r.cash, [1000, 1999, 1998.2, 1198.2], atol=1e-9)
    np.testing.assert_allclose(r.equity, [1000, 999, 1198.2, 1198.2], atol=1e-9)
    # s2 sizing used equity net of the s1 fee: the cover is still exactly the 100 held.
    assert r.fills["qty"].tolist() == [-100.0, 100.0]


@pytest.mark.parametrize("engine", ENGINES)
def test_short_pays_the_dividend(engine):
    # s2 is the ex-date: raw 100 -> 99 with a $2 dividend (ret = 1%), tr_close 100 -> 101.
    # Short 10 @ 100 at s1: s2 equity = 2000 - 10 * 101 = 990 (+$10 price, -$20 dividend).
    rows = path("A", closes=[100, 100, 99, 99], opens=[100, 100, 99, 99],
                rets=[np.nan, 0.0, 0.01, 0.0])
    _, r = go(engine, rows, {0: {"A": -1.0}})
    np.testing.assert_allclose(r.equity, [1000, 1000, 990, 990], atol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_market_neutral_long_short(engine):
    # Net 0, gross 2. s1 open: short 50 B @ 20 first (cash 2000), then buy 100 A @ 10 (1000).
    # s2: 1000 + 100*11 - 50*22 = 1000. s3: 1000 + 100*12 - 50*20 = 1200.
    rows = path("A", closes=[10, 10, 11, 12]) + path("B", closes=[20, 20, 22, 20])
    _, r = go(engine, rows, {0: {"A": 1.0, "B": -1.0}})
    f = r.fills
    assert f[["asset", "qty", "price"]].values.tolist() == [[1, -50.0, 20.0], [0, 100.0, 10.0]]
    np.testing.assert_allclose(r.cash, [1000, 1000, 1000, 1000], atol=1e-9)
    np.testing.assert_allclose(r.equity, [1000, 1000, 1000, 1200], atol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_long_to_short_is_one_order(engine):
    # s1: buy 100 @ 10. Decide -0.5 at s1 -> s2 open @ 10, equity 1000 -> target -50,
    # one sell of 150 that crosses through zero.
    rows = path("A", closes=[10, 10, 10, 10])
    _, r = go(engine, rows, {0: {"A": 1.0}, 1: {"A": -0.5}})
    assert r.fills[["session", "qty"]].values.tolist() == [[1, 100.0], [2, -150.0]]
    np.testing.assert_allclose(r.cash, [1000, 0, 1500, 1500], atol=1e-9)
    np.testing.assert_allclose(r.equity, [1000] * 4, atol=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_short_held_through_a_delisting_is_closed_at_the_final_value(engine):
    # A delists at s2 with -100%: the short keeps all 1000 of proceeds -> equity 2000.
    # s3: force-bought back at the 1e-9 floor, no fee.
    rows = path("A", closes=[100, 100, 1], rets=[np.nan, 0.0, -1.0]) + path("B", closes=[1] * 5)
    panel, r = go(engine, rows, {0: {"A": -1.0}})
    assert panel.liquidate[3, 0]
    np.testing.assert_allclose(r.equity, [1000, 1000, 2000, 2000 - 1e-8, 2000 - 1e-8],
                               atol=1e-9)
    assert r.positions[-1, 0] == 0.0


@pytest.mark.parametrize("engine", ENGINES)
def test_a_short_squeeze_past_zero_equity_stops_the_run(engine):
    # Short 100 @ 10 with 1000: at 25 equity is 2000 - 2500 < 0. No engine may carry on.
    rows = path("A", closes=[10, 10, 25, 25])
    with pytest.raises(AccountWipedOut):
        go(engine, rows, {0: {"A": -1.0}})


def test_net_and_gross_limits():
    rows = path("A", closes=[10] * 4) + path("B", closes=[10] * 4)
    panel = build_panel(long_frame(rows))
    with pytest.raises(ValueError, match="net weight"):
        weight_schedule(Fixed({0: {"A": 1.0, "B": 0.1}}), panel)
    with pytest.raises(ValueError, match="gross weight"):
        weight_schedule(Fixed({0: {"A": 0.9, "B": -1.2}}), panel)
    w, _ = weight_schedule(Fixed({0: {"A": 1.0, "B": -1.0}}), panel)  # 100/100 is allowed
    assert w[1].tolist() == [1.0, -1.0]


def test_exit_orders_on_a_short_are_rejected():
    rows = path("A", closes=[10] * 4)
    panel = build_panel(long_frame(rows))
    with pytest.raises(ValueError, match="long positions only"):
        weight_schedule(Fixed({0: Orders(weights={"A": -0.5}, stop_loss_pct={"A": 0.1})}),
                        panel)
    with pytest.raises(ValueError, match="entry_limit"):
        weight_schedule(Fixed({0: Orders(weights={"A": -0.5}, entry_limit={"A": 9.5})}),
                        panel)


@pytest.mark.parametrize("desired,pos,cap_buy,cap_sell,expected", [
    (100.0, -100.0, 0.0, np.inf, 100.0),  # covering is never blocked by unknown volume
    (-50.0, 0.0, 0.0, np.inf, 0.0),  # opening a short needs known volume, like a buy
    (-150.0, 100.0, 0.0, np.inf, -100.0),  # long -> short: only the exit part goes
    (-150.0, 100.0, 60.0, 60.0, -60.0),  # known cap applies once to the whole trade
    (-150.0, 100.0, 120.0, 120.0, -120.0),
    (30.0, -100.0, 60.0, 60.0, 30.0),
    (100.0, 0.0, 60.0, 60.0, 60.0),  # long-only cases are unchanged
    (-100.0, 100.0, 0.0, np.inf, -100.0),
])
def test_volume_cap_for_shorts(desired, pos, cap_buy, cap_sell, expected):
    assert _capped.py_func(desired, pos, cap_buy, cap_sell) == expected


@pytest.mark.parametrize("engine", ["reference"])
def test_leg_attribution_by_hand(engine):
    from concordance.reconcile import valuation_marks
    from concordance.report import leg_attribution

    # Market-neutral case above: long 100 A 10 -> 12 = +200; short 50 B 20 -> 20 = 0.
    rows = path("A", closes=[10, 10, 11, 12]) + path("B", closes=[20, 20, 22, 20])
    panel, r = go(engine, rows, {0: {"A": 1.0, "B": -1.0}})
    legs = leg_attribution(r, valuation_marks(panel.tr_close))
    assert legs["long_pnl"] == pytest.approx(200, abs=1e-9)
    assert legs["short_pnl"] == pytest.approx(0, abs=1e-9)
    # Exposure at the closes: s0 flat; s1 (1000+1000)/1000; s2 (1100+1100)/1000;
    # s3 (1200+1000)/1200. Net: 0, 0, 0, 200/1200.
    assert legs["avg_gross_exposure"] == pytest.approx((0 + 2 + 2.2 + 2200 / 1200) / 4)
    assert legs["avg_net_exposure"] == pytest.approx((200 / 1200) / 4)
    # Short-only fall case with borrow: short P&L +200, borrow 1.8 -> equity +198.2.
    rows = path("A", closes=[10, 10, 8, 8], opens=[10, 10, 9, 8])
    panel, r = go(engine, rows, {0: {"A": -1.0}, 2: {"A": 0.0}}, costs=BORROW_0P1PCT)
    legs = leg_attribution(r, valuation_marks(panel.tr_close))
    assert legs["short_pnl"] == pytest.approx(200, abs=1e-9)
    assert legs["total_borrow"] == pytest.approx(1.8, abs=1e-9)


def test_wiped_out_account_is_a_strategy_outcome(tmp_path):
    from concordance.api import run_backtest

    rows = path("A", closes=[10, 10, 25, 25])
    panel = build_panel(long_frame(rows))
    src = tmp_path / "s.py"
    src.write_text("from concordance.strategy import Strategy\n"
                   "class S(Strategy):\n"
                   "    def decide(self, view):\n"
                   "        return {'A': -1.0} if view.i == 0 else None\n"
                   "STRATEGY = S()\n")
    out = run_backtest({"strategy": {"path": str(src)}, "initial_capital": 1000.0,
                        "liquidity": None, "validation": {"lookahead_cuts": 0},
                        "data": {"start": str(panel.sessions[0].date()),
                                 "end": str(panel.sessions[-1].date())}},
                       tmp_path / "out", panel=panel)
    assert out["status"] == "strategy_error"
    assert "AccountWipedOut" in out["diagnostics"]["messages"][0]
