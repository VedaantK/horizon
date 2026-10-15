"""Hand-computed cases for stop-loss, take-profit and limit-entry orders, the volume cap with
carry-over, and market impact. Every engine must hit these numbers exactly.

Expected values are worked out in the comments from first principles (docs/SEMANTICS.md),
independent of the engine code. All paths start at a raw close of 100, so the total-return
scale equals the raw scale until a split or dividend says otherwise.
"""

import math

import numpy as np
import pandas as pd
import pytest
from conftest import DATES

from concordance.costs import ZERO_COST, LiquidityModel, exec_tables
from concordance.panel import build_panel
from concordance.runner import ALL_ENGINES, run_engine
from concordance.strategy import STOP, TAKE_PROFIT, Orders, Strategy, build_schedule

ENGINES = list(ALL_ENGINES)
EQ = dict(rtol=0, atol=1e-9)


def bars(asset, ohlc, rets=None, volume=None):
    """Rows from (open, high, low, close) tuples; ret = close/prev_close - 1 unless given.
    An open of None means no opening print that session (untradable)."""
    out = []
    for k, (o, h, lo, c) in enumerate(ohlc):
        r = rets[k] if rets is not None else (np.nan if k == 0 else c / ohlc[k - 1][3] - 1)
        out.append({"date": DATES[k], "asset": asset, "open": np.nan if o is None else o,
                    "high": h, "low": lo, "close": c, "ret": r,
                    "volume": volume[k] if volume is not None else 1e9})
    return out


class Script(Strategy):
    """Returns the given decision at given sessions and records the position state seen."""

    def __init__(self, decisions):
        self.decisions = decisions
        self.seen = {}

    def decide(self, view):
        self.seen[view.i] = view.position
        return self.decisions.get(view.i)


def go(engine, rows, decisions, init_cash=1000.0, liquidity=None):
    panel = build_panel(pd.DataFrame(rows))
    strat = Script(decisions)
    sched = build_schedule(strat, panel)
    tables = exec_tables(panel, ZERO_COST, liquidity)
    return run_engine(engine, panel, sched, tables, init_cash), sched, strat


def fills(r):
    return r.fills[["session", "phase", "asset", "qty", "price", "fee"]].values.tolist()


FLAT = (100, 100, 100, 100)


@pytest.mark.parametrize("engine", ENGINES)
def test_stop_inside_the_range_fills_at_the_stop(engine):
    # s0 decide: all in A, stop 95. s1 buy 10 @100 (low 99 > 95, no trigger).
    # s2 opens 100 > 95, low 94 <= 95: sell 10 @95 intraday -> cash 950.
    rows = bars("A", [FLAT, (100, 101, 99, 100), (100, 101, 94, 96), (96, 96, 96, 96)])
    r, s, strat = go(engine, rows, {0: Orders(weights={"A": 1.0}, stop_loss={"A": 95})})
    assert fills(r) == [[1, 0, 0, 10.0, 100.0, 0.0], [2, 1, 0, -10.0, 95.0, 0.0]]
    np.testing.assert_allclose(r.equity, [1000, 1000, 950, 950], **EQ)
    # The strategy sees the exit at the close of s2 and holds nothing afterwards.
    st = strat.seen[2]
    assert st.exit_today[0] == STOP and not st.held[0] and np.isnan(st.stop_level[0])
    assert strat.seen[1].held[0] and strat.seen[1].stop_level[0] == 95.0


@pytest.mark.parametrize("engine", ENGINES)
def test_gap_through_the_stop_sells_at_the_open(engine):
    # s2 opens at 93, already below the 95 stop: sell 10 @93 at the open (phase 0) -> 930.
    rows = bars("A", [FLAT, (100, 101, 99, 100), (93, 94, 92, 94), (94, 94, 94, 94)])
    r, s, _ = go(engine, rows, {0: Orders(weights={"A": 1.0}, stop_loss={"A": 95})})
    assert fills(r) == [[1, 0, 0, 10.0, 100.0, 0.0], [2, 0, 0, -10.0, 93.0, 0.0]]
    assert s.gap_exit[2, 0] and s.exit_kind[2, 0] == STOP
    np.testing.assert_allclose(r.equity, [1000, 1000, 930, 930], **EQ)


@pytest.mark.parametrize("engine", ENGINES)
def test_take_profit_pct_from_the_entry_price(engine):
    # Entry at the s1 open of 100 -> take profit at 100 * 1.05 = 105.
    # s1 high 104 < 105. s2 high 106 >= 105: sell 10 @105 -> 1050.
    rows = bars("A", [FLAT, (100, 104, 99, 102), (102, 106, 101, 103), (103,) * 4])
    r, _, strat = go(engine, rows, {0: Orders(weights={"A": 1.0},
                                              take_profit_pct={"A": 0.05})})
    assert fills(r) == [[1, 0, 0, 10.0, 100.0, 0.0], [2, 1, 0, -10.0, 105.0, 0.0]]
    np.testing.assert_allclose(r.equity, [1000, 1020, 1050, 1050], **EQ)
    assert strat.seen[1].entry_price[0] == 100.0 and strat.seen[1].entry_session[0] == 1
    assert strat.seen[2].exit_today[0] == TAKE_PROFIT


@pytest.mark.parametrize("engine", ENGINES)
def test_stop_wins_when_both_levels_are_touched(engine):
    # The documented path is open -> low -> high -> close, so with stop 95 and take
    # profit 105 both inside s2's range (94..106), the stop fills first: 10 @95 -> 950.
    rows = bars("A", [FLAT, FLAT, (100, 106, 94, 100), FLAT])
    r, _, _ = go(engine, rows, {0: Orders(weights={"A": 1.0}, stop_loss={"A": 95},
                                          take_profit={"A": 105})})
    assert fills(r) == [[1, 0, 0, 10.0, 100.0, 0.0], [2, 1, 0, -10.0, 95.0, 0.0]]
    np.testing.assert_allclose(r.equity, [1000, 1000, 950, 950], **EQ)


@pytest.mark.parametrize("engine", ENGINES)
def test_exact_touch_triggers(engine):
    # A low exactly at the stop counts as reached: 10 @95.
    rows = bars("A", [FLAT, FLAT, (100, 100, 95, 97), FLAT])
    r, _, _ = go(engine, rows, {0: Orders(weights={"A": 1.0}, stop_loss={"A": 95})})
    assert fills(r)[-1] == [2, 1, 0, -10.0, 95.0, 0.0]


@pytest.mark.parametrize("engine", ENGINES)
def test_limit_entry_fills_at_the_limit(engine):
    # s0: target 100% of A, but only at 97 or better. s1 opens 100 (> 97), low 96 <= 97:
    # buy round6(1000 / 97) = round6(10.3092783505) = 10.309278 @97 intraday.
    # Cost 10.309278 * 97 = 999.999966 -> cash 0.000034.
    # Close s1 at 100: 0.000034 + 1030.9278 = 1030.927834. s2 close 101: 0.000034 + 1041.237078.
    rows = bars("A", [FLAT, (100, 101, 96, 100), (100, 101, 99, 101), (101,) * 4])
    r, _, strat = go(engine, rows, {0: Orders(weights={"A": 1.0}, entry_limit={"A": 97})})
    assert fills(r) == [[1, 1, 0, 10.309278, 97.0, 0.0]]
    np.testing.assert_allclose(r.equity, [1000, 1030.927834, 1041.237112, 1041.237112],
                               rtol=0, atol=1e-6)
    assert strat.seen[1].entry_price[0] == 97.0


@pytest.mark.parametrize("engine", ENGINES)
def test_limit_entry_not_reached_is_cancelled(engine):
    # s1 low 98 > 97: no fill. It is a day order, so s2's low of 90 buys nothing.
    rows = bars("A", [FLAT, (100, 101, 98, 100), (100, 100, 90, 95), FLAT])
    r, _, _ = go(engine, rows, {0: Orders(weights={"A": 1.0}, entry_limit={"A": 97})})
    assert fills(r) == []
    np.testing.assert_allclose(r.equity, [1000] * 4, **EQ)


@pytest.mark.parametrize("engine", ENGINES)
def test_marketable_limit_buys_at_the_open(engine):
    # Limit 101 with an open of 100: already marketable, so a normal open buy of 10 @100.
    rows = bars("A", [FLAT, FLAT, FLAT])
    r, s, _ = go(engine, rows, {0: Orders(weights={"A": 1.0}, entry_limit={"A": 101})})
    assert fills(r) == [[1, 0, 0, 10.0, 100.0, 0.0]]
    assert np.isnan(s.entry_limit[1, 0])


@pytest.mark.parametrize("engine", ENGINES)
def test_stop_on_a_limit_entry_starts_next_session(engine):
    # Limit 97 fills on s1 (low 94): 10.309278 @97, cash 0.000034. Its 2% stop is at
    # 97 * 0.98 = 95.06, active from s2 only, so s1's low of 94 does not trigger it.
    # s2 low 95 <= 95.06: sell 10.309278 @95.06 = 979.99996668 -> cash 980.00000068.
    rows = bars("A", [FLAT, (100, 100, 94, 99), (99, 99, 95, 96), (96,) * 4])
    r, _, _ = go(engine, rows, {0: Orders(weights={"A": 1.0}, entry_limit={"A": 97},
                                          stop_loss_pct={"A": 0.02})})
    assert fills(r) == [[1, 1, 0, 10.309278, 97.0, 0.0], [2, 1, 0, -10.309278, 95.06, 0.0]]
    np.testing.assert_allclose(r.equity[2:], [980.00000068] * 2, rtol=0, atol=1e-8)


@pytest.mark.parametrize("engine", ENGINES)
def test_split_does_not_trigger_a_stop(engine):
    # 2:1 split on s2: raw prices halve, ret = 0. The raw stop of 95 set at s0 lives on the
    # total-return scale (95); s2's raw low of 49 is 98 on that scale, so nothing fires.
    # At the close of s2 the strategy sees the stop on today's raw scale: 95 / 2 = 47.5.
    rows = bars("A", [FLAT, FLAT, (50, 50.5, 49, 50), (50,) * 4], rets=[np.nan, 0, 0, 0])
    r, _, strat = go(engine, rows, {0: Orders(weights={"A": 1.0}, stop_loss={"A": 95})})
    assert fills(r) == [[1, 0, 0, 10.0, 100.0, 0.0]]
    np.testing.assert_allclose(r.equity, [1000] * 4, **EQ)
    assert strat.seen[2].stop_level[0] == pytest.approx(47.5, abs=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_no_trigger_on_a_session_without_an_open(engine):
    # s2 has no opening print: untradable, so its low of 90 cannot trigger the 95 stop.
    # s3 (low 96) does not either, and A is still held at the end.
    rows = bars("A", [FLAT, FLAT, (None, 100, 90, 99), (99, 99, 96, 98)])
    r, s, _ = go(engine, rows, {0: Orders(weights={"A": 1.0}, stop_loss={"A": 95})})
    assert fills(r) == [[1, 0, 0, 10.0, 100.0, 0.0]]
    np.testing.assert_allclose(r.equity, [1000, 1000, 990, 980], **EQ)


@pytest.mark.parametrize("engine", ENGINES)
def test_volume_cap_carries_the_rest_to_later_opens(engine):
    # Volume 30 shares every session, participation 10%, ADV over the 2 prior sessions with
    # both required. s1's ADV has only one prior session -> unknown -> buys blocked; the
    # whole target of 10 shares is carried. From s2 the cap is 0.1 * 30 = 3 shares:
    # s2 +3 (7 left), s3 +3 (4 left), s4 +3 (1 left), s5 +1.
    rows = bars("A", [FLAT] * 7, volume=[30] * 7)
    liq = LiquidityModel(participation=0.1, impact_coef=0.0, max_impact_bps=0.0, window=2,
                         min_obs=2)
    r, _, _ = go(engine, rows, {0: {"A": 1.0}}, liquidity=liq)
    assert fills(r) == [[2, 0, 0, 3.0, 100.0, 0.0], [3, 0, 0, 3.0, 100.0, 0.0],
                        [4, 0, 0, 3.0, 100.0, 0.0], [5, 0, 0, 1.0, 100.0, 0.0]]
    assert r.positions[-1, 0] == 10.0


@pytest.mark.parametrize("engine", ENGINES)
def test_new_targets_replace_carried_quantity(engine):
    # As above, but at the close of s2 the target drops to 0. s2 bought 3 (7 were still to
    # come); s3's new target sells the 3 held (cap 3) and nothing else is bought later.
    rows = bars("A", [FLAT] * 7, volume=[30] * 7)
    liq = LiquidityModel(participation=0.1, impact_coef=0.0, max_impact_bps=0.0, window=2,
                         min_obs=2)
    r, _, _ = go(engine, rows, {0: {"A": 1.0}, 2: {"A": 0.0}}, liquidity=liq)
    assert fills(r) == [[2, 0, 0, 3.0, 100.0, 0.0], [3, 0, 0, -3.0, 100.0, 0.0]]


@pytest.mark.parametrize("engine", ENGINES)
def test_stop_exit_is_not_capped(engine):
    # Volume 1e6 on s0, then 30. Window 1, participation 1%:
    #   s1 cap = 0.01 * 1e6 = 10,000 shares -> the 10-share buy @100 fits.
    #   s3 cap = 0.01 * 30 = 0.3 shares, but an exit order is never capped:
    #   s3 opens 100 > 95, low 94 -> sell all 10 @95.
    rows = bars("A", [FLAT, FLAT, FLAT, (100, 100, 94, 96), FLAT],
                volume=[1e6, 30, 30, 30, 30])
    liq = LiquidityModel(participation=0.01, impact_coef=0.0, max_impact_bps=0.0, window=1,
                         min_obs=1)
    r, _, _ = go(engine, rows, {0: Orders(weights={"A": 1.0}, stop_loss={"A": 95})},
                 liquidity=liq)
    assert fills(r) == [[1, 0, 0, 10.0, 100.0, 0.0], [3, 1, 0, -10.0, 95.0, 0.0]]


@pytest.mark.parametrize("engine", ENGINES)
def test_square_root_impact_fee(engine):
    # Closes 100, 102, 102 -> returns 0.02 and 0.0. With window 2 / min_obs 2, the s3 open
    # sees sigma = std(0.02, 0.0; ddof=1) = 0.02 / sqrt(2) and ADV = 100 shares (s1, s2;
    # raw and total-return shares agree, no split).
    # Decision at s2: 50% of 1000 in A at the s3 open of 102:
    #   qty = round6(500 / 102) = round6(4.90196078) = 4.901961
    #   impact = 1.0 * sigma * sqrt(4.901961 / 100)
    #   fee = 4.901961 * 102 * impact
    rows = bars("A", [FLAT, (102,) * 4, (102,) * 4, (102,) * 4, (102,) * 4],
                volume=[100] * 5)
    liq = LiquidityModel(participation=1.0, impact_coef=1.0, max_impact_bps=500, window=2,
                         min_obs=2)
    r, _, _ = go(engine, rows, {2: {"A": 0.5}}, liquidity=liq)
    q = 4.901961
    impact = (0.02 / math.sqrt(2)) * math.sqrt(q / 100)
    fee = q * 102 * impact
    assert len(r.fills) == 1
    got = r.fills.iloc[0]
    assert got["qty"] == q and got["price"] == 102.0
    assert got["fee"] == pytest.approx(fee, rel=1e-12, abs=1e-9)  # Nautilus money: 1e-9
    np.testing.assert_allclose(r.equity[3], 1000 - fee, rtol=0, atol=1e-8)


@pytest.mark.parametrize("engine", ENGINES)
def test_impact_ceiling_when_volume_is_unknown(engine):
    # Window 1, min_obs 1, participation 100%, ceiling 500 bp = 5%.
    # s1: ADV = s0's volume (1e6, known); sigma needs two returns (ddof=1) -> unknown ->
    #   impact at the 5% ceiling. Target 10 @100 costs 10 * 100 * 1.05 = 1050 > 1000, so the
    #   buy is scaled: k = 1000 / 1050 * (1 - 1e-9), q = floor6(10 * k) = 9.523809,
    #   fee = q * 100 * 0.05.
    # s2: ADV = s1's volume (missing) -> unknown. The stop exit still sells everything
    #   (never capped or blocked): q @95, fee = q * 95 * 0.05.
    rows = bars("A", [FLAT, FLAT, (100, 100, 94, 96), FLAT],
                volume=[1e6, np.nan, np.nan, np.nan])
    liq = LiquidityModel(participation=1.0, impact_coef=1.0, max_impact_bps=500, window=1,
                         min_obs=1)
    r, _, _ = go(engine, rows, {0: Orders(weights={"A": 1.0}, stop_loss={"A": 95})},
                 liquidity=liq)
    q = 9.523809
    assert fills(r)[0][:5] == [1, 0, 0, q, 100.0]
    assert r.fills.iloc[0]["fee"] == pytest.approx(q * 100 * 0.05, rel=1e-12, abs=1e-9)
    ex = r.fills.iloc[1]
    assert (ex["session"], ex["phase"], ex["qty"], ex["price"]) == (2, 1, -q, 95.0)
    assert ex["fee"] == pytest.approx(q * 95 * 0.05, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_hold_leaves_a_position_untouched(engine):
    # s1: buy 5 A @100 (50%). A closes at 200 on s2, so equity = 500 + 5 * 200 = 1500.
    # s2 decides: 25% in B, hold A as is. s3 open: B target = 0.25 * 1500 / 100 = 3.75
    # shares (375), cash 500 - 375 = 125; A keeps its 5 shares (no trade). Without `hold`
    # its weight of 0 would sell it.
    rows = (bars("A", [FLAT, FLAT, (200,) * 4, (200,) * 4])
            + bars("B", [FLAT, FLAT, FLAT, FLAT]))
    r, _, _ = go(engine, rows, {0: {"A": 0.5},
                                2: Orders(weights={"B": 0.25}, hold=["A"])})
    assert fills(r) == [[1, 0, 0, 5.0, 100.0, 0.0], [3, 0, 1, 3.75, 100.0, 0.0]]
    np.testing.assert_allclose(r.equity, [1000, 1000, 1500, 1500], **EQ)
    assert r.positions[-1].tolist() == [5.0, 3.75]


def test_entry_limit_misuse_is_rejected():
    rows = bars("A", [FLAT, FLAT, FLAT])
    panel = build_panel(pd.DataFrame(rows))
    with pytest.raises(ValueError, match="positive target weight"):
        build_schedule(Script({0: Orders(entry_limit={"A": 99})}), panel)
    with pytest.raises(ValueError, match="only for opening"):
        build_schedule(Script({0: {"A": 1.0},
                               1: Orders(weights={"A": 1.0}, entry_limit={"A": 99})}), panel)
    with pytest.raises(ValueError, match="hold"):
        build_schedule(Script({0: Orders(weights={"A": 1.0}, hold=["A"])}), panel)
    with pytest.raises(ValueError, match="hold only applies"):
        build_schedule(Script({0: Orders(hold=["A"])}), panel)
    with pytest.raises(ValueError, match="both set"):
        build_schedule(Script({0: Orders(weights={"A": 1.0}, stop_loss={"A": 95},
                                         stop_loss_pct={"A": 0.05})}), panel)
