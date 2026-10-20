"""Futures margin account, hand-computed (docs/SEMANTICS.md, "Futures").

One market AA: multiplier 10, tick 0.5, commission 2 per contract per side, slippage 1
tick, so the fee is 2 + 1 * 0.5 * 10 = 7 per contract per side. Margin rate 0.10, margin
cap 0.5 of equity. Risk-free rate 0.0001 per session. Starting cash 10,000.

Sessions d0..d4 = 2024-01-02, 03, 04, 05, 08 (XNYS).
Contract C1 (last trade d3, so with sessions_before=1 it rolls at the d2 close):
    d0 open 100 close 101 | d1 open 102 close 104 | d2 open 104 close 103
Contract C2: d2 open 106 close 106 | d3 open 107 close 108 | d4 open 108 close 105
Roll gap at d2 = 106 - 103 = 3, so C1 is shifted up 3 (Panama), and C2 is unadjusted:
    adjusted open/close: d0 103/104, d1 105/107, d2 107/106, d3 107/108, d4 108/105
Targets: +2.0x notional executed at the d1 open, -1.0x at the d3 open.

d0: cash 10000 * 1.0001 = 10001.0. Nothing held.
d1: E = 10001.0; cash 10001.0 * 1.0001 = 10002.0001.
    target 2.0 * 10001 / (10 * 102) = 19.6098 -> 20 contracts; margin 20*10*102*0.1 = 2040
    <= 5000.5. Buy 20 @ 102, fee 140 -> 9862.0001.
    close mark 20 * 10 * (107 - 105) = +400 -> 10262.0001
d2: cash 10262.0001 * 1.0001 = 10263.02630001; mark 20*10*(106-107) = -200 -> 10063.02630001
    roll at the settle: fee 2 * 20 * 7 = 280 -> 9783.02630001
d3: E = 9783.02630001; cash * 1.0001 = 9784.004602640001
    overnight mark 20*10*(107 - 106) = +200 -> 9984.004602640001
    target -1.0 * 9783.0263 / (10 * 107) = -9.143 -> -9; sell 29 @ 107, fee 203 -> 9781.004602640001
    close mark -9 * 10 * (108 - 107) = -90 -> 9691.004602640001
d4: cash * 1.0001 = 9691.973703100265; mark -9*10*(105 - 108) = +270 -> 9961.973703100265
"""

import numpy as np
import pandas as pd
import pytest

from concordance.futures.continuous import Contract, build_continuous
from concordance.futures.engine import MarginRules, run, target_contracts
from concordance.futures.panel import Spec, build_futures_panel
from concordance.strategy import Schedule

D = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])


def market():
    c1 = Contract("C1", last_trade=D[3], bars=pd.DataFrame(
        {"open": [100.0, 102.0, 104.0], "close": [101.0, 104.0, 103.0]}, index=D[:3]))
    c2 = Contract("C2", last_trade=pd.Timestamp("2024-03-15"), bars=pd.DataFrame(
        {"open": [106.0, 107.0, 108.0], "close": [106.0, 108.0, 105.0]}, index=D[2:]))
    series, table = build_continuous([c1, c2], sessions_before=1)
    spec = Spec("AA", multiplier=10.0, tick=0.5, margin_rate=0.10, commission=2.0,
                slippage_ticks=1.0)
    return build_futures_panel({"AA": series}, {"AA": spec}, pd.Series(0.0001, index=D),
                               sessions=D), table


def test_continuous_and_panel_fields():
    panel, table = market()
    assert table["roll_date"].tolist() == [D[2]]
    assert table["gap"].tolist() == [3.0]
    fd = panel.meta["futures"]
    np.testing.assert_allclose(fd.adj_close[:, 0], [104, 107, 106, 108, 105])
    np.testing.assert_allclose(fd.adj_open[:, 0], [103, 105, 107, 107, 108])
    np.testing.assert_allclose(fd.open_unadj[:, 0], [100, 102, 104, 107, 108])
    np.testing.assert_allclose(fd.close_unadj[:, 0], [101, 104, 103, 108, 105])
    assert fd.roll[:, 0].tolist() == [False, False, True, False, False]
    # Excess return: own point change over own previous settle.
    # d1 (104-101)/101, d2 (103-104)/104, d3 C2 (108-106)/106, d4 (105-108)/108
    np.testing.assert_allclose(panel["ret"][1:, 0], [3 / 101, -1 / 104, 2 / 106, -3 / 108])
    assert np.isnan(panel["ret"][0, 0])


def test_margin_account_by_hand():
    panel, _ = market()
    w = np.zeros((5, 1))
    has = np.zeros(5, bool)
    w[1, 0], has[1] = 2.0, True
    w[3, 0], has[3] = -1.0, True
    res = run(panel, Schedule.from_weights(w, has), 10_000.0, MarginRules(max_margin=0.5))
    np.testing.assert_allclose(
        res.equity,
        [10001.0, 10262.0001, 9783.02630001, 9691.004602640001, 9961.973703100265],
        rtol=0, atol=1e-8)
    assert res.positions[:, 0].tolist() == [0, 20, 20, -9, -9]
    f = res.fills
    assert f[["session", "phase", "qty", "price", "fee"]].values.tolist() == [
        [1, 0, 20.0, 102.0, 140.0], [2, 2, 0.0, 103.0, 280.0], [3, 0, -29.0, 107.0, 203.0]]


def test_margin_cap_scales_targets_toward_zero():
    # 8x on E = 10001 at price 102: raw 80008 / 1020 = 78.4392 -> 78, needing margin
    # 78 * 10 * 102 * 0.1 = 7956 > cap 5000.5. Scaled: floor(78.4392 * 5000.5 / 7956)
    # = floor(49.30) = 49.
    q, ok, scaled = target_contracts(np.array([8.0]), 10001.0, np.array([10.0]),
                                     np.array([102.0]), np.array([0.1]), MarginRules(0.5))
    assert scaled and q.tolist() == [49.0]


def test_futures_weight_limits_replace_cash_account_limits():
    from concordance.strategy import _to_row

    panel, _ = market()
    assert _to_row([3.0], panel).tolist() == [3.0]  # 3x notional is fine on margin
    with pytest.raises(ValueError):
        _to_row([9.0], panel)


def test_nautilus_matches_the_hand_computed_account():
    from concordance.futures import nautilus

    panel, _ = market()
    w = np.zeros((5, 1))
    has = np.zeros(5, bool)
    w[1, 0], has[1] = 2.0, True
    w[3, 0], has[3] = -1.0, True
    res = nautilus.run(panel, Schedule.from_weights(w, has), 10_000.0, MarginRules(0.5))
    np.testing.assert_allclose(
        res.equity,
        [10001.0, 10262.0001, 9783.02630001, 9691.004602640001, 9961.973703100265],
        rtol=0, atol=1e-6)
    assert res.positions[:, 0].tolist() == [0, 20, 20, -9, -9]
    assert res.fills[["session", "phase", "qty", "price", "fee"]].values.tolist() == [
        [1, 0, 20.0, 102.0, 140.0], [2, 2, 0.0, 103.0, 280.0], [3, 0, -29.0, 107.0, 203.0]]
