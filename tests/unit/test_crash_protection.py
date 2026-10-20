import numpy as np
import pandas as pd
import pytest

from concordance.data.crsp import xnys_sessions
from concordance.overlay import invested_mask, protect, summary


def flat_index_then_drop(drop_day="2021-01-28"):
    """Flat S&P for 2020, a -10% day in late January 2021, flat after."""
    s = xnys_sessions("2020-01-02", "2021-03-31")
    r = pd.Series(0.0, index=s)
    r[pd.Timestamp(drop_day)] = -0.10
    return r


def test_month_end_signal_switches_two_sessions_later():
    r = flat_index_then_drop()
    window = r.loc["2021-01-04":].index
    inv = invested_mask(window, r)
    # Jan 29 (last January session) closes at 0.9, below its 200-day average (~0.9995):
    # decided at that close, traded at the Feb 1 open, so Feb 1's return is still the
    # strategy's and the safe asset earns from Feb 2. At the Feb 26 close the level is
    # still 0.9 against an average of ~0.99 (mostly pre-drop days): stay protected.
    assert inv.loc[:"2021-02-01"].eq(1.0).all()
    assert inv.loc["2021-02-02":].eq(0.0).all()


def test_a_drop_inside_a_month_waits_for_month_end():
    r = flat_index_then_drop("2021-01-12")
    inv = invested_mask(r.loc["2021-01-04":"2021-01-29"].index, r)
    # The last session of the window is a month end, but there is no later session to
    # trade on, so nothing switches inside the window.
    assert inv.eq(1.0).all()


def test_needs_two_hundred_sessions_of_history():
    r = flat_index_then_drop()
    with pytest.raises(ValueError, match="needs 200 S&P sessions"):
        invested_mask(r.loc["2020-03-02":].index, r)


def test_protect_costs_and_safe_returns_by_hand():
    idx = pd.date_range("2021-01-04", periods=3, freq="B")
    rets = pd.Series([0.01, 0.02, 0.03], index=idx)
    inv = pd.Series([1.0, 0.0, 1.0], index=idx)
    safe = pd.Series(0.001, index=idx)
    out = protect(rets, inv, safe, cost_bps=10)
    # day 1 strategy; day 2 safe 0.001 minus a 10 bp switch; day 3 back: 0.03 - 0.001.
    assert out.tolist() == pytest.approx([0.01, 0.0, 0.029])


def test_summary_matches_report_definitions():
    r = pd.Series([0.1, -0.5, 0.2])
    s = summary(r, periods=3)  # one "year" of 3 periods
    assert s["cagr"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert s["max_drawdown"] == pytest.approx(-0.5)
    assert s["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(3))
