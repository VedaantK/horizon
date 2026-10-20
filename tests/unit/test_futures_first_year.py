import pandas as pd
import pytest

from concordance.data.crsp import xnys_sessions
from concordance.data.futures import RollGap, load_series


def write_lake(tmp, contracts):
    """contracts: (code, deadline, first bar, last bar); prices are flat 100."""
    rows, bars = [], []
    for i, (code, deadline, lo, hi) in enumerate(contracts):
        rows.append({"root": "LE", "futcode": i, "month_code": code[0], "year": int(code[1:]),
                     "liquid": True, "deadline": pd.Timestamp(deadline)})
        for d in xnys_sessions(lo, hi):
            bars.append({"root": "LE", "futcode": i, "date": d, "open": 100.0,
                         "settle": 100.0})
    pd.DataFrame(rows).to_parquet(tmp / "futures_contracts.parquet")
    pd.DataFrame(bars).to_parquet(tmp / "futures_bars.parquet")


def test_first_year_gap_starts_the_market_after_it(tmp_path):
    # Like LE 1979: Z79 can't roll because G80's history only begins 1980-01-02.
    write_lake(tmp_path, [("Z1979", "1979-11-30", "1979-01-02", "1979-12-18"),
                          ("G1980", "1980-01-31", "1980-01-02", "1980-02-20"),
                          ("J1980", "1980-03-31", "1980-01-02", "1980-04-18")])
    series, table, skipped = load_series("LE", tmp_path, xnys_sessions("1979-01-02",
                                                                       "1980-03-20"))
    assert skipped == [("LEZ1979", "first-year history")]
    assert series.index[0] == pd.Timestamp("1980-01-02")
    assert table["from_contract"].tolist() == ["LEG1980"]


def test_a_later_gap_still_raises(tmp_path):
    write_lake(tmp_path, [("Z1979", "1979-11-30", "1979-01-02", "1979-12-18"),
                          ("G1980", "1980-01-31", "1979-06-01", "1980-02-20"),
                          ("Z1980", "1980-11-28", "1980-01-02", "1980-12-19"),
                          ("G1981", "1981-01-30", "1981-01-02", "1981-02-20")])
    with pytest.raises(RollGap, match="LEZ1980"):
        load_series("LE", tmp_path, xnys_sessions("1979-01-02", "1981-01-20"))


def test_a_session_no_market_trades_is_dropped_and_its_interest_carried():
    from concordance.data.futures import _drop_closed_sessions

    sessions = pd.DatetimeIndex(pd.to_datetime(["1974-11-04", "1974-11-05", "1974-11-06"]))
    series = {"ZC": pd.DataFrame({"close": [1.0, float("nan"), 1.0]}, index=sessions)}
    rf = pd.Series([0.001, 0.002, 0.003], index=sessions)
    kept, rf_kept, closed = _drop_closed_sessions(series, sessions, rf)
    assert closed.tolist() == [pd.Timestamp("1974-11-05")]
    assert kept.tolist() == [pd.Timestamp("1974-11-04"), pd.Timestamp("1974-11-06")]
    # 11-06 earns its own day and election day's: 1.002 * 1.003 - 1 = 0.005006
    assert rf_kept.tolist() == pytest.approx([0.001, 0.005006])
