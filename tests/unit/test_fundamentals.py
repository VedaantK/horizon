import numpy as np
import pandas as pd
from conftest import long_frame, path

from concordance.data.fundamentals import (
    asof_matrix,
    availability,
    link_to_permno,
    quarterize_ytd,
)
from concordance.data.ingest import OPEN_END
from concordance.panel import build_panel


def test_ytd_to_quarterly():
    fq = pd.DataFrame({"gvkey": "001", "fyearq": 2020, "fqtr": [1, 2, 3, 4],
                       "oancfy": [10.0, 25.0, 45.0, 70.0]})
    assert quarterize_ytd(fq)["oancfq"].tolist() == [10, 15, 20, 25]
    gap = fq.drop(index=1)  # Q2 missing -> Q3 cannot be derived
    assert np.isnan(quarterize_ytd(gap)["oancfq"].tolist()[1])


def test_available_first_session_after_rdq():
    # rdq Friday 2020-01-17; Monday 01-20 is MLK day -> available Tuesday 2020-01-21.
    fq = pd.DataFrame({"gvkey": ["001"], "datadate": [pd.Timestamp("2019-12-31")],
                       "rdq": [pd.Timestamp("2020-01-17")]})
    assert availability(fq)["available_at"].iloc[0] == pd.Timestamp("2020-01-21")
    fq["rdq"] = pd.NaT  # no rdq: datadate + 90d = 2020-03-30 (Mon) -> 03-31
    out = availability(fq)
    assert out["available_at"].iloc[0] == pd.Timestamp("2020-03-31") and out["rdq_imputed"][0]


def test_asof_never_early_and_newest_period_wins():
    rows_px = path(77, closes=[10] * 12)
    panel = build_panel(long_frame(rows_px))  # sessions 2020-01-02 .. 2020-01-17
    s = panel.sessions
    rows = pd.DataFrame({
        "permno": [77, 77, 77],
        "datadate": pd.to_datetime(["2019-09-30", "2019-12-31", "2019-06-30"]),
        "available_at": [s[2], s[5], s[8]],  # the old Q2 arrives late (restated filing)
        "x": [1.0, 2.0, 99.0],
    })
    m = asof_matrix(rows, "x", panel)[:, 0]
    assert np.isnan(m[:2]).all()
    assert m[2:5].tolist() == [1, 1, 1]  # visible from the availability close, not before
    assert m[5:].tolist() == [2] * 7  # the late, older quarter never overrides


def test_link_valid_on_availability_date():
    fq = pd.DataFrame({"gvkey": ["001", "001"],
                       "datadate": pd.to_datetime(["2005-12-31", "2010-12-31"]),
                       "available_at": pd.to_datetime(["2006-02-01", "2011-02-01"])})
    links = pd.DataFrame({"gvkey": ["001", "001"], "permno": [1, 2], "linkprim": ["P", "P"],
                          "linkdt": pd.to_datetime(["2000-01-01", "2008-01-01"]),
                          "linkenddt": [pd.Timestamp("2007-12-31"), OPEN_END]})
    out = link_to_permno(fq, links)
    assert out.sort_values("datadate")["permno"].tolist() == [1, 2]


def test_fiscal_year_change_duplicates_resolved():
    from concordance.data.ingest import clean_fundq

    d = pd.to_datetime
    df = pd.DataFrame({
        "gvkey": ["1", "1", "1"], "datadate": d(["2006-03-31", "2006-03-31", "2006-06-30"]),
        "fyearq": [2005, 2006, 2006], "fqtr": [4, 1, 2],
        "rdq": d(["2006-03-31", "2006-05-30", "2006-08-01"]),
        "atq": [100.0, 100.0, 110.0], "revtq": [10.0, np.nan, 12.0]})
    out, stats = clean_fundq(df)
    assert stats["fiscal_year_change_duplicates_resolved"] == 1 and len(out) == 2
    first = out.iloc[0]
    # The fuller (old-calendar) row wins, but only from the later of the two report dates.
    assert first["fyearq"] == 2005 and first["revtq"] == 10
    assert first["rdq"] == pd.Timestamp("2006-05-30")


def test_ttm_needs_four_consecutive_quarters():
    from concordance.data.fundamentals import add_ttm

    ends = pd.to_datetime(["2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
                           "2020-03-31", "2020-09-30", "2020-12-31", "2021-03-31",
                           "2021-06-30"])
    fq = pd.DataFrame({"gvkey": "1", "datadate": ends,
                       "revtq": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan]})
    ttm = add_ttm(fq, fields=("revtq",))["revtq_ttm"].tolist()
    assert np.isnan(ttm[:3]).all()
    assert ttm[3:5] == [10, 14]
    # 2020-06-30 is missing: windows spanning the gap are not a trailing year.
    assert np.isnan(ttm[5:8]).all()
    assert np.isnan(ttm[8])  # a missing quarter blanks the sum


def test_view_fundamental_point_in_time_and_no_fallback():
    from concordance.data.fundamentals import Fundamentals
    from concordance.strategy import LookaheadError, PointInTimeView

    panel = build_panel(long_frame(path(77, closes=[10] * 12)))
    s = panel.sessions
    rows = pd.DataFrame({"permno": [77, 77],
                         "datadate": pd.to_datetime(["2019-09-30", "2019-12-31"]),
                         "available_at": [s[2], s[6]], "atq": [5.0, 6.0],
                         "revtq_ttm": [50.0, np.nan]})
    view = PointInTimeView(panel, 5)
    try:
        view.fundamental("atq")
        raise AssertionError("expected LookaheadError")
    except LookaheadError:
        pass
    from dataclasses import replace
    panel = replace(panel, fundamentals=Fundamentals(rows))
    at5 = PointInTimeView(panel, 5).fundamental("atq")[:, 0]
    assert at5.shape == (6,) and np.isnan(at5[:2]).all() and at5[2:].tolist() == [5] * 4
    full = PointInTimeView(panel, 11)
    assert full.fundamental("atq")[6:, 0].tolist() == [6] * 6
    # The newest quarter has no TTM revenue: NaN, not the older quarter's 50.
    assert np.isnan(full.fundamental("revtq_ttm")[6:, 0]).all()
    assert not full.fundamental("atq").flags.writeable


def test_synthetic_fundamentals_serve_text_asset_ids():
    from concordance.data.fundamentals import VIEW_FIELDS, random_fundamentals
    from concordance.strategy import PointInTimeView
    from concordance.synthetic import random_long_frame

    panel = random_fundamentals(build_panel(random_long_frame(5, n_assets=4,
                                                              n_sessions=300)), 5)
    T = panel.shape[0]
    view = PointInTimeView(panel, T - 1)
    for name in VIEW_FIELDS:
        assert view.fundamental(name).shape == (T, 4)
    assert np.isfinite(view.fundamental("atq")[-1]).any()
    assert np.isfinite(view.fundamental("revtq_ttm")[-1]).any()
