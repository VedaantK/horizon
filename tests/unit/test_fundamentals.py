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
