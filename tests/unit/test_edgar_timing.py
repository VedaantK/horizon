"""Hand-checked event dating (cases carried over from QuantMega test_eventlab + early close)."""

import pandas as pd

from concordance.events.edgar import (
    assign_day0,
    cluster,
    earnings_8k,
    needs_header_check,
    parse_header_acceptance,
    utc_to_et,
)


def day0(*ts):
    s = pd.Series(pd.to_datetime(list(ts), format="ISO8601"))
    return assign_day0(s)


def test_bmo_dmh_amc():
    out = day0("2021-04-28 07:00", "2021-04-28 09:30", "2021-04-28 15:59:59",
               "2021-04-28 16:00", "2021-04-28 16:30")
    assert out["timing"].tolist() == ["BMO", "DMH", "DMH", "AMC", "AMC"]
    assert out["day0"].dt.strftime("%m-%d").tolist() == ["04-28"] * 3 + ["04-29"] * 2


def test_amc_friday_and_holiday_rolls_forward():
    # Friday 2021-04-30 after close -> Monday 05-03. Saturday filing -> Monday.
    out = day0("2021-04-30 16:05", "2021-05-01 10:00")
    assert out["day0"].dt.strftime("%m-%d").tolist() == ["05-03", "05-03"]
    assert out["timing"].tolist() == ["AMC", "NON-SESSION"]
    # Good Friday 2021-04-02 is a holiday: a filing that day maps to Monday 04-05.
    assert day0("2021-04-02 08:00")["day0"].iloc[0] == pd.Timestamp("2021-04-05")


def test_early_close_day():
    # 2021-11-26 (day after Thanksgiving) closes at 13:00 ET: a 14:00 filing is AMC.
    out = day0("2021-11-26 12:59", "2021-11-26 14:00")
    assert out["timing"].tolist() == ["DMH", "AMC"]
    assert out["day0"].tolist() == [pd.Timestamp("2021-11-26"), pd.Timestamp("2021-11-29")]


def test_utc_to_et_handles_dst():
    # Apple-style 16:30 ET releases: 20:30Z in summer, 21:30Z in winter.
    et = utc_to_et(pd.Series(["2021-07-27T20:30:00.000Z", "2021-01-27T21:30:00.000Z"]))
    assert et.dt.strftime("%H:%M").tolist() == ["16:30", "16:30"]


def test_suspect_hours_and_header_parse():
    et = pd.Series(pd.to_datetime(["2009-04-23 16:17", "2009-04-23 12:17", "2009-04-23 20:00"]))
    # 12:17 ET could be a mislabelled 16:17 -> check header; 16:17 / 20:00 cannot move day0.
    assert needs_header_check(et).tolist() == [False, True, False]
    hdr = "<SEC-HEADER>\n<ACCEPTANCE-DATETIME>20090423161707\nACCESSION NUMBER: x"
    assert parse_header_acceptance(hdr) == pd.Timestamp("2009-04-23 16:17:07")
    assert parse_header_acceptance("no header") is None


def test_earnings_8k_filter():
    f = pd.DataFrame({
        "form": ["8-K", "8-K/A", "8-K", "10-Q", "8-K"],
        "items": ["2.02,9.01", "2.02", "5.02", "", "12.02"],
        "acceptanceDateTime": ["2021-04-28T20:30:00.000Z"] * 5,
        "accessionNumber": ["a", "b", "c", "d", "e"]})
    assert earnings_8k(f)["accession"].tolist() == ["a"]  # amendments and 12.02 excluded


def test_cluster_keeps_first_and_counts():
    ev = pd.DataFrame({
        "asset": [1, 1, 1, 2],
        "day0": pd.to_datetime(["2021-01-04", "2021-01-06", "2021-04-01", "2021-01-06"]),
        "accepted_et": pd.to_datetime(["2021-01-04 07:00", "2021-01-06 07:00",
                                       "2021-04-01 07:00", "2021-01-06 07:00"])})
    out = cluster(ev, gap_sessions=20)
    assert out["day0"].dt.strftime("%m-%d").tolist() == ["01-04", "04-01", "01-06"]
    assert out["n_clustered"].tolist() == [1, 0, 0]
