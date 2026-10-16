"""Point-in-time supplier -> customer links (data/customers.py).

Expected values below are worked out by hand in the comments, not copied from output.
"""

import json

import numpy as np
import pandas as pd
import pytest

from concordance.api import RequestError, resolve_request
from concordance.data.customers import (
    IMPUTED_SHARE,
    UNNAMED,
    build_links,
    clean_funda,
    clean_seg_customer,
    from_frame,
    match_customers,
    normalize_name,
    random_links,
)
from concordance.data.ingest import OPEN_END, IngestError, ingest
from concordance.panel import build_panel
from concordance.strategy import LookaheadError, PointInTimeView, Strategy
from concordance.synthetic import random_long_frame
from concordance.validation.lookahead import check_truncation_invariance

T = pd.Timestamp


def test_normalize_names_and_unnamed_placeholders():
    assert normalize_name("Amazon.com Inc") == "AMAZON"
    assert normalize_name("The Boeing Co.") == "BOEING"
    assert normalize_name("Wal-Mart Stores, Inc.") == "WAL MART STORES"
    assert normalize_name("AT&T Inc") == "AT AND T"
    assert normalize_name(None) == ""
    for s in ("NOT REPORTED", "5 CUSTOMERS", "TWO MAJOR CUSTOMERS", "CUSTOMER A",
              "OTHERS", "LARGEST CUSTOMER"):
        assert UNNAMED.fullmatch(s), s
    for s in ("TARGET", "CUSTOMERS BANCORP", "GENERAL MOTORS"):
        assert not UNNAMED.fullmatch(s), s


def seg_rows():
    return pd.DataFrame({
        "gvkey": [99.0, 99.0, 99.0, 99.0, 99.0, np.nan],  # float ids, as CSVs give them
        "srcdate": [T("2019-06-30")] * 6,
        "conm": ["SUPPLIER INC"] * 5 + [None],
        "cnms": ["Cisco Systems Inc", "Cisco Systems Inc", "Not Reported", "Canada",
                 "Unpriced Co", "Ghost"],
        "ctype": ["COMPANY", "COMPANY", "COMPANY", "GEOREG", "COMPANY", "COMPANY"],
        "salecs": [30.0, 10.0, 5.0, 100.0, np.nan, 1.0],
        "sid": [1, 2, 1, 1, 1, 1],
    })


def test_clean_seg_customer_sums_segments_and_drops_placeholders():
    out, stats = clean_seg_customer(seg_rows())
    # Cisco is reported under two segments: 30 + 10 = 40. "Not Reported" and the
    # geographic row go; the blank-id row is dropped and counted.
    assert out["gvkey"].unique().tolist() == ["000099"]
    by = out.set_index("cnms")["salecs"]
    assert by["Cisco Systems Inc"] == 40.0
    assert np.isnan(by["Unpriced Co"])  # all-missing sales stay missing, not 0
    assert set(out["cnms"]) == {"Cisco Systems Inc", "Unpriced Co"}
    assert stats["no_gvkey_dropped"] == 1 and stats["unnamed_dropped"] == 1


def test_clean_funda_keeps_industrial_format_and_refuses_conflicts():
    f = pd.DataFrame({"gvkey": ["1", "1"], "datadate": [T("2019-12-31")] * 2,
                      "indfmt": ["INDL", "FS"], "sale": [100.0, 90.0]})
    out, stats = clean_funda(f)
    assert out["sale"].tolist() == [100.0] and stats["non_indl_dropped"] == 1
    f["indfmt"] = "INDL"
    with pytest.raises(IngestError):
        clean_funda(f)


def universe():
    """Compustat names, links and S&P membership for a few companies."""
    funda = pd.DataFrame({
        "gvkey": ["000010", "000020", "000030", "000040", "000050", "000060", "000099"],
        "datadate": [T("2019-06-30")] * 7,
        "conm": ["CISCO SYSTEMS INC", "HP INC", "HELMERICH & PAYNE", "TINY CORP",
                 "ACME CORP", "ACME CORP", "SUPPLIER INC"],
        "tic": ["CSCO", "HPQ", "HP", "TNY", "ACM", "ACM2", "SUP"],
        "sale": [50e3, 60e3, 3e3, 10.0, 1e3, 1e3, 100.0],
    })
    links = pd.DataFrame({
        "gvkey": ["000010", "000020", "000030", "000040", "000050", "000060", "000099"],
        "permno": [100, 200, 300, 400, 500, 600, 999],
        "linktype": "LC", "linkprim": "P",
        "linkdt": [T("1990-01-01")] * 4 + [T("1990-01-01"), T("2015-01-01"),
                                            T("1990-01-01")],
        # Two different companies named ACME: the first link ends before the second starts.
        "linkenddt": [OPEN_END] * 4 + [T("2014-12-31"), OPEN_END, OPEN_END],
    })
    sp500 = pd.DataFrame({"permno": [100, 200, 300, 500, 600, 999],
                          "start": T("1990-01-01"), "end": OPEN_END})
    return funda, links, sp500


def seg_for(names, dates=None, sales=None):
    n = len(names)
    return pd.DataFrame({"gvkey": "000099", "srcdate": dates or [T("2019-06-30")] * n,
                         "cnms": names, "salecs": sales or [10.0] * n})


def test_exact_alias_ticker_precedence_and_review():
    funda, links, sp500 = universe()
    seg = seg_for(["Cisco Systems, Inc.", "Hewlett-Packard Co", "Tiny Corp", "Cisco Sys"])
    m, rev, rep = match_customers(seg, funda, links, sp500)
    got = m.set_index("cnms")["customer_gvkey"].to_dict()
    assert got["Cisco Systems, Inc."] == "000010"
    # HEWLETT PACKARD -> alias HP -> the company named HP (HP Inc), not the HP ticker.
    assert got["Hewlett-Packard Co"] == "000020"
    assert m.set_index("cnms").loc["Hewlett-Packard Co", "match"] == "alias"
    # Tiny Corp is not an S&P 500 company; "Cisco Sys" is not an exact name: both reviewed.
    assert set(rev["customer_name"]) == {"Tiny Corp", "Cisco Sys"}
    assert rep["matched_rows"] == 2


def test_reused_name_resolved_by_link_valid_at_fiscal_date():
    funda, links, sp500 = universe()
    seg = seg_for(["Acme Corp", "Acme Corp"], dates=[T("2010-06-30"), T("2019-06-30")])
    m, _, _ = match_customers(seg, funda, links, sp500)
    assert m.sort_values("srcdate")["customer_gvkey"].tolist() == ["000050", "000060"]


def test_review_decisions_accept_and_reject():
    funda, links, sp500 = universe()
    seg = seg_for(["Cisco Sys", "Tiny Corp"])
    decisions = pd.DataFrame({"customer_name": ["Cisco Sys", "Tiny Corp"],
                              "decision": ["accept", "reject"], "gvkey": ["10", ""]})
    m, rev, rep = match_customers(seg, funda, links, sp500, decisions)
    assert m.set_index("cnms")["customer_gvkey"].to_dict() == {"Cisco Sys": "000010"}
    assert rep["decisions_accepted"] == 1 and rep["decisions_rejected"] == 1
    bad = decisions.assign(decision=["yes", ""])
    with pytest.raises(ValueError, match="accept"):
        match_customers(seg, funda, links, sp500, bad)


def test_build_links_availability_share_and_imputation():
    funda, links, _ = universe()
    sessions = pd.bdate_range("2019-06-01", "2021-12-31")
    matched = pd.DataFrame({
        "gvkey": "000099", "srcdate": T("2019-06-30"),
        "cnms": ["Cisco", "HP", "Huge"], "customer_gvkey": ["000010", "000020", "000030"],
        "salecs": [25.0, np.nan, 150.0], "match": "exact"})
    out, stats = build_links(matched, funda, links, sessions)
    r = out.set_index("customer_permno")
    # Supplier sales 100: Cisco 25/100 = 0.25; HP has no amount -> 10% threshold, flagged;
    # "Huge" claims 150/100 > 100% and is dropped.
    assert r.loc[100, "share"] == pytest.approx(0.25) and not r.loc[100, "imputed"]
    assert r.loc[200, "share"] == IMPUTED_SHARE and r.loc[200, "imputed"]
    assert 300 not in r.index and stats["dropped_share_over_100pct"] == 1
    # 2019-06-30 + 6 months = 2019-12-30 (Mon); first session strictly after = Tue 12-31.
    assert r.loc[100, "available"] == T("2019-12-31")
    assert r.loc[100, "expires"] == T("2020-12-30")  # + 18 months
    assert set(out["supplier_permno"]) == {999}


def frame(rows):
    return pd.DataFrame(rows, columns=["supplier_permno", "customer_permno", "srcdate",
                                       "share", "imputed", "available", "expires"])


def test_weights_never_early_and_newest_fiscal_year_wins():
    sessions = pd.bdate_range("2020-01-01", periods=60)
    s = sessions
    links = from_frame(frame([
        # FY2019: supplier 1 -> customer 2 (30%), visible s[5] .. s[30] inclusive.
        (1, 2, T("2019-06-30"), 0.30, False, s[5], s[30]),
        # FY2020 filing arrives at s[15] and names only customer 3 (imputed share).
        (1, 3, T("2020-06-30"), 0.10, True, s[15], s[40]),
    ]), sessions, np.array([1, 2, 3]))
    w = lambda i, imp=True: links.weights_at(i, imp)  # noqa: E731
    assert not w(4).any()  # one session before availability: nothing
    assert w(5)[0, 1] == 0.30 and w(14)[0, 1] == 0.30
    # From s[15] the newer filing replaces the old one even though FY2019 has not expired.
    assert w(15)[0, 1] == 0 and w(15)[0, 2] == 0.10
    assert not w(15, imp=False).any()  # the only visible link is imputed
    assert w(40)[0, 2] == 0.10 and not w(41).any()  # expired after its last session
    assert not w(20).flags.writeable


def test_weights_match_brute_force_on_random_links():
    rng = np.random.default_rng(7)
    sessions = pd.bdate_range("2020-01-01", periods=120)
    rows = []
    for _ in range(80):
        a, b = rng.choice(6, 2, replace=False)
        start = int(rng.integers(0, 100))
        stop = start + int(rng.integers(1, 40))
        rows.append((int(a), int(b), T("2018-01-31") + pd.DateOffset(months=int(rng.integers(0,
                     12))), float(rng.uniform(0.1, 0.5)), bool(rng.random() < 0.3),
                     sessions[start], sessions[min(stop, 119)]))
    f = frame(rows)
    links = from_frame(f, sessions, np.arange(6))
    order = rng.permutation(120)  # out of order, to exercise the cache
    for i in order:
        d = sessions[i]
        vis = f[(f["available"] <= d) & (f["expires"] >= d)]
        newest = vis.groupby("supplier_permno")["srcdate"].transform("max")
        vis = vis[vis["srcdate"] == newest]
        want = np.zeros((6, 6))
        for r in vis.itertuples():
            want[r.supplier_permno, r.customer_permno] = r.share
        assert np.array_equal(links.weights_at(int(i)), want), i


class CustomerMomentum(Strategy):
    """Hold the supplier with the best revenue-weighted customer return over 5 sessions."""

    name = "customer_momentum_test"

    def __init__(self):
        self.params = {}

    def warmup(self):
        return 6

    def decide(self, view):
        tr = view["tr_close"]
        r = tr[-1] / tr[-6] - 1
        w = view.customer_weights()
        score = w @ np.nan_to_num(r) / np.where(w.sum(1) > 0, w.sum(1), np.nan)
        ok = np.isfinite(score) & view.members & np.isfinite(tr[-1])
        out = np.zeros(len(view.assets))
        if ok.any():
            out[np.flatnonzero(ok)[np.argmax(score[ok])]] = 0.5
        return out


def test_view_exposes_links_and_passes_truncation_check():
    panel = random_links(build_panel(random_long_frame(3, n_assets=12, n_sessions=300)), 3)
    view = PointInTimeView(panel, 260)
    assert view.customer_weights().shape == (12, 12)
    check_truncation_invariance(lambda p: CustomerMomentum(), panel, cuts=4)


def test_view_without_links_says_how_to_load_them():
    panel = build_panel(random_long_frame(3, n_assets=4, n_sessions=30))
    with pytest.raises(LookaheadError, match="customer_links"):
        PointInTimeView(panel, 10).customer_weights()


def test_request_validates_customer_links_flag():
    base = {"strategy": {"path": "s.py"}, "data": {"start": "2019-01-01", "end": "2019-06-01"}}
    assert resolve_request(base)["data"]["customer_links"] is False
    with pytest.raises(RequestError):
        resolve_request({**base, "data": {**base["data"], "customer_links": "yes"}})


def test_ingest_keeps_datasets_from_other_folders(tmp_path):
    lake, raw = tmp_path / "lake", tmp_path / "raw"
    lake.mkdir()
    raw.mkdir()
    pd.DataFrame({"x": [1]}).to_parquet(lake / "crsp_daily.parquet")
    (lake / "manifest.json").write_text(json.dumps({"files": {
        "crsp_daily": {"status": "ok", "rows": 1, "listing_gaps": []},
        "customer_links": {"status": "ok", "links": 5}}}))
    pd.DataFrame({"gvkey": ["001000"], "lpermno": [7], "linktype": ["LC"],
                  "linkprim": ["P"], "linkdt": ["2000-01-01"], "linkenddt": ["E"]}
                 ).to_csv(raw / "ccm_link.csv", index=False)
    files = ingest(raw, lake)["files"]
    assert files["ccm_link"]["status"] == "ok"
    assert files["crsp_daily"]["kept_from_previous_ingest"] and files["crsp_daily"]["rows"] == 1
    assert files["customer_links"]["links"] == 5
    assert files["comp_fundq"]["status"] == "missing"
