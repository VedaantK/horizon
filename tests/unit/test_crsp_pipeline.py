"""Synthetic CIZ-shaped WRDS exports -> ingest -> lake -> panel -> all engines."""

import numpy as np
import pandas as pd
import pytest

from concordance.costs import CostModel
from concordance.data import lockbox
from concordance.data.crsp import CoverageError, load_sp500_panel, xnys_sessions
from concordance.data.ingest import IngestError, ingest
from concordance.runner import run
from concordance.strategy import Strategy

SESS = xnys_sessions("2019-12-02", "2020-03-31")


def crsp_rows(permno, start, end, p0, rng, delist_ret=None, negprc_at=None):
    rows, prev = [], None
    for i in range(start, end + 1):
        c = (p0 if prev is None else prev) * float(np.exp(rng.normal(0, 0.01)))
        o = c * float(np.exp(rng.normal(0, 0.005)))
        ret = np.nan if prev is None else c / prev - 1
        if i == end and delist_ret is not None:
            ret = delist_ret
        prc = -c if i == negprc_at else c
        rows.append({"PERMNO": permno, "DlyCalDt": SESS[i].date(), "DlyOpen": o,
                     "DlyClose": c, "DlyPrc": prc, "DlyRet": ret, "DlyBid": c * 0.999,
                     "DlyAsk": c * 1.001, "DlyVol": 1e5, "Ticker": f"T{permno}",
                     "DlyDelFlg": "Y" if (i == end and delist_ret is not None) else "N"})
        prev = c
    return rows


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    T = len(SESS)
    rows = (crsp_rows(10001, 0, T - 1, 50, rng)
            + crsp_rows(10002, 0, 40, 30, rng, delist_ret=-0.45)  # delists mid-window
            + crsp_rows(10003, 10, T - 1, 80, rng, negprc_at=20)  # joins later
            + crsp_rows(10004, 0, T - 1, 20, rng))  # never a member
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.iloc[[5]]])  # an exact duplicate, as in real CRSP extracts
    raw = tmp_path / "raw"
    raw.mkdir()
    df.to_parquet(raw / "crsp_daily.parquet")
    pd.DataFrame({
        "PERMNO": [10001, 10002, 10003],
        "MbrStartDt": [SESS[0], SESS[0], SESS[15]],
        "MbrEndDt": [None, SESS[40], None],
    }).to_csv(raw / "sp500_membership.csv", index=False)
    return raw


class EqualWeightMembers(Strategy):
    def decide(self, view):
        m = view.members & np.isfinite(view["close"][-1])
        if view.i % 5 or not m.any():
            return None
        return m / m.sum() * 0.98


def test_pipeline_end_to_end(raw_dir, tmp_path):
    lake = tmp_path / "lake"
    man = ingest(raw_dir, lake)
    assert man["files"]["crsp_daily"]["exact_duplicates_dropped"] == 1
    assert man["files"]["crsp_daily"]["no_trade_rows"] == 1
    assert man["files"]["comp_fundq"]["status"] == "missing"

    panel = load_sp500_panel(SESS[0], SESS[-1], lake=lake)
    assert list(panel.assets) == [10001, 10002, 10003]  # 10004 never a member
    j2 = list(panel.assets).index(10002)
    assert panel.liquidate[41, j2]  # delisted at session 40 -> liquidated at 41's open
    assert panel.member[40, j2] and not panel.member[41, j2]
    j3 = list(panel.assets).index(10003)
    assert not panel.member[14, j3] and panel.member[15, j3]
    assert panel["close"][20, j3] > 0  # negative DlyPrc became |prc|

    out = run(panel, EqualWeightMembers(), CostModel(commission_bps=1), init_cash=1e6)
    assert out.report.ok, out.report.summary()
    ref = out.results["reference"]
    # The delisting return (-45%) is in the results, not dropped.
    liq = ref.fills[(ref.fills.session == 41) & (ref.fills.asset == j2)]
    assert len(liq) == 1 and liq["fee"].iloc[0] == 0.0


def test_conflicting_duplicates_rejected(raw_dir, tmp_path):
    df = pd.read_parquet(raw_dir / "crsp_daily.parquet")
    bad = df.iloc[[3]].copy()
    bad["DlyRet"] = 0.5
    pd.concat([df, bad]).to_parquet(raw_dir / "crsp_daily.parquet")
    with pytest.raises(IngestError, match="differ"):
        ingest(raw_dir, tmp_path / "lake")


def test_missing_column_rejected(raw_dir, tmp_path):
    df = pd.read_parquet(raw_dir / "crsp_daily.parquet").drop(columns=["DlyRet"])
    df.to_parquet(raw_dir / "crsp_daily.parquet")
    with pytest.raises(IngestError, match="missing required columns"):
        ingest(raw_dir, tmp_path / "lake")


def test_low_coverage_refused(raw_dir, tmp_path):
    df = pd.read_parquet(raw_dir / "crsp_daily.parquet")
    df = df[~((df.PERMNO == 10001) & (df.DlyCalDt > SESS[10].date()))]  # member data vanishes
    df.loc[df.PERMNO == 10001, "DlyDelFlg"] = "N"
    df.to_parquet(raw_dir / "crsp_daily.parquet")
    lake = tmp_path / "lake"
    ingest(raw_dir, lake)
    with pytest.raises(CoverageError, match="coverage too low"):
        load_sp500_panel(SESS[0], SESS[-1], lake=lake)


def test_lockbox(raw_dir, tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    ingest(raw_dir, lake)
    monkeypatch.setenv("CONCORDANCE_LOCKBOX", str(SESS[60].date()))
    monkeypatch.setenv("CONCORDANCE_LAKE", str(lake))
    with pytest.raises(lockbox.LockboxViolation):
        load_sp500_panel(SESS[0], SESS[-1], lake=lake)
    load_sp500_panel(SESS[0], SESS[59], lake=lake)  # before the lockbox: fine
    load_sp500_panel(SESS[0], SESS[-1], lake=lake, unlock_reason="final evaluation")
    assert "final evaluation" in (lake / "lockbox_ledger.jsonl").read_text()


def test_chunked_web_exports_with_wrds_date_formats(raw_dir, tmp_path):
    """Web-form downloads: CRSP split into date chunks (overlapping), YYYYMMDD dates, CSV."""
    df = pd.read_parquet(raw_dir / "crsp_daily.parquet")
    (raw_dir / "crsp_daily.parquet").unlink()
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"]).dt.strftime("%Y%m%d")
    cut = SESS[40].strftime("%Y%m%d")
    df[df.DlyCalDt <= cut].to_csv(raw_dir / "crsp_daily_part1.csv", index=False)
    df[df.DlyCalDt >= cut].to_csv(raw_dir / "crsp_daily_part2.csv", index=False)  # overlap
    lake = tmp_path / "lake"
    man = ingest(raw_dir, lake)["files"]["crsp_daily"]
    assert sorted(man["sources"]) == ["crsp_daily_part1.csv", "crsp_daily_part2.csv"]
    assert man["rows"] == len(df.drop_duplicates())
    panel = load_sp500_panel(SESS[0], SESS[-1], lake=lake)
    assert panel.shape == (len(SESS), 3)


def test_unparseable_dates_rejected(raw_dir, tmp_path):
    df = pd.read_parquet(raw_dir / "crsp_daily.parquet")
    df["DlyCalDt"] = df["DlyCalDt"].astype(str)
    df.loc[df.index[3], "DlyCalDt"] = "not a date"
    df.to_parquet(raw_dir / "crsp_daily.parquet")
    with pytest.raises(IngestError, match="unparseable dates"):
        ingest(raw_dir, tmp_path / "lake")


def test_permnos_command(raw_dir, tmp_path):
    from concordance.cli import main
    out = tmp_path / "permnos.txt"
    main(["permnos", str(raw_dir / "sp500_membership.csv"), "-o", str(out)])
    assert out.read_text().split() == ["10001", "10002", "10003"]


def _real_data_cases(raw_dir):
    """Add the patterns found in the real 1990-2025 export:
    10005 leaves its exchange (blank rows), relists with no bridging return (like AMR 2012-13);
    10006 fails with a final price but no DlyRet (like SVB on 2023-03-13);
    10007 has a one-day trading halt whose next return spans both days (like BIIB 2020-11-06).
    """
    rng = np.random.default_rng(7)
    T = len(SESS)
    df = pd.read_parquet(raw_dir / "crsp_daily.parquet")
    a = pd.DataFrame(crsp_rows(10005, 0, T - 1, 40, rng))
    a.loc[31:45, ["DlyOpen", "DlyClose", "DlyPrc", "DlyRet", "DlyBid", "DlyAsk", "DlyVol"]] = np.nan
    a.loc[46, "DlyRet"] = np.nan  # relisting: CRSP gives no return across the gap
    b = pd.DataFrame(crsp_rows(10006, 0, 50, 60, rng))
    b.loc[50, "DlyRet"] = np.nan
    b.loc[50, ["DlyPrc", "DlyClose"]] = b.loc[49, "DlyPrc"] * 0.01
    b.loc[50, "DlyOpen"] = np.nan
    c = pd.DataFrame(crsp_rows(10007, 0, T - 1, 70, rng))
    c.loc[20, ["DlyOpen", "DlyClose", "DlyPrc", "DlyRet", "DlyBid", "DlyAsk", "DlyVol"]] = np.nan
    c.loc[21, "DlyRet"] = c.loc[21, "DlyPrc"] / c.loc[19, "DlyPrc"] - 1
    pd.concat([df, a, b, c]).to_parquet(raw_dir / "crsp_daily.parquet")
    m = pd.read_csv(raw_dir / "sp500_membership.csv")
    extra = pd.DataFrame({"PERMNO": [10005, 10006, 10007],
                          "MbrStartDt": [str(SESS[0].date())] * 3,
                          "MbrEndDt": [None, str(SESS[50].date()), None]})
    pd.concat([m, extra]).to_csv(raw_dir / "sp500_membership.csv", index=False)
    return b.loc[49, "DlyPrc"]


def test_listing_gaps_and_failures(raw_dir, tmp_path):
    from concordance.data.crsp import RELIST_ID_OFFSET

    _real_data_cases(raw_dir)
    lake = tmp_path / "lake"
    man = ingest(raw_dir, lake)["files"]["crsp_daily"]
    assert man["final_return_from_price"] == [
        {"permno": 10006, "date": str(SESS[50].date()), "ret": -0.99}]
    assert man["listing_gaps"] == [{"permno": 10005, "last_priced": str(SESS[30].date()),
                                    "relisted": str(SESS[46].date()), "member_at_gap": True}]
    assert man["unchained_rows_left"] == 0

    panel = load_sp500_panel(SESS[0], SESS[-1], lake=lake)
    ids = list(panel.assets)
    first, second = ids.index(10005), ids.index(10005 + RELIST_ID_OFFSET)
    assert list(panel.meta["permno"][[first, second]]) == [10005, 10005]
    # First listing: forced exit at the session after its last price, at that value.
    assert panel.liquidate[31, first] and panel.liquidate.sum(0)[first] == 1
    assert panel.exec_price[31, first] == panel.tr_close[30, first]
    # Second listing: a fresh security from the relisting day, no history before it.
    assert np.isnan(panel.tr_close[:46, second]).all()
    assert panel.tr_close[46, second] == pytest.approx(panel["close"][46, second])
    # Membership follows the PERMNO across both listings.
    assert panel.member[10, first] and panel.member[50, second] and not panel.member[50, first]
    # The failed bank: -99% from the price, then liquidated.
    j6 = ids.index(10006)
    assert panel["ret"][50, j6] == pytest.approx(-0.99)
    assert panel.liquidate[51, j6]
    # The halt day is skipped; the next return spans both days.
    j7 = ids.index(10007)
    assert np.isnan(panel.tr_close[20, j7])
    assert panel.tr_close[21, j7] / panel.tr_close[19, j7] == pytest.approx(
        panel["close"][21, j7] / panel["close"][19, j7])

    out = run(panel, EqualWeightMembers(), CostModel(commission_bps=1), init_cash=1e6)
    assert out.report.ok, out.report.summary()
    from concordance.cli import listing_gap_exits
    ex = listing_gap_exits(panel, out.results["reference"])
    assert list(ex["permno"]) == [10005] and ex["date"].iloc[0] == SESS[31].date()


def test_unexplained_missing_return_still_refused(raw_dir, tmp_path):
    from concordance.panel import PanelError

    df = pd.read_parquet(raw_dir / "crsp_daily.parquet")
    df.loc[(df.PERMNO == 10001) & (df.DlyCalDt == SESS[30].date()), "DlyRet"] = np.nan
    df.to_parquet(raw_dir / "crsp_daily.parquet")
    lake = tmp_path / "lake"
    assert ingest(raw_dir, lake)["files"]["crsp_daily"]["unchained_rows_left"] == 1
    with pytest.raises(PanelError, match="missing total return mid-history"):
        load_sp500_panel(SESS[0], SESS[-1], lake=lake)


def test_benchmark_and_metrics_skip_warmup(raw_dir, tmp_path):
    from concordance.report import metrics

    df = pd.read_parquet(raw_dir / "crsp_daily.parquet").drop_duplicates()
    df["DlyCap"] = df["PERMNO"].map({10001: 100.0, 10002: 300.0, 10003: 50.0, 10004: 999.0})
    df["sprtrn"] = 0.0
    df.to_parquet(raw_dir / "crsp_daily.parquet")
    lake = tmp_path / "lake"
    ingest(raw_dir, lake)
    panel = load_sp500_panel(SESS[30], SESS[-1], lake=lake, warmup_sessions=30)
    b = panel.meta["benchmark"]
    assert b.index.equals(panel.sessions)
    # Hand check, session 35: members 10001 and 10002 (10003 joins at 15, so also a member;
    # 10004 never is), weighted by the previous day's cap 100 / 300 / 50.
    r = {a: panel["ret"][35, list(panel.assets).index(a)] for a in (10001, 10002, 10003)}
    assert b.iloc[35] == pytest.approx((100 * r[10001] + 300 * r[10002] + 50 * r[10003]) / 450)
    assert (panel.meta["sp500_price_return"].dropna() == 0.0).all()
    res = run(panel, EqualWeightMembers(), CostModel(), engines=["reference"]).result
    m = metrics(res, start=30, benchmark=b)
    assert m["start"] == str(SESS[30].date())
    eq = res.equity_series.iloc[30:]
    years = (len(eq) - 1) / 252
    assert m["cagr"] == pytest.approx((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)
    rb = b.iloc[31:]
    assert m["benchmark_cagr"] == pytest.approx((1 + rb).prod() ** (252 / len(rb)) - 1)


def test_run_ledger_follows_lake(raw_dir, tmp_path, monkeypatch):
    from concordance.cli import main

    lake, home_lake = tmp_path / "lake", tmp_path / "default_lake"
    ingest(raw_dir, lake)
    monkeypatch.setenv("CONCORDANCE_LAKE", str(home_lake))
    strat = tmp_path / "s.py"
    strat.write_text("from tests.unit.test_crsp_pipeline import EqualWeightMembers\n"
                     "STRATEGY = EqualWeightMembers()\n")
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).parents[2]))
    main(["run", str(strat), "--start", str(SESS[0].date()), "--end", str(SESS[-1].date()),
          "--lake", str(lake), "--engines", "reference", "--lookahead-cuts", "0",
          "--program", "t"])
    assert (lake / "ledger" / "t" / "trials.jsonl").exists()
    assert not (home_lake / "ledger").exists()


def test_gap_exit_valued_with_crsp_delisting_return(raw_dir, tmp_path):
    _real_data_cases(raw_dir)
    # v2 stkdelists layout, as the WRDS web form exports it.
    pd.DataFrame({
        "delactiontype": ["GDR", "MER"],
        "permno": [10005, 10002],
        "delistingdt": [str(SESS[30].date()), str(SESS[40].date())],
        "delret": [-0.6, -0.45],  # 10005 went dark: worth 60% less off-exchange
        "delnextprc": [np.nan, np.nan],
    }).to_csv(raw_dir / "crsp_delist.csv", index=False)
    lake = tmp_path / "lake"
    man = ingest(raw_dir, lake)["files"]
    assert man["crsp_delist"]["rows"] == 2
    c = man["crsp_daily"]
    assert c["gap_exits_from_delistings"] == [
        {"permno": 10005, "last_priced": str(SESS[30].date()), "relisted": str(SESS[46].date()),
         "date": str(SESS[31].date()), "ret": -0.6, "source": "delret"}]
    assert c["gap_exits_unknown"] == []
    # 10002's final row already carries its -45% delisting return, as CIZ data does.
    assert c["delisting_return_check"] == {"checked": 1, "mismatches": []}

    panel = load_sp500_panel(SESS[0], SESS[-1], lake=lake)
    j = list(panel.assets).index(10005)
    # The listing ends one session later at 40% of its last value; sold at the next open.
    assert panel.tr_close[31, j] == pytest.approx(panel.tr_close[30, j] * 0.4)
    assert panel.liquidate[32, j] and panel.exec_price[32, j] == panel.tr_close[31, j]
    out = run(panel, EqualWeightMembers(), CostModel(commission_bps=1), init_cash=1e6)
    assert out.report.ok, out.report.summary()
    from concordance.cli import listing_gap_exits
    assert listing_gap_exits(panel, out.results["reference"]).empty  # exit value is known
    fills = out.results["reference"].fills
    sold = fills[(fills.session == 32) & (fills.asset == j)]
    assert len(sold) == 1 and sold["price"].iloc[0] == panel.tr_close[31, j]
