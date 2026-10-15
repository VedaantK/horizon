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
