"""Annual signal timing, missing-year handling, link spells and execution."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from concordance.api import load_strategy, resolve_request, run_backtest
from concordance.data.annual import (
    AnnualFundamentals,
    attach,
    june_close,
    prepare,
    prepare_rows,
)
from concordance.panel import build_panel
from concordance.strategy import PointInTimeView

STRATEGY = Path(__file__).resolve().parents[2] / "examples/value_annual.py"


def financials(dates=("2017-12-31", "2018-12-31")):
    return pd.DataFrame([{"gvkey": str(i), "datadate": date, "curcd": "USD",
                          "indfmt": "INDL", "datafmt": "STD", "consol": "C",
                          "at": 100., "ceq": 100. - i, "revt": 100., "cogs": 10. * i}
                         for date in dates for i in range(1, 5)])


def links():
    return pd.DataFrame({"gvkey": [str(i) for i in range(1, 5)], "permno": range(1, 5),
                         "linktype": "LC", "linkprim": "P",
                         "linkdt": pd.Timestamp("2000-01-01"),
                         "linkenddt": pd.Timestamp("2099-12-31")})


def panel():
    dates = pd.to_datetime(["2018-06-28", "2018-06-29", "2018-07-02", "2018-07-03",
                            "2019-06-27", "2019-06-28", "2019-07-01", "2019-07-02"])
    frame = pd.DataFrame([{"date": date, "asset": i, "open": 10., "close": 10.,
                           "ret": 0., "mktcap_k": 100_000., "member": True}
                          for date in dates for i in range(1, 5)])
    p = build_panel(frame)
    p.fields["sp500"] = np.zeros(p.shape)
    return p


def test_june_weekend_formation_and_no_early_values():
    rows = prepare_rows(financials(), links())
    p = replace(panel(), fundamentals=AnnualFundamentals(rows))
    assert june_close(2018) == pd.Timestamp("2018-06-29")
    assert np.isnan(p.fundamentals.matrix("at", p)[0]).all()
    assert (p.fundamentals.matrix("at", p)[1] == 100).all()
    assert rows.available_at.min() == pd.Timestamp("2018-06-29")


def test_missing_year_and_missing_new_values_do_not_fall_back():
    f = financials()
    f = f[~((f.gvkey == "1") & (f.datadate == "2018-12-31"))]
    f.loc[(f.gvkey == "2") & (f.datadate == "2018-12-31"), "ceq"] = np.nan
    p = replace(panel(), fundamentals=AnnualFundamentals(prepare_rows(f, links())))
    values = p.fundamentals.matrix("ceq", p)
    assert np.isfinite(values[4, :2]).all()
    assert np.isnan(values[5:, :2]).all()


def test_choose_latest_fiscal_end_without_filling_from_other_periods():
    f = financials(("2017-03-31", "2017-12-31"))
    f.loc[f.datadate.eq("2017-12-31"), "ceq"] = np.nan
    rows = prepare_rows(f, links())
    assert len(rows) == 4 and rows.ceq.isna().all()
    assert rows.datadate.eq(pd.Timestamp("2017-12-31")).all()


def test_link_expiry_and_later_link_start():
    link_rows = links()
    link_rows.loc[link_rows.permno.eq(1), "linkenddt"] = pd.Timestamp("2018-07-02")
    link_rows.loc[link_rows.permno.eq(2), "linkdt"] = pd.Timestamp("2018-07-03")
    rows = prepare_rows(financials(), link_rows)
    p = replace(panel(), fundamentals=AnnualFundamentals(rows))
    values = p.fundamentals.matrix("at", p)
    assert values[2, 0] == 100 and np.isnan(values[3, 0])
    assert np.isnan(values[2, 1]) and values[3, 1] == 100


def test_conflicts_and_bad_numbers_fail():
    f = financials()
    bad = f.iloc[[0]].assign(at=999)
    with pytest.raises(ValueError, match="conflicting"):
        prepare_rows(pd.concat([f, bad]), links())
    f.loc[0, "at"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        prepare_rows(f, links())
    link_rows = pd.concat([links(), links().iloc[[0]].assign(gvkey="2")])
    with pytest.raises(ValueError, match="overlapping"):
        prepare_rows(financials(), link_rows)


def test_cache_uses_actual_asset_and_session_identity():
    rows = prepare_rows(financials(), links())
    adapter = AnnualFundamentals(rows)
    p = panel()
    a = adapter.matrix("ceq", p)
    rev = replace(p, assets=p.assets[::-1])
    b = adapter.matrix("ceq", rev)
    np.testing.assert_equal(a[:, ::-1], b)


def test_prepare_hashes_and_explicit_api_basis(tmp_path):
    source = tmp_path / "annual.csv"
    link = tmp_path / "links.parquet"
    financials().to_csv(source, index=False)
    links().to_parquet(link)
    dest = tmp_path / "prepared"
    prepare(source, link, dest)
    p = attach(panel(), dest)
    assert p.meta["fundamentals"]["basis"] == "annual_lagged"
    req = {"strategy": {"path": str(STRATEGY)}, "data": {
        "start": "2018-06-28", "end": "2019-07-02", "fundamentals": True,
        "fundamentals_basis": "annual_lagged", "annual_source": str(dest)}}
    assert resolve_request(req)["data"]["annual_source"] == str(dest)
    req["data"]["fundamentals_basis"] = "first_reported"
    with pytest.raises(ValueError, match="annual_source requires"):
        resolve_request(req)
    manifest = dest / "annual_manifest.json"
    m = json.loads(manifest.read_text())
    m["data_sha256"] = "invalid"
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="hash mismatch"):
        attach(panel(), dest)


def test_selection_exclusions_ties_and_no_mid_june_rebalance():
    p = replace(panel(), fundamentals=AnnualFundamentals(prepare_rows(financials(), links())))
    strategy = load_strategy(STRATEGY)
    assert strategy.decide(PointInTimeView(p, 0)) is None
    # Equities 99,98,97,96 -> cheaper half 1,2 -> most profitable = 1.
    np.testing.assert_equal(strategy.decide(PointInTimeView(p, 1)), [.8, 0, 0, 0])
    p.fields["sp500"][:, 0] = 1
    np.testing.assert_equal(strategy.decide(PointInTimeView(p, 1)), [0, .8, 0, 0])
    p.fields["sp500"][:] = 1
    np.testing.assert_equal(strategy.decide(PointInTimeView(p, 1)), np.zeros(4))


def test_three_engines_agree_and_orders_fill_after_formation(tmp_path):
    p = replace(panel(), fundamentals=AnnualFundamentals(prepare_rows(financials(), links())))
    req = {"strategy": {"path": str(STRATEGY)},
           "data": {"start": "2018-06-28", "end": "2019-07-02"},
           "engines": ["reference", "vectorbt", "nautilus"], "liquidity": None,
           "costs": {"commission_bps": 0, "default_half_spread_bps": 0},
           "validation": {"program": "annual_synthetic", "lookahead_cuts": 3}}
    result = run_backtest(req, tmp_path / "run", panel=p)
    assert result["status"] == "ok", result.get("diagnostics")
    fills = pd.read_parquet(tmp_path / "run/fills.parquet")
    assert not fills.empty
    assert pd.to_datetime(fills.date).min() == pd.Timestamp("2018-07-02")
