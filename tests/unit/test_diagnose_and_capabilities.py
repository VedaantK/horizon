"""`concordance diagnose` (where engines stop agreeing) and `concordance capabilities`."""

import dataclasses
import json
from pathlib import Path

import pytest

from concordance import diagnose as dg
from concordance.api import DEFAULTS, run_backtest
from concordance.capabilities import capabilities
from concordance.panel import build_panel
from concordance.runner import run_engine
from concordance.strategy import MAX_GROSS, MAX_NET
from concordance.synthetic import random_long_frame

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "swing_pullback.py"


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    panel = build_panel(random_long_frame(7, n_assets=30, n_sessions=320))
    out = tmp_path_factory.mktemp("run")
    req = {"strategy": {"path": str(EXAMPLE), "params": {"trend_days": 60}},
           "data": {"start": str(panel.sessions[0].date()),
                    "end": str(panel.sessions[-1].date())},
           "engines": ["reference", "vectorbt"],
           "liquidity": {"window": 10, "min_obs": 5},
           "validation": {"program": "t", "lookahead_cuts": 0}}
    res = run_backtest(req, out, panel=panel)
    assert res["status"] == "ok", res["diagnostics"]
    return panel, out


def test_agreeing_engines_report_ok(saved_run):
    panel, out = saved_run
    rep = dg.diagnose(out, panel=panel)
    assert rep["ok"] and rep["engines"]["vectorbt"]["ok"]
    assert "first_failure" not in rep["engines"]["vectorbt"]
    assert json.loads((out / "diagnose.json").read_text())["ok"]
    assert "agrees with the reference" in dg.render(rep)


def test_a_broken_fill_is_located(saved_run, monkeypatch):
    panel, out = saved_run

    def broken(engine, *args):
        r = run_engine(engine, *args)
        if engine != "vectorbt":
            return r
        f = r.fills.copy()
        k = len(f) // 2  # one fill buys 0.5 share more; positions follow from then on
        s, j = int(f.loc[k, "session"]), int(f.loc[k, "asset"])
        f.loc[k, "qty"] += 0.5
        pos = r.positions.copy()
        pos[s:, j] += 0.5
        broken.at = (s, j)
        return dataclasses.replace(r, fills=f, positions=pos)

    monkeypatch.setattr("concordance.runner.run_engine", broken)
    rep = dg.diagnose(out, panel=panel)
    s, j = broken.at
    e = rep["engines"]["vectorbt"]
    assert not rep["ok"] and not e["ok"]
    d = e["first_failure"]
    assert d["session"] == s and d["date"] == str(panel.sessions[s].date())
    row = next(r for r in d["fills_that_differ"] if r["asset"] == panel.assets[j])
    assert row["qty_gap"] == pytest.approx(0.5) and not row["explained"]
    assert any(p["asset"] == panel.assets[j] for p in d["positions_that_differ_at_close"])
    text = dg.render(rep)
    assert f"first unexplained difference on {d['date']}" in text


def test_capabilities_come_from_the_enforcing_code(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"files": {"crsp_daily": {"start": "1990-01-02", "end": "2025-12-31", "rows": 5}}}))
    cap = capabilities(tmp_path)
    assert cap["weights"] == {"max_net": MAX_NET, "max_gross": MAX_GROSS,
                              "short_selling": True}
    assert cap["costs"] == DEFAULTS["costs"] and "min_trade_bps" in cap["costs"]
    assert set(cap["orders"]["long_only"]) <= set(cap["orders"]["fields"])
    assert "tr_close" in cap["view_fields"] and "volume" in cap["view_fields"]
    assert cap["datasets"]["crsp_daily"]["end"] == "2025-12-31"
    assert cap["universe"] == ["sp500", "futures"] and cap["not_supported"]
    if cap["futures"]["available"]:  # the real lake has the Datastream futures tables
        from concordance.futures.panel import WEIGHT_LIMITS

        assert cap["futures"]["engines"] == ["reference", "nautilus"]
        assert cap["futures"]["weights"]["max_gross"] == WEIGHT_LIMITS["max_gross"]
