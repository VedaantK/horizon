"""The JSON run API, the strategy validator and the authoring driver's spec gate."""

import json
from pathlib import Path

import pandas as pd
import pytest

from concordance.api import run_backtest
from concordance.authoring.drive import check_spec
from concordance.authoring.validate import static_check, validate_file
from concordance.panel import build_panel
from concordance.synthetic import random_long_frame

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "swing_pullback.py"


@pytest.fixture(scope="module")
def panel():
    return build_panel(random_long_frame(7, n_assets=30, n_sessions=320))


def request(panel, path=EXAMPLE, **over):
    req = {"strategy": {"path": str(path), "params": {"trend_days": 60}},
           "data": {"start": str(panel.sessions[0].date()),
                    "end": str(panel.sessions[-1].date())},
           "engines": ["reference", "vectorbt"],
           "liquidity": {"window": 10, "min_obs": 5},
           "validation": {"program": "t", "lookahead_cuts": 2}}
    req.update(over)
    return req


def test_ok_run_writes_artifacts_and_is_deterministic(panel, tmp_path):
    a = run_backtest(request(panel), tmp_path / "a", panel=panel)
    assert a["status"] == "ok", a["diagnostics"]
    out = Path(a["output_dir"])
    for f in ("result.json", "request.json", "strategy.py", "equity.parquet",
              "fills.parquet", "positions.parquet", "orders.parquet"):
        assert (out / f).exists(), f
    saved = json.loads((out / "result.json").read_text())
    assert saved["status"] == "ok" and saved["manifest"]["strategy"]["sha256"]
    assert saved["manifest"]["request"]["strategy"]["params"] == {"trend_days": 60}
    fills = pd.read_parquet(out / "fills.parquet")
    assert set(fills["kind"]) <= {"rebalance", "carried", "stop", "take_profit",
                                  "stop_at_open", "take_profit_at_open", "limit_entry",
                                  "delisting"}
    assert a["metrics"]["orders"]["limit_entries_placed"] > 0
    b = run_backtest(request(panel), tmp_path / "b", panel=panel)
    assert b["metrics"]["equity_sha256"] == a["metrics"]["equity_sha256"]


def test_statuses(panel, tmp_path):
    r = run_backtest({"strategy": {"path": str(EXAMPLE)}}, tmp_path / "x")
    assert r["status"] == "invalid_request" and "data.start" in r["diagnostics"]["messages"][0]
    r = run_backtest(request(panel, bogus=1), tmp_path / "x")
    assert r["status"] == "invalid_request"

    bad = tmp_path / "bad.py"
    bad.write_text("from concordance.strategy import Strategy\n"
                   "class B(Strategy):\n    name = 'b'\n    params = {}\n"
                   "    def decide(self, view):\n        return {'nope': 1.0}\n"
                   "STRATEGY = B()\n")
    r = run_backtest(request(panel, path=bad, strategy={"path": str(bad)}), tmp_path / "y",
                     panel=panel)
    assert r["status"] == "strategy_error" and "unknown assets" in str(r["diagnostics"])

    peek = tmp_path / "peek.py"
    peek.write_text("import numpy as np\nfrom concordance.strategy import Strategy\n"
                    "class P(Strategy):\n    name = 'p'\n    params = {}\n"
                    "    def decide(self, view):\n"
                    "        fut = view._panel.tr_close[-1] / view['tr_close'][-1]\n"
                    "        w = np.zeros(len(view.assets))\n"
                    "        w[int(np.nanargmax(np.nan_to_num(fut)))] = 1.0\n"
                    "        return w\n"
                    "STRATEGY = P()\n")
    r = run_backtest(request(panel, strategy={"path": str(peek)}), tmp_path / "z",
                     panel=panel)
    assert r["status"] == "lookahead"


def test_dsr_check_is_a_verdict_not_a_failure(panel, tmp_path):
    def run(dsr_min, where):
        return run_backtest(request(panel, engines=["reference"], validation={
            "program": "t", "lookahead_cuts": 0, "dsr_min": dsr_min}), tmp_path / where,
            panel=panel)

    for dsr_min, where in ((0.01, "lo"), (0.99, "hi")):
        r = run(dsr_min, where)
        assert r["status"] == "ok", r["diagnostics"]
        c = r["metrics"]["dsr_check"]
        assert c["threshold"] == dsr_min and c["dsr"] == r["metrics"]["dsr"]["dsr"]
        assert c["passed"] == (c["dsr"] >= dsr_min)
        warned = any("deflated Sharpe check failed" in w for w in r["diagnostics"]["warnings"])
        assert warned == (not c["passed"])
    for bad in (0, 1, 1.5, True, "0.9"):
        assert run(bad, "bad")["status"] == "invalid_request"


def test_deflate_runs_matches_the_runs_own_dsr(panel, tmp_path):
    from concordance.api import deflate_runs

    def run(days, where):
        return run_backtest(request(panel, engines=["reference"],
                                    strategy={"path": str(EXAMPLE),
                                              "params": {"trend_days": days}},
                                    validation={"program": "t", "lookahead_cuts": 0}),
                            tmp_path / where, panel=panel)

    a, b = run(60, "a"), run(30, "b")
    # A fresh ledger per run (panel supplied), so each run's own DSR used one trial.
    alone = deflate_runs([a["output_dir"]])
    assert alone["n_effective"] == 1.0
    own = a["metrics"]["dsr"]
    assert alone["runs"][0]["sharpe"] == pytest.approx(own["sharpe"], rel=1e-12)
    assert alone["runs"][0]["dsr"] == pytest.approx(own["dsr"], abs=1e-12)
    assert alone["runs"][0]["sharpe"] == pytest.approx(a["metrics"]["sharpe"], rel=1e-9)
    # Choosing between two versions raises the bar for both.
    both = deflate_runs([a["output_dir"], b["output_dir"]])
    assert both["n_trials"] == 2 and 1.0 <= both["n_effective"] <= 2.0
    assert both["runs"][0]["dsr"] <= alone["runs"][0]["dsr"]
    assert both["runs"][0]["noise_ceiling"] >= alone["runs"][0]["noise_ceiling"]
    with pytest.raises(ValueError):
        deflate_runs([a["output_dir"]], dsr_min=1.0)


def test_static_check_flags_private_access_and_io():
    errs, _ = static_check("import os\nx = view._panel\nimport pandas as pd\n"
                           "pd.read_csv('f')\nSTRATEGY = 1\n")
    checks = {e["check"] for e in errs}
    assert {"imports", "lookahead", "forbidden"} <= checks
    errs, _ = static_check("def make_strategy(): pass\nself._x = 1\n")
    assert errs == []


def test_validator_accepts_the_example():
    rep = validate_file(EXAMPLE, timeout=600)
    assert rep["ok"], rep["errors"]
    assert rep["stats"]["market_1"]["fills"] > 0


def test_author_refuses_specs_with_open_questions():
    spec = {k: "x" for k in ("name", "hypothesis", "signal", "entry", "exit", "sizing",
                             "parameters")}
    assert check_spec(spec) == []
    spec["open_questions"] = ["raw or total-return prices?"]
    assert "open_questions" in check_spec(spec)[0]
    assert "missing 'name'" in check_spec({"hypothesis": "h"})


def test_validator_accepts_month_end_calendar_helper(tmp_path):
    # Synthetic markets used to include exchange holidays (e.g. Good Friday 2015), where
    # is_last_session_of_month raised NotSessionError.
    from concordance.authoring.validate import validate_file

    src = tmp_path / "month_end.py"
    src.write_text(
        "import numpy as np\n"
        "from concordance.calendar import is_last_session_of_month\n"
        "from concordance.strategy import Strategy\n\n\n"
        "class MonthEnd(Strategy):\n"
        "    name = 'month_end'\n\n"
        "    def __init__(self):\n"
        "        self.params = {}\n\n"
        "    def decide(self, view):\n"
        "        if not is_last_session_of_month(view.date):\n"
        "            return None\n"
        "        ok = view.members & np.isfinite(view['tr_close'][-1])\n"
        "        return np.where(ok, 0.9 / max(ok.sum(), 1), 0.0)\n\n\n"
        "def make_strategy(**params):\n"
        "    return MonthEnd(**params)\n")
    report = validate_file(src)
    assert report["ok"], report["errors"]
