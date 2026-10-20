"""The run API on a futures panel: margin engines, reconciliation, metrics, request rules."""

import numpy as np

from concordance.api import resolve_request, run_backtest
from concordance.futures.synthetic import random_futures

STRATEGY = '''
import numpy as np
from concordance.strategy import Strategy


class Flip(Strategy):
    name = "flip"
    params = {}

    def warmup(self):
        return 21

    def decide(self, view):
        if len(view["ret"]) % 21:
            return None
        r = np.nansum(np.asarray(view["ret"])[-21:], axis=0)
        return np.sign(r) * 0.5


def make_strategy():
    return Flip()
'''


def test_futures_run_reconciles_and_reports(tmp_path):
    (tmp_path / "s.py").write_text(STRATEGY)
    panel = random_futures(n_markets=3, seed=3)
    req = {"strategy": {"path": str(tmp_path / "s.py")},
           "data": {"universe": "futures", "start": "2015-01-02", "end": "2017-12-29"},
           "initial_capital": 1_000_000.0, "validation": {"lookahead_cuts": 0}}
    res = run_backtest(req, tmp_path / "out", panel=panel)
    assert res["status"] == "ok", res["diagnostics"]
    assert "[OK ] nautilus vs reference" in res["diagnostics"]["reconciliation"]
    assert res["metrics"]["n_rolls"] > 0 and np.isfinite(res["metrics"]["sharpe"])


def test_futures_requests_refuse_vectorbt_and_stock_data():
    base = {"strategy": {"path": "x.py"},
            "data": {"universe": "futures", "start": "2010-01-01", "end": "2011-01-01"}}
    assert resolve_request(base)["engines"] == ["reference", "nautilus"]
    for bad in ({"engines": ["reference", "vectorbt"]},
                {"data": {**base["data"], "fundamentals": True}},
                {"futures": {"markets": ["XX"]}}):
        try:
            resolve_request({**base, **bad})
        except Exception as e:  # RequestError
            assert "futures" in str(e) or "stock" in str(e) or "markets" in str(e)
        else:
            raise AssertionError(f"accepted {bad}")
