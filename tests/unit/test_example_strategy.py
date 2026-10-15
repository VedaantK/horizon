import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from concordance.costs import CostModel
from concordance.data.crsp import xnys_sessions
from concordance.panel import build_panel
from concordance.runner import run
from concordance.validation.lookahead import check_truncation_invariance

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("mom", ROOT / "examples" / "momentum_12_1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _market(n=30, start="2018-01-02", end="2019-12-31", seed=0):
    rng = np.random.default_rng(seed)
    sess = xnys_sessions(start, end)
    rows = []
    for j in range(n):
        drift = rng.normal(0.0004, 0.0004)
        c = float(rng.uniform(4, 200))
        for k, d in enumerate(sess):
            r = rng.normal(drift, 0.015)
            prev = c
            c = c * (1 + r)
            rows.append({"date": d, "asset": 1000 + j, "open": prev * (1 + r / 2), "close": c,
                         "ret": np.nan if k == 0 else r, "member": j % 5 != 0 or k > 300})
    return build_panel(pd.DataFrame(rows))


def test_momentum_example_runs_reconciles_and_has_no_lookahead():
    m = _load()
    panel = _market()
    out = run(panel, m.Momentum12_1(), CostModel(commission_bps=1), init_cash=1e6)
    assert out.report.ok, out.report.summary()
    ref = out.results["reference"]
    assert len(ref.fills) > 0
    # Rebalances only on month-end decisions -> fills only on the first session of a month.
    fill_days = panel.sessions[ref.fills["session"].unique()]
    prev = panel.sessions[ref.fills["session"].unique() - 1]
    assert (fill_days.month != prev.month).all()
    check_truncation_invariance(lambda p: m.Momentum12_1(), panel, cuts=4)
