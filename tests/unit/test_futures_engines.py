"""Reference and Nautilus futures engines must agree on random margined books."""

import numpy as np
import pytest

from concordance.futures import nautilus
from concordance.futures.engine import MarginRules, run
from concordance.futures.synthetic import random_futures
from concordance.strategy import Schedule


def random_schedule(panel, seed, every=7, scale=4.0):
    rng = np.random.default_rng(seed)
    T, N = panel.shape
    w = np.zeros((T, N))
    has = np.zeros(T, bool)
    for i in range(1, T, every):
        has[i] = True
        w[i] = rng.normal(0, 1, N)
        w[i] *= scale / np.abs(w[i]).sum()
    return Schedule.from_weights(w, has)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reference_and_nautilus_reconcile(seed):
    panel = random_futures(n_markets=4, seed=seed)
    sched = random_schedule(panel, seed)
    rules = MarginRules(max_margin=0.3)
    ref = run(panel, sched, 1_000_000.0, rules)
    nt = nautilus.run(panel, sched, 1_000_000.0, rules)
    assert ref.fills.shape[0] > 50 and (ref.fills["phase"] == 2).sum() > 5  # trades and rolls
    np.testing.assert_array_equal(ref.positions, nt.positions)
    cols = ["session", "phase", "asset", "qty"]
    np.testing.assert_array_equal(ref.fills[cols].to_numpy(), nt.fills[cols].to_numpy())
    np.testing.assert_allclose(ref.fills["fee"], nt.fills["fee"], rtol=0, atol=1e-6)
    # Money is held at 1e-9 in Nautilus: allow that per fill plus float noise.
    tol = 1e-9 * len(ref.fills) + 1e-9 * np.abs(ref.equity).max()
    assert np.abs(ref.equity - nt.equity).max() <= tol
    assert ref.scaled_sessions == nt.scaled_sessions


def test_adjusted_levels_can_go_negative_without_breaking_pnl():
    panel = random_futures(n_markets=2, seed=5)
    fd = panel.meta["futures"]
    # P&L uses only adjusted differences; they equal the held contract's own changes.
    d_adj = np.diff(fd.adj_close, axis=0)
    same = ~fd.roll[:-1]  # within a contract the unadjusted change must equal it
    d_raw = np.diff(fd.close_unadj, axis=0)
    ok = same & np.isfinite(d_adj) & np.isfinite(d_raw)
    np.testing.assert_allclose(d_adj[ok], d_raw[ok], atol=1e-9)
