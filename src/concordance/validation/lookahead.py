"""Truncation-invariance check: decisions up to session c must not change when every row
after c is removed. Catches strategies that peek at the full panel (e.g. precomputed in
__init__, or via a global) — something the read-only PointInTimeView alone cannot stop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from concordance.panel import MarketPanel, derive_prices
from concordance.strategy import LookaheadError, Strategy, weight_schedule


def truncate(panel: MarketPanel, last: int) -> MarketPanel:
    """Panel containing sessions 0..last only, with derived prices recomputed."""
    f = {k: v[: last + 1].copy() for k, v in panel.fields.items()}
    tr, ex, liq, val = derive_prices(f["open"], f["close"], f["ret"])
    return replace(panel, sessions=panel.sessions[: last + 1], fields=f,
                   member=panel.member[: last + 1].copy(), tr_close=tr, exec_price=ex,
                   liquidate=liq, val_price=val, meta=dict(panel.meta))


def check_truncation_invariance(make_strategy: Callable[[MarketPanel], Strategy],
                                panel: MarketPanel, cuts: int = 5, seed: int = 0) -> None:
    """`make_strategy(panel)` must build a fresh strategy (it may inspect the panel it gets)."""
    full_w, full_has = weight_schedule(make_strategy(panel), panel)
    T = panel.shape[0]
    rng = np.random.default_rng(seed)
    for c in sorted(rng.choice(np.arange(2, T - 1), size=min(cuts, T - 3), replace=False)):
        p = truncate(panel, int(c))
        w, has = weight_schedule(make_strategy(p), p)
        # Rows 0..c of the truncated schedule = decisions taken at sessions < c.
        if not (np.array_equal(has, full_has[: c + 1])
                and np.allclose(w, full_w[: c + 1], rtol=0, atol=0)):
            bad = np.flatnonzero((has != full_has[: c + 1])
                                 | (w != full_w[: c + 1]).any(axis=1))[0]
            raise LookaheadError(
                f"decision executed at {panel.sessions[bad].date()} changes when data after "
                f"{panel.sessions[c].date()} is removed: the strategy uses future data")
