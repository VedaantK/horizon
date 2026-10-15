"""Seeded random-portfolio placebo.

Keeps everything about the strategy except the choice of names: the same rebalance dates,
the same number of holdings and the same gross exposure, drawn at random from the assets
that were eligible (universe members with a price) on each decision date. If the real
strategy does not beat most placebos, its selection carries no information.

Replaces QuantMega's placebo.py, whose seeds came from `hash()` and were not reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from concordance.costs import ExecTables
from concordance.engines import reference
from concordance.panel import MarketPanel
from concordance.strategy import Schedule


@dataclass(frozen=True)
class PlaceboResult:
    real_sharpe: float
    placebo_sharpes: np.ndarray
    p_value: float  # (1 + #placebo >= real) / (1 + n)


def _sharpe(equity: np.ndarray) -> float:
    r = np.diff(equity) / equity[:-1]
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def random_selection_placebo(panel: MarketPanel, weights: np.ndarray, has: np.ndarray,
                             tables: ExecTables, init_cash: float = 1e6,
                             n: int = 300, seed: int = 0) -> PlaceboResult:
    """Weights only: stop-loss, take-profit and limit orders are not replayed (they belong
    to the names the strategy chose). The real run is re-done the same way, so both sides
    of the comparison ignore them."""
    def go(w):
        return reference.run(panel, Schedule.from_weights(w, has), tables, init_cash,
                             compiled=True)

    real = go(weights)
    rng = np.random.default_rng(seed)
    T, N = panel.shape
    # Eligible on the decision session (execution row i was decided at i - 1).
    eligible = np.zeros((T, N), dtype=bool)
    eligible[1:] = panel.member[:-1] & np.isfinite(panel.tr_close[:-1])
    rows = np.flatnonzero(has)
    sharpes = np.empty(n)
    for k in range(n):
        w = np.zeros_like(weights)
        for i in rows:
            held = int((weights[i] > 0).sum())
            pool = np.flatnonzero(eligible[i])
            if held == 0 or len(pool) == 0:
                continue
            pick = rng.choice(pool, size=min(held, len(pool)), replace=False)
            w[i, pick] = weights[i].sum() / len(pick)
        sharpes[k] = _sharpe(go(w).equity)
    s = _sharpe(real.equity)
    p = (1 + int((sharpes >= s).sum())) / (1 + n)
    return PlaceboResult(s, sharpes, p)
