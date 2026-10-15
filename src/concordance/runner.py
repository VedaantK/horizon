"""Run one strategy through several engines and reconcile them."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from concordance.costs import CostModel
from concordance.panel import MarketPanel
from concordance.reconcile import ReconciliationReport, reconcile
from concordance.result import BacktestResult
from concordance.strategy import Strategy, weight_schedule

EngineFn = Callable[..., BacktestResult]


def _engines() -> dict[str, EngineFn]:
    # Imported lazily: Nautilus and vectorbt are heavy to import.
    def reference(*a, **k):
        from concordance.engines import reference as m
        return m.run(*a, **k)

    def vectorbt(*a, **k):
        from concordance.engines import vbt_engine as m
        return m.run(*a, **k)

    def nautilus(*a, **k):
        from concordance.engines import nautilus_engine as m
        return m.run(*a, **k)

    return {"reference": reference, "vectorbt": vectorbt, "nautilus": nautilus}


ALL_ENGINES = ("reference", "vectorbt", "nautilus")


@dataclass
class Run:
    results: dict[str, BacktestResult]
    report: ReconciliationReport | None

    @property
    def result(self) -> BacktestResult:
        """The reference result (only meaningful once reconciliation passed)."""
        return self.results.get("reference") or next(iter(self.results.values()))


def run_engine(engine: str, panel: MarketPanel, weights: np.ndarray, has: np.ndarray,
               fee_rate: np.ndarray, init_cash: float) -> BacktestResult:
    return _engines()[engine](panel, weights, has, fee_rate, init_cash)


def run(panel: MarketPanel, strategy: Strategy, costs: CostModel | None = None,
        init_cash: float = 1_000_000.0, engines: Sequence[str] = ALL_ENGINES,
        strict: bool = True) -> Run:
    """Execute `strategy` in every engine. With `strict`, raise if they disagree."""
    costs = costs or CostModel()
    weights, has = weight_schedule(strategy, panel)
    fee_rate = costs.fee_rates(panel)
    results = {e: run_engine(e, panel, weights, has, fee_rate, init_cash) for e in engines}
    report = reconcile(results) if len(results) > 1 else None
    if strict and report is not None and not report.ok:
        raise AssertionError(f"engines disagree:\n{report.summary()}")
    return Run(results, report)
