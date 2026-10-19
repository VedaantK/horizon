"""Run one strategy through several engines and reconcile them."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from concordance.costs import CostModel, ExecTables, LiquidityModel, exec_tables
from concordance.panel import MarketPanel
from concordance.reconcile import ReconciliationReport, reconcile, valuation_marks
from concordance.result import BacktestResult
from concordance.strategy import Schedule, Strategy, build_schedule

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
    schedule: Schedule | None = None
    tables: ExecTables | None = None

    @property
    def result(self) -> BacktestResult:
        """The reference result (only meaningful once reconciliation passed)."""
        return self.results.get("reference") or next(iter(self.results.values()))


def run_engine(engine: str, panel: MarketPanel, sched: Schedule | tuple[np.ndarray, np.ndarray],
               tables: ExecTables | np.ndarray, init_cash: float) -> BacktestResult:
    """`sched` may be a (weights, has) pair and `tables` a plain fee-rate array, for
    weight-only strategies without the liquidity model."""
    if isinstance(sched, tuple):
        sched = Schedule.from_weights(*sched)
    if isinstance(tables, np.ndarray):
        inf = np.full(tables.shape, np.inf)
        nan = np.full(tables.shape, np.nan)
        tables = ExecTables(tables, inf, inf.copy(), nan, nan.copy())
    return _engines()[engine](panel, sched, tables, init_cash)


def run(panel: MarketPanel, strategy: Strategy, costs: CostModel | None = None,
        init_cash: float = 1_000_000.0, engines: Sequence[str] = ALL_ENGINES,
        strict: bool = True, liquidity: LiquidityModel | None = None) -> Run:
    """Execute `strategy` in every engine. With `strict`, raise if they disagree."""
    sched = build_schedule(strategy, panel)
    tables = exec_tables(panel, costs or CostModel(), liquidity)
    results = {e: run_engine(e, panel, sched, tables, init_cash) for e in engines}
    report = (reconcile(results, valuation_marks(panel.tr_close))
              if len(results) > 1 else None)
    if strict and report is not None and not report.ok:
        raise AssertionError(f"engines disagree:\n{report.summary()}")
    return Run(results, report, sched, tables)
