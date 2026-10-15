# Concordance — notes for AI contributors

Multi-engine backtester. Read `README.md` and `docs/SEMANTICS.md` first.

## Rules
- **Never loosen a reconciliation tolerance to make engines agree.** The tolerances in
  `src/concordance/reconcile.py` come from rounding granularity. If engines disagree, find the
  bug (example: Nautilus `Currency.from_str` silently created an 8-decimal currency).
- Any change to execution semantics must be made in **all** engines plus `docs/SEMANTICS.md`,
  with a hand-computed golden test in `tests/golden/`.
- Golden test expectations are derived by hand in comments, never copied from engine output.
- Raw prices: strategy logic only. P&L: total-return prices only.
- Data problems raise errors; never fill them silently.
- `./check.sh` must pass before committing.

## Layout
- `src/concordance/panel.py`: data contract and total-return derivation
- `strategy.py`: view, `Orders`, and `build_schedule` (decisions + order triggers, shared)
- `planner.py`: the shared sizing rule (numba)
- `costs.py`: cost model, liquidity model (volume cap, impact) -> `ExecTables`
- `engines/`: reference, vectorbt, Nautilus, and the hftbacktest bridge
- `reconcile.py`, `runner.py`: run every engine and compare
- `api.py`: JSON run interface, artifacts, manifest, reproduce (docs/RUN_API.md)
- `authoring/`: strategy validator and the Claude/Codex authoring driver; prompts in `prompts/`
- `synthetic.py`: synthetic CRSP-shaped markets (tests, demo, validator)
- `data/`: WRDS ingest, CRSP panels, point-in-time Compustat, lockbox
- `validation/`: DSR, PBO/CPCV, placebo, lookahead, ledger
- `events/`, `futures/`: EDGAR timing, Panama rolls
- `hft/`: separate uv project (numpy<2.3) for hftbacktest

## Environments
`uv sync` (Python 3.13, main) and `cd hft && uv sync` (isolated). The system Python is 3.14
and is not used.

## v1 limits (not yet implemented)
Long-only, no leverage, fractional shares (1e-6 grid), daily bars. Orders: market at the
open, stop/take-profit exits and next-session limit entries on the documented intraday path
(no stop entries, no limit exits other than take-profit, no trailing stops built in).
Costs: commission + prior-close half spread + square-root impact; volume cap on rebalances.
Fundamentals are restated values, not as-first-reported.
