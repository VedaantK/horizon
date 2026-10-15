# Concordance

**A backtest result counts only when independent engines agree.**

Concordance runs every strategy through three engines and accepts the result only if
their fills, positions, cash and equity reconcile:

| Engine | Role | Independent parts |
|---|---|---|
| **reference** | Plain-Python oracle, written to be audited | Everything |
| **vectorbt** 1.1.2 | Fast research and parameter sweeps | Fill engine, cash, positions, valuation |
| **Nautilus Trader** 1.231 | Event-driven matching engine, the same code that trades live | Matching, cash account, fee model, portfolio valuation |
| **hftbacktest** 2.4.4 | Queue position and latency for market making (separate environment) | Its own L2/L3 simulator |

Shared between the engines: the data panel and one documented sizing rule
([`docs/SEMANTICS.md`](docs/SEMANTICS.md)). Both have their own golden tests.

## What's enforced
- **No look-ahead:** strategies see a read-only view truncated at the decision close and fill
  at the next open. A truncation-invariance check catches strategies that peek another way.
- **No survivorship bias:** CRSP universes come from S&P 500 membership spells keyed by
  PERMNO. Delisting returns are kept. Positions in delisted stocks are force-liquidated at
  their final value, never dropped.
- **Correct corporate actions:** P&L chains CRSP total returns. Raw prices are used only for
  same-day logic such as a $5 floor.
- **No silent repairs:** a missing return mid-history, a conflicting duplicate, low coverage or
  a lockbox breach raises an error.
- **Multiple-testing honesty:** every run goes into a trial ledger. The Deflated Sharpe uses
  the measured effective number of independent trials. Also available: overfitting
  probability (PBO / CSCV), purged combinatorial cross-validation, and a seeded random-selection placebo.

## Quick start
```bash
uv sync                 # main environment (Python 3.13)
(cd hft && uv sync)     # isolated hftbacktest environment
./check.sh              # lint + all tests (golden, randomized reconciliation, hft)
uv run concordance demo --report reports/demo.html
```

With WRDS data (see [`docs/WRDS_EXPORT.md`](docs/WRDS_EXPORT.md)):
```bash
uv run concordance ingest ~/wrds_exports
uv run concordance run examples/momentum_12_1.py --start 2000-01-01 --end 2021-12-31 \
    --placebo 200 --report reports/mom.html
```
Set `CONCORDANCE_LOCKBOX=2022-01-01` to seal recent data until a logged one-shot evaluation.

## Writing a strategy
```python
from concordance.strategy import Strategy

class MyStrategy(Strategy):
    def warmup(self): return 252
    def decide(self, view):            # called at each session close
        tr = view["tr_close"]          # total-return prices up to today only
        ...
        return weights                 # {asset: w} / array; None = hold. Long-only, sum <= 1
STRATEGY = MyStrategy()
```

## What came from QuantMega
Kept, and only after fixes and new tests: the CRSP CIZ column conventions; the Compustat
pull field lists (plus a new point-in-time join); the eventlab backtester's design (as the
reference engine); the DSR core and `effective_independent_trials`; EDGAR 8-K timing; the
Panama roll builder. Everything else was discarded. See `docs/QUANTMEGA_AUDIT.md`.
