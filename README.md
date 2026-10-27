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
- **Realistic orders:** stop-loss, take-profit (absolute or % from entry) and limit-entry
  orders fill on a documented intraday path (open -> low -> high -> close). Nautilus places
  them as real orders and decides independently whether and where they fill.
- **Liquidity:** a rebalance trades at most a set share of trailing average volume per
  session (the rest carries to later opens), and every trade pays square-root market impact.
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
uv run concordance backtest request.json -o runs/x     # JSON in, JSON + parquet out
uv run concordance reproduce runs/x                    # re-run and compare
```
Every run writes a directory with `result.json` (status, metrics, diagnostics, a
reproducibility manifest), equity/fills/positions/orders as parquet, and copies of the
strategy and request. See [`docs/RUN_API.md`](docs/RUN_API.md).
Set `CONCORDANCE_LOCKBOX=2022-01-01` to seal recent data until a logged one-shot evaluation.

## Writing a strategy
```python
from concordance.strategy import Orders, Strategy

class MyStrategy(Strategy):
    name = "my_strategy"
    def __init__(self, lookback=20):
        self.params = {"lookback": lookback}
        self.lookback = lookback
    def warmup(self): return self.lookback + 1
    def decide(self, view):            # called at each session close
        tr = view["tr_close"]          # total-return prices up to today only
        ...
        return Orders(weights=w, stop_loss_pct=..., take_profit_pct=..., entry_limit=...)
        # or plain weights, or None = hold. Negative = short; net <= 1, gross <= 2

def make_strategy(**params):
    return MyStrategy(**params)
```
The full contract (data fields, order rules, common mistakes) is
[`docs/STRATEGY_CONTRACT.md`](docs/STRATEGY_CONTRACT.md). `concordance validate file.py`
checks a file against it. [`examples/swing_pullback.py`](examples/swing_pullback.py) uses
every order type.

## Writing strategies with AI tools
`concordance formalize` turns an idea into a schema-checked spec; `concordance author`
has Claude Code or Codex implement it in an isolated workspace, with the validator in the
loop. See [`prompts/README.md`](prompts/README.md).

## What came from QuantMega
Kept, and only after fixes and new tests: the CRSP CIZ column conventions; the Compustat
pull field lists (plus a new point-in-time join); the eventlab backtester's design (as the
reference engine); the DSR core and `effective_independent_trials`; EDGAR 8-K timing; the
Panama roll builder. Everything else was discarded. See `docs/QUANTMEGA_AUDIT.md`.

## Intraday research execution

`concordance intraday --bars minute_bars.parquet --signals signals.parquet -o runs/minute`
executes completed-bar stock signals at the next minute open. It is a separate
minute-bar interface; the daily `backtest` request and three-engine contract above are
unchanged. Minute execution reconciles a reference event loop with an independent
replay loop. It requires prior-minute NBBO quotes and volume, supports long and short
stops/targets, resolves same-minute stop/target ambiguity against the trade, and exits
positions at each asset's last regular-session bar. See the sibling
`/home/vkulk/intraday-research` repository for the data contract and research workflow.
