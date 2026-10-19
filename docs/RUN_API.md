# Run API

One backtest = one JSON request in, one JSON result plus artifact files out.

```bash
concordance backtest request.json -o runs/my_run      # exit code 0 only if status == "ok"
concordance reproduce runs/my_run                      # re-run it and compare
```

```python
from concordance.api import run_backtest
result = run_backtest(request_dict, output_dir="runs/my_run")
```

Failures come back as a `status`, never as an exception, so a coordinator can branch on
them.

## Request

Only `strategy.path`, `data.start` and `data.end` are required. Unknown fields are rejected.

```json
{
  "run_id": "optional; default <UTC timestamp>-<random>",
  "strategy": {"path": "candidates/x/strategy.py", "params": {"lookback": 20}},
  "data": {"universe": "sp500", "start": "2005-01-01", "end": "2019-12-31", "lake": null,
           "customer_links": false},
  "initial_capital": 1000000,
  "engines": ["reference", "vectorbt", "nautilus"],
  "costs": {"commission_bps": 1.0, "default_half_spread_bps": 5.0,
            "use_prior_quoted_spread": true},
  "liquidity": {"participation": 0.10, "impact_coef": 1.0, "max_impact_bps": 500,
                "window": 20, "min_obs": 10},
  "validation": {"program": "my_idea", "placebo": 0, "lookahead_cuts": 3},
  "lockbox": {"unlock_reason": null},
  "report": false
}
```

- `data.customer_links: true` loads `lake/customer_links.parquet` (built by
  `concordance link-customers`) so the strategy can call `view.customer_weights()`. The
  manifest records the file's SHA-256 and link counts.
- `strategy.params` go to the file's `make_strategy(**params)` (docs/STRATEGY_CONTRACT.md).
  A relative `strategy.path` in a request file is relative to that file.
- `data.start` is the first evaluated session; warm-up history before it is loaded
  automatically. `data.end` past the lockbox date (`CONCORDANCE_LOCKBOX`) is refused unless
  `lockbox.unlock_reason` is given, and that read is logged.
- `liquidity: null` turns off the volume cap and market impact.
- `validation.program` names the trial ledger. Every run is recorded in it, and the deflated
  Sharpe counts all trials in the program. Use one program per research question.
- `engines`: a result counts only when engines agree; a single engine is allowed for
  exploration and is flagged in `diagnostics.warnings`.

## Result (`result.json`)

```json
{
  "schema_version": 1,
  "run_id": "...",
  "status": "ok | rejected | lookahead | strategy_error | data_error | invalid_request | error",
  "diagnostics": {"messages": [], "warnings": [], "reconciliation": "...", "traceback": "..."},
  "metrics": {...},
  "artifacts": {"equity": "equity.parquet", "fills": "fills.parquet", ...},
  "manifest": {...}
}
```

| status | meaning |
|---|---|
| `ok` | engines agree; metrics are valid |
| `rejected` | engines disagree: a Concordance bug, never a strategy result |
| `lookahead` | the strategy's decisions change when future data is removed |
| `strategy_error` | the file failed to load, or `decide` raised / returned something invalid, or equity reached 0 with shorts open (`AccountWipedOut`) |
| `data_error` | coverage too low, lockbox, missing lake files |
| `invalid_request` | the request itself is wrong |
| `error` | anything else (traceback in diagnostics) |

### Metrics
Computed from the first evaluated session (after warm-up), on daily equity. No risk-free
rate is subtracted.

| key | definition |
|---|---|
| `cagr` | `(final / initial) ** (252 / sessions) - 1` |
| `vol` | stdev of daily returns * sqrt(252) |
| `sharpe` | mean / stdev of daily returns * sqrt(252) |
| `max_drawdown` | min of equity / running max - 1 |
| `annual_turnover` | traded notional / average equity / years |
| `total_fees` | commissions + spread + impact, in dollars |
| `benchmark_cagr`, `benchmark_sharpe` | rebuilt S&P 500 total return, same window |
| `long_pnl`, `short_pnl` | dollars earned by each side before costs; with `fees_after_start` and `total_borrow` they add up to the change in equity |
| `avg_gross_exposure`, `avg_net_exposure`, `avg_short_exposure` | (longs + shorts), (longs - shorts), shorts, as fractions of equity, averaged over closes |
| `factor_alpha_annual`, `factor_alpha_t` | intercept of daily excess returns on Fama-French 5 + momentum (annualized), with a Newey-West (5 lags) t-stat |
| `factor_betas`, `factor_r2`, `factor_n_days` | loadings on mktrf, smb, hml, rmw, cma, umd; fit; days used |
| `orders` | stop / take-profit exits (at the open and intraday), limit entries placed and filled |
| `dsr` | deflated Sharpe ratio using the ledger's effective number of trials |
| `trials_recorded`, `trials_effective` | trials in this ledger program |
| `placebo_p_value` | share of random-selection portfolios with Sharpe >= this one |
| `equity_sha256` | fingerprint of the equity curve (1e-6), for reproduction checks |

### Artifacts (all in the run directory)
| file | contents |
|---|---|
| `equity.parquet` | date, equity, cash, gross_exposure, n_positions, benchmark_return |
| `fills.parquet` | date, asset (PERMNO), phase (open/intraday), kind (rebalance, carried, stop, take_profit, stop_at_open, take_profit_at_open, limit_entry, delisting), qty, price, fee |
| `positions.parquet` | date, asset, qty, value, weight (non-zero positions only) |
| `orders.parquet` | every resting stop / take-profit / entry-limit order per session: level, triggered, fill price (total-return scale) |
| `strategy.py`, `request.json` | exact copies used for the run (reproduction) |
| `report.html` | quantstats tear sheet (if `report: true`) |

Prices in `fills`/`orders` are on the total-return scale (dividends and splits folded in).

### Manifest
`started_at`, `finished_at`, `elapsed_s`, the resolved request, the strategy file's
SHA-256, Concordance's git commit and whether the tree was dirty, the lake manifest's
SHA-256, the panel shape and coverage, Python/package versions. `concordance reproduce`
re-runs `request.json` with the copied `strategy.py` and reports which of status, equity,
strategy source, data and code match.
