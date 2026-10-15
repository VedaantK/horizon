# QuantMega salvage audit (2026-10-14)

Default verdict: discard. A piece was kept only if it was correct (or fixable), self-contained,
and not already provided by vectorbt, Nautilus, hftbacktest, skfolio or quantstats.

## Kept (ported, fixed, re-tested)
| QuantMega | Concordance | Fixes |
|---|---|---|
| `quantlab/crsp.py` conventions | `data/schema.py`, `data/ingest.py`, `data/crsp.py` | Duplicates that conflict now raise; parameterized SQL; membership keyed by PERMNO with inclusive end dates; coverage gate; reproducible export spec |
| `scripts/pull_wrds_compustat.py` field lists | `data/fundamentals.py` | Point-in-time join added (first session after `rdq`); YTD→quarterly; link valid on the availability date; a late-filed old quarter never overrides a newer one |
| `eventlab/backtest.py` design | `engines/reference.py` | A missing return now raises (was treated as 0); explicit liquidation at delisting; positions open at the window end are reported |
| `validation/deflated_sharpe.py` | `validation/dsr.py` | A non-positive variance term now gives DSR = 0 (was inflated); verified against the paper (0.9004) and vectorbt |
| `eventlab/{events,edgar}.py` timing | `events/edgar.py` | Early-close sessions handled (was a fixed 16:00 close) |
| `futures/continuous.py` Panama | `futures/continuous.py` | Real expiry and first-notice dates; roll only moves earlier; look-ahead in adjusted levels documented, causal `point_changes` added |
| Ideas | lockbox, coverage refusal, trial ledger, placebo | Placebo is now seeded and reproducible |

## Discarded (main reasons)
- `quantlab/backtest.py`:
  - drops trades still open at delisting or at the end of the data;
  - deletes returns outside [−99%, +100%];
  - applies the $5 filter to split-adjusted prices (look-ahead);
  - has no cash accounting;
  - lets rejected trades block later signals.
- `sp500_historical_constituents.csv`: ends 2019-01, keyed by ticker; the yfinance universe covers only 59% of names.
- `walkforward.py`: never re-fits parameters.
- `parameter_stability.py`: clamps negative integers to 1.
- `capital_feasibility.py`: same-day trades never exit; shorts are booked as longs.
- `placebo.py`: seeded from `hash()`, so not reproducible.
- `indicators.py`: RSI is 0 when there are no losses.
- `futures/backtest.py`: books roll gaps as P&L.
- `montecarlo.py`, `regime.py`, `scoring.py`, `metrics.py`, `costs.py`, `calendars.py`, `instruments.py`: redundant with libraries.
- All of `terminal/`: live-trading system, not backtesting.
