# Memory-bounded stock backtests

Use the run API's opt-in disk mode for the all-US stock universe:

```json
{
  "memory": {
    "mode": "disk",
    "asset_batch_size": 128,
    "bar_batch_sessions": 10
  }
}
```

The default remains `mode: ram`. Disk mode currently requires `data.universe: us_liquid`.
Both batch sizes must be integers in 1–4096; smaller batches trade memory for overhead.
Use a local SSD with sufficient free space. A full-history run can create tens of GB of
temporary files below its output directory. The run-scoped files are deleted on normal
completion or a Python exception; an OS kill can leave them behind. Durable parquet/JSON
result artifacts and the trial ledger remain.

## What changes

1. Universe breakpoints and lagged membership are computed across the entire universe
   once and staged to parquet using DuckDB's 2 GB working-memory limit. Prices are then
   read in bounded asset batches. Each asset keeps its full time series, historical
   membership, listing splits and total-return chain. Market coverage is checked globally.
   Benchmarks combine each batch's numerators and denominators; floating summation order
   can change the last bits of benchmark returns without changing strategy results.
2. Large panel, derived-price, annual-signal, schedule, execution and position arrays use
   run-scoped NumPy memory maps. Price transformations and rolling liquidity calculations
   operate on asset-column blocks; observations across time are not reset.
3. The reference and vectorbt engines still run the complete continuous portfolio.
   Nautilus streams whole-session bar batches using the same engine instance. Its cash,
   positions, orders and matching state survive `run(streaming=True)` / `clear_data()`.
   Only the input bar stream is cleared, with `end()` called once after the last batch.
   The bar cache retains one bar per instrument because execution uses the current bar
   and explicit carried valuation state, not a history of cached bars.
   After the full-universe decision schedule is fixed, instruments with no execution
   role are omitted from Nautilus registration. Each batch feeds bars for held positions,
   pending quantities and that batch's scheduled targets/resting orders. Sentinel bars
   advance and record cash-only sessions. Numeric execution arrays retain the original
   asset indices and dimensions. This reduces event traffic without narrowing the
   ranking universe or changing the order schedule.
4. Engines run sequentially. Each candidate is reconciled against the reference and then
   released; all three engine result histories are not retained in RAM at once. Existing
   reconciliation tolerances, execution rules, trade costs, liquidity and lookahead checks
   remain in force. Large reporting intermediates are also backed by disk where possible.

This is not a collection of independently restarted yearly accounts. There are no
artificial year-end sales, cash resets or lost pending orders at batch boundaries.

## Limits and measurement

Memory maps consume virtual address space even when most pages are not resident in RAM.
A strict 16 GB `RLIMIT_AS` can therefore reject this mode despite modest physical RAM
usage. Measure resident memory (`VmRSS` / maximum RSS) and available system RAM rather
than treating virtual mapping size as physical consumption. The implementation is not
a hard RAM quota: vectorbt's internal simulation buffers, Python objects and OS page
cache still use memory. It does not change OS resource limits.

The run manifest records allocated map bytes/files and batch sizes. Use `/usr/bin/time
-v` for peak RSS. For a long run, also monitor system available memory and interruption
behavior. Temporary disk bytes are cumulative allocation evidence, not peak physical RAM.

Tests compare disk and RAM panels (including relistings), rolling costs, schedules, fills,
cash, positions, borrow fees and equity. Streaming tests force boundaries through pending
liquidity-limited trades, limit orders, stops, shorts and forced liquidations. A separate
real-data smoke comparison must pass before interpreting any full-history result.

## Verified full-history run (2026-10-21)

The annual Value request completed 1998-01-02 through 2019-12-31, with 21 prior warmup
sessions: 5,556 sessions and 7,974 listing histories from 7,725 PERMNOs. All three engines
agreed across 4,780 fills. Maximum equity gaps versus the reference were $1.024e-8 for
vectorbt and $4.377e-8 for Nautilus, within the unchanged reconciliation tolerances.

With asset batches of 128 and bar batches of 10, wall time was 320.05 seconds and maximum
resident memory was 16,876,676 KiB (16.095 GiB). Thus this run fits the actual 28 GiB
machine; it does not establish a strict physical-memory bound below 16 GiB. The earlier
16 GiB address-space cap is a different constraint and is unsuitable for the many
disk-backed virtual mappings. No OS memory limits were changed by the implementation.
All temporary array files were removed after successful artifact generation.

Evidence: `/home/vkulk/wrds_exports/annual_strategy_2026_10_21/memory_completion.json`.
The final output is `full_disk_run_final/` in that directory, with the reusable request
in `full_disk_request.json`. Exact RAM-versus-disk historical smoke parity is recorded
in `memory_final_parity.json`. Two earlier development attempts were interrupted by the
operator for I/O and event-traffic improvements; neither is an accepted result.

The annual accounting revision limitation and one last-price exit at a CRSP listing
gap remain explicit in the final report. Memory optimization does not resolve those
data limitations or provide independent out-of-sample investment evidence.
