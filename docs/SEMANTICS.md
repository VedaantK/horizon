# Execution semantics (the contract every engine implements)

All engines (reference, vectorbt, Nautilus) implement exactly these rules. A result is
accepted only if every engine produces the same fills, cash, positions and equity within
a tolerance that comes from known rounding granularity, never from tuning.

## Timeline
- Sessions `i = 0 … T-1` are exchange sessions (XNYS for US equities).
- At the **close** of session `i`, the strategy sees data for sessions `0 … i` only (the
  `PointInTimeView` physically truncates every array at `i`). It returns target weights
  or `None` (= leave positions unchanged).
- Decisions are executed at the **open of session i+1**. A decision made at the last
  session is never executed.

## Prices
- Raw CRSP prices (`open`, `close`, …) are visible to strategies for price-level logic
  (e.g. a $5 floor). They are never used for P&L.
- P&L uses a **total-return price** built causally from the first observation:
  `tr_close[i] = tr_close[prev] * (1 + ret[i])` and `tr_open[i] = tr_close[i] * open[i] / close[i]`.
  CRSP `ret` includes dividends, splits and (in CIZ/v2) the delisting return. The
  intraday leg uses the raw same-day ratio, so a buyer at the open of an ex-date does
  not receive the dividend and the seller does.
- Prices are rounded to 1e-9; quantities to 1e-6. All engines see the same rounded values.

## Order planning at the open of session i (`planner.plan_orders`)
1. **Forced liquidation:** an asset whose data ended at `i-1` while the panel continues
   (delisting or leaving the dataset) is sold in full at `exec_price = tr_close[last]`
   with zero fee. Its target weight is ignored.
2. `V_pre = cash + Σ pos_j * mark_j`, where `mark_j` is the execution price if the asset
   trades this session, otherwise its last valuation price.
3. With pending weights `w`: `target_j = round6(w_j * V_pre / exec_price_j)` for tradable
   assets. An asset with no open price this session is **not tradable**: it keeps its
   position. Without pending weights, targets equal current positions (except liquidations).
4. Orders are `delta_j = target_j - pos_j`: all sells first (ascending asset index), then
   all buys (ascending asset index).
5. Fees are `|delta| * price * fee_rate[i, j]`. If buys plus fees would exceed the cash
   available after the sells, every buy is scaled by the same factor and rounded **down**
   to the 1e-6 grid. Cash never goes negative, so no engine ever shrinks or rejects an order.
6. Long-only in v1: a negative weight is an error.

## Valuation
- Equity at the close of `i` = `cash + Σ pos_j * val_j`, where `val_j = tr_close[i, j]`, or
  the last finite `tr_close` if the asset has no close that session.
- Positions still open at the end of the window are marked at the last close and reported
  as open. They are never dropped.

## Costs (`costs.CostModel`)
`fee_rate[i, j] = commission_bps/1e4 + half_spread[i, j]`, where the half-spread is the quoted
`(ask - bid) / (ask + bid)` from the **previous** session close, which is known before the open.
If the quote is missing, `default_half_spread_bps` is used. Liquidation fills have zero fee.

## Tolerance
Nautilus stores money at 1e-9 precision and rounds each fee. The reference and vectorbt use
float64. The allowed equity difference is `1e-9 * cumulative_fills + 1e-9 * |equity|`.
Fills must match exactly in session, asset and price, with quantities equal to 1e-6.
