# Execution semantics (the contract every engine implements)

All engines (reference, vectorbt, Nautilus) implement exactly these rules. A result is
accepted only if every engine produces the same fills, cash, positions and equity within
a tolerance that comes from known rounding granularity, never from tuning.

## Timeline
- Sessions `i = 0 … T-1` are exchange sessions (XNYS for US equities).
- At the **close** of session `i`, the strategy sees data for sessions `0 … i` only (the
  `PointInTimeView` physically truncates every array at `i`). It returns target weights,
  an `Orders` object, or `None` (= leave positions and orders unchanged).
- Decisions are executed at the **open of session i+1**, or for resting orders during
  session i+1. A decision made at the last session is never executed.

## Prices
- Raw CRSP prices (`open`, `close`, …) are visible to strategies for price-level logic
  (e.g. a $5 floor). They are never used for P&L.
- P&L uses a **total-return price** built causally from the first observation:
  `tr_close[i] = tr_close[prev] * (1 + ret[i])` and `tr_open[i] = tr_close[i] * open[i] / close[i]`.
  CRSP `ret` includes dividends, splits and (in CIZ/v2) the delisting return. The
  intraday leg uses the raw same-day ratio, so a buyer at the open of an ex-date does
  not receive the dividend and the seller does.
- Prices are rounded to 1e-9; quantities to 1e-6. All engines see the same rounded values.

## CRSP rows without a return (`data/ingest._unchained_rows`)
A missing `ret` on a priced row is never treated as 0. Three patterns occur in the real
1990–2025 export, and each is handled by an explicit rule recorded in `lake/manifest.json`:
- **Trading halt** (blank row, no price and no return, e.g. BIIB 2020-11-06): the blank day
  is skipped. CRSP measures the next return from the last valid price, so it spans the halt.
  The asset has no open that day, so it can't be traded.
- **Final row with a price but no return** (e.g. SVB and Signature Bank on 2023-03-13):
  `ret = close / previous close - 1`, listed under `final_return_from_price`.
- **Listing gap**: the stock left its exchange and later came back, and CRSP has no return
  spanning the gap (e.g. AMR 2012–13, SMCI 2018–20). It is listed under `listing_gaps`.
  The loader splits the PERMNO into one security per listing:
  - The first listing keeps the PERMNO as its asset id; the k-th relisting gets
    `PERMNO + k * 1_000_000` (`crsp.permno_of` recovers the PERMNO). Ids only reflect
    relistings that have already happened.
  - A position in the earlier listing is force-sold at the last price before the gap.
    This is an assumption, so `concordance run` prints a warning whenever a strategy held
    such a position. Gaps that began while the stock was an S&P member are flagged
    `member_at_gap`.
  - No return or signal is ever chained across the gap.
  - Membership and fundamentals follow the PERMNO across all of its listings.

Any other priced row without a return still makes the panel refuse to build.

## Decisions and the order schedule (`strategy.build_schedule`)
A decision at the close of session `i` is target weights, an `Orders` object (weights plus
exit and entry orders), or `None`. `build_schedule` runs the strategy once and resolves
everything that depends only on prices and the strategy's own levels (never on cash):

- **Level scale.** A raw price level set at the close of `d` is stored on the total-return
  scale: `level_tr = round9(raw * tr_close[d] / close[d])` (ratio forward-filled if the
  asset has no close on `d`). A split or dividend therefore never triggers an order. A `_pct`
  level is `entry_tr * (1 - pct)` (stop) or `entry_tr * (1 + pct)` (take-profit), where
  `entry_tr` is the fill price of the trade that opened the position.
- **Persistence.** Stop and take-profit levels persist while the position is open, and are
  dropped when it closes. Giving a stop field (`stop_loss` or `stop_loss_pct`) replaces every
  stop; likewise for take-profits. An entry limit lives for the next session only.
- **Intended holdings.** The schedule tracks which assets the orders hold (`held`). An
  asset is held after the open of `i` if (with new targets) its weight is > 0, or it is in
  `hold`, or (without new targets) it was held; an untradable asset keeps its state. This
  is what `view.position` shows. Engines may hold less than intended (cash or volume
  limits), never more.

### Session i, in order
1. **Open.** Forced liquidations (below). Then, for tradable assets:
   - an entry limit with `tr_open <= level` is marketable: it is an ordinary open buy;
     otherwise it rests (step 3);
   - a position held after the open whose stop is at or above `tr_open`, or whose
     take-profit is at or below it, is **exited at the open** (`gap_exit`): its target is 0,
     overriding the weights, and an asset about to be entered is not entered.
2. **Resting exits.** For a position held after the open, the session's path is assumed to
   be **open -> low -> high -> close** on the total-return scale, where low/high are
   `round9(tr_close * raw_low / raw_close)` (likewise high) and the range is widened to
   include the open and the close (vendor ranges that exclude them occur). The stop fills at
   its level if `low <= stop`; otherwise the take-profit fills at its level if
   `high >= take_profit`. Touching counts. The whole position is sold.
3. **Resting entry limit.** Fills at the level if `low <= level`; the position's exit
   levels (from `_pct`) start working the next session. Otherwise it is cancelled.
4. **Close.** Valuation; the strategy decides for session `i+1`.

Entry limits may only open a flat position with a positive weight in the same `Orders`;
`hold` assets must have weight 0. Violations raise `ValueError`.

## Order planning at the open of session i (`planner.plan_session`)
1. **Forced liquidation:** an asset whose data ended at `i-1` while the panel continues
   (delisting or leaving the dataset) is sold in full at `exec_price = tr_close[last]`
   with zero fee. Its target weight is ignored. A `gap_exit` likewise sells the whole
   position, at the open price with normal fees.
2. `V_pre = cash + Σ pos_j * mark_j`, where `mark_j` is the execution price if the asset
   trades this session, otherwise its last valuation price.
3. With new weights `w` (assets not in `hold`): `target_j = round6(w_j * V_pre /
   exec_price_j)` for tradable assets and `desired_j = round6(target_j - pos_j)`, except a
   full exit (`target_j = 0`), which is `-pos_j` exactly. New weights drop any carried
   quantity (step 5) for those assets. An asset with no open price this session is **not
   tradable**: it keeps its position. Without new weights (or in `hold`), `desired_j` is the
   carried quantity; a carried sell within half a grid step of the position is a full exit.
4. **Resting limit buy:** `limit_qty_j = round6(round6(w_j * V_pre / level_j) - pos_j)`,
   capped like a buy; nothing trades at the open for that asset.
5. **Volume cap:** a rebalance trades at most `cap_buy` / `cap_sell` shares per session
   (`costs.exec_tables`); `delta_j` is `desired_j` clipped to the cap (rounded down to the
   grid), and `round6(desired_j - delta_j)` is **carried** to the next open. Exits (gap,
   stop, take-profit) and forced liquidations are never capped.
6. Fees are `|q| * price * rate_j(q)` with `rate_j(q) = base_fee[i, j] + impact_j(q)`
   (0 for forced liquidations). If open buys plus resting limit buys (at their level), with
   fees, exceed the cash available after the open sells, all of them are scaled by one
   factor and rounded **down** to the 1e-6 grid; fees are recomputed at the new sizes
   (impact only falls). Cash for limit buys is thus reserved at the open, so intraday fill
   order never matters, and cash never goes negative: no engine ever shrinks or rejects an
   order. Scaling is not carried over.
7. Open orders execute sells first (ascending asset index), then buys.
8. **Exit quantity:** for a position with a resting stop/take-profit, `exit_qty_j = pos_j +
   delta_j` (the position after the open, exactly). If a resting exit triggers, any carried
   quantity for the asset is dropped, also when the engine held nothing.
9. Long-only in v1: a negative weight is an error.

Fills carry a `phase`: 0 at the open, 1 intraday. An asset has at most one fill per
(session, phase).

## Valuation
- Equity at the close of `i` = `cash + Σ pos_j * val_j`, where `val_j = tr_close[i, j]`, or
  the last finite `tr_close` if the asset has no close that session.
- Positions still open at the end of the window are marked at the last close and reported
  as open. They are never dropped.

## Costs (`costs.CostModel`)
`base_fee[i, j] = commission_bps/1e4 + half_spread[i, j]`, where the half-spread is the quoted
`(ask - bid) / (ask + bid)` from the **previous** session close, which is known before the open.
If the quote is missing, `default_half_spread_bps` is used. Liquidation fills have zero fee.

## Liquidity (`costs.LiquidityModel`, optional)
Known before the open of `i`, from sessions `< i` only (rolling `window`, at least
`min_obs` observations, else unknown):
- `ADV[i, j]` = mean of `volume * close / tr_close`: share volume on the total-return scale,
  so splits do not change it;
- `sigma[i, j]` = standard deviation (ddof 1) of `ret`.

`cap_buy = cap_sell = participation * ADV`. Unknown ADV: buys are blocked (cap 0), sells are
uncapped. Impact rate `impact_j(q) = min(impact_coef * sigma * sqrt(q / ADV), max_impact)`;
`max_impact` also applies when ADV or sigma is unknown. With the model off, caps are infinite
and impact is 0.

## How each engine executes intraday orders
- **reference** and **vectorbt** fill resting orders where the schedule says they
  triggered, at the schedule's price (vectorbt in flexible mode, after the open orders).
- **Nautilus** places real orders after the open fills: STOP_MARKET and LIMIT sells
  (reduce-only; when one fills the other is cancelled) and LIMIT buys. It receives two
  range bars per asset with a resting order, `(O, H=O, L, C=L)` then `(O=L, H, L, C)`,
  which walk the documented path, and its own matching engine decides whether and at what
  price each order fills. Unfilled orders are cancelled at the close. Its fee model computes
  impact itself. So the trigger rule is checked independently every run.

## Tolerance
Nautilus stores money at 1e-9 precision and rounds each fee. The reference and vectorbt use
float64. The allowed equity difference is `1e-9 * cumulative_fills + 1e-9 * |equity|`.
Fills must match exactly in session, asset and price, with quantities equal to 1e-6.

**Grid flips.** Each engine sizes orders from its own pre-trade value, and Nautilus's value can
differ from the float64 engines by the 1e-9 money rounding above. When `w * V_pre / price`
sits within that distance of a half step of the 1e-6 grid, `round6` lands on neighbouring
steps (seen on real CRSP data: 10 of 45,344 fills in a weekly-rebalanced 2000–2022 run). Such
a fill may differ by exactly one step (1e-6), never more. Its fee may differ by at most
`1.5 * fee/|qty| * 1e-6` (the slope bound for a linear rate plus square-root impact). Cash may
additionally differ by the money those flips explain, `sum(price * 1e-6 + |fee gap|)` so far,
and equity also by `|position gap| * valuation price` while a one-step position gap is open.
A position may differ by at most one step per flip so far in that asset. Anything else is
still a mismatch.
