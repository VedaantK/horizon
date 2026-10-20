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
   carried quantity; a carried trade that would leave within half a grid step of flat is a
   full exit (`-pos_j`), for longs and shorts alike.
   **Minimum trade:** with `min_trade = CostModel.min_trade_bps / 1e4 > 0`, a new-weight
   trade with `target_j != 0` and `|desired_j| * exec_price_j < min_trade * V_pre` is
   skipped (`desired_j = 0`; the position stays as it is). This covers small resizes and
   tiny new positions. Full exits (`target_j = 0`), forced liquidations, gap exits and
   carried quantities always trade. A trade within float noise of the threshold could be
   decided differently by two engines; reconciliation would then reject the run.
4. **Resting limit buy:** `limit_qty_j = round6(round6(w_j * V_pre / level_j) - pos_j)`,
   capped like a buy; nothing trades at the open for that asset.
5. **Volume cap:** a rebalance trades at most `cap_buy` / `cap_sell` shares per session
   (`costs.exec_tables`); `delta_j` is `desired_j` clipped to the cap (rounded down to the
   grid), and `round6(desired_j - delta_j)` is **carried** to the next open. Shares that
   reduce the position (a sale of a long, a cover of a short) are limited by `cap_sell`;
   shares that open or add to a position on either side by whatever is left of `cap_buy`.
   With known volume the two caps are equal, so the whole trade is capped once; without
   volume history `cap_buy = 0` and `cap_sell = inf`, so exits and covers go through and new
   positions (long or short) wait. Exits (gap, stop, take-profit) and forced liquidations
   are never capped.
6. Fees are `|q| * price * rate_j(q)` with `rate_j(q) = base_fee[i, j] + impact_j(q)`
   (0 for forced liquidations). If open buys plus resting limit buys (at their level), with
   fees, exceed the cash available after the open sells, all of them are scaled by one
   factor and rounded **down** to the 1e-6 grid; fees are recomputed at the new sizes
   (impact only falls). Cash for limit buys is thus reserved at the open, so intraday fill
   order never matters, and a buy never takes cash below zero: no engine ever shrinks or
   rejects an order. Scaling is not carried over. If borrow fees have taken cash below zero
   (see Short positions), buys are scaled to nothing until sells restore it.
7. Open orders execute sells first (ascending asset index), then buys. Short sales are
   sells, so their proceeds are available to the same session's buys. A trade that crosses
   zero (long to short or back) is one order.
8. **Exit quantity:** for a position with a resting stop/take-profit, `exit_qty_j = pos_j +
   delta_j` (the position after the open, exactly). If a resting exit triggers, any carried
   quantity for the asset is dropped, also when the engine held nothing.
9. Exit and entry orders are long-only: a stop-loss, take-profit or entry limit on an
   asset whose target weight is negative is an error.

Fills carry a `phase`: 0 at the open, 1 intraday. An asset has at most one fill per
(session, phase).

## Valuation
- Equity at the close of `i` = `cash + Σ pos_j * val_j`, where `val_j = tr_close[i, j]`, or
  the last finite `tr_close` if the asset has no close that session.
- Positions still open at the end of the window are marked at the last close and reported
  as open. They are never dropped.

## Short positions
- A negative weight is a short: `target_j = round6(w_j * V_pre / exec_price_j) < 0`. Weights
  must satisfy **net** `Σ w_j <= 1` and **gross** `Σ |w_j| <= 2` (`strategy.MAX_NET`,
  `MAX_GROSS`). Net <= 1 means longs are funded only by equity and short proceeds, so a
  rebalance never borrows cash (`cash = V(1 - net)` before costs); gross <= 2 is Reg T's
  50% initial margin (e.g. 100/100, 130/30).
- A short sale credits its proceeds to cash. Valuation is unchanged (`cash + Σ pos_j *
  val_j` with signed `pos_j`), so a short loses whatever the total-return price gains:
  **dividends are paid by the short** through the total-return price, with no separate
  cash flow.
- **Stock borrow:** at each close, after valuation prices are updated, cash is charged
  `borrow_rate * Σ_{pos_j < 0} |pos_j| * val_j`, with `borrow_rate = borrow_bps / 1e4 / 252`
  per session (`CostModel.borrow_bps`, default 50, flat for every stock: the lake has no
  borrow-fee or availability data, and S&P 500 names are almost always general
  collateral). A short opened and covered within one session pays none. Every stock is
  assumed borrowable.
- Borrow fees can take a fully invested book's cash below zero. That debit earns and pays
  no interest; the planner scales buys to nothing while cash is negative (step 6).
- A short in a delisting stock is bought back at its final value, like a long (step 1).
- No maintenance margin or forced buy-ins. If equity reaches 0 with shorts open, the run
  stops with `AccountWipedOut` in every engine.
- Engines: the reference engine charges borrow in its close step. vectorbt trades with
  `Direction.Both`; borrow is computed in its post-segment callback from vectorbt's own
  positions and forward-filled close, kept outside vectorbt (which refuses orders at cash
  <= 0): the planner sees vectorbt's cash minus the running fee total, and the reported
  cash and value are vectorbt's minus the cumulative fee. Nautilus shorts on its cash
  account with `allow_cash_borrowing`, charges borrow through the simulated exchange's
  `adjust_account`, and values positions with signed quantities (its `net_exposures`
  reports a short's notional as positive).

## Costs (`costs.CostModel`)
`base_fee[i, j] = commission_bps/1e4 + half_spread[i, j]`, where the half-spread is the quoted
`(ask - bid) / (ask + bid)` from the **previous** session close, which is known before the open.
If the quote is missing, `default_half_spread_bps` is used. Liquidation fills have zero fee.
`min_trade_bps` (default 0 = off) sets the minimum trade of planning step 3.

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
Fills must match exactly in session, asset and price, with quantities equal to 1e-6. Each
borrow charge is one more rounded money amount (Nautilus rounds it to 1e-9), so it adds one
`1e-9` to the cumulative allowance.

**Grid flips.** Each engine sizes orders from its own pre-trade value, and Nautilus's value can
differ from the float64 engines by the 1e-9 money rounding above. When `w * V_pre / price`
sits within that distance of a half step of the 1e-6 grid, `round6` lands on neighbouring
steps (seen on real CRSP data: 10 of 45,344 fills in a weekly-rebalanced 2000–2022 run). Such
a fill may differ by exactly one step (1e-6), never more. Its fee may differ by at most
`1.5 * fee/|qty| * 1e-6` (the slope bound for a linear rate plus square-root impact). Cash may
additionally differ by the money those flips explain, `sum(price * 1e-6 + |fee gap|)` so far,
and equity also by `|position gap| * valuation price` while a one-step position gap is open.
A flip leaves a small, explained cash gap. On a session whose buys were scaled to fit cash
(step 6, factor `k = cash_after / B` with `B` the buy cost), that gap moves every scaled buy
in proportion, so a scaled buy may also differ by `|qty| * G / B + 1e-6`, where `G` is the
observed cash gap at the previous close, plus the money of the session's differing sells,
plus `Σ_buys price * 1e-6 * (1 + fee rate)` (any pre-scaling buy size may have flipped,
which moves `B`); the extra step covers the rounding-down of the scaled size. A sale that closes a position
may differ by the position gap the asset already had. Any fill at the open may also differ
by `|position after| * G_V / V`: targets are `w * V_pre / price`, so a gap `G_V` in the
pre-trade value (the cash gap plus open position gaps at their marks, at the previous
close) moves every target in proportion (seen on an EWO/RSI variant: a 525-share buy on a
~$30k account moved 2 steps on a $0.0001 value gap). These differences add their money to
the cash/equity allowance exactly like flips. Seen on real CRSP data: a weekly reversal
strategy on $100k, 2000–2022, where 2008 buys of a few thousand dollars moved 2–3 steps (allowances there: median 3.5
steps, at most 240 steps = $0.03 of stock).
A position may differ by at most the sum of explained fill gaps so far in that asset. While a one-step
gap in a short is open it also changes that night's borrow fee, so cash may additionally
differ by `borrow_rate * Σ(open one-step gap values)` so far. Anything else is still a
mismatch.

## Futures (margin account: `futures/engine.py`, `futures/nautilus.py`)

Futures panels have one asset per market. They're built from Panama continuous series
(`futures/continuous.py`), with a roll at the close `sessions_before` sessions before the
front contract's last trade or first-notice deadline.

**What strategies see:**
- `open` and `close`: the held contract's own (unadjusted) prices.
- `ret`: its point change over its own previous settle. That's an excess return with no
  interest included.
- Strategies never see adjusted levels, because those include future roll gaps.

**What engines use:** only differences of adjusted prices. These equal the held contract's
own changes, including across a roll.

Weights are **notional exposure as a fraction of equity**. Net and gross limits come from
the panel: `|net|, gross <= 8` instead of the cash account's 1 and 2.

### Session i, in order

1. **Interest.** Cash earns `rf[i]`, the daily Fama-French risk-free return. Cash is equity,
   because P&L settles daily and positions cost no cash.
2. **Open, when a target was decided at the previous close:**
   - Held contracts mark from the previous adjusted settle to the adjusted open.
   - Targets become integer contracts: `round_half_away(w * E / (mult * open))`, where E is
     equity at the previous close and `open` is unadjusted.
   - If the initial margin needed exceeds `max_margin * E`, every target is scaled by the
     same factor and rounded toward zero. Margin is `|q| * mult * open * margin_rate`.
   - Each change trades at `open`, for a fee per contract of
     `commission + slippage_ticks * tick * mult`.
3. **Close.** Positions mark to the adjusted settle. A market with no bar marks nothing, and
   the next bar marks from the last settle.
4. **Roll.** If the series moves to the next contract after this close, the position keeps
   its contract count. It pays `2 * |q| * fee per contract` (two legs) and is recorded as a
   phase-2 fill with qty 0 at the old contract's settle.

**Engines:** futures reconcile between **reference and Nautilus**. vectorbt has no margined
futures model. Nautilus uses a MARGIN account and `FuturesContract` instruments with the
market multiplier. Its prices are shifted per market by a constant so they stay positive
(differences are unchanged). Interest and roll fees are booked through the venue ledger.
Hand-computed: `tests/golden/test_golden_futures.py`. Random books:
`tests/unit/test_futures_engines.py`.
