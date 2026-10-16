# Strategy contract

A Concordance strategy is **one Python file**. This page is everything an author (human or
AI) needs to write one that the backtester will run. Check a file with:

```bash
concordance validate strategy.py            # synthetic markets, ~15 s
concordance validate strategy.py --real 2015-01-01:2016-12-31   # plus real CRSP data
```

`"ok": true` means the file follows this contract. It says nothing about whether the
strategy is any good.

## File shape

```python
import numpy as np

from concordance.strategy import Orders, Strategy


class MyStrategy(Strategy):
    name = "my_strategy"                     # unique snake_case name

    def __init__(self, lookback=20, top=10):
        self.params = {"lookback": lookback, "top": top}   # EVERY parameter, JSON values
        self.lookback, self.top = lookback, top

    def warmup(self) -> int:                 # sessions of history needed before deciding
        return self.lookback + 1

    def decide(self, view):                  # called after every session close
        ...
        return weights                       # or Orders(...), or None


def make_strategy(**params):                 # how a run request builds it
    return MyStrategy(**params)
```

Rules:
- Allowed imports: `numpy`, `pandas`, `math`, `statistics`, `dataclasses`, `typing`,
  `collections`, `functools`, `itertools`, `warnings`, `enum`, `concordance.strategy`,
  `concordance.calendar`. Nothing that touches files, network, processes or the clock.
- No private attributes of other objects (`view._panel` holds future data and is rejected).
- Deterministic: two fresh instances must make identical decisions. If you need
  randomness, use `np.random.default_rng(<constant>)` created in `__init__`.
- `decide` must be fast: real runs call it ~6,500 times over ~1,300 securities. Use numpy
  over all assets at once; no Python loops over assets.

## Timing

`decide(view)` runs at the **close** of session `i` and sees data up to and including that
close. Whatever it returns executes at the **open of session i+1** (or intraday on i+1 for
resting orders). It can never act on the close it just saw.

## What `view` gives you

| Expression | Shape | Meaning |
|---|---|---|
| `view["tr_close"]` | (i+1, N) | total-return price (dividends and splits included). Use it for **returns, momentum, averages, volatility**. |
| `view["open"]`, `["high"]`, `["low"]`, `["close"]` | (i+1, N) | raw prices. Use them for **price levels**: a $5 floor, a stop/limit price, a breakout above a raw high. |
| `view["ret"]` | (i+1, N) | daily total return (NaN on the first day of a listing) |
| `view["retx"]` | (i+1, N) | daily return without dividends |
| `view["volume"]` | (i+1, N) | shares traded (raw) |
| `view["bid"]`, `view["ask"]` | (i+1, N) | closing quotes |
| `view["shrout"]`, `view["mktcap_k"]` | (i+1, N) | shares outstanding (thousands), market cap ($ thousands) |
| `view["member"]` / `view.members` | (i+1, N) / (N,) | S&P 500 membership; `view.members` is today's row |
| `view.assets` | (N,) | asset ids (CRSP PERMNOs). Columns of every array follow this order. |
| `view.date`, `view.i`, `view.sessions` | | today's date, session index, all dates so far |
| `view.position` | | your own orders' state (below) |
| `view.customer_weights(imputed=True)` | (N, N) | supplier x customer revenue shares known at this close (below). Needs `data.customer_links: true` in the run request. |

Arrays are read-only and contain **NaN** wherever a security has no data (not listed yet,
delisted, halted). Always guard: `np.isfinite(x)`. Row `-1` is today. The universe is every
stock that was in the S&P 500 at any time in the run window; select with `view.members`.

Raw vs total-return: never compute a return from raw prices (a split looks like a -50%
day). Never compare a raw price level to a total-return price.

`view.position` (all arrays of length N, raw prices on today's scale):
- `held` (bool), `entry_session` (int, -1 if flat), `entry_price`,
- `stop_level`, `take_profit_level` (NaN = none),
- `exit_today` (0 none, 1 stop, 2 take-profit filled today).

These describe what your orders intended. The engines may hold less (cash or volume
limits), never more.

`view.customer_weights()` is built from each supplier's 10-K list of customers worth 10%+
of revenue (Compustat customer segments, fiscal years 1998 on). Row = supplier, column =
customer, value = that customer's share of the supplier's revenue (0.35 = 35%); zero where
there is no link. A fiscal year's links appear at the first session after fiscal year-end
+ 6 months and vanish after + 18 months; a newer filing replaces the older one. Shares that
were not disclosed are set to 0.10 (the disclosure threshold); `imputed=False` drops them.
Customers are matched to S&P 500 companies only, so most rows sum to well under 1.
Typical use: `w = view.customer_weights(); s = w.sum(1); cust_ret = (w @ np.nan_to_num(r))
/ np.where(s > 0, s, np.nan)` (customers with a NaN return count as 0 there; mask them in
`w` first if that matters). The matrix is read-only and cached, so calling it every session
is cheap.

`concordance.calendar.is_last_session_of_month(view.date)` is safe to use (the holiday
calendar is published in advance).

## What `decide` returns

- **`None`**: change nothing. Positions and exit orders stay.
- **Weights**: a numpy array of length N, a `dict {asset_id: weight}`, or a pandas Series.
  Target fraction of current equity per asset. Missing or NaN = 0, which **sells** that
  asset. Weights must be >= 0 (long-only) and sum to <= 1 (no leverage). Leftover is cash.
  Returning weights rebalances **every** asset to its target at the next open: include the
  positions you want to keep, or list them in `hold` to leave them exactly as they are.
- **`Orders(...)`**: weights plus exit and entry orders:

```python
Orders(
    weights=w,                        # as above, or None to leave targets alone
    stop_loss={asset: raw_price},     # sell the whole position if the price falls to it
    take_profit={asset: raw_price},   # sell the whole position if the price rises to it
    stop_loss_pct={asset: 0.08},      # stop at entry price * (1 - 0.08)
    take_profit_pct={asset: 0.10},    # take profit at entry price * (1 + 0.10)
    entry_limit={asset: raw_price},   # buy (to its weight) only if the price falls to this
                                      # level next session; cancelled at that close
    hold=view.position.held & ~exit,  # with weights: leave these positions exactly as they
)                                     # are (weight 0/NaN for them). Ids or a bool array.
```

Every order field accepts a dict, a Series, or an array of length N with NaN for "no order".

Exit order rules:
- Exit levels **persist** while the position is open, until you replace them. They vanish
  when the position closes.
- Giving `stop_loss` or `stop_loss_pct` **replaces all stops**: assets not in it lose their
  stop. Same for take-profits. So when you send exit orders, send them for every position
  that should keep one. A field left as `None` changes nothing.
- A `_pct` level is measured from the position's entry price. Send it with the entry
  decision. For a position already held, it is applied to that position's entry price.
- For one asset, give either `stop_loss` or `stop_loss_pct`, not both.
- A trailing stop is a stop you raise yourself each day (e.g.
  `max(view.position.stop_level, close - 2 * atr)`).
- A time stop: compare `view.i - view.position.entry_session` with your limit and set the
  weight to 0.

Entry limit rules:
- Only for opening a position that is currently flat, and the same `Orders` must give that
  asset a positive weight.
- If the next open is already at or below the limit, it fills at the open. Otherwise it
  fills at the limit price if the session's low reaches it, and is cancelled if not.
- The position's stop and take-profit start working the session **after** a limit fill.

## How orders fill (the short version; docs/SEMANTICS.md is authoritative)

- Market orders (weights) fill at the next open.
- A stop or take-profit already crossed at the open fills at the open, and the asset is not
  entered that session.
- Otherwise the session is assumed to move **open -> low -> high -> close**. A stop fills at
  its level when the low reaches it. A take-profit fills at its level when the high reaches
  it. If both are reached the same day, the stop wins. Touching a level counts.
- Costs: commission + half the prior close's bid-ask spread on every trade, plus
  square-root market impact. A rebalance trades at most 10% of the stock's 20-day average
  volume per session; the rest carries to later opens. Exit orders are never capped.
- Delisted stocks are sold at their final value automatically.

## Common mistakes the validator reports

| Mistake | Fix |
|---|---|
| returns from `view["close"]` | use `view["tr_close"]` |
| `np.argsort` over NaNs picks delisted names | mask with `np.isfinite(...) & view.members` first |
| weights for only the new names | weights replace the whole book; include held names or put them in `hold` |
| stop only for new entries | a stop field replaces all stops; restate held ones |
| `view["close"][-1] * 0.95` as a stop for tomorrow's entry | fine, but `stop_loss_pct` measures from the actual fill |
| Python loops over assets | vectorize; the validator warns above 50 ms per decision |
| `self.params` missing a parameter | every constructor argument goes in `self.params` |
