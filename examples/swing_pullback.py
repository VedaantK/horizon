"""Swing pullback on S&P 500 members: buy a short dip inside a long uptrend, with a limit
entry below the close, an 8% stop, a 10% take-profit and a 10-session time stop.

Hypothesis: short-term overreaction in stocks that are in a long-term uptrend tends to
partly reverse within days (short-horizon reversal conditional on trend).

Run:  uv run concordance run examples/swing_pullback.py --start 2010-01-01 --end 2019-12-31
"""

import numpy as np

from concordance.strategy import Orders, Strategy


class SwingPullback(Strategy):
    name = "swing_pullback"

    def __init__(self, trend_days=200, dip_days=3, dip_pct=0.04, max_positions=10,
                 entry_discount=0.01, stop_pct=0.08, take_profit_pct=0.10, max_hold=10,
                 min_price=5.0):
        self.params = {"trend_days": trend_days, "dip_days": dip_days, "dip_pct": dip_pct,
                       "max_positions": max_positions, "entry_discount": entry_discount,
                       "stop_pct": stop_pct, "take_profit_pct": take_profit_pct,
                       "max_hold": max_hold, "min_price": min_price}
        self.trend_days, self.dip_days, self.dip_pct = trend_days, dip_days, dip_pct
        self.max_positions, self.entry_discount = max_positions, entry_discount
        self.stop_pct, self.take_profit_pct = stop_pct, take_profit_pct
        self.max_hold, self.min_price = max_hold, min_price

    def warmup(self) -> int:
        return self.trend_days + 1

    def decide(self, view):
        tr = view["tr_close"]  # total-return prices: right for returns and averages
        close = view["close"][-1]  # raw: right for price levels and the $5 floor
        window = tr[-self.trend_days:]
        n_obs = np.isfinite(window).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            # average over observed days; require 90% of the window (trading halts)
            sma = np.where(n_obs >= 0.9 * self.trend_days, np.nansum(window, axis=0) / n_obs,
                           np.nan)
            dip = tr[-1] / tr[-1 - self.dip_days] - 1.0
            in_trend = tr[-1] > sma
            dipped = dip < -self.dip_pct
        pos = view.position
        held = pos.held
        age = np.where(held, view.i - pos.entry_session, 0)
        time_out = held & (age >= self.max_hold)
        keep = held & ~time_out

        slots = self.max_positions - int(keep.sum())
        ok = (view.members & ~held & in_trend & dipped & np.isfinite(close)
              & (close >= self.min_price))
        new = np.zeros(len(view.assets), dtype=bool)
        if slots > 0 and ok.any():
            idx = np.flatnonzero(ok)
            new[idx[np.argsort(dip[idx], kind="stable")[:slots]]] = True  # deepest dips
        if not time_out.any() and not new.any():
            return None  # nothing changes: keep positions and their exit orders

        w = np.zeros(len(view.assets))
        w[new] = 1.0 / self.max_positions
        exits = keep | new
        return Orders(
            weights=w,
            hold=keep,  # kept positions are not resized; time-outs (weight 0) are sold
            entry_limit=np.where(new, close * (1 - self.entry_discount), np.nan),
            # Exit fields replace all exit orders, so restate them for kept positions too
            # (a % level on a held position is measured from its own entry price).
            stop_loss_pct=np.where(exits, self.stop_pct, np.nan),
            take_profit_pct=np.where(exits, self.take_profit_pct, np.nan),
        )


def make_strategy(**params):
    return SwingPullback(**params)
