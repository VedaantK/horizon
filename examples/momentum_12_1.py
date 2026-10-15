"""12-1 momentum on S&P 500 members: hold the top decile by trailing 12-month total return
skipping the most recent month, equal-weighted, rebalanced on the last session of each month.

Run:  uv run concordance run examples/momentum_12_1.py --start 2000-01-01 --end 2021-12-31
"""

import numpy as np

from concordance.calendar import is_last_session_of_month
from concordance.strategy import Strategy

LOOKBACK = 252
SKIP = 21


class Momentum12_1(Strategy):
    name = "momentum_12_1"

    def __init__(self, top: float = 0.1, min_price: float = 5.0, gross: float = 0.98):
        self.params = {"top": top, "min_price": min_price, "gross": gross}
        self.top, self.min_price, self.gross = top, min_price, gross

    def warmup(self) -> int:
        return LOOKBACK

    def decide(self, view):
        if not is_last_session_of_month(view.date):
            return None
        tr = view["tr_close"]  # total-return prices: right for returns
        with np.errstate(invalid="ignore", divide="ignore"):
            mom = tr[-SKIP - 1] / tr[-LOOKBACK - 1] - 1.0
        raw_close = view["close"][-1]  # raw prices: right for a price floor
        ok = view.members & np.isfinite(mom) & (raw_close >= self.min_price)
        if ok.sum() < 10:
            return None
        k = max(1, int(ok.sum() * self.top))
        idx = np.flatnonzero(ok)
        winners = idx[np.argsort(mom[idx], kind="stable")[-k:]]
        w = np.zeros(len(view.assets))
        w[winners] = self.gross / k
        return w


STRATEGY = Momentum12_1()
