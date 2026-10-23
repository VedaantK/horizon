"""Exploratory annual Value-then-profitability, formed at the last June close.

Requires data.fundamentals_basis=annual_lagged. Current annual values can contain
later revisions. This is a separate strategy from the first-reported quarterly study.
"""

import numpy as np

from concordance.calendar import is_last_session_of_month
from concordance.strategy import Strategy


class ValueAnnual(Strategy):
    name = "value_then_profitability_annual_lagged"
    fundamentals_basis = "annual_lagged"

    def __init__(self, value_fraction=0.5, profit_fraction=0.2, gross=0.8):
        if not all(np.isfinite(v) and 0 < v <= 1
                   for v in (value_fraction, profit_fraction, gross)):
            raise ValueError("fractions and gross must be finite and in (0, 1]")
        self.value_fraction = float(value_fraction)
        self.profit_fraction = float(profit_fraction)
        self.gross = float(gross)
        self.params = {"value_fraction": value_fraction,
                       "profit_fraction": profit_fraction, "gross": gross}

    def warmup(self):
        return 1

    def decide(self, view):
        # Exchange sessions, not the next row of the panel: truncating a run must
        # not turn an arbitrary June date into an annual rebalance.
        if view.date.month != 6 or not is_last_session_of_month(view.date):
            return None
        equity = view.fundamental("ceq")[-1]
        assets = view.fundamental("at")[-1]
        revenue = view.fundamental("revt")[-1]
        cogs = view.fundamental("cogs")[-1]
        cap = view["mktcap_k"][-1] / 1000  # CRSP thousands -> Compustat millions.
        close = view["close"][-1]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = equity / cap
            profitability = (revenue - cogs) / assets
        valid = (view.members & (view["sp500"][-1] == 0)
                 & np.isfinite(close) & (close >= 5)
                 & np.isfinite(equity) & (equity > 0)
                 & np.isfinite(assets) & (assets > 0)
                 & np.isfinite(cap) & (cap > 0)
                 & np.isfinite(revenue) & np.isfinite(cogs)
                 & np.isfinite(value) & np.isfinite(profitability))
        candidates = np.flatnonzero(valid)
        cheap = sorted(candidates, key=lambda i: (-float(value[i]), str(view.assets[i])))
        cheap = cheap[:max(1, int(len(cheap) * self.value_fraction))]
        profitable = sorted(cheap, key=lambda i: (-float(profitability[i]), str(view.assets[i])))
        count = min(1000, max(1, int(len(profitable) * self.profit_fraction)))
        selected = profitable[:count]
        weights = np.zeros(len(view.assets))
        if selected:
            weights[selected] = self.gross / len(selected)
        return weights


def make_strategy(**params):
    return ValueAnnual(**params)
