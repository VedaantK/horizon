"""Synthetic futures markets for tests and the strategy validator.

Each market is a sequence of quarterly contracts that overlap for a month, priced as a
random walk around a shared spot with a random roll basis, so Panama rolls, gaps (and,
for drifting markets, negative adjusted levels) all occur. Pure function of the seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from concordance.calendar import calendar
from concordance.futures.continuous import Contract, build_continuous
from concordance.futures.panel import Spec, build_futures_panel


def random_futures(n_markets=4, start="2015-01-02", end="2017-12-29", seed=0,
                   sessions_before=5):
    rng = np.random.default_rng(seed)
    cal = calendar("XNYS")
    days = pd.DatetimeIndex(cal.sessions_in_range(start, end)).tz_localize(None)
    series, specs = {}, {}
    for m in range(n_markets):
        root = f"M{m}"
        mult = float(rng.choice([10, 50, 100, 1000]))
        tick = float(rng.choice([0.01, 0.25, 0.5]))
        spot = 50 + 100 * rng.random()
        path = spot * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(days))))
        expiries = pd.date_range(days[0] + pd.Timedelta(days=80), days[-1] + pd.Timedelta(
            days=200), freq="QE-MAR") + pd.Timedelta(days=15)
        contracts, begin = [], days[0]
        for k, exp in enumerate(expiries):
            live = days[(days >= begin) & (days <= exp)]
            if len(live) < 25:
                continue
            basis = 1 + rng.normal(0, 0.01)
            idx = np.searchsorted(days, live)
            close = np.round(path[idx] * basis / tick) * tick
            gap = rng.normal(0, 0.004, len(live))
            open_ = np.round(close * (1 - gap) / tick) * tick
            contracts.append(Contract(f"{root}{k}", last_trade=exp,
                                      bars=pd.DataFrame({"open": open_, "close": close},
                                                        index=live)))
            begin = exp - pd.Timedelta(days=35)  # next contract starts a month before
        s, _ = build_continuous(contracts, sessions_before=sessions_before)
        series[root] = s
        specs[root] = Spec(root, multiplier=mult, tick=tick,
                           margin_rate=float(rng.choice([0.03, 0.06, 0.1])),
                           commission=float(rng.choice([1.0, 2.5])), slippage_ticks=1.0)
    rf = pd.Series(rng.uniform(0, 0.0002), index=days)
    return build_futures_panel(series, specs, rf, sessions=days)
