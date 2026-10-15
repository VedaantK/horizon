"""Random CRSP-shaped markets with every awkward case the engines must agree on."""

import numpy as np
import pandas as pd

from concordance.strategy import Strategy


def random_long_frame(seed: int, n_assets: int | None = None, n_sessions: int | None = None):
    rng = np.random.default_rng(seed)
    N = n_assets or int(rng.integers(2, 9))
    T = n_sessions or int(rng.integers(15, 60))
    dates = pd.bdate_range("2015-01-02", periods=T)
    rows = []
    for j in range(N):
        start = int(rng.integers(0, T // 3))
        end = T - 1 if rng.random() < 0.6 else int(rng.integers(start + 2, T))
        raw = float(rng.uniform(3, 300))
        prev_close = None
        for i in range(start, end + 1):
            if prev_close is not None and rng.random() < 0.06 and i != end:
                continue  # halt: no row today; next return is measured from the last close
            gap = rng.normal(0, 0.01)
            intraday = rng.normal(0, 0.015)
            split = 1.0
            div = 0.0
            if prev_close is not None and rng.random() < 0.03:
                split = float(rng.choice([2.0, 3.0, 0.5]))
            if prev_close is not None and rng.random() < 0.05:
                div = prev_close * float(rng.uniform(0.002, 0.03))
            base = raw if prev_close is None else prev_close
            o = base * np.exp(gap) / split - (div / split if div else 0.0)
            o = max(o, 0.01)
            c = max(o * np.exp(intraday), 0.01)
            if prev_close is None:
                ret = np.nan
            else:
                ret = (c * split + div) / prev_close - 1.0
            if i == end and end < T - 1 and rng.random() < 0.5:
                ret = -1.0 if rng.random() < 0.3 else float(rng.uniform(-0.6, 0.1))
            row = {"date": dates[i], "asset": f"S{j}", "open": round(o, 4), "close": round(c, 4),
                   "ret": ret}
            if rng.random() < 0.8:
                half = c * float(rng.uniform(0.0002, 0.01))
                row["bid"], row["ask"] = c - half, c + half
            if rng.random() < 0.03:
                row["open"] = np.nan  # no open print: untradable today
            rows.append(row)
            prev_close = c
    df = pd.DataFrame(rows)
    # Rounding prices after computing ret mimics CRSP: ret is authoritative, prices rounded.
    for col in ("bid", "ask"):
        if col not in df:
            df[col] = np.nan
    return df


class RandomRebalancer(Strategy):
    def __init__(self, seed: int, p: float = 0.35):
        self.rng = np.random.default_rng(seed + 10_000)
        self.p = p

    def decide(self, view):
        if self.rng.random() > self.p:
            return None
        n = len(view.assets)
        k = int(self.rng.integers(1, n + 1))
        chosen = self.rng.choice(n, size=k, replace=False)
        w = np.zeros(n)
        w[chosen] = self.rng.dirichlet(np.ones(k)) * float(self.rng.uniform(0.3, 1.0))
        return w
