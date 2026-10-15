"""Random CRSP-shaped markets with every awkward case the engines must agree on."""

import numpy as np

from concordance.strategy import Strategy
from concordance.synthetic import random_long_frame  # noqa: F401  (re-exported for tests)


class RandomOrders(Strategy):
    """Random targets plus random stop-loss, take-profit (absolute and %) and limit entries."""

    def __init__(self, seed: int, p: float = 0.35):
        self.rng = np.random.default_rng(seed + 20_000)
        self.p = p

    def decide(self, view):
        from concordance.strategy import Orders

        rng = self.rng
        n = len(view.assets)
        close = view["close"][-1]
        st = view.position
        o = Orders()
        if rng.random() < self.p:
            k = int(rng.integers(1, n + 1))
            chosen = rng.choice(n, size=k, replace=False)
            w = np.zeros(n)
            w[chosen] = rng.dirichlet(np.ones(k)) * float(rng.uniform(0.3, 1.0))
            if rng.random() < 0.3 and st.held.any():
                keep = st.held & (rng.random(n) < 0.5)
                w[keep] = 0.0
                o.hold = keep
            o.weights = w
            flat = (w > 0) & ~st.held & np.isfinite(close)
            if rng.random() < 0.5 and flat.any():
                o.entry_limit = np.where(flat & (rng.random(n) < 0.5),
                                         close * rng.uniform(0.95, 1.01, n), np.nan)
        if rng.random() < 0.5:
            ok = np.isfinite(close)
            if rng.random() < 0.5:
                o.stop_loss = np.where(ok & (rng.random(n) < 0.7),
                                       close * rng.uniform(0.95, 1.0, n), np.nan)
            else:
                o.stop_loss_pct = np.where(rng.random(n) < 0.7, rng.uniform(0.005, 0.05, n),
                                           np.nan)
        if rng.random() < 0.4:
            o.take_profit_pct = np.where(rng.random(n) < 0.6, rng.uniform(0.005, 0.05, n),
                                         np.nan)
        if o.weights is None and o.stop_loss is None and o.stop_loss_pct is None \
                and o.take_profit_pct is None:
            return None
        return o


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
