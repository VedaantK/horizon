"""Transaction cost model, expressed as a per-(session, asset) fee rate on traded notional."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from concordance.panel import MarketPanel


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.0
    # Half the quoted spread at the previous close, known before the open we trade at.
    use_prior_quoted_spread: bool = True
    default_half_spread_bps: float = 5.0
    max_half_spread_bps: float = 500.0

    def fee_rates(self, panel: MarketPanel) -> np.ndarray:
        T, N = panel.shape
        half = np.full((T, N), self.default_half_spread_bps / 1e4)
        if self.use_prior_quoted_spread and "bid" in panel.fields and "ask" in panel.fields:
            bid, ask = panel["bid"], panel["ask"]
            with np.errstate(invalid="ignore", divide="ignore"):
                quoted = (ask - bid) / (ask + bid)
            ok = np.isfinite(quoted) & (bid > 0) & (ask >= bid)
            prior = np.full((T, N), np.nan)
            prior[1:] = np.where(ok[:-1], quoted[:-1], np.nan)
            half = np.where(np.isfinite(prior), prior, half)
        half = np.minimum(half, self.max_half_spread_bps / 1e4)
        rate = self.commission_bps / 1e4 + half
        return np.where(panel.liquidate, 0.0, rate)


ZERO_COST = CostModel(commission_bps=0.0, use_prior_quoted_spread=False,
                      default_half_spread_bps=0.0)
