"""Transaction costs and liquidity limits.

`CostModel` gives a per-(session, asset) fee rate on traded notional (commission + half
spread) and the stock-borrow fee on short positions. `LiquidityModel` adds a volume cap
and a square-root market-impact charge, both measured against average daily volume from
sessions *before* the trade. `exec_tables` combines them into the arrays every engine reads.
See docs/SEMANTICS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from concordance.panel import MarketPanel, PanelError

SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.0
    # Half the quoted spread at the previous close, known before the open we trade at.
    use_prior_quoted_spread: bool = True
    default_half_spread_bps: float = 5.0
    max_half_spread_bps: float = 500.0
    # Annual stock-borrow fee on the value of short positions, charged each session at the
    # close as borrow_bps / 1e4 / 252. Flat: there is no per-stock borrow data in the lake,
    # and S&P 500 names are almost always general collateral (roughly 0.25-0.5% a year).
    borrow_bps: float = 50.0
    # Rebalancing trades worth less than this share of portfolio value (bps) are skipped:
    # the position stays as it is. Full exits and forced exits always trade. 0 = off.
    min_trade_bps: float = 0.0
    spread_multiplier: float = 1.0

    def borrow_rate(self) -> float:
        """Borrow fee per session, as a rate on short market value."""
        return self.borrow_bps / 1e4 / SESSIONS_PER_YEAR

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
        half = half * self.spread_multiplier
        rate = self.commission_bps / 1e4 + half
        return np.where(panel.liquidate, 0.0, rate)


ZERO_COST = CostModel(commission_bps=0.0, use_prior_quoted_spread=False,
                      default_half_spread_bps=0.0, borrow_bps=0.0)


@dataclass(frozen=True)
class LiquidityModel:
    """Volume cap and market impact.

    participation   max shares traded per asset per session, as a fraction of ADV
    impact_coef     impact rate = impact_coef * daily_vol * sqrt(shares / ADV)
                    (square-root law; 1.0 is at the conservative end of published estimates)
    max_impact_bps  ceiling on the impact rate; also charged when ADV is unknown
    window/min_obs  trailing sessions for ADV and daily volatility, and the observations
                    required before they count as known
    """

    participation: float = 0.10
    impact_coef: float = 1.0
    max_impact_bps: float = 500.0
    window: int = 20
    min_obs: int = 10


@dataclass(frozen=True)
class ExecTables:
    base_fee: np.ndarray  # (T, N) commission + half spread (0 on forced liquidations)
    cap_buy: np.ndarray  # (T, N) max shares bought per session (inf = no cap, 0 = blocked)
    cap_sell: np.ndarray  # (T, N) max shares sold per session by rebalancing (inf = no cap)
    adv: np.ndarray  # (T, N) ADV in total-return shares, NaN if unknown
    sigma: np.ndarray  # (T, N) daily return volatility, NaN if unknown
    impact_coef: float = 0.0
    max_impact: float = 0.0  # as a rate, not bps
    borrow_rate: float = 0.0  # per session, on short market value at the close
    min_trade: float = 0.0  # fraction of pre-trade value below which a rebalance is skipped

    @property
    def has_liquidity(self) -> bool:
        return self.impact_coef > 0 or self.max_impact > 0 or bool(
            np.isfinite(self.cap_buy).any() or np.isfinite(self.cap_sell).any())


def trailing_liquidity(panel: MarketPanel, window: int, min_obs: int):
    """(adv, sigma) known before the open of each session: rows i use sessions < i.

    Volume is converted to total-return shares (raw shares * raw close / tr_close), so a
    split does not change the measured liquidity.
    """
    if "volume" not in panel.fields:
        raise PanelError("the liquidity model needs a 'volume' field in the panel")
    with np.errstate(invalid="ignore", divide="ignore"):
        tr_shares = panel["volume"] * panel["close"] / panel.tr_close
    roll = dict(window=window, min_periods=min_obs)
    adv = pd.DataFrame(tr_shares).rolling(**roll).mean().shift(1).to_numpy()
    sigma = pd.DataFrame(panel["ret"]).rolling(**roll).std(ddof=1).shift(1).to_numpy()
    return adv, sigma


def exec_tables(panel: MarketPanel, costs: CostModel | None = None,
                liquidity: LiquidityModel | None = None) -> ExecTables:
    costs = costs or CostModel()
    base = costs.fee_rates(panel)
    T, N = panel.shape
    inf = np.full((T, N), np.inf)
    if liquidity is None:
        nan = np.full((T, N), np.nan)
        return ExecTables(base, inf, inf.copy(), nan, nan.copy(),
                          borrow_rate=costs.borrow_rate(), min_trade=costs.min_trade_bps / 1e4)
    adv, sigma = trailing_liquidity(panel, liquidity.window, liquidity.min_obs)
    cap = liquidity.participation * adv
    known = np.isfinite(cap)
    cap_buy = np.where(known, cap, 0.0)  # no volume history: cannot size a buy, so none
    cap_sell = np.where(known, cap, np.inf)  # but an exit is never blocked
    return ExecTables(base, cap_buy, cap_sell, adv, sigma, liquidity.impact_coef,
                      liquidity.max_impact_bps / 1e4, costs.borrow_rate(),
                      costs.min_trade_bps / 1e4)
