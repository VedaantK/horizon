"""Engine-neutral backtest result. Every engine adapter must return exactly this."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# phase 0 = at the open (rebalance, gap exits, liquidations); 1 = intraday (resting stop,
# take-profit and limit orders). An asset has at most one fill per (session, phase).
FILL_COLUMNS = ["session", "phase", "asset", "qty", "price", "fee"]


@dataclass
class BacktestResult:
    engine: str
    sessions: pd.DatetimeIndex
    assets: pd.Index
    equity: np.ndarray  # (T,) at each session close
    cash: np.ndarray  # (T,) at each session close
    positions: np.ndarray  # (T, N) quantities at each session close
    fills: pd.DataFrame  # FILL_COLUMNS; session/asset are integer indices; qty signed
    scaled_sessions: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def equity_series(self) -> pd.Series:
        return pd.Series(self.equity, index=self.sessions, name=self.engine)

    @property
    def returns(self) -> pd.Series:
        return self.equity_series.pct_change().fillna(0.0)

    def fills_frame(self) -> pd.DataFrame:
        """Fills with dates and asset ids instead of indices."""
        f = self.fills.copy()
        f.insert(0, "date", self.sessions[f["session"].to_numpy()])
        f["asset"] = self.assets[f["asset"].to_numpy()]
        return f.drop(columns="session")

    def open_positions(self) -> pd.Series:
        last = pd.Series(self.positions[-1], index=self.assets)
        return last[last != 0]


def empty_fills() -> pd.DataFrame:
    return pd.DataFrame({
        "session": pd.Series(dtype=np.int64), "phase": pd.Series(dtype=np.int64),
        "asset": pd.Series(dtype=np.int64),
        "qty": pd.Series(dtype=float), "price": pd.Series(dtype=float),
        "fee": pd.Series(dtype=float)})
