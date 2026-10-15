"""Strategy interface and the point-in-time view that prevents look-ahead.

A strategy sees only a `PointInTimeView` truncated at the decision session. Its decisions
are turned into a dense weight schedule once, and every engine executes that same schedule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
import pandas as pd

from concordance.panel import MarketPanel


class LookaheadError(RuntimeError):
    pass


class PointInTimeView:
    """Data known at the close of session `i`: every array is sliced to rows 0..i (read-only)."""

    def __init__(self, panel: MarketPanel, i: int):
        self._panel = panel
        self.i = i
        self.date = panel.sessions[i]
        self.sessions = panel.sessions[: i + 1]
        self.assets = panel.assets

    def _cut(self, arr: np.ndarray) -> np.ndarray:
        out = arr[: self.i + 1].view()
        out.flags.writeable = False
        return out

    def __getitem__(self, name: str) -> np.ndarray:
        if name == "tr_close":
            return self._cut(self._panel.tr_close)
        if name == "member":
            return self._cut(self._panel.member)
        return self._cut(self._panel.fields[name])

    def frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self[name], index=self.sessions, columns=self.assets)

    @property
    def members(self) -> np.ndarray:
        """Boolean mask of universe members on the decision session."""
        return self._panel.member[self.i].copy()


class Strategy(ABC):
    """Override `decide`. Return {asset: weight} / Series / array of length N, or None to hold."""

    name: str = "strategy"

    def warmup(self) -> int:
        """Sessions of history required before the first decision."""
        return 0

    @abstractmethod
    def decide(self, view: PointInTimeView):
        ...


def _to_row(decision, panel: MarketPanel) -> np.ndarray:
    n = len(panel.assets)
    if isinstance(decision, pd.Series):
        row = decision.reindex(panel.assets).fillna(0.0).to_numpy(float)
        unknown = decision.index.difference(panel.assets)
        if len(unknown):
            raise ValueError(f"weights for unknown assets: {list(unknown)[:5]}")
    elif isinstance(decision, Mapping):
        row = _to_row(pd.Series(decision, dtype=float), panel)
    else:
        row = np.asarray(decision, dtype=float)
        if row.shape != (n,):
            raise ValueError(f"weight array must have shape ({n},), got {row.shape}")
        row = np.nan_to_num(row)
    if (row < 0).any():
        raise ValueError("negative weights are not supported yet (long-only v1)")
    if row.sum() > 1.0 + 1e-12:
        raise ValueError(f"gross weight {row.sum():.6f} exceeds 1 (no leverage in v1)")
    return row


def weight_schedule(strategy: Strategy, panel: MarketPanel) -> tuple[np.ndarray, np.ndarray]:
    """Run the strategy session by session.

    Returns (weights, has_weights) aligned to the **execution** session: row i+1 holds the
    decision taken at the close of session i.
    """
    T, N = panel.shape
    weights = np.zeros((T, N))
    has = np.zeros(T, dtype=bool)
    for i in range(strategy.warmup(), T - 1):
        decision = strategy.decide(PointInTimeView(panel, i))
        if decision is None:
            continue
        weights[i + 1] = _to_row(decision, panel)
        has[i + 1] = True
    return weights, has
