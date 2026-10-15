"""Continuous futures series by Panama (difference) back-adjustment.

Salvaged from QuantMega `futures/continuous.py` (its offset arithmetic was correct), with:
- roll deadlines from **reference data** (last trade date, and first notice date for
  physically settled contracts), not "the last bar we happen to have";
- the roll may only move **earlier** to find an overlap session, never later (later could
  pass first notice and invite delivery);
- an explicit warning about look-ahead in adjusted *levels*: every past level is shifted
  by roll gaps that occur after it. Only point changes (`point_changes`) are causal; never
  use Panama levels for percentage returns or price thresholds at time t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from concordance.calendar import calendar

OHLC = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class Contract:
    id: str
    last_trade: pd.Timestamp
    first_notice: pd.Timestamp | None = None  # physically settled contracts only
    bars: pd.DataFrame | None = None  # daily OHLC indexed by session date

    @property
    def deadline(self) -> pd.Timestamp:
        """Last day we may hold: the session before first notice, else the last trade day."""
        if self.first_notice is not None:
            return pd.Timestamp(self.first_notice) - pd.Timedelta(days=1)
        return pd.Timestamp(self.last_trade)


def roll_dates(contracts: list[Contract], sessions_before: int, cal_name: str = "XNYS"
               ) -> list[pd.Timestamp]:
    """Roll at the close `sessions_before` exchange sessions before each front's deadline,
    moved earlier to the nearest session on which both contracts have a close."""
    cal = calendar(cal_name)
    out = []
    for front, nxt in zip(contracts[:-1], contracts[1:], strict=True):
        dl = cal.date_to_session(front.deadline, direction="previous")
        cand = cal.session_offset(dl, -sessions_before).tz_localize(None) \
            if sessions_before else pd.Timestamp(dl).tz_localize(None)
        both = front.bars["close"].dropna().index.intersection(nxt.bars["close"].dropna().index)
        ok = both[both <= cand]
        if len(ok) == 0:
            raise ValueError(f"{front.id}->{nxt.id}: no overlapping close on or before "
                             f"{cand.date()}; cannot roll in time")
        out.append(ok[-1])
    return out


def build_continuous(contracts: list[Contract], sessions_before: int = 5,
                     cal_name: str = "XNYS") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (series, roll_table). `series` has Panama-adjusted OHLC, `close_unadj`
    and `contract`; the newest segment is unadjusted."""
    if len(contracts) < 2:
        raise ValueError("need at least two contracts")
    contracts = sorted(contracts, key=lambda c: c.last_trade)
    for c in contracts:
        if c.bars is None or not isinstance(c.bars.index, pd.DatetimeIndex):
            raise ValueError(f"{c.id}: bars must be a DataFrame indexed by date")
    rolls = roll_dates(contracts, sessions_before, cal_name)
    gaps = [float(n.bars.loc[d, "close"] - f.bars.loc[d, "close"])
            for f, n, d in zip(contracts[:-1], contracts[1:], rolls, strict=True)]

    pieces, prev = [], None
    for k, c in enumerate(contracts):
        end = rolls[k] if k < len(rolls) else c.bars.index[-1]
        mask = c.bars.index <= end
        if prev is not None:
            mask &= c.bars.index > prev
        seg = c.bars.loc[mask].copy()
        offset = float(np.sum(gaps[k:]))  # every roll at/after this segment
        seg["close_unadj"] = seg["close"]
        for col in OHLC:
            if col in seg:
                seg[col] = seg[col] + offset
        seg["contract"] = c.id
        pieces.append(seg)
        prev = end
    series = pd.concat(pieces).sort_index()
    if series.index.has_duplicates:
        raise ValueError("overlapping segments")
    table = pd.DataFrame({"roll_date": rolls, "from_contract": [c.id for c in contracts[:-1]],
                          "to_contract": [c.id for c in contracts[1:]], "gap": gaps})
    return series, table


def point_changes(series: pd.DataFrame) -> pd.Series:
    """Close-to-close point change of the held contract — causal (no future gaps)."""
    return series["close"].diff()
