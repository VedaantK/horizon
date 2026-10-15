import numpy as np
import pandas as pd
import pytest

from concordance.strategy import Strategy

DATES = pd.bdate_range("2020-01-02", periods=12)


def long_frame(rows):
    """rows: list of (session_idx, asset, open, close, ret[, extra dict])."""
    out = []
    for r in rows:
        i, asset, o, c, ret = r[:5]
        d = {"date": DATES[i], "asset": asset, "open": o, "close": c, "ret": ret}
        if len(r) > 5:
            d.update(r[5])
        out.append(d)
    return pd.DataFrame(out)


def path(asset, closes, opens=None, start=0, rets=None):
    """Rows for a simple price path; ret = close/prev_close - 1 unless given."""
    opens = opens or closes
    rows = []
    for k, (o, c) in enumerate(zip(opens, closes, strict=True)):
        if rets is not None:
            r = rets[k]
        else:
            r = np.nan if k == 0 else c / closes[k - 1] - 1
        rows.append((start + k, asset, o, c, r))
    return rows


class Fixed(Strategy):
    """Emits given weights at given decision sessions."""

    def __init__(self, schedule):
        self.schedule = schedule

    def decide(self, view):
        return self.schedule.get(view.i)


@pytest.fixture
def fixed():
    return Fixed
