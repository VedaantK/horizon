"""Exchange-calendar helpers. The holiday schedule is published in advance, so using it in
a decision is not look-ahead (unlike asking whether tomorrow has a price)."""

from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=4)
def calendar(name: str = "XNYS"):
    return xcals.get_calendar(name, start="1990-01-01")


def is_last_session_of_month(date, name: str = "XNYS") -> bool:
    d = pd.Timestamp(date)
    return calendar(name).next_session(d).month != d.month
