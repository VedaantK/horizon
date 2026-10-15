"""Point-in-time CRSP equity panels from the lake.

Conventions (carried over from QuantMega's validated CIZ layer):
- `ret` = DlyRet, a TOTAL return including dividends, and on a delisting day the delisting
  return. Multi-day P&L chains through it; dividends are never added a second time.
- Raw prices (`open`, `close` = |DlyPrc|, `bid`, `ask`) are for same-day ratios and
  price-level screens only.
- Universe membership comes from S&P 500 spells keyed by PERMNO (never ticker), end inclusive.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import exchange_calendars as xcals
import numpy as np
import pandas as pd

from concordance.data import lockbox
from concordance.data.ingest import default_lake
from concordance.panel import MarketPanel, build_panel


class CoverageError(RuntimeError):
    pass


PANEL_COLUMNS = ["open", "high", "low", "close", "ret", "retx", "volume", "bid", "ask",
                 "mktcap_k", "shrout"]


def xnys_sessions(start, end) -> pd.DatetimeIndex:
    cal = xcals.get_calendar("XNYS", start=pd.Timestamp(start) - pd.Timedelta(days=7))
    s = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return pd.DatetimeIndex(s).tz_localize(None)


def load_membership(lake: Path | None = None) -> pd.DataFrame:
    lake = Path(lake or default_lake())
    return pd.read_parquet(lake / "sp500_membership.parquet")


def member_mask(long: pd.DataFrame, spells: pd.DataFrame) -> np.ndarray:
    """True where (permno, date) falls inside a membership spell (both ends inclusive)."""
    con = duckdb.connect()
    con.register("rows", long[["permno", "date"]].reset_index())
    con.register("spells", spells[["permno", "start", "end"]])
    hit = con.execute("""
        SELECT r.index FROM rows r JOIN spells s
          ON r.permno = s.permno AND r.date BETWEEN s.start AND s."end"
    """).fetchnumpy()["index"]
    con.close()
    mask = np.zeros(len(long), dtype=bool)
    mask[hit] = True
    return mask


def coverage_report(long: pd.DataFrame, sessions: pd.DatetimeIndex,
                    spells: pd.DataFrame) -> pd.DataFrame:
    """Per year: member-days implied by the spells vs member-days with a valid close."""
    days = pd.DataFrame({"date": sessions})
    con = duckdb.connect()
    con.register("days", days)
    con.register("spells", spells[["permno", "start", "end"]])
    expected = con.execute("""
        SELECT d.date, s.permno FROM days d JOIN spells s ON d.date BETWEEN s.start AND s."end"
    """).df()
    con.close()
    have = long.loc[long["member"] & long["close"].notna(), ["date", "permno"]]
    merged = expected.merge(have.assign(ok=True), on=["date", "permno"], how="left")
    merged["ok"] = merged["ok"].fillna(False).astype(bool)
    rep = merged.groupby(merged["date"].dt.year)["ok"].agg(["mean", "size"])
    rep.columns = ["coverage", "member_days"]
    return rep


def load_sp500_panel(start, end, lake: Path | None = None, warmup_sessions: int = 0,
                     min_coverage: float = 0.85, min_year_coverage: float = 0.75,
                     unlock_reason: str | None = None) -> MarketPanel:
    """All stocks that were S&P 500 members at any point in [start, end].

    Includes their non-member days (so a stock that leaves the index can still be sold and
    its delisting return is kept). `panel.member` marks member-days for universe selection.
    Raises CoverageError if member-day price coverage is below the thresholds.
    """
    lockbox.enforce(end, unlock_reason)
    lake = Path(lake or default_lake())
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    in_range = xnys_sessions(start, end)
    if len(in_range) == 0:
        raise CoverageError(f"no XNYS sessions between {start.date()} and {end.date()}")
    sessions = in_range
    if warmup_sessions:
        before = xnys_sessions(start - pd.Timedelta(days=int(warmup_sessions * 1.6) + 10),
                               start - pd.Timedelta(days=1))
        sessions = before[-warmup_sessions:].append(in_range)
    spells = load_membership(lake)
    # Universe: anyone who was a member at some point inside [start, end].
    spells_in = spells[(spells["start"] <= in_range[-1]) & (spells["end"] >= in_range[0])]
    permnos = sorted(spells_in["permno"].unique().tolist())
    lo, hi = sessions[0], sessions[-1]

    con = duckdb.connect()
    con.register("ids", pd.DataFrame({"permno": permnos}))
    path = (lake / "crsp_daily.parquet").as_posix()
    available = set(duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").df()
                    ["column_name"])
    cols = ", ".join(f"d.{c}" for c in ["permno", "date", *PANEL_COLUMNS] if c in available)
    long = con.execute(f"""
        SELECT {cols} FROM read_parquet('{path}') d JOIN ids USING (permno)
        WHERE d.date BETWEEN ? AND ? ORDER BY d.permno, d.date
    """, [lo, hi]).df()
    con.close()

    long["date"] = pd.to_datetime(long["date"])
    off = ~long["date"].isin(sessions)
    if off.any():
        raise CoverageError(f"{int(off.sum())} CRSP rows on non-XNYS sessions, e.g. "
                            f"{long.loc[off, 'date'].iloc[0].date()}")
    empty = sessions.difference(pd.DatetimeIndex(long["date"].unique()))
    if len(empty):
        raise CoverageError(f"no CRSP rows at all on XNYS sessions {list(empty[:5].date)}")

    long["member"] = member_mask(long, spells_in)
    rep = coverage_report(long, in_range, spells_in)
    overall = float((rep["coverage"] * rep["member_days"]).sum() / rep["member_days"].sum())
    if overall < min_coverage or rep["coverage"].min() < min_year_coverage:
        raise CoverageError(f"member-day coverage too low (overall {overall:.1%}, "
                            f"worst year {rep['coverage'].min():.1%}):\n{rep}")

    # The first row of a permno inside the window has no prior close in the window:
    # its return is not needed (the total-return price is anchored there).
    first = ~long.duplicated("permno")
    long.loc[first, "ret"] = np.nan
    long = long.rename(columns={"permno": "asset"})
    panel = build_panel(long, sessions=sessions)
    panel.meta.update({"source": "crsp", "coverage": rep, "coverage_overall": overall,
                       "warmup_sessions": warmup_sessions})
    return panel
