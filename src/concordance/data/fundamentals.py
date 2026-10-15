"""Point-in-time Compustat quarterly fundamentals mapped onto a panel.

A quarterly row becomes visible at the close of the first XNYS session strictly after its
report date `rdq` (an announcement after the close on `rdq` is only usable the next day).
Rows with no `rdq` use `datadate + 90 days` and are flagged. QuantMega pulled these tables
but never built this join; the YTD->quarterly conversion and link-date matching are new.

Limitation: comp.fundq holds restated values, not as-first-reported figures. Values can
therefore contain information that was not public at `rdq` (restatements). For
as-reported data use Compustat Point-in-Time / Snapshot if your WRDS licence has it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from concordance.data.crsp import xnys_sessions
from concordance.data.ingest import default_lake
from concordance.panel import MarketPanel

YTD_FIELDS = {"oancfy": "oancfq", "capxy": "capxq"}
NO_RDQ_LAG = pd.Timedelta(days=90)


def quarterize_ytd(fq: pd.DataFrame) -> pd.DataFrame:
    """Convert year-to-date cash-flow items to single-quarter values."""
    fq = fq.sort_values(["gvkey", "fyearq", "fqtr"]).copy()
    g = fq.groupby(["gvkey", "fyearq"])
    for ytd, q in YTD_FIELDS.items():
        if ytd not in fq:
            continue
        prev_val = g[ytd].shift()
        prev_q = g["fqtr"].shift()
        fq[q] = np.where(fq["fqtr"] == 1, fq[ytd],
                         np.where(prev_q == fq["fqtr"] - 1, fq[ytd] - prev_val, np.nan))
    return fq


def availability(fq: pd.DataFrame) -> pd.DataFrame:
    fq = fq.copy()
    basis = fq["rdq"].where(fq["rdq"].notna(), fq["datadate"] + NO_RDQ_LAG)
    fq["rdq_imputed"] = fq["rdq"].isna()
    lo, hi = basis.min(), basis.max() + pd.Timedelta(days=10)
    sess = xnys_sessions(lo, hi)
    # First session strictly after the basis date.
    idx = np.searchsorted(sess.values, basis.values.astype("datetime64[ns]"), side="right")
    fq["available_at"] = sess[np.minimum(idx, len(sess) - 1)]
    return fq


def link_to_permno(fq: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Attach the PERMNO whose link is valid on the availability date (primary link wins)."""
    m = fq.merge(links[["gvkey", "permno", "linkprim", "linkdt", "linkenddt"]], on="gvkey")
    m = m[(m["available_at"] >= m["linkdt"]) & (m["available_at"] <= m["linkenddt"])]
    m["prim_rank"] = (m["linkprim"] != "P").astype(int)
    m = m.sort_values(["gvkey", "datadate", "prim_rank"]).drop_duplicates(
        ["gvkey", "datadate", "permno"])
    return m.drop(columns=["prim_rank", "linkprim", "linkdt", "linkenddt"])


def asof_matrix(rows: pd.DataFrame, field: str, panel: MarketPanel,
                max_age_days: int = 400) -> np.ndarray:
    """(T, N) latest value of `field` known at each session close, per asset.

    Among rows already available, the one with the latest fiscal `datadate` wins, so a
    late-filed old quarter never overrides a newer one. Values older than `max_age_days`
    (from datadate) are dropped as stale.
    """
    T, N = panel.shape
    out = np.full((T, N), np.nan)
    col = {a: j for j, a in enumerate(panel.assets)}
    sess = panel.sessions.values.astype("datetime64[ns]")
    for permno, g in rows.dropna(subset=[field]).groupby("permno"):
        j = col.get(permno)
        if j is None:
            continue
        g = g.sort_values(["available_at", "datadate"])
        # Drop rows that arrive after a newer fiscal period is already known.
        newer_known = g["datadate"].cummax().shift().fillna(pd.Timestamp.min)
        g = g[g["datadate"] > newer_known]
        avail = g["available_at"].values.astype("datetime64[ns]")
        k = np.searchsorted(avail, sess, side="right") - 1  # last row available by the close
        ok = k >= 0
        vals = np.full(T, np.nan)
        vals[ok] = g[field].to_numpy(float)[k[ok]]
        dd = np.full(T, np.datetime64("NaT"), dtype="datetime64[ns]")
        dd[ok] = g["datadate"].values.astype("datetime64[ns]")[k[ok]]
        stale = ok & ((sess - dd) > np.timedelta64(max_age_days, "D"))
        vals[stale] = np.nan
        out[:, j] = vals
    return out


def attach_quarterly(panel: MarketPanel, fields: list[str], lake: Path | None = None,
                     max_age_days: int = 400) -> MarketPanel:
    """Add `fq_<field>` arrays to the panel (in place on panel.fields) and return it."""
    lake = Path(lake or default_lake())
    fq = pd.read_parquet(lake / "comp_fundq.parquet")
    links = pd.read_parquet(lake / "ccm_link.parquet")
    rows = link_to_permno(availability(quarterize_ytd(fq)), links)
    for f in fields:
        panel.fields[f"fq_{f}"] = asof_matrix(rows, f, panel, max_age_days)
    panel.meta["fundamentals_rdq_imputed_share"] = float(rows["rdq_imputed"].mean()) \
        if len(rows) else 0.0
    return panel
