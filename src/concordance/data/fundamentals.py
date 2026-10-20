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

# What `view.fundamental(name)` serves. Balance-sheet items are as of the quarter end; flow
# items are the single quarter, and `<flow>_ttm` is the sum of the last four quarters.
STOCK_FIELDS = ("atq", "ltq", "seqq", "ceqq", "cheq", "actq", "lctq", "dlttq", "dlcq",
                "pstkq", "txditcq", "cshoq", "prccq")
FLOW_FIELDS = ("revtq", "saleq", "cogsq", "xsgaq", "ibq", "niq", "oiadpq", "xrdq",
               "oancfq", "capxq")
VIEW_FIELDS = (*STOCK_FIELDS, *FLOW_FIELDS, *(f"{f}_ttm" for f in FLOW_FIELDS))
# A quarter older than this (from its fiscal quarter end) is stale: one missed report is
# tolerated, a second one blanks the stock.
VIEW_MAX_AGE_DAYS = 275
# Four rows form a trailing year when the first and last quarter ends are ~9 months apart.
TTM_SPAN_DAYS = (250, 300)


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


def add_ttm(fq: pd.DataFrame, fields=FLOW_FIELDS) -> pd.DataFrame:
    """`<field>_ttm` = sum of the row's quarter and the three before it, only when all four
    are present and consecutive (quarter ends spanning ~9 months); otherwise NaN."""
    fq = fq.sort_values(["gvkey", "datadate"]).copy()
    g = fq.groupby("gvkey")
    span = (fq["datadate"] - g["datadate"].shift(3)).dt.days
    ok = span.between(*TTM_SPAN_DAYS)
    for f in fields:
        if f not in fq:
            continue
        total = g[f].transform(lambda x: x.rolling(4, min_periods=4).sum())
        fq[f"{f}_ttm"] = total.where(ok)
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


def _asset_key(p):
    """PERMNOs compare as ints (lake parquet vs panel meta dtypes); synthetic ids as-is."""
    try:
        return int(p)
    except (TypeError, ValueError):
        return p


def asof_matrix(rows: pd.DataFrame, field: str, panel: MarketPanel,
                max_age_days: int = 400, fallback: bool = True) -> np.ndarray:
    """(T, N) latest value of `field` known at each session close, per asset.

    Among rows already available, the one with the latest fiscal `datadate` wins, so a
    late-filed old quarter never overrides a newer one. Values older than `max_age_days`
    (from datadate) are dropped as stale. With `fallback=False` a missing value in the
    newest quarter shows as NaN instead of the last quarter that had one, so every field read
    on a date comes from the same quarter.
    """
    T, N = panel.shape
    out = np.full((T, N), np.nan)
    # One PERMNO can span several listing ids (see crsp.RELIST_ID_OFFSET).
    permnos = panel.meta.get("permno", np.asarray(panel.assets))
    col: dict = {}
    for j, p in enumerate(permnos):
        col.setdefault(_asset_key(p), []).append(j)
    sess = panel.sessions.values.astype("datetime64[ns]")
    src = rows.dropna(subset=[field]) if fallback else rows
    for permno, g in src.groupby("permno"):
        js = col.get(_asset_key(permno))
        if js is None:
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
        dd = np.full(T, np.datetime64("NaT", "ns"))
        dd[ok] = g["datadate"].values.astype("datetime64[ns]")[k[ok]]
        stale = ok & ((sess - dd) > np.timedelta64(max_age_days, "D"))
        vals[stale] = np.nan
        out[:, js] = vals[:, None]
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


class Fundamentals:
    """Point-in-time quarterly fundamentals for one run, built into (T, N) arrays on first
    use. `rows` has permno, datadate, available_at and the VIEW_FIELDS columns."""

    def __init__(self, rows: pd.DataFrame, max_age_days: int = VIEW_MAX_AGE_DAYS):
        self.rows = rows
        self.max_age_days = max_age_days
        self._cache: dict = {}

    def matrix(self, name: str, panel: MarketPanel) -> np.ndarray:
        if name not in VIEW_FIELDS:
            raise KeyError(f"unknown fundamental {name!r}; available: {', '.join(VIEW_FIELDS)}")
        key = (name, len(panel.sessions), panel.sessions[0], panel.shape[1])
        if key not in self._cache:
            self._cache[key] = asof_matrix(self.rows, name, panel, self.max_age_days,
                                           fallback=False)
        return self._cache[key]


def load_rows(lake: Path | None = None, fundq: str = "comp_fundq.parquet") -> pd.DataFrame:
    """Lake fundq -> USD rows with quarterized cash flows, TTM sums, availability and PERMNO.
    `fundq`: comp_fundq.parquet (S&P 500 firms) or comp_fundq_us.parquet (all US firms)."""
    lake = Path(lake or default_lake())
    fq = pd.read_parquet(lake / fundq)
    if "curcdq" in fq:
        fq = fq[fq["curcdq"].isna() | (fq["curcdq"] == "USD")]
    links = pd.read_parquet(lake / "ccm_link.parquet")
    return link_to_permno(availability(add_ttm(quarterize_ytd(fq))), links)


def attach_fundamentals(panel: MarketPanel, lake: Path | None = None,
                        fundq: str = "comp_fundq.parquet") -> MarketPanel:
    """Return a copy of `panel` whose view serves `view.fundamental(name)`."""
    from dataclasses import replace

    rows = load_rows(lake, fundq)
    meta = dict(panel.meta)
    meta["fundamentals"] = {"quarters": int(len(rows)),
                            "rdq_imputed_share": round(float(rows["rdq_imputed"].mean()), 4)
                            if len(rows) else 0.0,
                            "max_age_days": VIEW_MAX_AGE_DAYS}
    return replace(panel, fundamentals=Fundamentals(rows), meta=meta)


def random_fundamentals(panel: MarketPanel, seed: int) -> MarketPanel:
    """Synthetic quarters for the strategy validator: one every 63 sessions per asset,
    announced 15-60 sessions after the quarter end, some missing."""
    from dataclasses import replace

    rng = np.random.default_rng(seed)
    T, N = panel.shape
    sess = panel.sessions
    permnos = panel.meta.get("permno", np.asarray(panel.assets))
    out = []
    for j in range(N):
        for end in range(0, T, 63):
            if rng.random() < 0.1:
                continue
            row = {"gvkey": f"S{j}", "permno": permnos[j], "datadate": sess[end],
                   "available_at": sess[min(end + int(rng.integers(15, 61)), T - 1)]}
            for f in (*STOCK_FIELDS, *FLOW_FIELDS):
                row[f] = np.nan if rng.random() < 0.05 else float(rng.lognormal(5, 1))
            out.append(row)
    rows = pd.DataFrame(out)
    if len(rows):
        rows = add_ttm(rows)
    return replace(panel, fundamentals=Fundamentals(rows, max_age_days=36_500))
