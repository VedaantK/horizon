"""Build the full historical universe in bounded asset batches.

Batching is only by independent asset columns. Each asset retains its complete
time series, return chain, listing splits and lagged liquidity history.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from concordance.data import lockbox
from concordance.data.crsp import CoverageError, permno_of, xnys_sessions
from concordance.data.us_universe import load_us_panel
from concordance.memory import STORE, allocate
from concordance.panel import MarketPanel


def load_disk_us_panel(start, end, lake, warmup_sessions=0, unlock_reason=None,
                       min_price=5., size_pct=.2, min_coverage=.85, min_year_coverage=.75):
    lockbox.enforce(end, unlock_reason)
    store = STORE.get()
    if store is None:
        raise ValueError("disk panel requires a run-scoped ArrayStore")
    lake = Path(lake)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    sessions = xnys_sessions(start, end)
    if not len(sessions):
        raise CoverageError("no sessions in requested period")
    first = sessions[0]
    if warmup_sessions:
        before = xnys_sessions(start - pd.Timedelta(days=int(warmup_sessions * 1.6) + 10),
                               start - pd.Timedelta(days=1))
        sessions = before[-warmup_sessions:].append(sessions)
    tr_path = Path(store.temp.name) / "tradability.parquet"
    with duckdb.connect() as db:
        db.execute("SET memory_limit='2GB'; SET threads=2; SET preserve_insertion_order=false")
        # Write the complete membership history once, rather than materializing it
        # as one enormous pandas frame or recalculating breakpoints per batch.
        query = """WITH bp AS (
            SELECT date, quantile_cont(mktcap_k, ?) AS cut FROM read_parquet(?)
            WHERE primaryexch='N' AND mktcap_k>0 AND date BETWEEN ? AND ? GROUP BY date),
          t AS (SELECT d.permno,d.date,
            coalesce(d.close>=? AND d.mktcap_k>=bp.cut,false) AS tradable
            FROM read_parquet(?) d JOIN bp USING(date) WHERE d.date BETWEEN ? AND ?)
          SELECT *,lag(tradable,1,false) OVER(PARTITION BY permno ORDER BY date) AS member
          FROM t ORDER BY permno,date"""
        lo = sessions[0] - pd.Timedelta(days=10)
        path = str(lake / "crsp_daily_us.parquet")
        db.execute(f"COPY ({query}) TO '{tr_path}' (FORMAT PARQUET)",
                   [size_pct, path, lo, sessions[-1], min_price, path, lo, sessions[-1]])
        ids = db.execute("SELECT DISTINCT permno FROM read_parquet(?) "
                         "WHERE member AND date>=? ORDER BY permno", [str(tr_path), first]
                         ).fetchnumpy()["permno"].tolist()
    if not ids:
        raise CoverageError("no stock passes the universe rule")
    chunks = []
    cap_sum = pd.Series(0., index=sessions)
    cap_ret = cap_sum.copy()
    count_sum = cap_sum.copy()
    ew_ret = cap_sum.copy()
    coverage = []
    gaps = []
    orphan = 0
    for offset in range(0, len(ids), store.batch_assets):
        batch = ids[offset:offset + store.batch_assets]
        p = load_us_panel(start, end, lake, warmup_sessions, min_price, size_pct,
                          min_coverage=0, min_year_coverage=0, unlock_reason=unlock_reason,
                          _ids=batch, _tradability_path=tr_path)
        den, count = p.meta.pop("_cap_denominator"), p.meta.pop("_ew_count")
        cap_sum += den
        cap_ret += p.meta["benchmark"].fillna(0) * den
        count_sum += count
        ew_ret += p.meta["benchmark_ew"].fillna(0) * count
        coverage.append(p.meta.pop("_coverage"))
        orphan += p.meta.pop("_orphan_opens")
        gaps.append(p.meta["listing_gaps"])
        # All large chunk arrays are already mapped by build_panel.
        chunks.append(p)
        print(f"disk panel: {min(offset + len(batch), len(ids))}/{len(ids)} securities", flush=True)
    counts = pd.concat(coverage).groupby(level=0).sum()
    overall = float(counts["sum"].sum() / counts["count"].sum())
    yearly = counts["sum"] / counts["count"]
    if overall < min_coverage or yearly.min() < min_year_coverage:
        raise CoverageError(f"member-day coverage too low: {overall:.1%}, {yearly.min():.1%}")
    assets = pd.Index(sorted(a for p in chunks for a in p.assets), name="asset")
    if not assets.is_unique:
        raise ValueError("overlapping asset batches")
    shape = len(sessions), len(assets)
    fields = {name: allocate(shape, value.dtype)
              for name, value in chunks[0].fields.items()}
    values = {name: allocate(shape, bool if name in ("member", "liquidate") else float)
              for name in ("member", "tr_close", "exec_price", "liquidate", "val_price")}
    indices = [assets.get_indexer(p.assets) for p in chunks]
    for lo in range(0, len(sessions), 128):
        sl = slice(lo, lo + 128)
        for p, cols in zip(chunks, indices, strict=True):
            for name, out in fields.items():
                out[sl, cols] = p.fields[name][sl]
            for name, out in values.items():
                out[sl, cols] = getattr(p, name)[sl]
    meta = dict(chunks[0].meta)
    meta.update(coverage_overall=overall, coverage_by_year=yearly.round(4).to_dict(),
                orphan_opens_dropped=orphan, permno=permno_of(assets),
                listing_gaps=pd.concat(gaps).drop_duplicates().reset_index(drop=True),
                benchmark=(cap_ret / cap_sum.where(cap_sum > 0)).rename(
                    "US liquid stocks (cap-weighted)"),
                benchmark_ew=(ew_ret / count_sum.where(count_sum > 0)).rename(
                    "US liquid stocks (equal-weighted)"),
                members_per_day=float(values["member"][warmup_sessions:].sum(1).mean()),
                storage="disk", asset_batch_size=store.batch_assets)
    return MarketPanel(sessions=sessions, assets=assets, fields=fields, meta=meta, **values)
