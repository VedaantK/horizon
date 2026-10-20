"""All US common stocks (the `us_liquid` universe): ingest into `*_us` lake datasets.

The S&P 500 datasets (crsp_daily, crsp_delist, comp_fundq) stay exactly as they were, so
earlier experiments keep their data. The all-stocks exports (docs/WRDS_WEB_EXPORT.md,
"Small caps") are validated and cleaned by the same functions as the S&P files and
written beside them:

    crsp_delist_us.parquet   all delistings (CRSP v2)
    comp_fundq_us.parquet    Compustat quarterly, whole database
    crsp_daily_us.parquet    CRSP daily, US common stocks (ingested by `ingest_daily_us`)
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from concordance.data.ingest import clean_delist, clean_fundq, find_exports, read_exports
from concordance.data.schema import COMP_FUNDQ, CRSP_DELIST

SUFFIX = "_us"
TABLES = ((CRSP_DELIST, clean_delist), (COMP_FUNDQ, clean_fundq))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _record(lake: Path, name: str, entry: dict) -> None:
    """Add one dataset to the lake manifest, keeping every other entry."""
    path = lake / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"files": {}}
    manifest.setdefault("files", {})[name] = entry
    path.write_text(json.dumps(manifest, indent=2))


def ingest_us(raw_dir: Path, lake: Path) -> dict:
    """Delistings and quarterly fundamentals for all US stocks. Returns per-table stats."""
    raw_dir, lake = Path(raw_dir), Path(lake)
    out = {}
    for spec, clean in TABLES:
        paths = find_exports(raw_dir, spec)
        name = spec.name + SUFFIX
        if not paths:
            out[name] = {"status": "missing"}
            continue
        df, stats = clean(read_exports(paths, spec))
        df.to_parquet(lake / f"{name}.parquet", index=False)
        entry = {"status": "ok", "sources": {p.name: _sha256(p) for p in paths}, **stats,
                 "ingested_at": datetime.now(UTC).isoformat()}
        _record(lake, name, entry)
        out[name] = entry
    return out


# ---------- CRSP daily, all US common stocks ----------

COMMON_STOCK = {  # CIZ screens; applied when the export carries the column
    "sharetype": "('NS')", "securitytype": "('EQTY')", "securitysubtype": "('COM')",
    "usincflg": "('Y')", "primaryexch": "('N','A','Q')",
}
NUMERIC = ("dlyopen", "dlyhigh", "dlylow", "dlyclose", "dlyprc", "dlybid", "dlyask", "dlyvol",
           "dlyret", "dlyretx", "dlycap", "shrout")


def _stage(paths, staging: Path) -> dict:
    """Stream the CSV(s) into a filtered, renamed, permno-sorted parquet (DuckDB)."""
    import duckdb

    from concordance.data.schema import CRSP_DAILY

    con = duckdb.connect()
    con.execute("SET memory_limit='12GB'; SET preserve_insertion_order=false")
    files = "[" + ", ".join(f"'{p.as_posix()}'" for p in paths) + "]"
    src = f"read_csv({files}, header=true, union_by_name=true, all_varchar=true)"
    cols = {c.lower() for c in con.execute(f"DESCRIBE SELECT * FROM {src}").df()["column_name"]}
    screen = " AND ".join(f"{c} IN {v}" for c, v in COMMON_STOCK.items() if c in cols) or "TRUE"
    # The screen judges the security, not the row: CRSP writes a stock's delisting-day row
    # (DlyDelFlg = Y, carrying the delisting return) with exchange X and share/security type
    # N/A, so a row-level screen would drop every delisting return.
    where = screen
    if "dlydelflg" in cols:
        where = (f"({screen}) OR (dlydelflg = 'Y' AND permno IN "
                 f"(SELECT DISTINCT permno FROM {src} WHERE {screen}))")
    select = []
    for raw, name in CRSP_DAILY.columns.items():
        if raw not in cols:
            continue
        if raw == "permno":
            select.append("CAST(permno AS BIGINT) AS permno")
        elif raw == "dlycaldt":
            select.append("CAST(dlycaldt AS DATE) AS date")
        elif raw == "siccd":
            select.append("TRY_CAST(NULLIF(siccd, '') AS BIGINT) AS siccd")
        elif raw in NUMERIC:
            select.append(f"TRY_CAST(NULLIF({raw}, '') AS DOUBLE) AS {name}")
        else:
            select.append(f"{raw} AS {name}")
    total = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    con.execute(f"COPY (SELECT {', '.join(select)} FROM {src} WHERE {where} "
                f"ORDER BY permno, date) TO '{staging.as_posix()}' (FORMAT parquet)")
    kept = con.execute(f"SELECT count(*), count(DISTINCT permno) FROM "
                       f"read_parquet('{staging.as_posix()}')").fetchone()
    delisting_rows = (con.execute(f"SELECT count(*) FROM {src} WHERE dlydelflg = 'Y' AND NOT "
                                  f"({screen})").fetchone()[0] if "dlydelflg" in cols else 0)
    con.close()
    return {"rows_in_export": int(total), "rows_common_stock": int(kept[0]),
            "delisting_day_rows_outside_screen": int(delisting_rows),
            "permnos": int(kept[1]), "screens": sorted(c for c in COMMON_STOCK if c in cols)}


def ingest_daily_us(raw_dir: Path, lake: Path, batch: int = 1500) -> dict:
    """CRSP daily for all US common stocks -> crsp_daily_us.parquet, cleaned by the same
    `clean_crsp_daily` as the S&P file, in batches of PERMNOs (every rule in it works
    within one PERMNO) with each batch's own delistings."""
    import duckdb
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from concordance.data.ingest import IngestError, clean_crsp_daily
    from concordance.data.schema import CRSP_DAILY

    raw_dir, lake = Path(raw_dir), Path(lake)
    paths = find_exports(raw_dir, CRSP_DAILY)
    if not paths:
        raise IngestError(f"no crsp_daily export in {raw_dir}")
    delist_path = lake / "crsp_delist_us.parquet"
    if not delist_path.exists():
        raise IngestError("ingest the all-stock delistings first (concordance ingest-us)")
    delist = pd.read_parquet(delist_path)
    staging = lake / "_staging_crsp_daily_us.parquet"
    stats = _stage(paths, staging)
    permnos = duckdb.sql(f"SELECT DISTINCT permno FROM read_parquet('{staging.as_posix()}') "
                         "ORDER BY 1").df()["permno"].tolist()
    out_path = lake / "crsp_daily_us.parquet"
    tmp = lake / "_crsp_daily_us.writing.parquet"
    writer, schema = None, None
    agg = {"rows": 0, "no_trade_rows": 0, "final_return_from_price": 0, "listing_gaps": 0,
           "gap_exits_from_delistings": 0, "gap_exits_unknown": 0, "unchained_rows_left": 0,
           "delisting_checked": 0, "delisting_mismatches": [], "exact_duplicates_dropped": 0,
           "unchained_rows_split_as_new_listing": 0}
    con = duckdb.connect()
    try:
        for start in range(0, len(permnos), batch):
            ids = permnos[start:start + batch]
            con.register("ids", pd.DataFrame({"permno": ids}))
            df = con.execute(f"SELECT s.* FROM read_parquet('{staging.as_posix()}') s "
                             "JOIN ids USING (permno) ORDER BY permno, date").df()
            con.unregister("ids")
            df["date"] = pd.to_datetime(df["date"])
            clean, st = clean_crsp_daily(df, delist[delist["permno"].isin(ids)])
            # A priced row with no return after a few missing sessions (12 in 1990-2025,
            # nearly all sub-$1): no return can be chained across it without knowing about
            # splits or dividends, so it starts a new listing, like a listing gap.
            px = clean["close"].notna()
            seen = px.groupby(clean["permno"]).cumsum() - px.astype(int) > 0
            split = px & clean["ret"].isna() & seen & ~clean["relist"]
            clean.loc[split, "relist"] = True
            agg["unchained_rows_split_as_new_listing"] += int(split.sum())
            agg["rows"] += st["rows"]
            agg["no_trade_rows"] += st["no_trade_rows"]
            agg["exact_duplicates_dropped"] += st["exact_duplicates_dropped"]
            agg["final_return_from_price"] += len(st["final_return_from_price"])
            agg["listing_gaps"] += len(st["listing_gaps"])
            agg["gap_exits_from_delistings"] += len(st["gap_exits_from_delistings"])
            agg["gap_exits_unknown"] += len(st["gap_exits_unknown"])
            agg["unchained_rows_left"] += st["unchained_rows_left"]
            agg["delisting_checked"] += st["delisting_return_check"]["checked"]
            agg["delisting_mismatches"] += st["delisting_return_check"]["mismatches"]
            table = pa.Table.from_pandas(clean, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema)
            writer.write_table(table.cast(schema))
    finally:
        con.close()
        if writer is not None:
            writer.close()
    tmp.replace(out_path)
    staging.unlink()
    dates = duckdb.sql(f"SELECT min(date), max(date), count(DISTINCT permno) FROM "
                       f"read_parquet('{out_path.as_posix()}')").fetchone()
    entry = {"status": "ok", "sources": {p.name: _sha256(p) for p in paths}, **stats, **agg,
             "delisting_mismatches": agg["delisting_mismatches"][:50],
             "delisting_mismatch_count": len(agg["delisting_mismatches"]),
             "start": str(pd.Timestamp(dates[0]).date()), "end": str(pd.Timestamp(dates[1]).date()),
             "permnos_out": int(dates[2]), "ingested_at": datetime.now(UTC).isoformat()}
    _record(lake, "crsp_daily_us", entry)
    return entry


# ---------- the us_liquid panel ----------

MIN_PRICE = 5.0
NYSE_SIZE_PCT = 0.20  # Fama-French's microcap line: the NYSE 20th percentile of market cap


def tradable_rows(lake: Path, lo, hi, min_price=MIN_PRICE, size_pct=NYSE_SIZE_PCT):
    """(permno, date, tradable) for [lo, hi]: closed at or above `min_price` with market cap
    at or above that day's NYSE `size_pct` breakpoint. A stock is a universe member on a
    session if it was tradable at the previous close it has (no same-day information)."""
    import duckdb

    path = (Path(lake) / "crsp_daily_us.parquet").as_posix()
    return duckdb.execute(f"""
        WITH bp AS (
            SELECT date, quantile_cont(mktcap_k, {size_pct}) AS cut
            FROM read_parquet('{path}')
            WHERE primaryexch = 'N' AND mktcap_k > 0 AND date BETWEEN ? AND ?
            GROUP BY date)
        SELECT d.permno, d.date,
               coalesce(d.close >= {min_price} AND d.mktcap_k >= bp.cut, false) AS tradable,
               bp.cut AS nyse_cut_k
        FROM read_parquet('{path}') d JOIN bp USING (date)
        WHERE d.date BETWEEN ? AND ?
        ORDER BY d.permno, d.date""", [lo, hi, lo, hi]).df()


def load_us_panel(start, end, lake: Path | None = None, warmup_sessions: int = 0,
                  min_price: float = MIN_PRICE, size_pct: float = NYSE_SIZE_PCT,
                  min_coverage: float = 0.85, min_year_coverage: float = 0.75,
                  unlock_reason: str | None = None):
    """All US common stocks that were universe members at some point in [start, end], with
    their other days too (so they can be sold, and delisting returns are kept).

    `panel.member`: tradable at the previous close (price and NYSE-size rule above).
    `panel.fields["sp500"]`: 1 on S&P 500 member days, so strategies can exclude them.
    Benchmarks: cap-weighted (`meta["benchmark"]`) and equal-weighted
    (`meta["benchmark_ew"]`) total return of the members.
    """
    import duckdb
    import numpy as np
    import pandas as pd

    from concordance.data import lockbox
    from concordance.data.crsp import (
        PANEL_COLUMNS,
        CoverageError,
        listing_asset_ids,
        load_membership,
        member_mask,
        permno_of,
        xnys_sessions,
    )
    from concordance.data.ingest import default_lake
    from concordance.panel import build_panel

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
    lo, hi = sessions[0], sessions[-1]
    # One extra week before `lo` so the first session has a previous close to judge.
    tr = tradable_rows(lake, lo - pd.Timedelta(days=10), hi, min_price, size_pct)
    tr["member"] = tr.groupby("permno")["tradable"].shift(fill_value=False).astype(bool)
    tr = tr[tr["date"] >= lo]
    ids = sorted(tr.loc[tr["member"] & (tr["date"] >= in_range[0]), "permno"].unique())
    if not ids:
        raise CoverageError("no stock passes the universe rule in this window")

    con = duckdb.connect()
    con.register("ids", pd.DataFrame({"permno": ids}))
    path = (lake / "crsp_daily_us.parquet").as_posix()
    available = set(duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").df()
                    ["column_name"])
    cols = ", ".join(f"d.{c}" for c in ["permno", "date", *PANEL_COLUMNS] if c in available)
    long = con.execute(f"""SELECT {cols} FROM read_parquet('{path}') d JOIN ids USING (permno)
        WHERE d.date BETWEEN ? AND ? ORDER BY d.permno, d.date""", [lo, hi]).df()
    known = "d.gap_exit_known" if "gap_exit_known" in available else "false"
    relist = con.execute(f"""SELECT d.permno, d.date, {known} AS gap_exit_known
        FROM read_parquet('{path}') d JOIN ids USING (permno) WHERE d.relist""").df()
    con.close()
    long["date"] = pd.to_datetime(long["date"])
    tr["date"] = pd.to_datetime(tr["date"])
    off = ~long["date"].isin(sessions)
    if off.any():
        raise CoverageError(f"{int(off.sum())} CRSP rows on non-XNYS sessions, e.g. "
                            f"{long.loc[off, 'date'].iloc[0].date()}")
    long = long.merge(tr[["permno", "date", "member"]], on=["permno", "date"], how="left")
    long["member"] = long["member"].fillna(False).astype(bool)

    # Coverage: member-days must have a close.
    mem = long[long["member"] & (long["date"] >= in_range[0])]
    by_year = mem["close"].notna().groupby(mem["date"].dt.year).mean()
    overall = float(mem["close"].notna().mean())
    if overall < min_coverage or by_year.min() < min_year_coverage:
        raise CoverageError(f"member-day close coverage too low (overall {overall:.1%}, "
                            f"worst year {by_year.min():.1%})")
    # An open with no close is unusable; CRSP has a few (no trade, quote-only days).
    # Dropping the open is the panel's documented rule here, and the count is reported.
    orphan_open = long["open"].notna() & long["close"].isna()
    long.loc[orphan_open, "open"] = np.nan

    sp = load_membership(lake)
    long["sp500"] = member_mask(long, sp)
    long["asset"] = listing_asset_ids(long, relist)
    first = ~long.duplicated("asset")
    long.loc[first, "ret"] = np.nan

    # Benchmarks over member days: previous-close cap weights, and equal weights.
    cap_prev = long.groupby("asset")["mktcap_k"].shift()
    ok = long["member"] & long["ret"].notna()
    cw = long.loc[ok & (cap_prev > 0), ["date", "ret"]].assign(cap=cap_prev[ok & (cap_prev > 0)])
    cw["rc"] = cw["ret"] * cw["cap"]
    g = cw.groupby("date")[["rc", "cap"]].sum()
    bench = (g["rc"] / g["cap"]).reindex(sessions).rename("US liquid stocks (cap-weighted)")
    bench_ew = long.loc[ok].groupby("date")["ret"].mean().reindex(sessions).rename(
        "US liquid stocks (equal-weighted)")

    sp500_flag = long.pivot(index="date", columns="asset", values="sp500")
    panel = build_panel(long.drop(columns=["permno", "sp500"]), sessions=sessions)
    panel.fields["sp500"] = sp500_flag.reindex(index=sessions, columns=panel.assets).fillna(
        False).to_numpy(float)
    gaps = relist[(relist["date"] > sessions[0]) & (relist["date"] <= sessions[-1])]
    panel.meta.update({
        "source": "crsp_us", "universe": "us_liquid",
        "universe_rule": {"min_price": min_price, "nyse_size_pct": size_pct,
                          "judged_at": "previous close"},
        "coverage_overall": overall, "coverage_by_year": by_year.round(4).to_dict(),
        "orphan_opens_dropped": int(orphan_open.sum()),
        "warmup_sessions": warmup_sessions, "permno": permno_of(panel.assets),
        "listing_gaps": gaps.reset_index(drop=True),
        "benchmark": bench, "benchmark_ew": bench_ew,
        "members_per_day": float(panel.member[len(sessions) - len(in_range):].sum(1).mean()),
    })
    return panel
