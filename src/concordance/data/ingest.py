"""Ingest user-exported WRDS files into the Parquet lake, validating as we go.

Nothing is silently repaired. Exact duplicate rows are dropped and counted. Conflicting
duplicates, impossible returns and missing columns raise `IngestError`.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from concordance.data.schema import (
    ALL_SPECS,
    CCM_LINK,
    COMP_FUNDQ,
    CRSP_DAILY,
    SP500_MEMBERSHIP,
    ExportSpec,
)

OPEN_END = pd.Timestamp("2262-04-11")  # pandas max; stands for "still active"


class IngestError(ValueError):
    pass


def default_lake() -> Path:
    return Path(os.environ.get("CONCORDANCE_LAKE", Path.home() / "concordance" / "lake"))


def find_exports(raw_dir: Path, spec: ExportSpec) -> list[Path]:
    """`<name>.ext` or chunked `<name>_<anything>.ext` (web-form downloads split by date)."""
    found = []
    for ext in (".parquet", ".csv.gz", ".csv"):
        found += [p for p in raw_dir.glob(f"{spec.name}*{ext}")
                  if p.name == f"{spec.name}{ext}" or p.name.startswith(f"{spec.name}_")]
    return sorted(set(found))


def _parse_dates(col: pd.Series, label: str) -> pd.Series:
    """WRDS web forms emit YYYY-MM-DD, YYYYMMDD or 02JAN2020 depending on the chosen format.
    Unparseable non-empty values raise instead of becoming NaT."""
    s = col.astype("string").str.strip().replace({"": pd.NA, "E": pd.NA, ".": pd.NA})
    out = pd.Series(pd.NaT, index=col.index, dtype="datetime64[ns]")
    ymd = s.str.fullmatch(r"\d{8}(\.0)?").fillna(False).astype(bool)
    out[ymd] = pd.to_datetime(s[ymd].str[:8], format="%Y%m%d")
    rest = s.notna() & ~ymd
    out[rest] = pd.to_datetime(s[rest], format="mixed", errors="coerce")
    bad = rest & out.isna()
    if bad.any():
        raise IngestError(f"{label}: unparseable dates, e.g. {s[bad].iloc[0]!r}")
    return out


def read_export(path: Path, spec: ExportSpec) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    keep = {src: dst for src, dst in spec.columns.items() if src in df.columns}
    df = df[list(keep)].rename(columns=keep)
    df = df.loc[:, ~df.columns.duplicated()]
    missing = [c for c in spec.required if c not in df.columns]
    if missing:
        raise IngestError(f"{path.name}: missing required columns {missing} "
                          f"(expected from {spec.wrds_table}; see docs/WRDS_WEB_EXPORT.md). "
                          f"Columns found: {sorted(df.columns)}")
    for c in spec.date_columns:
        if c in df.columns:
            df[c] = _parse_dates(df[c], f"{path.name}:{c}")
    return df


def read_exports(paths: list[Path], spec: ExportSpec) -> pd.DataFrame:
    return pd.concat([read_export(p, spec) for p in paths], ignore_index=True)


def _dedupe(df: pd.DataFrame, keys: list[str], label: str) -> tuple[pd.DataFrame, int]:
    exact = df.duplicated(keep="first")
    n_exact = int(exact.sum())
    df = df.loc[~exact]
    conflict = df.duplicated(keys, keep=False)
    if conflict.any():
        ex = df.loc[conflict].sort_values(keys).head(6)
        raise IngestError(f"{label}: {int(conflict.sum())} rows share {keys} but differ:\n{ex}")
    return df, n_exact


def clean_crsp_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["permno"] = df["permno"].astype("int64")
    if df["date"].isna().any():
        raise IngestError("crsp_daily: unparseable dates")
    df, n_dup = _dedupe(df, ["permno", "date"], "crsp_daily")
    # CRSP prc < 0 means no trade: the price is the bid/ask midpoint. Close = |prc|.
    df["close"] = df["prc"].abs()
    df["no_trade"] = df["prc"] < 0
    df.loc[df["close"] == 0, "close"] = np.nan
    bad = df["ret"] < -1.0 - 1e-12
    if bad.any():
        raise IngestError(f"crsp_daily: returns below -100%:\n{df.loc[bad].head()}")
    if (df["date"].dt.dayofweek >= 5).any():
        raise IngestError("crsp_daily: weekend dates present")
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    df, rules = _unchained_rows(df)
    stats = {"rows": len(df), "exact_duplicates_dropped": n_dup,
             "permnos": int(df["permno"].nunique()),
             "start": str(df["date"].min().date()), "end": str(df["date"].max().date()),
             "no_trade_rows": int(df["no_trade"].sum()), **rules}
    return df, stats


GAP_DAYS = 7  # calendar days between priced rows that count as a listing gap


def _unchained_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Priced rows with no CRSP return, which the panel cannot chain on its own.

    Two cases are resolved by explicit rules and listed in the manifest:
    - final priced row of a PERMNO (a delisting with a price but no DlyRet, e.g. SVB on
      2023-03-13): ret = close / previous close - 1.
    - first priced row after a listing gap (the security was off-exchange and CRSP has no
      return bridging the gap): flagged `relist`. The panel loader treats each listing as
      its own security, so no return or signal is ever chained across the gap.
    Any other priced row without a return is left missing, and the panel refuses it.
    `df` must be sorted by permno, date.
    """
    pno, px = df["permno"], df["close"].notna()
    n_px = px.groupby(pno).cumsum()
    seen_before = (n_px - px.astype(int)) > 0
    later = (px.groupby(pno).transform("sum") - n_px) > 0
    prev_close = df["close"].groupby(pno).ffill().groupby(pno).shift()
    prev_date = df["date"].where(px).groupby(pno).ffill().groupby(pno).shift()
    prev_row_priced = px.groupby(pno).shift(fill_value=False).astype(bool)
    unchained = px & df["ret"].isna() & seen_before
    gap = ~prev_row_priced | ((df["date"] - prev_date).dt.days > GAP_DAYS)
    relist = unchained & gap
    final = unchained & ~gap & ~later
    df = df.copy()
    df.loc[final, "ret"] = df.loc[final, "close"] / prev_close[final] - 1.0
    df["relist"] = relist
    df["ret_from_price"] = final
    fmt = lambda d: str(pd.Timestamp(d).date())  # noqa: E731
    return df, {
        "final_return_from_price": [
            {"permno": int(r.permno), "date": fmt(r.date), "ret": round(float(r.ret), 6)}
            for r in df.loc[final].itertuples()],
        "listing_gaps": [
            {"permno": int(df.at[i, "permno"]), "last_priced": fmt(prev_date[i]),
             "relisted": fmt(df.at[i, "date"])} for i in np.flatnonzero(relist)],
        "unchained_rows_left": int((unchained & ~relist & ~final).sum()),
    }


def clean_membership(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["permno"] = df["permno"].astype("int64")
    df["end"] = df["end"].fillna(OPEN_END)  # end is inclusive in CRSP
    if (df["end"] < df["start"]).any():
        raise IngestError("sp500_membership: spell ends before it starts")
    df = df.drop_duplicates().sort_values(["permno", "start"]).reset_index(drop=True)
    # Overlapping spells for one permno would double-count membership.
    prev_end = df.groupby("permno")["end"].shift()
    if (df["start"] <= prev_end).any():
        raise IngestError("sp500_membership: overlapping spells for the same permno")
    return df, {"spells": len(df), "permnos": int(df["permno"].nunique())}


def clean_ccm_link(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df = df[df["linktype"].isin(["LU", "LC"]) & df["linkprim"].isin(["P", "C"])]
    df = df.dropna(subset=["permno"])
    df["permno"] = df["permno"].astype("int64")
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    df["linkenddt"] = df["linkenddt"].fillna(OPEN_END)  # null / 'E' = still active
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {"links": len(df)}


def clean_fundq(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    df, n_dup = _dedupe(df, ["gvkey", "datadate"], "comp_fundq")
    return df.reset_index(drop=True), {"rows": len(df), "exact_duplicates_dropped": n_dup,
                                       "missing_rdq": int(df["rdq"].isna().sum())}


CLEANERS = {CRSP_DAILY.name: clean_crsp_daily, SP500_MEMBERSHIP.name: clean_membership,
            CCM_LINK.name: clean_ccm_link, COMP_FUNDQ.name: clean_fundq}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flag_member_gaps(manifest: dict, lake: Path) -> None:
    """Mark listing gaps that began while the stock was an S&P 500 member. A strategy holding
    one exits at the last price before the gap, which is an assumption, not a CRSP return."""
    crsp, mem = manifest["files"].get("crsp_daily", {}), manifest["files"].get(
        SP500_MEMBERSHIP.name, {})
    if crsp.get("status") != "ok" or mem.get("status") != "ok":
        return
    spells = pd.read_parquet(lake / f"{SP500_MEMBERSHIP.name}.parquet")
    for g in crsp["listing_gaps"]:
        s, d = spells[spells["permno"] == g["permno"]], pd.Timestamp(g["last_priced"])
        g["member_at_gap"] = bool(((s["start"] <= d) & (s["end"] >= d)).any())


def ingest(raw_dir: Path, lake: Path | None = None) -> dict:
    """Ingest every export found in `raw_dir`. Returns the manifest (also saved to the lake)."""
    raw_dir, lake = Path(raw_dir), Path(lake or default_lake())
    lake.mkdir(parents=True, exist_ok=True)
    manifest = {"ingested_at": datetime.now(UTC).isoformat(), "files": {}}
    for spec in ALL_SPECS:
        paths = find_exports(raw_dir, spec)
        if not paths:
            manifest["files"][spec.name] = {"status": "missing"}
            continue
        df = read_exports(paths, spec)
        clean = CLEANERS.get(spec.name)
        df, stats = clean(df) if clean else (df, {"rows": len(df)})
        df.to_parquet(lake / f"{spec.name}.parquet", index=False)
        manifest["files"][spec.name] = {
            "status": "ok", "sources": {p.name: _sha256(p) for p in paths}, **stats}
    _flag_member_gaps(manifest, lake)
    (lake / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
