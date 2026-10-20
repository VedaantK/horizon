"""Daily risk-free rate before the WRDS Fama-French table starts (1990). Source: Ken French's
data library, F-F_Research_Data_Factors_daily (zip or csv, values in percent). Only RF is
kept; where it overlaps ff_factors_daily it must match exactly, so both are one series."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

TOLERANCE = 1e-9


def read_french_daily(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
    else:
        text = path.read_text(encoding="latin-1")
    lines = text.splitlines()
    header = next(i for i, ln in enumerate(lines) if ln.replace(" ", "").startswith(",Mkt-RF"))
    rows = []
    for ln in lines[header + 1:]:
        if not ln[:8].strip().isdigit():
            break  # the daily table ends at the first non-date line
        rows.append(ln)
    names = ["date"] + [c.strip() for c in lines[header].split(",")[1:]]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=names)
    if "RF" not in df.columns:
        raise ValueError(f"{path.name}: no RF column")
    out = pd.DataFrame({"date": pd.to_datetime(df["date"].astype(str), format="%Y%m%d"),
                        "rf": df["RF"].astype(float) / 100})
    if out["date"].duplicated().any() or not out["date"].is_monotonic_increasing:
        raise ValueError(f"{path.name}: dates are not unique and increasing")
    return out


def ingest_rf(raw: Path, lake: Path) -> dict:
    raw, lake = Path(raw), Path(lake)
    rf = read_french_daily(raw)
    stats = {"rows": len(rf), "start": str(rf["date"].min().date()),
             "end": str(rf["date"].max().date())}
    ff_path = lake / "ff_factors_daily.parquet"
    if ff_path.exists():
        ff = pd.read_parquet(ff_path, columns=["date", "rf"])
        ff["date"] = pd.to_datetime(ff["date"])
        both = rf.merge(ff, on="date", suffixes=("", "_wrds"))
        diff = (both["rf"] - both["rf_wrds"]).abs()
        if (diff > TOLERANCE).any():
            bad = both.loc[diff > TOLERANCE, "date"]
            raise ValueError(f"RF differs from ff_factors_daily on {len(bad)} days "
                             f"(first {bad.iloc[0].date()})")
        stats["overlap_days_matching_wrds"] = len(both)
    rf.to_parquet(lake / "rf_daily.parquet", index=False)
    path = lake / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"files": {}}
    manifest.setdefault("files", {})["rf_daily"] = {
        "status": "ok", **stats,
        "sources": {raw.name: hashlib.sha256(raw.read_bytes()).hexdigest()},
        "ingested_at": datetime.now(UTC).isoformat()}
    path.write_text(json.dumps(manifest, indent=2))
    return stats


def load_rf(lake: Path) -> pd.Series:
    """Daily risk-free rate: the WRDS table where it has a value, Ken French's before."""
    lake = Path(lake)
    ff = pd.read_parquet(lake / "ff_factors_daily.parquet", columns=["date", "rf"])
    rf = ff.set_index(pd.to_datetime(ff["date"]))["rf"]
    extra = lake / "rf_daily.parquet"
    if extra.exists():
        fr = pd.read_parquet(extra)
        rf = rf.combine_first(fr.set_index(pd.to_datetime(fr["date"]))["rf"])
    return rf.sort_index()
