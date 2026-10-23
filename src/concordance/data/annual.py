"""Lagged annual Compustat research inputs; these are not original-vintage data.

Prior-calendar-year statements become usable at the last June session close.
Each annual cohort expires before the next June formation; missing years never
inherit older numbers. Annual link assignments are valid only within CCM spells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from concordance.data.crsp import xnys_sessions
from concordance.data.fundamentals import _asset_key
from concordance.memory import allocate

BASIS = "annual_lagged"
FILE = "annual_lagged.parquet"
MANIFEST = "annual_manifest.json"
FIELDS = ("at", "ceq", "revt", "cogs")
LIMITATION = ("Current annual STD values may include revisions; "
              "not authenticated first-reported data.")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def june_close(year):
    return xnys_sessions(f"{year}-06-01", f"{year}-06-30")[-1]


def prepare_rows(frame, links, cutoff="2019-12-31"):
    required = {"gvkey", "datadate", "curcd", "indfmt", "datafmt", "consol", *FIELDS}
    if not required <= set(frame):
        raise ValueError(f"annual export missing fields: {sorted(required - set(frame))}")
    d = frame.copy()
    d["gvkey"] = d.gvkey.astype("string").str.zfill(6)
    d["datadate"] = pd.to_datetime(d.datadate, errors="raise")
    if d[["gvkey", "datadate"]].isna().any().any():
        raise ValueError("missing company or statement date")
    mask = (d.curcd.eq("USD") & d.indfmt.eq("INDL") & d.datafmt.eq("STD")
            & d.consol.eq("C") & d.datadate.le(pd.Timestamp(cutoff)))
    if "popsrc" in d:
        mask &= d.popsrc.eq("D")
    d = d.loc[mask, ["gvkey", "datadate", *FIELDS]].drop_duplicates()
    if d.duplicated(["gvkey", "datadate"]).any():
        raise ValueError("conflicting annual records for company/date")
    for field in FIELDS:
        d[field] = pd.to_numeric(d[field], errors="raise")
        if np.isinf(d[field]).any():
            raise ValueError(f"infinite annual {field}")
    d["formation_year"] = d.datadate.dt.year + 1
    # Fiscal-year transitions can produce several ends in one calendar year.
    # Choose the latest end as a whole row, including its missing values.
    d = d.sort_values("datadate").drop_duplicates(["gvkey", "formation_year"], keep="last")
    years = d.formation_year.unique()
    starts = {int(y): june_close(int(y)) for y in years}
    ends = {int(y): june_close(int(y) + 1) - pd.Timedelta(days=1) for y in years}
    d["available_at"] = d.formation_year.map(starts)
    d["valid_until"] = d.formation_year.map(ends)
    d = d[d.available_at.le(pd.Timestamp(cutoff))]
    link_rows = links.copy()
    link_rows.columns = link_rows.columns.str.lower()
    link_rows = link_rows.rename(columns={"lpermno": "permno"})
    link_rows["gvkey"] = link_rows.gvkey.astype("string").str.zfill(6)
    keep = link_rows.linktype.isin(["LC", "LU"]) & link_rows.linkprim.isin(["P", "C"])
    link_rows = link_rows[keep].copy()
    for col, default in [("linkdt", "1900-01-01"), ("linkenddt", "2099-12-31")]:
        raw = link_rows[col].astype("string")
        open_date = raw.isna() | raw.isin(["", "B", "E", ".B", ".E"])
        link_rows[col] = pd.to_datetime(raw.mask(open_date, default), errors="raise")
    link_rows["permno"] = pd.to_numeric(link_rows.permno, errors="raise").astype("Int64")
    link_rows = link_rows.dropna(subset=["permno"])
    if (link_rows.linkenddt < link_rows.linkdt).any():
        raise ValueError("CCM link ends before it starts")
    # Include links beginning during the holding year, but never expose them early.
    m = d.merge(link_rows[["gvkey", "permno", "linkdt", "linkenddt"]].drop_duplicates(), on="gvkey")
    m["available_at"] = m[["available_at", "linkdt"]].max(axis=1)
    m["valid_until"] = m[["valid_until", "linkenddt"]].min(axis=1)
    m = m[m.available_at.le(m.valid_until) & m.available_at.le(pd.Timestamp(cutoff))]
    m["valid_until"] = m.valid_until.clip(upper=pd.Timestamp(cutoff))
    m = m.drop(columns=["linkdt", "linkenddt"]).drop_duplicates()
    # Multiple active companies per PERMNO must not be chosen by incidental row order.
    for _, g in m.groupby("permno", sort=False):
        g = g.sort_values("available_at")
        if (g.available_at <= g.valid_until.cummax().shift()).any():
            raise ValueError("overlapping annual company/link assignments for PERMNO")
    return m.sort_values(["permno", "available_at"]).reset_index(drop=True)


def prepare(source, links, destination, cutoff="2019-12-31"):
    source, links, destination = Path(source), Path(links), Path(destination)
    if (destination / FILE).exists() or (destination / MANIFEST).exists():
        raise ValueError("annual destination already exists; use a new directory")
    frame = pd.read_csv(source, dtype={"gvkey": "string"}, low_memory=False)
    linkframe = (pd.read_parquet(links) if links.suffix == ".parquet"
                 else pd.read_csv(links, dtype={"gvkey": "string"}, low_memory=False))
    rows = prepare_rows(frame, linkframe, cutoff)
    if rows.empty:
        raise ValueError("no linked annual data before cutoff")
    destination.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(destination / FILE, index=False)
    manifest = {"schema_version": 1, "basis": BASIS, "limitation": LIMITATION,
                "timing": "prior_calendar_year_at_last_june_close", "cutoff": cutoff,
                "source": str(source), "source_sha256": sha256(source),
                "links": str(links), "links_sha256": sha256(links),
                "data_sha256": sha256(destination / FILE), "rows": len(rows),
                "companies": int(rows.gvkey.nunique()), "permnos": int(rows.permno.nunique()),
                "popsrc_verified_from_column": "popsrc" in frame,
                "missing_values": rows[list(FIELDS)].isna().sum().to_dict()}
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


class AnnualFundamentals:
    def __init__(self, rows):
        self.rows = rows
        self._cache = {}

    def matrix(self, name, panel):
        if name not in FIELDS:
            raise KeyError(f"unknown annual field {name!r}; available: {FIELDS}")
        permnos = tuple(map(_asset_key, panel.meta.get("permno", panel.assets)))
        dates = panel.sessions.to_numpy(dtype="datetime64[ns]")
        key = (name, dates.tobytes(), permnos)
        if key not in self._cache:
            out = allocate(panel.shape, fill=np.nan, order="F")
            columns = {}
            for j, permno in enumerate(permnos):
                columns.setdefault(permno, []).append(j)
            for permno, group in self.rows.groupby("permno", sort=False):
                js = columns.get(_asset_key(permno))
                if not js:
                    continue
                group = group.sort_values("available_at")
                k = np.searchsorted(group.available_at.to_numpy(), dates, side="right") - 1
                valid = k >= 0
                idx = np.maximum(k, 0)
                valid &= dates <= group.valid_until.to_numpy()[idx]
                values = np.full(len(dates), np.nan)
                values[valid] = group[name].to_numpy(float)[idx[valid]]
                out[:, js] = values[:, None]
            self._cache[key] = out
        return self._cache[key]


def attach(panel, source):
    source = Path(source)
    manifest = json.loads((source / MANIFEST).read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("basis") != BASIS
            or manifest.get("timing") != "prior_calendar_year_at_last_june_close"
            or manifest.get("data_sha256") != sha256(source / FILE)):
        raise ValueError("annual manifest or data hash mismatch")
    if panel.sessions[-1] > pd.Timestamp(manifest["cutoff"]):
        raise ValueError("annual source cutoff precedes requested panel end")
    rows = pd.read_parquet(source / FILE)
    meta = dict(panel.meta)
    meta["fundamentals"] = manifest
    return replace(panel, fundamentals=AnnualFundamentals(rows), meta=meta)


def random_annual(panel, seed=0):
    """Synthetic annual fixtures for the strategy validator; never vendor evidence."""
    rng = np.random.default_rng(seed)
    records = []
    permnos = panel.meta.get("permno", panel.assets)
    for year in range(panel.sessions[0].year - 1, panel.sessions[-1].year + 1):
        for permno in permnos:
            values = {field: float(rng.lognormal(5, 1)) for field in FIELDS}
            if rng.random() < .1:
                values["ceq"] = np.nan
            records.append({"permno": permno, "datadate": pd.Timestamp(year - 1, 12, 31),
                            "available_at": june_close(year),
                            "valid_until": june_close(year + 1) - pd.Timedelta(days=1),
                            **values})
    return replace(panel, fundamentals=AnnualFundamentals(pd.DataFrame(records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--links", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--cutoff", default="2019-12-31")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.links, args.destination, args.cutoff), indent=2))


if __name__ == "__main__":
    main()
