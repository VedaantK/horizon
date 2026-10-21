"""An explicit, fail-closed adapter for separately admitted original-vintage data.

No fallback to restated Compustat. A signed vendor feed is not inferred from an
operator JSON file: the source evidence still requires independent review.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from concordance.data.crsp import xnys_sessions
from concordance.data.first_reported import KEYS, VALUES, events, require
from concordance.data.fundamentals import VIEW_MAX_AGE_DAYS, _asset_key

FILE = "first_reported_quarterly.parquet"
ADMISSION = "first_reported_admission.json"


class FirstReportedFundamentals:
    def __init__(self, rows, max_age_days=VIEW_MAX_AGE_DAYS):
        self.events = events(rows)
        self.max_age_days = max_age_days
        self._cache = {}

    def matrix(self, name, panel):
        require(name in (*VALUES, "revtq_ttm", "cogsq_ttm"), "unsupported first-reported field")
        permnos = tuple(map(_asset_key, panel.meta.get("permno", np.asarray(panel.assets))))
        dates = panel.sessions.values.astype("datetime64[ns]")
        key = (name, dates.tobytes(), permnos)
        if key not in self._cache:
            out = np.full(panel.shape, np.nan)
            for permno, group in self.events.groupby("permno"):
                columns = [i for i, value in enumerate(permnos) if value == _asset_key(permno)]
                if not columns:
                    continue
                group = group.sort_values("available_at")
                index = np.searchsorted(group["available_at"].values, dates, side="right") - 1
                known = index >= 0
                values = np.full(len(dates), np.nan)
                values[known] = group[name].to_numpy(float)[index[known]]
                periods = group["datadate"].values.astype("datetime64[ns]")
                age = dates[known] - periods[index[known]]
                values[np.flatnonzero(known)[age > np.timedelta64(self.max_age_days, "D")]] = np.nan
                out[:, columns] = values[:, None]
            self._cache[key] = out
        return self._cache[key]


def load_admitted_rows(lake):
    lake = Path(lake)
    path, admission_path = lake / FILE, lake / ADMISSION
    require(path.is_file() and admission_path.is_file(), "admitted first-reported data required")
    require(not path.is_symlink() and not admission_path.is_symlink(),
            "regular source files required")
    admission = json.loads(admission_path.read_text())
    require(set(admission) == {"schema_version", "basis", "data_sha256", "evidence_sha256",
                               "approved_by", "cutoff"}, "invalid first-reported admission fields")
    require(admission["schema_version"] == 1 and admission["basis"] == "as_first_reported",
            "first-reported admission required")
    require(isinstance(admission["approved_by"], str) and admission["approved_by"].strip(),
            "source review identity required")
    require(isinstance(admission["evidence_sha256"], str)
            and len(admission["evidence_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in admission["evidence_sha256"]),
            "source evidence hash required")
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    require(sha == admission["data_sha256"], "first-reported data differs from admission")
    table = pq.read_table(path)
    require(not table.schema.metadata and all(not field.metadata for field in table.schema),
            "canonical first-reported file must not contain unapproved metadata")
    rows = table.to_pandas()
    require(set(rows.columns) == set(KEYS + VALUES + ("available_at",)),
            "canonical first-reported columns required")
    require(not rows.empty and rows["is_original"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and value).all(),
        "original reports required")
    for name in ("datadate", "available_at"):
        rows[name] = pd.to_datetime(rows[name], errors="raise")
        require(rows[name].dt.tz is None and rows[name].notna().all()
                and rows[name].eq(rows[name].dt.normalize()).all(), "date-only event keys required")
    for name in ("published_at", "vintage_at"):
        require(rows[name].map(lambda value: pd.Timestamp(value).tzinfo is not None).all(),
                "source timestamps require timezones")
        known = pd.to_datetime(rows[name], utc=True, errors="raise")
        require(known.notna().all(), "missing source timestamp")
        # Earliest safe date is strictly after both timestamp calendar dates.
        require((rows["available_at"] > known.dt.tz_localize(None).dt.normalize()).all(),
                "report availability precedes source timestamps")
        require((known.dt.tz_localize(None).dt.normalize() >= rows["datadate"]).all(),
                "a quarter cannot be known before it ends")
    basis = pd.DataFrame({name: pd.to_datetime(rows[name], utc=True)
                          for name in ("published_at", "vintage_at")}).max(axis=1)
    basis = basis.dt.tz_localize(None).dt.normalize()
    sessions = xnys_sessions(basis.min(), basis.max() + pd.Timedelta(days=14))
    expected = sessions[np.searchsorted(sessions.values, basis.to_numpy(), side="right")]
    require(np.array_equal(rows["available_at"].to_numpy(), expected.values),
            "canonical availability is not the next exchange session")
    for name in ("permno", "fyearq", "fqtr"):
        values = pd.to_numeric(rows[name], errors="raise")
        require(values.notna().all() and np.isfinite(values).all()
                and values.eq(np.floor(values)).all(), "integer identifiers/fiscal keys required")
        rows[name] = values.astype("int64")
    require(rows["permno"].gt(0).all() and rows["fqtr"].between(1, 4).all(),
            "invalid identifiers or fiscal quarters")
    require(rows["currency"].eq("USD").all() and rows["units"].eq("USD_millions").all(),
            "first-reported amounts must be USD millions")
    require(rows["available_at"].le(pd.Timestamp(admission["cutoff"])).all(),
            "first-reported records cross admitted cutoff")
    require(not rows.duplicated(["permno", "datadate"]).any(), "ambiguous original quarters")
    require(not rows.duplicated(["permno", "fyearq", "fqtr"]).any(), "ambiguous fiscal quarters")
    for _, group in rows.groupby("permno"):
        ordered = group.sort_values("datadate")
        ordinal = ordered["fyearq"] * 4 + ordered["fqtr"]
        require((ordinal.diff().dropna() > 0).all(), "fiscal quarter order conflicts with dates")
    for name in VALUES:
        rows[name] = pd.to_numeric(rows[name], errors="raise")
        require(not np.isinf(rows[name]).any(), "infinite financial value")
    return rows, admission, sha


def attach(panel, lake):
    rows, admission, sha = load_admitted_rows(lake)
    meta = {**panel.meta, "fundamentals": {
        "basis": "as_first_reported", "source_evidence_sha256": admission["evidence_sha256"],
        "data_sha256": sha, "quarters": len(rows), "max_age_days": VIEW_MAX_AGE_DAYS,
        "provenance": "operator-reviewed source attestation; not vendor-authenticated by code",
    }}
    return replace(panel, fundamentals=FirstReportedFundamentals(rows), meta=meta)
