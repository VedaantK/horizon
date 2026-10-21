"""Vendor-neutral first-reported quarterly contract and conservative as-of event builder.

This validates supplied evidence; it does not authenticate a vendor or turn restated
Compustat into vintage data. Importing does not admit data into a sealed generation.
Amounts are USD millions, flows are individual quarters, identifiers are PERMNOs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from concordance.data.crsp import xnys_sessions

VALUES = ("atq", "seqq", "ceqq", "pstkq", "txditcq", "revtq", "cogsq")
KEYS = ("permno", "datadate", "fyearq", "fqtr", "published_at", "vintage_at",
        "source_record_id", "is_original", "currency", "units")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(frame: pd.DataFrame, provenance: dict, source_sha256: str,
              cutoff="2019-12-31") -> pd.DataFrame:
    required = {"schema_version", "basis", "provider", "dataset", "reviewed_by",
                "reviewed_at", "source_sha256", "methodology"}
    require(required <= provenance.keys(), "missing independent source-review evidence")
    require(provenance["schema_version"] == 1
            and provenance["basis"] == "as_first_reported", "first-reported provenance required")
    require(provenance["source_sha256"] == source_sha256, "review does not match source bytes")
    require(all(isinstance(provenance[k], str) and provenance[k].strip()
                for k in required - {"schema_version"}), "empty source-review evidence")
    require(set(KEYS + VALUES) == set(frame.columns), "canonical export columns do not match")
    require(not frame.empty, "empty first-reported export")
    rows = frame.copy()
    rows.attrs = {}  # Vendor dataframe attributes are not approved research metadata.
    require(rows["is_original"].map(lambda v: isinstance(v, (bool, np.bool_)) and v).all(),
            "only explicitly original reports are supported; restated rows are rejected")
    require(rows["currency"].eq("USD").all() and rows["units"].eq("USD_millions").all(),
            "explicit USD million units required; no silent currency conversion")
    for column in ("published_at", "vintage_at"):
        require(rows[column].map(lambda v: pd.Timestamp(v).tzinfo is not None).all(),
                "publication and vintage timestamps must include a timezone")
        rows[column] = pd.to_datetime(rows[column], utc=True, errors="raise")
        require(rows[column].notna().all(), "missing publication/vintage timestamp")
    rows["datadate"] = pd.to_datetime(rows["datadate"], errors="raise")
    require(rows["datadate"].dt.tz is None and rows["datadate"].notna().all(),
            "quarter ends must be timezone-free dates")
    require(rows["datadate"].eq(rows["datadate"].dt.normalize()).all(),
            "quarter end must not include a time")
    for column in ("permno", "fyearq", "fqtr"):
        values = pd.to_numeric(rows[column], errors="raise")
        require(values.notna().all() and np.isfinite(values).all()
                and values.eq(np.floor(values)).all(), "integer identifier/fiscal keys required")
        rows[column] = values.astype("int64")
    require(rows["permno"].gt(0).all() and rows["fqtr"].between(1, 4).all(),
            "invalid PERMNO or fiscal quarter")
    require(rows["source_record_id"].map(lambda v: isinstance(v, str) and bool(v.strip())).all(),
            "source record identity is required")
    known = rows[["published_at", "vintage_at"]].max(axis=1)
    basis = known.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    require((basis >= rows["datadate"]).all(), "a quarter cannot be known before it ends")
    cutoff_date = pd.Timestamp(cutoff)
    require(cutoff_date.tzinfo is None and cutoff_date == cutoff_date.normalize(),
            "cutoff must be a date")
    sessions = xnys_sessions(basis.min(), max(basis.max(), cutoff_date) + pd.Timedelta(days=14))
    indices = np.searchsorted(sessions.values, basis.to_numpy(), side="right")
    rows["available_at"] = sessions[indices].values
    rows = rows[rows["available_at"] <= cutoff_date].copy()
    require(not rows.empty, "no reports available by development cutoff")
    for column in VALUES:
        rows[column] = pd.to_numeric(rows[column], errors="raise")
        require(not np.isinf(rows[column]).any(), "infinite financial value")
    rows = rows.drop_duplicates()
    require(not rows.duplicated(["permno", "datadate"]).any(),
            "conflicting original reports for the same asset and quarter")
    require(not rows.duplicated(["permno", "fyearq", "fqtr"]).any(),
            "ambiguous fiscal calendar; normalize the source explicitly")
    for _, group in rows.groupby("permno"):
        ordered = group.sort_values("datadate")
        ordinal = ordered["fyearq"] * 4 + ordered["fqtr"]
        require((ordinal.diff().dropna() > 0).all(), "fiscal quarter order conflicts with dates")
    return rows.sort_values(["permno", "available_at", "datadate"]).reset_index(drop=True)


def events(rows: pd.DataFrame) -> pd.DataFrame:
    """Derive TTM only from reports visible at that event; late quarters cannot backfill."""
    output = []
    for permno, group in rows.groupby("permno", sort=True):
        known = {}
        for available, arriving in group.groupby("available_at", sort=True):
            for row in arriving.to_dict("records"):
                known[row["datadate"]] = row
            latest = known[max(known)]
            item = {"permno": permno, "datadate": latest["datadate"], "available_at": available,
                    **{name: latest[name] for name in VALUES}}
            ordinal = latest["fyearq"] * 4 + latest["fqtr"]
            quarters = sorted((r for r in known.values()
                               if ordinal - 3 <= r["fyearq"] * 4 + r["fqtr"] <= ordinal),
                              key=lambda r: r["datadate"])
            complete = (len(quarters) == 4
                        and {r["fyearq"] * 4 + r["fqtr"] for r in quarters}
                        == set(range(ordinal - 3, ordinal + 1))
                        and 250 <= (quarters[-1]["datadate"]
                                    - quarters[0]["datadate"]).days <= 300)
            for flow in ("revtq", "cogsq"):
                values = [r[flow] for r in quarters]
                item[flow + "_ttm"] = (sum(values) if complete
                                       and np.isfinite(values).all() else np.nan)
            output.append(item)
    return pd.DataFrame(output)


def import_export(source, review, destination, cutoff="2019-12-31"):
    source, review, destination = map(Path, (source, review, destination))
    require(not source.is_symlink() and not review.is_symlink(), "regular source files required")
    require(not destination.exists(), "destination must be new; no sealed files are amended")
    with source.open("rb") as stream:
        raw_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    evidence = json.loads(review.read_text())
    frame = (pd.read_parquet(source) if source.suffix == ".parquet"
             else pd.read_csv(source, dtype={"source_record_id": "string"}))
    with source.open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == raw_hash,
                "source changed during import")
    rows = normalize(frame, evidence, raw_hash, cutoff)
    destination.mkdir(parents=True, mode=0o700)
    path = destination / "first_reported_quarterly.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(rows, preserve_index=False).replace_schema_metadata(None)
    pq.write_table(table, path, compression="zstd")
    manifest = {"status": "validated_export_not_admitted_to_research", "cutoff": cutoff,
                "basis": "as_first_reported", "retained_rows": len(rows),
                "available_start": str(rows["available_at"].min().date()),
                "available_end": str(rows["available_at"].max().date()),
                "source_review": evidence,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "warning": "Source-review claims require independent verification; "
                           "a schema check alone does not establish historical provenance."}
    (destination / "VALIDATION.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--cutoff", default="2019-12-31")
    args = parser.parse_args()
    result = import_export(args.source, args.review, args.destination, args.cutoff)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
