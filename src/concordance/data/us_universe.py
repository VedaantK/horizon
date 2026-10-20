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
