"""Original-vintage adapter semantics, entirely on synthetic fixtures."""

import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.unit.test_first_reported import evidence, export

from concordance.api import load_panel
from concordance.capabilities import capabilities
from concordance.data.first_reported import normalize
from concordance.data.first_reported_panel import (
    ADMISSION,
    FILE,
    FirstReportedFundamentals,
    attach,
)
from concordance.panel import build_panel
from concordance.strategy import PointInTimeView


def panel(dates=None, asset=77):
    dates = dates if dates is not None else pd.bdate_range("2019-01-02", "2019-04-01")
    return build_panel(pd.DataFrame({"date": dates, "asset": asset,
                                     "open": 10.0, "close": 10.0, "ret": 0.0}))


def admitted(directory, frame=None):
    frame = export() if frame is None else frame
    rows = normalize(frame, evidence(), "fixture-hash")
    path = directory / FILE
    table = pa.Table.from_pandas(rows, preserve_index=False).replace_schema_metadata(None)
    pq.write_table(table, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"schema_version": 1, "basis": "as_first_reported", "data_sha256": sha,
              "evidence_sha256": "a" * 64, "approved_by": "SYNTHETIC_TEST_NOT_REAL_EVIDENCE",
              "cutoff": "2019-12-31"}
    (directory / ADMISSION).write_text(json.dumps(record))
    return rows


def test_late_quarter_ttm_never_backfills_and_newest_balance_sheet_wins(tmp_path):
    frame = export()
    frame.loc[1, "published_at"] = pd.Timestamp("2019-03-01T12:00:00Z")
    frame.loc[1, "vintage_at"] = pd.Timestamp("2019-03-01T12:00:00Z")
    admitted(tmp_path, frame)
    loaded = attach(panel(), tmp_path)
    matrix = loaded.fundamentals.matrix("revtq_ttm", loaded)[:, 0]
    assert np.isnan(matrix[loaded.sessions < "2019-03-04"]).all()
    assert (matrix[loaded.sessions >= "2019-03-04"] == 10).all()
    assets = loaded.fundamentals.matrix("atq", loaded)[:, 0]
    assert (assets[loaded.sessions >= "2019-02-11"] == 4).all()


def test_point_in_time_view_and_staleness(tmp_path):
    admitted(tmp_path)
    loaded = attach(panel(pd.bdate_range("2019-01-02", "2019-12-31")), tmp_path)
    early = PointInTimeView(loaded, 3).fundamental("revtq_ttm")
    assert len(early) == 4 and np.isnan(early).all()
    assert early.flags.writeable is False
    late = loaded.fundamentals.matrix("atq", loaded)
    assert np.isnan(late[loaded.sessions > "2019-10-03"]).all()


def test_cache_distinguishes_asset_identity_and_unknown_fields():
    rows = normalize(export(), evidence(), "fixture-hash")
    provider = FirstReportedFundamentals(rows)
    first, other = panel(), panel(asset=88)
    assert np.isfinite(provider.matrix("atq", first)).any()
    assert np.isnan(provider.matrix("atq", other)).all()
    with pytest.raises(ValueError, match="unsupported"):
        provider.matrix("future_field", first)


def test_admission_is_required_and_hash_mismatch_fails(tmp_path):
    with pytest.raises(ValueError, match="admitted"):
        attach(panel(), tmp_path)
    admitted(tmp_path)
    with (tmp_path / FILE).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="differs from admission"):
        attach(panel(), tmp_path)


def test_api_does_not_fallback_to_restated_data(tmp_path, monkeypatch):
    monkeypatch.setattr("concordance.data.crsp.load_sp500_panel", lambda *args, **kwargs: panel())

    def forbidden(*args, **kwargs):
        raise AssertionError("restated data were accessed")

    monkeypatch.setattr("concordance.data.fundamentals.attach_fundamentals", forbidden)
    request = {"data": {"lake": str(tmp_path), "universe": "sp500", "start": "2019-01-02",
                         "end": "2019-04-01", "customer_links": False, "fundamentals": True,
                         "fundamentals_basis": "first_reported"},
               "lockbox": {"unlock_reason": None}}
    strategy = type("Synthetic", (), {"warmup": lambda self: 0})()
    with pytest.raises(ValueError, match="admitted"):
        load_panel(request, strategy)
    admitted(tmp_path)
    _, _, data = load_panel(request, strategy)
    assert data["fundamentals"]["basis"] == "as_first_reported"
    assert data["fundamentals"]["file"] == FILE


def test_import_discards_vendor_dataframe_attributes():
    frame = export()
    frame.attrs["future_diagnostics"] = "SYNTHETIC_WITHHELD_CANARY"
    normalized = normalize(frame, evidence(), "fixture-hash")
    assert normalized.attrs == {}


def test_unapproved_parquet_metadata_is_not_admitted_even_if_hash_matches(tmp_path):
    admitted(tmp_path)
    table = pq.read_table(tmp_path / FILE).replace_schema_metadata({b"future": b"synthetic canary"})
    pq.write_table(table, tmp_path / FILE)
    admission = json.loads((tmp_path / ADMISSION).read_text())
    admission["data_sha256"] = hashlib.sha256((tmp_path / FILE).read_bytes()).hexdigest()
    (tmp_path / ADMISSION).write_text(json.dumps(admission))
    with pytest.raises(ValueError, match="unapproved metadata"):
        attach(panel(), tmp_path)


def test_capabilities_distinguish_first_reported_from_restated(tmp_path):
    files = {key: {"status": "ok"} for key in (
        "crsp_daily_us", "first_reported_quarterly", "first_reported_admission")}
    (tmp_path / "manifest.json").write_text(json.dumps({"files": files}))
    caps = capabilities(tmp_path)
    assert caps["first_reported"]["available"]
    assert "ceqq" in caps["first_reported"]["fields"]
    assert "no restated fallback" in caps["us_liquid"]["fundamentals"].lower()
    assert any("annual fundamentals" in s for s in caps["not_supported"])


def test_short_signal_still_loads_enough_history_for_execution_liquidity(tmp_path, monkeypatch):
    observed = []

    def loader(*args, **kwargs):
        observed.append(kwargs["warmup_sessions"])
        return panel()

    monkeypatch.setattr("concordance.data.crsp.load_sp500_panel", loader)
    request = {"data": {"lake": str(tmp_path), "universe": "sp500", "start": "2019-01-02",
                         "end": "2019-04-01", "customer_links": False, "fundamentals": False},
               "liquidity": {"window": 20}, "lockbox": {"unlock_reason": None}}
    strategy = type("ShortLookback", (), {"warmup": lambda self: 2})()
    load_panel(request, strategy)
    assert observed == [21]


def test_liquidity_history_does_not_invent_rows_before_the_lake_starts(tmp_path, monkeypatch):
    observed = []

    def loader(*args, **kwargs):
        observed.append(kwargs["warmup_sessions"])
        return panel()

    monkeypatch.setattr("concordance.data.crsp.load_sp500_panel", loader)
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {
        "crsp_daily": {"start": "2019-01-02"}}}))
    request = {"data": {"lake": str(tmp_path), "universe": "sp500", "start": "2019-01-02",
                         "end": "2019-04-01", "customer_links": False, "fundamentals": False},
               "liquidity": {"window": 20}, "lockbox": {"unlock_reason": None}}
    strategy = type("NoLookback", (), {"warmup": lambda self: 0})()
    _, _, data = load_panel(request, strategy)
    assert observed == [0]
    assert data["partial_execution_history"]["loaded"] == 0
