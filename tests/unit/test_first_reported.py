"""Synthetic source contracts; these fixtures are not historical provider evidence."""

import numpy as np
import pandas as pd
import pytest

from concordance.data.first_reported import VALUES, events, normalize


def export():
    dates = pd.to_datetime(["2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31"])
    return pd.DataFrame([
        {"permno": 77, "datadate": date, "fyearq": 2018, "fqtr": i + 1,
         "published_at": (date + pd.Timedelta(days=40)).tz_localize("UTC"),
         "vintage_at": (date + pd.Timedelta(days=40)).tz_localize("UTC"),
         "source_record_id": f"synthetic-{i}", "is_original": True,
         "currency": "USD", "units": "USD_millions",
         **{name: float(i + 1) for name in VALUES}}
        for i, date in enumerate(dates)
    ])


def evidence():
    return {"schema_version": 1, "basis": "as_first_reported", "provider": "synthetic",
            "dataset": "unit-test-only", "reviewed_by": "synthetic test fixture",
            "reviewed_at": "2026-01-01T00:00:00Z", "source_sha256": "fixture-hash",
            "methodology": "Synthetic original filings; no real historical evidence"}


def test_source_bytes_restated_basis_and_original_flags_are_enforced():
    with pytest.raises(ValueError, match="source bytes"):
        normalize(export(), evidence(), "different")
    review = {**evidence(), "basis": "restated"}
    with pytest.raises(ValueError, match="first-reported provenance"):
        normalize(export(), review, "fixture-hash")
    frame = export()
    frame.loc[0, "is_original"] = False
    with pytest.raises(ValueError, match="restated rows"):
        normalize(frame, evidence(), "fixture-hash")


def test_no_imputed_dates_or_silent_currency_and_units_conversion():
    for column, value in (("published_at", None), ("published_at", "2018-05-01"),
                          ("currency", "EUR"), ("units", "USD_thousands")):
        frame = export()
        frame[column] = frame[column].astype(object)
        frame.loc[0, column] = value
        with pytest.raises(ValueError):
            normalize(frame, evidence(), "fixture-hash")


def test_later_vintage_delays_availability_and_cutoff_excludes_it():
    frame = export()
    frame.loc[3, "vintage_at"] = pd.Timestamp("2020-01-03T16:00:00Z")
    normalized = normalize(frame, evidence(), "fixture-hash")
    assert len(normalized) == 3
    assert normalized["available_at"].max() <= pd.Timestamp("2019-12-31")


def test_after_close_friday_and_exchange_holiday_are_not_traded_early():
    frame = export().iloc[[3]].copy()
    frame["published_at"] = pd.Timestamp("2019-01-18T23:00:00Z")
    frame["vintage_at"] = pd.Timestamp("2019-01-18T23:00:00Z")
    normalized = normalize(frame, evidence(), "fixture-hash")
    # January 21 is the exchange's MLK holiday; first later session is January 22.
    assert normalized["available_at"].iloc[0] == pd.Timestamp("2019-01-22")


def test_conflicting_originals_and_ambiguous_fiscal_keys_fail():
    frame = export()
    duplicate = frame.iloc[[0]].copy()
    duplicate["atq"] = 999
    with pytest.raises(ValueError, match="conflicting original"):
        normalize(pd.concat([frame, duplicate]), evidence(), "fixture-hash")
    frame.loc[1, "fqtr"] = 1
    with pytest.raises(ValueError, match="ambiguous fiscal"):
        normalize(frame, evidence(), "fixture-hash")


def test_late_quarter_completes_ttm_only_when_it_arrives():
    frame = export()
    frame.loc[1, "published_at"] = pd.Timestamp("2019-03-01T12:00:00Z")
    frame.loc[1, "vintage_at"] = pd.Timestamp("2019-03-01T12:00:00Z")
    normalized = normalize(frame, evidence(), "fixture-hash")
    derived = events(normalized)
    assert np.isnan(derived.iloc[-2]["revtq_ttm"])
    assert derived.iloc[-1]["revtq_ttm"] == 10
    assert derived.iloc[-1]["datadate"] == pd.Timestamp("2018-12-31")
    # Adding the late filing cannot rewrite earlier events.
    before = events(normalized.iloc[:-1])
    pd.testing.assert_frame_equal(before, derived.iloc[:-1].reset_index(drop=True))


def test_missing_quarter_or_missing_flow_does_not_invent_ttm():
    frame = export()
    frame.loc[1, "cogsq"] = np.nan
    derived = events(normalize(frame, evidence(), "fixture-hash"))
    assert derived.iloc[-1]["revtq_ttm"] == 10
    assert np.isnan(derived.iloc[-1]["cogsq_ttm"])
    missing = events(normalize(frame.drop(index=1), evidence(), "fixture-hash"))
    assert np.isnan(missing.iloc[-1]["revtq_ttm"])


def test_source_review_and_record_identity_cannot_be_empty():
    with pytest.raises(ValueError, match="independent source-review"):
        normalize(export(), {}, "fixture-hash")
    frame = export()
    frame.loc[0, "source_record_id"] = ""
    with pytest.raises(ValueError, match="source record identity"):
        normalize(frame, evidence(), "fixture-hash")
