import pandas as pd
import pytest

from concordance.data.riskfree import ingest_rf, load_rf, read_french_daily

FRENCH = """This file was created by using the 202608 CRSP database.
The Tbill return is the simple daily rate.

,Mkt-RF,SMB,HML,RF
19891228,    0.50,   -0.10,    0.20,    0.03
19891229,    0.40,    0.10,   -0.20,    0.03
19900102,    1.44,   -0.68,   -0.06,    0.03
19900103,   -0.20,    0.10,    0.00,    0.02

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def lake_with_ff(tmp_path, rf_1990=(0.0003, 0.0002)):
    pd.DataFrame({"date": pd.to_datetime(["1990-01-02", "1990-01-03"]), "mktrf": 0.0,
                  "rf": list(rf_1990)}).to_parquet(tmp_path / "ff_factors_daily.parquet")
    raw = tmp_path / "F-F_Research_Data_Factors_daily.csv"
    raw.write_text(FRENCH)
    return raw


def test_reads_percent_rows_and_stops_at_footer(tmp_path):
    raw = lake_with_ff(tmp_path)
    df = read_french_daily(raw)
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "1989-12-28", "1989-12-29", "1990-01-02", "1990-01-03"]
    assert df["rf"].tolist() == pytest.approx([0.0003, 0.0003, 0.0003, 0.0002])


def test_ingest_checks_overlap_and_load_prefers_wrds(tmp_path):
    raw = lake_with_ff(tmp_path)
    stats = ingest_rf(raw, tmp_path)
    assert stats["overlap_days_matching_wrds"] == 2
    rf = load_rf(tmp_path)
    assert rf.index.strftime("%Y-%m-%d").tolist() == [
        "1989-12-28", "1989-12-29", "1990-01-02", "1990-01-03"]
    assert rf.tolist() == pytest.approx([0.0003, 0.0003, 0.0003, 0.0002])


def test_ingest_refuses_a_mismatch_with_wrds(tmp_path):
    raw = lake_with_ff(tmp_path, rf_1990=(0.0003, 0.0009))
    with pytest.raises(ValueError, match="differs from ff_factors_daily on 1 days"):
        ingest_rf(raw, tmp_path)
    assert not (tmp_path / "rf_daily.parquet").exists()
