import json

import pandas as pd

from concordance.data.us_universe import ingest_us


def test_ingest_us_writes_suffixed_datasets_and_leaves_sp500_files_alone(tmp_path):
    raw, lake = tmp_path / "raw", tmp_path / "lake"
    raw.mkdir(), lake.mkdir()
    pd.DataFrame({"permno": [10001], "x": [1]}).to_parquet(lake / "crsp_delist.parquet")
    (lake / "manifest.json").write_text(json.dumps({"files": {"crsp_delist": {"status": "ok"}}}))
    pd.DataFrame({"permno": [10001, 10002], "delistingdt": ["2017-08-03", "2001-05-01"],
                  "delret": [0.0, -0.3], "delactiontype": ["MER", "GDR"]}).to_csv(
        raw / "crsp_delist_all.csv", index=False)
    pd.DataFrame({"gvkey": [1417, 1417], "datadate": ["2020-03-31", "2020-06-30"],
                  "fyearq": [2020, 2020], "fqtr": [1, 2],
                  "rdq": ["2020-05-01", None], "atq": [10.0, 11.0], "curcdq": ["USD", "USD"]}
                 ).to_csv(raw / "comp_fundq_all.csv", index=False)
    out = ingest_us(raw, lake)
    assert out["crsp_delist_us"]["rows"] == 2 and out["comp_fundq_us"]["rows"] == 2
    assert pd.read_parquet(lake / "crsp_delist.parquet")["x"].tolist() == [1]  # untouched
    fq = pd.read_parquet(lake / "comp_fundq_us.parquet")
    assert fq["gvkey"].tolist() == ["001417", "001417"]
    manifest = json.loads((lake / "manifest.json").read_text())["files"]
    assert manifest["crsp_delist"] == {"status": "ok"}
    assert {"crsp_delist_us", "comp_fundq_us"} <= set(manifest)
