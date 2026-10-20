import gzip
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


def test_daily_us_keeps_common_stocks_and_cleans_them(tmp_path):
    from concordance.data.us_universe import ingest_daily_us

    raw, lake = tmp_path / "raw", tmp_path / "lake"
    raw.mkdir(), lake.mkdir()
    pd.DataFrame({"permno": [20001], "date": pd.to_datetime(["2021-01-07"]), "delret": [-0.5]}
                 ).to_parquet(lake / "crsp_delist_us.parquet")  # as ingest_us writes it
    head = ("PERMNO,DlyCalDt,Ticker,SICCD,PrimaryExch,ShareType,SecurityType,SecuritySubType,"
            "USIncFlg,DlyOpen,DlyPrc,DlyRet,DlyRetx,DlyCap,ShrOut,DlyVol,DlyDelFlg\n")
    rows = [
        # A common stock delisted on 2021-01-07. CRSP writes the delisting-day row with
        # exchange X, types N/A/UNK and price 0; its DlyRet is the delisting return.
        "20001,2021-01-05,AAA,3674,Q,NS,EQTY,COM,Y,10,10,,,1000,100,5,N",
        "20001,2021-01-06,AAA,3674,Q,NS,EQTY,COM,Y,10,11,0.1,0.1,1100,100,5,N",
        "20001,2021-01-07,AAA,3674,X,N/A,N/A,UNK,N,,0,-0.5,-0.5,,,,Y",
        # An ETF (security type FUND) and an ADR (share type AD): both screened out, and so
        # is the ETF's own delisting row.
        "30001,2021-01-05,ETF,0,A,NS,FUND,ETF,Y,50,50,,,5000,100,5,N",
        "30001,2021-01-06,ETF,0,X,N/A,N/A,UNK,N,,0,0.01,0.01,,,,Y",
        "40001,2021-01-05,ADR,2834,N,AD,EQTY,COM,N,20,20,,,2000,100,5,N",
    ]
    with gzip.open(raw / "crsp_daily_all_2021.csv.gz", "wt") as f:
        f.write(head + "\n".join(rows) + "\n")
    out = ingest_daily_us(raw, lake, batch=1)
    assert (out["rows_in_export"], out["rows_common_stock"], out["permnos_out"]) == (6, 3, 1)
    assert out["delisting_day_rows_outside_screen"] == 2  # AAA's kept, the ETF's not
    assert out["delisting_checked"] == 1 and out["delisting_mismatch_count"] == 0
    df = pd.read_parquet(lake / "crsp_daily_us.parquet")
    assert df["ret"].tolist()[-1] == -0.5  # the delisting return survives
    assert df["close"].iloc[:2].tolist() == [10.0, 11.0] and pd.isna(df["close"].iloc[2])
    assert not (lake / "_staging_crsp_daily_us.parquet").exists()


def test_us_panel_membership_by_hand(tmp_path):
    from concordance.data.us_universe import load_us_panel

    lake = tmp_path
    days = pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-05", "2021-01-06",
                           "2021-01-07", "2021-01-08"])
    rows = []
    for permno, exch, cap, prices in (
        (1, "N", 1000.0, [10, 10, 10, 10, 10, 10]),
        (2, "N", 100.0, [10, 10, 10, 10, 10, 10]),
        (3, "Q", 500.0, [4, 4, 6, 6, 6, 6]),
    ):
        prev = None
        for d, p in zip(days, prices, strict=True):
            rows.append({"permno": permno, "date": d, "open": float(p), "close": float(p),
                         "ret": None if prev is None else p / prev - 1, "mktcap_k": cap,
                         "primaryexch": exch, "relist": False, "gap_exit_known": False})
            prev = p
    pd.DataFrame(rows).to_parquet(lake / "crsp_daily_us.parquet")
    pd.DataFrame({"permno": [1], "start": pd.to_datetime(["2021-01-06"]),
                  "end": pd.to_datetime(["2030-01-01"])}).to_parquet(
        lake / "sp500_membership.parquet")
    panel = load_us_panel("2021-01-04", "2021-01-08", lake=lake)
    # NYSE caps 1000 and 100: the 20th percentile is 100 + 0.2 x 900 = 280, so stock 2
    # (cap 100) is never tradable and never enters the panel.
    assert list(panel.meta["permno"]) == [1, 3]
    # Stock 1 was tradable at every previous close; stock 3 first closed >= $5 on 01-05,
    # so it is a member from 01-06.
    assert panel.member[:, 0].tolist() == [True] * 5
    assert panel.member[:, 1].tolist() == [False, False, True, True, True]
    assert panel.fields["sp500"][:, 0].tolist() == [0, 0, 1, 1, 1]
    # 01-05: only stock 1 is a member (flat); 01-06: members 1 and 3, both flat.
    assert panel.meta["benchmark"].loc["2021-01-05"] == 0.0
