"""Storage and streaming must preserve data, execution, and cross-batch state."""


import numpy as np
import pandas as pd
import pytest

from concordance.api import RequestError, resolve_request, run_backtest
from concordance.costs import ZERO_COST, CostModel, LiquidityModel, exec_tables
from concordance.data.crsp import xnys_sessions
from concordance.data.disk_panel import load_disk_us_panel
from concordance.data.us_universe import load_us_panel
from concordance.memory import STORE, ArrayStore, allocate
from concordance.panel import build_panel
from concordance.report import leg_attribution
from concordance.runner import ALL_ENGINES, run_engine
from concordance.strategy import Orders, Strategy, build_schedule


class Script(Strategy):
    def __init__(self, decisions):
        self.decisions = decisions

    def decide(self, view):
        return self.decisions.get(view.i)


def test_mapped_arrays_clean_up_and_context_does_not_leak(tmp_path):
    with ArrayStore(tmp_path) as store:
        arr = allocate((200, 2000), fill=3)
        assert isinstance(arr, np.memmap)
        assert arr[17, 42] == 3
        directory = store.temp.name
        assert store.files == 1
    assert STORE.get() is None
    from pathlib import Path

    assert not Path(directory).exists()


@pytest.mark.parametrize("width", [1, 3])
def test_streaming_preserves_pending_limits_stops_shorts_and_liquidation(tmp_path, width):
    dates = xnys_sessions("2020-01-02", "2020-01-17")
    rows = []
    for j in range(3):
        for i, date in enumerate(dates):
            if j == 2 and i > 7:
                continue  # forced liquidation must survive a streaming boundary
            close = 98. if j == 0 and i >= 4 else 100.
            rows.append({"date": date, "asset": j, "open": close, "close": close,
                         "high": close + 2, "low": close - 3, "volume": 15.,
                         "ret": -.02 if j == 0 and i == 4 else 0.})
    long = pd.DataFrame(rows)
    decisions = {0: Orders(weights=[.4, -.1, .2], stop_loss={0: 97.}, entry_limit={2: 99.}),
                 5: Orders(weights=[.2, -.1, .3], stop_loss={}),
                 8: [0., 0., 0.]}
    p = build_panel(long, sessions=dates)
    s = build_schedule(Script(decisions), p)
    tables = exec_tables(p, CostModel(commission_bps=1),
                         LiquidityModel(participation=.2, window=3, min_obs=1))
    expected = {e: run_engine(e, p, s, tables, 1000.) for e in ALL_ENGINES}
    with ArrayStore(tmp_path, batch_assets=1, bar_sessions=width):
        mapped = build_panel(long, sessions=dates)
        schedule = build_schedule(Script(decisions), mapped)
        execution = exec_tables(mapped, CostModel(commission_bps=1),
                                LiquidityModel(participation=.2, window=3, min_obs=1))
        for name, arr in s.arrays().items():
            np.testing.assert_equal(arr, schedule.arrays()[name])
        for name in ("base_fee", "adv", "sigma", "cap_buy", "cap_sell"):
            np.testing.assert_equal(getattr(tables, name), getattr(execution, name))
        for engine in ALL_ENGINES:
            actual = run_engine(engine, mapped, schedule, execution, 1000.)
            old = expected[engine]
            np.testing.assert_equal(actual.equity, old.equity)
            np.testing.assert_equal(actual.cash, old.cash)
            np.testing.assert_equal(actual.positions, old.positions)
            np.testing.assert_equal(actual.borrow, old.borrow)
            pd.testing.assert_frame_equal(actual.fills, old.fills)
            assert actual.scaled_sessions == old.scaled_sessions
            token = STORE.set(None)
            try:
                expected_report = leg_attribution(old, p.val_price)
            finally:
                STORE.reset(token)
            assert leg_attribution(actual, mapped.val_price) == expected_report


def test_disk_loader_matches_ram_including_relistings_and_benchmarks(tmp_path):
    lake = tmp_path / "lake"
    lake.mkdir()
    dates = xnys_sessions("2018-05-01", "2018-07-10")
    records = []
    for j in range(1, 8):
        for i, date in enumerate(dates):
            close = 10 + j + i / 100
            records.append({"permno": j, "date": date, "open": close, "close": close,
                            "high": close + 1, "low": close - 1, "ret": 0.001,
                            "volume": 1000., "mktcap_k": j * 1000. + i,
                            "shrout": 100., "primaryexch": "N",
                            "relist": j == 4 and i == 30, "gap_exit_known": True})
    pd.DataFrame(records).to_parquet(lake / "crsp_daily_us.parquet")
    pd.DataFrame({"permno": [3], "start": [dates[10]], "end": [dates[-1]]}).to_parquet(
        lake / "sp500_membership.parquet")
    ram = load_us_panel("2018-06-01", "2018-07-10", lake, warmup_sessions=3)
    with ArrayStore(tmp_path, batch_assets=2):
        disk = load_disk_us_panel("2018-06-01", "2018-07-10", lake, warmup_sessions=3)
        pd.testing.assert_index_equal(ram.assets, disk.assets)
        pd.testing.assert_index_equal(ram.sessions, disk.sessions)
        for field in ram.fields:
            np.testing.assert_equal(ram.fields[field], disk.fields[field])
        for field in ("member", "tr_close", "exec_price", "liquidate", "val_price"):
            np.testing.assert_equal(getattr(ram, field), getattr(disk, field))
        for field in ("benchmark", "benchmark_ew"):
            np.testing.assert_allclose(ram.meta[field], disk.meta[field], atol=1e-15)
        assert ram.meta["coverage_overall"] == disk.meta["coverage_overall"]
        np.testing.assert_equal(ram.meta["permno"], disk.meta["permno"])


def test_panel_derivation_is_identical_across_column_batches(tmp_path):
    from concordance.synthetic import random_long_frame

    long = random_long_frame(23, n_assets=15, n_sessions=80)
    ram = build_panel(long)
    with ArrayStore(tmp_path, batch_assets=3):
        disk = build_panel(long)
        for field in ("tr_close", "exec_price", "val_price", "liquidate"):
            np.testing.assert_equal(getattr(ram, field), getattr(disk, field))
        old = exec_tables(ram, ZERO_COST, LiquidityModel(window=10, min_obs=3))
        new = exec_tables(disk, ZERO_COST, LiquidityModel(window=10, min_obs=3))
        for name in ("base_fee", "adv", "sigma", "cap_buy", "cap_sell"):
            np.testing.assert_equal(getattr(old, name), getattr(new, name))


def test_disk_run_api_records_storage_and_cleans_up(tmp_path):
    source = tmp_path / "strategy.py"
    source.write_text("from concordance.strategy import Strategy\n"
                      "class S(Strategy):\n"
                      "    name='memory_fixture'\n"
                      "    params={}\n"
                      "    def decide(self, view):\n"
                      "        return [0.5] if view.i == 0 else None\n"
                      "STRATEGY=S()\n")
    dates = xnys_sessions("2020-01-02", "2020-01-10")
    p = build_panel(pd.DataFrame({"date": dates, "asset": 1, "open": 10.,
                                 "close": 10., "ret": 0.}))
    request = {"strategy": {"path": str(source)},
               "data": {"universe": "us_liquid", "start": "2020-01-02", "end": "2020-01-10"},
               "engines": ["reference"], "liquidity": None,
               "validation": {"program": "memory_fixture", "lookahead_cuts": 2}}
    ram = run_backtest(request, tmp_path / "ram", panel=p)
    request["memory"] = {"mode": "disk", "asset_batch_size": 1, "bar_batch_sessions": 1}
    disk = run_backtest(request, tmp_path / "disk", panel=p)
    assert ram["status"] == disk["status"] == "ok"
    assert ram["metrics"]["equity_sha256"] == disk["metrics"]["equity_sha256"]
    assert disk["manifest"]["memory"]["mode"] == "disk"
    assert not list((tmp_path / "disk").glob("arrays-*"))
    assert STORE.get() is None
    request["memory"]["asset_batch_size"] = 0
    with pytest.raises(RequestError, match="asset_batch_size"):
        resolve_request(request)


def test_streamed_cash_days_delisting_and_later_asset_activation(tmp_path):
    dates = xnys_sessions("2020-01-02", "2020-01-15")
    records = []
    for i, date in enumerate(dates):
        for asset in range(3):
            if (asset == 0 and i >= 4) or (asset == 1 and i < 4):
                continue
            records.append({"date": date, "asset": asset, "open": 10.+asset,
                            "close": 10.+asset, "ret": 0.})
    p = build_panel(pd.DataFrame(records), sessions=dates)
    s = build_schedule(Script({0: [.6, 0, 0], 6: [0, .5, 0]}), p)
    tables = exec_tables(p, ZERO_COST)
    expected = run_engine("nautilus", p, s, tables, 1000.)
    with ArrayStore(tmp_path, bar_sessions=2):
        actual = run_engine("nautilus", p, s, tables, 1000.)
        np.testing.assert_equal(actual.positions, expected.positions)
        np.testing.assert_equal(actual.equity, expected.equity)
        pd.testing.assert_frame_equal(actual.fills, expected.fills)
