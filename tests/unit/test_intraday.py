import pandas as pd

from concordance.intraday import run_intraday


def market(highs, lows, opens=None, asset="A"):
    n = len(highs)
    ts = pd.date_range("2026-10-26 13:30", periods=n, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "asset": asset,
            "open": opens or [10.0] * n,
            "high": highs,
            "low": lows,
            "close": [10.0] * n,
            "volume": [10000] * n,
            "bid": [9.99] * n,
            "ask": [10.01] * n,
        }
    )


def signal(t, direction=1):
    return pd.DataFrame(
        [
            {
                "asset": "A",
                "signal_timestamp": t,
                "direction": direction,
                "stop_pct": 0.01,
                "reward_risk": 1,
                "max_minutes": 5,
                "timeframe": 1,
            }
        ]
    )


def test_next_open_and_ambiguous_bar_uses_stop():
    bars = market([10.1] * 4, [9.8] * 4)
    result = run_intraday(bars, signal(bars.timestamp.iloc[0]))
    assert result["status"] == "ok"
    fills = result["fills"]
    assert len(fills) == 2
    assert fills.timestamp.iloc[0] == bars.timestamp.iloc[1]
    assert fills.reason.iloc[-1] == "stop"
    assert result["metrics"]["net_profit"] < 0
    assert result["metrics"]["max_drawdown"] < 0


def test_short_gap_uses_open_and_covers():
    bars = market([10, 10, 10.5], [10, 10, 9.5], opens=[10, 10, 10.5])
    result = run_intraday(bars, signal(bars.timestamp.iloc[0], -1))
    assert result["status"] == "ok"
    assert result["fills"].qty.iloc[0] < 0
    assert result["fills"].qty.iloc[-1] > 0
    assert result["fills"].reason.iloc[-1] in ("stop", "session_end")


def test_each_asset_closes_on_its_last_bar():
    a = market([10] * 3, [10] * 3)
    b = market([10] * 2, [10] * 2, asset="B")
    bars = pd.concat([a, b], ignore_index=True)
    signals = pd.DataFrame(
        [
            {
                "asset": "B",
                "signal_timestamp": b.timestamp.iloc[0],
                "direction": 1,
                "stop_pct": 0.01,
                "reward_risk": 2,
                "max_minutes": 5,
                "timeframe": 1,
            }
        ]
    )
    result = run_intraday(bars, signals)
    assert result["status"] == "ok"
    assert result["fills"].reason.iloc[-1] == "session_end"
    assert result["fills"].timestamp.iloc[-1] == b.timestamp.iloc[-1]


def test_entry_uses_prior_quote_and_volume():
    bars = market([10] * 3, [10] * 3)
    bars.loc[0, "volume"] = 100
    bars.loc[1, "bid"] = 9.0
    bars.loc[1, "ask"] = 11.0
    result = run_intraday(bars, signal(bars.timestamp.iloc[0]))
    assert result["status"] == "ok"
    entry = result["fills"].iloc[0]
    assert entry.qty == 1  # 1% of prior minute volume
    assert entry.price < 10.02  # prior quote, not the current wide quote


def test_trailing_stop_uses_only_prior_bar_extreme():
    bars = market([10, 10.2, 10.16, 10.1], [10, 10.0, 10.14, 10.05], opens=[10, 10, 10.15, 10.1])
    bars["close"] = [10, 10.15, 10.15, 10.08]
    sig = signal(bars.timestamp.iloc[0])
    sig["reward_risk"] = 10
    sig["trailing_pct"] = 0.005
    result = run_intraday(bars, sig)
    assert result["status"] == "ok"
    assert result["fills"].reason.iloc[-1] == "stop"
    assert result["fills"].timestamp.iloc[-1] == bars.timestamp.iloc[2]


def test_time_exit_counts_entry_minute():
    bars = market([10] * 5, [10] * 5)
    sig = signal(bars.timestamp.iloc[0])
    sig["max_minutes"] = 2
    result = run_intraday(bars, sig)
    assert result["status"] == "ok"
    assert result["fills"].reason.iloc[-1] == "time"
    assert result["fills"].timestamp.iloc[-1] == bars.timestamp.iloc[2]
