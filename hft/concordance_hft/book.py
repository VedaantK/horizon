"""Small helpers to build hftbacktest event arrays by hand (for tests and converters)."""

from __future__ import annotations

import numpy as np
from hftbacktest import (
    BUY_EVENT,
    DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
    BacktestAsset,
    HashMapMarketDepthBacktest,
)
from hftbacktest.types import event_dtype

BOTH = EXCH_EVENT | LOCAL_EVENT


def events(rows) -> np.ndarray:
    """rows: (kind, side, exch_ts, local_ts, px, qty) with kind in snapshot/depth/trade and
    side 'bid'/'ask' (book) or 'buy'/'sell' (trade aggressor)."""
    kinds = {"snapshot": DEPTH_SNAPSHOT_EVENT, "depth": DEPTH_EVENT, "trade": TRADE_EVENT}
    sides = {"bid": BUY_EVENT, "ask": SELL_EVENT, "buy": BUY_EVENT, "sell": SELL_EVENT}
    out = np.zeros(len(rows), dtype=event_dtype)
    for k, (kind, side, ets, lts, px, qty) in enumerate(rows):
        if lts < ets:
            raise ValueError("local_ts must be >= exch_ts (feed latency is positive)")
        out[k] = (kinds[kind] | sides[side] | BOTH, ets, lts, px, qty, 0, 0, 0.0)
    return out


def backtest(data: np.ndarray, tick: float = 0.1, lot: float = 1.0,
             maker_fee: float = -0.0001, taker_fee: float = 0.0005,
             entry_latency: int = 10, resp_latency: int = 10):
    asset = (BacktestAsset()
             .data([data])
             .linear_asset(1.0)
             .constant_order_latency(entry_latency, resp_latency)
             .risk_adverse_queue_model()
             .no_partial_fill_exchange()
             .trading_value_fee_model(maker_fee, taker_fee)
             .tick_size(tick)
             .lot_size(lot))
    return HashMapMarketDepthBacktest([asset])
