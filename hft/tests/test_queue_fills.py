"""Hand-computed queue-position and fill cases for hftbacktest.

Expected outcomes are derived from the documented rules (order_fill.html):
RiskAdverseQueueModel advances the queue only on trades at our price; NoPartialFillExchange
fills fully once the queue ahead is exhausted, on any trade-through, and fills crossing
orders at the best opposite price.
"""

import pytest
from concordance_hft.book import backtest, events
from hftbacktest import FILLED, GTC, GTX, LIMIT

SNAP = [("snapshot", "bid", 0, 0, 100.0, 5.0), ("snapshot", "ask", 0, 0, 100.1, 5.0)]


def _run(trades, price, qty=2.0, tif=GTX):
    hbt = backtest(events(SNAP + trades))
    hbt.elapse(1_000)
    assert hbt.submit_buy_order(0, 1, price, qty, tif, LIMIT, False) == 0
    hbt.elapse(10_000)
    sv = hbt.state_values(0)
    order = hbt.orders(0).get(1)
    out = (order.status if order is not None else None, sv.position, sv.balance, sv.fee)
    hbt.close()
    return out


def test_queue_not_exhausted_no_fill():
    # Queue ahead 5; one 3-lot sell at our price -> 2 still ahead -> resting.
    status, pos, bal, fee = _run([("trade", "sell", 2_000, 2_000, 100.0, 3.0)], 100.0)
    assert status != FILLED and pos == 0.0 and bal == 0.0 and fee == 0.0


def test_queue_exhausted_fills_at_limit_with_maker_rebate():
    # 3 + 3 lots trade at 100.0: queue ahead 5 -> 2 -> exhausted -> full fill of 2 @ 100.0.
    trades = [("trade", "sell", 2_000, 2_000, 100.0, 3.0),
              ("trade", "sell", 3_000, 3_000, 100.0, 3.0)]
    status, pos, bal, fee = _run(trades, 100.0)
    assert status == FILLED and pos == 2.0
    assert bal == pytest.approx(-200.0)
    assert fee == pytest.approx(200.0 * -0.0001)  # maker rebate


def test_trade_through_fills_regardless_of_queue():
    # A sell trade at 99.9 (< our 100.0 bid) fills us even with 5 lots ahead.
    status, pos, bal, fee = _run([("trade", "sell", 2_000, 2_000, 99.9, 1.0)], 100.0)
    assert status == FILLED and pos == 2.0 and bal == pytest.approx(-200.0)


def test_crossing_order_fills_at_best_ask_as_taker():
    status, pos, bal, fee = _run([], 100.3, tif=GTC)
    assert status == FILLED and pos == 2.0
    assert bal == pytest.approx(-200.2)  # 2 @ best ask 100.1, not our 100.3 limit
    assert fee == pytest.approx(200.2 * 0.0005)


def test_order_before_entry_latency_cannot_fill():
    # Trades at t=1005 happen before our order reaches the exchange (1000 + 10ns latency).
    trades = [("trade", "sell", 1_005, 1_005, 99.9, 1.0)]
    status, pos, *_ = _run(trades, 100.0)
    assert pos == 0.0 and status != FILLED


def test_negative_feed_latency_rejected():
    with pytest.raises(ValueError):
        events([("trade", "sell", 2_000, 1_999, 100.0, 1.0)])
