"""Nautilus Trader engine: Nautilus's matching engine, cash account and portfolio do the work.

Each session becomes two bars per asset on the same bar type:
- an **open bar** at 14:30Z with O=H=L=C=execution price (the total-return open, or the
  final value for a forced liquidation), and
- a **close bar** at 21:00Z with O=H=L=C=total-return close.

Market orders submitted once every open bar of a session has arrived fill at that open
price (bar execution, L1). Equity at each close is read from Nautilus: account balance plus
`portfolio.net_exposures` marked at the last bar price. Money uses a 9-decimal currency and
quantities a 6-decimal grid, matching docs/SEMANTICS.md.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    AssetClass,
    BarAggregation,
    CurrencyType,
    InstrumentClass,
    OmsType,
    OrderSide,
    PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy as NautilusStrategy

from concordance.panel import PRICE_DECIMALS, QTY_DECIMALS, MarketPanel
from concordance.planner import plan_orders
from concordance.result import FILL_COLUMNS, BacktestResult

VENUE = Venue("SIM")
OPEN_OFFSET = pd.Timedelta(hours=14, minutes=30)
CLOSE_OFFSET = pd.Timedelta(hours=21)
BAR_VOLUME = Quantity(1_000_000_000, QTY_DECIMALS)


def _currency() -> Currency:
    # Not Currency.from_str: for an unknown code it silently invents an 8-decimal currency.
    ccy = Currency("CUSD", PRICE_DECIMALS, 0, "Concordance USD", CurrencyType.FIAT)
    Currency.register(ccy, overwrite=True)
    assert Currency.from_str("CUSD", strict=True).precision == PRICE_DECIMALS
    return ccy


class _TableFee(FeeModel):
    """Fee = |qty| * price * fee_rate[session, asset], looked up from the order's timestamp."""

    def __init__(self, fee_rate, ts_to_session, inst_to_asset, ccy):
        super().__init__()
        self.fee_rate = fee_rate
        self.ts_to_session = ts_to_session
        self.inst_to_asset = inst_to_asset
        self.ccy = ccy

    def get_commission(self, order, fill_qty, fill_px, instrument):
        i = self.ts_to_session[order.ts_init]
        j = self.inst_to_asset[instrument.id]
        notional = float(fill_qty) * float(fill_px)
        return Money(notional * float(self.fee_rate[i, j]), self.ccy)


class _Executor(NautilusStrategy):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.seen = defaultdict(int)
        self.open_px = {}
        self.last_close = np.full(ctx["N"], np.nan)
        self.fills = []
        self.problems = []

    def on_start(self):
        for bt in self.ctx["bar_types"]:
            self.subscribe_bars(bt)

    def on_bar(self, bar: Bar):
        ts = bar.ts_event
        j = self.ctx["inst_to_asset"][bar.bar_type.instrument_id]
        kind, i = self.ctx["ts_kind"][ts]
        if kind == "open":
            self.open_px[j] = bar.close.as_double()
        else:
            self.last_close[j] = bar.close.as_double()
        self.seen[ts] += 1
        if self.seen[ts] == self.ctx["expected"][ts]:
            if kind == "open":
                self._rebalance(i)
            else:
                self._record(i)

    def _positions(self) -> np.ndarray:
        pos = np.zeros(self.ctx["N"])
        for j, inst in enumerate(self.ctx["instruments"]):
            pos[j] = float(self.portfolio.net_position(inst.id))
        return pos

    def _cash(self) -> float:
        return self.portfolio.account(VENUE).balance_total(self.ctx["ccy"]).as_double()

    def _rebalance(self, i: int):
        c = self.ctx
        exec_price = np.full(c["N"], np.nan)
        for j, px in self.open_px.items():
            exec_price[j] = px
        self.open_px = {}
        delta, order, m, scaled = plan_orders(
            self._cash(), self._positions(), exec_price, self.last_close.copy(),
            c["fee_rate"][i], c["weights"][i], bool(c["has"][i]), c["liquidate"][i])
        if scaled:
            c["scaled"].append(i)
        for k in range(m):
            j = int(order[k])
            qty = Quantity(abs(float(delta[j])), QTY_DECIMALS)
            side = OrderSide.BUY if delta[j] > 0 else OrderSide.SELL
            self.submit_order(self.order_factory.market(c["instruments"][j].id, side, qty))

    def _record(self, i: int):
        c = self.ctx
        cash = self._cash()
        exposures = self.portfolio.net_exposures(VENUE) or {}
        exposure = exposures.get(c["ccy"])
        c["cash"][i] = cash
        c["equity"][i] = cash + (exposure.as_double() if exposure is not None else 0.0)
        c["positions"][i] = self._positions()
        c["recorded"][i] = True

    def on_order_filled(self, event):
        c = self.ctx
        _, i = c["ts_kind"][event.ts_event]
        j = c["inst_to_asset"][event.instrument_id]
        sign = 1.0 if event.order_side == OrderSide.BUY else -1.0
        self.fills.append((i, j, sign * float(event.last_qty), event.last_px.as_double(),
                           event.commission.as_double()))

    def on_order_denied(self, event):
        self.problems.append(f"order denied: {event.reason}")

    def on_order_rejected(self, event):
        self.problems.append(f"order rejected: {event.reason}")


def _instrument(j: int, ccy: Currency) -> Instrument:
    sym = Symbol(f"A{j}")
    return Instrument(
        InstrumentId(sym, VENUE), sym, AssetClass.EQUITY, InstrumentClass.SPOT, ccy, False,
        PRICE_DECIMALS, QTY_DECIMALS, Quantity(10.0 ** -QTY_DECIMALS, QTY_DECIMALS),
        Quantity.from_int(1), Decimal(0), Decimal(0), Decimal(0), Decimal(0), 0, 0,
        price_increment=Price(10.0 ** -PRICE_DECIMALS, PRICE_DECIMALS))


def run(panel: MarketPanel, weights: np.ndarray, has_weights: np.ndarray,
        fee_rate: np.ndarray, init_cash: float) -> BacktestResult:
    T, N = panel.shape
    ccy = _currency()
    instruments = [_instrument(j, ccy) for j in range(N)]
    bar_types = [BarType(inst.id, BarSpecification(1, BarAggregation.DAY, PriceType.LAST),
                         AggregationSource.EXTERNAL) for inst in instruments]

    ts_kind, expected, bars = {}, {}, []
    for i, day in enumerate(panel.sessions):
        t_open = (day + OPEN_OFFSET).tz_localize("UTC").value
        t_close = (day + CLOSE_OFFSET).tz_localize("UTC").value
        ts_kind[t_open] = ("open", i)
        ts_kind[t_close] = ("close", i)
        for t, row in ((t_open, panel.exec_price[i]), (t_close, panel.tr_close[i])):
            js = np.flatnonzero(np.isfinite(row))
            expected[t] = len(js)
            for j in js:
                p = Price(float(row[j]), PRICE_DECIMALS)
                bars.append(Bar(bar_types[j], p, p, p, p, BAR_VOLUME, t, t))

    ctx = {
        "N": N, "ccy": ccy, "instruments": instruments, "bar_types": bar_types,
        "inst_to_asset": {inst.id: j for j, inst in enumerate(instruments)},
        "ts_kind": ts_kind, "expected": expected,
        "fee_rate": fee_rate, "weights": weights, "has": has_weights,
        "liquidate": panel.liquidate, "scaled": [],
        "equity": np.full(T, np.nan), "cash": np.full(T, np.nan),
        "positions": np.zeros((T, N)), "recorded": np.zeros(T, dtype=bool),
    }

    engine = BacktestEngine(BacktestEngineConfig(
        trader_id=TraderId("CONCORDANCE-001"),
        logging=LoggingConfig(bypass_logging=True),
        risk_engine=RiskEngineConfig(max_order_submit_rate="1000000/00:00:01"),
    ))
    fee_model = _TableFee(fee_rate, {t: i for t, (_, i) in ts_kind.items()},
                          ctx["inst_to_asset"], ccy)
    engine.add_venue(VENUE, OmsType.NETTING, AccountType.CASH,
                     starting_balances=[Money(init_cash, ccy)], base_currency=ccy,
                     fee_model=fee_model, bar_execution=True)
    for inst in instruments:
        engine.add_instrument(inst)
    engine.add_data(bars)
    strat = _Executor(ctx)
    engine.add_strategy(strat)
    try:
        engine.run()
    finally:
        engine.dispose()

    if strat.problems:
        raise RuntimeError("Nautilus refused orders: " + "; ".join(strat.problems[:5]))
    missing = np.flatnonzero(~ctx["recorded"])
    if len(missing):
        raise RuntimeError(f"no close recorded for sessions {missing[:5].tolist()}")

    fills = pd.DataFrame(strat.fills, columns=FILL_COLUMNS).astype(
        {"session": np.int64, "asset": np.int64})
    return BacktestResult("nautilus", panel.sessions, panel.assets, ctx["equity"], ctx["cash"],
                          ctx["positions"], fills, sorted(ctx["scaled"]))
