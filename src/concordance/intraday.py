"""Minute-bar stock execution alongside Concordance's unchanged daily backtester.

Signals are decisions at completed bars. The two engines independently resolve orders
from 1-minute OHLC, then their trade ledgers and equity must agree. Quotes must be
present; this interface refuses to fabricate transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class IntradayError(ValueError):
    pass


@dataclass(frozen=True)
class IntradayConfig:
    initial_capital: float = 100_000.0
    risk_fraction: float = 0.005
    max_positions: int = 5
    max_gross: float = 1.0
    participation: float = 0.01
    commission_bps: float = 1.0
    slippage_bps: float = 1.0
    impact_bps: float = 1.0
    borrow_bps_annual: float = 50.0


def _utc(values: pd.Series) -> pd.Series:
    if not isinstance(values.dtype, pd.DatetimeTZDtype):
        if not values.astype("string").str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$").all():
            raise IntradayError("timestamps must include a UTC offset")
    return pd.to_datetime(values, utc=True)


def _prepare(bars: pd.DataFrame, signals: pd.DataFrame):
    need = {"timestamp", "asset", "open", "high", "low", "close", "volume", "bid", "ask"}
    if not need <= set(bars):
        raise IntradayError(f"minute bars missing {sorted(need - set(bars))}")
    if bars.empty:
        raise IntradayError("no minute bars")
    b = bars.copy()
    b["timestamp"] = _utc(b["timestamp"])
    if b[["timestamp", "asset"]].duplicated().any():
        raise IntradayError("duplicate asset/minute")
    if (
        (b["open"] <= 0)
        | (b["low"] <= 0)
        | (b["close"] <= 0)
        | (b["high"] < b[["open", "close"]].max(axis=1))
        | (b["low"] > b[["open", "close"]].min(axis=1))
        | (b["high"] < b["low"])
        | (b["volume"] < 0)
        | (b["bid"] <= 0)
        | (b["ask"] < b["bid"])
    ).any():
        raise IntradayError("invalid price, quote, or volume")
    b["session"] = b["timestamp"].dt.tz_convert("America/New_York").dt.date.astype(str)
    b = b.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    b["cost_bid"] = b.groupby(["asset", "session"])["bid"].shift(1)
    b["cost_ask"] = b.groupby(["asset", "session"])["ask"].shift(1)
    b["prior_volume"] = b.groupby(["asset", "session"])["volume"].shift(1)
    b = b.sort_values(["timestamp", "asset"]).reset_index(drop=True)
    s = signals.copy()
    required = {
        "asset",
        "signal_timestamp",
        "direction",
        "stop_pct",
        "reward_risk",
        "max_minutes",
        "timeframe",
    }
    if not required <= set(s):
        raise IntradayError(f"signals missing {sorted(required - set(s))}")
    s["signal_timestamp"] = _utc(s["signal_timestamp"])
    if "trailing_pct" not in s:
        s["trailing_pct"] = 0.0
    if (
        not s["direction"].isin([-1, 1]).all()
        or not (
            (s["stop_pct"] > 0)
            & (s["stop_pct"] < 0.1)
            & (s["reward_risk"] > 0)
            & (s["reward_risk"] <= 10)
        ).all()
        or not s["timeframe"].isin([1, 5, 15]).all()
        or not s["max_minutes"].between(1, 390).all()
        or not s["trailing_pct"].between(0, 0.09999).all()
    ):
        raise IntradayError("invalid signal direction or exit")
    s["entry_timestamp"] = s["signal_timestamp"] + pd.to_timedelta(s["timeframe"], unit="m")
    if s[["asset", "entry_timestamp"]].duplicated().any():
        raise IntradayError("duplicate entry decision")
    return b, s.sort_values(["entry_timestamp", "asset"]).reset_index(drop=True)


def _cost(bar, cfg: IntradayConfig) -> float:
    if not np.isfinite(bar.cost_bid) or not np.isfinite(bar.cost_ask):
        raise IntradayError("no prior quote for execution")
    half = (bar.cost_ask - bar.cost_bid) / (bar.cost_ask + bar.cost_bid)
    return half + (cfg.slippage_bps + cfg.impact_bps) / 1e4


def _fill(price: float, direction: int, cost: float) -> float:
    return price * (1 + direction * cost)


def _resolve(bar, position):
    d = position["direction"]
    stop, target = position["stop"], position["target"]
    trail = position.get("trailing_pct", 0.0)
    if trail:
        peak = position["peak"]
        stop = max(stop, peak * (1 - trail)) if d == 1 else min(stop, peak * (1 + trail))
    gap_stop = bar.open <= stop if d == 1 else bar.open >= stop
    hit_stop = bar.low <= stop if d == 1 else bar.high >= stop
    gap_target = bar.open >= target if d == 1 else bar.open <= target
    hit_target = bar.high >= target if d == 1 else bar.low <= target
    if gap_stop or hit_stop:
        return (bar.open if gap_stop else stop), "stop"
    if gap_target or hit_target:
        return (bar.open if gap_target else target), "target"
    return None


def _reference(b: pd.DataFrame, s: pd.DataFrame, cfg: IntradayConfig):
    cash = cfg.initial_capital
    positions = {}
    fills, marks = [], []
    lookup = {(r.asset, r.entry_timestamp): r for r in s.itertuples(index=False)}
    last = b.groupby(["session", "asset"])["timestamp"].max().to_dict()
    prices = {}
    for bar in b.itertuples(index=False):
        key = (bar.asset, bar.timestamp)
        equity = cash + sum(p["qty"] * prices.get(a, p["entry"]) for a, p in positions.items())
        sig = lookup.get(key)
        if sig is not None and bar.asset not in positions and len(positions) < cfg.max_positions:
            cost = _cost(bar, cfg)
            entry = _fill(bar.open, sig.direction, cost)
            stop = entry * (1 - sig.direction * sig.stop_pct)
            target = entry * (1 + sig.direction * sig.stop_pct * sig.reward_risk)
            capacity = max(
                0.0,
                cfg.max_gross * equity
                - sum(abs(p["qty"] * prices.get(a, p["entry"])) for a, p in positions.items()),
            )
            qty = int(
                min(
                    equity * cfg.risk_fraction / (entry * sig.stop_pct),
                    capacity / entry,
                    cfg.participation * bar.prior_volume,
                )
            )
            if qty > 0:
                signed = sig.direction * qty
                fee = abs(signed) * entry * cfg.commission_bps / 1e4
                cash -= signed * entry + fee
                positions[bar.asset] = {
                    "qty": signed,
                    "entry": entry,
                    "direction": sig.direction,
                    "stop": stop,
                    "target": target,
                    "trailing_pct": float(sig.trailing_pct),
                    "peak": entry,
                    "deadline": bar.timestamp + pd.Timedelta(minutes=int(sig.max_minutes) - 1),
                }
                fills.append((bar.timestamp, bar.asset, signed, entry, fee, "entry"))
        if bar.asset in positions:
            p = positions[bar.asset]
            outcome = _resolve(bar, p)
            if outcome is None and bar.timestamp >= p["deadline"]:
                outcome = (bar.close, "time")
            if outcome is None and bar.timestamp == last[(bar.session, bar.asset)]:
                outcome = (bar.close, "session_end")
            if outcome is None:
                p["peak"] = (
                    max(p["peak"], bar.high) if p["direction"] == 1 else min(p["peak"], bar.low)
                )
            if outcome is not None:
                px, reason = outcome
                exit_qty = -p["qty"]
                fill = _fill(px, 1 if exit_qty > 0 else -1, _cost(bar, cfg))
                fee = abs(exit_qty) * fill * cfg.commission_bps / 1e4
                if p["direction"] == -1:
                    fee += abs(exit_qty) * fill * cfg.borrow_bps_annual / 1e4 / 252
                cash -= exit_qty * fill + fee
                fills.append((bar.timestamp, bar.asset, exit_qty, fill, fee, reason))
                del positions[bar.asset]
        prices[bar.asset] = bar.close
        marks.append(
            (
                bar.timestamp,
                cash + sum(p["qty"] * prices.get(a, p["entry"]) for a, p in positions.items()),
            )
        )
    return pd.DataFrame(
        fills, columns=["timestamp", "asset", "qty", "price", "fee", "reason"]
    ), pd.DataFrame(marks, columns=["timestamp", "equity"])


def _replay(b: pd.DataFrame, s: pd.DataFrame, cfg: IntradayConfig):
    """Independent event ordering and cash accounting for reconciliation."""
    cash = float(cfg.initial_capital)
    book = {}
    events = []
    equity_rows = []
    signals = s.set_index(["entry_timestamp", "asset"])
    last = b.groupby(["session", "asset"])["timestamp"].max().to_dict()
    last_prices = {}
    for time, frame in b.groupby("timestamp", sort=True):
        for bar in frame.itertuples(index=False):
            key = (time, bar.asset)
            if key in signals.index and bar.asset not in book and len(book) < cfg.max_positions:
                sig = signals.loc[key]
                nav = cash + sum(v[0] * last_prices.get(a, v[1]) for a, v in book.items())
                gross = sum(abs(v[0] * last_prices.get(a, v[1])) for a, v in book.items())
                direction = int(sig.direction)
                price = _fill(bar.open, direction, _cost(bar, cfg))
                shares = int(
                    min(
                        nav * cfg.risk_fraction / (price * sig.stop_pct),
                        max(0, cfg.max_gross * nav - gross) / price,
                        cfg.participation * bar.prior_volume,
                    )
                )
                if shares:
                    q = direction * shares
                    fee = abs(q) * price * cfg.commission_bps / 1e4
                    cash = cash - q * price - fee
                    book[bar.asset] = (
                        q,
                        price,
                        price * (1 - direction * sig.stop_pct),
                        price * (1 + direction * sig.stop_pct * sig.reward_risk),
                        time + pd.Timedelta(minutes=int(sig.max_minutes) - 1),
                        float(sig.trailing_pct),
                        price,
                    )
                    events.append((time, bar.asset, q, price, fee, "entry"))
            if bar.asset in book:
                q, entry, stop, target, deadline, trail, peak = book[bar.asset]
                direction = 1 if q > 0 else -1
                p = {
                    "direction": direction,
                    "stop": stop,
                    "target": target,
                    "trailing_pct": trail,
                    "peak": peak,
                }
                hit = _resolve(bar, p)
                if hit is None and time >= deadline:
                    hit = (bar.close, "time")
                if hit is None and time == last[(bar.session, bar.asset)]:
                    hit = (bar.close, "session_end")
                if hit is not None:
                    exit_px, reason = hit
                    xq = -q
                    price = _fill(exit_px, 1 if xq > 0 else -1, _cost(bar, cfg))
                    fee = abs(xq) * price * cfg.commission_bps / 1e4
                    if direction == -1:
                        fee += abs(xq) * price * cfg.borrow_bps_annual / 1e4 / 252
                    cash = cash - xq * price - fee
                    events.append((time, bar.asset, xq, price, fee, reason))
                    book.pop(bar.asset)
                else:
                    next_peak = max(peak, bar.high) if direction == 1 else min(peak, bar.low)
                    book[bar.asset] = (q, entry, stop, target, deadline, trail, next_peak)
            last_prices[bar.asset] = bar.close
            equity_rows.append(
                (time, cash + sum(v[0] * last_prices.get(a, v[1]) for a, v in book.items()))
            )
    return pd.DataFrame(
        events, columns=["timestamp", "asset", "qty", "price", "fee", "reason"]
    ), pd.DataFrame(equity_rows, columns=["timestamp", "equity"])


def run_intraday(
    bars: pd.DataFrame, signals: pd.DataFrame, config: IntradayConfig | None = None
) -> dict:
    config = config or IntradayConfig()
    if not (
        config.initial_capital > 0
        and 0 < config.risk_fraction <= 1
        and config.max_positions >= 1
        and 0 < config.max_gross <= 2
        and 0 < config.participation <= 1
    ):
        raise IntradayError("invalid portfolio configuration")
    b, s = _prepare(bars, signals)
    first = _reference(b, s, config)
    second = _replay(b, s, config)
    a, ae = first
    c, ce = second
    if not a.equals(c) or not np.allclose(ae["equity"], ce["equity"], rtol=0, atol=1e-6):
        return {
            "status": "rejected",
            "reason": "intraday engines disagree",
            "fills": a,
            "equity": ae,
        }
    daily = ae.groupby(ae["timestamp"].dt.tz_convert("America/New_York").dt.date)["equity"].last()
    returns = daily.pct_change().dropna()
    drawdown = (
        daily / np.maximum.accumulate(np.r_[config.initial_capital, daily.to_numpy()])[1:] - 1
    )
    return {
        "status": "ok",
        "fills": a,
        "equity": ae,
        "daily_equity": daily,
        "metrics": {
            "final_equity": float(daily.iloc[-1]),
            "net_profit": float(daily.iloc[-1] - config.initial_capital),
            "max_drawdown": float(drawdown.min()),
            "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
            if len(returns) > 1 and returns.std(ddof=1) > 0
            else None,
            "n_trades": int((a["reason"] == "entry").sum()),
        },
    }
