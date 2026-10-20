"""Crash protection: one fixed rule layered on a run's reconciled daily returns.

Rule (declared before any strategy was scored with it, never tuned per strategy): at the
close of each month's last session, if the S&P 500 total-return index (rebuilt from the
members, lake/sp500_tr_daily.parquet) is below its 200-session simple average, hold the safe
asset until a later month-end close is back at or above it; otherwise hold the strategy.

- The switch trades at the next session's open; that whole session still earns the old
  state's return (close-to-close returns only, so the delay is a full session: the stated
  conservative side).
- Each switch costs one full portfolio turnover at commission + half spread.
- Safe assets: T-bills (the daily risk-free rate) or Treasury bonds (10-year note futures
  excess return + the risk-free rate, i.e. a fully collateralized ZN position).

The strategy's own returns on protected days are discarded; on the switch back it buys the
book the strategy holds that day. This is a report on a reconciled return series, not a
fourth engine: the strategy's fills are unchanged (docs/SEMANTICS.md, "Crash protection").
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SMA_SESSIONS = 200
INDEX_FILE = "sp500_tr_daily.parquet"


def build_sp500_index(lake: Path, start="1993-01-04", end=None) -> dict:
    """lake/sp500_tr_daily.parquet: the rebuilt S&P 500 total return (cap-weighted members,
    data/crsp.sp500_total_return) over the whole lake. 1993 is the first start whose panel
    validates (1992 has opens without closes)."""
    from concordance.data.crsp import load_sp500_panel

    lake = Path(lake)
    man_path = lake / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"files": {}}
    end = end or manifest.get("files", {}).get("crsp_daily", {}).get("end", "2025-12-31")
    bench = load_sp500_panel(start, end, lake).meta["benchmark"]
    if bench is None:
        raise ValueError("no mktcap_k in crsp_daily: cannot rebuild the S&P 500")
    gaps = bench.iloc[1:].isna()
    if gaps.any():
        raise ValueError(f"S&P 500 return missing on {int(gaps.sum())} sessions, "
                         f"e.g. {bench.iloc[1:][gaps].index[0].date()}")
    df = pd.DataFrame({"date": bench.index, "ret": bench.fillna(0.0).to_numpy()})
    df.to_parquet(lake / INDEX_FILE, index=False)
    stats = {"status": "ok", "rows": len(df), "start": str(df["date"].iloc[0].date()),
             "end": str(df["date"].iloc[-1].date()),
             "built_from": "crsp_daily + sp500_membership",
             "built_at": datetime.now(UTC).isoformat()}
    manifest.setdefault("files", {})["sp500_tr_daily"] = stats
    man_path.write_text(json.dumps(manifest, indent=2))
    return stats


def invested_mask(sessions: pd.DatetimeIndex, index_ret: pd.Series,
                  sma: int = SMA_SESSIONS) -> pd.Series:
    """1.0 on sessions whose return the strategy earns, 0.0 where the safe asset does.
    `index_ret` must start at least `sma` sessions before sessions[0] and use no data after
    sessions[-1] (the caller trims it)."""
    r = index_ret.loc[:sessions[-1]]
    before = r.index[r.index < sessions[0]]
    if len(before) < sma:
        raise ValueError(f"crash protection needs {sma} S&P sessions before "
                         f"{sessions[0].date()}; the index has {len(before)}")
    level = (1 + r).cumprod()
    above = level >= level.rolling(sma).mean()
    month = pd.Series(r.index.to_period("M"), index=r.index)
    month_end = month != month.shift(-1)
    if r.index[-1] == sessions[-1]:
        month_end.iloc[-1] = False  # the window's last close: no later session to trade on
    state = pd.Series(np.nan, index=r.index)
    state[month_end] = above[month_end].astype(float)
    state = state.ffill().fillna(1.0)
    # Decided at close t, traded at the open of t+1, so the return of t+1 is still the old
    # state's: the new state first earns the return of t+2.
    return state.shift(2).fillna(1.0).reindex(sessions)


def protect(returns: pd.Series, invested: pd.Series, safe: pd.Series,
            cost_bps: float) -> pd.Series:
    inv = invested.reindex(returns.index).to_numpy()
    switch = np.abs(np.diff(np.concatenate([[1.0], inv])))
    out = inv * returns.to_numpy() + (1 - inv) * safe.reindex(returns.index).to_numpy()
    return pd.Series(out - switch * cost_bps / 1e4, index=returns.index)


def summary(r: pd.Series, periods: int = 252) -> dict:
    """The same definitions as report.metrics, on a daily return series."""
    eq = np.concatenate([[1.0], np.cumprod(1 + r.to_numpy())])
    years = len(r) / periods
    sd = r.std(ddof=1)
    return {"cagr": float(eq[-1] ** (1 / years) - 1) if years > 0 else np.nan,
            "vol": float(sd * np.sqrt(periods)),
            "sharpe": float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan,
            "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min())}


def _safe_assets(lake: Path, sessions: pd.DatetimeIndex) -> dict[str, pd.Series]:
    from concordance.data.futures import load_futures_panel
    from concordance.data.riskfree import load_rf

    rf = load_rf(lake)
    missing = sessions.difference(rf.index)
    if len(missing) > 5:
        raise ValueError(f"risk-free rate missing on {len(missing)} sessions")
    rf = rf.reindex(sessions).fillna(0.0)
    out = {"tbills": rf}
    if (lake / "futures_bars.parquet").exists():
        zn = load_futures_panel(sessions[0], sessions[-1], lake, roots=["ZN"])
        ret = pd.Series(zn.fields["ret"][:, 0], index=zn.sessions)
        holes = int(ret.iloc[1:].isna().sum())
        if holes > 5:
            raise ValueError(f"10-year note return missing on {holes} sessions")
        # Sessions the panel dropped (bond market shut, e.g. Columbus Day) have no move: the
        # next settle's return spans them.
        out["bonds"] = rf + ret.reindex(sessions).fillna(0.0)
    return out


def crash_protection(returns: pd.Series, lake: Path, cost_bps: float,
                     benchmark: pd.Series | None = None) -> tuple[dict, pd.DataFrame]:
    """Metrics of `returns` (a run's evaluated daily returns) under the rule, per safe
    asset, plus the S&P 500 under the same rule for comparison. Also the daily frame."""
    lake = Path(lake)
    path = lake / INDEX_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `concordance build-sp500`")
    idx = pd.read_parquet(path)
    index_ret = idx.set_index(pd.to_datetime(idx["date"]))["ret"]
    sessions = pd.DatetimeIndex(returns.index)
    if sessions[-1] > index_ret.index[-1]:
        raise ValueError(f"S&P index ends {index_ret.index[-1].date()}, before the run")
    inv = invested_mask(sessions, index_ret)
    bench = (benchmark.reindex(sessions) if benchmark is not None
             else index_ret.reindex(sessions))
    frame = pd.DataFrame({"invested": inv.to_numpy()}, index=sessions)
    out = {"rule": f"S&P 500 below its {SMA_SESSIONS}-session average at a month-end close",
           "cost_bps_per_switch": cost_bps,
           "share_protected": float(1 - inv.mean()),
           "switches": int(np.abs(np.diff(np.concatenate([[1.0], inv.to_numpy()]))).sum())}
    for name, safe in _safe_assets(lake, sessions).items():
        p = protect(returns, inv, safe, cost_bps)
        frame[f"return_{name}"] = p.to_numpy()
        out[name] = {**summary(p),
                     "benchmark": summary(protect(bench.fillna(0.0), inv, safe, cost_bps))}
    return out, frame
