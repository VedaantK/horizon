"""Performance summary and HTML tear sheet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from concordance.result import BacktestResult


def metrics(res: BacktestResult, periods: int = 252, start: int = 0,
            benchmark: pd.Series | None = None) -> dict:
    """Performance from session `start` on (pass the warm-up length: before it the strategy
    cannot trade, so including it would dilute every rate)."""
    eq = res.equity_series.iloc[start:]
    r = eq.pct_change().dropna()
    years = len(r) / periods
    sd = r.std(ddof=1)
    dd = eq / eq.cummax() - 1
    f = res.fills[res.fills["session"] >= start]
    traded = (f["qty"].abs() * f["price"]).sum()
    avg_eq = eq.mean()
    return {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        "final_equity": float(eq.iloc[-1]),
        "cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan,
        "vol": float(sd * np.sqrt(periods)),
        "sharpe": float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "annual_turnover": float(traded / avg_eq / years) if years > 0 else np.nan,
        "total_fees": float(f["fee"].sum()),
        "n_fills": int(len(f)),
        "open_positions_at_end": int((res.positions[-1] != 0).sum()),
        "scaled_sessions": len(res.scaled_sessions),
        **_benchmark_metrics(benchmark, eq.index, periods),
    }


FACTORS = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
NW_LAGS = 5  # Newey-West lags for the alpha t-stat (daily returns: a trading week)


def factor_exposures(returns: pd.Series, factors: pd.DataFrame | None,
                     periods: int = 252) -> dict:
    """OLS of daily excess returns on Fama-French 5 + momentum: annualized alpha, its
    Newey-West t-stat, betas and R^2. A strategy whose "edge" is a known factor shows a
    big beta and an alpha near zero."""
    if factors is None:
        return {}
    f = factors.set_index("date").reindex(pd.DatetimeIndex(returns.index))
    cols = [c for c in FACTORS if c in f.columns]
    ok = f[cols + ["rf"]].notna().all(axis=1).to_numpy() & np.isfinite(returns.to_numpy())
    if ok.sum() < 60:
        return {"factor_model": "not enough factor data for this window"}
    y = returns.to_numpy()[ok] - f["rf"].to_numpy()[ok]
    X = np.column_stack([np.ones(ok.sum()), f[cols].to_numpy()[ok]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n = len(y)
    xtx_inv = np.linalg.inv(X.T @ X)
    xe = X * e[:, None]
    s = xe.T @ xe
    for lag in range(1, NW_LAGS + 1):
        w = 1 - lag / (NW_LAGS + 1)
        g = xe[lag:].T @ xe[:-lag]
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    se_alpha = float(np.sqrt(cov[0, 0]))
    sst = ((y - y.mean()) ** 2).sum()
    return {
        "factor_model": "FF5+UMD",
        "factor_alpha_annual": float(beta[0] * periods),
        "factor_alpha_t": float(beta[0] / se_alpha) if se_alpha > 0 else np.nan,
        "factor_betas": {c: float(b) for c, b in zip(cols, beta[1:], strict=True)},
        "factor_r2": float(1 - (e ** 2).sum() / sst) if sst > 0 else np.nan,
        "factor_n_days": int(n),
    }


def leg_attribution(res: BacktestResult, marks: np.ndarray, start: int = 0) -> dict:
    """Profit by side from session `start` on: long and short P&L before costs, fees and
    stock borrow, which add up exactly to the change in equity; plus average gross and net
    exposure as fractions of equity. A day's P&L on an asset belongs to the side it was
    held on at the previous close (or, if flat then, at this close)."""
    T, N = res.positions.shape
    pos = res.positions
    held = np.where(pos != 0, pos * np.nan_to_num(marks), 0.0)
    trade = np.zeros((T, N))
    f = res.fills
    np.add.at(trade, (f["session"].to_numpy(), f["asset"].to_numpy()),
              (f["qty"] * f["price"]).to_numpy())
    pnl = np.diff(held, axis=0, prepend=np.zeros((1, N))) - trade
    prev = np.vstack([np.zeros((1, N)), pos[:-1]])
    side = np.where(prev != 0, np.sign(prev), np.sign(pos))
    rows = slice(start + 1, None)  # P&L earned after the start close
    fees = f.loc[f["session"] > start, "fee"].sum()
    borrow = res.borrow[rows].sum() if res.borrow is not None else 0.0
    eq = res.equity[start:]
    long_v = np.where(pos > 0, held, 0.0).sum(axis=1)[start:]
    short_v = -np.where(pos < 0, held, 0.0).sum(axis=1)[start:]
    return {
        "long_pnl": float(pnl[rows][side[rows] > 0].sum()),
        "short_pnl": float(pnl[rows][side[rows] < 0].sum()),
        "fees_after_start": float(fees),
        "total_borrow": float(borrow),
        "avg_gross_exposure": float(np.mean((long_v + short_v) / eq)),
        "avg_net_exposure": float(np.mean((long_v - short_v) / eq)),
        "avg_short_exposure": float(np.mean(short_v / eq)),
    }


def _benchmark_metrics(b: pd.Series | None, index: pd.DatetimeIndex, periods: int) -> dict:
    if b is None:
        return {}
    r = b.reindex(index).iloc[1:]
    if r.isna().any() or len(r) == 0:
        return {"benchmark": "incomplete for this window"}
    years = len(r) / periods
    return {"benchmark_cagr": float((1 + r).prod() ** (1 / years) - 1),
            "benchmark_sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(periods))}


def html_report(res: BacktestResult, path: Path, title: str,
                benchmark: pd.Series | None = None, extra: dict | None = None,
                start: int = 0) -> Path:
    """quantstats tear sheet with Concordance's reconciliation/validation block prepended."""
    import logging

    import quantstats as qs

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rets = res.returns.iloc[max(start, 1):]
    rets.index = pd.DatetimeIndex(rets.index)
    if benchmark is not None:
        benchmark = benchmark.reindex(rets.index)
        if benchmark.isna().any():
            benchmark = None
    qs.reports.html(rets, benchmark=benchmark, output=str(path), title=title,
                    download_filename=path.name)
    if extra:
        rows = "".join(f"<tr><td>{k}</td><td><pre>{v}</pre></td></tr>" for k, v in extra.items())
        block = (f"<div style='margin:16px;font-family:monospace'><h2>Concordance</h2>"
                 f"<table border=1 cellpadding=4>{rows}</table></div>")
        html = path.read_text()
        path.write_text(html.replace("<body>", "<body>" + block, 1))
    return path
