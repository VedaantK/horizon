"""Performance summary and HTML tear sheet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from concordance.result import BacktestResult


def metrics(res: BacktestResult, periods: int = 252) -> dict:
    eq = res.equity_series
    r = eq.pct_change().dropna()
    years = len(r) / periods
    sd = r.std(ddof=1)
    dd = eq / eq.cummax() - 1
    traded = (res.fills["qty"].abs() * res.fills["price"]).sum()
    avg_eq = eq.mean()
    return {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        "final_equity": float(eq.iloc[-1]),
        "cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan,
        "vol": float(sd * np.sqrt(periods)),
        "sharpe": float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "annual_turnover": float(traded / avg_eq / years) if years > 0 else np.nan,
        "total_fees": float(res.fills["fee"].sum()),
        "n_fills": int(len(res.fills)),
        "open_positions_at_end": int((res.positions[-1] != 0).sum()),
        "scaled_sessions": len(res.scaled_sessions),
    }


def html_report(res: BacktestResult, path: Path, title: str,
                benchmark: pd.Series | None = None, extra: dict | None = None) -> Path:
    """quantstats tear sheet with Concordance's reconciliation/validation block prepended."""
    import logging

    import quantstats as qs

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rets = res.returns.iloc[1:]
    rets.index = pd.DatetimeIndex(rets.index)
    qs.reports.html(rets, benchmark=benchmark, output=str(path), title=title,
                    download_filename=path.name)
    if extra:
        rows = "".join(f"<tr><td>{k}</td><td><pre>{v}</pre></td></tr>" for k, v in extra.items())
        block = (f"<div style='margin:16px;font-family:monospace'><h2>Concordance</h2>"
                 f"<table border=1 cellpadding=4>{rows}</table></div>")
        html = path.read_text()
        path.write_text(html.replace("<body>", "<body>" + block, 1))
    return path
