"""Why did engines disagree? `concordance diagnose RUN_DIR` reruns a saved run's engines and
shows the first session where each engine stops agreeing with the reference.

It reads the run's own request.json and strategy.py copy (like `reproduce`), so it works on
rejected runs, which keep no per-engine artifacts. Nothing is loosened or changed: the
report is the same reconciliation, with each engine's state and fills at that session.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from concordance.reconcile import QTY_TOL, _qty_allowance, reconcile, valuation_marks

TOP = 10  # rows per table


def _id(x):
    """An asset id as JSON: a PERMNO-based integer, or a string on synthetic panels."""
    return x.item() if isinstance(x, np.generic) else x


def _num(x):
    x = float(x)
    return None if not np.isfinite(x) else x


def session_detail(panel, sched, a, b, i: int, marks: np.ndarray, top: int = TOP) -> dict:
    """Engine `b` against reference `a` at session index `i`."""
    assets = np.asarray(panel.assets)
    prev = max(i - 1, 0)
    detail = {
        "session": int(i),
        "date": str(panel.sessions[i].date()),
        "new_weights": bool(sched.has[i]),
        "buys_scaled": {a.engine: i in a.scaled_sessions, b.engine: i in b.scaled_sessions},
        "before_open": {
            "cash": [_num(a.cash[prev]), _num(b.cash[prev])],
            "equity": [_num(a.equity[prev]), _num(b.equity[prev])],
        } if i > 0 else None,
        "at_close": {"cash": [_num(a.cash[i]), _num(b.cash[i])],
                     "equity": [_num(a.equity[i]), _num(b.equity[i])]},
    }
    key = ["session", "phase", "asset"]
    m = a.fills.merge(b.fills, on=key, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m["_merge"] == "both"].reset_index(drop=True)
    dq = (both["qty_a"] - both["qty_b"]).abs()
    fee_gap = (both["fee_a"] - both["fee_b"]).abs()
    allowed = _qty_allowance(a, b, both, dq, fee_gap, marks) if len(both) else np.zeros(0)
    both["allowed"] = allowed
    both["dq"] = dq
    here = m[m["session"] == i]
    rows = []
    for _, r in here.iterrows():
        j = int(r["asset"])
        match = both[(both["session"] == i) & (both["phase"] == r["phase"])
                     & (both["asset"] == j)]
        gap = float(match["dq"].iloc[0]) if len(match) else None
        if r["_merge"] == "both" and gap is not None and gap <= QTY_TOL \
                and abs(r["fee_a"] - r["fee_b"]) <= 1e-9:
            continue
        rows.append({
            "asset": _id(assets[j]), "phase": int(r["phase"]),
            "qty": [_num(r["qty_a"]), _num(r["qty_b"])],
            "qty_gap": gap,
            "allowed_gap": float(match["allowed"].iloc[0]) if len(match) else None,
            "explained": bool(gap is not None and gap <= match["allowed"].iloc[0] + QTY_TOL),
            "price": [_num(r["price_a"]), _num(r["price_b"])],
            "fee": [_num(r["fee_a"]), _num(r["fee_b"])],
            "target_weight": _num(sched.weights[i, j]) if sched.has[i] else None,
            "position_before": [_num(a.positions[prev, j]), _num(b.positions[prev, j])]
            if i > 0 else [0.0, 0.0],
            "only_in": None if r["_merge"] == "both"
            else (a.engine if r["_merge"] == "left_only" else b.engine),
        })
    price = np.abs(np.nan_to_num(marks[i]))
    rows.sort(key=lambda x: (x["explained"], -(x["qty_gap"] or np.inf)
                             * price[list(assets).index(x["asset"])]))
    detail["fills_that_differ"] = rows[:top]
    detail["n_fills_that_differ"] = len(rows)
    dpos = np.abs(a.positions[i] - b.positions[i])
    order = np.argsort(-dpos * price)
    detail["positions_that_differ_at_close"] = [
        {"asset": _id(assets[j]), "qty": [_num(a.positions[i, j]), _num(b.positions[i, j])],
         "value_gap": _num(dpos[j] * price[j])}
        for j in order[:top] if dpos[j] > QTY_TOL]
    return detail


def first_difference(a, b) -> int | None:
    """The first session with any difference at all, explained or not."""
    diff = np.abs(a.positions - b.positions).max(axis=1) > QTY_TOL
    diff |= np.abs(a.cash - b.cash) > 1e-6
    return int(np.argmax(diff)) if diff.any() else None


def diagnose(run_dir: str | Path, engines=None, panel=None, top: int = TOP) -> dict:
    from concordance.api import load_panel, load_strategy, request_tables, resolve_request
    from concordance.runner import run_engine
    from concordance.strategy import build_schedule

    run_dir = Path(run_dir).resolve()
    req = json.loads((run_dir / "request.json").read_text())
    req["strategy"]["path"] = str(run_dir / "strategy.py")
    req = resolve_request(req)
    names = list(engines or req["engines"])
    if "reference" not in names:
        names.insert(0, "reference")
    if len(names) < 2:
        raise ValueError("diagnose compares engines: name at least one besides the reference")
    strategy = load_strategy(req["strategy"]["path"], req["strategy"]["params"])
    if panel is None:
        panel, _, _ = load_panel(req, strategy)
    sched = build_schedule(strategy, panel)
    tables = request_tables(req, panel)
    results = {e: run_engine(e, panel, sched, tables, req["initial_capital"]) for e in names}
    marks = valuation_marks(panel.tr_close)
    report = reconcile(results, marks)
    ref = results["reference"]
    out = {"run_dir": str(run_dir), "ok": report.ok, "summary": report.summary(),
           "engines": {}}
    for chk in report.checks:
        b = results[chk.engine]
        first = first_difference(ref, b)
        entry = {"ok": chk.ok, "problems": chk.problems[:20],
                 "max_equity_gap": chk.max_equity_diff, "max_cash_gap": chk.max_cash_diff,
                 "first_difference": str(panel.sessions[first].date()) if first is not None
                 else None}
        if not chk.ok and chk.first_bad_session is not None:
            entry["first_failure"] = session_detail(panel, sched, ref, b,
                                                    chk.first_bad_session, marks, top)
        out["engines"][chk.engine] = entry
    (run_dir / "diagnose.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def render(out: dict) -> str:
    lines = [out["summary"], ""]
    for engine, e in out["engines"].items():
        if e["ok"]:
            lines.append(f"{engine}: agrees with the reference"
                         + (f" (small explained gaps from {e['first_difference']})"
                            if e["first_difference"] else ""))
            continue
        d = e.get("first_failure")
        lines.append(f"{engine}: first unexplained difference on {d['date'] if d else '?'}"
                     f" (first difference of any kind: {e['first_difference']})")
        if not d:
            continue
        if d["before_open"]:
            c, q = d["before_open"]["cash"], d["before_open"]["equity"]
            lines.append(f"  before the open: cash {c[0]:.6f} vs {c[1]:.6f} "
                         f"(gap {c[0] - c[1]:+.3e}), equity gap {q[0] - q[1]:+.3e}")
        lines.append(f"  new weights: {d['new_weights']}; buys scaled to fit cash: "
                     + ", ".join(f"{k}={v}" for k, v in d["buys_scaled"].items()))
        if d["fills_that_differ"]:
            lines.append(f"  fills that differ ({d['n_fills_that_differ']}):")
            frame = pd.DataFrame([{
                "asset": r["asset"], "phase": r["phase"], "ref_qty": r["qty"][0],
                f"{engine}_qty": r["qty"][1], "gap": r["qty_gap"],
                "allowed": r["allowed_gap"], "explained": r["explained"],
                "ref_fee": r["fee"][0], f"{engine}_fee": r["fee"][1],
                "only_in": r["only_in"]} for r in d["fills_that_differ"]])
            lines.extend("    " + x for x in frame.to_string(index=False).splitlines())
        if d["positions_that_differ_at_close"]:
            lines.append("  positions that differ at the close:")
            for p in d["positions_that_differ_at_close"]:
                lines.append(f"    asset {p['asset']}: {p['qty'][0]} vs {p['qty'][1]} "
                             f"(${p['value_gap']:.6f})")
    lines.append("")
    lines.append(f"full report: {out['run_dir']}/diagnose.json")
    return "\n".join(lines)
