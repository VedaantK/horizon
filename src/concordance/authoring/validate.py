"""Check a strategy file against docs/STRATEGY_CONTRACT.md and say exactly what is wrong.

Two layers:
- **static** (AST, no code runs): allowed imports only, no file/network/process access, no
  private attributes such as `view._panel` that would bypass the point-in-time view;
- **dynamic** (in a child process with a timeout): the file loads, the strategy runs on
  synthetic CRSP-shaped markets, is deterministic, passes the look-ahead truncation check,
  and its orders execute and reconcile on all three engines; optionally a short run on real
  CRSP data.

The report is JSON made for a feedback loop: each error has a `check`, a `message`, and when
it comes from the strategy's own code a `line` and the source text there.

This is a contract check, not a security sandbox: run untrusted files in an isolated
workspace as well.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ALLOWED_IMPORTS = {
    "__future__", "numpy", "pandas", "math", "statistics", "dataclasses", "typing",
    "collections", "functools", "itertools", "warnings", "enum", "concordance.strategy",
    "concordance.calendar",
}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
                   "globals", "locals", "vars", "exit", "quit", "help"}
FORBIDDEN_ATTRS = {"load", "save", "savez", "savez_compressed", "loadtxt", "savetxt",
                   "genfromtxt", "fromfile", "tofile", "memmap", "to_csv", "to_parquet",
                   "to_pickle", "to_json", "to_excel", "to_sql", "to_hdf", "to_feather",
                   "read_pickle", "system", "popen"}
ALLOWED_DUNDERS = {"__init__", "__name__"}


def _err(check: str, message: str, line: int | None = None, code: str | None = None) -> dict:
    out = {"check": check, "message": message}
    if line is not None:
        out["line"] = line
    if code:
        out["code"] = code.strip()
    return out


def static_check(src: str) -> tuple[list[dict], list[dict]]:
    errors, warnings = [], []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [_err("syntax", e.msg, e.lineno, e.text)], []
    lines = src.splitlines()

    def code(n):
        return lines[n.lineno - 1] if 0 < getattr(n, "lineno", 0) <= len(lines) else None

    defines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""])
            for m in mods:
                if m not in ALLOWED_IMPORTS and m.split(".")[0] not in {"numpy", "pandas"}:
                    errors.append(_err("imports", f"import of '{m}' is not allowed; allowed: "
                                       f"{sorted(ALLOWED_IMPORTS)}", node.lineno, code(node)))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                errors.append(_err("forbidden", f"'{node.func.id}()' is not allowed in a "
                                   "strategy", node.lineno, code(node)))
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a in FORBIDDEN_ATTRS or a.startswith("read_"):
                errors.append(_err("forbidden", f"'.{a}' reads or writes files; strategies "
                                   "use only the data in `view`", node.lineno, code(node)))
            elif a.startswith("__") and a not in ALLOWED_DUNDERS:
                errors.append(_err("forbidden", f"dunder attribute '.{a}' is not allowed",
                                   node.lineno, code(node)))
            elif a.startswith("_") and not (isinstance(node.value, ast.Name)
                                            and node.value.id in {"self", "cls"}):
                errors.append(_err("lookahead", f"private attribute '.{a}' on another object "
                                   "(e.g. view._panel holds the FUTURE); use view[...] and "
                                   "view.position", node.lineno, code(node)))
        elif isinstance(node, ast.FunctionDef | ast.ClassDef) and node in tree.body:
            defines.add(node.name)
        elif isinstance(node, ast.Assign) and node in tree.body:
            defines.update(t.id for t in node.targets if isinstance(t, ast.Name))
    if not {"make_strategy", "STRATEGY"} & defines:
        errors.append(_err("contract", "define `def make_strategy(**params)` (preferred) or "
                           "`STRATEGY = MyStrategy()` at module level"))
    if "make_strategy" not in defines:
        warnings.append(_err("contract", "no make_strategy(**params): parameters cannot be "
                             "varied from a run request"))
    return errors, warnings


# ------------------------------------------------------------------------- dynamic (child)


def _frames_in(path: str, exc: BaseException) -> dict:
    """The innermost traceback frame inside the strategy file, for a precise pointer."""
    tb = traceback.extract_tb(exc.__traceback__)
    mine = [f for f in tb if Path(f.filename).resolve() == Path(path).resolve()]
    if not mine:
        return {}
    f = mine[-1]
    return {"line": f.lineno, "code": f.line}


HINTS = {
    "negative weights": "long-only: weights must be >= 0 (shorting is not supported yet)",
    "exceeds 1": "weights must sum to <= 1 (no leverage); scale them down",
    "shape": "arrays must have one entry per asset: len(view.assets)",
    "unknown assets": "dict keys must be asset ids from view.assets",
    "entry_limit": "entry_limit opens a flat position only, with a positive weight in the "
                   "same Orders",
    "KeyError": "unknown data field; available: open high low close ret retx volume bid ask "
                "shrout mktcap_k tr_close member",
}


def _hint(msg: str) -> str | None:
    for k, v in HINTS.items():
        if k in msg:
            return v
    return None


def _child(path: str, params: dict | None, real_window: tuple[str, str] | None) -> dict:
    import numpy as np

    from concordance.api import StrategyError, load_strategy
    from concordance.costs import CostModel, LiquidityModel, exec_tables
    from concordance.panel import build_panel
    from concordance.reconcile import reconcile
    from concordance.runner import ALL_ENGINES, run_engine
    from concordance.strategy import LookaheadError, build_schedule
    from concordance.synthetic import random_long_frame
    from concordance.validation.lookahead import check_truncation_invariance

    errors, warnings, stats = [], [], {}

    def fresh():
        return load_strategy(path, params)

    try:
        s = fresh()
    except StrategyError as e:
        cause = e.__cause__ or e
        return {"errors": [_err("load", str(e), **_frames_in(path, cause))],
                "warnings": [], "stats": {}}
    except Exception as e:
        return {"errors": [_err("load", f"{type(e).__name__}: {e}", **_frames_in(path, e))],
                "warnings": [], "stats": {}}

    name = getattr(s, "name", None)
    if not isinstance(name, str) or name in ("", "strategy"):
        errors.append(_err("contract", "set a class attribute `name = \"...\"` (unique, "
                           "snake_case)"))
    p = getattr(s, "params", None)
    if not isinstance(p, dict):
        errors.append(_err("contract", "set `self.params = {...}` with every parameter: the "
                           "trial ledger identifies a trial by name + params"))
    else:
        try:
            json.dumps(p)
        except TypeError as e:
            errors.append(_err("contract", f"self.params must be JSON-serializable: {e}"))
        for k, v in (params or {}).items():
            if p.get(k) != v:
                errors.append(_err("contract", f"param {k}={v!r} was passed but self.params "
                                   f"has {p.get(k)!r}"))
    try:
        warm = int(s.warmup())
        if warm < 0 or warm != s.warmup():
            raise ValueError
    except Exception:
        errors.append(_err("contract", "warmup() must return a non-negative int"))
        warm = 0
    stats["name"], stats["params"], stats["warmup"] = name, p, warm
    if errors:
        return {"errors": errors, "warnings": warnings, "stats": stats}

    T = max(500, warm + 260)
    liq = LiquidityModel()
    markets = ((101, "realistic", 0.002, 0.001), (202, "hostile", 0.06, 0.03))
    active = 0
    for k, (seed, kind, halts, no_open) in enumerate(markets):
        panel = build_panel(random_long_frame(seed, n_assets=50, n_sessions=T,
                                              halt_rate=halts, missing_open_rate=no_open))
        tag = f"synthetic market {k + 1} ({kind}: {T} sessions x 50 assets)"
        t0 = time.perf_counter()
        try:
            sched = build_schedule(fresh(), panel)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            err = _err("runtime", f"{tag}: {msg}", **_frames_in(path, e))
            h = _hint(msg)
            if h:
                err["hint"] = h
            errors.append(err)
            break
        dt = time.perf_counter() - t0
        calls = max(T - 1 - warm, 1)
        stats[f"market_{k + 1}"] = {
            "ms_per_decision": round(1e3 * dt / calls, 3),
            "rebalances": int(sched.has.sum()),
            "limit_entries": int(np.isfinite(sched.entry_limit).sum()),
            "stop_or_tp_exits": int(np.isfinite(sched.exit_price).sum() + sched.gap_exit.sum()),
        }
        for key, n in sched.notes.items():
            if n:
                warnings.append(_err("orders", f"{tag}: {n} {key.replace('_', ' ')} "
                                     "(they trigger at the next open)"))
        again = build_schedule(fresh(), panel).arrays()
        for nm, a in sched.arrays().items():
            b = again[nm]
            same = (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b
            if not same.all():
                errors.append(_err("determinism", f"{tag}: two fresh instances produced "
                                   f"different {nm}; remove randomness or seed it with a "
                                   "constant"))
                break
        try:
            check_truncation_invariance(lambda p: fresh(), panel, cuts=3)
        except LookaheadError as e:
            errors.append(_err("lookahead", f"{tag}: {e}"))
        except Exception as e:
            errors.append(_err("runtime", f"{tag} (truncated data): {type(e).__name__}: {e}",
                               **_frames_in(path, e)))
        if errors:
            break
        tables = exec_tables(panel, CostModel(commission_bps=1.0), liq)
        engines = ALL_ENGINES if k == 0 else ("reference",)
        try:
            res = {e: run_engine(e, panel, sched, tables, 1e6) for e in engines}
        except Exception as e:
            errors.append(_err("execution", f"{tag}: an engine failed: {type(e).__name__}: "
                               f"{e}"))
            break
        if len(res) > 1:
            rep = reconcile(res)
            if not rep.ok:
                errors.append(_err("execution", f"{tag}: engines disagree (a Concordance bug "
                                   f"- report it):\n{rep.summary()}"))
        f = res["reference"].fills
        stats[f"market_{k + 1}"]["fills"] = int(len(f))
        active += int(sched.has.any())
        if stats[f"market_{k + 1}"]["ms_per_decision"] > 50:
            warnings.append(_err("speed", f"{tag}: {stats[f'market_{k + 1}']['ms_per_decision']}"
                                 " ms per decision on 50 assets; real data has ~1,300 assets "
                                 "and ~6,500 sessions. Vectorize with numpy."))

    if not errors and not active:
        warnings.append(_err("activity", "the strategy never set target weights on either "
                             "synthetic market, so its orders were not exercised. Fine if "
                             "its signal is rare; check with --real."))
    if not errors and real_window:
        from concordance.data.crsp import load_sp500_panel

        t0 = time.perf_counter()
        try:
            panel = load_sp500_panel(real_window[0], real_window[1], warmup_sessions=warm)
            sched = build_schedule(fresh(), panel)
            tables = exec_tables(panel, CostModel(commission_bps=1.0), liq)
            r = run_engine("reference", panel, sched, tables, 1e6)
            stats["real"] = {"window": list(real_window), "sessions": int(panel.shape[0]),
                             "securities": int(panel.shape[1]), "fills": int(len(r.fills)),
                             "rebalances": int(sched.has.sum()),
                             "final_equity": float(r.equity[-1]),
                             "seconds": round(time.perf_counter() - t0, 1)}
            if not len(r.fills):
                warnings.append(_err("activity", "no trades on the real window"))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            err = _err("real_data", msg, **_frames_in(path, e))
            h = _hint(msg)
            if h:
                err["hint"] = h
            errors.append(err)
    return {"errors": errors, "warnings": warnings, "stats": stats}


# ---------------------------------------------------------------------------------- entry


def validate_file(path: str | Path, params: dict | None = None,
                  real_window: tuple[str, str] | None = None, timeout: float = 300.0) -> dict:
    path = Path(path).resolve()
    rep = {"ok": False, "file": str(path), "sha256": None, "errors": [], "warnings": [],
           "stats": {}}
    if not path.is_file():
        rep["errors"].append(_err("load", f"no such file: {path}"))
        return rep
    src = path.read_text()
    rep["sha256"] = hashlib.sha256(src.encode()).hexdigest()
    errors, warnings = static_check(src)
    rep["errors"] += errors
    rep["warnings"] += warnings
    if errors:
        return rep
    arg = json.dumps({"path": str(path), "params": params, "real": real_window})
    try:
        cp = subprocess.run([sys.executable, "-I", "-m", "concordance.authoring.validate",
                             "--child", arg], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        rep["errors"].append(_err("timeout", f"validation took longer than {timeout:.0f}s: "
                                  "the strategy is too slow or loops forever"))
        return rep
    try:
        child = json.loads(cp.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        tail = (cp.stderr or cp.stdout).strip().splitlines()[-15:]
        rep["errors"].append(_err("crash", "validator child process crashed:\n"
                                  + "\n".join(tail)))
        return rep
    rep["errors"] += child["errors"]
    rep["warnings"] += child["warnings"]
    rep["stats"] = child["stats"]
    rep["ok"] = not rep["errors"]
    return rep


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        import warnings as _w

        _w.filterwarnings("ignore")
        a = json.loads(sys.argv[2])
        out = _child(a["path"], a["params"], tuple(a["real"]) if a["real"] else None)
        print(json.dumps(out, default=str))
    else:
        raise SystemExit("usage: concordance validate STRATEGY.py")
