"""Have an AI coding tool (Claude Code or Codex CLI) implement a StrategySpec as a strategy
file, then check it with the validator ourselves and send back any errors.

    concordance author spec.json --tool claude --out candidates/my_idea_claude
    concordance author spec.json --tool codex  --out candidates/my_idea_codex

Each tool runs headless in its own workspace holding only the spec, the contract and an
example; it may edit files there and run the validator, nothing else. The tool's own claim
that validation passed is never trusted: the driver re-runs `validate` after every round and
the round counts only if that passes.

Billing: the tools run on their subscription logins. ANTHROPIC_API_KEY / OPENAI_API_KEY are
removed from the child environment so a stray key cannot switch them to paid API billing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from concordance.authoring.validate import validate_file

REPO = Path(__file__).resolve().parents[3]
PROMPTS = REPO / "prompts"
TOOLS = ("claude", "codex")
API_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY")
REQUIRED_SPEC = ("name", "hypothesis", "signal", "entry", "exit", "sizing", "parameters")


def _validate_cmd() -> str:
    exe = Path(sys.executable).parent / "concordance"
    return f"{exe} validate strategy.py"


def _env(ws: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in API_KEYS}
    env["NUMBA_CACHE_DIR"] = str(ws / ".numba_cache")  # writable inside a sandbox
    return env


def _call_claude(prompt: str, ws: Path, schema: str, model: str | None, timeout: float):
    vc = _validate_cmd()
    cmd = ["claude", "-p", "--output-format", "json", "--json-schema", schema,
           "--permission-mode", "acceptEdits", "--no-session-persistence",
           "--allowedTools", "Read", "Write", "Edit", "Glob", "Grep", f"Bash({vc}:*)",
           "--disallowedTools", "WebFetch", "WebSearch", "Agent"]
    if model:
        cmd += ["--model", model]
    cp = subprocess.run(cmd, input=prompt, cwd=ws, env=_env(ws), capture_output=True,
                        text=True, timeout=timeout)
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": (cp.stderr or cp.stdout)[-2000:]}
    out = data.get("structured_output")
    if out is None:
        try:
            out = json.loads(data.get("result") or "null")
        except json.JSONDecodeError:
            out = None
    return {"ok": not data.get("is_error") and cp.returncode == 0, "answer": out,
            "session_id": data.get("session_id"), "turns": data.get("num_turns"),
            "error": data.get("result") if data.get("is_error") else None}


def _call_codex(prompt: str, ws: Path, schema_path: Path, model: str | None,
                timeout: float):
    last = ws / ".codex_last_message.json"
    last.unlink(missing_ok=True)
    cmd = ["codex", "exec", "--cd", str(ws), "--sandbox", "workspace-write",
           "--skip-git-repo-check", "--ephemeral", "--output-schema", str(schema_path),
           "-o", str(last), "-"]
    if model:
        cmd[2:2] = ["--model", model]
    cp = subprocess.run(cmd, input=prompt, cwd=ws, env=_env(ws), capture_output=True,
                        text=True, timeout=timeout)
    answer = None
    if last.exists():
        try:
            answer = json.loads(last.read_text())
        except json.JSONDecodeError:
            answer = None
    return {"ok": cp.returncode == 0, "answer": answer,
            "error": None if cp.returncode == 0 else (cp.stderr or cp.stdout)[-2000:]}


def check_spec(spec: dict) -> list[str]:
    problems = [f"missing '{k}'" for k in REQUIRED_SPEC if k not in spec]
    if spec.get("open_questions"):
        problems.append("spec has open_questions; answer them before implementing: "
                        + "; ".join(spec["open_questions"]))
    return problems


def author(spec_path: str | Path, tool: str, workspace: str | Path, model: str | None = None,
           rounds: int = 3, timeout: float = 1200.0, allow_open_questions: bool = False) -> dict:
    if tool not in TOOLS:
        raise ValueError(f"tool must be one of {TOOLS}")
    if shutil.which(tool) is None:
        raise RuntimeError(f"'{tool}' is not on PATH")
    spec = json.loads(Path(spec_path).read_text())
    problems = check_spec(spec)
    if not allow_open_questions:
        if problems:
            return {"ok": False, "stage": "spec", "problems": problems}
    elif any(not p.startswith("spec has open_questions") for p in problems):
        return {"ok": False, "stage": "spec", "problems": problems}

    ws = Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "spec.json").write_text(json.dumps(spec, indent=2))
    shutil.copyfile(REPO / "docs" / "STRATEGY_CONTRACT.md", ws / "STRATEGY_CONTRACT.md")
    shutil.copyfile(REPO / "examples" / "swing_pullback.py", ws / "example_strategy.py")
    schema_path = PROMPTS / "author_result.schema.json"
    schema = schema_path.read_text()
    vc = _validate_cmd()
    prompt = (PROMPTS / "author.md").read_text().replace("{validate_cmd}", vc)

    attempts = []
    report = None
    for k in range(rounds):
        t0 = time.time()
        try:
            if tool == "claude":
                res = _call_claude(prompt, ws, schema, model, timeout)
            else:
                res = _call_codex(prompt, ws, schema_path, model, timeout)
        except subprocess.TimeoutExpired:
            res = {"ok": False, "error": f"{tool} timed out after {timeout:.0f}s"}
        report = validate_file(ws / "strategy.py")
        attempts.append({"round": k + 1, "tool_ok": res.get("ok"),
                         "tool_error": res.get("error"), "answer": res.get("answer"),
                         "seconds": round(time.time() - t0, 1),
                         "validation_ok": report["ok"], "errors": report["errors"]})
        if report["ok"]:
            break
        if res.get("error") and not (ws / "strategy.py").exists():
            break  # the tool itself failed (login, usage limit): don't burn more rounds
        prompt = (PROMPTS / "repair.md").read_text().replace("{validate_cmd}", vc).replace(
            "{report}", json.dumps({"errors": report["errors"],
                                    "warnings": report["warnings"]}, indent=2))
    result = {"ok": bool(report and report["ok"]), "tool": tool, "model": model,
              "workspace": str(ws), "strategy_file": str(ws / "strategy.py"),
              "sha256": report.get("sha256") if report else None,
              "rounds": len(attempts), "attempts": attempts,
              "validation": report}
    (ws / "author_result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def formalize(idea: str, tool: str, out_path: str | Path, model: str | None = None,
              timeout: float = 600.0) -> dict:
    """Plain-language idea -> StrategySpec JSON (no tools: the model only answers)."""
    if tool not in TOOLS:
        raise ValueError(f"tool must be one of {TOOLS}")
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path = PROMPTS / "strategy_spec.schema.json"
    prompt = (PROMPTS / "formalize.md").read_text().replace("{idea}", idea.strip())
    ws = out_path.parent
    if tool == "claude":
        cmd = ["claude", "-p", "--output-format", "json", "--json-schema",
               schema_path.read_text(), "--no-session-persistence",
               "--disallowedTools", "Bash", "Edit", "Write", "WebFetch", "WebSearch", "Agent"]
        if model:
            cmd += ["--model", model]
        cp = subprocess.run(cmd, input=prompt, cwd=ws, env=_env(ws), capture_output=True,
                            text=True, timeout=timeout)
        try:
            data = json.loads(cp.stdout)
            spec = data.get("structured_output") or json.loads(data.get("result") or "null")
        except json.JSONDecodeError:
            return {"ok": False, "error": (cp.stderr or cp.stdout)[-2000:]}
    else:
        last = ws / ".codex_spec.json"
        cmd = ["codex", "exec", "--cd", str(ws), "--sandbox", "read-only",
               "--skip-git-repo-check", "--ephemeral", "--output-schema", str(schema_path),
               "-o", str(last), "-"]
        if model:
            cmd[2:2] = ["--model", model]
        cp = subprocess.run(cmd, input=prompt, cwd=ws, env=_env(ws), capture_output=True,
                            text=True, timeout=timeout)
        try:
            spec = json.loads(last.read_text())
        except (OSError, json.JSONDecodeError):
            return {"ok": False, "error": (cp.stderr or cp.stdout)[-2000:]}
    if not isinstance(spec, dict):
        return {"ok": False, "error": "no structured answer"}
    out_path.write_text(json.dumps(spec, indent=2))
    return {"ok": True, "spec": str(out_path), "open_questions": spec.get("open_questions", []),
            "inferred_defaults": spec.get("inferred_defaults", []),
            "unsupported": spec.get("unsupported", [])}
