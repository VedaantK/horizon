"""Bridge to the isolated hftbacktest environment (hft/.venv).

hftbacktest pins numpy<2.3, which conflicts with vectorbt, so it runs in its own venv.
Data crosses the boundary as files (.npz event arrays / parquet); code runs via subprocess.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HFT_DIR = Path(__file__).resolve().parents[3] / "hft"
HFT_PYTHON = HFT_DIR / ".venv" / "bin" / "python"


def run_hft(module: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run `python -m module args...` inside hft/.venv with hft/ on the path."""
    if not HFT_PYTHON.exists():
        raise FileNotFoundError(f"{HFT_PYTHON} missing: run `cd hft && uv sync`")
    return subprocess.run([str(HFT_PYTHON), "-m", module, *args], cwd=HFT_DIR,
                          capture_output=True, text=True, check=check)


def self_test() -> str:
    """Run the hand-computed hftbacktest queue/fill tests; raises on failure."""
    out = run_hft("pytest", "tests", "-rA", check=False)
    if out.returncode != 0:
        raise RuntimeError("hftbacktest self-test failed:\n" + out.stdout + out.stderr)
    return out.stdout
