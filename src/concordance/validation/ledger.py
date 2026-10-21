"""Trial ledger: every evaluated configuration is recorded, so the deflated Sharpe always
counts the trials actually made (QuantMega's idea; storage is new and file-based).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from concordance.validation.dsr import effective_independent_trials


def spec_hash(name: str, params: dict) -> str:
    """Stable hash of a strategy name + parameters (order-independent)."""
    blob = json.dumps({"name": name, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class TrialLedger:
    def __init__(self, root: Path | None = None, program: str = "default"):
        lake = Path(os.environ.get("CONCORDANCE_LAKE", Path.home() / "concordance" / "lake"))
        self.dir = Path(root or lake / "ledger") / program
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / "trials.jsonl"

    def record(
        self,
        name: str,
        params: dict,
        returns: pd.Series,
        note: str = "",
        context: dict | None = None,
    ) -> str:
        # Source, execution and data identity matter even when names/parameters match.
        identity = {
            "name": name,
            "params": params,
            "context": context or {},
            "returns": hashlib.sha256(
                pd.util.hash_pandas_object(returns, index=True).values.tobytes()
            ).hexdigest(),
        }
        h = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        with (self.dir / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.dir / f"{h}.parquet"
            if not path.exists():
                tmp = self.dir / f"{h}.tmp"
                returns.rename("ret").to_frame().to_parquet(tmp)
                os.replace(tmp, path)
        with self.index.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(
                json.dumps(
                    {
                        "hash": h,
                        "name": name,
                        "params": params,
                        "note": note,
                        "context": context or {},
                        "schema": 2,
                        "at": datetime.now(UTC).isoformat(),
                        "start": str(returns.index[0].date()),
                        "end": str(returns.index[-1].date()),
                    },
                    default=str,
                )
                + "\n"
            )
            f.flush()
            os.fsync(f.fileno())
        return h

    def trials(self) -> pd.DataFrame:
        if not self.index.exists():
            return pd.DataFrame(columns=["hash", "name", "params"])
        with self.index.open() as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            rows = [json.loads(line) for line in stream if line.strip()]
        return pd.DataFrame(rows).drop_duplicates("hash", keep="last")

    def returns_matrix(self) -> pd.DataFrame:
        cols = {h: pd.read_parquet(self.dir / f"{h}.parquet")["ret"] for h in self.trials()["hash"]}
        return pd.DataFrame(cols)

    def n_trials(self) -> int:
        return len(self.trials())

    def n_effective(self) -> float:
        m = self.returns_matrix()
        if not m.shape[1]:
            return 0.0
        # Disjoint windows cannot establish correlations; never reset their search to one.
        if len(m.dropna()) < 60:
            return float(m.shape[1])
        return effective_independent_trials(m.to_numpy())

    def sharpe_variance(self) -> float:
        m = self.returns_matrix().to_numpy()
        s = m.mean(0) / m.std(0, ddof=1)
        return float(np.var(s[np.isfinite(s)], ddof=1)) if m.shape[1] > 1 else np.nan
