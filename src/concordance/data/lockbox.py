"""Lockbox guard: data on/after the lockbox date stays sealed until a logged one-shot run.

Idea carried over from QuantMega's `_enforce_lockbox`; the one-shot ledger is new.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


class LockboxViolation(PermissionError):
    pass


def lockbox_date() -> pd.Timestamp | None:
    v = os.environ.get("CONCORDANCE_LOCKBOX")
    return pd.Timestamp(v) if v else None


def enforce(end, unlock_reason: str | None = None, ledger: Path | None = None,
            lockbox: pd.Timestamp | None = None) -> None:
    """Raise if `end` reaches the lockbox, unless an unlock reason is given (and logged)."""
    lb = lockbox if lockbox is not None else lockbox_date()
    if lb is None or pd.Timestamp(end) < lb:
        return
    if not unlock_reason:
        raise LockboxViolation(
            f"requested data through {pd.Timestamp(end).date()} reaches the lockbox "
            f"({lb.date()}). Pass unlock_reason=... for a registered one-shot evaluation.")
    ledger = ledger or Path(os.environ.get("CONCORDANCE_LAKE", ".")) / "lockbox_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps({"at": datetime.now(UTC).isoformat(), "end": str(end),
                            "lockbox": str(lb.date()), "reason": unlock_reason}) + "\n")
