"""Overfitting diagnostics for parameter searches.

- `pbo_cscv`: Probability of Backtest Overfitting via combinatorially symmetric
  cross-validation (Bailey, Borwein, López de Prado, Zhu 2017).
- `cpcv_select`: choose the best configuration on purged/embargoed training folds and
  collect its out-of-sample returns on every test path (skfolio CombinatorialPurgedCV).
Replaces QuantMega's walkforward.py, which never re-fit parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from skfolio.model_selection import CombinatorialPurgedCV


def _sharpe(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = x.mean(axis=0) / sd
    return np.where(sd > 0, s, -np.inf)


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray
    n_combinations: int


def pbo_cscv(returns: np.ndarray, n_blocks: int = 16) -> PBOResult:
    """`returns` is (T, K): one column per configuration tried. Returns PBO in [0, 1]."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("need a (T, K) matrix with K >= 2 configurations")
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    blocks = np.array_split(np.arange(r.shape[0]), n_blocks)
    logits = []
    for train_ids in combinations(range(n_blocks), n_blocks // 2):
        test_ids = [b for b in range(n_blocks) if b not in train_ids]
        tr = np.concatenate([blocks[b] for b in train_ids])
        te = np.concatenate([blocks[b] for b in test_ids])
        best = int(np.argmax(_sharpe(r[tr])))
        oos = _sharpe(r[te])
        rank = (np.sum(oos < oos[best]) + 0.5 * (np.sum(oos == oos[best]) - 1) + 1)
        w = rank / (r.shape[1] + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return PBOResult(float(np.mean(logits <= 0)), logits, len(logits))


@dataclass(frozen=True)
class CPCVResult:
    chosen: list[int]  # configuration chosen on each split's training folds
    oos_paths: np.ndarray  # (T_path, n_paths) out-of-sample returns of the chosen configs
    path_sharpes: np.ndarray


def cpcv_select(returns: np.ndarray, n_folds: int = 10, n_test_folds: int = 2,
                purged_size: int = 5, embargo_size: int = 5) -> CPCVResult:
    """Pick the best-Sharpe configuration per training split, evaluate it out of sample.

    `purged_size` / `embargo_size` should be at least the strategy's holding horizon so
    labels that straddle a train/test boundary don't leak.
    """
    r = np.asarray(returns, dtype=float)
    cv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test_folds,
                               purged_size=purged_size, embargo_size=embargo_size)
    chosen = []
    path_ids = cv.get_path_ids()
    n_paths = cv.n_test_paths
    pieces: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {p: [] for p in range(n_paths)}
    for s, (train, tests) in enumerate(cv.split(r)):
        best = int(np.argmax(_sharpe(r[train])))
        chosen.append(best)
        for k, te in enumerate(tests):
            pieces[int(path_ids[s][k])].append((te, r[te, best]))
    paths = []
    for p in range(n_paths):
        segs = sorted(pieces[p], key=lambda x: x[0][0])
        paths.append(np.concatenate([v for _, v in segs]))
    L = min(len(p) for p in paths)
    oos = np.column_stack([p[:L] for p in paths])
    return CPCVResult(chosen, oos, _sharpe(oos))
