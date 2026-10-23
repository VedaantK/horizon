"""Run-scoped disk arrays and bounded column work; no global NumPy monkeypatching."""

from __future__ import annotations

import itertools
import tempfile
from contextvars import ContextVar
from pathlib import Path

import numpy as np

STORE = ContextVar("concordance_array_store", default=None)


class ArrayStore:
    def __init__(self, directory, batch_assets=256, bar_sessions=20):
        self.directory = Path(directory)
        self.batch_assets = batch_assets
        self.bar_sessions = bar_sessions
        self.counter = itertools.count()
        self.bytes = 0
        self.files = 0

    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="arrays-", dir=self.directory)
        self.token = STORE.set(self)
        return self

    def __exit__(self, *exc):
        STORE.reset(self.token)
        self.temp.cleanup()


def allocate(shape, dtype=float, fill=None, order="C"):
    store = STORE.get()
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if store is not None and size >= 1024 * 1024:
        path = Path(store.temp.name) / f"{next(store.counter)}.bin"
        out = np.memmap(path, mode="w+", dtype=dtype, shape=shape, order=order)
        store.bytes += size
        store.files += 1
    else:
        out = np.empty(shape, dtype=dtype, order=order)
    if fill is not None:
        out[...] = fill
    return out


def stored(array):
    if STORE.get() is None:
        return array
    out = allocate(array.shape, array.dtype)
    # Row slices keep disk-backed arrays in sequential order.
    for i in range(0, len(array), 128):
        out[i:i + 128] = array[i:i + 128]
    return out


def blocks(n):
    width = STORE.get().batch_assets if STORE.get() is not None else n
    for j in range(0, n, width):
        yield slice(j, min(j + width, n))
