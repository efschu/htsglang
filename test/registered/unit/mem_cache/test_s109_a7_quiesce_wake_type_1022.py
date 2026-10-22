"""S109-A7-TYPE (W65 22.10., boot 1022_003738): "S109-A7 QUIESCE-WAKE quiesce failed: TypeError('only length-1
arrays can be converted to Python scalars')" on D TP0 00:40:15 (x3) and P PP0 00:39:02 (x2) -- the first quiesce of the
boot, arena still empty (PUBLISH-SWEEP unbacked=0). ``unsecured_keys_lo(..., keys=False)`` promises the COUNT; its two
early exits (empty census, failed census) returned the empty key ARRAY, and ``queue_unsecured_all`` did ``int(...)`` on
it inside a logging ``except``. RED on a4f4b8fec1, GREEN: an int on every path, no failure line.
(The conftest seam replaces ``unsecured_keys_lo`` by the writer's form; this file calls the real one explicitly.)"""
from __future__ import annotations

import logging
import os
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from flliper.srt.mem_cache import l3_secured as l3s  # noqa: E402

_REAL = l3s.unsecured_keys_lo   # bound at import, before the per-test conftest seam


class _EmptyArena:
    def complete_census(self):
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.uint64)


class _BrokenArena:
    def complete_census(self):
        raise RuntimeError("census unavailable")


@pytest.mark.parametrize("arena", [_EmptyArena(), _BrokenArena()], ids=["empty", "census_failed"])
def test_count_form_is_an_int_on_the_early_exits(arena):
    n = _REAL(arena, None, keys=False)
    assert isinstance(n, int) and n == 0
    k = _REAL(arena, None)
    assert isinstance(k, np.ndarray) and k.shape == (0,) and k.dtype == np.uint64


def test_quiesce_wake_on_an_empty_arena_names_no_failure(monkeypatch, caplog):
    monkeypatch.setattr(l3s, "unsecured_keys_lo", _REAL)
    backend = types.SimpleNamespace(_l3wb_thread=threading.current_thread())
    monkeypatch.setattr(l3s, "_REG", [(_EmptyArena(), backend), (_EmptyArena(), backend)])
    with caplog.at_level(logging.WARNING):
        assert l3s.queue_unsecured_all("quiesce") == 0
    assert not any("QUIESCE-WAKE" in m and "failed" in m for m in caplog.messages), caplog.messages
