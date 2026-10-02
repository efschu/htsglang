# SPDX-License-Identifier: Apache-2.0
"""L15-HOLDMAP: every import a P take/deposit makes from D's hold is released
together after the copies -- a lingering mapping or fd pins D's hold VRAM on
the shared card after D frees it."""

from __future__ import annotations

import inspect
import os

import pytest

from sglang.srt.weg2 import l15_share_admit
from sglang.srt.weg2.l15_hold_share import HoldMapper, L15ShareError


class _T:
    def __init__(self, ptr):
        self._p = ptr

    def data_ptr(self):
        return self._p


def _mapper(order):
    maps = []

    def map_fn(fd, size):
        maps.append((fd, size))
        return _T(0x1000 * (len(maps)))

    m = HoldMapper(0, map_fn=map_fn,
                   unmap_fn=lambda va, size: order.append(("unmap", va, size)),
                   sync_fn=lambda: order.append(("sync",)))
    return m, maps


def test_one_mapping_per_fd_then_sync_before_every_unmap_and_fds_closed():
    order = []
    m, maps = _mapper(order)
    r, w = os.pipe()
    m.own_fds([r, w])
    a = m(r, 4096)
    assert m(r, 4096) is a                 # KV and anchor pieces share it
    m(w, 8192)
    assert maps == [(r, 4096), (w, 8192)] and m.mapped == 2
    assert m.close() == (2, 2)
    assert order[0] == ("sync",)           # copies finished before any unmap
    assert sorted(o[2] for o in order[1:]) == [4096, 8192]
    for f in (r, w):
        with pytest.raises(OSError):
            os.fstat(f)
    assert m.close() == (0, 0)             # idempotent


def test_same_fd_other_size_is_refused():
    m, _ = _mapper([])
    m(5, 4096)
    with pytest.raises(L15ShareError):
        m(5, 8192)


def test_fetch_owns_the_received_fds_even_if_nothing_is_mapped():
    m, _ = _mapper([])
    r, w = os.pipe()
    d, fds = m.fetch(lambda rank: ({"rank": rank}, [r, w]), 1)
    assert d == {"rank": 1}
    assert m.close() == (0, 2)


def test_hot_admission_releases_its_mapper_in_finally():
    src = inspect.getsource(l15_share_admit.admit_for_sched)
    assert "mapper = HoldMapper(dev)" in src
    assert "map_extent=mapper" in src and "mapper.fetch(" in src
    tail = src[src.index("mapper = HoldMapper(dev)"):]
    assert "finally:\n        mapper.close()" in tail
    assert "map_hold_extent" not in src
