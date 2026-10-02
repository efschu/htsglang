# SPDX-License-Identifier: Apache-2.0
"""L15-FLIPCOST-5: ArenaMHAHostPool.slot_gens answers by sort+searchsorted
over the COMPLETE census -- identical to the old dict lookup (-1 when not
COMPLETE), unsorted census and duplicates in the request included."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool


def test_identical_to_the_dict_lookup():
    rng = np.random.default_rng(3)
    cs = rng.permutation(5000)[:2000].astype(np.int64)
    cg = rng.integers(0, 9, size=cs.size).astype(np.int64)
    fake = SimpleNamespace(arena=SimpleNamespace(complete_census=lambda: (cs, cg, None, None)))
    want = list(rng.integers(0, 5000, size=700)) + [int(cs[0]), int(cs[0])]
    d = {int(s): int(g) for s, g in zip(cs.tolist(), cg.tolist())}
    assert ArenaMHAHostPool.slot_gens(fake, want) == [d.get(int(s), -1) for s in want]


def test_edges():
    empty = SimpleNamespace(arena=SimpleNamespace(
        complete_census=lambda: (np.zeros(0, np.int64), np.zeros(0, np.int64), None, None)))
    assert ArenaMHAHostPool.slot_gens(empty, [1, 2]) == [-1, -1]
    assert ArenaMHAHostPool.slot_gens(empty, []) == []
    assert ArenaMHAHostPool.slot_gens(SimpleNamespace(arena=None), [7]) == [-1]
