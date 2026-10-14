"""h18port 1014: SELF-EVICT (port e937898ced) against the h16h metal of weg2-58-558.

h16h 853534a9ff, P.log ``..._1014_180031``, 19:21:06-11, weg2-58-558 (1158 pages, 1155 in the store):
* PP2 ``#248e ORDERED-EVICT n=1 site=l3fill need=56 took=49 victims=[weg2-58-555:rank=1,pages=49,lowest_page=807]``
  -- the fill of THIS read made its room from the kept pages of the sibling turn weg2-58-555, which share the
  session's prefix: the very pages 807.. this read had found in L2 at its find step.
* then ``#1028B FETCH CAP kv=807 claimed=784`` and ``SA SHORT-READ ANCHOR landed=807 of 1155 pages anchor=784`` on
  ALL THREE stages (the P arena is one), ``#257 ... claim cut to 50176``, 16640 tokens recomputed.

So the read ended at page 807 because its own fill evicted it there, not at a missing anchor: the anchor 784 is only
the deepest state INSIDE the 807 landed pages. SELF-EVICT puts every stem of the operation on the fill's keep list
(``fill_protect_scope``): the room comes from pages that are not this read's, the read keeps its L2 pages.

Driven with the real C arena, the real fill and the real ordered clock evict (the #248e fixture).
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache import hicache_storage as hs  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402

from test_weg2_handoff_evict_order_0930 import (  # noqa: E402,F401
    PARK, SB, _read_order, _states, _to_l3, env, pytestmark,
)

#: the read's tail only the L3 holds (on the metal the 217 l3fill pages)
FILL = ["n0", "n1"]


def _read_58_558(env, protect: bool):
    """The read = the sibling's kept chain (found in L2: env.h, the hand-off) + two pages only in L3. The sibling
    is not in the read order, so the clock ranks its pages first -- the metal's victim."""
    _to_l3(env.store, FILL, 0x70)
    _read_order([PARK])
    op = types.SimpleNamespace(hash_value=list(env.h) + FILL, _weg2_protect_stems=list(env.h) + FILL)
    with envs.SGLANG_WEG2_L3FILL_PROTECT_READ.override(protect):
        with hs.fill_protect_scope(env.be, op):
            out = HiCacheFile.arena_fill_from_disk(env.be, env.kv, FILL, SB, prefix=True)
    return out


def test_the_fill_never_evicts_the_reads_own_l2_pages(env):
    out = _read_58_558(env, protect=True)
    assert all(o is not None for o in out), out
    assert _states(env.kv, env.h) == [2] * 8, "the read's own L2 pages stay: the walk goes past them"
    assert _states(env.kv, env.a[2:4]) != [2, 2], "the room came from pages that are not this read's"


def test_mutant_protect_off_is_the_metal_cut(env):
    """PROTECT_READ off (the metal boot): the fill takes the read's own pages from the tail (h7, h6) -- its walk
    then ends there, on every stage that shares the arena (landed=807 of 1155)."""
    out = _read_58_558(env, protect=False)
    assert all(o is not None for o in out), out
    assert _states(env.kv, env.h[6:]) != [2, 2], "the read's own pages made the room"
    assert _states(env.kv, env.h[:6]) == [2] * 6
