"""EVICT-KEEP (z30y12, 30.09. 23:35:32-23:35:50): the clock evict
(``HiCacheFile._arena_evict_to_disk``) emptied the 112-slot MAMBA arena in one
round and took the six D-parked requests' anchors with it.

Metal: P PP0 ``ARENA-EVICT n=11 want=256`` 23:35:32 (callers floor ``want`` at
256); 3 s later ``ARENA-REF-CENSUS arena-78446592.bin slots=112 complete=0``,
D's MAMBA ``park_kept=6/12`` -> ``0/12``. At the wake all six resumes read
their KV back but no anchor (``#904 match-census refusers=MambaComponent:
absent``), the X gate priced each whole (W31) and 381k tokens went to P. The
second collapse (D TP0-2 ``ARENA-EVICT want=256`` 23:42:10, complete 110 -> 8)
has the same shape. The claim path (``_evict_for_claim``) had passed over the
#243/#248 keep for a long time; the clock evict never did.

Red on df1f81516c (every slot of the small arena goes, the parked ones first),
green with the fix (at most an eighth of the arena per round, kept pages passed
over).
"""
from __future__ import annotations

import os
import shutil
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from sglang.srt.weg2 import handoff_pending as hp  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weg2"))
from test_weg2_park_l3_248 import P, SB, SLOTS, _hicache, _kv_pool, _publish, _Store  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")

PARKED = "weg2-35-112"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_QUEUE_REFS", "1")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path / "arena"))
    monkeypatch.setenv("SGLANG_WEG2_HANDOFF", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    (tmp_path / "l3").mkdir()
    arena = ShmArena(str(tmp_path / "arena-mamba.bin"), SB, SLOTS)
    store = _Store(str(tmp_path / "l3"))
    be = _hicache(store)
    del be._arena_evict_to_disk            # the real clock evict, not the stub
    be.arena_secure_to_disk = lambda a, cands: {"on_disk": 0, "written": len(cands), "lost": 0}
    yield types.SimpleNamespace(arena=arena, store=store, be=be)
    arena.close()


def _complete(arena, stems):
    return [st for (_s, st) in arena.find_slots(stems)].count(2)


def test_the_clock_evict_keeps_the_parked_anchor_and_takes_one_eighth(env):
    """A full small arena: a D park (3 pages, kept by order, unreferenced) and
    5 other unreferenced pages; the caller asks for 256 like the metal."""
    pool = _kv_pool(env.arena, env.store)
    chain = ["aA", "aB", "aC"]
    parked_stems = pool._stems(chain)
    _publish(env.arena, parked_stems)                     # the park's pages first: the oldest
    others = [f"other{i}.sfx" for i in range(SLOTS - len(chain))]
    _publish(env.arena, others)
    assert hp.mark_park(PARKED, chain, P)
    env.be.register_keep_pool(env.arena, pool)
    env.be._arena_evict_to_disk(env.arena, max(256, 1))
    assert _complete(env.arena, parked_stems) == 3, "the parked anchor left L2 (W31 at the wake)"
    assert _complete(env.arena, others) >= len(others) - max(1, SLOTS // 8), \
        "one round emptied the arena (z30y12: 112 -> 0)"


def test_an_unregistered_arena_still_evicts_within_the_cap(env):
    """No pool bound (a width nobody keeps): the clock evict works as before,
    only capped -- it frees something, never the whole arena."""
    _publish(env.arena, [f"p{i}.sfx" for i in range(SLOTS)])
    freed_before = _complete(env.arena, [f"p{i}.sfx" for i in range(SLOTS)])
    env.be._arena_evict_to_disk(env.arena, 256)
    left = _complete(env.arena, [f"p{i}.sfx" for i in range(SLOTS)])
    assert freed_before == SLOTS and 0 < SLOTS - left <= max(1, SLOTS // 8)
