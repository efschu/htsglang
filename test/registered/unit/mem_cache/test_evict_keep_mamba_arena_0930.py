"""EVICT-KEEP (Port von 27B 3f7378473f; z30y12, 30.09. 23:35:32-23:35:50): the
clock evict (``HiCacheFile._arena_evict_to_disk``) emptied the 112-slot MAMBA
arena in one round and took the six D-parked requests' anchors with it.

Metal (27B z30y12): P PP0 ``ARENA-EVICT n=11 want=256`` 23:35:32 (callers floor
``want`` at 256); 3 s later ``ARENA-REF-CENSUS arena-78446592.bin slots=112
complete=0``, D's MAMBA ``park_kept=6/12`` -> ``0/12``. At the wake all six
resumes read their KV back but no anchor (``#904 match-census
refusers=MambaComponent: absent``), the X gate priced each whole (W31) and 381k
tokens went to P. NF y5l: P PP0 ``ARENA-EVICT n=1..3 want=256`` -- the class is
there.

The NF form (#248e) already finds the mamba arena's keep (``bind_arena`` in
``ArenaMambaPoolHost.bind``, role ``anchor``: the chain's last
``ANCHOR_TAIL_KEYS``) -- but with ``need`` defaulting to ``want`` = 256, stage
(i) takes every unkept page and stage (ii) then spends the kept anchors in hold
order, so the keep did not hold. RED on 1cb9ceefcd (8 of 8 slots evicted, the
anchors among them), green with the cap (at most an eighth of the arena per
round; an explicit ``need`` lifts ``want`` back to it).
"""
from __future__ import annotations

import os
import shutil
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from flliper.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from flliper.srt.pdflip import handoff_pending as hp  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pdflip"))
from test_pdflip_park_l3_248 import P, SB, SLOTS, _hicache, _publish, _Store  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")

PARKED = "pdflip-35-112"


class _MStore(_Store):
    """The backend surface the mamba pool's ``_stems`` touches."""

    def _log_key(self, pool_name, h):
        return f"{pool_name}-{h}"


def _mamba_pool(arena, store):
    """The MAMBA arena host pool as ``bind`` leaves it (role ``anchor``)."""
    pool = object.__new__(ArenaMambaPoolHost)
    pool.arena = arena
    pool.arena_slots = SLOTS
    pool._backend = store
    hp.bind_arena(pool, arena)  # what ArenaMambaPoolHost.bind does (#248e)
    return pool


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_QUEUE_REFS", "1")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path / "arena"))
    monkeypatch.setenv("FLLIPER_PDFLIP_HANDOFF", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    (tmp_path / "l3").mkdir()
    arena = ShmArena(str(tmp_path / "arena-mamba.bin"), SB, SLOTS)
    store = _MStore(str(tmp_path / "l3"))
    be = _hicache(store)
    del be._arena_evict_to_disk            # the real clock evict, not the stub
    be.arena_secure_to_disk = lambda a, cands, writer="claim_room": {
        "on_disk": 0, "written": len(cands), "lost": 0}
    yield types.SimpleNamespace(arena=arena, store=store, be=be)
    hp.note_read_order([])
    getattr(hp, "_ARENA_POOLS", {}).clear()
    arena.close()


def _complete(arena, stems):
    return [st for (_s, st) in arena.find_slots(stems)].count(2)


def test_the_clock_evict_keeps_the_parked_anchor_and_takes_one_eighth(env):
    """A full small MAMBA arena: a D park (chain of 3, its 2 tail anchors kept
    by order, unreferenced) and 5 other unreferenced pages; the caller asks
    for 256 like the metal (no ``need``: the arena-write / L3-promote path)."""
    pool = _mamba_pool(env.arena, env.store)
    chain = ["aA", "aB", "aC"]
    stems = pool._stems(chain)
    anchors = stems[-hp.ANCHOR_TAIL_KEYS:]
    _publish(env.arena, stems)                            # the park's pages first: the oldest
    others = [f"other{i}.sfx" for i in range(SLOTS - len(chain))]
    _publish(env.arena, others)
    assert hp.mark_park(PARKED, chain, P)
    assert len(hp.keep_for(pool)) == len(anchors), "the mamba arena's keep is the anchor tail"
    env.be._arena_evict_to_disk(env.arena, max(256, 1))
    assert _complete(env.arena, anchors) == len(anchors), "the parked anchor left L2 (W31 at the wake)"
    left = _complete(env.arena, stems[:-len(anchors)] + others)
    assert left >= SLOTS - len(anchors) - max(1, SLOTS // 8), \
        "one round emptied the arena (z30y12: 112 -> 0)"


def test_an_unregistered_arena_still_evicts_within_the_cap(env):
    """No pool bound (a width nobody keeps): the clock evict works as before,
    only capped -- it frees something, never the whole arena."""
    pages = [f"p{i}.sfx" for i in range(SLOTS)]
    _publish(env.arena, pages)
    assert _complete(env.arena, pages) == SLOTS
    env.be._arena_evict_to_disk(env.arena, 256)
    assert 0 < SLOTS - _complete(env.arena, pages) <= max(1, SLOTS // 8)


def test_an_explicit_need_lifts_the_capped_want(env):
    """The L3 fill passes its real need (``need=len(full)``): the cap never
    leaves it short -- stage (i) takes ``need`` unkept pages, no kept anchor."""
    pool = _mamba_pool(env.arena, env.store)
    chain = ["bA", "bB", "bC"]
    stems = pool._stems(chain)
    anchors = stems[-hp.ANCHOR_TAIL_KEYS:]
    _publish(env.arena, stems)
    others = [f"o{i}.sfx" for i in range(SLOTS - len(chain))]
    _publish(env.arena, others)
    assert hp.mark_park(PARKED, chain, P)
    need = 3                                              # > SLOTS // 8 == 1
    HiCacheFile._arena_evict_to_disk(env.be, env.arena, max(256, need), need=need)
    assert _complete(env.arena, anchors) == len(anchors), "a kept anchor went while unkept pages were left"
    assert _complete(env.arena, stems[:-len(anchors)] + others) == SLOTS - len(anchors) - need
