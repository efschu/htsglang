"""S24 1018: SIBLING-EVICT (h16v5 1af79968a7, D 03:31:55-56, weg2-2-54 / weg2-0-25) against the real C arena.

D.log: the wake issues two hold reads at once (``WAKE-READ issued=2 ['weg2-2-54', 'weg2-0-25']``, WAKE-PARALLEL io
threads). weg2-0-25's fill found the arena full (``ARENA-EVICT want=259 need=259``) and its ordered clock evict took
``victims=[weg2-2-54:rank=4,pages=162,lowest_page=1538]`` -- the tail of the SIBLING's chain, whose fill (184 stems,
slots 5566.. claimed by TP2, joined by TP0/TP1) had completed them and not yet referenced them. weg2-2-54 landed 1538
of 1700 on all three ranks: TP2/TP1 ``cause=fill_none``, TP0 ``cause=arena_chain_recheck`` (one event, three names).
``fill_protect_scope`` was thread-local: every read protected only its own stems.
"""
from __future__ import annotations

import threading
import types

from sglang.srt.mem_cache import hicache_storage as hs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile

from test_weg2_handoff_evict_order_0930 import (  # noqa: F401
    HANDOFF, PARK, SB, _read_order, _states, _to_l3, env, pytestmark,
)


def _sibling_in_flight():
    """Read A = the hand-off chain (env.h), its scope open in ANOTHER thread for the run of B's fill."""
    ready, done = threading.Event(), threading.Event()

    def run(op):
        with hs.fill_protect_scope(None, op):
            ready.set()
            done.wait(10)

    return ready, done, run


def _b_fill(env, protect_a: bool):
    _read_order([PARK, HANDOFF])        # HANDOFF read last: the clock ranks its tail first
    op_a = types.SimpleNamespace(hash_value=list(env.h), _weg2_protect_stems=list(env.h))
    op_b = types.SimpleNamespace(hash_value=list(env.a[4:]), _weg2_protect_stems=list(env.a[4:]))
    ready, done, run = _sibling_in_flight()
    t = threading.Thread(target=run, args=(op_a,))
    t.start()
    assert ready.wait(10)
    try:
        out_box = []

        def fill():
            with hs.fill_protect_scope(env.be, op_b):
                out_box.append(HiCacheFile.arena_fill_from_disk(env.be, env.kv, env.a[4:], SB, prefix=True,
                                                                room_wait=False))
        tb = threading.Thread(target=fill)
        tb.start()
        tb.join(30)
    finally:
        done.set()
        t.join(10)
    return out_box[0]


def test_a_sibling_reads_chain_is_never_the_victim_of_another_reads_room_clock(env):
    out = _b_fill(env, True)
    assert _states(env.kv, env.h) == [2] * 8, "the in-flight sibling's chain stays whole in L2 (metal: tail 162 gone)"
    assert all(o is not None for o in out), f"B still got its room, from pages that are nobody's read: {out}"


def test_protection_ends_with_the_scope(env):
    """No stale protection: once the sibling's scope is closed, its chain is an ordinary kept chain again."""
    _b_fill(env, True)
    assert hs.fill_protected_stems() == []


def test_the_refused_reference_of_a_landed_slot_is_named_not_fill_none():
    """The refused ref (evicted between COMPLETE and REF) is recorded on the answer list; the reason line names it."""
    from sglang.srt.weg2 import read_depth as rd

    class _Arena:
        def ref_slots(self, slots, d):
            return 0 if d > 0 else 1

    out = hs._RefOnLand(_Arena(), 3)
    out[1] = 7
    assert out[1] is None and out.refused_at == [1]
    hs._read_depth_fill_reasons(["a", "b", "c"], {"a", "b", "c"}, out, [], [], [], {"todo": 1}, None)
    assert rd.fill_reasons_take() == {1: rd.FILL_EVICTED_BEFORE_REF}
