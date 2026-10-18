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

import pytest
import torch

from sglang.srt.mem_cache import hicache_storage as hs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile

from test_weg2_handoff_evict_order_0930 import (  # noqa: F401
    HANDOFF, PARK, SB, _read_order, _states, _to_l3, env, pytestmark,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Registry hygiene (review S24 A3b): empty before and after every test."""
    hs._FILL_ACTIVE.clear()
    yield
    hs._FILL_ACTIVE.clear()


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


def test_protection_ends_with_the_scope_regression_anchor(env):
    """(green on the base too: a regression anchor, no red value.) No stale protection: once the sibling's scope is closed, its chain is an ordinary kept chain again."""
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


# --- review S24 (1018): probes P1-P3, taken over -------------------------------------------------------------------

def _scope_in_thread(op):
    ready, done = threading.Event(), threading.Event()

    def run():
        with hs.fill_protect_scope(None, op):
            ready.set()
            done.wait(10)

    t = threading.Thread(target=run)
    t.start()
    assert ready.wait(10)
    return t, done


def test_p1_claim_room_does_not_take_an_in_flight_reads_chain(env):
    """Auflage 2: the scheduler thread's claim room (``_evict_for_claim``, every stage) honours the fill protection."""
    _read_order([PARK, HANDOFF])        # HANDOFF read last: its tail goes first
    op_a = types.SimpleNamespace(hash_value=list(env.h), _weg2_protect_stems=list(env.h))
    t, done = _scope_in_thread(op_a)
    try:
        ids = env.pool.alloc_write(["new0", "new1"])     # the scheduler thread's claim (D retain / P chunk)
    finally:
        done.set()
        t.join(10)
    assert ids is not None
    assert _states(env.kv, env.h) == [2] * 8, "the in-flight read's chain stays whole (claim stage ii took its tail)"


def _clock_in_the_complete_ref_window(env, monkeypatch):
    """Another rank's clock (its process registry knows nothing of this read) runs right after the complete."""
    real = env.kv.complete_slots
    fired = []

    def other_process_clock():
        saved = dict(hs._FILL_ACTIVE)
        hs._FILL_ACTIVE.clear()
        try:
            HiCacheFile._arena_evict_to_disk(env.be, env.kv, 2, need=2)
        finally:
            hs._FILL_ACTIVE.update(saved)

    def complete_then_clock(*a, **k):
        cs = real(*a, **k)
        if not fired:
            fired.append(1)
            t = threading.Thread(target=other_process_clock)
            t.start()
            t.join(10)
        return cs

    monkeypatch.setattr(env.kv, "complete_slots", complete_then_clock)
    return fired


def test_p2_other_rank_process_clock_in_the_complete_ref_window(env, monkeypatch):
    """Auflage 1: reference BEFORE complete -- the slot is never COMPLETE and unreferenced."""
    _read_order([HANDOFF, PARK])        # PARK read last: its tail (a5, a4 = this fill) goes first
    fired = _clock_in_the_complete_ref_window(env, monkeypatch)
    op_b = types.SimpleNamespace(hash_value=list(env.a[4:]), _weg2_protect_stems=list(env.a[4:]))
    with hs.fill_protect_scope(env.be, op_b):
        out = HiCacheFile.arena_fill_from_disk(env.be, env.kv, env.a[4:], SB, prefix=True, room_wait=False,
                                               take_ref=True)
    assert fired
    assert all(o is not None for o in out), f"the landed pages were taken before their reference: {list(out)}"
    assert env.kv.slot_refs([int(o) for o in out]) == [1, 1], "exactly one reference per landed slot (adopted, not doubled)"


def test_p3_a_scope_left_by_an_exception_protects_nothing():
    op = types.SimpleNamespace(hash_value=["x0", "x1"], _weg2_protect_stems=["x0", "x1"])
    with pytest.raises(RuntimeError):
        with hs.fill_protect_scope(None, op):
            assert set(hs.fill_protected_stems()) == {"x0", "x1"}
            raise RuntimeError("fill died")
    assert hs.fill_protected_stems() == [] and hs._FILL_ACTIVE == {}


def test_the_reference_before_complete_is_given_back_when_the_complete_is_refused(env, monkeypatch):
    """The pre-reference of a slot whose complete is refused (reaped / recycled) is returned: no leaked reference."""
    _read_order([HANDOFF, PARK])
    real = env.kv.complete_slots
    monkeypatch.setattr(env.kv, "complete_slots", lambda slots, gens, ext: [6] * len(slots))
    out = HiCacheFile.arena_fill_from_disk(env.be, env.kv, env.a[4:], SB, prefix=True, room_wait=False,
                                           take_ref=True)
    assert all(o is None for o in out)
    refs = env.kv.slot_refs(list(range(12)))
    assert max(refs) == 0, f"a leaked pre-reference: {refs}"
    monkeypatch.setattr(env.kv, "complete_slots", real)


# --- the same window in the draft and mamba fills (Auflage 1, 'mitnehmen') ------------------------------------------

def test_draft_fill_takes_its_reference_before_the_complete(env, monkeypatch):
    from sglang.srt.managers.cache_controller import HiCacheController as C

    comp = "draft-x"
    keys = ["d0", "d1"]
    stems = [f"{k}.{comp}" for k in keys]
    _to_l3(env.store, stems, 0x60)
    env.be._suffix_for_key = lambda k: ("",)
    _read_order([HANDOFF, PARK])
    dpool = types.SimpleNamespace(arena_read=True, staging_rows=4, arena=env.kv, _page_bytes=SB,
                                  ensure_bound=lambda *a, **k: True,
                                  resolve_draft_rows=lambda rows, slots: setattr(dpool, "slots", list(slots)))
    ctl = types.SimpleNamespace(mem_pool_host_draft=dpool, storage_backend=env.be, page_size=1,
                                _draft_component_name=lambda: comp, _draft_l3_hits=0, _draft_l3_misses=0)
    fired = _clock_in_the_complete_ref_window(env, monkeypatch)
    flags = C._draft_page_get_generic(ctl, keys, torch.tensor([4, 5], dtype=torch.int64))
    assert fired
    assert flags == [True, True], f"a landed draft page was taken before its reference: {flags}"
    assert env.kv.slot_refs([int(x) for x in dpool.slots]) == [1, 1], "one reference per landed draft page"


def test_mamba_fill_takes_its_reference_before_the_complete(env, monkeypatch):
    from sglang.srt.mem_cache.pool_host import arena_mamba_pool as amp

    monkeypatch.setenv(amp.ENV_ANCHOR_BYPASS, "0")
    stem = "m0.mamba"
    _to_l3(env.store, [stem], 0x70)
    env.be._get_suffixed_key = lambda k: k + ".mamba"
    _read_order([HANDOFF, PARK])
    p = object.__new__(amp.ArenaMambaPoolHost)
    p.staging_rows = 4
    p.arena = env.kv
    p._page_bytes = SB
    p.ensure_bound = lambda *a, **k: True
    fired = _clock_in_the_complete_ref_window(env, monkeypatch)
    hi = torch.tensor([99], dtype=torch.int64)
    out = p.arena_resolve_reads(env.be, hi, ["m0"])
    assert fired
    assert out == [True], f"the landed anchor was taken before its reference: {out}"
    slot = int(hi[0]) - 4
    assert env.kv.slot_refs([slot]) == [1], "one reference on the anchor blob"
