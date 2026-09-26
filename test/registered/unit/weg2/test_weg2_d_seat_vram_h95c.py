"""H95c (Nutzer 26.09.): D's seat posts are PAGES only for the phase's seats.

  "soll ja dynamisches bs geben, 1,6gb experten cache kostet es nur bei
  tatsaechlich 6 sitzen. bei weniger sitzen kostet es weniger experten cache"

H95 B decided n per flip but kept the posts of --d-bs = 6 seats allocated (38
GDN slots x 56.2 MiB on TP0, the expert bank at the cap's rows). H95c keeps
every VIRTUAL range (the captured graphs) and maps per phase only the slots
of n seats; the pages the other seats leave back extra expert-LRU rows.

WHAT MUST HOLD.
(1) The arithmetic: the GDN slots of a phase are mapped per layer (outward
    rounded, the allocation's slack kept), the expert bank as one prefix; the
    extra rows k(n) never make the phase map more than the cap. NF x177 form
    (36 GDN layers x 39 slots x 1.5 MiB, 48 MoE layers w13 1.5625 + w2 0.78125
    MiB/row, granule 2 MiB): k = 13/10/8/5/1/0 for n = 1..6.
(2) The pool tables carry the seat rows as DEVICE VALUES: an OFF row is never
    a victim and never routed, switching k is a table write (no recapture),
    the capture bound uses the capacity with every seat occupied, and every
    host-side reader (sync, seed, bijection) leaves OFF rows OFF.
(3) The slot allocator hands out only slots 1..L(n); refuses the limit while a
    slot above it is live; a flush keeps it; the pool's reset zeroes only the
    mapped slots.
(4) The runtime: the first request of a wake without a count resets to the
    cap form, the request with n sets the Mamba plan (paused) and grows the
    expert bank in place (mapped), the rest of a wake repeats nothing; every
    rank derives the same slot limit and admission cap; a batch wider than n
    is refused (W-SEAT); one line per wake.
(5) The saver's span map (tms_csrc patch 3), built against a mock driver:
    resume maps only the plan, ``now`` keeps mapped extents (and their bytes)
    and maps the rest, pause/free give every extent back, tag bytes are the
    physical bytes, a failed map rolls back; without a plan: the stock path.
(6) The planner shows each n's own rows (n=1: TP0 133 = 120 + 13) and the
    launcher writes the switch and the rows' reservation into --env-d.
(7) Off (default): nothing of the above runs.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import textwrap
import types
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.layers.moe import expert_pool_device as ep  # noqa: E402
from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402

G = 2 << 20
MIB = 1 << 20
NF_MODEL = (
    "/spinning/llm_stuff/club-3090/models-cache/"
    "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
)


def _nf_geometry(rows_boot=120, x=15):
    slot = dsv.SlotTensorGeom("gdn_temporal", 36, 39, 48 * 128 * 128 * 2,
                              dsv.align_up(36 * 39 * 48 * 128 * 128 * 2, G))
    rows = []
    for _layer in range(48):
        for rb in (160 * 2560 * 4, 40 * 5120 * 4):  # Marlin w13 / w2 packed
            rows.append(dsv.RowTensorGeom("t", rows_boot, rows_boot + x, rb,
                                          dsv.align_up((rows_boot + x) * rb, G)))
    return slot, rows


# ---- (1) the arithmetic -----------------------------------------------------

def test_slot_spans_keep_every_layers_live_slots_and_the_slack():
    g = dsv.SlotTensorGeom("t", layers=3, slots=5, slot_bytes=3 * MIB, alloc_bytes=46 * MIB)
    # layer blocks at 0 / 15 / 30 MiB, tensor 45 MiB, allocation 46 MiB
    spans = dsv.slot_spans(g, 2, G)
    assert spans == ((0, 6 * MIB), (14 * MIB, 22 * MIB), (30 * MIB, 36 * MIB), (44 * MIB, 46 * MIB))
    for layer in range(3):  # every live byte of slots 0..1 is inside a span
        lo, hi = layer * 15 * MIB, layer * 15 * MIB + 6 * MIB
        assert any(a <= lo and hi <= b for a, b in spans)
    assert dsv.slot_spans(g, 5, G) == ((0, 46 * MIB),)
    assert dsv.slot_spans(g, 9, G) == ((0, 46 * MIB),)
    assert all(a % G == 0 and b % G == 0 for a, b in spans)


def test_row_spans_are_one_granule_aligned_prefix():
    g = dsv.RowTensorGeom("w", rows_boot=10, rows_max=14, row_bytes=3 * MIB // 2,
                          alloc_bytes=dsv.align_up(14 * 3 * MIB // 2, G))
    assert dsv.row_spans(g, 10, G) == ((0, 16 * MIB),)  # 15 MiB -> 16
    assert dsv.row_spans(g, 12, G) == ((0, 18 * MIB),)
    assert dsv.row_spans(g, 14, G) == ((0, g.alloc_bytes),)


def test_phase_slot_limit_is_the_pool_a_boot_for_n_seats_would_have():
    assert [dsv.phase_slot_limit(38, n, 6) for n in range(1, 7)] == [7, 13, 19, 25, 32, 38]
    # a pool sized otherwise (another ratio / a ceiling fit): proportional, up
    assert [dsv.phase_slot_limit(23, n, 6) for n in range(1, 7)] == [4, 8, 12, 16, 20, 23]
    assert dsv.phase_slot_limit(38, 9, 6) == 38


def test_nf_x177_extra_rows_per_phase_never_map_more_than_the_cap():
    slot, rows = _nf_geometry()
    tab = dsv.seat_vram_rows([slot], rows, cap=6, pool_size=38, extra_max=15, granule=G)
    assert [r.extra_rows for r in tab] == [13, 10, 8, 5, 1, 0]
    assert [r.slot_limit for r in tab] == [7, 13, 19, 25, 32, 38]
    for r in tab:
        assert r.mapped <= r.cap_mapped
    assert tab[-1].mapped == tab[-1].cap_mapped
    # n=1: 1620 MiB of GDN pages go (of 31 x 1.5 x 36 = 1674 ideal: inward rounding)
    assert (tab[0].mamba_cap - tab[0].mamba_mapped) == 1620 * MIB
    # the reservation bounds k; without expert tensors nothing moves
    assert dsv.seat_vram_rows([slot], rows, cap=6, pool_size=38, extra_max=4)[0].extra_rows == 4
    assert dsv.seat_vram_rows([slot], [], cap=6, pool_size=38, extra_max=15)[0].extra_rows == 0


def test_extra_rows_env_per_rank():
    assert dsv.extra_rows_for_rank(0, "14,0,0") == 14
    assert dsv.extra_rows_for_rank(1, "14,0,0") == 0
    assert dsv.extra_rows_for_rank(5, "14,0,0") == 0
    assert dsv.extra_rows_for_rank(2, "7") == 7
    assert dsv.extra_rows_for_rank(0, "") == 0
    assert dsv.extra_rows_for_rank(0, "x,1") == 0


# ---- (2) the pool tables ------------------------------------------------------

def _seat_tables(E=24, R=3, C=7, S=2, X=4):
    hot = {e: e for e in range(R)}
    host_row = [(-1 if e in hot else e) for e in range(E)]
    t = ep.allocate_pool_tables("cpu", E, R + C + X, R, S, hot, host_row, seat_rows=X)
    return t, hot, host_row


def test_seat_form_layout_and_capacity():
    t, _hot, _hr = _seat_tables()
    # [0,3) residents | [3,5) staging | [5,10) LRU | [10,14) seat rows OFF
    assert (t.pool_rows, t.lru_start, ep.resident_count(t)) == (14, 5, 3)
    assert t.staging_rows.tolist() == [3, 4]
    assert t.row_key.tolist()[10:] == [ep.SEAT_OFF_KEY] * 4
    assert t.row_use.tolist()[10:] == [ep.ROW_USE_NEVER] * 4
    assert ep.seat_off_range(t) == (10, 14)
    assert ep.pool_row_capacity(t) == 7  # C: every seat occupied
    ep.set_seat_rows_on(t, 3)
    assert ep.pool_row_capacity(t) == 10
    assert t.row_key.tolist()[10:] == [-1, -1, -1, ep.SEAT_OFF_KEY]
    assert ep.seat_off_range(t) == (13, 14)
    with pytest.raises(ValueError):
        ep.set_seat_rows_on(t, 5)


def test_an_off_row_is_never_a_victim_and_never_routed():
    t, _hot, _hr = _seat_tables()
    b = ep.allocate_step_buffers("cpu", t.num_experts, 64)
    seen_rows = set()
    for step in range(12):
        ids = torch.tensor([3 + (step * 5 + i) % 21 for i in range(5)], dtype=torch.int32)
        pairs, _ = ep.step_reference(t, ids, b, waves=2, spill=True)
        seen_rows |= {d for _s, d in pairs}
        seen_rows |= {int(r) for r in b.routes.tolist() if r >= 0}
    assert seen_rows and max(seen_rows) < 10  # rows 10..13 are OFF
    assert int(t.error[0]) == 0
    # ON: the seat rows become ordinary LRU rows
    ep.set_seat_rows_on(t, 4)
    for step in range(12):
        ids = torch.tensor([3 + (step * 7 + i) % 21 for i in range(6)], dtype=torch.int32)
        pairs, _ = ep.step_reference(t, ids, b, waves=2)
        seen_rows |= {d for _s, d in pairs}
    assert {10, 11, 12, 13} & seen_rows


def test_turning_rows_off_drops_the_experts_they_held():
    t, _hot, _hr = _seat_tables()
    ep.set_seat_rows_on(t, 4)
    pairs = ep.seed_lru_rows(t, list(range(3, 12)))
    held = {r: s for s, r in pairs}  # host row == expert id here
    assert max(held) >= 10  # the seed reached the seat rows
    ep.set_seat_rows_on(t, 1)
    hot = t.hot_phys.tolist()
    for r, e in held.items():
        if r >= 11:
            assert hot[e] == -1 and int(t.row_key[r]) == ep.SEAT_OFF_KEY
        else:
            assert hot[e] == r
    assert ep.bijection_breaks(t) == 0


def test_reinit_sync_seed_keep_off_rows_off():
    t, hot, host_row = _seat_tables()
    ep.set_seat_rows_on(t, 2, device_write=False)  # recorded for the rearm
    ep.reinit_pool_tables(t, hot, host_row)
    assert t.row_key.tolist()[10:] == [-1, -1, ep.SEAT_OFF_KEY, ep.SEAT_OFF_KEY]
    assert t.staging_rows.tolist() == [3, 4]
    # an eager pass wrote rows 5, 6 and (wrongly) an OFF row: the OFF row stays OFF
    rep = ep.sync_tables(t, {5: 7, 6: 8, 12: 9}, keep_unwritten=False)
    assert rep.owned == 2
    assert t.row_key.tolist()[12:] == [ep.SEAT_OFF_KEY, ep.SEAT_OFF_KEY]
    assert int(t.hot_phys[9]) == -1
    free = [r for r, k in enumerate(t.row_key.tolist()) if k < 0 and r >= t.lru_start]
    assert 12 not in free and 13 not in free
    assert all(r < 12 for _s, r in ep.seed_lru_rows(t, list(range(3, 24))))
    assert ep.bijection_breaks(t) == 0


def test_capture_bound_uses_every_seat_occupied_and_the_true_residents():
    t, _hot, _hr = _seat_tables(E=40, R=3, C=7, S=2, X=4)
    b = ep.allocate_step_buffers("cpu", 40, 64)
    ids = torch.arange(3, 3 + 15, dtype=torch.int32)
    # min(15, E - R = 37) = 15 > 2 x C (7) at k = 0
    with pytest.raises(ValueError):
        ep.step_reference(t, ids, b, waves=2)
    ep.step_reference(t, ids[:14], b, waves=2, spill=True)  # 14 <= 2 x 7


def test_without_seat_rows_the_tables_are_those_before_h95c():
    hot = {0: 0, 1: 1}
    hr = [-1, -1] + list(range(8))
    t = ep.allocate_pool_tables("cpu", 10, 8, 2, 2, hot, hr)
    assert (t.pool_rows, t.lru_start, t.resident_rows, t.seat_rows) == (6, 2, None, 0)
    assert t.staging_rows.tolist() == [6, 7]
    assert ep.pool_row_capacity(t) == 6 and ep.seat_off_range(t) is None
    assert ep.resident_count(t) == 2
    ep.reinit_pool_tables(t, hot, hr)
    assert t.staging_rows.tolist() == [6, 7]


# ---- (3) the slot allocator and the pool's reset -----------------------------

def test_the_allocator_hands_out_only_the_phases_slots():
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

    a = MambaSlotAllocator(size=38, device="cpu")
    assert a.set_phase_limit(7)
    assert a.available_size() == 7 and a.phase_limit == 7
    got = a.alloc(7)
    assert sorted(got.tolist()) == list(range(1, 8)) and a.alloc(1) is None
    a.free(got[:3])
    assert a.available_size() == 3
    a.clear()  # a flush inside the phase keeps the limit
    assert a.available_size() == 7 and a.free_slots.max().item() == 7
    assert a.set_phase_limit(None) and a.available_size() == 38
    s = a.alloc(20)
    assert not a.set_phase_limit(7)  # slot 20 is live
    assert a.phase_limit is None and a.available_size() == 18
    a.free(s)
    assert a.set_phase_limit(13) and a.available_size() == 13
    assert a.set_phase_limit(25) and a.available_size() == 25
    assert a.set_phase_limit(38) and a.phase_limit is None and a.available_size() == 38


def test_the_pools_reset_zeroes_only_mapped_slots():
    from sglang.srt.mem_cache.memory_pool import MambaPool

    temporal = torch.ones(2, 6, 3)
    fake = types.SimpleNamespace(
        mamba_cache=MambaPool.State(conv=[torch.ones(2, 6, 4)], temporal=temporal),
        SpeculativeState=MambaPool.SpeculativeState, device="cpu",
        replayssm_write_pos=None, _sync_device=lambda: None, _weg2_seat_keep=3)
    MambaPool.reset_state(fake)
    assert temporal[:, :3].abs().sum() == 0 and temporal[:, 3:].eq(1).all()
    fake._weg2_seat_keep = None
    MambaPool.reset_state(fake)
    assert temporal.abs().sum() == 0


# ---- (4) the runtime ------------------------------------------------------------

class FakeTms:
    """The span map's semantics in Python (the C++ is tested in (5))."""

    def __init__(self):
        self.allocs = {}
        self.calls = []

    available = True

    def add(self, t, active=True):
        size = dsv.align_up(t.numel() * t.element_size(), 4096)
        self.allocs[t.data_ptr()] = {"size": size, "plan": None, "mapped": [(0, size)] if active else [],
                                     "active": active}

    def info(self, ptr):
        a = self.allocs.get(int(ptr))
        if a is None:
            return None
        planned = sum(h - l for l, h in (a["plan"] or [(0, a["size"])]))
        return dsv.AllocInfo(a["size"], sum(h - l for l, h in a["mapped"]), planned, a["active"])

    def set_spans(self, ptr, spans, *, now):
        a = self.allocs[int(ptr)]
        self.calls.append((int(ptr), tuple(spans), now))
        a["plan"] = list(spans)
        if now and a["active"]:
            a["mapped"] = list(spans)
        return 0

    def pause(self, ptr):
        self.allocs[int(ptr)].update(active=False, mapped=[])

    def resume(self, ptr):
        a = self.allocs[int(ptr)]
        a.update(active=True, mapped=list(a["plan"] or [(0, a["size"])]))


class FakeCache:
    def __init__(self, layer_id, R, C, S, X, row_elems):
        from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

        self.layer = types.SimpleNamespace(layer_id=layer_id)
        self.seat_rows = X
        self.planner = types.SimpleNamespace(buffer_size=R + C)
        self._resident = {
            "w13": torch.zeros(R + C + X, row_elems, dtype=torch.int32),
            "w13_scales": torch.zeros(R + C + X, 2, dtype=torch.int32),
        }
        hot = {e: e for e in range(R)}
        self._pool_tables = ep.allocate_pool_tables(
            "cpu", 40, R + C + X, R, S, hot, [(-1 if e in hot else e) for e in range(40)],
            seat_rows=X)
        self._pool_ready = True
        self.set_seat_rows_on = types.MethodType(MoEExpertOffloadCache.set_seat_rows_on, self)


def _rank(fake_tms, *, with_pages=True, layers=2, slots=11, X=4):
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

    temporal = torch.zeros(3, slots + 1, 1024, dtype=torch.int32)  # 4 KiB per slot
    pool = types.SimpleNamespace(size=slots, mamba_cache=types.SimpleNamespace(temporal=temporal))
    alloc = MambaSlotAllocator(size=slots, device="cpu")
    caches = [FakeCache(i, 3, 7, 2, X if with_pages else 0, 1024) for i in range(layers)]
    if with_pages:
        fake_tms.add(temporal, active=True)
        for c in caches:
            for buf in c._resident.values():
                fake_tms.add(buf, active=True)
    modules = [types.SimpleNamespace(_expert_offload=c) for c in caches]
    model = types.SimpleNamespace(modules=lambda: iter(modules))
    rtp = types.SimpleNamespace(mamba_pool=pool, mamba_allocator=alloc)
    sched = types.SimpleNamespace(
        server_args=types.SimpleNamespace(max_running_requests=2 if slots < 20 else 6),
        tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(
            req_to_token_pool=rtp, model=model)))
    return sched, temporal, pool, alloc, caches


def _armed(fake_tms):
    from sglang.srt.environ import envs

    stack = [mock.patch.dict(os.environ, {dsv.GROUP_ENV: "D"}),
             envs.SGLANG_OPT_WEG2_D_SEAT_VRAM.override(True),
             mock.patch.object(dsv, "_TMS", fake_tms),
             mock.patch.object(dsv, "granule_for", lambda _dev: 4096)]
    return stack


class _Ctx:
    def __init__(self, stack):
        self.stack = stack

    def __enter__(self):
        for s in self.stack:
            s.__enter__()
        return self

    def __exit__(self, *a):
        for s in reversed(self.stack):
            s.__exit__(*a)


def _seats(n):
    from sglang.srt.weg2 import d_seats

    return d_seats.phase_seats(n, 0, cap=2)


def test_weights_leg_first_then_kv_leg_with_n_grows_the_bank_in_place():
    tms = FakeTms()
    with _Ctx(_armed(tms)):
        sched, temporal, pool, alloc, caches = _rank(tms)
        w = types.SimpleNamespace(epoch="e1")
        # D slept: every page is paused
        for p in list(tms.allocs):
            tms.pause(p)
        st = dsv.on_wake(sched, w, None)  # weights leg: no count -> cap form
        assert st.n == 2 and st.slot_limit is None and alloc.phase_limit is None
        ctl = sched._weg2_d_seat_vram
        assert ctl.applied.extra_rows == 0
        for p in list(tms.allocs):  # the saver resumes the weights (and later kv)
            if p != temporal.data_ptr():
                tms.resume(p)
        assert all(c._pool_tables.seat_on == 0 for c in caches)
        kv = types.SimpleNamespace(epoch="e1", handoff_n=1, parked_n=0)
        st = dsv.on_wake(sched, kv, _seats(1))
        assert st.n == 1 and st.slot_limit == dsv.phase_slot_limit(11, 1, 2) == 6
        assert alloc.available_size() == 6
        k = ctl.applied.extra_rows
        assert k > 0
        # Mamba (paused): the plan of 1 seat, set BEFORE its resume
        plan = tms.allocs[temporal.data_ptr()]["plan"]
        assert plan == list(dsv.slot_spans(ctl.slot_tensors[0].geom, 7, 4096))
        assert pool._weg2_seat_keep == 7
        # experts (mapped): grown in place (now=True) and ON in the device tables
        grow = [c for c in tms.calls if c[2]]
        assert grow and all(c[1] == ((0, dsv.align_up((10 + k) * 4096, 4096)),) for c in grow)
        for c in caches:
            assert c._pool_tables.seat_on == k
            assert c._pool_tables.row_key.tolist()[10:10 + k] == [-1] * k
        # the wake's later request changes nothing
        n_calls = len(tms.calls)
        assert dsv.on_wake(sched, types.SimpleNamespace(epoch="e1"), None) is None
        assert len(tms.calls) == n_calls
        # the admission cap and the guard
        assert dsv.admission_cap(sched) == 1
        dsv.guard(sched, types.SimpleNamespace(reqs=[1]))
        with pytest.raises(dsv.Weg2DSeatOverrun, match="W-SEAT"):
            dsv.guard(sched, types.SimpleNamespace(reqs=[1, 2]))
        # the next wake starts in the cap form again
        for p in list(tms.allocs):
            tms.pause(p)
        dsv.on_wake(sched, types.SimpleNamespace(epoch="e2"), None)
        assert alloc.phase_limit is None and dsv.admission_cap(sched) is None
        assert tms.allocs[temporal.data_ptr()]["plan"] == [(0, tms.allocs[temporal.data_ptr()]["size"])]
        assert pool._weg2_seat_keep is None
        assert all(c._pool_tables.seat_on == 0 for c in caches)


def test_kv_leg_first_sets_both_plans_before_any_resume():
    tms = FakeTms()
    with _Ctx(_armed(tms)):
        sched, temporal, pool, alloc, caches = _rank(tms)
        for p in list(tms.allocs):
            tms.pause(p)
        dsv.on_wake(sched, types.SimpleNamespace(epoch="e3", handoff_n=1), _seats(1))
        k = sched._weg2_d_seat_vram.applied.extra_rows
        assert k > 0 and not any(c[2] for c in tms.calls)  # nothing mapped live
        # the weights leg of the same wake repeats nothing
        assert dsv.on_wake(sched, types.SimpleNamespace(epoch="e3"), None) is None
        for c in caches:  # recorded for the rearm, not written
            assert c._pool_tables.seat_on == k
            assert c._pool_tables.row_key.tolist()[10:] == [ep.SEAT_OFF_KEY] * 4


def test_a_live_mamba_pool_never_shrinks():
    tms = FakeTms()
    with _Ctx(_armed(tms)):
        sched, temporal, pool, alloc, caches = _rank(tms)
        applied = dsv.on_wake(sched, types.SimpleNamespace(epoch="e4", handoff_n=1), _seats(1))
        ctl = sched._weg2_d_seat_vram
        assert ctl.applied.extra_rows == 0 and "live" in ctl.applied.note
        assert tms.allocs[temporal.data_ptr()]["plan"] is None
        assert applied.slot_limit == 6  # the replicated half still holds


def test_every_rank_sets_the_same_limit_and_cap_pages_or_not():
    tms = FakeTms()
    with _Ctx(_armed(tms)):
        s0, *_ = _rank(tms, with_pages=True)
        s1, _t1, _p1, a1, _c1 = _rank(tms, with_pages=False)
        for s in (s0, s1):
            for p in list(tms.allocs):
                if tms.allocs[p]["active"]:
                    tms.pause(p)
            st = dsv.on_wake(s, types.SimpleNamespace(epoch="e5", handoff_n=1), _seats(1))
            assert (st.n, st.slot_limit) == (1, 6)
        assert s1._weg2_d_seat_vram is False  # a worker: no pages of its own
        assert a1.available_size() == 6
        assert dsv.admission_cap(s0) == dsv.admission_cap(s1) == 1


def test_a_live_slot_above_the_limit_keeps_the_cap_form_on_every_rank():
    tms = FakeTms()
    with _Ctx(_armed(tms)):
        sched, temporal, pool, alloc, caches = _rank(tms)
        alloc.alloc(9)  # slot 9 is live
        for p in list(tms.allocs):
            tms.pause(p)
        st = dsv.on_wake(sched, types.SimpleNamespace(epoch="e6", handoff_n=1), _seats(1))
        assert st.slot_limit is None and "still in use" in st.note
        assert sched._weg2_d_seat_vram.applied.extra_rows == 0
        assert dsv.admission_cap(sched) == 1  # the seat count itself is replicated


def test_one_line_per_wake(caplog):
    import logging

    tms = FakeTms()
    with _Ctx(_armed(tms)), caplog.at_level(logging.INFO, logger=dsv.__name__):
        sched, *_ = _rank(tms)
        for p in list(tms.allocs):
            tms.pause(p)
        dsv.on_wake(sched, types.SimpleNamespace(epoch="e7", handoff_n=1), _seats(1))
    lines = [r.getMessage() for r in caplog.records if "n=1 of cap 2" in r.getMessage()]
    assert len(lines) == 1
    ln = lines[0]
    assert ln.startswith("WEG2 D-SEAT-VRAM (H95c) n=1 of cap 2 epoch=e7 mamba_slots=1..6 ")
    assert "expert_rows=+" in ln and "(vs n=2)" in ln and "mapped_mib=" in ln


def test_off_nothing_runs():
    from sglang.srt.environ import envs

    assert envs.SGLANG_OPT_WEG2_D_SEAT_VRAM.get() is False  # the default
    sched = types.SimpleNamespace(server_args=types.SimpleNamespace(max_running_requests=6))
    with mock.patch.dict(os.environ, {dsv.GROUP_ENV: "D"}):
        assert dsv.on_wake(sched, types.SimpleNamespace(epoch="x"), _seats(1)) is None
        assert dsv.admission_cap(sched) is None
        dsv.guard(sched, types.SimpleNamespace(reqs=list(range(9))))
        assert dsv.presplit_seat_rows(types.SimpleNamespace(moe_tp_rank=0)) == 0
    with envs.SGLANG_OPT_WEG2_D_SEAT_VRAM.override(True), mock.patch.dict(
            os.environ, {dsv.GROUP_ENV: "P"}):
        assert dsv.on_wake(sched, types.SimpleNamespace(epoch="x"), _seats(1)) is None
        assert dsv.presplit_seat_rows(types.SimpleNamespace(moe_tp_rank=0)) == 0


def test_the_presplit_reserves_the_ranks_rows_only_with_a_span_map():
    from sglang.srt.environ import envs

    env = {dsv.GROUP_ENV: "D", dsv.POOL_GRAPH_MODE_ENV: "pool"}
    layer0, layer1 = types.SimpleNamespace(moe_tp_rank=0), types.SimpleNamespace(moe_tp_rank=1)
    with mock.patch.dict(os.environ, env), envs.SGLANG_OPT_WEG2_D_SEAT_VRAM.override(True), \
            envs.SGLANG_WEG2_D_SEAT_EXPERT_ROWS.override("14,0,0"):
        with mock.patch.object(dsv, "_TMS", FakeTms()):
            assert dsv.presplit_seat_rows(layer0) == 14
            assert dsv.presplit_seat_rows(layer1) == 0
        stock = types.SimpleNamespace(available=False)
        with mock.patch.object(dsv, "_TMS", stock):
            assert dsv.presplit_seat_rows(layer0) == 0  # a saver without patch 3


def test_the_seat_bank_is_born_trimmed_to_its_cap_form():
    class BornTms(FakeTms):
        def info(self, ptr):
            if int(ptr) not in self.allocs:
                self.allocs[int(ptr)] = {"size": 11 * 4096, "plan": None,
                                         "mapped": [(0, 11 * 4096)], "active": True}
            return super().info(ptr)

    tms = BornTms()
    buf = dsv.seat_expert_buffer(rows=8, extra=3, tail=(1024,), dtype=torch.int32,
                                 device="cpu", spans=tms, granule=4096)
    assert tuple(buf.shape) == (11, 1024)
    (ptr, spans, now), = tms.calls
    assert ptr == buf.data_ptr() and now and spans == ((0, 8 * 4096),)
    small = dsv.seat_expert_buffer(rows=8, extra=1, tail=(16,), dtype=torch.int32,
                                   device="cpu", spans=tms, granule=4096)
    assert tuple(small.shape) == (9, 16) and len(tms.calls) == 1  # kept whole

    class NoAlloc(FakeTms):
        def info(self, ptr):
            return None

    with pytest.raises(dsv.Weg2DSeatVramRefused, match="not one saver allocation"):
        dsv.seat_expert_buffer(rows=8, extra=3, tail=(1024,), dtype=torch.int32,
                               device="cpu", spans=NoAlloc(), granule=4096)


# ---- the wiring -------------------------------------------------------------------

def test_the_scheduler_delegates_and_the_resume_calls_before_the_tags():
    from sglang.srt.managers import scheduler as sch
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    from sglang.srt.weg2 import d_park_runtime

    for name in ("weg2_d_seat_vram_wake", "_weg2_d_seat_cap", "_weg2_d_seat_guard"):
        assert callable(getattr(sch.Scheduler, name, None)), name
    for name in ("seat_vram_wake", "seat_cap", "seat_guard"):
        assert callable(getattr(d_park_runtime, name, None)), name
    src = open(sch.__file__).read()
    i = src.index("    def get_num_allocatable_reqs(self, running_bs):")
    body = src[i:i + 1200]
    assert "_seat_cap = self._weg2_d_seat_cap()" in body
    assert "limit = min(limit, _seat_cap)" in body
    j = src.index("    def _run_batch_forward(")
    assert "self._weg2_d_seat_guard(batch)" in src[j:j + 600]
    wsrc = open(wu.__file__).read()
    k = wsrc.index("    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):")
    wbody = wsrc[k:k + 5000]
    c = wbody.index('"weg2_d_seat_vram_wake", None)')
    assert wbody.index("return replay") < c < wbody.index("tags = recv_req.tags")
    assert wbody.index('"weg2_d_note_wake_seats", None)') < c


def test_the_expert_bank_carries_the_seat_rows():
    from sglang.srt.layers.moe import expert_offload as eo

    src = open(eo.__file__).read()
    assert "_seat_x = _seat_vram.presplit_seat_rows(layer)" in src
    assert "buf = _seat_vram.seat_expert_buffer(" in src
    assert "layer._weg2_seat_rows = int(_seat_x)" in src
    assert "buf_rows = buf_slots + self.seat_rows" in src
    assert "seat_rows=self.seat_rows," in src
    assert "need = pool_waves_for(int(n_ids), t.num_experts, resident_count(t), C)" in src


# ---- (5) the saver's span map, built against a mock driver --------------------

_MOCK = r'''
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <map>
#include <cstring>
#include <cstdlib>
#include <cstdint>
static std::map<CUdeviceptr, size_t> g_res;                       // VA reservations
static std::map<CUdeviceptr, std::pair<size_t, uint64_t>> g_map;   // addr -> (size, handle)
static std::map<uint64_t, size_t> g_handles;                       // live handles
static uint64_t g_next = 1; static CUdeviceptr g_va = 0x100000000ULL;
static long g_fail_after = -1;
extern "C" {
size_t mock_live_bytes() { size_t s = 0; for (auto& h : g_handles) s += h.second; return s; }
size_t mock_mapped_bytes() { size_t s = 0; for (auto& m : g_map) s += m.second.first; return s; }
int mock_is_mapped(CUdeviceptr a) { for (auto& m : g_map) if (a >= m.first && a < m.first + m.second.first) return 1; return 0; }
int mock_extents() { return (int) g_map.size(); }
void mock_fail_create_after(long n) { g_fail_after = n; }
CUresult cuMemCreate(CUmemGenericAllocationHandle* h, size_t size, const CUmemAllocationProp*, unsigned long long) {
  if (size % (2u << 20)) return CUDA_ERROR_INVALID_VALUE;
  if (g_fail_after == 0) { g_fail_after = -1; return CUDA_ERROR_OUT_OF_MEMORY; }
  if (g_fail_after > 0) --g_fail_after;
  *h = (CUmemGenericAllocationHandle) g_next; g_handles[g_next++] = size; return CUDA_SUCCESS; }
CUresult cuMemRelease(CUmemGenericAllocationHandle h) {
  if (!g_handles.count((uint64_t) h)) return CUDA_ERROR_INVALID_VALUE;
  for (auto& m : g_map) if (m.second.second == (uint64_t) h) return CUDA_ERROR_INVALID_VALUE;
  g_handles.erase((uint64_t) h); return CUDA_SUCCESS; }
CUresult cuMemAddressReserve(CUdeviceptr* p, size_t size, size_t, CUdeviceptr, unsigned long long) {
  *p = g_va; g_res[g_va] = size; g_va += size + (64u << 20); return CUDA_SUCCESS; }
CUresult cuMemAddressFree(CUdeviceptr p, size_t) { g_res.erase(p); return CUDA_SUCCESS; }
CUresult cuMemMap(CUdeviceptr a, size_t size, size_t off, CUmemGenericAllocationHandle h, unsigned long long) {
  if (off != 0 || !g_handles.count((uint64_t) h) || g_handles[(uint64_t) h] != size) return CUDA_ERROR_INVALID_VALUE;
  bool inside = false; for (auto& r : g_res) if (a >= r.first && a + size <= r.first + r.second) inside = true;
  if (!inside) return CUDA_ERROR_INVALID_VALUE;
  for (auto& m : g_map) if (a < m.first + m.second.first && m.first < a + size) return CUDA_ERROR_INVALID_VALUE;
  g_map[a] = std::make_pair(size, (uint64_t) h); return CUDA_SUCCESS; }
CUresult cuMemUnmap(CUdeviceptr a, size_t size) {
  auto it = g_map.find(a); if (it == g_map.end() || it->second.first != size) return CUDA_ERROR_INVALID_VALUE;
  g_map.erase(it); return CUDA_SUCCESS; }
CUresult cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) { return CUDA_SUCCESS; }
CUresult cuMemGetAllocationGranularity(size_t* g, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) { *g = 2u << 20; return CUDA_SUCCESS; }
CUresult cuDeviceGetAttribute(int* v, CUdevice_attribute, CUdevice) { *v = 0; return CUDA_SUCCESS; }
CUresult cuGetErrorString(CUresult, const char** s) { *s = "mock"; return CUDA_SUCCESS; }
CUresult cuGetErrorName(CUresult, const char** s) { *s = "mock"; return CUDA_SUCCESS; }
CUresult cuCtxGetDevice(CUdevice* d) { *d = 0; return CUDA_SUCCESS; }
CUresult cuDeviceGet(CUdevice* d, int) { *d = 0; return CUDA_SUCCESS; }
CUresult cuDeviceGetUuid(CUuuid* u, CUdevice) { memset(u, 0, sizeof(*u)); return CUDA_SUCCESS; }
cudaError_t cudaMallocHost(void** p, size_t n) { *p = malloc(n); return cudaSuccess; }
cudaError_t cudaFreeHost(void* p) { free(p); return cudaSuccess; }
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
cudaError_t cudaHostRegister(void*, size_t, unsigned int) { return cudaSuccess; }
cudaError_t cudaGetLastError(void) { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "mock"; }
}
'''


@pytest.fixture(scope="module")
def tms_lib(tmp_path_factory):
    gxx = shutil.which("g++")
    inc = "/usr/local/cuda/include"
    if gxx is None or not os.path.isfile(os.path.join(inc, "cuda.h")):
        pytest.skip("no g++ / CUDA headers to build the saver against a mock driver")
    import sglang.srt.weg2 as w

    src = os.path.join(os.path.dirname(w.__file__), "tms_csrc")
    out = tmp_path_factory.mktemp("tms")
    (out / "mock_cuda.cpp").write_text(textwrap.dedent(_MOCK))
    so = out / "libtms_mock.so"
    cmd = [gxx, "-std=c++17", "-shared", "-fPIC", "-DUSE_CUDA", "-DTMS_HOOK_MODE_PRELOAD",
           "-I" + inc, "-I" + src] + [os.path.join(src, f) for f in (
               "core.cpp", "entrypoint.cpp", "host_ring.cpp", "api_forwarder.cpp")] + [
           str(out / "mock_cuda.cpp"), "-o", str(so), "-Wl,-Bsymbolic", "-ldl", "-lpthread"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    lib = ctypes.CDLL(str(so))
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.tms_set_current_tag.argtypes = [ctypes.c_char_p]
    lib.tms_set_interesting_region.argtypes = [ctypes.c_bool]
    lib.tms_pause.argtypes = [ctypes.c_char_p]
    lib.tms_resume_rc.argtypes = [ctypes.c_char_p]
    lib.tms_tag_bytes.argtypes = [ctypes.c_char_p]
    lib.tms_tag_bytes.restype = ctypes.c_uint64
    for name in ("mock_live_bytes", "mock_mapped_bytes"):
        getattr(lib, name).restype = ctypes.c_size_t
    lib.mock_is_mapped.argtypes = [ctypes.c_uint64]
    lib.mock_fail_create_after.argtypes = [ctypes.c_long]
    return lib


def _malloc(lib, tag, size):
    lib.tms_set_current_tag(tag.encode())
    lib.tms_set_interesting_region(True)
    p = ctypes.c_void_p()
    assert lib.cudaMalloc(ctypes.byref(p), size) == 0
    lib.tms_set_interesting_region(False)
    return int(p.value)


def test_span_map_resume_maps_only_the_plan(tms_lib):
    lib = tms_lib
    spans = dsv.TmsSpans(symbol=lambda name: getattr(lib, name))
    live0 = lib.mock_live_bytes()
    p = _malloc(lib, "seat_kv", 20 * MIB)
    assert spans.info(p) == dsv.AllocInfo(20 * MIB, 20 * MIB, 20 * MIB, True)
    assert spans.info(p + 1) is None  # not an allocation base
    # plan for the NEXT resume: two ranges
    assert spans.set_spans(p, ((0, 4 * MIB), (10 * MIB, 12 * MIB)), now=False) == 0
    assert lib.tms_tag_bytes(b"seat_kv") == 20 * MIB  # mapped now (physical)
    lib.tms_pause(b"seat_kv")
    assert lib.tms_tag_bytes(b"seat_kv") == 6 * MIB  # planned for the next resume
    assert lib.mock_live_bytes() == live0 and spans.info(p).mapped == 0
    assert lib.tms_resume_rc(b"seat_kv") == 0
    assert lib.mock_live_bytes() - live0 == 6 * MIB
    assert lib.mock_is_mapped(p) and lib.mock_is_mapped(p + 11 * MIB)
    assert not lib.mock_is_mapped(p + 5 * MIB)
    assert spans.info(p) == dsv.AllocInfo(20 * MIB, 6 * MIB, 6 * MIB, True)
    lib.tms_pause(b"seat_kv")
    assert lib.mock_live_bytes() == live0
    # the plan persists; an empty plan is the stock whole mapping again
    assert spans.set_spans(p, (), now=False) == 0
    assert lib.tms_resume_rc(b"seat_kv") == 0
    assert lib.mock_live_bytes() - live0 == 20 * MIB and lib.mock_is_mapped(p + 19 * MIB)


def test_span_map_now_keeps_mapped_extents_and_grows(tms_lib):
    lib = tms_lib
    spans = dsv.TmsSpans(symbol=lambda name: getattr(lib, name))
    live0 = lib.mock_live_bytes()
    p = _malloc(lib, "seat_w", 16 * MIB)
    # born, then trimmed to the cap form (the stock whole handle is not inside)
    assert spans.set_spans(p, ((0, 6 * MIB),), now=True) == 0
    assert lib.mock_live_bytes() - live0 == 6 * MIB and not lib.mock_is_mapped(p + 7 * MIB)
    ext0 = lib.mock_extents()
    # grow: the [0, 6) extent stays (same handle), [6, 10) is new
    assert spans.set_spans(p, ((0, 10 * MIB),), now=True) == 0
    assert lib.mock_extents() == ext0 + 1
    assert lib.mock_live_bytes() - live0 == 10 * MIB
    assert lib.tms_tag_bytes(b"seat_w") == 10 * MIB
    # shrink back to the cap form: only the grown extent goes
    assert spans.set_spans(p, ((0, 6 * MIB),), now=True) == 0
    assert lib.mock_extents() == ext0 and lib.mock_live_bytes() - live0 == 6 * MIB
    # malformed plans are refused, nothing changes
    assert spans.set_spans(p, ((0, 3 * MIB),), now=True) == -3  # unaligned
    assert spans.set_spans(p, ((4 * MIB, 6 * MIB), (0, 2 * MIB)), now=True) == -3
    assert spans.set_spans(p, ((0, 18 * MIB),), now=True) == -3  # past the end
    assert spans.set_spans(p + 2 * MIB, ((0, 2 * MIB),), now=True) == -1
    assert lib.mock_live_bytes() - live0 == 6 * MIB
    lib.tms_pause(b"seat_w")
    assert lib.mock_live_bytes() == live0


def test_span_map_failed_resume_rolls_the_whole_tag_back(tms_lib):
    lib = tms_lib
    spans = dsv.TmsSpans(symbol=lambda name: getattr(lib, name))
    live0 = lib.mock_live_bytes()
    a = _malloc(lib, "seat_rb", 8 * MIB)
    b = _malloc(lib, "seat_rb", 8 * MIB)
    for p in (a, b):
        assert spans.set_spans(p, ((0, 2 * MIB), (4 * MIB, 6 * MIB)), now=False) == 0
    lib.tms_pause(b"seat_rb")
    lib.mock_fail_create_after(3)  # the 4th create (b's second range) fails
    assert lib.tms_resume_rc(b"seat_rb") != 0
    assert lib.mock_live_bytes() == live0
    assert not lib.mock_is_mapped(a) and not lib.mock_is_mapped(b)
    assert not spans.info(a).active and not spans.info(b).active
    assert lib.tms_resume_rc(b"seat_rb") == 0  # a retry after the refund is legal
    assert lib.mock_live_bytes() - live0 == 8 * MIB


def test_span_map_without_a_plan_is_the_stock_saver(tms_lib):
    lib = tms_lib
    live0 = lib.mock_live_bytes()
    p = _malloc(lib, "stock", 6 * MIB)
    assert lib.tms_tag_bytes(b"stock") == 6 * MIB
    lib.tms_pause(b"stock")
    assert lib.mock_live_bytes() == live0
    assert lib.tms_resume_rc(b"stock") == 0
    assert lib.mock_live_bytes() - live0 == 6 * MIB and lib.mock_is_mapped(p + 5 * MIB)


# ---- (6) planner and launcher ---------------------------------------------------

def _rows_fixture():
    from sglang.srt.planner import expert_residency as er

    def row(n, max_rows):
        return er.SeatTableRow(
            seats=n, ids_per_step=40 * n, waves=2, mamba_slots=7, host_mamba_mib=0.0,
            host_spec_mib=0.0, max_rows=max_rows, scratch_given=(100, 48, 48),
            waves_given=(1, 1, 1), fraction_given=(None, None, None),
            scratch_min=(None, None, None), fraction_max=(None, None, None), refusal=None)

    return tuple(row(n, (136 - (16 * (n - 1)) // 5, 140, 141)) for n in range(1, 7))


def test_the_seat_table_shows_each_ns_own_rows():
    from sglang.srt.planner import expert_residency as er

    form = er.SeatVramForm(temporal_slot_bytes=(48 * 128 * 128 * 2, 0, 0), gdn_layers=36,
                           expert_row_bytes=2534448, moe_layers=48, small_row_bytes=76848)
    rows = er._seat_vram_columns(_rows_fixture(), form)
    assert rows[-1].max_rows[0] == 120
    assert [r.runtime_rows[0] for r in rows] == [133, 130, 128, 125, 121, 120]
    assert [r.runtime_rows[1:] for r in rows] == [(140, 141)] * 6
    assert [r.seat_slot_limit for r in rows] == [7, 13, 19, 25, 32, 38]
    assert er.seat_expert_rows_value(rows) == "14,0,0"
    assert er.seat_fixed_mib(rows, form, "14,0,0") == (49.2, 0.0, 0.0)
    line = er.describe_seat_table(rows, marker="M", label="D")[0]
    assert "H95c Laufzeit-Zeilen [133, 140, 141]" in line and "'+13'" in line


def test_seat_vram_form_from_the_nf_config():
    from sglang.srt.planner import expert_residency as er

    cfg = {"linear_num_value_heads": 48, "linear_value_head_dim": 128,
           "linear_key_head_dim": 128, "layer_types": ["linear_attention"] * 36 + ["full_attention"] * 12,
           "moe_intermediate_size": 640, "hidden_size": 2560, "mamba_ssm_dtype": "float32"}
    f = er.seat_vram_form(cfg, ssm_dtype="bfloat16", rank_tp_ratio="1,0,0", n_ranks=3,
                          expert_row_bytes=2534448.0, moe_layers=48)
    assert f.temporal_slot_bytes == (1572864, 0, 0) and f.gdn_layers == 36
    assert f.small_row_bytes == 2534448 - 3 * 640 * 2560 // 2
    assert er.seat_vram_form({}, ssm_dtype=None, rank_tp_ratio="1", n_ranks=1,
                             expert_row_bytes=1.0, moe_layers=1) is None


@pytest.mark.skipif(not os.path.isdir(NF_MODEL), reason="needs the NF checkpoint header")
def test_dry_run_of_the_x177_form_writes_the_rows_into_env_d():
    from sglang.srt.planner import expert_residency as er
    from sglang.srt.weg2 import launcher as L

    ns = types.SimpleNamespace(model=NF_MODEL, profile=L.PROFILE_NEXTFLASH, d_bs=6,
                               extra_d="--max-running-requests 6",
                               env_d="SGLANG_MOE_SCRATCH_SLOTS=100,48,48")
    assert "D-SITZ-VRAM (H95c)" in L.apply_profile_d_seat_vram_default(ns)
    L.apply_profile_d_pool_waves_default(ns)
    env_d = dict(L.parse_group_env(ns.env_d), SGLANG_UNEVEN_MOE_EXPERT_SHARD="1",
                 SGLANG_MOE_OFFLOAD_GRAPH_MODE="pool", SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL="1",
                 SGLANG_WEG2_DRAFT_SHARE_EMBED="1")
    kw = dict(model_path=NF_MODEL, budgets_mib=[29624, 18664, 18672], ratios=[183, 137, 168],
              fractions=[0.06, 0.51, 0.48], scratch_rows=[100, 48, 48], rank_tp_ratio="1,0,0",
              env_d=env_d, reference_logs="", kv_tokens=262144, label="D",
              marker=L.D_RANK_SOLVE_MARKER, card_reference_logs="", seat_graph_mib=None,
              reference_seats=1)
    with mock.patch.object(L, "d_replayssm_spec_plan_form", lambda _ns: er.ReplaySSMSpecForm(
            ring_len=16, draft_tokens=4, max_running=6, ssm_dtype="bfloat16")):
        lines = L.d_seat_table_lines(ns, er, kw, "D")
    assert any("H95c Laufzeit-Zeilen [133, 140, 141]" in ln for ln in lines if "n=1:" in ln)
    assert any("H95c Laufzeit-Zeilen [120, 140, 141]" in ln for ln in lines if "n=6:" in ln)
    assert L.parse_group_env(ns.env_d)["SGLANG_WEG2_D_SEAT_EXPERT_ROWS"] == "14,0,0"
    assert L.parse_group_env(ns.env_d)["SGLANG_OPT_WEG2_D_SEAT_VRAM"] == "1"


def test_the_launcher_default_and_an_operators_word():
    from sglang.srt.weg2 import DEFAULT_D_SEAT_VRAM_NEXTFLASH
    from sglang.srt.weg2 import launcher as L

    assert DEFAULT_D_SEAT_VRAM_NEXTFLASH is True
    told = types.SimpleNamespace(profile=L.PROFILE_NEXTFLASH, env_d="SGLANG_OPT_WEG2_D_SEAT_VRAM=0")
    assert L.apply_profile_d_seat_vram_default(told) is None
    assert told.env_d == "SGLANG_OPT_WEG2_D_SEAT_VRAM=0"
    other = types.SimpleNamespace(profile=L.PROFILE_QWEN27B, env_d="")
    assert L.apply_profile_d_seat_vram_default(other) is None and other.env_d == ""
    src = open(L.__file__).read()
    j = src.index("    ns = build_parser().parse_args(")  # FL6: the argv goes through _canonical_flags
    assert "apply_profile_d_seat_vram_default(ns)" in src[j:j + 900]
    ns = types.SimpleNamespace(env_d="SGLANG_WEG2_D_SEAT_EXPERT_ROWS=9,0,0")
    from sglang.srt.planner import expert_residency as er

    form = er.SeatVramForm(temporal_slot_bytes=(1572864, 0, 0), gdn_layers=36,
                           expert_row_bytes=2534448, moe_layers=48, small_row_bytes=76848)
    rows = er._seat_vram_columns(_rows_fixture(), form)
    (line,) = L.apply_d_seat_expert_rows(ns, er, rows, form, "D")
    assert "nennt SGLANG_WEG2_D_SEAT_EXPERT_ROWS=9,0,0 selbst" in line
    assert ns.env_d == "SGLANG_WEG2_D_SEAT_EXPERT_ROWS=9,0,0"
