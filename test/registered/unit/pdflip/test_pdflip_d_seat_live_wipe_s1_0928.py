"""S1-Wisch (rc12z17, 28.09.): a live span move must keep the bytes it keeps.

THE METAL. rc12z17 D TP0 10:51:24: the wake's weights leg planned the paused
expert bank in the cap form (n=6, S0, rows_on=33); the saver's resume mapped
that plan as ONE extent ``[0, R+C+33)`` and the flip wrote the residents into
it. The kv leg then asked for n=4 at stage S1 (demand 365291 > 262144):
rows_on 33 -> 23 on the LIVE bank. core.cpp ``set_spans(now=True)`` keeps
only extents wholly inside the new plan -- the one extent was not, so it was
released and ``[0, R+C+23)`` mapped fresh: every resident, staging and LRU
row of all 48 layers was uninitialised memory. accept len 2.3 -> 1.05 from the
first round on, and it stayed there through the later S0 phases (the next
sleep handed the wiped residents to the store). Every earlier wake on metal
had n=5 (+34) or n=6 (+33): growth only, which keeps the old extent.

WHAT MUST HOLD.
(1) The plans of the bank and of the KV are cut at the lattice of every phase
    a rank can take; between any two such plans no extent straddles.
(2) S0 -> S1 at fewer seats with the bank LIVE: the rows the new plan keeps
    keep their extents (their bytes); only whole cells above the new end go.
(3) S1 -> S0 with the KV LIVE: the S0 prefix keeps its extents.
(4) A live move that WOULD release kept bytes stops by name (W-SEAT-WIPE)
    before anything moved -- tables, plans and pages untouched.
(5) Every live move writes one LIVE-SPANS line with wiped=0.

The fake saver below is core.cpp's span logic extent by extent: an extent is
an id; a released extent's id never comes back, so "same ids over a range"
is "same bytes".
"""
from __future__ import annotations

import logging
import os
import types
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.layers.moe import expert_pool_device as ep  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402

G = 4096
PAGE = 16
STAGES = "64,128,192"


class CoreTms:
    """``tms_set_spans`` / pause / resume exactly as core.cpp: one extent per
    mapped range; a live apply keeps an extent only if it lies wholly inside
    ONE range of the new plan, releases every other one and maps the
    uncovered parts of each range with fresh extents."""

    available = True

    def __init__(self):
        self.allocs = {}
        self.calls = []
        self._next = 1

    def _fresh(self, lo, hi):
        i = self._next
        self._next += 1
        return (int(lo), int(hi), i)

    def add(self, t, active=True):
        size = dsv.align_up(t.numel() * t.element_size(), G)
        self.allocs[t.data_ptr()] = {"size": size, "plan": [], "active": active,
                                     "ext": [self._fresh(0, size)] if active else []}

    def info(self, ptr):
        a = self.allocs.get(int(ptr))
        if a is None:
            return None
        planned = sum(h - l for l, h in (a["plan"] or [(0, a["size"])]))
        mapped = sum(h - l for l, h, _ in a["ext"])
        return dsv.AllocInfo(a["size"], mapped, planned, a["active"])

    def set_spans(self, ptr, spans, *, now):
        a = self.allocs[int(ptr)]
        self.calls.append((int(ptr), tuple(spans), now))
        prev_hi = 0
        for i, (lo, hi) in enumerate(spans):
            if hi <= lo or lo % G or hi % G or hi > a["size"] or (i and lo < prev_hi):
                return -3
            prev_hi = hi
        plan = [tuple(r) for r in spans]
        if plan == [(0, a["size"])]:
            plan = []
        a["plan"] = plan
        if not now or not a["active"]:
            return 0
        want = plan or [(0, a["size"])]
        kept = [e for e in a["ext"] if any(lo <= e[0] and e[1] <= hi for lo, hi in want)]
        new = []
        for lo, hi in want:
            cur = lo
            while cur < hi:
                cover = [e for e in kept if e[0] <= cur < e[1]]
                if cover:
                    cur = cover[0][1]
                    continue
                nxt = min([e[0] for e in kept if cur < e[0] < hi] + [hi])
                new.append(self._fresh(cur, nxt))
                cur = nxt
        a["ext"] = sorted(kept + new)
        return 0

    def pause(self, ptr):
        self.allocs[int(ptr)].update(active=False, ext=[])

    def resume(self, ptr):
        a = self.allocs[int(ptr)]
        a.update(active=True, ext=[self._fresh(lo, hi) for lo, hi in (a["plan"] or [(0, a["size"])])])

    def ids(self, ptr, lo, hi):
        """The extents holding the bytes ``[lo, hi)``."""
        return {e[2] for e in self.allocs[int(ptr)]["ext"] if e[0] < hi and lo < e[1]}

    def mapped(self, ptr):
        return sum(h - l for l, h, _ in self.allocs[int(ptr)]["ext"])


class FakeCache:
    def __init__(self, layer_id, R, C, S, X, row_elems):
        from flliper.srt.layers.moe.expert_offload import MoEExpertOffloadCache

        self.layer = types.SimpleNamespace(layer_id=layer_id)
        self.seat_rows = X
        self.planner = types.SimpleNamespace(buffer_size=R + C)
        self._resident = {"w13": torch.zeros(R + C + X, row_elems, dtype=torch.int32)}
        hot = {e: e for e in range(R)}
        self._pool_tables = ep.allocate_pool_tables(
            "cpu", 60, R + C + X, R, S, hot, [(-1 if e in hot else e) for e in range(60)],
            seat_rows=X)
        self._pool_ready = True
        self.set_seat_rows_on = types.MethodType(MoEExpertOffloadCache.set_seat_rows_on, self)


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


def _armed(tms):
    from flliper.srt.environ import envs

    return _Ctx([mock.patch.dict(os.environ, {dsv.GROUP_ENV: "D"}),
                 envs.FLLIPER_OPT_PDFLIP_D_SEAT_VRAM.override(True),
                 envs.FLLIPER_PDFLIP_D_KV_STAGE_TOKENS.override(STAGES),
                 envs.FLLIPER_PDFLIP_D_KV_STAGE_ROWS.override(32),
                 envs.FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS.override(""),
                 mock.patch.object(dsv, "_TMS", tms),
                 mock.patch.object(dsv, "granule_for", lambda _d: G),
                 mock.patch.object(dsv, "_KV_BORN", [])])


def _rank(tms):
    """The #251c desk rank (test_pdflip_d_kv_stage_runtime_251c): GDN pool 11
    slots x 4 KiB x 3 layers, 2 seats; two MoE layers of 4 KiB rows
    (R3 + C7 + X40); two KV tensors of the top stage (192 + 16 tokens,
    1 KiB a token). S0 = 64 tokens with 32 stage rows ON, S1 = 128."""
    from flliper.srt.mem_cache.allocator.mamba import MambaSlotAllocator
    from flliper.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from flliper.srt.mem_cache.memory_pool import MHATokenToKVPool

    temporal = torch.zeros(3, 12, 1024, dtype=torch.int32)
    pool = types.SimpleNamespace(size=11, mamba_cache=types.SimpleNamespace(temporal=temporal))
    alloc = MambaSlotAllocator(size=11, device="cpu")
    caches = [FakeCache(i, 3, 7, 2, 40, 1024) for i in range(2)]
    with mock.patch.object(dsv, "stage_form", lambda env=None: None):
        kv_pool = MHATokenToKVPool(192, PAGE, torch.bfloat16, 1, 8, 2, "cpu", False,
                                   enable_alt_stream=False)
    tms.add(temporal, active=True)
    for c in caches:
        for buf in c._resident.values():
            tms.add(buf, active=True)
            # the bank's birth trim (seat_expert_buffer): the cap form's rows
            tms.set_spans(buf.data_ptr(), dsv.row_spans(
                dsv.RowTensorGeom("b", 10, 50, G, 50 * G), 10, G), now=True)
    kv = []
    for i in range(2):
        t = torch.zeros(192 + PAGE, 256, dtype=torch.int32)
        tms.add(t, active=True)
        kv.append(dsv.kv_stage_born(t, pool_size=192, page_size=PAGE, name="k%d" % i))
    modules = [types.SimpleNamespace(_expert_offload=c) for c in caches]
    model = types.SimpleNamespace(modules=lambda: iter(modules))
    rtp = types.SimpleNamespace(mamba_pool=pool, mamba_allocator=alloc)
    kv_alloc = PagedTokenToKVPoolAllocator(192, page_size=PAGE, dtype=torch.int64, device="cpu",
                                           kvcache=types.SimpleNamespace(), need_sort=False)
    dsv.kv_stage_boot_cap(kv_alloc, PAGE)
    runner = types.SimpleNamespace(req_to_token_pool=rtp, model=model, token_to_kv_pool=kv_pool,
                                   token_to_kv_pool_allocator=kv_alloc, page_size=PAGE)
    sched = types.SimpleNamespace(server_args=types.SimpleNamespace(max_running_requests=2),
                                  tp_worker=types.SimpleNamespace(model_runner=runner))
    banks = [b for c in caches for b in c._resident.values()]
    return types.SimpleNamespace(sched=sched, temporal=temporal, caches=caches, kv=kv,
                                 banks=banks, kv_alloc=kv_alloc)


def _seats(n):
    from flliper.srt.pdflip import d_seats

    return d_seats.phase_seats(n, 0, cap=2)


def _pause_all(tms):
    for p in list(tms.allocs):
        tms.pause(p)


def _kv_leg(epoch, n, demand):
    return types.SimpleNamespace(epoch=epoch, handoff_n=n, parked_n=0, phase_kv_tokens=demand)


# ---- (1) the lattice ----------------------------------------------------------

def test_plans_of_all_phases_share_one_lattice():
    rg = dsv.RowTensorGeom("b", 10, 50, 3000, dsv.align_up(50 * 3000, G))
    ks = (0, 16, 23, 32, 33, 40)
    cuts = [10 + k for k in ks]
    plans = {k: dsv.row_spans(rg, 10 + k, G, cuts=cuts) for k in ks}
    for a in ks:
        for b in ks:
            assert dsv.straddling(plans[a], plans[b]) is None, (a, b)
        assert dsv.span_bytes(plans[a]) == dsv.span_bytes(dsv.row_spans(rg, 10 + a, G))
    # the uncut form of rc12z17: one range -- the shrink 33 -> 23 straddles it
    assert dsv.straddling(dsv.row_spans(rg, 43, G), dsv.row_spans(rg, 33, G)) is not None
    sg = dsv.SlotTensorGeom("kv", 3, 40, 1000, dsv.align_up(3 * 40 * 1000, G) + G)
    stage_keeps = (10, 20, 30)
    kp = {s: dsv.slot_spans(sg, s, G, cuts=stage_keeps) for s in stage_keeps}
    for a in stage_keeps:
        for b in stage_keeps:
            assert dsv.straddling(kp[a], kp[b]) is None, (a, b)
        assert dsv.span_bytes(kp[a]) == dsv.span_bytes(dsv.slot_spans(sg, a, G))


# ---- (2) S0 -> S1 with the bank live: the metal sequence ----------------------

def test_s0_to_s1_on_a_live_bank_keeps_the_rows_it_keeps(caplog):
    tms = CoreTms()
    with _armed(tms):
        r = _rank(tms)
        _pause_all(tms)
        # weights leg: the cap form, S0, 32 stage rows ON -- the bank is paused
        st = dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
        ctl = r.sched._pdflip_d_seat_vram
        assert (st.stage, ctl.applied.extra_rows) == (0, 32)
        for b in r.banks:
            tms.resume(b.data_ptr())  # the flip writes the residents into these pages
        k1 = 32 - 16
        keep_bytes = (10 + k1) * G
        before = {b.data_ptr(): tms.ids(b.data_ptr(), 0, keep_bytes) for b in r.banks}
        caplog.set_level(logging.INFO)
        # kv leg: 2 seats, demand 100 > S0 = 64 -> S1 with 16 rows ON: a LIVE shrink
        st = dsv.on_wake(r.sched, _kv_leg("e1", 2, 100), _seats(2))
        assert (st.stage, ctl.applied.extra_rows) == (1, k1)
        for b in r.banks:
            p = b.data_ptr()
            assert tms.ids(p, 0, keep_bytes) == before[p], "the kept rows were remapped fresh"
            assert tms.mapped(p) == keep_bytes  # only whole cells above the new end went
        lines = [m for m in caplog.messages if dsv.LIVE_MARK in m]
        assert lines and "rows_on 32->16" in lines[-1] and "wiped=0" in lines[-1]
        assert "cells_freed=" in lines[-1] and "cells_freed=0 " not in lines[-1]


# ---- (3) S0 -> S1 -> S0 with the KV live ----------------------------------------

def test_s1_back_to_s0_on_a_live_kv_keeps_the_s0_prefix():
    tms = CoreTms()
    with _armed(tms):
        r = _rank(tms)
        _pause_all(tms)
        dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
        st = dsv.on_wake(r.sched, _kv_leg("e1", 2, 100), _seats(2))
        assert st.stage == 1
        for p in list(tms.allocs):
            tms.resume(p)  # the phase runs: KV S1 mapped, bytes written
        s0_bytes = (64 + PAGE) * 1024
        before = {t.data_ptr(): tms.ids(t.data_ptr(), 0, s0_bytes) for t in r.kv}
        # the next wake's weights leg resets to S0 while the KV is still mapped
        st = dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e2"), None)
        assert st.stage == 0
        for t in r.kv:
            p = t.data_ptr()
            assert tms.ids(p, 0, s0_bytes) == before[p], "the S0 prefix was remapped fresh"
            assert tms.mapped(p) == s0_bytes


# ---- (4) the named stop ------------------------------------------------------------

def test_a_live_move_that_would_release_kept_bytes_stops_by_name():
    tms = CoreTms()
    with _armed(tms):
        r = _rank(tms)
        _pause_all(tms)
        dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
        ctl = r.sched._pdflip_d_seat_vram
        b0 = r.banks[0]
        # rc12z17's state: the resume mapped the cap form as ONE extent
        tms.allocs[b0.data_ptr()]["plan"] = [(0, 42 * G)]
        for b in r.banks:
            tms.resume(b.data_ptr())
        ctl.spans_by_ptr[b0.data_ptr()] = ((0, 42 * G),)
        calls, seat_on = len(tms.calls), [c._pool_tables.seat_on for c in r.caches]
        ids = tms.ids(b0.data_ptr(), 0, 26 * G)
        with pytest.raises(dsv.PdFlipDSeatVramRefused, match=dsv.WIPE_CODE):
            dsv.on_wake(r.sched, _kv_leg("e1", 2, 100), _seats(2))
        assert len(tms.calls) == calls  # nothing moved
        assert [c._pool_tables.seat_on for c in r.caches] == seat_on
        assert tms.ids(b0.data_ptr(), 0, 26 * G) == ids


# ---- (5) growth stays what it was ----------------------------------------------------

def test_a_live_grow_keeps_every_extent_and_writes_the_line(caplog):
    tms = CoreTms()
    with _armed(tms):
        r = _rank(tms)
        _pause_all(tms)
        dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
        for b in r.banks:
            tms.resume(b.data_ptr())
        before = {b.data_ptr(): tms.ids(b.data_ptr(), 0, 42 * G) for b in r.banks}
        caplog.set_level(logging.INFO)
        # one seat, demand within S0: more rows ON than the cap form's 32
        dsv.on_wake(r.sched, _kv_leg("e1", 1, 40), _seats(1))
        ctl = r.sched._pdflip_d_seat_vram
        assert ctl.applied.extra_rows > 32
        for b in r.banks:
            assert tms.ids(b.data_ptr(), 0, 42 * G) == before[b.data_ptr()]
        line = [m for m in caplog.messages if dsv.LIVE_MARK in m][-1]
        assert "cells_freed=0 " in line and "wiped=0" in line


# ---- PA review of 1bab093912 ------------------------------------------------------

def test_a_small_layer_major_kv_is_born_at_the_stage_lattice():
    """(3) A layer-major tensor whose layers are small against the granule (the
    QSA keys on metal) coalesced at its birth into ONE extent across the stage
    lattice: the first live KV move would release it (W-SEAT-WIPE mid-run).
    RED on 1bab093912: born uncut."""
    tms = CoreTms()
    with _armed(tms):
        q = torch.zeros(3, 24 * 64, dtype=torch.int32)  # 3 layers x 24 slots x 256 B
        tms.add(q, active=True)
        dsv.kv_stage_born(q, pool_size=192, page_size=PAGE, name="qsa", tokens_per_slot=4,
                          layers=3, slots=24)
        geom = dsv._KV_BORN[-1][1]
        cuts = [geom.slots_for(t) for t in (64, 128, 192)]
        ext = [(lo, hi) for lo, hi, _ in tms.allocs[q.data_ptr()]["ext"]]
        assert len(ext) > 1
        for t in (64, 128, 192):
            plan = dsv.slot_spans(geom.geom, geom.slots_for(t), G, cuts=cuts)
            assert dsv.straddling(ext, plan) is None, (t, ext, plan)


def test_a_birth_across_the_lattice_stops_at_the_controllers_build():
    """(3) The named stop: a KV tensor recorded with a birth extent across the
    lattice stops the controller's build with W-SEAT-WIPE (not swallowed into
    "no controller on this rank"), before any phase ran."""
    tms = CoreTms()
    with _armed(tms), mock.patch.dict(dsv._KV_BORN_SPANS, {}, clear=True):
        r = _rank(tms)
        p = r.kv[0].data_ptr()
        key = [k for k in dsv._KV_BORN_SPANS if k[0] == p][0]
        dsv._KV_BORN_SPANS[key] = ((0, 144 * 1024),)  # across the S0 cut at 80 KiB
        with pytest.raises(dsv.PdFlipDSeatVramWipe, match="controller's build"):
            dsv.controller(r.sched)


def test_apply_turns_rows_off_before_a_live_shrink_unmaps():
    """(2) H95c ``apply`` (no stage form in the phase): a live shrink turns the
    rows OFF in the tables and synchronizes BEFORE the pages go, as
    ``apply_stage`` does. RED on 1bab093912: set_spans(now=True) first."""
    tms = CoreTms()
    with _armed(tms):
        r = _rank(tms)
        tms.pause(r.temporal.data_ptr())  # the Mamba pool may shrink
        ctl = dsv.controller(r.sched)
        ctl.apply(1)
        k1 = ctl.rows_on
        events = []
        banks = {b.data_ptr() for b in r.banks}
        real_set = tms.set_spans

        def _spans(ptr, spans, *, now):
            if int(ptr) in banks and now:
                events.append("pages")
            return real_set(ptr, spans, now=now)

        tms.set_spans = _spans
        for c in r.caches:
            real = c.set_seat_rows_on
            c.set_seat_rows_on = (lambda k, device_write=False, _r=real:
                                  (events.append("rows"), _r(k, device_write=device_write))[1])
        ctl.apply(2)
        assert ctl.rows_on < k1, (k1, ctl.rows_on)
        assert "pages" in events and events.index("rows") < events.index("pages"), events


def test_the_wipe_refusal_does_not_claim_nothing_changed():
    """(4) The wake set its phase limit and KV cap before the apply; the text
    names what did not move instead of 'Nothing was changed'."""
    tms = CoreTms()
    with _armed(tms):
        r = _rank(tms)
        ctl = dsv.controller(r.sched)
        p = r.banks[0].data_ptr()
        ctl.spans_by_ptr[p] = ((0, 42 * G),)
        with pytest.raises(dsv.PdFlipDSeatVramRefused) as exc:
            ctl.refuse_wipes([(p, "b", ((0, 26 * G), (26 * G, 42 * G)), True)])
        assert "Nothing was changed" not in str(exc.value)
        assert "phase limit" in str(exc.value)
