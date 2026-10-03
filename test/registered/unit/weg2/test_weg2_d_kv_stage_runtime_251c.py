"""#251c runtime: D's KV stages on the Form A attention host, at the wake.

WHAT MUST HOLD.
(1) The form values: fewer than two stages, another group or the switch off
    = no form (H95c byte-identical); the pool is VIRTUALLY the top stage and
    the budget must hold S0 on the attention host (a named refusal below it,
    a Form A worker takes the rows without a budget).
(2) A KV tensor of a stage-form pool is born trimmed to S0 (one tensor at a
    time -- the boot never holds the top stage's pages); a tensor the saver
    does not own is refused by name; every other tensor passes untouched.
(3) The allocator hands out only the stage's pages (KvRowCap, replicated),
    the flush zeroes only mapped rows (safe_zero_rows).
(4) The wake: the weights leg resets to the boot form (S0, stage rows ON);
    the kv leg's demand picks the smallest stage that holds it; the KV plan
    grows (paused), the LIVE expert bank shrinks with its rows turned OFF in
    the tables BEFORE a page goes; the next wake with less demand falls back;
    above the top stage the youngest parks (over). A worker rank without
    pages takes the same stage and the same page cap from the same request.
(5) A form TP0's pages cannot fund is refused by name at the first wake.
"""
from __future__ import annotations

import logging
import os
import types
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.layers.moe import expert_pool_device as ep  # noqa: E402
from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402


@pytest.fixture(autouse=True)
def _kv_stage_table_path():
    """29.09.: #251d is the default now; this file covers the table path, so
    the switch is pinned off here (a test that wants demand overrides it)."""
    from sglang.srt.environ import envs

    with envs.SGLANG_WEG2_D_KV_STAGE_BY_DEMAND.override(False):
        yield


G = 4096
PAGE = 16
STAGES = "64,128,192"


class FakeTms:
    def __init__(self):
        self.allocs = {}
        self.calls = []

    available = True

    def add(self, t, active=True):
        size = dsv.align_up(t.numel() * t.element_size(), G)
        self.allocs[t.data_ptr()] = {"size": size, "plan": None,
                                     "mapped": [(0, size)] if active else [], "active": active}

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

    def mapped(self, ptr):
        return sum(h - l for l, h in self.allocs[int(ptr)]["mapped"])


class FakeCache:
    def __init__(self, layer_id, R, C, S, X, row_elems, log):
        from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

        self.layer = types.SimpleNamespace(layer_id=layer_id)
        self.seat_rows = X
        self.planner = types.SimpleNamespace(buffer_size=R + C)
        self._resident = {"w13": torch.zeros(R + C + X, row_elems, dtype=torch.int32)}
        hot = {e: e for e in range(R)}
        self._pool_tables = ep.allocate_pool_tables(
            "cpu", 60, R + C + X, R, S, hot, [(-1 if e in hot else e) for e in range(60)],
            seat_rows=X)
        self._pool_ready = True
        real = types.MethodType(MoEExpertOffloadCache.set_seat_rows_on, self)

        def logged(k, *, device_write):
            log.append(("tables", int(k)))
            return real(k, device_write=device_write)

        self.set_seat_rows_on = logged


def _env(**extra):
    from sglang.srt.environ import envs

    return [mock.patch.dict(os.environ, {dsv.GROUP_ENV: "D"}),
            envs.SGLANG_OPT_WEG2_D_SEAT_VRAM.override(True),
            envs.SGLANG_WEG2_D_KV_STAGE_TOKENS.override(extra.get("tokens", STAGES)),
            envs.SGLANG_WEG2_D_KV_STAGE_ROWS.override(extra.get("rows", 32)),
            envs.SGLANG_WEG2_D_KV_STAGE_MAX_BY_SEATS.override(extra.get("max_by", ""))]


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


def _kv_allocator():
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    return PagedTokenToKVPoolAllocator(192, page_size=PAGE, dtype=torch.int64, device="cpu",
                                       kvcache=types.SimpleNamespace(), need_sort=False)


def _real_kv_pool():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    with mock.patch.object(dsv, "stage_form", lambda env=None: None):
        return MHATokenToKVPool(192, PAGE, torch.bfloat16, 1, 8, 2, "cpu", False,
                                enable_alt_stream=False)


def _rank(tms, *, pages=True, X=40):
    """A D rank: the GDN pool (11 slots x 4 KiB x 3 layers, 2 seats), two
    MoE layers of 4 KiB rows (R3 + C7 + X), two KV tensors born at the top
    stage (192 tokens + a 16-token page, 1 KiB a token). One stage step of
    64 tokens costs 128 KiB = 16 expert rows."""
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

    log = []
    temporal = torch.zeros(3, 12, 1024, dtype=torch.int32)
    pool = types.SimpleNamespace(size=11, mamba_cache=types.SimpleNamespace(temporal=temporal))
    alloc = MambaSlotAllocator(size=11, device="cpu")
    caches = [FakeCache(i, 3, 7, 2, X if pages else 0, 1024, log) for i in range(2)]
    # rc12z13: a REAL MHATokenToKVPool -- its safe_zero_rows is a property
    # (#656) that the stage must feed, never assign. Built with the form off:
    # its own tensors are not the FakeTms allocations this rank trims.
    kv_pool = _real_kv_pool()
    kv = []
    if pages:
        tms.add(temporal, active=True)
        for c in caches:
            for buf in c._resident.values():
                tms.add(buf, active=True)
        for i in range(2):
            t = torch.zeros(192 + PAGE, 256, dtype=torch.int32)
            tms.add(t, active=True)
            kv.append(dsv.kv_stage_born(t, pool_size=192, page_size=PAGE, name="k%d" % i))
    modules = [types.SimpleNamespace(_expert_offload=c) for c in caches]
    model = types.SimpleNamespace(modules=lambda: iter(modules))
    rtp = types.SimpleNamespace(mamba_pool=pool, mamba_allocator=alloc)
    kv_alloc = _kv_allocator()
    dsv.kv_stage_boot_cap(kv_alloc, PAGE)
    runner = types.SimpleNamespace(req_to_token_pool=rtp, model=model,
                                   token_to_kv_pool=kv_pool if pages else None,
                                   token_to_kv_pool_allocator=kv_alloc, page_size=PAGE)
    sched = types.SimpleNamespace(server_args=types.SimpleNamespace(max_running_requests=2),
                                  tp_worker=types.SimpleNamespace(model_runner=runner))
    return types.SimpleNamespace(sched=sched, temporal=temporal, caches=caches, kv=kv,
                                 kv_pool=kv_pool, kv_alloc=kv_alloc, log=log)


def _armed(tms, **extra):
    return _Ctx(_env(**extra) + [mock.patch.object(dsv, "_TMS", tms),
                                 mock.patch.object(dsv, "granule_for", lambda _d: G),
                                 mock.patch.object(dsv, "_KV_BORN", [])])


def _seats(n):
    from sglang.srt.weg2 import d_seats

    return d_seats.phase_seats(n, 0, cap=2)


# ---- (1) the form values ------------------------------------------------------

def test_no_form_off_one_stage_or_another_group():
    from sglang.srt.environ import envs

    with _Ctx(_env()):
        form = dsv.stage_form()
        assert form.tokens == (64, 128, 192) and form.rows_on == 32 and form.max_stage(2) == 2
    with _Ctx(_env(tokens="262144")):
        assert dsv.stage_form() is None
    with _Ctx(_env() + [mock.patch.dict(os.environ, {dsv.GROUP_ENV: "P"})]):
        assert dsv.stage_form() is None
    with _Ctx(_env() + [envs.SGLANG_OPT_WEG2_D_SEAT_VRAM.override(False)]):
        assert dsv.stage_form() is None
        assert dsv.kv_stage_pool_tokens(262144) == 262144
    with _Ctx(_env(max_by="0,1")):
        form = dsv.stage_form()
        assert (form.max_stage(1), form.max_stage(2), form.max_stage(6)) == (0, 1, 1)


def test_the_pool_is_the_top_stage_and_the_budget_must_hold_s0():
    with _Ctx(_env()):
        assert dsv.kv_stage_pool_tokens(100) == 192
        assert dsv.kv_stage_pool_tokens(10, is_form_a_worker=True) == 192
        with pytest.raises(dsv.Weg2DSeatVramRefused, match="below stage S0 = 64"):
            dsv.kv_stage_pool_tokens(63)


def test_a_draft_pool_takes_the_stage_form_only_with_the_targets_slot_ids():
    """Review 28.09. (#251c note (c)): only the TARGET allocator is capped to
    S0; a draft sized at the top stage is sound only while it writes at the
    target's slot ids (MTP/EAGLE). A draft with its own allocator (DFlash solo
    host) is refused by name, never silently handed ids above S0."""
    with _Ctx(_env()):
        assert dsv.kv_stage_pool_tokens(100, is_draft_worker=True) == 192
        assert dsv.kv_stage_pool_tokens(100, is_draft_worker=True, draft_shares_slots=True) == 192
        with pytest.raises(dsv.Weg2DSeatVramRefused, match="own allocator"):
            dsv.kv_stage_pool_tokens(100, is_draft_worker=True, draft_shares_slots=False)
        # the target is unaffected by the draft flag default
        assert dsv.kv_stage_pool_tokens(100, draft_shares_slots=False) == 192
    with _Ctx(_env() + [__import__("sglang.srt.environ", fromlist=["envs"]).envs.SGLANG_OPT_WEG2_D_SEAT_VRAM.override(False)]):
        # form off: a solo-host draft keeps its own number, no refusal
        assert dsv.kv_stage_pool_tokens(100, is_draft_worker=True, draft_shares_slots=False) == 100


def test_the_kv_mixin_passes_the_draft_role_and_the_solo_host():
    import inspect

    from sglang.srt.model_executor import model_runner_kv_cache_mixin as mx

    src = inspect.getsource(mx)
    assert 'is_draft_worker=bool(getattr(self, "is_draft_worker", False))' in src
    assert 'draft_shares_slots=not bool(getattr(self, "is_draft_solo_host", False))' in src


# ---- (2) born trimmed -----------------------------------------------------------

def test_a_kv_tensor_is_born_with_only_s0s_pages():
    tms = FakeTms()
    with _armed(tms):
        t = torch.zeros(192 + PAGE, 256, dtype=torch.int32)
        tms.add(t, active=True)
        assert dsv.kv_stage_born(t, pool_size=192, page_size=PAGE, name="k0") is t
        # S0 = 64 tokens + the page = 80 rows of 1 KiB: 80 KiB mapped, the rest unmapped
        assert tms.mapped(t.data_ptr()) == 80 * 1024
        assert [p for p, _ in dsv._KV_BORN] == [t.data_ptr()]
        # a layer-major QSA-like tensor: one slot per 4 tokens, 3 layers
        q = torch.zeros(3, 52 * 64, dtype=torch.int32)
        tms.add(q, active=True)
        dsv.kv_stage_born(q, pool_size=192, page_size=PAGE, name="qsa", tokens_per_slot=4,
                          layers=3, slots=52)
        geom = dsv._KV_BORN[-1][1]
        assert geom.geom.layers == 3 and geom.slots_for(64) == 20
        # untouched: a smaller pool, and a tensor the saver does not own is refused
        small = torch.zeros(80, 256, dtype=torch.int32)
        tms.add(small, active=True)
        n = len(tms.calls)
        dsv.kv_stage_born(small, pool_size=64, page_size=PAGE)
        assert len(tms.calls) == n
        with pytest.raises(dsv.Weg2DSeatVramRefused, match="not a saver allocation"):
            dsv.kv_stage_born(torch.zeros(208, 256, dtype=torch.int32), pool_size=192,
                              page_size=PAGE, name="k9")


def test_off_nothing_is_trimmed():
    tms = FakeTms()
    with _Ctx([mock.patch.object(dsv, "_TMS", tms), mock.patch.object(dsv, "_KV_BORN", [])]):
        t = torch.zeros(208, 256, dtype=torch.int32)
        tms.add(t, active=True)
        assert dsv.kv_stage_born(t, pool_size=192, page_size=PAGE) is t
        assert not tms.calls and not dsv._KV_BORN


# ---- (3) the allocator's cap --------------------------------------------------

def test_the_allocator_hands_out_only_the_stages_pages():
    with _Ctx(_env()):
        a = _kv_allocator()
        assert dsv.kv_stage_boot_cap(a, PAGE) == 4  # S0 = 64 tokens = 4 pages
        assert a.available_size() == 4 * PAGE
        assert a.residency_withheld_slots == 8 * PAGE
        got = a.alloc(4 * PAGE)
        assert got is not None and int(got.max()) < 5 * PAGE and a.alloc(PAGE) is None
        assert dsv.max_live_page(a) == 4
        a.free(got)
        assert dsv.max_live_page(a) == 0
        dsv._engage_kv_cap(a, 192, PAGE)  # the top stage: nothing withheld
        assert a.available_size() == 12 * PAGE and a.residency_withheld_slots == 0


# ---- (4) the wake -------------------------------------------------------------------

def test_demand_picks_the_stage_the_bank_shrinks_tables_first_and_the_next_wake_falls_back(caplog):
    tms = FakeTms()
    with _armed(tms):
        r = _rank(tms)
        for p in list(tms.allocs):
            tms.pause(p)
        # weights leg: the boot form -- S0, the 32 stage rows ON
        st = dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
        ctl = r.sched._weg2_d_seat_vram
        assert (st.stage, ctl.applied.extra_rows) == (0, 32)
        for c in r.caches:
            for buf in c._resident.values():
                tms.resume(buf.data_ptr())
        r.log.clear()
        # kv leg, 2 seats, demand 100 tokens > S0 (64): stage S1 (128)
        caplog.set_level(logging.INFO)
        kv = types.SimpleNamespace(epoch="e1", handoff_n=2, parked_n=0, phase_kv_tokens=100)
        st = dsv.on_wake(r.sched, kv, _seats(2))
        assert (st.stage, st.stage_tokens, st.over) == (1, 128, False)
        k1 = ctl.applied.extra_rows
        assert k1 == 32 - 16
        # the LIVE bank shrank: the tables turned the rows OFF BEFORE a page went
        assert r.log[0] == ("tables", k1)
        shrink = [c for c in tms.calls if c[2] and c[0] in {b.data_ptr() for x in r.caches
                                                              for b in x._resident.values()}]
        # S1-Wisch: the prefix is cut at the lattice of all phases (whole cells go)
        assert shrink and all(c[1][0][0] == 0 and c[1][-1][1] == (10 + k1) * G
                              and dsv.span_bytes(c[1]) == (10 + k1) * G for c in shrink[-2:])
        # the KV (paused) got S1's plan: 128 + 16 rows of 1 KiB
        for t in r.kv:
            # S1-Wisch: cut at S0 (64 + 16 rows) -- a later live S1 -> S0 keeps that cell
            assert tms.allocs[t.data_ptr()]["plan"] == [(0, 80 * 1024), (80 * 1024, 144 * 1024)]
            assert not tms.allocs[t.data_ptr()]["active"]  # a plan, mapped at the resume
        assert r.kv_alloc.available_size() == 8 * PAGE  # S1 = 8 pages
        assert r.kv_pool.safe_zero_rows == 128 + PAGE
        assert any(dsv.STAGE_MARK in m and "stage=S1" in m for m in caplog.messages)
        # the next wake with less demand falls back to S0 at once
        for p in list(tms.allocs):
            tms.pause(p)
        dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e2"), None)
        st = dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e2", handoff_n=1, parked_n=0,
                                                        phase_kv_tokens=40), _seats(1))
        assert st.stage == 0 and r.kv_alloc.available_size() == 4 * PAGE
        # S1 -> S0: the flush bound falls back with the pages (rc12z13)
        assert r.kv_pool.safe_zero_rows == 64 + PAGE
        assert r.kv_pool._committed_row_bound() == 64 + PAGE
        # one seat: its GDN pages fund more rows than the boot form's 32
        assert ctl.applied.extra_rows > 32


def test_above_the_top_stage_the_youngest_parks_and_workers_agree():
    tms = FakeTms()
    with _armed(tms):
        host = _rank(tms)
        worker = _rank(FakeTms(), pages=False)
        wake = types.SimpleNamespace(epoch="e1", handoff_n=2, parked_n=0, phase_kv_tokens=500)
        for p in list(tms.allocs):
            tms.pause(p)
        a = dsv.on_wake(host.sched, wake, _seats(2))
        b = dsv.on_wake(worker.sched, wake, _seats(2))
        assert (a.stage, a.over) == (b.stage, b.over) == (2, True)
        assert host.kv_alloc.available_size() == worker.kv_alloc.available_size() == 12 * PAGE
        assert host.sched._weg2_d_seat_vram.applied.extra_rows == 0
        # no demand on the wake (an older front): S0, the boot form
        c = dsv.on_wake(worker.sched, types.SimpleNamespace(epoch="e2", handoff_n=2), _seats(2))
        assert (c.stage, c.over) == (0, False)


# ---- (5) a form the pages cannot fund ----------------------------------------------

def test_a_form_the_pages_cannot_fund_is_refused_by_name(caplog):
    tms = FakeTms()
    with _armed(tms):
        r = _rank(tms, X=20)  # 20 reserved rows cannot pay S2's 32 at two seats
        for p in list(tms.allocs):
            tms.pause(p)
        caplog.set_level(logging.WARNING)
        dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
        # the controller refused the form by name (a missing piece names itself)
        assert r.sched._weg2_d_seat_vram is False
        assert any("cannot fund" in m for m in caplog.messages)


# ---- (6) the capture floor over the rank's own cells ------------------------------

def test_the_capture_floor_comes_from_the_ranks_cells_and_workers_have_none():
    tms = FakeTms()
    with _armed(tms, max_by="2,1"), mock.patch.dict(dsv._CAPTURE, {"runner": None,
                                                                     "floors": None}):
        r = _rank(tms)
        runner = r.sched.tp_worker.model_runner
        runner.server_args = r.sched.server_args
        dsv.note_capture_context(runner)
        ctl = dsv.SeatVram.from_runtime(cap=2, req_to_token_pool=runner.req_to_token_pool,
                                        model=runner.model)
        want = dsv.capture_floors(ctl.form, ctl.cells, 2)
        assert want[0] == min(ctl.cells[(1, 2)].extra_rows, ctl.cells[(2, 1)].extra_rows)
        assert want[1] == ctl.cells[(2, 1)].extra_rows
        # 4 verify tokens x top-2 per seat: 8 ids = bs1, 16 ids = bs2, past the cap: 0
        assert dsv.capture_floor_rows(8, 8) == want[0]
        assert dsv.capture_floor_rows(16, 8) == want[1]
        assert dsv.capture_floor_rows(24, 8) == 0
    with _armed(FakeTms()), mock.patch.dict(dsv._CAPTURE, {"runner": None, "floors": None}):
        w = _rank(FakeTms(), pages=False)
        runner = w.sched.tp_worker.model_runner
        runner.server_args = w.sched.server_args
        dsv.note_capture_context(runner)
        assert dsv.capture_floor_rows(8, 8) == 0


def test_pages_held_above_the_forms_stage_stop_the_wake_by_name():
    tms = FakeTms()
    with _armed(tms, max_by="2,0"):
        r = _rank(tms)
        for p in list(tms.allocs):
            tms.pause(p)
        dsv._engage_kv_cap(r.kv_alloc, 192, PAGE)
        held = r.kv_alloc.alloc(9 * PAGE)  # pages up to 9 > S1's 8: only S2 maps them
        assert held is not None and dsv.max_live_page(r.kv_alloc) == 9
        wake = types.SimpleNamespace(epoch="e1", handoff_n=2, parked_n=0, phase_kv_tokens=40)
        with pytest.raises(dsv.Weg2DSeatVramRefused, match="above the S0 the form allows 2"):
            dsv.on_wake(r.sched, wake, _seats(2))
        # one seat may take S2: the same pages are fine there
        st = dsv.on_wake(r.sched, types.SimpleNamespace(
            epoch="e2", handoff_n=1, parked_n=0, phase_kv_tokens=40), _seats(1))
        assert st.stage == 2
