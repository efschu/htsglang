"""TAIL-STAGE-WORKER (30.09., NF P->D flip): the E2 tail staging runs on ONE
long-lived worker thread per rank (a queue) and copies into ONE host arena
pinned once from the form -- no thread and no pin_memory() per rid/tensor.

Metal (IPC flip_first_work P>D + PDFLIP-BAR1 lane-time, D TP0, 5-6 seats):
* y4w 62357f2ba1 (TSE at the legs' start): D collect issue_ms median 452
  (y4l 247), P PP0 credit waits 1069 (y4l 457), legs 1840 (y4l 1522);
* y4x 72af174665 (TSE behind the legs, TSAL): collect issue 251, credit 603,
  but the expert rearm 375 ms (y4w 90, max 1401) and TAIL-READY waited
  back at 338 ms -- the staging's own cost collides wherever it runs.

Hermetic (CPU): the arena is a plain host tensor; what is pinned is the
serial worker, the arena's span accounting and release, the boot pin from
the form, and the per-wake instrument line.
"""

import ast
import inspect
import logging
import pathlib
import threading
import time
import types

import pytest
import torch

from flliper.srt.environ import envs
from flliper.srt.pdflip import tail_adopt as ta
from flliper.srt.pdflip import tail_handoff as th

REPO = pathlib.Path(__file__).resolve().parents[4]
LOGGER = "flliper.srt.pdflip.tail_adopt"


def _held():
    bf = str(torch.bfloat16)
    return ta.HeldShapes(fa={3: [ta.RowSpec([2, 8], bf), ta.RowSpec([2, 8], bf)]},
                         gdn={0: [ta.RowSpec([4, 16], bf), ta.RowSpec([3, 16], bf)]}, qsa_ratio=0)


@pytest.fixture
def rig(monkeypatch):
    ta._JOBS.clear()
    ta._AGREED.clear()
    ta.SKIP_PLANS.clear()
    ta.PENDING_INSTALLS.clear()
    ta._VERIFYING.clear()
    ta._STAGE_WORKER[0] = ta.StageWorker(seats=2, pin=False)
    state = {"order": [], "active": 0, "max_active": 0, "lock": threading.Lock()}
    monkeypatch.setattr(ta, "_candidate", lambda rid: rid.startswith("pdflip-"))
    monkeypatch.setattr(ta, "adopt_enabled", lambda: True)
    monkeypatch.setattr(th, "headers_for", lambda rid: [types.SimpleNamespace(rid=rid)])
    monkeypatch.setattr(th, "manifest_state", lambda headers: ("complete", 3, 3))
    monkeypatch.setattr(ta, "held_shapes", lambda kv, r2t: _held())

    def fake_stage_into(box, headers, held, check_digest, device):
        with state["lock"]:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.02)
        # the staging's copies go through the arena of this thread
        staged = ta._pin(torch.ones(2, 8, dtype=torch.bfloat16), torch.cuda.is_available())
        state["order"].append((headers[0].rid, staged))
        with state["lock"]:
            state["active"] -= 1
        box.append(types.SimpleNamespace(ok=True, end_ok=True, e1=True))

    monkeypatch.setattr(ta, "_stage_into", fake_stage_into)
    tree = types.SimpleNamespace(token_to_kv_pool_allocator=types.SimpleNamespace(get_kvcache=lambda: None),
                                 req_to_token_pool=None, page_size=4)
    yield state, tree
    for job in list(ta._JOBS.values()):
        if job.thread is not None:
            job.thread.join(5)
    ta._JOBS.clear()
    ta._AGREED.clear()
    ta._VERIFYING.clear()
    ta._STAGE_WORKER[0] = None


def _workers():
    return [t for t in threading.enumerate() if t.name.startswith("pdflip-tail-stage")]


def test_switch_default_off():
    assert envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.get() is False
    assert ta.stage_worker_enabled() is False


def test_one_worker_thread_stages_every_rid_in_order(rig):
    state, tree = rig
    rids = [f"pdflip-32-{i}" for i in range(48, 54)]
    with envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.override(True):
        for rid in rids:
            ta.stage(rid, tree)
        for rid in rids:
            ta._JOBS[rid].thread.join(5)
    assert [r for r, _t in state["order"]] == rids          # submission order
    assert state["max_active"] == 1                          # serial: never two at once
    names = {t.name for t in _workers()}
    assert names == {"pdflip-tail-stage-worker"}               # one thread, no thread per rid
    assert all(ta.vote_hold(rid) == 0 for rid in rids)       # staged: the hold does not wait
    arena = ta._STAGE_WORKER[0].arena
    for _rid, t in state["order"]:                           # the copy lives IN the arena
        base = arena.buf.data_ptr()
        assert base <= t.data_ptr() < base + arena.nbytes


def test_off_is_one_thread_per_rid(rig):
    state, tree = rig
    with envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.override(False):
        ta.stage("pdflip-2-5", tree)
        job = ta._JOBS["pdflip-2-5"]
        assert isinstance(job.thread, threading.Thread) and job.thread.name == "pdflip-tail-stage"
        job.thread.join(5)
    assert ta._STAGE_WORKER[0].arena is None


def test_arena_spans_first_fit_release_and_fallback():
    a = ta.PinArena(4 * ta.ARENA_ALIGN, pin=False)
    x = a.take("r1", (ta.ARENA_ALIGN // 2,), torch.bfloat16)      # one aligned span
    y = a.take("r2", (2, ta.ARENA_ALIGN), torch.uint8)            # two spans
    assert x is not None and y is not None and a.used() == 3 * ta.ARENA_ALIGN
    assert a.take("r3", (2, ta.ARENA_ALIGN), torch.uint8) is None  # no room: caller falls back
    assert a.release("r1") == ta.ARENA_ALIGN and a.release("r1") == 0
    assert a.release("r2") == 2 * ta.ARENA_ALIGN
    assert a.used() == 0 and a._free == [(0, 4 * ta.ARENA_ALIGN)]  # merged back to one run
    assert a.peak == 3 * ta.ARENA_ALIGN
    z = a.take("r4", (4 * ta.ARENA_ALIGN,), torch.uint8)          # the whole buffer again
    assert z is not None and z.data_ptr() == a.buf.data_ptr()


def test_pin_uses_the_arena_and_never_pin_memory_in_scope(monkeypatch):
    a = ta.PinArena(8 * ta.ARENA_ALIGN, pin=False)
    calls = []
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a_, **k: calls.append(1) or self)
    ta._TLS.arena, ta._TLS.rid = a, "pdflip-1-3"
    try:
        src = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        got = ta._pin(src, True)
        back = ta._readback_buffers({3: (src,)}, {}, {}, None, True)
    finally:
        ta._TLS.arena, ta._TLS.rid = None, None
    assert torch.equal(got, src) and got.data_ptr() != src.data_ptr()
    assert back["fa3"][0].shape == (4, 32) and back["fa3"][0].dtype == torch.uint8
    assert calls == []                                            # no pin_memory per tensor
    assert a.owners() == ["pdflip-1-3"]


def test_arena_full_falls_back_counted(monkeypatch):
    a = ta.PinArena(ta.ARENA_ALIGN, pin=False)
    before = ta._WORKER_STATS["fallback"]
    ta._TLS.arena, ta._TLS.rid = a, "pdflip-1-3"
    try:
        big = ta._pin(torch.zeros(4 * ta.ARENA_ALIGN, dtype=torch.uint8), False)
    finally:
        ta._TLS.arena, ta._TLS.rid = None, None
    assert big.numel() == 4 * ta.ARENA_ALIGN
    assert ta._WORKER_STATS["fallback"] == before + 1


def test_reclaim_gives_back_what_nothing_holds_and_keeps_the_rest(rig):
    state, tree = rig
    w = ta._STAGE_WORKER[0]
    w.prepare(_held(), 4, False)
    for rid in ("pdflip-a", "pdflip-b", "pdflip-c", "pdflip-d"):
        assert w.arena.take(rid, (8,), torch.uint8) is not None
    ta._AGREED["pdflip-a"] = object()                               # agreed, not yet admitted
    ta.SKIP_PLANS["pdflip-b"] = object()                            # admitted, END install pending
    ta._VERIFYING["pdflip-c"] = 1                                   # rows in flight to the device
    try:
        w.reclaim()
        assert sorted(w.arena.owners()) == ["pdflip-a", "pdflip-b", "pdflip-c"]   # pdflip-d: dropped -> back
        ta._release_verified("pdflip-c")                            # the verify thread saw it land
        assert "pdflip-c" not in w.arena.owners() and "pdflip-c" not in ta._VERIFYING
    finally:
        ta._AGREED.pop("pdflip-a", None)
        ta.SKIP_PLANS.pop("pdflip-b", None)


def test_finish_waits_the_copies_before_the_span_goes_back(rig, monkeypatch):
    state, tree = rig
    w = ta._STAGE_WORKER[0]
    w.prepare(_held(), 4, False)
    w.arena.take("pdflip-9-9", (16,), torch.uint8)
    spec = th.TailSpec(rid="pdflip-9-9", n_tokens=10, page_prefix=8, cut=10, key="k")
    inst = ta.Install(spec=spec, headers=[], fa={}, gdn={}, fa_dst={}, gdn_dst={},
                      rows=torch.zeros(2, dtype=torch.int64), groups=None, slot=None, t0=time.perf_counter())
    seen = []
    monkeypatch.setattr(ta, "_log_adopt", lambda *a, **k: seen.append(1))
    with envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.override(True):
        ta._finish(inst, False)
        for t in [t for t in threading.enumerate() if t.name == "pdflip-tail-verify"]:
            t.join(5)
    assert seen and "pdflip-9-9" not in w.arena.owners() and not ta._VERIFYING


def test_boot_pins_the_arena_from_the_form(rig, caplog):
    state, tree = rig
    held = _held()
    per = ta.rid_bytes(held, 64, verify=True)
    # K, V: 3 pages x 2x8 bf16; GDN twice (E1 + END); + one alignment per staged tensor (8); x2 readback
    raw = 2 * (3 * 64 * 16 * 2) + 2 * (4 * 16 * 2 + 3 * 16 * 2)
    assert per == 2 * (raw + 8 * ta.ARENA_ALIGN)
    runner = types.SimpleNamespace(token_to_kv_pool=None, req_to_token_pool=None)
    args = types.SimpleNamespace(max_running_requests=6, page_size=64)
    with envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.override(True), envs.FLLIPER_PDFLIP_TAIL_VERIFY.override(True), \
            caplog.at_level(logging.INFO, logger=LOGGER):
        n = ta.prepare_arena(runner, args)
        assert ta.prepare_arena(runner, args) == n                # pinned ONCE
    assert n == 6 * per
    lines = [r.getMessage() for r in caplog.records if ta.LEDGER_LINE in r.getMessage()]
    assert len(lines) == 1 and "seats=6" in lines[0]
    with envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.override(False):
        assert ta.prepare_arena(runner, args) is None


def test_one_instrument_line_per_drained_batch(rig, caplog):
    state, tree = rig
    with envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.override(True), caplog.at_level(logging.INFO, logger=LOGGER):
        for rid in ("pdflip-5-1", "pdflip-5-2", "pdflip-5-3"):
            ta.stage(rid, tree)
        for rid in ("pdflip-5-1", "pdflip-5-2", "pdflip-5-3"):
            ta._JOBS[rid].thread.join(5)
        deadline = time.time() + 5
        while time.time() < deadline and not any(ta.WORKER_LINE in r.getMessage() for r in caplog.records):
            time.sleep(0.01)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(ta.WORKER_LINE)]
    assert 1 <= len(lines) <= 3
    joined = " ".join(lines)
    for field in ("worker_issue_ms=", "read_ms=", "digest_ms=", "arena_used_mib=", "fallback=",
                  "digest=worker:pre-verdict"):
        assert field in joined
    assert sum(int(l.split("jobs=")[1].split()[0]) for l in lines) == 3


def test_the_part_reader_times_read_and_digest(tmp_path, monkeypatch):
    th.stage_clock_reset()
    assert th.stage_clock() == (0.0, 0.0)
    th._clock_add("digest_ms", time.perf_counter() - 0.002)
    r, d = th.stage_clock()
    assert r == 0.0 and d >= 1.0


def test_no_synchronize_on_the_scheduler_thread_side():
    """submit / reclaim / _pin / _live_rids run on the scheduler thread (or
    the staging thread): no device sync, no .item(), no collective."""
    forbidden = {"synchronize", "item", "all_reduce", "barrier", "all_gather", "tolist"}
    for fn in (ta.StageWorker.submit, ta.StageWorker.reclaim, ta._live_rids, ta._pin, ta._pin_empty,
               ta.PinArena.take, ta.PinArena.release):
        tree = ast.parse(inspect.getsource(fn).lstrip() if not inspect.getsource(fn).startswith(" ")
                         else "class _X:\n" + inspect.getsource(fn))
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert not (calls & forbidden), (fn.__qualname__, calls & forbidden)


def test_the_d_worker_pins_it_at_boot():
    """y5a 0110f8e132: at the spec worker's init the pools did not exist yet
    (``boot pin n/a (AttributeError: 'ModelRunner' object has no attribute
    'token_to_kv_pool')`` on TP0-2, the first staging pinned 1386 MiB inside
    the first flip, pin_ms=757.4 on TP0). The pin rides the pool allocation."""
    spec = (REPO / "python/flliper/srt/speculative/eagle_worker_v2.py").read_text()
    assert "prepare_arena" not in spec
    src = (REPO / "python/flliper/srt/managers/tp_worker.py").read_text()
    i_def = src.index("    def alloc_memory_pool(")
    i_alloc = src.index("self.model_runner.alloc_memory_pool(memory_pool_config)", i_def)
    i_pin = src.index("_tail_adopt.prepare_arena(self.model_runner, self.server_args)", i_alloc)
    i_next = src.index("    def init_attention_backends(", i_def)
    assert i_def < i_alloc < i_pin < i_next
    assert "if not self.is_draft_worker:" in src[i_alloc:i_pin]


def test_prepare_arena_after_the_pools_pins_from_their_shapes(monkeypatch):
    runner = types.SimpleNamespace(token_to_kv_pool="kv", req_to_token_pool="r2t")
    seen = []
    monkeypatch.setattr(ta, "held_shapes", lambda kv, r2t: seen.append((kv, r2t)) or _held())
    monkeypatch.setattr(ta, "adopt_enabled", lambda: True)
    ta._STAGE_WORKER[0] = ta.StageWorker(seats=2, pin=False)
    try:
        with envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.override(True):
            n = ta.prepare_arena(runner, types.SimpleNamespace(max_running_requests=6, page_size=64))
        assert seen == [("kv", "r2t")] and n == 6 * ta.rid_bytes(_held(), 64, ta.verify_enabled())
    finally:
        ta._STAGE_WORKER[0] = None
