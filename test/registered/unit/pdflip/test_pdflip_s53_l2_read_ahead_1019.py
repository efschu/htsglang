"""S53 (19.10.) L2-READ-AHEAD: the queue head the front's pool plan (H91c) holds is read L3 -> L2 on P while its
predecessor computes.

rc14 boot ..._92dd662831_1019_010742, P epoch 05:18:55-05:20:26Z: pdflip-0-22 waited in the front from 05:18:53
(``P-PHASE epoch=1 ... pool_holds=290``), left it at 05:19:10.27 when pdflip-0-21 came back, and read
``pages=2087 l3fill_pages=2087 total_ms=2403`` from the disk while PP0's card idled (``#PGAP gpu_gap_ms=5689.5``).
pdflip-1-27, whose pages were in L2, read ``pages=2026 l3fill_pages=82 total_ms=216``.

Driven for real: the front's ``_p_drain_pool`` and ``Front._maybe_l2_read_ahead``; P's ``build_ple_prefetch_hint``
and ``Scheduler.handle_ple_prefetch_hint``; the worker of ``pdflip/l2_read_ahead.py``; the fill against the real C
arena (gcc) with the real ``HiCacheFile.arena_fill_from_disk`` over a temp L3 directory.
"""
from __future__ import annotations

import asyncio
import collections
import os
import shutil
import types
from array import array

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.pdflip import front  # noqa: E402
from flliper.srt.pdflip import l2_read_ahead as ra  # noqa: E402
from flliper.srt.pdflip import phase_policy  # noqa: E402

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")


# ============================================================================ front: the pool plan's hold


def _drain(items, costs, budget, *, limit=4, max_dispatch=0, on_hold=None, finish_after=0.05):
    """The real ``_p_drain_pool``; every leg finishes ``finish_after`` s after its dispatch."""
    held = []

    async def main():
        q = collections.deque(items)
        order = []

        async def one(it):
            order.append(it)
            await asyncio.sleep(finish_after)
            return it

        await front._p_drain_pool(
            q, limit, one, lambda it: None, lambda: True, max_dispatch=max_dispatch,
            cost=lambda it: costs[it], budget=budget, stats={},
            on_pool_hold=(on_hold if on_hold is not None else held.append))
        return order

    order = asyncio.run(main())
    return order, held


def test_metal_shape_the_held_head_is_named_to_the_hook():
    """pdflip-0-21 (199014) in flight, pdflip-0-22 (152764) does not fit P's 262144: the hook gets 0-22 -- RED on
    the base (no hook: TypeError on the keyword)."""
    order, held = _drain(["0-21", "0-22"], {"0-21": 199014, "0-22": 152764}, 262144)
    assert order == ["0-21", "0-22"]
    assert held and set(held) == {"0-22"}


def test_a_head_that_fits_is_never_held():
    order, held = _drain(["a", "b"], {"a": 100000, "b": 100000}, 262144)
    assert order == ["a", "b"] and held == []


def test_the_phase_cap_is_no_pool_hold():
    """Danger direction (a read-ahead for a request that waits for the NEXT P phase): the cap stops the drain, the
    hook stays silent."""
    order, held = _drain(["a", "b"], {"a": 1000, "b": 1000}, 262144, max_dispatch=1)
    assert order == ["a"] and held == []


def test_only_the_head_is_named_one_successor():
    """Three waiting behind one big leg: only the head is held (the drain breaks there) -- never the second."""
    costs = {"big": 200000, "h1": 150000, "h2": 150000, "h3": 150000}
    _order, held = _drain(["big", "h1", "h2", "h3"], costs, 262144, limit=4)
    first_hold = held[0]
    assert first_hold == "h1"
    # each item is named only while it is the head
    seen_heads = list(dict.fromkeys(held))
    assert seen_heads == ["h1", "h2", "h3"]


def test_a_raising_hook_never_stops_the_drain():
    def boom(_it):
        raise RuntimeError("x")

    order, _ = _drain(["a", "b"], {"a": 200000, "b": 200000}, 262144, on_hold=boom)
    assert order == ["a", "b"]


class _FrontStandIn:
    def __init__(self, awake="P"):
        self.awake = awake
        self.groups = {"P": object()}
        self.session = object()
        self.counters = collections.Counter()
        self.posts = []

    async def rpc(self, g, path, body, timeout):
        self.posts.append((path, body))
        return 200, {}


def _pending(rid="pdflip-0-22", span=151040, known=True, skip=False):
    return front.Pending(rid=rid, path="/v1/messages", payload={"rid": rid, "max_tokens": 1, "stream": True},
                         text="", t_arrive=0.0, fut=None, span_known=known, store_span_est=span, skip_leg1=skip)


def _hint(fs, p):
    async def main():
        front.Front._maybe_l2_read_ahead(fs, p)
        await asyncio.sleep(0)
        for t in list(fs.__dict__.get("_ple_hint_tasks", ())):
            await t

    asyncio.run(main())


def test_front_hints_the_held_head_once_with_read_ahead_and_no_ple():
    fs, p = _FrontStandIn(), _pending()
    _hint(fs, p)
    _hint(fs, p)
    assert len(fs.posts) == 1, "once per request"
    path, body = fs.posts[0]
    assert path == "/pdflip/ple_prefetch_hint"
    assert body["read_ahead"] is True and body["ple"] is False and body["start_hint"] == 151040
    assert body["payload"]["rid"] == "pdflip-0-22" and "stream" not in body["payload"]


@pytest.mark.parametrize("kw,awake", [
    ({}, "D"),                       # P asleep: the H43 PLE hint's case, not this one
    ({"known": False}, "P"),         # no store span priced: nothing known to read
    ({"span": 1024}, "P"),           # below MIN_SPAN_TOKENS
    ({"skip": True}, "P"),           # no leg 1 on P
])
def test_front_sends_no_hint_outside_its_case(kw, awake):
    fs = _FrontStandIn(awake)
    _hint(fs, _pending(**kw))
    assert fs.posts == []


def test_front_switch_off_wires_no_hook(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_L2_READ_AHEAD", "0")
    assert ra.enabled() is False
    monkeypatch.setenv("FLLIPER_PDFLIP_L2_READ_AHEAD", "1")
    assert ra.enabled() is True


def test_the_drain_call_site_wires_the_hook_behind_the_switch():
    import inspect

    src = inspect.getsource(front)
    assert "on_pool_hold=((lambda p: Front._maybe_l2_read_ahead(self, p))" in src
    assert "if _l2ra.enabled() else None)," in src


# ============================================================================ P: the hint and the scheduler


def test_the_p_hint_carries_read_ahead_and_ple():
    from flliper.srt.pdflip.ple_admit_hint import build_ple_prefetch_hint

    body = {"path": "/generate", "payload": {"rid": "r1", "input_ids": [5, 6, 7]}, "start_hint": 8192,
            "read_ahead": True, "ple": False}
    h = asyncio.run(build_ple_prefetch_hint(body, serving_chat=None, serving_completion=None, encode=None))
    assert (h.rid, h.input_ids, h.start_hint, h.read_ahead, h.ple) == ("r1", [5, 6, 7], 8192, True, False)
    old = {"path": "/generate", "payload": {"rid": "r2", "input_ids": [1]}}
    h2 = asyncio.run(build_ple_prefetch_hint(old, serving_chat=None, serving_completion=None, encode=None))
    assert (h2.read_ahead, h2.ple) == (False, True), "an H43 hint is unchanged"


def _sched(pp_rank=0, dormant=False, **kw):
    kw.setdefault("chunked_req", None)
    return types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=pp_rank, tp_rank=0), pdflip_dormant=dormant,
                                 chunked_prefill_size=16384, page_size=64, **kw)


def test_scheduler_read_ahead_hint_skips_the_ple_admission(monkeypatch):
    from flliper.srt.managers.scheduler import Scheduler
    from flliper.srt.models import qwen4_exp_ple_admit as ple
    from flliper.srt.pdflip import short_read as sr

    admitted, noted, queued = [], [], []
    monkeypatch.setattr(ple, "admit_ple_hint", lambda rid, *a, **k: admitted.append(rid))
    monkeypatch.setattr(sr, "note_hint", lambda rid, s: noted.append(rid))
    monkeypatch.setattr(ra, "on_hint", lambda s, r: queued.append(r.rid))
    req = types.SimpleNamespace(rid="pdflip-0-22", input_ids=[1] * 9000, start_hint=8192, read_ahead=True, ple=False)
    Scheduler.handle_ple_prefetch_hint(_sched(), req)
    assert queued == ["pdflip-0-22"]
    assert admitted == [] and noted == [], "an awake P's PLE ring is the running request's"
    old = types.SimpleNamespace(rid="pdflip-0-21", input_ids=[1, 2], start_hint=64)
    Scheduler.handle_ple_prefetch_hint(_sched(dormant=True), old)
    assert admitted == ["pdflip-0-21"] and noted == ["pdflip-0-21"] and queued == ["pdflip-0-22"]


def test_on_hint_is_pp0_only_and_never_on_a_dormant_group(monkeypatch):
    made = []

    class _RA:
        def __init__(self, s):
            made.append(s)

        def submit(self, rid, ids, span):
            return "queued"

    monkeypatch.setattr(ra, "ReadAhead", _RA)
    req = types.SimpleNamespace(rid="r", input_ids=[1] * 9000, start_hint=8192)
    assert ra.on_hint(_sched(pp_rank=1), req) is None, "a follower never fills (the arena is shared)"
    assert ra.on_hint(_sched(dormant=True), req) is None
    assert ra.on_hint(_sched(), types.SimpleNamespace(rid="r", input_ids=[1] * 9000, start_hint=1024)) is None
    assert ra.on_hint(_sched(), req) == "queued" and len(made) == 1
    monkeypatch.setenv("FLLIPER_PDFLIP_L2_READ_AHEAD", "0")
    assert ra.on_hint(_sched(), req) is None


def test_on_hint_never_raises(monkeypatch):
    class _RA:
        def __init__(self, s):
            raise RuntimeError("no thread")

    monkeypatch.setattr(ra, "ReadAhead", _RA)
    assert ra.on_hint(_sched(), types.SimpleNamespace(rid="r", input_ids=[1] * 9000, start_hint=8192)) is None


# ============================================================================ P: the worker (one successor, lead)


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _worker(sched, fills):
    clk = _Clock()

    def fill(s, ids, span):
        fills.append(span)
        return {"verdict": "done", "pages": span // 64}

    return ra.ReadAhead(sched, fill=fill, now=clk, poll_s=0.0, start=False), clk


def test_one_successor_a_newer_hint_replaces_the_waiting_one():
    fills = []
    w, _ = _worker(_sched(), fills)
    assert w.submit("a", [1], 8192) == "queued"
    assert w.submit("b", [1], 16384) == "replaced"
    assert w.submit("b", [1], 16384) == "dup"
    job = w._take()
    assert job["rid"] == "b" and w.waiting is None
    assert w.run_job(job) == "done" and fills == [16384]


def _chunked(total, filled):
    return types.SimpleNamespace(origin_input_ids=[0] * total, fill_ids=[0] * filled)


def test_metal_shape_the_read_waits_for_the_lead():
    """pdflip-0-21 (199014 tokens, 163584 read): before its first chunk 35430 tokens are left -- more than two
    chunks (32768): the read-ahead of pdflip-0-22 waits; after the first chunk 19046 are left: it starts."""
    fills = []
    s = _sched(chunked_req=_chunked(199014, 163584))
    w, _ = _worker(s, fills)
    w.submit("pdflip-0-22", [1], 151040)
    job = w._take()
    assert w.step(job) is None and fills == [], "35430 left > two chunks: wait"
    s.chunked_req = _chunked(199014, 179968)
    assert w.step(job) == "done" and fills == [151040]


def test_no_chunked_request_left_starts_at_once():
    fills = []
    w, _ = _worker(_sched(), fills)
    w.submit("r", [1], 8192)
    assert w.step(w._take()) == "done"


def test_the_lead_switch(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_L2_READ_AHEAD_LEAD_TOKENS", "4096")
    assert ra.lead_tokens(_sched()) == 4096
    monkeypatch.delenv("FLLIPER_PDFLIP_L2_READ_AHEAD_LEAD_TOKENS")
    assert ra.lead_tokens(_sched()) == 32768


@pytest.mark.parametrize("why", ["dormant", "stale", "read_started"])
def test_a_job_is_dropped_when_it_lost_its_point(why):
    fills = []
    s = _sched(tree_cache=types.SimpleNamespace(ongoing_prefetch={}))
    w, clk = _worker(s, fills)
    w.submit("r", [1], 8192)
    job = w._take()
    if why == "dormant":
        s.pdflip_dormant = True         # P flipped away
    elif why == "stale":
        clk.t += ra.MAX_AGE_S + 1
    else:
        s.tree_cache.ongoing_prefetch["r"] = object()   # its own store read began
    assert w.run_job(job) == "skip:" + why and fills == []


def test_a_waiting_job_yields_to_a_newer_hint():
    fills = []
    s = _sched(chunked_req=_chunked(300000, 16384))
    w, _ = _worker(s, fills)
    w.submit("old", [1], 8192)
    job = w._take()
    w.submit("new", [1], 8192)
    assert w.run_job(job) == "superseded" and fills == []
    assert w.waiting["rid"] == "new"


def test_a_raising_fill_is_contained():
    def fill(s, ids, span):
        raise OSError("disk")

    w = ra.ReadAhead(_sched(), fill=fill, poll_s=0.0, start=False)
    w.submit("r", [1], 8192)
    assert w.run_job(w._take()) == "error" and w.running is None


def test_the_thread_runs_a_job_end_to_end():
    import threading

    ran = threading.Event()

    def fill(s, ids, span):
        ran.set()
        return {"verdict": "done"}

    w = ra.ReadAhead(_sched(), fill=fill, poll_s=0.01)
    try:
        w.submit("r", [1], 8192)
        assert ran.wait(5)
    finally:
        w.stop()


# ============================================================================ P: the keys are the fetch's keys


@pytest.mark.parametrize("bigram", [True, False])
def test_span_hashes_are_the_fetchs_keys_cut_at_the_span(bigram):
    from flliper.srt.managers import pdflip_store_told as st
    from flliper.srt.mem_cache.utils import get_hash_str

    ids = list(range(1000, 1000 + 64 * 10 + 7))
    s = types.SimpleNamespace(page_size=64, tree_cache=types.SimpleNamespace(is_eagle=bigram))
    got = ra.span_hashes(s, ids, 64 * 6 + 5)            # a span off the page grain: cut down to 6 pages
    key, _ = st._probe_key(s, types.SimpleNamespace(extra_key=None), array("q", ids), 64 * 6)
    want = list(get_hash_str(key, None, page_size=64))[:6]
    assert got == want and len(got) == 6
    other = ra.span_hashes(types.SimpleNamespace(page_size=64, tree_cache=types.SimpleNamespace(is_eagle=not bigram)),
                           ids, 64 * 6)
    assert other != got, "bigram and unigram chains differ: the tree's flag decides"


# ============================================================================ P: the fill against the real C arena

SB = 256
SLOTS = 16


@pytest.fixture
def arena_env(tmp_path, monkeypatch):
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile
    from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_QUEUE_REFS", "1")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path / "arena"))
    (tmp_path / "l3").mkdir()
    root = str(tmp_path / "l3")

    def path(stem):
        return os.path.join(root, stem + ".bin")

    be = object.__new__(HiCacheFile)
    be._stat_stems = lambda stems: {s: os.path.getsize(path(s)) for s in stems if os.path.exists(path(s))}
    be._sharded_path = path
    be._existing_path = path
    be._ensure_shard_dir = lambda p: os.makedirs(os.path.dirname(p), exist_ok=True)
    be._evictor = types.SimpleNamespace(reserve=lambda *a, **k: True, commit=lambda s: None, abort=lambda s: None)
    be._key_geom = {"is_mla_model": False}
    arena = ShmArena(str(tmp_path / "arena-kv.bin"), SB, SLOTS)

    def to_l3(stems, fill):
        for i, stem in enumerate(stems):
            with open(path(stem), "wb") as f:
                f.write(bytes([fill + i]) * SB)

    def publish(stems, fill):
        for i, stem in enumerate(stems):
            (sl, st, g), = arena.claim_slots([stem], [SB])
            assert st == 0
            arena.slot_view(sl, SB)[:] = bytes([fill + i]) * SB
            assert arena.complete_slots([sl], [g], [(0, SB)]) == [1]

    yield types.SimpleNamespace(be=be, arena=arena, to_l3=to_l3, publish=publish, path=path)
    arena.close()


def _states(arena, stems):
    return [int(st) for _, st in arena.find_slots(stems)]


def _slots(arena, stems):
    return [int(sl) for sl, _ in arena.find_slots(stems)]


@needs_gcc
def test_metal_shape_the_disk_prefix_lands_complete_and_unreferenced(arena_env):
    """pdflip-0-22's 2087 pages (8 here) only on L3: after the read-ahead every page is COMPLETE in L2 with the
    disk's bytes and NO reader reference (danger: a pinned read-ahead withholds slots from the running publish)."""
    e = arena_env
    stems = [f"p{i}" for i in range(8)]
    e.to_l3(stems, 0x20)
    res = ra.fill_arena(e.be, e.arena, stems, SB)
    assert res == {"pages": 8, "resident": 0, "asked": 8, "landed": 8}
    assert _states(e.arena, stems) == [2] * 8
    assert e.arena.slot_refs(_slots(e.arena, stems)) == [0] * 8, "no reference taken"
    for sl, stem in zip(_slots(e.arena, stems), stems):
        assert bytes(e.arena.slot_view(sl, SB)) == open(e.path(stem), "rb").read()


@needs_gcc
def test_pages_already_in_l2_are_not_asked(arena_env):
    """LMCache #3540's lesson: no I/O for what the faster tier holds -- 5 of 8 in L2, 3 asked."""
    e = arena_env
    stems = [f"p{i}" for i in range(8)]
    e.to_l3(stems, 0x20)
    e.publish(stems[:5], 0x20)
    res = ra.fill_arena(e.be, e.arena, stems, SB)
    assert res["resident"] == 5 and res["asked"] == 3 and res["landed"] == 3
    assert _states(e.arena, stems) == [2] * 8


@needs_gcc
def test_the_walk_ends_at_the_first_page_neither_in_l2_nor_on_disk(arena_env):
    e = arena_env
    stems = [f"p{i}" for i in range(8)]
    e.to_l3(stems[:3] + stems[4:], 0x20)        # p3 is a hole
    res = ra.fill_arena(e.be, e.arena, stems, SB)
    assert res["landed"] == 3
    assert _states(e.arena, stems[:3]) == [2, 2, 2]
    assert _states(e.arena, stems[3:]) == [0] * 5, "nothing past the hole: the real read ends there too"


@needs_gcc
def test_a_full_arena_ends_the_read_ahead_without_a_room_wait(arena_env, monkeypatch):
    """Every slot COMPLETE and referenced (a running read): the read-ahead gets no room and ends -- no READ-ROOM post
    (danger: a room wait would ask the trees to spill for a request that is not even on P yet)."""
    from flliper.srt.mem_cache import hicache_storage as hs

    e = arena_env
    busy = [f"b{i}" for i in range(SLOTS)]
    e.publish(busy, 0x01)
    e.to_l3(busy, 0x01)
    e.arena.ref_slots(_slots(e.arena, busy), +1)
    posts = []
    monkeypatch.setattr(hs._read_room, "post", lambda *a, **k: posts.append(a))
    stems = [f"p{i}" for i in range(4)]
    e.to_l3(stems, 0x20)
    res = ra.fill_arena(e.be, e.arena, stems, SB)
    assert res["landed"] == 0 and posts == []
    assert _states(e.arena, busy) == [2] * SLOTS, "the running read's pages stay"


@needs_gcc
def test_the_read_ahead_never_evicts_its_own_resident_pages(arena_env):
    """The arena full of unreferenced pages: 4 of the span resident, 4 on disk -- the fill's room comes from the
    OTHER pages (fill_protect_scope over the whole span), never from the span's resident head."""
    e = arena_env
    stems = [f"p{i}" for i in range(8)]
    other = [f"o{i}" for i in range(SLOTS - 4)]
    e.to_l3(stems + other, 0x20)
    e.publish(stems[:4], 0x20)
    e.publish(other, 0x60)
    res = ra.fill_arena(e.be, e.arena, stems, SB)
    assert res["resident"] == 4
    assert _states(e.arena, stems[:4]) == [2] * 4, "the span's own L2 head stays"
    assert res["landed"] == 4 and _states(e.arena, stems) == [2] * 8


@needs_gcc
def test_fill_span_reads_kv_and_qsa_through_the_shared_arenas(arena_env, monkeypatch):
    """The scheduler path: KV stems through the bound pool's arena (``pdflip_suffixed_stems``), QSA stems
    ``{hash}.qsa_indexer`` through ``_arena_for(qsa width)``; the Mamba anchor is not read ahead."""
    from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

    e = arena_env
    qarena = ShmArena(os.path.join(os.path.dirname(e.path("x")), "..", "arena-qsa.bin"), 64, SLOTS)
    try:
        e.be._suffix_for_key = lambda k: (".s", True)
        e.be._get_suffixed_key = lambda k: k + ".s"
        e.be.canonical_qsa_page = types.SimpleNamespace(total_bytes=64)
        e.be._arena_for = lambda tb: qarena if int(tb) == 64 else None
        pool = types.SimpleNamespace(arena_read=True, arena=e.arena, _page_bytes=SB,
                                     ensure_bound=lambda be, role: True)
        sched = types.SimpleNamespace(page_size=4, tree_cache=types.SimpleNamespace(
            is_eagle=True, cache_controller=types.SimpleNamespace(storage_backend=e.be, mem_pool_host=pool)))
        ids = list(range(50, 50 + 4 * 6 + 1))
        hashes = ra.span_hashes(sched, ids, 4 * 6)
        kv = [h + ".s" for h in hashes]
        qsa = [h + ".qsa_indexer.s" for h in hashes]
        e.to_l3(kv, 0x30)
        for i, st in enumerate(qsa):
            with open(e.path(st), "wb") as f:
                f.write(bytes([0x70 + i]) * 64)
        res = ra.fill_span(sched, ids, 4 * 6)
        assert res["verdict"] == "done" and res["pages"] == 6
        assert res["kv"]["landed"] == 6 and res["qsa"]["landed"] == 6
        assert _states(e.arena, kv) == [2] * 6 and _states(qarena, qsa) == [2] * 6
        assert e.arena.slot_refs(_slots(e.arena, kv)) == [0] * 6
        assert qarena.slot_refs(_slots(qarena, qsa)) == [0] * 6
    finally:
        qarena.close()


def test_fill_span_without_an_arena_pool_does_nothing():
    pool = types.SimpleNamespace(arena_read=False)
    sched = types.SimpleNamespace(page_size=64, tree_cache=types.SimpleNamespace(
        is_eagle=True, cache_controller=types.SimpleNamespace(storage_backend=object(), mem_pool_host=pool)))
    assert ra.fill_span(sched, [1] * 9000, 8192) == {"verdict": "no_arena"}


def test_phase_policy_unchanged_the_device_plan_stays():
    """H91c stays: the read-ahead never changes which request P admits."""
    assert phase_policy.p_overlap_admits(199014, 1, 152764, 262144) is False
    assert phase_policy.p_overlap_admits(0, 0, 999999, 262144) is True
