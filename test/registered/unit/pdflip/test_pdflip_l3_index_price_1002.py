# SPDX-License-Identifier: Apache-2.0
"""L3-INDEX PRICE (02.10.): an arrival's X credit is the page-granular store
prefix of its exact ids, asked with the key P/D read with.

y7d (9bdfe50185), front log ..._1002_085437.front.log:
  * pdflip-0-2 (08:57:43): priced 115213 ``presence_src=none``, LONG, flip pair;
    P hit 111232 of 111256 from the PERSISTENT L3 (earlier boots);
  * pdflip-2-19 (09:01:20, tokenizer ready): ``X-EXACT-PRICE pending=30076
    tokens=30076 credit=0 src=none``, LONG, flip pair; P hit 30016 of 30076 --
    a page prefix of an earlier, longer request of this boot, no END-ANCHOR.

The store here is the real thing in small: a real #1459 shared stem index
(``l3_index.L3Index``, the arena.c table), stems written with the radix tree's
own page-hash function (``compute_node_hash_values`` over bigram RadixKeys,
chained node to node), the store's ``L3_SUFFIXES`` records and shard files.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import time
import types
from array import array

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from flliper.srt.mem_cache.storage.file.l3_index import L3Index  # noqa: E402
from flliper.srt.mem_cache.utils import compute_node_hash_values  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import front_store as FS  # noqa: E402
from flliper.srt.pdflip.front_tokens import Count, TokenSpans  # noqa: E402

PAGE = 64
SFX = "_Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist_898fe1bf454ff7c1"
X = 4855


def _ids(n, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.integers(1, 240000, size=n, dtype=np.int64).astype(np.int32)
    out[0] = 248045  # <|im_start|>
    return out


def _tree_page_hashes(ids, split=640):
    """The page keys as the radix tree writes them to the store: two nodes,
    bigram RadixKeys, the child chained on the parent's last page hash."""
    raw = array("q", [int(t) for t in ids])
    parent = types.SimpleNamespace(key=RadixKey(raw[: split + 1], None, is_bigram=True),
                                   parent=None, hash_value=None)
    parent.hash_value = compute_node_hash_values(parent, PAGE)
    child = types.SimpleNamespace(key=RadixKey(raw[split:], None, is_bigram=True),
                                  parent=parent, hash_value=None)
    hv = parent.hash_value + compute_node_hash_values(child, PAGE)
    full = (len(ids) - 1) // PAGE
    return hv[:full]


class _Store:
    """An L3 store directory + the shared stem index a rank opened."""

    def __init__(self, tmp, cap=1 << 16):
        self.dir = os.path.join(tmp, "l3-nextflash")
        os.makedirs(self.dir)
        self.arena = os.path.join(tmp, "pdflip-arena-t")
        for g, extra in (("P", "_0_1_3_0"), ("D", "_0_3")):
            with open(os.path.join(self.dir, f"L3_SUFFIXES.{g}.json"), "w") as f:
                json.dump({"group": g, "suffixes": [SFX + extra, SFX]}, f)
        os.makedirs(FS.l3_index_path(self.arena).rsplit("/", 1)[0])
        self.index = L3Index(FS.l3_index_path(self.arena), cap=cap)  # the rank creates it

    def write(self, hashes, kv=None, mamba=(), qsa=None):
        """KV (+ QSA) pages ``kv`` (default: all) and mamba blobs at ``mamba``."""
        kv = range(len(hashes)) if kv is None else kv
        qsa = kv if qsa is None else qsa
        classes = [[f"{hashes[i]}{SFX}" for i in kv],
                   [f"{hashes[i]}.qsa_indexer{SFX}" for i in qsa],
                   [f"{hashes[i]}.mamba{SFX}" for i in mamba]]
        for stems in classes:
            self.index.add(stems)
            for s in stems[:16]:  # a few real page files, for the component census
                d = os.path.join(self.dir, s[:2])
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, s + ".bin"), "wb").close()

    def probe(self, hybrid=True):
        probe, why = FS.open_store_presence(self.dir, self.arena, PAGE, True, hybrid)
        assert probe is not None, why
        return probe


@pytest.fixture
def store(tmp_path):
    return _Store(str(tmp_path))


# ---- the probe ------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_early_flip(monkeypatch):
    # EARLY-FLIP (02.10.) would begin a D->P flip for these big arrivals on an
    # idle D; this file tests the pricing in isolation, on the path without it
    monkeypatch.setenv("FLLIPER_PDFLIP_EARLY_FLIP", "0")

def test_the_front_keys_pages_exactly_as_the_tree_writes_them():
    ids = _ids(5000)
    assert FS.bigram_page_hasher(ids, PAGE, True) == _tree_page_hashes(ids)
    assert len(_tree_page_hashes(ids)) == (5000 - 1) // PAGE


def test_pdflip_0_2_an_earlier_boots_111k_prefix_is_its_depth(store):
    prev = _ids(111256)
    h = _tree_page_hashes(prev)
    store.write(h, mamba=[len(h) - 1])  # P's END-ANCHOR of an earlier boot
    d = store.probe().depth(_ids(111256))
    assert (d.tokens, d.pages, d.kv_pages) == (111232, 1738, 1738)


def test_pdflip_2_19_a_page_prefix_of_a_longer_request_is_credited(store):
    longer = _ids(52000, seed=3)
    h = _tree_page_hashes(longer)
    # P's anchors of the longer prompt: its end and the multi-anchor tail at 30016
    store.write(h, mamba=[468, len(h) - 1])
    cur = np.concatenate([longer[:30040], _ids(36, seed=9)])  # 30076 tokens
    d = store.probe().depth(cur)
    assert d.kv_pages == 469 and d.tokens == 30016, "no END-ANCHOR, a page prefix"
    assert 30076 - d.tokens == 60 <= X


def test_the_credit_ends_at_the_deepest_anchor_on_the_present_path(store):
    p = _ids(20000, seed=4)
    h = _tree_page_hashes(p)
    store.write(h, mamba=[99, 199])
    assert store.probe().depth(p).tokens == 200 * PAGE, "KV runs to the end, anchors at 100/200"
    store.index.remove([f"{h[199]}.mamba{SFX}"])
    assert store.probe().depth(p).tokens == 100 * PAGE, "anchor 200 evicted: the one below"


def test_a_kv_gap_and_a_missing_qsa_page_cut_the_run(store):
    p = _ids(20000, seed=5)
    h = _tree_page_hashes(p)
    store.write(h, kv=list(range(120)) + list(range(121, len(h))), mamba=[99, 200, len(h) - 1])
    assert store.probe().depth(p).tokens == 100 * PAGE, "KV page 120 missing: anchor 99"


def test_qsa_is_an_all_pages_pool(tmp_path):
    st = _Store(str(tmp_path))
    p = _ids(20000, seed=6)
    h = _tree_page_hashes(p)
    st.write(h, qsa=range(50), mamba=[30, 60, len(h) - 1])
    pr = st.probe()
    assert pr.all_pages == ("qsa_indexer",) and pr.trailing == ("mamba",)
    assert pr.depth(p).tokens == 31 * PAGE


def test_a_hybrid_group_without_any_anchor_gets_no_credit(store):
    p = _ids(9000, seed=7)
    store.write(_tree_page_hashes(p), mamba=[])
    assert store.probe(hybrid=True).depth(p).tokens == 0


def test_the_front_never_creates_the_index(tmp_path):
    d = os.path.join(str(tmp_path), "s")
    os.makedirs(d)
    with open(os.path.join(d, "L3_SUFFIXES.P.json"), "w") as f:
        json.dump({"suffixes": [SFX]}, f)
    arena = os.path.join(str(tmp_path), "pdflip-arena-x")
    probe, why = FS.open_store_presence(d, arena, PAGE, True, True)
    assert probe is not None and probe.index is None and "l3_index=absent" in why
    assert probe.depth(_ids(3000)).tokens == 0
    assert not os.path.exists(FS.l3_index_path(arena)), "joined only, never created"
    assert not os.path.exists(arena)


def test_no_shared_suffix_no_credit(tmp_path):
    d = os.path.join(str(tmp_path), "s")
    os.makedirs(d)
    for g, s in (("P", SFX + "_0_1_3_0"), ("D", SFX + "_0_3")):
        with open(os.path.join(d, f"L3_SUFFIXES.{g}.json"), "w") as f:
            json.dump({"suffixes": [s]}, f)
    assert FS.shared_suffix(d)[0] is None


# ---- the front: priced at arrival -------------------------------------------------------

class _Tok:
    state = "ready"
    why = ""

    def __init__(self, ids):
        self.ids = ids
        self.m = {}
        self.executor = None

    def ids_for(self, text):
        return self.m.get(text)

    def remember(self, text, ids):
        self.m[text] = ids

    def count(self, path, payload):
        return Count(n=int(self.ids.size), ids=self.ids, ms=1.0, reused=0, encoded=int(self.ids.size))


def _front(probe, ids, awake="D"):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = 0
    f.awake = awake
    f.state = "serving"
    f.tp_prefill_max_tokens = X
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = _Tok(ids)
    f._x_exact_rid = collections.OrderedDict()
    f.store_probe = probe
    return f


def test_front_prices_a_boot_start_request_short_on_the_l3_prefix(store, caplog):
    caplog.set_level(logging.INFO)
    prev = _ids(111256)
    h = _tree_page_hashes(prev)
    store.write(h, mamba=[len(h) - 1])
    f = _front(store.probe(), _ids(111256))
    xx = asyncio.run(f._x_exact_price("pdflip-0-2", "/v1/messages", {"m": 1}, "0-2", 115213, 115213))
    assert (xx.pending, xx.credit, xx.src) == (24, 111232, "l3_index")
    assert xx.pending <= X, "SHORT"
    assert any(m.startswith("PDFLIP X-EXACT-PRICE rid=pdflip-0-2 pending=24 tokens=111256 credit=111232 "
                            "src=l3_index") for m in caplog.messages)
    assert any("PDFLIP L3-INDEX-PRESENCE rid=pdflip-0-2 tier=l3_index depth=111232" in m
               for m in caplog.messages)
    assert f.counters["l3_index_credit"] == 1


def test_front_prices_pdflip_2_19_short(store, caplog):
    caplog.set_level(logging.INFO)
    longer = _ids(52000, seed=3)
    h = _tree_page_hashes(longer)
    store.write(h, mamba=[468, len(h) - 1])
    cur = np.concatenate([longer[:30040], _ids(36, seed=9)])
    f = _front(store.probe(), cur)
    xx = asyncio.run(f._x_exact_price("pdflip-2-19", "/v1/messages", {"m": 1}, "2-19", 30076, 30076))
    assert (xx.pending, xx.credit, xx.src) == (60, 30016, "l3_index")


def test_a_queued_reprice_keeps_the_store_credit(store):
    prev = _ids(111256)
    h = _tree_page_hashes(prev)
    store.write(h, mamba=[len(h) - 1])
    ids = _ids(111256)
    f = _front(store.probe(), ids)
    asyncio.run(f._x_exact_price("pdflip-0-2", "/v1/messages", {"m": 1}, "0-2", 115213, 115213))
    p = F.Pending("pdflip-0-2", "/v1/messages", {}, "0-2", time.time(),
                  asyncio.new_event_loop().create_future(), est_prompt=111256, est_uncached=24)
    f.queue.append(p)
    f._x_exact_reprice_queue("test")
    assert p.est_uncached == 24
    assert f.tspans.pending(ids, epoch=3)[3] == "l3_index", "a store fact, beyond the epoch"


def test_a_finish_reading_replaces_the_l3_label():
    ts = TokenSpans(agent_span=True)
    ids = _ids(1000)
    assert ts.record_store_depth(ids, 960) == 960
    assert ts.pending(ids)[1:] == (960, True, "l3_index")
    ts.record_presence(ids, 990, prompt_tokens=1000, resumable_depth=960)
    assert ts.pending(ids)[3] == "d_leg2_cached"
    assert ts.record_store_depth(ids, 640) == 0, "a deeper reading stands"


def test_no_probe_no_credit_and_no_failure(caplog):
    f = _front(None, _ids(3000))
    f._store_probe_t = time.monotonic()
    xx = asyncio.run(f._x_exact_price("r", "/v1/messages", {"m": 1}, "t", 3000, 3000))
    assert (xx.credit, xx.src) == (0, "none")


def test_wiring():
    import inspect

    src = inspect.getsource(F.Front._x_exact_price)
    i = src.index("l3, tier = await self._store_probe_depth(")
    assert i < src.index("pending, credit, known, src = self.tspans.pending(c.ids, epoch=epoch)")
    assert 'self.tspans.record_store_depth(c.ids, l3, source=tier)' in src
    assert "await asyncio.get_running_loop().run_in_executor(ft.executor, self._store_probe_open)" \
        in inspect.getsource(F.Front._x_exact_boot_load)
    assert '"l3_index": int(self.counters["l3_index_credit"])' in inspect.getsource(F.Front.state_dict)


# ---- BOOT-START HOLD: the route decision waits for the tokenizer ----------------------

class _Req:
    def __init__(self, payload, path="/v1/messages"):
        self._p = payload
        self.path = path

    async def json(self):
        return self._p


class _LoadingTokens(_Tok):
    """Loading at the first arrival; records the state every count saw."""
    state = "loading"

    def __init__(self, ids):
        from concurrent.futures import ThreadPoolExecutor

        super().__init__(ids)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.states_at_count = []

    def count(self, path, payload):
        self.states_at_count.append(self.state)
        return super().count(path, payload)


def _boot_front(probe, ids):
    from flliper.srt.environ import envs

    with envs.FLLIPER_PDFLIP_FRONT_EXACT_TOKENS.override(True):
        f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="bootstart",
                    store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
                    weight_chunks=2, tp_prefill_max_tokens=X, flip_min_work_tokens=X)
    f.state = "serving"
    f.routed = []

    async def seat(rid, est, refused=None, **kw):   # Q-712: _acquire_short_seat(..., uncached=...)
        return 1

    async def leg2(request, rid, payload, text, stream, pending=None, **kw):
        f.routed.append(("short", rid))
        return F.web.json_response({})

    async def solo(rid, rem):
        return True

    f._acquire_short_seat = seat
    f.leg2 = leg2
    f._x_solo_admits = solo
    f._kick_controller = lambda *a, **k: None
    f.ftok = _LoadingTokens(ids)
    f.store_probe = probe
    return f


def test_y7d_boot_start_request_waits_for_the_tokenizer_then_prices_short_on_l3(store, caplog):
    """y7d 08:57:43: pdflip-0-2 arrived 0.8 s after 'PDFLIP-FRONT up', was priced
    115213 by chars/3 (reason=tokenizer_loading, presence_src=none), LONG, flip
    pair -- P hit 111232 of 111256 from the persistent L3. Now: held until the
    tokenizer is ready, then exact and credited by the store: SHORT."""
    caplog.set_level(logging.INFO, logger="pdflip.front")
    prev = _ids(111256)
    h = _tree_page_hashes(prev)
    store.write(h, mamba=[len(h) - 1])
    f = _boot_front(store.probe(), _ids(111256))
    payload = {"model": "m", "max_tokens": 10,
               "messages": [{"role": "user", "content": "a" * 345637}]}

    async def go():
        t = asyncio.create_task(f.handle_generate(_Req(payload)))
        await asyncio.sleep(0.2)
        held = not t.done() and not f.routed and not f.counters["route_long"]
        f.ftok.state = "ready"  # the load finished
        f._x_exact_ready_event().set()
        await asyncio.sleep(0.3)
        if not t.done():
            t.cancel()
        return held

    assert asyncio.run(go()), "the route decision waited while the tokenizer loaded"
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("tokenizer_loading" in m for m in msgs)
    assert f.ftok.states_at_count == ["ready"], "never counted/priced while loading"
    assert any(m.startswith("PDFLIP X-EXACT-HOLD rid=pdflip-0-1 ") for m in msgs)
    price = [m for m in msgs if m.startswith("PDFLIP X-EXACT-PRICE")]
    assert len(price) == 1 and "pending=24 tokens=111256 credit=111232 src=l3_index" in price[0]
    verdict = [m for m in msgs if m.startswith("PDFLIP ROUTE-VERDICT")][0]
    assert "verdict=short uncached=24 " in verdict and "presence_src=l3_index" in verdict
    assert f.routed == [("short", "pdflip-0-1")] and not f.counters["route_long"]


def test_the_boot_task_releases_the_hold_even_when_the_load_fails():
    async def go():
        f = object.__new__(F.Front)

        async def boom():
            raise RuntimeError("no group answered")

        f._x_exact_boot_load = boom
        try:
            await f._x_exact_boot()
        except RuntimeError:
            pass
        return f._x_exact_ready_event().is_set()

    assert asyncio.run(go())


def test_the_hold_is_bounded_when_the_load_never_ends(store, caplog, monkeypatch):
    """HOLD-BOUND (02.10.): the load waits for a group's /get_server_info with no
    end of its own, and a front whose startup hook never ran has no load at all
    (the h91c harness: test_27b_park2_metal's fairness requests never reached
    D). A route decision must never wait unbounded: after the bound it goes on
    by name (chars/3, reason=tokenizer_loading). Red on the parent: the hold
    waited forever."""
    caplog.set_level(logging.INFO, logger="pdflip.front")
    monkeypatch.setattr(F.Front, "X_EXACT_HOLD_MAX_S", 0.2, raising=False)
    f = _boot_front(store.probe(), _ids(30076))
    payload = {"model": "m", "max_tokens": 10,
               "messages": [{"role": "user", "content": "a" * 90000}]}

    async def go():
        # a LONG then waits in P's queue (no controller here): the subject is
        # the route decision, not the answer
        t = asyncio.create_task(f.handle_generate(_Req(payload)))
        for _ in range(50):
            await asyncio.sleep(0.1)
            if any("ROUTE-VERDICT" in r.getMessage() for r in caplog.records):
                break
        t.cancel()

    asyncio.run(go())
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("PDFLIP X-EXACT-HOLD-TIMEOUT rid=pdflip-0-1 ") for m in msgs)
    assert f.counters["x_exact_hold_timeout"] == 1
    assert any("PDFLIP ROUTE-VERDICT rid=pdflip-0-1" in m for m in msgs)
    assert f.ftok.states_at_count == [], "never counted while loading"


# ---- launcher: no declaration of the deleted CARRIER route --------------------------

def test_the_launcher_no_longer_declares_the_carrier_route():
    """y7d boot log: 'PDFLIP-LAUNCH DEVIATION (declared): zero-remainder: a BATCH
    prompt longer than group D's host staging pool ... is served by ONE prefill
    on D (front route CARRIER-EXCEEDS, no leg 1)' -- a route 9bdfe50185 deleted.
    The declaration names what the boot does: every BATCH prompt on P's leg 1."""
    import importlib.util

    with open(importlib.util.find_spec("flliper.srt.pdflip.launcher").origin) as f:
        src = f.read()
    i = src.index("state.deviations = [")
    block = src[i:src.index("]\n", i)]
    assert "CARRIER-EXCEEDS, no leg 1" not in block
    assert "is served by ONE prefill on D" not in block
    assert "every BATCH prompt takes P's leg 1" in block


# ---- L2-ARENA PRICE: pages COMPLETE only in the shared host arena -----------------------

SLOT = 64


def _arena(store, width=SLOT, slots=16384):
    """The shared host arena a rank creates (one file per page width)."""
    from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

    os.makedirs(store.arena, exist_ok=True)
    return ShmArena(os.path.join(store.arena, f"arena-{width}.bin"), width, slots)


def _complete(arena, stems):
    import ctypes

    buf = (ctypes.c_char * SLOT)()
    st = arena.write(list(stems), [SLOT] * len(stems), [((0, SLOT),)] * len(stems),
                     [ctypes.addressof(buf)] * len(stems))
    assert set(st) <= {1, 2}, st


def _l2_write(store, arena, hashes, kv=None, mamba=()):
    kv = range(len(hashes)) if kv is None else kv
    _complete(arena, [f"{hashes[i]}{SFX}" for i in kv])
    _complete(arena, [f"{hashes[i]}.qsa_indexer{SFX}" for i in kv])
    _complete(arena, [f"{hashes[i]}.mamba{SFX}" for i in mamba])


def test_y7d_pdflip_2_19_an_l2_only_in_boot_page_prefix_is_priced_short(store, caplog):
    """y7d 09:01:20 pdflip-2-19: 'X-EXACT-PRICE pending=30076 credit=0 src=none',
    LONG, flip pair, P hit 30016 of 30076. The longer request's pages were
    written this boot; the L3 write-behind lags L2 by minutes (27B N4a
    'deferred=86750'), so at pricing time they are COMPLETE in the shared
    host arena only. Credited from L2 -- SHORT, src=l2_arena."""
    caplog.set_level(logging.INFO)
    probe = store.probe()  # boot start: the arena does not exist yet
    arena = _arena(store)  # a rank creates it later, lazily
    longer = _ids(52000, seed=3)
    h = _tree_page_hashes(longer)
    _l2_write(store, arena, h, mamba=[468, len(h) - 1])
    cur = np.concatenate([longer[:30040], _ids(36, seed=9)])
    d = probe.depth(cur)
    assert (d.tokens, d.tier, d.l3_pages) == (30016, "l2_arena", 0)
    f = _front(probe, cur)
    xx = asyncio.run(f._x_exact_price("pdflip-2-19", "/v1/messages", {"m": 1}, "2-19", 30076, 30076))
    assert (xx.pending, xx.credit, xx.src) == (60, 30016, "l2_arena") and xx.pending <= X
    msgs = caplog.messages
    assert any(m.startswith("PDFLIP X-EXACT-PRICE rid=pdflip-2-19 pending=60 tokens=30076 credit=30016 "
                            "src=l2_arena") for m in msgs)
    assert any("PDFLIP L3-INDEX-PRESENCE rid=pdflip-2-19 tier=l2_arena depth=30016" in m for m in msgs)
    assert f.counters["l2_arena_credit"] == 1


def test_the_two_tiers_are_one_union_per_page(store):
    """KV pages 0..299 already in L3, 300.. only in L2, the anchor only in L2:
    the run is the union (what batch_exists_v2 reads, arena first); L3 alone
    proves 200 pages (its own anchor), so the depth needs L2."""
    arena = _arena(store)
    p = _ids(30000, seed=11)
    h = _tree_page_hashes(p)
    store.write(h[:300], mamba=[199])
    _l2_write(store, arena, h, kv=range(300, len(h)), mamba=[400])
    d = store.probe().depth(p)
    assert (d.pages, d.l3_pages, d.tier) == (401, 200, "l2_arena")
    _complete(arena, [f"{h[i]}.mamba{SFX}" for i in (199,)])  # already in L3: no change
    assert store.probe().depth(p).tier == "l2_arena"


def test_an_l3_proven_depth_is_named_l3_index(store):
    arena = _arena(store)
    p = _ids(9000, seed=12)
    h = _tree_page_hashes(p)
    store.write(h, mamba=[len(h) - 1])
    _l2_write(store, arena, h[:50], mamba=[49])  # L2 holds a shallower copy
    d = store.probe().depth(p)
    assert (d.pages, d.tier) == (len(h), "l3_index")


def test_a_claimed_not_complete_l2_page_is_no_credit(store):
    import ctypes

    arena = _arena(store)
    p = _ids(5000, seed=13)
    h = _tree_page_hashes(p)
    _l2_write(store, arena, h, kv=range(10), mamba=[9])
    stem = f"{h[10]}{SFX}"
    rows = arena.claim_slots([stem], [SLOT])  # a writer claimed page 10, bytes not landed
    assert rows and rows[0][1] == 0
    _complete(arena, [f"{h[i]}{SFX}" for i in range(11, len(h))])
    _complete(arena, [f"{h[i]}.qsa_indexer{SFX}" for i in range(10, len(h))])
    _complete(arena, [f"{h[len(h) - 1]}.mamba{SFX}"])
    assert store.probe().depth(p).pages == 10, "CLAIMED is not COMPLETE: the run stops at 10"
    del ctypes


def test_the_arena_view_never_creates_or_initialises(tmp_path):
    from flliper.srt.mem_cache.storage.file.hicache_arena import ArenaView

    path = str(tmp_path / "arena-64.bin")
    with pytest.raises(FileNotFoundError):
        ArenaView(path)
    assert not os.path.exists(path)
    open(path, "wb").close()  # a rank between create and init
    with pytest.raises(RuntimeError):
        ArenaView(path)
    assert os.path.getsize(path) == 0, "untouched"
