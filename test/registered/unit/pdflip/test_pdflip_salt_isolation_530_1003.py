# SPDX-License-Identifier: Apache-2.0
"""Q-530 SALT-ISOLATION on NF (port of 27B Q-460 058eda6506; NF base 044316dd1a).

27B metal (y8r 55c95a89c7, probe_27b_y8q.py, cache_salt black box over
usage.cached_tokens): salt B and an unsalted request got ``cached=8187/8963``
and ``8192/8964`` on salt A's prefix -- a store read (``HiCache prefetch
matched=39 loaded=8153``), and t2b ``8983/9019`` from salt A's pages in P's
tree. NF carries the same store key (identical ``get_hash_str``), so the L2/L3
leak is the same; what differs on NF:

  * no ``leg1_input_ids_payload`` (27B-only, LEG1-INPUT-IDS): P's leg 1 gets the
    client path, so the client's cache_salt reaches P's serving layer -- the
    first root does not exist here;
  * ``prefetch_namespace`` (y3k-korr) files the span under the REQUEST's salt:
    with a namespace-blind page key that put another tenant's pages beside the
    salted request's own path;
  * no L15 (``l15_share_admit``): the front keeps its own namespace note
    (``Front._ns_note`` / ``payload_namespace``);
  * the front's resume-via-P leg 1 is built here from D's needs-P record and
    carried no extra_key; the Anthropic adapter dropped cache_salt
    (``extra="ignore"``); the token spans are ids-keyed (a salted request took
    and gave credit across tenants).

Red on 044316dd1a, green with the port. The unsalted key stays bit-identical.
"""

from __future__ import annotations

import asyncio
import collections
import os
import tempfile
import time
from array import array
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import front_store as FS  # noqa: E402
from flliper.srt.pdflip.front_tokens import Count, TokenSpans  # noqa: E402

CHAT = {"model": "m", "rid": "pdflip-4-16", "max_tokens": 1,
        "messages": [{"role": "user", "content": "hello"}]}
IDS = list(range(100, 100 + 4 * 64 + 1))


# ------------------------------------------------- 1. the namespace of a payload
def test_payload_namespace_is_what_the_serving_layer_builds():
    # serving_base._compute_extra_key: cache_salt + extra_key, concatenated
    assert F.payload_namespace(dict(CHAT)) == (True, None)
    assert F.payload_namespace(dict(CHAT, cache_salt="probe-A")) == (True, "probe-A")
    assert F.payload_namespace(dict(CHAT, cache_salt="A", extra_key="x")) == (True, "Ax")
    assert F.payload_namespace(dict(CHAT, cache_salt="")) == (True, None)
    # LoRA: the group folds the adapter into the key -- the front cannot form it
    assert F.payload_namespace(dict(CHAT, model="base:ad"))[0] is False
    assert F.payload_namespace(dict(CHAT, lora_path="ad"))[0] is False
    assert F.payload_namespace(dict(CHAT, cache_salt=7))[0] is False


# ------------------------------------------------------------ 2. store page keys
def _rk(ids, ek, bigram=False):
    from flliper.srt.mem_cache.radix_cache import RadixKey

    return RadixKey(array("q", ids), ek, is_bigram=bigram)


def test_store_page_keys_differ_per_namespace():
    from flliper.srt.mem_cache.utils import get_hash_str

    for bigram in (False, True):
        a = get_hash_str(_rk(IDS, "probe-A", bigram), None, page_size=64)
        b = get_hash_str(_rk(IDS, "probe-B", bigram), None, page_size=64)
        n = get_hash_str(_rk(IDS, None, bigram), None, page_size=64)
        assert len(a) == len(b) == len(n) >= 4
        # no page of one namespace's chain is a page of another's
        assert not (set(a) & set(b)) and not (set(a) & set(n)) and not (set(b) & set(n)), bigram


def test_unsalted_store_key_is_unchanged():
    """The persistent L3 of earlier boots stays readable: the default
    namespace hashes exactly as before (plain ids == RadixKey(None))."""
    from flliper.srt.mem_cache.utils import get_hash_str

    assert get_hash_str(list(IDS), None, page_size=64) == get_hash_str(_rk(IDS, None), None, page_size=64)
    assert get_hash_str(list(IDS), "", page_size=64) == get_hash_str(list(IDS), None, page_size=64)


def test_write_key_equals_read_key_in_a_namespace():
    """P writes through the node's key (compute_node_hash_values on a child
    of the root); D / P read with the prefetch RadixKey (``_storage_hit_query``)
    or a plain-id probe given the namespace: one chain."""
    from flliper.srt.mem_cache.utils import compute_node_hash_values, get_hash_str

    root = SimpleNamespace(key=_rk([], None), hash_value=[], parent=None)
    node = SimpleNamespace(key=_rk(IDS[:256], "probe-A"), parent=root, hash_value=None)
    written = compute_node_hash_values(node, 64)
    read = get_hash_str(_rk(IDS[:256], "probe-A"), None, page_size=64)
    probe = get_hash_str(list(IDS[:256]), None, page_size=64, extra_key="probe-A")
    assert written == read == probe
    # the continuation below a namespaced node chains from the node's last hash
    child = SimpleNamespace(key=_rk(IDS[256:], "probe-A"), parent=node, hash_value=None)
    node.hash_value = written
    assert compute_node_hash_values(child, 1) == get_hash_str(_rk(IDS[256:], "probe-A"), written[-1], page_size=1)
    other = SimpleNamespace(key=_rk(IDS[:256], "probe-B"), parent=root, hash_value=None)
    assert not set(compute_node_hash_values(other, 64)) & set(written)


# ------------------------------------------------------------ 3. presence probes
class _Backend:
    def __init__(self):
        self.asked = []

    def batch_exists(self, keys, extra=None):
        self.asked.append(list(keys))
        return 0


def test_store_presence_probe_asks_in_the_namespace():
    from flliper.srt.managers.cache_controller import HiCacheController
    from flliper.srt.mem_cache.utils import get_hash_str, namespace_root_hash

    seen = []

    def _hash(tokens, prior, page_size=None):
        seen.append(prior)
        return get_hash_str(tokens, prior, page_size=page_size)

    be = _Backend()
    cc = SimpleNamespace(get_hash_str=_hash, page_size=64, storage_backend=be,
                         _presence_pool_transfers=lambda: None)
    HiCacheController.store_presence_pages(cc, list(IDS), None, extra_key="probe-B")
    HiCacheController.store_presence_pages(cc, list(IDS), None)
    assert seen == [namespace_root_hash("probe-B"), None]
    salted, plain = be.asked
    assert not set(salted) & set(plain)
    assert salted == get_hash_str(_rk(IDS, "probe-B"), None, page_size=64)[: len(salted)]


def test_told_clamp_probe_asks_in_the_namespace():
    from flliper.srt.managers import pdflip_store_told as st
    from flliper.srt.mem_cache.utils import get_hash_str

    be = _Backend()
    cc = SimpleNamespace(get_hash_str=get_hash_str, storage_backend=be,
                         _presence_pool_transfers=lambda: None)
    st._anchored_pages_full_span(cc, list(IDS), 64, extra_key="probe-A")
    st._anchored_pages_full_span(cc, _rk(IDS, "probe-A"), 64)
    st._anchored_pages_full_span(cc, list(IDS), 64)
    a_list, a_key, plain = be.asked
    assert a_list == a_key and not set(a_list) & set(plain)


def test_front_store_price_is_the_namespace_chain():
    from flliper.srt.pdflip import front_store as fs

    ids = np.asarray(IDS, dtype=np.int32)
    a = fs.bigram_page_hasher(ids, 64, True, "probe-A")
    n = fs.bigram_page_hasher(ids, 64, True)
    assert a and n and not set(a) & set(n)
    calls = []

    def hasher(*args):
        calls.append(args[3:])
        return []

    sp = fs.StorePresence.__new__(fs.StorePresence)
    sp.hasher, sp.page_size, sp.bigram = hasher, 64, True
    sp._rejoin = lambda: None
    sp._depth_fast = lambda hashes, t0: fs.Depth(tokens=0, kv_pages=0, pages=0, ms=0.0)
    sp.depth(ids, fast=True, extra_key="probe-B")
    sp.depth(ids, fast=True)
    assert calls == [("probe-B",), ()]


def test_unsalted_store_key_is_bit_identical_to_the_native_hash():
    """Against the hash function itself, not against get_hash_str: the
    persistent L3 of earlier boots stays readable."""
    from flliper.srt.mem_cache.cpp_utils.native_hash import get_native_hash
    from flliper.srt.mem_cache.utils import get_hash_str

    for ps in (1, 64):
        assert get_hash_str(list(IDS), None, page_size=ps) == get_native_hash(list(IDS), None, ps)
        assert get_hash_str(_rk(IDS, None), None, page_size=ps) == get_native_hash(_rk(IDS, None), None, ps)
    first = get_hash_str(list(IDS), None, page_size=64)
    assert get_hash_str(list(IDS[64:]), first[0], page_size=64) == \
        get_native_hash(list(IDS[64:]), bytes.fromhex(first[0]), 64)


def test_prefetch_span_of_a_salted_request_reads_its_own_chain():
    """y3k-korr files the prefetched span under the REQUEST's salt
    (prefetch_namespace); the read key (the prefetch RadixKey) is that chain's
    root-seeded key: another namespace's pages are not found."""
    from flliper.srt.mem_cache.unified_radix_cache import prefetch_namespace
    from flliper.srt.mem_cache.utils import get_hash_str

    ek = prefetch_namespace(anchor_extra_key=None, request_extra_key="probe-B")
    assert ek == "probe-B"
    read_b = get_hash_str(_rk(IDS, ek), None, page_size=64)
    written_a = get_hash_str(_rk(IDS, "probe-A"), None, page_size=64)
    written_none = get_hash_str(_rk(IDS, None), None, page_size=64)
    assert not set(read_b) & (set(written_a) | set(written_none))


# ---------------------------------------------------- 4. front token spans
def test_front_never_records_a_namespaced_reading():
    fake = SimpleNamespace(x_exact=True, tspans=object(),
                           _ns_ek={"pdflip-8-48": (True, "probe-A"), "pdflip-9-1": (False, None),
                                   "pdflip-9-2": (True, None)})
    assert F.Front._ns_isolated(fake, "pdflip-8-48")
    assert F.Front._ns_isolated(fake, "pdflip-9-1")      # LoRA: unknown = isolated
    assert not F.Front._ns_isolated(fake, "pdflip-9-2")  # default namespace
    assert not F.Front._ns_isolated(fake, "pdflip-unknown")
    # P's END-ANCHOR of a salted request credits no one (tspans untouched)
    assert F.Front._p_anchor_presence(fake, "pdflip-8-48", "text", None) == 0
    # D's in-flight anchor and the sequence mark likewise
    assert F.Front._d_inflight_presence(fake, "pdflip-8-48", "text", None, "d_direct") == 0
    fake.tspans = SimpleNamespace(record_seq=lambda *a, **k: (_ for _ in ()).throw(AssertionError("fed")))
    assert F.Front._seq_record(fake, "pdflip-8-48", np.arange(4), "64:abcd", "finish") is False


def test_arrival_note_marks_the_rid():
    fake = SimpleNamespace()
    F.Front._ns_note(fake, "r-salt", dict(CHAT, cache_salt="A"))
    F.Front._ns_note(fake, "r-plain", dict(CHAT))
    F.Front._ns_note(fake, "r-lora", dict(CHAT, model="b:l"))
    assert F.Front._ns_isolated(fake, "r-salt") and F.Front._ns_isolated(fake, "r-lora")
    assert not F.Front._ns_isolated(fake, "r-plain")


def test_p_flush_skips_a_namespaced_leg():
    ids = np.arange(300, dtype=np.int32)
    fed = []
    ts = SimpleNamespace(own_anchor=lambda n: 0,
                         record_store_anchor=lambda i, pt, source=None: fed.append(pt) or 64)
    ft = SimpleNamespace(ids_for=lambda text: ids)
    fake = SimpleNamespace(
        _p_phase_served=collections.OrderedDict(a=("ta", 300), b=("tb", 300)),
        tspans=ts, ftok=ft, counters=collections.Counter(),
        _ns_ek={"a": (True, "probe-A"), "b": (True, None)},
        epoch=1, _x_exact_reprice_queue=lambda why: None,
        _mm_persist_anchor=lambda *a: None, _mm_persist_flush=lambda: None)
    fake._ns_isolated = lambda rid: F.Front._ns_isolated(fake, rid)
    assert F.Front._p_flush_store_presence(fake) == 1
    assert fed == [300] and fake.counters["store_presence_ns_isolated"] == 1


# ---------------------------------------------------- 4b. the price at arrival
PAGE = 64
X = 4855


class _NsProbe:
    """A store holding the page chains of the namespaces ``store`` -- asked with
    the front's own hasher (bigram_page_hasher), so the key is the real one."""

    def __init__(self, store):
        self.have = set()
        for ids, ek in store:
            self.have |= set(FS.bigram_page_hasher(ids, PAGE, True, ek))
        self.asked = []

    def depth(self, ids, fast=None, extra_key=None):
        self.asked.append(extra_key)
        hashes = FS.bigram_page_hasher(ids, PAGE, True, extra_key)
        n = 0
        for h in hashes:
            if h not in self.have:
                break
            n += 1
        return FS.Depth(tokens=n * PAGE, kv_pages=n, pages=n, ms=0.0, tier="l3_index")


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


def _front(probe, ids):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = 0
    f.awake = "D"
    f.state = "serving"
    f.tp_prefill_max_tokens = X
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = _Tok(ids)
    f._x_exact_rid = collections.OrderedDict()
    f.store_probe = probe
    return f


def _price(f, rid, payload):
    Front = F.Front
    Front._ns_note(f, rid, payload)
    return asyncio.run(f._x_exact_price(rid, "/v1/messages", payload, "t-" + rid, 9000, 9000))


def _ids(n=9000, seed=1):
    rng = np.random.default_rng(seed)
    out = rng.integers(1, 240000, size=n, dtype=np.int64).astype(np.int32)
    out[0] = 248045
    return out


def test_price_is_taken_from_the_requests_own_namespace(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_EARLY_FLIP", "0")
    ids = _ids()
    probe = _NsProbe([(ids, "probe-A")])         # salt A's pages are in the store
    f = _front(probe, ids)
    a = _price(f, "pdflip-2-11", dict(CHAT, cache_salt="probe-A"))
    b = _price(f, "pdflip-4-16", dict(CHAT, cache_salt="probe-B"))
    n = _price(f, "pdflip-6-30", dict(CHAT))
    assert a.credit > 8000 and a.pending < X, a
    assert (b.credit, b.src) == (0, "none") and b.pending == ids.size, b   # not A's pages
    assert (n.credit, n.src) == (0, "none") and n.pending == ids.size, n
    assert probe.asked == ["probe-A", "probe-B", None]
    # a salted request gave the spans nothing: an unsalted twin finds no credit there
    assert f.tspans.pending(ids, epoch=0)[1] == 0
    assert f.counters["ns_isolated_priced"] == 2


def test_unsalted_price_unchanged_and_feeds_the_spans(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_EARLY_FLIP", "0")
    ids = _ids(seed=2)
    f = _front(_NsProbe([(ids, None)]), ids)
    n = _price(f, "pdflip-1-1", dict(CHAT))
    assert n.credit > 8000 and n.src == "l3_index"
    assert f.tspans.pending(ids, epoch=0)[1] == n.credit
    assert f.counters["ns_isolated_priced"] == 0


def test_a_salted_request_does_not_take_a_span_credit(monkeypatch):
    """t2b pdflip-8-49: priced 8988 from t2a's reading (ids-keyed span)."""
    monkeypatch.setenv("FLLIPER_PDFLIP_EARLY_FLIP", "0")
    ids = _ids(seed=3)
    f = _front(_NsProbe([]), ids)
    f.tspans.record_presence(ids, 8988, prompt_tokens=9000, held_epoch=0)  # t2a's D reading
    assert f.tspans.pending(ids, epoch=0)[1] > 8000
    b = _price(f, "pdflip-8-49", dict(CHAT, cache_salt="probe-B"))
    assert (b.credit, b.src) == (0, "none"), b
    plain = _price(f, "pdflip-8-50", dict(CHAT))
    assert plain.credit > 8000, "the unsalted twin keeps the span credit"


def test_a_lora_request_gets_no_store_ask_and_no_credit(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_EARLY_FLIP", "0")
    ids = _ids(seed=4)
    probe = _NsProbe([(ids, None)])
    f = _front(probe, ids)
    r = _price(f, "pdflip-3-3", dict(CHAT, model="base:ad"))
    assert (r.credit, r.src) == (0, "none") and probe.asked == []


def test_requeue_reprice_leaves_a_namespaced_request_alone():
    ids = _ids(seed=5)
    f = _front(None, ids)
    f.ftok.remember("t-salt", ids)
    f.tspans.record_presence(ids, 8988, prompt_tokens=9000, held_epoch=0)
    F.Front._ns_note(f, "pdflip-8-49", dict(CHAT, cache_salt="probe-B"))
    p = F.Pending("pdflip-8-49", "/v1/messages", {}, "t-salt", time.time(),
                  asyncio.new_event_loop().create_future(), est_prompt=9000, est_uncached=9000)
    f.queue.append(p)
    f._x_exact_reprice_queue("test")
    assert p.est_uncached == 9000


# ------------------------------------------------ 5b. the front's resume-via-P leg
def test_front_resume_via_p_leg_carries_the_namespace(monkeypatch):
    from flliper.srt.pdflip import resume_via_p as rvp

    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
        monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", d)
        monkeypatch.delenv(rvp.ENV, raising=False)
        monkeypatch.setenv(rvp.ENV_OPEN_STREAM, "0")
        ns = SimpleNamespace(tag="dkrtest", queue=[], counters=collections.Counter(), kicks=[])
        ns._kick_controller = lambda why: ns.kicks.append(why)
        for name in ("_rvp_state", "_note_front_price", "_rvp_take", "_rvp_p_finished", "_rvp_resumed"):
            setattr(ns, name, getattr(F.Front, name).__get__(ns))
        # D wrote the namespace into the needs-P record
        rvp.write_request("pdflip-8-49", list(range(300)), 280, 128, "x_refusal_midstream",
                          extra_key="probe-B")
        # an old record without it: the arrival's note supplies it
        rvp.write_request("pdflip-8-50", list(range(300)), 280, 128, "x_refusal_midstream")
        rvp.write_request("pdflip-8-51", list(range(300)), 280, 128, "x_refusal_midstream")
        F.Front._ns_note(ns, "pdflip-8-50", dict(CHAT, cache_salt="probe-C"))
        F.Front._ns_note(ns, "pdflip-8-51", dict(CHAT))
        for rid in ("pdflip-8-49", "pdflip-8-50", "pdflip-8-51"):
            ns._note_front_price(rid, 10)
        async def _take():
            return ns._rvp_take()

        assert asyncio.new_event_loop().run_until_complete(_take()) == 3
        got = {p.rid: p.payload for p in ns.queue}
    assert got["pdflip-8-49"]["extra_key"] == "probe-B"
    assert got["pdflip-8-50"]["extra_key"] == "probe-C"
    assert "extra_key" not in got["pdflip-8-51"]


# -------------------------------------------------------- 5. resume via P
def test_resume_via_p_record_carries_the_namespace():
    from flliper.srt.pdflip import resume_via_p as rvp

    with tempfile.TemporaryDirectory() as d:
        rvp.write_request("pdflip-8-49", [1, 2, 3], 3, 2, "x", directory=d, extra_key="probe-B")
        rvp.write_request("pdflip-8-50", [1, 2, 3], 3, 2, "x", directory=d)
        got = {r["rid"]: r for r in rvp.take_requests(d)}
    assert got["pdflip-8-49"]["extra_key"] == "probe-B"
    assert "extra_key" not in got["pdflip-8-50"]


# -------------------------------------------------------- 6. Anthropic wire
def test_anthropic_request_keeps_cache_salt():
    from flliper.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest

    r = AnthropicMessagesRequest(model="m", max_tokens=4, cache_salt="probe-A", extra_key="x",
                                 messages=[{"role": "user", "content": "hi"}])
    assert r.cache_salt == "probe-A" and r.extra_key == "x"


def test_anthropic_conversion_passes_the_namespace():
    from flliper.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from flliper.srt.entrypoints.anthropic.serving import AnthropicServing

    chat = SimpleNamespace(tokenizer_manager=None, reasoning_parser=None,
                           supports_native_reasoning_history=lambda: False,
                           apply_reasoning_enabled=lambda req, enabled: None)
    srv = AnthropicServing.__new__(AnthropicServing)
    srv.openai_serving_chat = chat
    srv._merge_inline_system = False
    srv._inline_system_in_place = False
    r = AnthropicMessagesRequest(model="m", max_tokens=4, cache_salt="probe-A",
                                 messages=[{"role": "user", "content": "hi"}])
    out = srv._convert_to_chat_completion_request(r)
    assert out.cache_salt == "probe-A"


# -------------------------------------------------------- 7. draft KV keys
def test_dflash_draft_page_keys_are_namespaced():
    from flliper.srt.speculative import dflash_draft_kv_producer as dp

    def req(ek):
        return SimpleNamespace(rid="r", fill_ids=list(IDS[:64]), extra_key=ek)

    a = dp.chunk_page_hashes(req("probe-A"), 0, 63, bigram=True)
    n = dp.chunk_page_hashes(req(None), 0, 63, bigram=True)
    assert a and n and not set(a) & set(n)
