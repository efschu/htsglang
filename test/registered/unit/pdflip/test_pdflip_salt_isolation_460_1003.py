# SPDX-License-Identifier: Apache-2.0
"""Q-460 SALT-ISOLATION (27B y8r 55c95a89c7, boot ..._1003_090656).

Metal (probe_27b_y8q.py, cache_salt black box over usage.cached_tokens):
  A1 salt A  pdflip-1-7   cached=0
  A2 salt A  pdflip-2-11  cached=8187/8963   (P: HiCache prefetch matched=39 loaded=8153)
  B1 salt B  pdflip-4-16  cached=8187/8963   FAIL (same store read as A2)
  N1 no salt pdflip-6-30  cached=8192/8964   FAIL (D-direct, D read the same 8192 store pages)
  t2b salt B pdflip-8-49  cached=8983/9019   FAIL (P: resident=8988 registered=8988 -- t2a's
                                                 salt-A pages in P's DEVICE tree)

Two holes, both shown red here:
  1. the front's leg 1 (LEG1-INPUT-IDS, P's /generate with the front's ids)
     dropped cache_salt / extra_key: every leg 1 ran on P under namespace None;
  2. the store page key (L2 arena / L3) was a function of the TOKENS alone
     (``get_hash_str`` ignores ``RadixKey.extra_key``): a page written under
     salt A was found by salt B and by an unsalted request -- on P and on D.
     Upstream has the same key (``compute_node_hash_values`` / ``_storage_hit_query``
     hash token ids only; 3639655dda only keys the prefetched SPAN's insert).
Plus the edges that must agree with the new key: the presence probes
(store_presence_pages, the told clamp, the front's L3/L2 price), the
resume-via-P re-prefill leg, the Anthropic adapter (it dropped cache_salt, so
D and the front disagreed about the namespace), and the front's token spans
(ids-keyed credit: t2b was priced from t2a's salt-A reading).
"""

from __future__ import annotations

import json
import os
import tempfile
from array import array
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402

CHAT = {"model": "m", "rid": "pdflip-4-16", "max_tokens": 1,
        "messages": [{"role": "user", "content": "hello"}]}
IDS = list(range(100, 100 + 4 * 64 + 1))


# --------------------------------------------------------------- 1. P intake
def test_leg1_ids_body_carries_the_salt():
    ids = np.asarray([5, 6, 7], dtype=np.int32)
    for path in ("/v1/chat/completions", "/v1/messages"):
        b = F.leg1_input_ids_payload(path, dict(CHAT, cache_salt="probe-A"), ids)
        assert b is not None and b.get("extra_key") == "probe-A", (path, b)
        b = F.leg1_input_ids_payload(path, dict(CHAT, cache_salt="A", extra_key="x"), ids)
        assert b["extra_key"] == "Ax", b  # serving_base._compute_extra_key: concatenated
    b = F.leg1_input_ids_payload("/generate", dict(CHAT, extra_key="k"), ids)
    assert b["extra_key"] == "k", b


def test_leg1_unsalted_body_unchanged_and_lora_keeps_the_client_path():
    ids = [5, 6, 7]
    b = F.leg1_input_ids_payload("/v1/chat/completions", dict(CHAT), ids)
    assert "extra_key" not in b
    # LoRA: the group folds the adapter into the key -- the front cannot form it
    assert F.leg1_input_ids_payload("/v1/chat/completions", dict(CHAT, model="base:ad"), ids) is None
    assert F.leg1_input_ids_payload("/generate", dict(CHAT, lora_path="ad"), ids) is None


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


# ---------------------------------------------------- 4. front token spans
def test_front_never_records_a_namespaced_reading():
    fake = SimpleNamespace(x_exact=True, tspans=object(),
                           _l15_ek={"pdflip-8-48": (True, "probe-A"), "pdflip-9-1": (False, None),
                                    "pdflip-9-2": (True, None)})
    assert F.Front._ns_isolated(fake, "pdflip-8-48")
    assert F.Front._ns_isolated(fake, "pdflip-9-1")      # LoRA: unknown = isolated
    assert not F.Front._ns_isolated(fake, "pdflip-9-2")  # default namespace
    assert not F.Front._ns_isolated(fake, "pdflip-unknown")
    # P's END-ANCHOR of a salted request credits no one (tspans untouched)
    assert F.Front._p_anchor_presence(fake, "pdflip-8-48", "text", None) == 0


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
