# SPDX-License-Identifier: Apache-2.0
"""L3-INDEX PRICE (NF 2c1150fa7f) on the 27B tree: page size ONE token, bigram keys.

27B facts (boot ..._1002_083747, 6dd8d7e68c): both groups 'page_size=1',
'speculative_algorithm=DFLASH', env SGLANG_HICACHE_BIGRAM_KEYS=1, the shared
index '#1459 L3-INDEX joined at /dev/shm/weg2-arena-<tag>-l3idx/l3idx.bin (cap
8388608, entries 1806831)', the persistent store
'/var/lib/htsglang/hicache-weg2/l3-qwen27b-...', mamba anchors every
SGLANG_WEG2_MAMBA_ANCHOR_INTERVAL=4096 plus P's chunk-end / END anchors.

* N3y weg2-8-15 (08:42:38, D phase, X=4896): 'X-EXACT-PRICE pending=111256
  tokens=111256 credit=0 src=none ... reused=109889' -> LONG, P hit 109703 of
  111256. 109703 = 60551 + 48 x 1024: an INNER anchor of weg2-0-4's P prefill
  (0-4: prompt 118498, cached 60551, 1024-token chunks) -- no END-ANCHOR any
  presence record names, but the store holds it: the L3 index credits it.
* N3y weg2-0-2 (08:40:12) is NOT store credit: its 65536 hit came from its
  twin weg2-0-1 computed in the same P phase ('#TW TWIN-DEFER rid=weg2-0-2
  shared=65603 sources=['weg2-0-1']', 0-1 cached_tokens=0) -- at its price the
  store held nothing on that path, so the probe must credit 0 (no over-credit).
"""
from __future__ import annotations

import json
import os
import types
from array import array

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.storage.file.l3_index import L3Index  # noqa: E402
from sglang.srt.mem_cache.utils import compute_node_hash_values  # noqa: E402
from sglang.srt.weg2 import front_store as FS  # noqa: E402

PAGE = 1  # the 27B tree's page
SFX = "_Qwen3.8-27B-INT8-gdncov-vocabembed_27bstoretest"


def _ids(n, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.integers(1, 240000, size=n, dtype=np.int64).astype(np.int32)
    out[0] = 248045
    return out


class _Store:
    def __init__(self, tmp):
        self.dir = os.path.join(tmp, "l3-qwen27b-test")
        os.makedirs(self.dir)
        self.arena = os.path.join(tmp, "weg2-arena-dkr27b")
        for g, extra in (("P", "_0_1_3_0"), ("D", "_0_3")):
            with open(os.path.join(self.dir, f"L3_SUFFIXES.{g}.json"), "w") as f:
                json.dump({"group": g, "suffixes": [SFX + extra, SFX]}, f)
        os.makedirs(FS.l3_index_path(self.arena).rsplit("/", 1)[0])
        self.index = L3Index(FS.l3_index_path(self.arena), cap=1 << 19)

    def write(self, hashes, kv_upto, mamba):
        for stems in ([f"{h}{SFX}" for h in hashes[:kv_upto]],
                      [f"{hashes[i]}.mamba{SFX}" for i in mamba]):
            for off in range(0, len(stems), 4096):
                self.index.add(stems[off:off + 4096])
            for s in stems[:8]:
                d = os.path.join(self.dir, s[:2])
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, s + ".bin"), "wb").close()

    def probe(self):
        probe, why = FS.open_store_presence(self.dir, self.arena, PAGE, True, True)
        assert probe is not None, why
        assert "page=1 bigram=1" in why and "trailing=['mamba']" in why, why
        return probe


@pytest.fixture
def store(tmp_path):
    return _Store(str(tmp_path))


def test_page_one_keys_are_the_trees_chained_keys():
    ids = _ids(3000, seed=1)
    raw = array("q", [int(t) for t in ids])
    parent = types.SimpleNamespace(key=RadixKey(raw[:1201], None, is_bigram=True), parent=None,
                                   hash_value=None)
    parent.hash_value = compute_node_hash_values(parent, PAGE)
    child = types.SimpleNamespace(key=RadixKey(raw[1200:], None, is_bigram=True), parent=parent,
                                  hash_value=None)
    tree = parent.hash_value + compute_node_hash_values(child, PAGE)
    assert FS.bigram_page_hasher(ids, PAGE, True) == tree[:2999]


def test_n3y_weg2_8_15_is_credited_at_0_4s_inner_anchor(store):
    ids_0_4 = _ids(118498, seed=4)
    h = FS.bigram_page_hasher(ids_0_4, PAGE, True)
    # 0-4's P prefill from 60551 in 1024 chunks: anchors at chunk ends (kept ones) + its END-ANCHOR
    inner = [60551 + 1024 * k for k in (46, 47, 48)]
    store.write(h, kv_upto=len(h), mamba=[d - 1 for d in inner] + [118496 - 1])
    ids_8_15 = np.concatenate([ids_0_4[:109889], _ids(111256 - 109889, seed=15)])
    d = store.probe().depth(ids_8_15)
    assert d.kv_pages >= 109888
    assert d.tokens == 109703, "the deepest anchor on the shared path = P's real hit"
    assert 111256 - d.tokens == 1553 <= 4896, "SHORT (the log: LONG, P computed 1553)"


def test_n3y_weg2_0_2_gets_no_store_credit_before_its_twin_is_computed(store):
    # an unrelated earlier prefix in the store, nothing on 0-2's path
    other = _ids(20000, seed=99)
    store.write(FS.bigram_page_hasher(other, PAGE, True), kv_upto=19999, mamba=[16383, 19998])
    assert store.probe().depth(_ids(65762, seed=2)).tokens == 0


def test_kv_without_an_anchor_is_no_credit(store):
    p = _ids(9000, seed=7)
    store.write(FS.bigram_page_hasher(p, PAGE, True), kv_upto=8999, mamba=[])
    d = store.probe().depth(p)
    assert d.kv_pages == 8999 and d.tokens == 0, "hybrid: no anchor, no resume point"


def test_the_backward_anchor_scan_equals_the_forward_answer(store):
    p = _ids(12000, seed=8)
    h = FS.bigram_page_hasher(p, PAGE, True)
    store.write(h, kv_upto=11000, mamba=[100, 4095, 8191, 11500])  # 11500 lies past the KV run
    pr = store.probe()
    assert pr.depth(p).tokens == 8192
    assert pr._deepest(h, 11000, "mamba") == 8191 and pr._deepest(h, 100, "mamba") == -1
