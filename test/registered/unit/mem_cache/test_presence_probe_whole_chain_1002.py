# SPDX-License-Identifier: Apache-2.0
"""The presence probe asks the WHOLE key chain when it asks about the mamba anchor.

Metal gmps14 (boot ...dual1mpsleepbar1fs10020256_4d088c5af7, D TP0, pdflip-0-19):
P's leg 1 (N=19941, 17450 cached, tail 2491) had written the tail to the store
before D's first probe -- ``#1028B FETCH CAP n=1: kv=1024 claimed=0 lost=1024
keys=1024 anchors_in_range mamba (0, -1)`` -- yet ``H108 PRESENCE-PROBE ...
covered=2490 pages=0 present=False`` for 21.5 s and 790 passes, then the refusal
and a second P prefill. ``store_presence_pages`` cut the chain to
``STORAGE_BATCH_SIZE`` (1024 on the boot) and asked ``batch_exists_v2``, whose
answer is capped to the last page carrying a mamba anchor; a hand-back tail's only
anchor is its end anchor at N-1, outside the first 1024 pages. Every tail longer
than the batch was invisible (gmps13: all 14 tails >= 1087 tokens refused, all 24
<= 963 admitted). The request was served only once D's device copy had been
evicted and the host chain made the probe unnecessary -- the real fetch
(``_storage_hit_query``) asks the whole span in one call.

Hermetic, real file backend, real method bodies (the #869b harness).
"""

import tempfile

import pytest

from flliper.srt.managers import cache_controller as CC
from flliper.srt.managers.cache_controller import HiCacheController
from flliper.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig, PoolName
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

TAIL = 2491          # pdflip-0-19's tail, pages at page_size 1
BATCH = 1024         # FLLIPER_HICACHE_STORAGE_BATCH on the dual boots


def _backend(tmpdir):
    cfg = HiCacheStorageConfig(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
                               is_mla_model=False, enable_storage_metrics=False, is_page_first_layout=True,
                               model_name="test/1002")
    be = HiCacheFile(cfg, file_path=tmpdir)
    be.file_path = tmpdir
    return be


def _write(be, key, component=None):
    stem = be._get_component_key(key, component)
    path = be._sharded_path(stem)
    be._ensure_shard_dir(path)
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 8)


class _Probe:
    store_presence_pages = HiCacheController.store_presence_pages

    def _presence_pool_transfers(self):
        return HiCacheController._presence_pool_transfers(self)

    def __init__(self, backend, hashes):
        self.storage_backend = backend
        self.page_size = 1
        self._hashes = hashes

    def get_hash_str(self, token_ids, last_hash, page_size=None):
        return list(self._hashes)


@pytest.fixture(scope="module")
def tail_store():
    """P's hand-back tail in the store: every KV page, the mamba anchor at the end only."""
    be = _backend(tempfile.mkdtemp(prefix="t1002wc_"))
    keys = ["p%05d" % i for i in range(TAIL)]
    for k in keys:
        _write(be, k)
    _write(be, keys[-1], PoolName.MAMBA)
    be.register_mem_host_pool_v2(object(), PoolName.MAMBA)
    return be, keys


def test_a_tail_longer_than_the_batch_is_seen_with_its_end_anchor(monkeypatch, tail_store):
    monkeypatch.setattr(CC, "STORAGE_BATCH_SIZE", BATCH)
    be, keys = tail_store
    assert _Probe(be, keys).store_presence_pages(list(range(TAIL + 1)), None) == TAIL


def test_the_hand_off_keys_are_asked_whole_too(monkeypatch, tail_store):
    # H108: P's keys spliced in (D's own hashes differ) -- the same whole-chain question
    monkeypatch.setattr(CC, "STORAGE_BATCH_SIZE", BATCH)
    be, keys = tail_store
    own = ["own%05d" % i for i in range(TAIL)]
    assert _Probe(be, own).store_presence_pages(list(range(TAIL + 1)), None, page_keys=keys) == TAIL


def test_a_tail_within_the_batch_is_unchanged(monkeypatch):
    monkeypatch.setattr(CC, "STORAGE_BATCH_SIZE", BATCH)
    be = _backend(tempfile.mkdtemp(prefix="t1002ws_"))
    keys = ["s%04d" % i for i in range(963)]       # gmps13: <= 963 always admitted
    for k in keys:
        _write(be, k)
    _write(be, keys[-1], PoolName.MAMBA)
    be.register_mem_host_pool_v2(object(), PoolName.MAMBA)
    assert _Probe(be, keys).store_presence_pages(list(range(964)), None) == 963


def test_no_anchor_is_still_absent(monkeypatch):
    """The #869b counter-arm holds for the whole chain: KV without a state is not a match."""
    monkeypatch.setattr(CC, "STORAGE_BATCH_SIZE", BATCH)
    be = _backend(tempfile.mkdtemp(prefix="t1002wn_"))
    keys = ["n%04d" % i for i in range(1500)]
    for k in keys:
        _write(be, k)
    be.register_mem_host_pool_v2(object(), PoolName.MAMBA)
    assert _Probe(be, keys).store_presence_pages(list(range(1501)), None) == 0


def test_the_cut_chain_mutant_is_the_gmps14_zero(monkeypatch, tail_store):
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(HiCacheController.store_presence_pages))
    fixed = "page_hashes, pool_transfers, extra_info"
    assert src.count(fixed) == 1, "the call moved -- re-aim the mutant"
    ns = dict(vars(CC))
    exec(compile(src.replace(fixed, "batch, pool_transfers, extra_info"), CC.__file__, "exec"), ns)
    monkeypatch.setattr(_Probe, "store_presence_pages", ns["store_presence_pages"])
    with pytest.raises(AssertionError):
        test_a_tail_longer_than_the_batch_is_seen_with_its_end_anchor(monkeypatch, tail_store)
