"""ZERO-ANCHOR 1015: every "cache hit ignored" case carries a cause.

Metal (NF h16p 226aad7105 + all 17 boots of 15.10., replay of the corrected
Prefill-Cache monitor, report h98): 74 P-leg-1 cases where the store's
presence stood >= 4096 tokens above ``cached``.

* 17 carried NO P-log marker. Two of them (boot 133855) were the store's
  COMPONENT cap: pdflip-5-52 ``#1028B FETCH CAP n=6: kv=1730 claimed=27 lost=1703
  caps={qsa_indexer: 47, mamba: 27}`` (108608 tokens re-prefilled), pdflip-40-179
  ``#1028B FETCH CAP n=30: kv=865 claimed=852`` (60352) -- the line carries no
  rid and stops after 40 per process. One (pdflip-0-6, boot 122948, 33856) was a
  claim whose KV prefix itself ended at page 47 of 576 -- ``final == kv``, so
  nothing printed at all.
* 38 were labelled BELOW-ANCHOR. Over the 35 with a PP0 line the cut to the
  deepest anchor inside the landed pages cost 78784 tokens, the pages the short
  landing never brought 1915520 -- e.g. pdflip-2-47 (boot 011215)
  ``read=36224 of 97280 anchored=35776``: cut 448, unlanded 61056.

Pinned (the real ``HybridCacheController._storage_hit_query`` on the real
``HiCacheFile.batch_exists_v2`` with the real L3 store of the QS pair test;
the real ``UnifiedRadixCache.check_prefetch_progress`` on the #257 harness):
  1. a component-capped claim prints ``PDFLIP-CLAIM-SHORT rid=... kv= claimed=
     lost= by=qsa_indexer``, a KV-prefix-short claim ``by=kv_prefix``; a full
     or nearly full claim prints nothing; the answer is unchanged;
  2. every #257 cut prints ``PDFLIP-ANCHOR-CUT rid=... cut_loss= unlanded=
     landing=<READ-DEPTH cause>``; the cut itself is unchanged;
  3. ``FLLIPER_PDFLIP_CLAIM_LINES=0`` prints neither (default ON).
RED on 226aad7105: ``flliper.srt.pdflip.claim_line`` does not exist, no line."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache.hicache_storage import (  # noqa: E402
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

_HERE = os.path.dirname(__file__)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qs = _load("_t_qs_pair_0930", "test_l3_qsa_pair_writers_0930.py")


@pytest.fixture(autouse=True)
def _awake_gate():
    try:
        from flliper.srt.mem_cache import l3_write_behind as gate
    except ImportError:
        yield
        return
    gate._reset_for_tests()
    yield
    gate._reset_for_tests()


def _store(tmp_path, monkeypatch, tag, kv_pages, qsa_pages):
    """The L3 store after the LRU split the pair: ``kv_pages`` KV pages on
    disk, the QSA index for the leading ``qsa_pages`` (pdflip-5-52: 1730 / 47)."""
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_WRITE_BEHIND_S", "0")
    be = qs._boot(tmp_path, monkeypatch, tag)
    hs = qs._hashes(tag[0])
    kv_arena, q_arena = be._arenas[qs.KV_TOTAL], be._arenas[qs.Q_TOTAL]
    for i, h in enumerate(hs[:kv_pages]):
        qs._put(kv_arena, be._get_suffixed_key(h), qs.KV_TOTAL, 0x40 + i)
    be._arena_copy_pages_to_disk(kv_arena, [be._get_suffixed_key(h) for h in hs[:kv_pages]])
    for h in hs[:qsa_pages]:
        qs._put(q_arena, be._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}"), qs.Q_TOTAL, 1)
    be._arena_copy_pages_to_disk(
        q_arena, [be._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}") for h in hs[:qsa_pages]])
    return be, hs


def _probe(be, hs, rid, caplog):
    """The real ``_storage_hit_query`` of the hybrid controller, PP0, the
    v2 arm (a QSA ALL_PAGES transfer, no draft tier)."""
    from flliper.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )

    ctl = types.SimpleNamespace(
        page_size=1, pp_rank=0, storage_backend=be,
        get_hash_str=lambda ids, last, page_size=1: list(hs[: len(ids)]),
        _draft_presence_transfer=lambda: None,
    )
    res = types.SimpleNamespace(kv=None)
    op = types.SimpleNamespace(
        request_id=rid, token_ids=list(range(len(hs))), last_hash=None, prefix_keys=None,
        pool_transfers=[PoolTransfer(name=PoolName.QSA_INDEXER, keys=[],
                                     hit_policy=PoolHitPolicy.ALL_PAGES)],
        is_terminated=lambda: False,
        pool_storage_result=types.SimpleNamespace(
            update_kv_hit_pages=lambda n: setattr(res, "kv", int(n))),
    )
    with caplog.at_level(logging.WARNING):
        keys, tokens = HybridCacheController._storage_hit_query(ctl, op)
    lines = [m for m in caplog.messages if "PDFLIP-CLAIM-SHORT" in m]
    return keys, tokens, res.kv, lines


@pytest.fixture
def _min2(monkeypatch):
    # the metal threshold is 64 pages (4096 tokens) on 64-token pages; the
    # store here has 8 one-token pages -> the same rule at 2 pages
    monkeypatch.setenv("FLLIPER_PDFLIP_CLAIM_LINE_MIN_PAGES", "2")
    monkeypatch.delenv("FLLIPER_PDFLIP_CLAIM_LINES", raising=False)


def test_pdflip_5_52_component_cap_names_rid_and_pool(tmp_path, monkeypatch, caplog, _min2):
    """KV on disk for all 8 pages, the QSA index for 3 (1730 / 47 on the
    metal): the claim is 3, the line names the rid, the loss and the pool."""
    be, hs = _store(tmp_path, monkeypatch, "a552", kv_pages=qs.PAGES, qsa_pages=qs.GAP)
    keys, tokens, kv, lines = _probe(be, hs, "pdflip-5-52", caplog)
    assert (len(keys), tokens, kv) == (qs.GAP, qs.GAP, qs.GAP)     # the answer is unchanged
    assert len(lines) == 1, caplog.messages
    assert ("PDFLIP-CLAIM-SHORT rid=pdflip-5-52 pp=0 asked=8 kv=8 claimed=3 lost=5 short=5 "
            "by=qsa_indexer tokens=3") in lines[0]


def test_pdflip_0_6_kv_prefix_short_is_named_too(tmp_path, monkeypatch, caplog, _min2):
    """The store's KV itself ends at page 3 (47 of 576 on the metal): no
    component cut, ``#1028B`` silent -- the new line says ``by=kv_prefix``."""
    be, hs = _store(tmp_path, monkeypatch, "b06", kv_pages=qs.GAP, qsa_pages=qs.GAP)
    keys, tokens, kv, lines = _probe(be, hs, "pdflip-0-6", caplog)
    assert (len(keys), kv) == (qs.GAP, qs.GAP)
    assert not any("#1028B FETCH CAP" in m for m in caplog.messages)
    assert len(lines) == 1
    assert "asked=8 kv=3 claimed=3 lost=0 short=5 by=kv_prefix" in lines[0]


def test_a_full_or_nearly_full_claim_prints_nothing(tmp_path, monkeypatch, caplog, _min2):
    be, hs = _store(tmp_path, monkeypatch, "cfull", kv_pages=qs.PAGES, qsa_pages=qs.PAGES)
    _k, _t, kv, lines = _probe(be, hs, "pdflip-9-9", caplog)
    assert kv == qs.PAGES and lines == []
    be2, hs2 = _store(tmp_path, monkeypatch, "dnear", kv_pages=qs.PAGES - 1, qsa_pages=qs.PAGES - 1)
    caplog.clear()
    _k, _t, kv2, lines2 = _probe(be2, hs2, "pdflip-9-8", caplog)
    assert kv2 == qs.PAGES - 1 and lines2 == []                   # 1 page short < 2


def test_switch_off_prints_nothing_and_answers_the_same(tmp_path, monkeypatch, caplog, _min2):
    monkeypatch.setenv("FLLIPER_PDFLIP_CLAIM_LINES", "0")
    be, hs = _store(tmp_path, monkeypatch, "eoff", kv_pages=qs.PAGES, qsa_pages=qs.GAP)
    keys, _t, kv, lines = _probe(be, hs, "pdflip-5-52", caplog)
    assert (len(keys), kv, lines) == (qs.GAP, qs.GAP, [])


# --- the #257 cut (real check_prefetch_progress on the #257 harness) --------

b257 = _load("_t_b257", "test_prefetch_below_anchor_257.py")


def _cut_lines(caplog):
    return [m for m in caplog.messages if "PDFLIP-ANCHOR-CUT" in m]


def test_pdflip_2_47_cut_names_its_cost_and_the_landing(monkeypatch, caplog):
    """16-page span, the read landed 10 (the walk stopped at page 10 for
    arena_full), the deepest anchor inside at 6: cut_loss 4, unlanded 6
    (pdflip-2-47: 448 / 61056). The claim is cut to 6 exactly as before."""
    from flliper.srt.pdflip import read_depth as rd

    monkeypatch.delenv("FLLIPER_PDFLIP_CLAIM_LINES", raising=False)
    cache, _store_ = b257._short_read(anchor_pages=6)
    op = cache.ongoing_prefetch[b257.REQ][3]
    rd.note_cut(op, b257.READ, rd.ARENA_FULL)
    with caplog.at_level(logging.WARNING):
        cache.check_prefetch_progress(b257.REQ)
    assert int(cache.prefetch_loaded_tokens_by_reqid[b257.REQ]) == 6
    lines = _cut_lines(caplog)
    assert len(lines) == 1, caplog.messages
    assert (f"PDFLIP-ANCHOR-CUT rid={b257.REQ} pp=") in lines[0]
    assert ("read=10 of=16 anchored=6 cut_to=6 cut_loss=4 unlanded=6 landing=arena_full"
            in lines[0])


def test_the_cut_line_is_not_rate_limited(monkeypatch, caplog):
    """The #257 line stops after 16 per process; this one never does."""
    monkeypatch.delenv("FLLIPER_PDFLIP_CLAIM_LINES", raising=False)
    cache, _ = b257._short_read(anchor_pages=0)
    cache._257_below_anchor_n = 100          # past the #257 line's cap
    with caplog.at_level(logging.WARNING):
        cache.check_prefetch_progress(b257.REQ)
    assert not any("#257 PREFETCH BELOW-ANCHOR" in m for m in caplog.messages)
    lines = _cut_lines(caplog)
    assert len(lines) == 1 and "cut_to=0 cut_loss=10 unlanded=6 landing=-" in lines[0]


def test_cut_line_switch_off(monkeypatch, caplog):
    monkeypatch.setenv("FLLIPER_PDFLIP_CLAIM_LINES", "0")
    cache, _ = b257._short_read(anchor_pages=6)
    with caplog.at_level(logging.WARNING):
        cache.check_prefetch_progress(b257.REQ)
    assert int(cache.prefetch_loaded_tokens_by_reqid[b257.REQ]) == 6
    assert _cut_lines(caplog) == []
