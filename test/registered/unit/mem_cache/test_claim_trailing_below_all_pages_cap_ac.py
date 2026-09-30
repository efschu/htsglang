"""AC: a store claim ends on a trailing (mamba) anchor BELOW the all-pages caps.

Metal (NF y3v 5327bdfa17, P log PP0, weg2-23-56, prompt 86598):

* 01:08:59 ``#1028B FETCH CAP n=5: kv=1350 claimed=716 caps={mamba: 1350,
  qsa_indexer: 716} ... anchors_in_range {mamba: (26, 1349), qsa_indexer:
  (1348, 1349)}`` -- the QSA index was missing at page 716 (a hole the L3-PAIR
  fix 7f6abb3a59 closes; y3v ran without it), the mamba boundary was taken
  over the whole KV prefix (1350) and the claim was their MIN: 716 pages,
  ending at page 715 where no anchor exists.
* 01:09:01 the #1416 clamp asked the SAME keys (``kv=716 claimed=0 ...
  mamba: (0, -1)``, identical KV presence = identical key form) and cut told
  to 0 -- correctly: there was no anchor below the hole. The read had loaded
  716 KV pages (45824 tokens, ~328 MB on PP0) for nothing.
* Same shape on y3u 5bedac26f1 PP0 00:36:06, weg2-0-5: ``claimed=47 caps=
  {mamba: 1996, qsa_indexer: 47}`` -> clamp ``completed=3008 anchored=0``.

The trailing-pages contract is "the last pages OF THE FINAL PREFIX": the
anchor is searched below the cap the all-pages pools leave. With an anchor
below the hole the claim keeps it (read instead of recompute); without one
the claim is 0 and no useless page is read.

Real HiCacheFile (tempdir), its real ``batch_exists_v2``; only the per-file
existence set is given. RED on c378ba4002, GREEN with AC.
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import (  # noqa: E402
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)

PAGES = 20
KEYS = [f"ac{i:04d}" for i in range(PAGES)]


def _store(d, *, mamba_at, qsa_upto, draft_upto=None):
    cfg = HiCacheStorageConfig(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
        is_mla_model=False, enable_storage_metrics=False, is_page_first_layout=False,
        model_name="unit-ac",
    )
    store = HiCacheFile(storage_config=cfg, file_path=d)
    present = {f"{store._get_component_key(k)}.bin" for k in KEYS}
    present |= {f"{store._get_component_key(KEYS[i], PoolName.MAMBA)}.bin" for i in mamba_at}
    present |= {f"{store._get_component_key(KEYS[i], PoolName.QSA_INDEXER)}.bin" for i in range(qsa_upto)}
    if draft_upto is not None:
        present |= {f"{store._get_component_key(KEYS[i], PoolName.DRAFT)}.bin" for i in range(draft_upto)}
    store._collect_existing_component_keys = lambda keys, transfers=None: present
    store._arena_kv_present_prefix = lambda keys: None
    return store


def _mamba():
    return PoolTransfer(name=PoolName.MAMBA, keys=["t"], hit_policy=PoolHitPolicy.TRAILING_PAGES)


def _qsa():
    return PoolTransfer(name=PoolName.QSA_INDEXER, hit_policy=PoolHitPolicy.ALL_PAGES,
                        indices_from_pool=PoolName.KV)


@pytest.mark.parametrize("order", ["mamba_first", "qsa_first"])
def test_the_claim_ends_on_the_deepest_anchor_below_the_qsa_hole(order):
    """RED on c378ba4002: 10 (the QSA hole at page 10; page 9 carries no
    anchor, the metal's 716). GREEN: 8, the anchor at page 7."""
    xs = [_mamba(), _qsa()] if order == "mamba_first" else [_qsa(), _mamba()]
    with tempfile.TemporaryDirectory() as d:
        res = _store(d, mamba_at={3, 7, 15, 19}, qsa_upto=10).batch_exists_v2(KEYS, xs)
    assert res.kv_hit_pages == 8
    assert res.extra_pool_hit_pages[PoolName.MAMBA] == 8


def test_no_anchor_below_the_hole_claims_nothing():
    """weg2-23-56 / weg2-0-5: every anchor sits beyond the hole. RED: 10
    stateless pages claimed and read. GREEN: 0, nothing read."""
    with tempfile.TemporaryDirectory() as d:
        res = _store(d, mamba_at={15, 19}, qsa_upto=10).batch_exists_v2(KEYS, [_mamba(), _qsa()])
    assert res.kv_hit_pages == 0


def test_without_an_all_pages_hole_the_claim_is_unchanged():
    with tempfile.TemporaryDirectory() as d:
        res = _store(d, mamba_at={3, 7, 15}, qsa_upto=PAGES).batch_exists_v2(KEYS, [_mamba(), _qsa()])
    assert res.kv_hit_pages == 16


def test_a_presence_only_pool_keeps_its_whole_prefix_boundary():
    """The draft page (caps_claim False) reports over the whole KV prefix;
    its consumer decides trim vs. cold from it -- it caps nothing."""
    draft = PoolTransfer(name=PoolName.DRAFT, keys=["t"], hit_policy=PoolHitPolicy.TRAILING_PAGES,
                         caps_claim=False)
    with tempfile.TemporaryDirectory() as d:
        res = _store(d, mamba_at={3, 7, 15, 19}, qsa_upto=10, draft_upto=PAGES).batch_exists_v2(
            KEYS, [draft, _mamba(), _qsa()]
        )
    assert res.kv_hit_pages == 8
    assert res.extra_pool_hit_pages[PoolName.DRAFT] == PAGES
