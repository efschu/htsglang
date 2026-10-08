"""SA: a short store read still reads the recurrent state at its deepest anchor.

Metal (NF y3v 5327bdfa17, P log, pdflip-46-98, prompt 54226, session c34ae69c6e):

* PP0 01:16:04 ``PDFLIP-READ-STAGES req=pdflip-46-98 pages=757 ... kv_ms=249`` --
  no ``extra_ms`` term: the hybrid pools were never read.
* PP0 01:16:06 ``#1028B FETCH CAP ... kv=757 claimed=716 ... anchors_in_range
  {mamba: (15, 715)}`` and ``#257 PREFETCH BELOW-ANCHOR read=48448 of 52672
  anchored=45824``; ``#1423 INSERT-PLACED ... inserted=45824``.
* PP1/PP2 ``PREFETCH-COMPLETE completed=45824 min_synced=45824`` (full reads
  of the told span, state included, ``extra_ms=105`` / ``73``).
* PP0 01:16:11 ``[#904 match-census] verdict=refused reached=45824 accepted=0
  refusers=MambaComponent:45824 why=MambaComponent:absent=45824`` ->
  ``#TF TOLD-FIDELITY told=45824 depth=45824 pp0_admissible=0`` -> told=0 on
  every rank; front ``PDFLIP-SERVED group=P ... cached_tokens=0 wall=37.68s``.

The root: ``HybridCacheController._page_transfer`` skips the extra pools for
EVERY read whose KV ended short, and #257 (b) cut the claim to the anchor the
store merely REPORTS -- a stateless prefix. The fix reads the state at the
deepest anchor inside the landed pages (plus the QSA index pages up to it),
and that read page is the rank's #257 vote.

Hermetic, CPU only. RED on 339476bee2 (the short read reads nothing; the cut
follows the presence answer), GREEN with SA.
"""

from __future__ import annotations

import importlib.util
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt import rank_role  # noqa: E402
from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers import cache_controller as cc  # noqa: E402
from flliper.srt.mem_cache.hicache_phase_binding import binding_state  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import (  # noqa: E402
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from flliper.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (  # noqa: E402
    HybridCacheController,
)
from flliper.srt.mem_cache.short_read_anchor import state_depth_mismatch  # noqa: E402

# the metal in pages of 64: 823 probed, 757 landed, deepest anchor at page 715
PROBED, LANDED, ANCHOR_IDX = 823, 757, 715
KEYS = [f"sa{i:04d}" for i in range(PROBED)]
MAMBA_AT = {35, 99, 400, 715, 822}  # 822 = the anchor past the landed pages
PAGE = 1  # one token per page keeps the host indices small


class _Store:
    """The store's answer, by the file backend's rule: KV present for the
    landed pages, the trailing mamba boundary = deepest landed page carrying
    a blob, QSA (ALL_PAGES) present everywhere."""

    abstains_from_claim_vote = False

    def __init__(self, fail_read=False):
        self.gets = []
        self.fail_read = fail_read

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        final = len(keys)
        for t in pool_transfers or []:
            if t.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                b = max((i + 1 for i in range(len(keys)) if KEYS.index(keys[i]) in MAMBA_AT), default=0)
                final = min(final, b)
        return types.SimpleNamespace(kv_hit_pages=final, extra_pool_hit_pages={})

    def batch_get_v2(self, transfers, extra_info=None):
        self.gets.append({t.name: (list(t.keys or []), int(t.host_indices.numel())) for t in transfers})
        return {t.name: [not self.fail_read] * len(t.keys or []) for t in transfers}


def _mamba():
    return PoolTransfer(name=PoolName.MAMBA, keys=[KEYS[-1]], hit_policy=PoolHitPolicy.TRAILING_PAGES,
                        host_indices=torch.tensor([7]))


def _qsa():
    return PoolTransfer(name=PoolName.QSA_INDEXER, keys=None, hit_policy=PoolHitPolicy.ALL_PAGES,
                        indices_from_pool=PoolName.KV)


def _ctl(store, tp_group=None, consumer=True):
    me = object.__new__(HybridCacheController)
    me.storage_backend = store
    me.page_size = PAGE
    me.tp_group = tp_group
    me.short_read_anchor_consumer = consumer
    return me


def _op(transfers):
    return types.SimpleNamespace(
        request_id="pdflip-46-98", hash_value=list(KEYS), completed_tokens=0,
        host_indices=torch.arange(PROBED * PAGE), pool_transfers=transfers,
        pool_storage_result=PoolTransferResult.empty(), pool_transfers_done=False,
        _pdflip_short_anchor_pages=None,
    )


@pytest.fixture
def short_kv(monkeypatch):
    """The KV read breaks at page 757 (pages 757.. neither in L2 nor on disk)."""
    monkeypatch.setattr(rank_role, "form_a_worker_holds_kv", lambda: False)
    monkeypatch.setattr(cc.HiCacheController, "_page_transfer",
                        lambda self, op: setattr(op, "completed_tokens", LANDED * PAGE))


# ------------------------------------------------------------ the controller
def test_pdflip_46_98_the_short_read_reads_the_state_at_its_deepest_anchor(short_kv):
    """RED on 339476bee2: nothing is read (store.gets == []), the operation
    carries no read anchor and the mamba hit stays 0 -- the metal's
    READ-STAGES without extra_ms. GREEN: the state at page 715 is read."""
    store = _Store()
    m, q = _mamba(), _qsa()
    op = _op([m, q])
    HybridCacheController._page_transfer(_ctl(store), op)
    assert len(store.gets) == 1
    got = store.gets[0]
    assert got[PoolName.MAMBA][0] == [KEYS[ANCHOR_IDX]]        # the anchor page, not 822
    assert got[PoolName.QSA_INDEXER] == (KEYS[: ANCHOR_IDX + 1], (ANCHOR_IDX + 1) * PAGE)
    assert op._pdflip_short_anchor_pages == ANCHOR_IDX + 1        # 716 pages = 45824 tokens
    assert op.pool_storage_result.extra_pool_hit_pages[PoolName.MAMBA] == 1
    assert op.pool_transfers_done


def test_a_failed_state_read_names_no_anchor(short_kv):
    store = _Store(fail_read=True)
    op = _op([_mamba()])
    HybridCacheController._page_transfer(_ctl(store), op)
    assert op._pdflip_short_anchor_pages == 0
    assert op.pool_storage_result.extra_pool_hit_pages.get(PoolName.MAMBA, 0) == 0


def test_no_anchor_inside_the_landed_pages_reads_nothing(short_kv, monkeypatch):
    monkeypatch.setattr(cc.HiCacheController, "_page_transfer",
                        lambda self, op: setattr(op, "completed_tokens", 30 * PAGE))
    store = _Store()
    op = _op([_mamba()])
    HybridCacheController._page_transfer(_ctl(store), op)
    assert store.gets == [] and op._pdflip_short_anchor_pages == 0


def test_a_tp_group_keeps_the_old_skip(short_kv, monkeypatch):
    """The MIN over a TP group's hit pages must never mix states of
    different depths: there the short read reads nothing, as before."""
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 3)
    store = _Store()
    op = _op([_mamba()])
    HybridCacheController._page_transfer(_ctl(store, tp_group=object()), op)
    assert store.gets == [] and op._pdflip_short_anchor_pages is None


def test_the_switch_off_is_the_old_skip(short_kv):
    store = _Store()
    op = _op([_mamba()])
    with envs.FLLIPER_PDFLIP_ENABLE_SHORT_READ_ANCHOR.override(False):
        HybridCacheController._page_transfer(_ctl(store), op)
    assert store.gets == [] and op._pdflip_short_anchor_pages is None


def test_a_tree_that_does_not_cut_to_the_read_anchor_keeps_the_old_skip(short_kv):
    """HiMambaRadixCache (same controller) inserts the whole landed span and
    files the state at its END: a state read at page 715 would sit on the
    node at 757 there. Only a consumer tree (UnifiedRadixCache) arms SA."""
    store = _Store()
    op = _op([_mamba()])
    HybridCacheController._page_transfer(_ctl(store, consumer=False), op)
    assert store.gets == [] and op._pdflip_short_anchor_pages is None
    assert HybridCacheController.short_read_anchor_consumer is False


def test_the_unified_stack_arms_its_controller():
    from flliper.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa

    ctl = object.__new__(HybridCacheController)
    ctl.layer_done_counter = object()
    cache = types.SimpleNamespace(components={})
    kv = types.SimpleNamespace(register_layer_transfer_counter=lambda c: None)
    result = types.SimpleNamespace(
        host_pool_group=None, cache_controller=ctl, component_host_pools={}, sidecars=[],
        register_req_to_token_counter=False,
    )
    try:
        hpa._apply_stack_result(cache, kv, None, result)
    except Exception:  # noqa: BLE001 - the attach log after the wiring needs a real stack
        pass
    assert ctl.short_read_anchor_consumer is True


def test_a_full_read_is_unchanged(short_kv, monkeypatch):
    monkeypatch.setattr(cc.HiCacheController, "_page_transfer",
                        lambda self, op: setattr(op, "completed_tokens", PROBED * PAGE))
    store = _Store()
    op = _op([_mamba()])
    HybridCacheController._page_transfer(_ctl(store), op)
    assert store.gets[0][PoolName.MAMBA][0] == [KEYS[-1]]
    assert op._pdflip_short_anchor_pages is None


# ------------------------------------------------------------ the reap (#257 cut)
_spec = importlib.util.spec_from_file_location(
    "_t_1157_sa", os.path.join(os.path.dirname(__file__), "test_1157_reaper_prices_requested_span.py")
)
h1157 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h1157)
REQ, SPAN, READ = h1157.REAP_REQ, h1157.REAP_TOKENS, 10


class _Presence:
    """The store's presence answer: an anchor at page 8 (it only SAW it)."""

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        return types.SimpleNamespace(kv_hit_pages=len(keys), extra_pool_hit_pages={PoolName.MAMBA: 8})


def _reap(read_anchor_pages):
    binding_state().reset()
    cache, op = h1157._reap_scenario(probed=False)
    op.hash_value = [f"h{i}" for i in range(SPAN)]
    op.probed_hit_tokens = SPAN
    op.increment(READ)
    op.mark_terminate()
    op._pdflip_short_anchor_pages = read_anchor_pages
    cache.cache_controller.storage_backend = _Presence()
    cache.cache_controller._presence_pool_transfers = lambda: [types.SimpleNamespace(name=PoolName.MAMBA)]
    cache.check_prefetch_progress(REQ)
    return int(cache.prefetch_loaded_tokens_by_reqid[REQ])


def test_the_cut_lands_on_the_page_whose_state_was_read():
    """RED on 339476bee2: 8 (the presence answer, a page with no state on the
    host). GREEN: 6, the page the controller read the state at."""
    try:
        assert _reap(6) == 6
    finally:
        binding_state().reset()


def test_no_read_state_no_claim():
    try:
        assert _reap(0) == 0
    finally:
        binding_state().reset()


def test_without_the_arm_the_presence_answer_stands():
    try:
        assert _reap(None) == 8
    finally:
        binding_state().reset()


# ------------------------------------------------------------ the commit guard
def test_a_state_is_committed_only_at_the_cut():
    hv = [f"k{i}" for i in range(12)]
    m = PoolTransfer(name=PoolName.MAMBA, keys=["k9"], hit_policy=PoolHitPolicy.TRAILING_PAGES)
    q = PoolTransfer(name=PoolName.QSA_INDEXER, keys=hv[:10], hit_policy=PoolHitPolicy.ALL_PAGES)
    hits = {PoolName.MAMBA: 1, PoolName.QSA_INDEXER: 10}
    assert state_depth_mismatch(transfers=[m, q], hash_value=hv, cut_pages=10, hit_pages=hits) == []
    assert state_depth_mismatch(transfers=[m, q], hash_value=hv, cut_pages=8, hit_pages=hits) == [PoolName.MAMBA]
    assert state_depth_mismatch(transfers=[m], hash_value=hv, cut_pages=0, hit_pages=hits) == [PoolName.MAMBA]
    # an unread state is nobody's concern
    assert state_depth_mismatch(transfers=[m], hash_value=hv, cut_pages=8, hit_pages={}) == []
