"""Rank form (28.09.): ONE predicate for "this rank holds KV token rows only".

Agreed with 27B (rank-form review, answers 3/4/6): ``rank_role.kv_only_rank()``
= dense weight share 0 AND owns token rows, from the installed Form A plan
(#239 token cut, metal form [0,32,32] -> S=2, TP1 [0,1), TP2 [1,2)) OR from
the weightless-KV lane (every rank but the head). Its readers ask it, never a
backend name:

* ``claim_vote_min_only`` / ``split_host_state_pools`` (B5, 6d685fb1ae):
  the KV-only rank answers for its KV rows only, the mamba anchor is the
  weight rank's;
* the F14 page window (``#239 F14 KV-WORKER-WINDOW``, marker text unchanged
  -- the launcher counts it);
* the RankState record: "holds GDN state" is False on a KV-only rank however
  its placeholder mamba pool is sized (the lane sizes it to one slot);
* tail adoption: ``held_shapes`` reports no GDN rows on a KV-only rank.

On the NF form (lane off) the predicate IS ``form_a_worker_holds_kv`` --
pinned below over the installed plan, so B5 did not move. The lane cases are
RED on d44f9dbccd/7bd3541c4f (the readers asked the Form A predicate only),
GREEN with the change.
"""

from __future__ import annotations

import contextlib
import inspect
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flliper.srt import rank_role
from flliper.srt.distributed import utils as du
from flliper.srt.managers import cache_controller as cc
from flliper.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from flliper.srt.pdflip import tail_adopt as ta

ROLES = ("host", "worker", "worker")
#: the metal cut [0, 32, 32] reduced (0, 1, 1): S=2, TP1 [0,1), TP2 [1,2)
CUT_BOUNDS = {0: (2, 0, 0), 1: (2, 0, 1), 2: (2, 1, 2)}


@pytest.fixture(autouse=True)
def _classic_process():
    """Each test starts on a classic process -- no Form A plan, no lane -- and
    leaves the installed plan as it found it. 27B 28.09.: a plan another
    suite left installed made the classic case order-dependent."""
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = None, 0
    try:
        with mock.patch.object(du, "_WEIGHTLESS_KV_HEAD_RANK", None):
            yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


@contextlib.contextmanager
def _form_a(rank, *, bounds=None):
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    rank_role.set_form_a_role_plan(rank_role.RankRolePlan(ROLES), rank)
    b = CUT_BOUNDS[rank] if bounds is None else bounds
    try:
        with mock.patch.object(du, "uneven_dcp_owner_bounds", lambda: b):
            yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


@contextlib.contextmanager
def _lane(rank, *, head=0, bounds="even"):
    """The weightless-KV lane installed, this process = ``rank``."""
    b = None if bounds == "even" else bounds
    with mock.patch.object(du, "_WEIGHTLESS_KV_HEAD_RANK", head), mock.patch(
        "flliper.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_rank=rank),
    ), mock.patch.object(du, "uneven_dcp_owner_bounds", lambda: b):
        yield


# ---------------------------------------------------------------- predicate
def test_a_classic_boot_has_no_kv_only_rank():
    assert rank_role.installed_role_plan() is None and not du.weightless_kv_active()
    assert rank_role.kv_only_rank() is False


def test_on_the_nf_cut_it_is_exactly_form_a_worker_holds_kv():
    for r in (0, 1, 2):
        with _form_a(r):
            assert rank_role.kv_only_rank() == rank_role.form_a_worker_holds_kv() == (r != 0)
    with _form_a(1, bounds=(2, 0, 0)):  # a worker with share 0 stays byteless
        assert rank_role.kv_only_rank() is False


def test_on_the_lane_every_rank_but_the_head_is_kv_only():
    with _lane(0):
        assert rank_role.kv_only_rank() is False  # the head holds the weights
    for r in (1, 2):
        with _lane(r):
            assert rank_role.kv_only_rank() is True  # even DCP: owns rows
        with _lane(r, bounds=(3, r, r + 1)):
            assert rank_role.kv_only_rank() is True  # weighted
    with _lane(1, bounds=(3, 1, 1)):  # an empty owner range owns no rows
        assert rank_role.kv_only_rank() is False


# ------------------------------------------------------- claim vote / B5 split
def _ctl():
    return SimpleNamespace(storage_backend=SimpleNamespace(abstains_from_claim_vote=False))


def _transfers():
    return [
        PoolTransfer(name=PoolName.KV, keys=["k"]),
        PoolTransfer(name=PoolName.MAMBA, keys=["m"], hit_policy=PoolHitPolicy.TRAILING_PAGES),
    ]


def test_a_lane_kv_rank_votes_min_only_and_hands_the_anchor_to_the_head():
    with _lane(1):
        assert cc.claim_vote_min_only(_ctl()) is True  # base: False (Form A only)
        own, host = cc.split_host_state_pools(_ctl(), _transfers())
        assert [t.name for t in own] == [PoolName.KV]
        assert [t.name for t in host] == [PoolName.MAMBA]
    with _lane(0):
        assert cc.claim_vote_min_only(_ctl()) is False
        assert cc.split_host_state_pools(_ctl(), _transfers())[1] == []


def test_the_nf_cut_worker_votes_as_before():
    with _form_a(1):
        assert cc.claim_vote_min_only(_ctl()) is True
    with _form_a(0):
        assert cc.claim_vote_min_only(_ctl()) is False


# ------------------------------------------------------------ RankState input
def _with_mamba(pool=object()):
    return SimpleNamespace(mem_pool_device_hybrid=SimpleNamespace(mamba_pool=pool))


def test_a_kv_only_rank_holds_no_gdn_state_whatever_its_pool():
    with _lane(1):
        assert cc.rank_holds_gdn_state(_with_mamba()) is False  # the placeholder
    with _lane(0):
        assert cc.rank_holds_gdn_state(_with_mamba()) is True
        assert cc.rank_holds_gdn_state(_with_mamba(None)) is False
    with _form_a(2):
        assert cc.rank_holds_gdn_state(_with_mamba()) is False


def test_the_record_does_not_demand_a_blob_of_a_lane_kv_rank():
    from flliper.srt.pdflip.rank_state import build_rank_state

    with _lane(2):
        st = build_rank_state(
            group="D", tp_rank=2, tp_size=3, pp_rank=0, pp_size=1,
            form_a_worker=rank_role.this_rank_is_form_a_worker(),
            canonical_on=True, canonical_kv_built=True, canonical_blob_built=False,
            has_mamba_pool=cc.rank_holds_gdn_state(_with_mamba()),
            page_size=64, owner_ctx=None, seq=1,
        )
    assert st.kv_page_applicable and st.kv_page_active
    assert st.gdn_blob_applicable is False  # base: True -> W7 'blob applicable, not active'


def test_the_storage_config_reads_the_predicate_and_keeps_the_marker():
    from flliper.srt.pdflip.launcher import FORM_A_KV_WORKER_CANONICAL_MARKER

    src = inspect.getsource(cc.HiCacheController)
    assert "if kv_only_rank() and canonical_on:" in src
    assert "has_mamba_pool=rank_holds_gdn_state(self)" in src
    # the launcher counts this exact text; the predicate must not reword it
    assert FORM_A_KV_WORKER_CANONICAL_MARKER.split(": ", 1)[0] in src
    assert '"#239 F14 KV-WORKER-WINDOW: Form A worker owns token rows %s of "' in src


# --------------------------------------------------------------- tail adoption
def _pools():
    from flliper.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from flliper.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    kv = object.__new__(QSATokenToKVPool)
    kv.full_kv_pool = SimpleNamespace(k_buffer=[torch.zeros(8, 2, 4)], v_buffer=[torch.zeros(8, 2, 4)])
    kv.full_attention_layer_id_mapping = {3: 0}
    kv.qsa_compress_ratio = 4
    kv.qsa_compressed_k_buffer_pool = []
    kv.qsa_key_state_buffer_pool = [torch.zeros(8, 1, 4)]
    kv.qsa_rope_position_buffer = torch.zeros(8, 3, dtype=torch.int64)
    rp = object.__new__(HybridReqToTokenPool)
    rp.mamba_map = {0: 0, 1: 1}
    # the lane's one-slot placeholder: shape[1:] is the FULL state shape
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=torch.zeros(2, 1, 4, 4, 4), conv=[torch.zeros(2, 1, 6, 3)],
    ))
    return kv, rp


def test_held_shapes_reports_no_gdn_rows_on_a_kv_only_rank():
    with _lane(1), mock.patch.object(du, "uneven_dcp_active", lambda: False):
        held = ta.held_shapes(*_pools())
    assert held.fa and held.gdn == {}  # base: {0: [...], 1: [...]} from the placeholder


def test_held_shapes_on_the_weight_rank_is_unchanged():
    with _lane(0), mock.patch.object(du, "uneven_dcp_active", lambda: False):
        held = ta.held_shapes(*_pools())
    assert sorted(held.gdn) == [0, 1]
