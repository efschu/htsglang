"""AP L15-12c-B (1001): the wake ACTS on the group verdict "fallback".

The fence tail of ``resume_memory_occupation`` computes the group's L15
verdict ("none" / "fallback" / "hold"; plan L15-12-PART3-PLAN.md sections 3
and 6).  Part 3 makes every rank ACT on it:

  * "fallback": every rank runs the default restore shape WITHOUT the
    re-reservation -- ``tree_cache.reset()`` + ``req_to_token_pool.clear()``
    (full, no keep; the flush shape has no separate mamba allocator, the
    mamba rows go with the req clear) + ``token_to_kv_pool_allocator.clear()``
    + the TMS keep-set clear.  Ranks that kept nothing experience the plain
    flush; ranks that kept something discard the held bytes (L2 stayed the
    authority).  Idempotent: a second drop changes nothing observable and
    reports zero dropped.
  * "deferred" (a sibling's kv resume refused, W114): NO rank restores,
    refills or drops -- the hold stays armed.  Touching the still-paused
    pool of a refused rank is the xsn408 fault class.
  * "none" and master off: zero new calls (byte-identical wake).

Hermetic: the methods run UNBOUND on a FakeSelf with fake pools and a fake
memory_saver_adapter; the KV tensor is real (CPU) so the keep-set buffer
walk is exercised on actual values, not mocks.
"""

import pathlib
import sys
from types import SimpleNamespace

import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from flliper.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402

WU = wu.SchedulerWeightUpdaterManager


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class _TreeCache:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class _ReqToTokenPool:
    def __init__(self):
        self.clears = 0

    def clear(self, *a, **k):
        self.clears += 1


class _Allocator:
    def __init__(self):
        self.clears = 0

    def clear(self):
        self.clears += 1


class _Adapter:
    """Records one entry per set_keep_byte_spans call; the helper must call
    it with the EMPTY tuple (the keep-set clear, not a re-arm)."""

    def __init__(self):
        self.sets = []

    def alloc_info_ok(self, base):
        return True

    def set_keep_byte_spans(self, base, spans):
        self.sets.append((base, tuple(spans)))


def _mk_sched():
    kv = torch.zeros(4, 8)
    return SimpleNamespace(
        tree_cache=_TreeCache(),
        req_to_token_pool=_ReqToTokenPool(),
        token_to_kv_pool_allocator=_Allocator(),
        memory_saver_adapter=_Adapter(),
        _kv_pools_for_flush=lambda: [
            SimpleNamespace(k_buffer=kv, v_buffer=None, kv_buffer=None)
        ],
    )


def _mk_manifest(slots):
    return SimpleNamespace(spans=[SimpleNamespace(slots=list(slots))])


class FakeSelf:
    """The unbound-method fake-self style of test_pdflip_l15_wake_restore_1001:
    the production methods run verbatim against the instance attributes."""

    _l15_clear_tms_keep_spans = WU._l15_clear_tms_keep_spans
    _l15_fallback_drop = WU._l15_fallback_drop
    _l15_wake_act = WU._l15_wake_act
    _l15_fence_manifest = WU._l15_fence_manifest

    def __init__(self, manifest=None):
        self._l15_wake_manifest = manifest


def _state_flags(sched):
    """Observable end state: WHAT the pools look like, not how often the
    fakes were called (a repeat drop must not change it)."""
    return dict(
        reset=sched.tree_cache.resets > 0,
        req=sched.req_to_token_pool.clears > 0,
        alloc=sched.token_to_kv_pool_allocator.clears > 0,
        keep=len(sched.memory_saver_adapter.sets) > 0,
    )


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_fallback_drops_trio_once_per_rank_over_a_3_rank_group():
    held = ([0, 3], [5, 6, 7], [])       # rank 0 holds 2 (0 is padding-excl->1)
    # slot 0 is the allocator padding slot: excluded from the dropped count
    expect = (1, 3, 0)
    ranks = [(FakeSelf(_mk_manifest(h)), _mk_sched()) for h in held]
    got = []
    for self_, sched in ranks:
        got.append(WU._l15_wake_act(self_, sched, "fallback",
                                    group_ok=True, master_on=True))
    for (self_, sched), n, exp in zip(ranks, got, expect):
        assert sched.tree_cache.resets == 1
        assert sched.req_to_token_pool.clears == 1
        assert sched.token_to_kv_pool_allocator.clears == 1
        # keep-set cleared exactly once, with the EMPTY span set
        assert [s for _, s in sched.memory_saver_adapter.sets] == [()]
        assert n == exp


def test_deferred_refused_sibling_touches_nothing_and_keeps_hold():
    self_, sched = FakeSelf(_mk_manifest([4, 8])), _mk_sched()
    n = WU._l15_wake_act(self_, sched, "deferred", group_ok=False,
                         master_on=True)
    assert n == 0
    assert _state_flags(sched) == dict(reset=False, req=False, alloc=False,
                                       keep=False)
    # the hold stays ARMED: the manifest is not consumed
    assert self_._l15_wake_manifest is not None


def test_refusal_outranks_fallback_verdict():
    # a refused group defers even where the fingerprint verdict said fallback
    self_, sched = FakeSelf(_mk_manifest([4])), _mk_sched()
    n = WU._l15_wake_act(self_, sched, "fallback", group_ok=False,
                         master_on=True)
    assert n == 0
    assert _state_flags(sched) == dict(reset=False, req=False, alloc=False,
                                       keep=False)
    assert self_._l15_wake_manifest is not None


def test_verdict_none_touches_nothing():
    self_, sched = FakeSelf(_mk_manifest([1])), _mk_sched()
    n = WU._l15_wake_act(self_, sched, "none", group_ok=True, master_on=True)
    assert n == 0
    assert _state_flags(sched) == dict(reset=False, req=False, alloc=False,
                                       keep=False)


def test_second_fallback_changes_nothing():
    self_, sched = FakeSelf(_mk_manifest([2, 9])), _mk_sched()
    n1 = WU._l15_fallback_drop(self_, sched)
    after = _state_flags(sched)
    n2 = WU._l15_fallback_drop(self_, sched)
    assert after == dict(reset=True, req=True, alloc=True, keep=True)
    assert _state_flags(sched) == after
    assert (n1, n2) == (2, 0)  # second drop: already empty, nothing held


def test_master_off_no_call():
    self_, sched = FakeSelf(_mk_manifest([1])), _mk_sched()
    n = WU._l15_wake_act(self_, sched, "fallback", group_ok=True,
                         master_on=False)
    assert n == 0
    assert _state_flags(sched) == dict(reset=False, req=False, alloc=False,
                                       keep=False)


# --------------------------------------------------------------------------
# F11 (lead review 01.10.): the fence must see the manifest the wake restore
# already read-and-unlinked -- a second load_for_wake on the same path always
# returns None and pins _l15_fp to None, so the part-2 verdict can never be
# "hold".  The stashed field wins; only a rank that did not consume the
# record (cap 0 etc.) asks the file.
# --------------------------------------------------------------------------

def test_fence_manifest_prefers_the_stashed_record():
    sentinel = _mk_manifest([1, 2])
    self_ = FakeSelf(sentinel)
    got = WU._l15_fence_manifest(self_, "/nonexistent/l15_manifest.json")
    assert got is sentinel


def test_fence_manifest_falls_back_to_the_file_when_not_stashed():
    self_ = FakeSelf(None)
    got = WU._l15_fence_manifest(self_, "/nonexistent/l15_manifest.json")
    assert got is None


# --------------------------------------------------------------------------
# F10 (lead review 01.10.): the keep-set walk must reach the hybrid pool's
# sub-pools -- on the 27B the _kv_pools_for_flush() entry is a
# HybridLinearKVPool and its buffers live on .full_kv_pool (and .swa_kv_pool
# where present), not on the entry itself.
# --------------------------------------------------------------------------

def test_hybrid_pool_keep_set_cleared_per_sub_pool_base():
    fk = torch.zeros(2, 8)
    fv = torch.zeros(2, 8)
    hybrid = SimpleNamespace(
        k_buffer=None, v_buffer=None, kv_buffer=None,
        full_kv_pool=SimpleNamespace(k_buffer=fk, v_buffer=fv,
                                     kv_buffer=None),
        swa_kv_pool=None,
    )
    sched = _mk_sched()
    sched._kv_pools_for_flush = lambda: [hybrid]
    self_ = FakeSelf(None)
    cleared = WU._l15_clear_tms_keep_spans(self_, sched)
    assert cleared == 2  # k and v bases of the full_kv_pool
    spans = [s for _, s in sched.memory_saver_adapter.sets]
    assert spans == [(), ()]
    ptrs = {b.data_ptr() for b, _ in sched.memory_saver_adapter.sets}
    assert ptrs == {fk.data_ptr(), fv.data_ptr()}
