"""H99 (rc9o, boot dkrnfbar1rc9o09260808 on 2eed285057, rid weg2-33-33):
on a Form A D group the #580 prefetch span is the attention host's.

MEASURED (D log 85540-85632, 08:29:00): ``#1042 EXTENT LIFECYCLE set
rid=weg2-33-33 extent=32768`` on TP0, ``extent=26368`` on TP1/TP2, then on
TP1/TP2 ``HiCacheCollectiveDesyncError: W65 Weg2PrefetchSpanSplit ...
DIFFERENT pre-vote spans (min=320 max=6720) ... group length of 320``, raised
from ``_add_request_to_queue -> _prefetch_kvcache -> prefetch_from_storage``
-- at INTAKE, before any admission, so H98's follow (admission only) never
saw the rid. The span each rank voted is ``fill[_matched_len:_match_end]``
with ``_matched_len`` from the intake re-match of its OWN tree: TP0's host
anchor at 32768, the expert workers' byteless shadow anchors at 26368.

Driven through the REAL ``UnifiedRadixCache.prefetch_from_storage`` on three
simulated ranks joined by a mock gloo group (the RU harness), each rank with
its Form A role. RED on 2eed285057: W65 on every rank. GREEN with H99: the
workers abstain in the span pair (x22), every rank registers the same group
length, and the workers' request bookkeeping takes the host's prefix/span.
"""

from __future__ import annotations

import math
import threading
import types
import unittest
from typing import Dict, List
from unittest import mock

import torch

from sglang.srt import rank_role
from sglang.srt.managers import tp_match_floor as m
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

WORLD = 3
THRESHOLD = 256
ROLES = ("host", "worker", "worker")
SWITCH = "SGLANG_WEG2_ENABLE_FORM_A_TP0_FOLLOW"
SPAN_ATTR = "_tp_match_floor_prefetch_group_span"
NF_D = types.SimpleNamespace(rank_tp_ratio=[1, 0, 0], hicache_size=4)

MATCH_END = 33088
PROMPT = list(range(MATCH_END))
TP0_PREFIX = 32768
WORKER_PREFIX = 26368


def _gloo_chunk_bytes(t: torch.Tensor) -> int:
    return max(1, math.ceil(t.numel() / (2 * WORLD))) * t.element_size()


class CollectiveMismatch(RuntimeError):
    pass


class MockGlooGroup:
    """Three ranks in threads; one sequence of collectives; gloo's check."""

    def __init__(self, timeout: float = 3.0):
        self.barrier = threading.Barrier(WORLD, timeout=timeout)
        self.slots: Dict[int, tuple] = {}
        self.log: Dict[int, List[tuple]] = {r: [] for r in range(WORLD)}
        self.errors: List[str] = []
        self._lock = threading.Lock()

    def all_reduce(self, rank, tensor, op, label):
        self.log[rank].append((label, tensor.numel(), str(tensor.dtype)))
        with self._lock:
            self.slots[rank] = (label, tensor.clone())
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            raise CollectiveMismatch(f"rank {rank} alone in {label}")
        entries = [self.slots[r] for r in range(WORLD)]
        if len({_gloo_chunk_bytes(e[1]) for e in entries}) > 1:
            with self._lock:
                self.errors.append(f"size mismatch {[(e[0], e[1].numel()) for e in entries]}")
            self.barrier.abort()
            raise CollectiveMismatch(self.errors[-1])
        stack = torch.stack([e[1] for e in entries])
        red = stack.max(dim=0).values if op == torch.distributed.ReduceOp.MAX else stack.min(dim=0).values
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            raise CollectiveMismatch(f"rank {rank} lost the group after {label}")
        tensor.copy_(red.to(tensor.dtype))


_ROLE = threading.local()


def _thread_is_worker() -> bool:
    return bool(getattr(_ROLE, "worker", False))


def run_ranks(fn):
    results, errors = {}, {}

    def _t(r):
        _ROLE.worker = ROLES[r] == "worker"
        try:
            results[r] = fn(r)
        except BaseException as e:  # noqa: BLE001
            errors[r] = e

    ts = [threading.Thread(target=_t, args=(r,)) for r in range(WORLD)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    return results, errors


class _HostPool:
    def alloc(self, n):
        return torch.arange(n)

    def available_size(self):
        return 1 << 20


class _Controller:
    def __init__(self):
        self.mem_pool_host = _HostPool()
        self.prefetch_tokens_occupied = 0
        self.released = 0

    def prefetch_rate_limited(self):
        return False

    def prefetch(self, req_id, host_indices, prefetch_key, *a, **kw):
        return types.SimpleNamespace(
            request_id=req_id, host_indices=host_indices, hash_value=[],
            completed_tokens=0, start_time=0.0, is_terminated=lambda: False,
        )

    def append_host_mem_release(self, host_indices=None, **kw):
        self.released += 0 if host_indices is None else len(host_indices)


def _host_node():
    return types.SimpleNamespace(
        key=None, backuped=True, parent=None,
        get_last_hash_value=lambda: None,
        get_prefix_hash_values=lambda parent: None,
    )


def _carrier(rank, group):
    cache = types.SimpleNamespace(
        enable_storage=True, cache_controller=_Controller(), tp_world_size=WORLD,
        is_eagle=False, page_size=1, prefetch_threshold=THRESHOLD,
        prefetch_stop_policy="wait_complete", ongoing_prefetch={},
        _retired_prefetch=[], _retired_prefetch_attempts={}, _retired_prefetch_recompute=0,
        _components_tuple=(),
        inc_host_lock_ref=lambda node: types.SimpleNamespace(to_dec_params=lambda: ("dec", node)),
        dec_host_lock_ref=lambda node, params: None,
        evict_host=lambda n: 0,
        _build_sidecar_transfers=lambda phase, kv_xfer, comp_xfers: [],
        _all_reduce_attn_groups=lambda t, op, label="hicache": group.all_reduce(rank, t, op, label),
    )
    for name in (
        "prefetch_from_storage", "_retire_ongoing_prefetch", "_prefetch_line_terms",
        "_log_prefetch_refused", "_log_prefetch_truncated", "_weg2_extent_topup",
        "_hicache_prefetch_symmetric",
    ):
        setattr(cache, name, types.MethodType(getattr(UnifiedRadixCache, name), cache))
    return cache


def _registered_len(cache, rid):
    rec = cache.ongoing_prefetch.get(rid)
    if rec is None:
        return None
    return len(rec[1]), len(rec[2])  # prefetch_key, host_indices


class _Env:
    """NF D form (#580 vote in force), Form A plan installed, per-thread role."""

    def __init__(self, switch=True):
        self.switch = switch
        self.stack = []

    def __enter__(self):
        from sglang.srt.environ import envs

        self.prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
        rank_role.set_form_a_role_plan(rank_role.RankRolePlan(ROLES), 0)
        for p in (
            mock.patch("sglang.srt.runtime_context.get_server_args", return_value=NF_D),
            mock.patch.object(urc, "uneven_dcp_active", return_value=False),
            mock.patch.object(rank_role, "this_rank_is_form_a_worker", _thread_is_worker),
        ):
            p.__enter__()
            self.stack.append(p)
        field = getattr(envs, SWITCH, None)
        if field is not None:
            ov = field.override(self.switch)
            ov.__enter__()
            self.stack.append(ov)
        return self

    def __exit__(self, *a):
        for p in reversed(self.stack):
            p.__exit__(*a)
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = self.prev


def _intake(spans, rid, switch=True):
    """One intake pass: every rank enters the vote with its own span."""
    group = MockGlooGroup()
    caches = {}

    def _rank(r):
        caches[r] = c = _carrier(r, group)
        c.prefetch_from_storage(rid, _host_node(), PROMPT[spans[r]:], last_hash=None, prefix_keys=None)
        return _registered_len(c, rid)

    with _Env(switch):
        results, errors = run_ranks(_rank)
    return caches, results, errors, group


RC9O = {0: TP0_PREFIX, 1: WORKER_PREFIX, 2: WORKER_PREFIX}


class TestRc9oPrefetchSpanIsTheHosts(unittest.TestCase):
    def test_rc9o_no_w65_and_one_registration_length(self):
        _c, results, errors, group = _intake(RC9O, "weg2-33-33")
        self.assertEqual(
            errors, {},
            "rc9o: the expert workers' shadow-anchor spans (6720) split the #580 "
            f"vote against TP0's 320 -> W65: {errors}",
        )
        self.assertEqual(results, {0: (320, 320), 1: (320, 320), 2: (320, 320)}, results)
        self.assertEqual(group.errors, [])

    def test_workers_release_their_surplus_rows(self):
        caches, _r, errors, _g = _intake(RC9O, "weg2-33-33")
        self.assertEqual(errors, {})
        self.assertEqual({r: c.cache_controller.released for r, c in caches.items()},
                         {0: 0, 1: 6400, 2: 6400})
        self.assertEqual({r: c.cache_controller.prefetch_tokens_occupied for r, c in caches.items()}
                         , {0: 320, 1: 320, 2: 320})

    def test_worker_bookkeeping_takes_the_host_prefix(self):
        caches, _r, errors, _g = _intake(RC9O, "weg2-33-33")
        self.assertEqual(errors, {})
        with _Env():
            _ROLE.worker = True
            try:
                req = types.SimpleNamespace(
                    rid="weg2-33-33", _prefetch_span_tokens=MATCH_END - WORKER_PREFIX,
                    _prefetch_registered_prefix_len=WORKER_PREFIX,
                )
                got = m.adopt_host_prefetch_span(caches[1], req, MATCH_END)
            finally:
                _ROLE.worker = False
        self.assertEqual(got, TP0_PREFIX)
        self.assertEqual(req._prefetch_span_tokens, MATCH_END - TP0_PREFIX)
        self.assertEqual(req._prefetch_registered_prefix_len, TP0_PREFIX)
        self.assertEqual(getattr(caches[1], SPAN_ATTR), {}, "consumed once")

    def test_worker_shorter_span_truncates_the_group_uniformly(self):
        # A worker whose shadow anchor sits DEEPER than the host's: its length
        # caps the group (it cannot register more than its key) -- a uniform
        # group truncation, never W65.
        _c, results, errors, _g = _intake({0: 26368, 1: 32768, 2: 26368}, "r-deep")
        self.assertEqual(errors, {}, errors)
        self.assertEqual(set(results.values()), {(320, 320)}, results)


class TestSwitchOffIsBase(unittest.TestCase):
    def test_switch_off_keeps_w65(self):
        _c, _r, errors, _g = _intake(RC9O, "weg2-33-33", switch=False)
        self.assertEqual(set(errors), {0, 1, 2})
        self.assertTrue(all("W65" in str(e) for e in errors.values()), errors)


class TestPieces(unittest.TestCase):
    def test_host_prefix_formula(self):
        f = m.host_prefix_from_group_span
        self.assertEqual(f(match_end=33088, group_span=320, page_size=1, bigram=False), 32768)
        # page 64, bigram: key len = floor64(33130 - 32768 - 1) = 320
        self.assertEqual(f(match_end=33130, group_span=320, page_size=64, bigram=True), 32768)
        for end in range(32768 + 64, 32768 + 64 + 200):
            span = (end - 32768 - 1) // 64 * 64
            self.assertEqual(f(match_end=end, group_span=span, page_size=64, bigram=True), 32768)

    def test_host_votes_its_span(self):
        with _Env():
            _ROLE.worker = False
            self.assertEqual(m.form_a_prefetch_span_vote(320), (320, -320))
            _ROLE.worker = True
            try:
                self.assertEqual(
                    m.form_a_prefetch_span_vote(6720),
                    (m.PREFETCH_SPAN_ABSTAIN, m.PREFETCH_SPAN_ABSTAIN),
                )
            finally:
                _ROLE.worker = False

    def test_scheduler_delegates_after_the_vote(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._prefetch_kvcache)
        i = src.index("locally_eligible=locally_eligible,")
        self.assertLess(i, src.index("tp_match_floor.adopt_host_prefetch_span("))


if __name__ == "__main__":
    unittest.main()
