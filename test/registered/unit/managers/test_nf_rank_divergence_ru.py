"""RU (nf_rank_divergence): the NF D group must never split on a rank-local
match or prefetch verdict.

Two deaths under real agent load on the NF D group (TP3, --rank-tp-ratio
1,0,0, dcp_size=1 -- TP0 on the arena host pool, TP1/TP2 on zero-width plain
pools):

* rc9k (dkrnfbar1final09260301, rid weg2-21-19): TP0 ``#915 PREFETCH REFUSED
  reason=too_short need=192``, TP1/TP2 ``#904 match-census verdict=refused
  ... MambaComponent:absent=29888`` and a registered prefetch; TP1/TP2 then
  sat in ``can_terminate_prefetch`` (3 x int32) while TP0 posted the packed
  int64 MIN of ``_update_uniform_pool_budget`` -> gloo ``248 vs 4``.
* rc9i (dkrnfbar1agent09252237, rid weg2-32-26): TP1/TP2 ``[#928 anchor]
  REFUSING resume ... host_hit=0``, TP0 ``host_hit=18112`` + #988 LOADBACK +
  tail skip-extend -> different forwards -> TP0 hung in chain-recv, watchdog.

Every test here drives REAL production code (the #580 predicate,
``prefetch_from_storage``, ``check_prefetch_progress`` /
``can_terminate_prefetch``, ``MambaComponent.finalize_match_result``) on
three simulated ranks joined by a mock gloo group that enforces what gloo
enforces: all ranks in the same collective, same byte size, or the group dies.
RED on a3b9479f29, GREEN with fix.patch.
"""

from __future__ import annotations

import math
import threading
import types
import unittest
from typing import Dict, List, Optional
from unittest import mock

import torch

from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

#: The attribute the fix plants on the tree for one plan call. Written by
#: name here so this file COLLECTS and runs on the unfixed tree too, where
#: nothing reads it (that is the red).
FLOOR_ATTR = "_tp_match_floor_group"

WORLD = 3
THRESHOLD = 256
#: packed int64 MIN of `_update_uniform_pool_budget`: 186 int64 = 1488 B,
#: which gloo's ring cuts into 2*world = 6 chunks of 248 B -- the "248".
PACKED_BUDGET_ELEMS = 186


def _gloo_chunk_bytes(t: torch.Tensor) -> int:
    per = max(1, math.ceil(t.numel() / (2 * WORLD)))
    return per * t.element_size()


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

    def all_reduce(self, rank: int, tensor: torch.Tensor, op, label: str):
        self.log[rank].append((label, tensor.numel(), str(tensor.dtype)))
        with self._lock:
            self.slots[rank] = (label, tensor.clone())
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            raise CollectiveMismatch(f"rank {rank} alone in {label}")
        entries = [self.slots[r] for r in range(WORLD)]
        chunks = {_gloo_chunk_bytes(e[1]) for e in entries}
        if len(chunks) > 1 or len({e[1].numel() for e in entries}) > 1:
            msg = (
                "op.preamble.length <= op.nbytes. %d vs %d. Received data size "
                "doesn't match expected size: %s"
                % (max(chunks), min(chunks), [(e[0], e[1].numel()) for e in entries])
            )
            with self._lock:
                self.errors.append(msg)
            self.barrier.abort()
            raise CollectiveMismatch(msg)
        stack = torch.stack([e[1] for e in entries])
        if op == torch.distributed.ReduceOp.MAX:
            red = stack.max(dim=0).values
        else:
            red = stack.min(dim=0).values
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            raise CollectiveMismatch(f"rank {rank} lost the group after {label}")
        tensor.copy_(red.to(tensor.dtype))


def run_ranks(fn):
    results: Dict[int, object] = {}
    errors: Dict[int, BaseException] = {}

    def _t(r):
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


# --------------------------------------------------------------------------
# Server-args forms
# --------------------------------------------------------------------------

NF_D = types.SimpleNamespace(rank_tp_ratio=[1, 0, 0], hicache_size=4)
EVEN_TP = types.SimpleNamespace(rank_tp_ratio=None, hicache_size=4)
EVEN_RATIO = types.SimpleNamespace(rank_tp_ratio=[1, 1, 1], hicache_size=4)


def _patched_form(server_args, dcp_uneven: bool):
    return (
        mock.patch(
            "sglang.srt.runtime_context.get_server_args", return_value=server_args
        ),
        mock.patch.object(urc, "uneven_dcp_active", return_value=dcp_uneven),
    )


class _Form:
    def __init__(self, server_args, dcp_uneven):
        self.ps = _patched_form(server_args, dcp_uneven)

    def __enter__(self):
        for p in self.ps:
            p.__enter__()

    def __exit__(self, *a):
        for p in reversed(self.ps):
            p.__exit__(*a)


def _symmetric(server_args, dcp_uneven: bool, tp_world_size: int = 3) -> bool:
    carrier = types.SimpleNamespace(
        enable_storage=True, cache_controller=object(), tp_world_size=tp_world_size
    )
    with _Form(server_args, dcp_uneven):
        return bool(UnifiedRadixCache._hicache_prefetch_symmetric(carrier))


class TestPrefetchVoteInForceOnNfForm(unittest.TestCase):
    """The #580 participation vote is the existing agreement; on NF D it was
    switched off because its predicate only knew uneven DCP."""

    def test_nf_d_form_decides_prefetch_by_group(self):
        self.assertTrue(
            _symmetric(NF_D, dcp_uneven=False),
            "--rank-tp-ratio 1,0,0 gives the TP ranks different host-pool "
            "classes (arena vs zero-width plain); prefetch participation must "
            "be a group vote there, or one rank's `too_short` splits the group",
        )

    def test_27b_form_unchanged(self):
        self.assertTrue(_symmetric(types.SimpleNamespace(rank_tp_ratio=[58, 25, 25]), True))

    def test_even_tp_stays_local(self):
        self.assertFalse(_symmetric(EVEN_TP, dcp_uneven=False))
        self.assertFalse(_symmetric(EVEN_RATIO, dcp_uneven=False))

    def test_single_rank_never_votes(self):
        self.assertFalse(_symmetric(NF_D, dcp_uneven=False, tp_world_size=1))


# --------------------------------------------------------------------------
# rc9k: prefetch registration, then the drain, then the packed budget MIN
# --------------------------------------------------------------------------


class _HostPool:
    def __init__(self, capacity: int = 1 << 20):
        self.capacity = capacity

    def alloc(self, n: int):
        return None if n > self.capacity else list(range(n))

    def available_size(self) -> int:
        return self.capacity


class _Controller:
    def __init__(self):
        self.mem_pool_host = _HostPool()
        self.prefetch_tokens_occupied = 0
        self.prefetch_capacity_limit = 1 << 20
        self.released = []

    def prefetch_rate_limited(self) -> bool:
        return False

    def prefetch(self, req_id, host_indices, prefetch_key, *a, **kw):
        return types.SimpleNamespace(
            request_id=req_id,
            host_indices=host_indices,
            hash_value=[],
            completed_tokens=0,
            start_time=0.0,
            is_terminated=lambda: False,
        )

    def append_host_mem_release(self, *a, **kw):
        self.released.append((a, kw))


def _host_node():
    return types.SimpleNamespace(
        key=None,
        backuped=True,
        parent=None,
        get_last_hash_value=lambda: None,
        get_prefix_hash_values=lambda parent: None,
    )


def _prefetch_carrier(rank: int, group: MockGlooGroup):
    cache = types.SimpleNamespace(
        enable_storage=True,
        cache_controller=_Controller(),
        tp_world_size=WORLD,
        is_eagle=False,
        page_size=1,
        prefetch_threshold=THRESHOLD,
        prefetch_stop_policy="wait_complete",
        ongoing_prefetch={},
        _retired_prefetch=[],
        _retired_prefetch_attempts={},
        _retired_prefetch_recompute=0,
        _components_tuple=(),
        inc_host_lock_ref=lambda node: types.SimpleNamespace(
            to_dec_params=lambda: ("dec", node)
        ),
        dec_host_lock_ref=lambda node, params: None,
        evict_host=lambda n: 0,
        _build_sidecar_transfers=lambda phase, kv_xfer, comp_xfers: [],
        _all_reduce_attn_groups=lambda t, op, label="hicache": group.all_reduce(
            rank, t, op, label
        ),
    )
    for name in (
        "prefetch_from_storage",
        "_retire_ongoing_prefetch",
        "_prefetch_line_terms",
        "_log_prefetch_refused",
        "_log_prefetch_truncated",
        "_weg2_extent_topup",
        "_hicache_prefetch_symmetric",
        "check_prefetch_progress",
        "can_terminate_prefetch",
    ):
        setattr(cache, name, types.MethodType(getattr(UnifiedRadixCache, name), cache))
    return cache


RID = "ru-21-19"
PROMPT = list(range(29888))


def _rc9k_rank(rank: int, group: MockGlooGroup):
    """One rank of the rc9k pass: register (rank-local span), drain, budget."""
    cache = _prefetch_carrier(rank, group)
    # TP0 matched 29,696 tokens (host anchor present), so its span is 192 --
    # below the 256 threshold. TP1/TP2 had the whole path refused by the
    # mamba validator (MambaComponent:absent=29888) and ask for all of it.
    span = PROMPT[29696:] if rank == 0 else PROMPT
    cache.prefetch_from_storage(RID, _host_node(), span, last_hash=None, prefix_keys=None)
    registered = RID in cache.ongoing_prefetch
    # `_drain_prefetch_progress` -> the real check_prefetch_progress.
    cache.check_prefetch_progress(RID)
    # the packed MIN reduce that follows the drain in _update_uniform_pool_budget
    group.all_reduce(
        rank,
        torch.zeros(PACKED_BUDGET_ELEMS, dtype=torch.int64),
        torch.distributed.ReduceOp.MIN,
        "uniform_pool_budget",
    )
    return registered


class TestRc9kPrefetchRegistrationSplit(unittest.TestCase):
    def _run(self):
        group = MockGlooGroup()
        with _Form(NF_D, dcp_uneven=False):
            results, errors = run_ranks(lambda r: _rc9k_rank(r, group))
        return group, results, errors

    def test_group_survives_the_rc9k_pass(self):
        group, results, errors = self._run()
        self.assertEqual(
            group.errors,
            [],
            "the TP ranks issued different collectives on one group (the rc9k "
            f"gloo abort): {group.errors}; per-rank sequences: {group.log}",
        )
        self.assertEqual(errors, {}, f"rank errors: {errors}")

    def test_registration_is_uniform(self):
        group, results, errors = self._run()
        self.assertEqual(len(set(results.values())), 1, f"registered per rank: {results}")

    def test_collective_sequences_identical(self):
        group, _results, _errors = self._run()
        seqs = {tuple(v) for v in group.log.values()}
        self.assertEqual(len(seqs), 1, f"per-rank collective sequences differ: {group.log}")


# --------------------------------------------------------------------------
# rc9i: the anchor verdict at admission
# --------------------------------------------------------------------------


def _mamba_data(value=None, host_value=None):
    return types.SimpleNamespace(value=value, host_value=host_value)


def _tree(rank: int):
    root = types.SimpleNamespace(name="root")
    tree = types.SimpleNamespace(
        root_node=root,
        cache_controller=object(),
        is_chunk_cache=lambda: False,
        supports_mamba=lambda: True,
    )
    # The node the 18,112-token match ends on: TP0 holds its recurrent state
    # in the host tier (arena rank), TP1/TP2 hold a tombstone (rc9i
    # `deepest=385` on TP0 vs `deepest=None` on TP1/TP2).
    data = [None, None, _mamba_data(host_value=(torch.tensor([3]) if rank == 0 else None))]
    node = types.SimpleNamespace(name="n385", component_data=data)
    return tree, node


def _component(tree):
    comp = types.SimpleNamespace(
        component_type=ComponentType.MAMBA,
        mamba_checkpoint_interval=None,
        mamba_ckpt_strict_resume=False,
        cache=tree,
        _stateless_resume_refusals=0,
        _foreign_pool_resume_refusals=0,
    )
    comp.finalize_match_result = types.MethodType(
        MambaComponent.finalize_match_result, comp
    )
    return comp


def _match(tree, node, req, cow: bool):
    comp = _component(tree)
    result = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=tree.root_node,
        last_host_node=node,
        best_match_node=node,
        host_hit_length=18112,
    )
    params = types.SimpleNamespace(cow_mamba=cow, req=req)
    return comp.finalize_match_result(
        result=result, params=params, value_chunks=[torch.zeros(1)], best_value_len=1
    )


def _group_usable(votes: Dict[int, int]) -> Optional[Dict[str, int]]:
    """What the fixed scheduler plants: the MIN-reduced usable-match arm.
    On the unfixed tree the arm does not exist -> nothing is planted."""
    try:
        from sglang.srt.managers import tp_match_floor
    except ImportError:
        return None
    canonical = [RID9I]
    reduced = torch.stack(
        [
            torch.tensor(
                tp_match_floor.build_usable_match_payload(canonical, {RID9I: v}, 32)
            )
            for v in votes.values()
        ]
    ).min(dim=0).values.tolist()
    return tp_match_floor.decode_group_usable(canonical, reduced)


RID9I = "weg2-32-26"


def _rc9i_rank(rank: int, planted: Optional[Dict[str, int]]):
    tree, node = _tree(rank)
    req = types.SimpleNamespace(rid=RID9I, mamba_pool_idx=0)
    setattr(tree, FLOOR_ATTR, planted)
    out = _match(tree, node, req, cow=True)
    return len(out.device_indices) + int(out.host_hit_length)


def _rc9i_votes() -> Dict[int, int]:
    """Each rank's vote from the head vote's own walk (cow_mamba=False, which
    never reaches #928), through the fix's usable-match rule when present."""
    votes = {}
    for rank in range(WORLD):
        tree, node = _tree(rank)
        req = types.SimpleNamespace(rid=RID9I, best_match_node=None, mamba_pool_idx=0)
        out = _match(tree, node, req, cow=False)
        n = len(out.device_indices) + int(out.host_hit_length)
        req.best_match_node = out.best_match_node
        try:
            from sglang.srt.managers import tp_match_floor

            n = tp_match_floor.local_usable_matches(tree, {RID9I: req}, {RID9I: n})[RID9I]
        except ImportError:
            pass
        votes[rank] = n
    return votes


class TestRc9iAnchorVerdictSplit(unittest.TestCase):
    def test_vote_walk_alone_does_not_see_928(self):
        # Documents WHY a new arm was needed: the head vote walk runs with
        # cow_mamba=False, so the raw match length it votes is 18112 on every
        # rank even where admission's #928 will refuse.
        for rank in range(WORLD):
            tree, node = _tree(rank)
            out = _match(tree, node, types.SimpleNamespace(rid=RID9I), cow=False)
            self.assertEqual(int(out.host_hit_length), 18112)

    def test_admission_geometry_is_uniform(self):
        planted = _group_usable(_rc9i_votes())
        geometry = {r: _rc9i_rank(r, planted) for r in range(WORLD)}
        self.assertEqual(
            len(set(geometry.values())),
            1,
            "rc9i: the ranks admitted the same rid with different prefixes "
            f"{geometry} -- TP0 resumes from its host anchor while TP1/TP2 "
            "re-prefill, so they run different forwards",
        )
        self.assertEqual(geometry[0], 0, "the only prefix every rank can materialize is 0")

    def test_no_floor_no_change(self):
        # Outside the plan call nothing is planted: TP0 keeps its resume.
        self.assertEqual(_rc9i_rank(0, None), 18112)

    def test_floor_never_raises_a_rank(self):
        # A group usable match can only LOWER a rank (delay-never-force).
        self.assertEqual(_rc9i_rank(1, {RID9I: 18112}), 0)  # #928 still refuses locally
        self.assertEqual(_rc9i_rank(0, {RID9I: 18112}), 18112)


# --------------------------------------------------------------------------
# tp_match_floor, pure (exists only with the fix)
# --------------------------------------------------------------------------


class TestFloorVerdictPure(unittest.TestCase):
    def setUp(self):
        from sglang.srt.managers import tp_match_floor

        self.m = tp_match_floor

    def test_verdicts(self):
        v = self.m.floor_verdict
        self.assertEqual(v(18112, None), "no_opinion")
        self.assertEqual(v(0, 0), "agree")
        self.assertEqual(v(18112, 18112), "agree")
        self.assertEqual(v(18112, 0), "zero")
        self.assertEqual(v(18304, 15744), "above_group")

    def test_absent_rank_abstains(self):
        canonical = ["a", "b"]
        r0 = self.m.build_usable_match_payload(canonical, {"a": 5, "b": 7}, 4)
        r1 = self.m.build_usable_match_payload(canonical, {"a": 5}, 4)
        reduced = [min(x, y) for x, y in zip(r0, r1)]
        self.assertEqual(self.m.decode_group_usable(canonical, reduced), {"a": 5})

    def test_rank_tp_plan(self):
        f = self.m.rank_tp_plan_uneven
        self.assertTrue(f([1, 0, 0]))
        self.assertTrue(f([58, 25, 25]))
        self.assertFalse(f([1, 1, 1]))
        self.assertFalse(f(None))
        self.assertFalse(f("auto"))

    def test_usable_arm_sits_before_the_tail_indexed_ballot(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._update_uniform_pool_budget)
        self.assertLess(
            src.index("_usable_at = len(vals)"),
            src.index("prefetch_ballot.build_prefetch_ballot_payload"),
            "the ballot is read from the TAIL; an arm after it moves its slice",
        )
        self.assertLess(
            src.index("_usable_at = len(vals)"),
            src.index("torch.distributed.all_reduce(t"),
        )


if __name__ == "__main__":
    unittest.main()
