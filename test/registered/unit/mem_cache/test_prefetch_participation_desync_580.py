"""Falsifiers for #580 -- prefetch participation must be rank-uniform.

Production crash (2026-08-05 06:23:38, TP=3 uneven DCP + hicache file
backend), verbatim::

    terminate called after throwing an instance of 'gloo::EnforceNotMet'
      what():  [enforce fail at .../gloo/transport/tcp/pair.cc:456]
      op.preamble.length <= op.nbytes. 16 vs 4. Received data size doesn't
      match expected size. Is there a distributed collective mismatch in
      your code?

The two sides of "16 vs 4", read off the tracebacks in that same log:

* TP0 died inside ``unified_radix_cache.prefetch_from_storage`` ->
  ``_all_reduce_attn_groups`` -- the participation vote, ONE int32, so gloo
  registered a 4-byte receive buffer.
* TP1/TP2 were inside ``kv_pressure_runtime._consensus_check`` -- eight
  int64, 64 bytes, which gloo's ring splits into 2*world_size = 6 segments
  of 16 bytes.

Both run on the SAME gloo group (``_ATTN_CP = _TP`` without a separate CP
group, and ``tp_cache_group = tp_cpu_group`` without DP attention), so
ProcessGroupGloo matched them as the same sequence number.

Why TP0 and only TP0 died: gloo's check is ``preamble.length <= nbytes``,
an INEQUALITY. The ranks holding the big buffer accept the short read and
sail on; only the rank holding the small buffer aborts the process. That
asymmetry is why the log shows TP1/TP2 continuing into the consensus and
dying later of "Connection closed by peer".

Root cause (the #431 family, "rank-local condition in front of a group
collective"): every predicate upstream of the participation vote is
rank-local under uneven DCP --

* ``Scheduler._prefetch_kvcache``: ``last_host_node.backuped`` is "full KV
  present in THIS rank's host pool", and uneven DCP gives the ranks host
  pools of different sizes;
* ``prefetch_from_storage``: ``prefetch_rate_limited()`` reads the per-rank
  ``prefetch_tokens_occupied`` counter.

A rank that trips either one returns BEFORE the vote, so it never enters
the collective its peers are entering. The vote was built to symmetrize the
host allocation; it cannot symmetrize participation in itself.
"""

import os
import subprocess
import sys
import types
import unittest

import torch

from sglang.srt.mem_cache.unified_radix_cache import (
    HiCacheCollectiveError,
    UnifiedRadixCache,
)
from sglang.test.ci.ci_register import register_cpu_ci

# ~25s: TestGlooReproduction spawns three child interpreters that each import
# sglang and stand up a real gloo group.
register_cpu_ci(est_time=30, suite="base-a-test-cpu")

#: Payload posted by the peers in the production crash: the kv-pressure
#: ladder consensus proposal, four (v, -v) int64 pairs.
KV_PRESSURE_CONSENSUS_ELEMS = 8

#: The exact gloo diagnostic from the production log.
PRODUCTION_ABORT_SIGNATURE = "op.preamble.length <= op.nbytes. 16 vs 4"


class _FakeHostPool:
    def __init__(self, capacity: int):
        self.capacity = capacity

    def alloc(self, n: int):
        if n > self.capacity:
            return None
        return list(range(n))

    def available_size(self) -> int:
        return self.capacity

    def free(self, _indices) -> None:
        pass


class _FakeController:
    def __init__(self, *, rate_limited: bool, capacity: int = 4096):
        self._rate_limited = rate_limited
        self.mem_pool_host = _FakeHostPool(capacity)
        self.prefetch_tokens_occupied = 0
        self.released = []

    def prefetch_rate_limited(self) -> bool:
        return self._rate_limited

    def prefetch(self, *args, **kwargs):
        return types.SimpleNamespace(host_indices=[0], id=0)

    def append_host_mem_release(self, *args, **kwargs):
        self.released.append((args, kwargs))


def _node(backuped: bool = True):
    """A host node the prefetch path can anchor on."""
    return types.SimpleNamespace(
        key=None,
        backuped=backuped,
        parent=None,
        get_last_hash_value=lambda: None,
        get_prefix_hash_values=lambda parent: None,
    )


def make_cache(
    *,
    rate_limited: bool,
    symmetric: bool,
    all_reduce,
    page_size: int = 1,
    prefetch_threshold: int = 4,
):
    """Minimal carrier exposing exactly what ``prefetch_from_storage`` touches.

    The method under test is the REAL one, bound to this carrier; only the
    collaborators around it are stubs.
    """
    cache = types.SimpleNamespace(
        enable_storage=True,
        cache_controller=_FakeController(rate_limited=rate_limited),
        is_eagle=False,
        page_size=page_size,
        prefetch_threshold=prefetch_threshold,
        ongoing_prefetch={},
        # #939: the registration retires any incumbent record under the same
        # req_id before taking the slot. Carried here as the REAL bound method
        # (like prefetch_from_storage below) rather than a stub, so this
        # stand-in keeps exercising what production runs.
        _retired_prefetch=[],
        _retired_prefetch_attempts={},
        _retired_prefetch_recompute=0,
        _components_tuple=(),
        _all_reduce_attn_groups=all_reduce,
        _hicache_prefetch_symmetric=lambda: symmetric,
        inc_host_lock_ref=lambda node: types.SimpleNamespace(
            to_dec_params=lambda: ("dec", node)
        ),
        dec_host_lock_ref=lambda node, params: None,
        evict_host=lambda n: 0,
        _build_sidecar_transfers=lambda phase, kv_xfer, comp_xfers: [],
    )
    cache.prefetch_from_storage = types.MethodType(
        UnifiedRadixCache.prefetch_from_storage, cache
    )
    cache._retire_ongoing_prefetch = types.MethodType(
        UnifiedRadixCache._retire_ongoing_prefetch, cache
    )
    # #1068 (slice 4): every exit behind the gate names itself through these
    # real methods (L1/L2 lines); carried as the REAL bound methods too.
    for name in (
        "_prefetch_line_terms",
        "_log_prefetch_refused",
        "_log_prefetch_truncated",
    ):
        setattr(cache, name, types.MethodType(getattr(UnifiedRadixCache, name), cache))
    return cache


def _run_rank(*, rate_limited: bool, symmetric: bool):
    """Run one rank's prefetch attempt; return the collectives it issued."""
    issued = []

    def _all_reduce(tensor, op, label="hicache"):
        issued.append((label, tensor.numel(), tensor.dtype))
        # MIN across the group: this rank is the pessimistic one whenever it
        # is ineligible, which is exactly what the vote is supposed to carry.
        if rate_limited and tensor.numel() >= 1:
            tensor[-1] = 0

    cache = make_cache(
        rate_limited=rate_limited, symmetric=symmetric, all_reduce=_all_reduce
    )
    cache.prefetch_from_storage(
        "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
    )
    return issued


class TestParticipationIsRankUniform(unittest.TestCase):
    """RED before the fix: the rate-limited rank issues no collective at all
    while its peers issue the participation vote."""

    def test_rate_limited_rank_still_enters_the_vote(self):
        healthy = _run_rank(rate_limited=False, symmetric=True)
        limited = _run_rank(rate_limited=True, symmetric=True)

        self.assertTrue(healthy, "the eligible rank must issue the participation vote")
        self.assertEqual(
            [(label, numel, dt) for label, numel, dt in healthy],
            [(label, numel, dt) for label, numel, dt in limited],
            "a rank-local predicate must not decide participation in a group "
            "collective: the rate-limited rank has to enter the same vote "
            "with the same payload shape and lower its own value instead",
        )

    def test_vote_payload_shape_is_identical_across_ranks(self):
        shapes = {
            tuple((n, d) for _, n, d in _run_rank(rate_limited=rl, symmetric=True))
            for rl in (False, True)
        }
        self.assertEqual(
            len(shapes),
            1,
            f"payload shapes diverge across ranks: {shapes}",
        )

    def test_default_even_tp_path_is_unchanged(self):
        # symmetric=False is stock HiCache: no vote, and a rate-limited rank
        # still returns early. That path must stay exactly as it was.
        self.assertEqual(_run_rank(rate_limited=False, symmetric=False), [])
        self.assertEqual(_run_rank(rate_limited=True, symmetric=False), [])


class TestForeignPayloadIsLoud(unittest.TestCase):
    """gloo aborts the process only when the mismatched buffers differ in
    size. Equal-sized traffic from another collective is accepted silently and
    would corrupt the vote, so the payload identifies itself."""

    def test_foreign_same_width_payload_raises_named_error(self):
        def _foreign(tensor, op, label="hicache"):
            # A same-width vector from somewhere else on this group.
            tensor.copy_(torch.tensor([7, -7, 1], dtype=tensor.dtype))

        cache = make_cache(rate_limited=False, symmetric=True, all_reduce=_foreign)
        # Asserted on the base class, so this file also COLLECTS against the
        # unfixed tree (where the named subclass does not exist yet) and the
        # can-fail proof covers every case in it.
        with self.assertRaises(HiCacheCollectiveError) as ctx:
            cache.prefetch_from_storage(
                "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
            )
        self.assertEqual(type(ctx.exception).__name__, "HiCacheCollectiveDesyncError")
        self.assertIn("prefetch_participation_vote", str(ctx.exception))

    def test_matching_tag_is_accepted(self):
        def _honest(tensor, op, label="hicache"):
            pass

        cache = make_cache(rate_limited=False, symmetric=True, all_reduce=_honest)
        cache.prefetch_from_storage(
            "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
        )
        self.assertIn("req-0", cache.ongoing_prefetch)


class TestSchedulerPrefetchGateIsRankUniform(unittest.TestCase):
    """RED before the fix: ``_prefetch_kvcache`` skips the whole call -- and
    therefore the vote inside it -- when THIS rank's host pool happens not to
    hold the node."""

    def _driver(self, *, backuped: bool, symmetric: bool):
        from sglang.srt.managers.scheduler import Scheduler

        calls = []
        root = _node()
        host_node = root if backuped else _node(backuped=False)
        tree_cache = types.SimpleNamespace(
            root_node=root,
            hicache_storage_pass_prefix_keys=False,
            prefetch_participation_is_collective=lambda: symmetric,
            prefetch_from_storage=lambda *a, **kw: calls.append((a, kw)),
        )
        req = types.SimpleNamespace(
            rid="req-0",
            last_host_node=host_node,
            prefix_indices=[],
            host_hit_length=0,
            full_untruncated_fill_ids=list(range(64)),
            init_next_round_input=lambda tree, cow_mamba=False: None,
            _compute_max_prefix_len=lambda n: n,
        )
        sched = types.SimpleNamespace(
            enable_hicache_storage=True, tree_cache=tree_cache
        )
        sched._prefetch_kvcache = types.MethodType(Scheduler._prefetch_kvcache, sched)
        sched._prefetch_kvcache(req)
        return calls

    def test_non_backuped_rank_still_reaches_the_collective(self):
        backed = self._driver(backuped=True, symmetric=True)
        unbacked = self._driver(backuped=False, symmetric=True)
        self.assertEqual(len(backed), 1)
        self.assertEqual(
            len(unbacked),
            1,
            "under group-decided participation every rank must call "
            "prefetch_from_storage, or the ranks that do are alone in the vote",
        )
        self.assertIs(unbacked[0][1].get("locally_eligible"), False)

    def test_default_path_still_skips_non_backuped_nodes(self):
        self.assertEqual(self._driver(backuped=False, symmetric=False), [])
        self.assertEqual(len(self._driver(backuped=True, symmetric=False)), 1)


# --------------------------------------------------------------------------
# End-to-end falsifier: the real method, a real gloo group, three processes.
# --------------------------------------------------------------------------

_CHILD_ENV = "SGLANG_580_GLOO_CHILD"


def _gloo_child() -> None:
    """One rank of the reproduction. Mirrors the production interleaving:
    the prefetch attempt, then the kv-pressure consensus, on one gloo group.
    """
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", rank=rank, world_size=3)
    group = dist.new_group(ranks=[0, 1, 2], backend="gloo")

    holder = types.SimpleNamespace(
        attn_cp_group=None,
        attn_tp_group=None,
        tp_group=group,
        tp_world_size=3,
        collective_timeout_s=0.0,
    )
    holder._wait_bounded = types.MethodType(UnifiedRadixCache._wait_bounded, holder)
    all_reduce = types.MethodType(UnifiedRadixCache._all_reduce_attn_groups, holder)

    # Rank 0 is eligible; ranks 1 and 2 are rate-limited. Under uneven DCP the
    # per-rank prefetch counters really do drift apart like this.
    cache = make_cache(rate_limited=(rank != 0), symmetric=True, all_reduce=all_reduce)
    cache.prefetch_from_storage(
        "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
    )

    # The next collective on this group in the production loop: the ladder
    # consensus. Eight int64 -> 16-byte ring segments against the vote's 4.
    consensus = torch.zeros(KV_PRESSURE_CONSENSUS_ELEMS, dtype=torch.int64)
    dist.all_reduce(consensus, op=dist.ReduceOp.MIN, group=group)
    print(f"rank {rank} survived", flush=True)


class TestGlooReproduction(unittest.TestCase):
    """RED before the fix: rank 0 aborts with the production signature.

    This is the byte-level provenance proof -- the same ``16 vs 4`` gloo
    diagnostic, produced by the real ``prefetch_from_storage`` against a real
    gloo group.
    """

    def test_three_rank_prefetch_then_consensus_survives(self):
        env = dict(os.environ)
        env.update(
            {
                _CHILD_ENV: "1",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29583",
                # Hermetic: this test must never touch a GPU.
                "CUDA_VISIBLE_DEVICES": "99",
            }
        )
        procs = []
        for rank in range(3):
            child_env = dict(env, RANK=str(rank))
            procs.append(
                subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__)],
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        outs = []
        for proc in procs:
            try:
                out, _ = proc.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
                out += "\n[TIMEOUT]"
            outs.append((proc.returncode, out))

        joined = "\n".join(out for _, out in outs)
        self.assertNotIn(
            PRODUCTION_ABORT_SIGNATURE,
            joined,
            "the #580 production abort reproduced: the ranks entered "
            f"different collectives.\n{joined}",
        )
        for rank, (code, out) in enumerate(outs):
            self.assertEqual(code, 0, f"rank {rank} exited {code}:\n{out}")
            self.assertIn(f"rank {rank} survived", out)


if __name__ == "__main__":
    if os.environ.get(_CHILD_ENV):
        _gloo_child()
    else:
        unittest.main()
