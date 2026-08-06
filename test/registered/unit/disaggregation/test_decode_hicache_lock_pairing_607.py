"""Hermetic (CPU-only) tests for the #607 lock-pairing and collective-gating
defects in the PD decode HiCache path.

Three defects are pinned here, all latent in production (which runs
``disaggregation_mode=null``) but live for PD deployments:

B (lock pairing across a rematch)
    ``DecodePreallocQueue._match_prefix_and_lock`` (decode.py:545) locks
    ``result.last_device_node`` and ``match_prefix_for_req`` points
    ``req.last_node`` at that same node. The restore path then re-matches
    (``_try_hicache_queue_load_back``), and ``match_prefix_for_req``
    (schedule_policy.py:123) silently re-points ``req.last_node`` at the newly
    matched node while the lock stays on the old one. Every teardown releases
    ``req.last_node`` (release_kv_cache -> cache_finished_req), so the old node
    leaks a lock and the new node is decremented without ever being
    incremented.

C (double release of the restore lock)
    ``_commit_hicache_local_restore_to_req`` moves the restore lock onto
    ``req.last_node`` but used to leave ``decode_req.hicache_restored_node``
    set, so a subsequent ``_clean_hicache_prefetch_resources`` released the
    very node the request teardown was about to release.

E (rank-local gate in front of a collective)
    ``check_prefetch_progress`` runs ``_all_reduce_attn_groups`` collectives
    (unified_radix_cache.py:2356 and :2400) and ``prefetch_from_storage`` runs
    the #580 participation vote (:2265). Both used to sit behind
    ``l3_storage_hit_length > 0``, which derives from
    ``last_host_node.backuped`` -- "full KV in THIS rank's host pool" -- and
    diverges across ranks under uneven DCP. A subset of ranks entered the
    collectives while their peers walked past: a hang, not a wrong answer.

The collective tests run two threads as two TP ranks against a shared barrier,
so a rank that skips a collective its peer enters shows up as a broken barrier
rather than as a silent pass.
"""

import dataclasses
import threading
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

_BARRIER_TIMEOUT_S = 3.0


class FakeNode:
    """Radix node stand-in; only identity and the hash helpers are used."""

    def __init__(self, name: str, backuped: bool = True):
        self.name = name
        self.backuped = backuped
        self.parent = None

    def get_last_hash_value(self):
        return f"hash-{self.name}"

    def get_prefix_hash_values(self, _parent):
        return []

    def __repr__(self):
        return f"FakeNode({self.name})"


class FakeReqToTokenPool:
    def __init__(self):
        self.writes = []

    def write(self, where, values):
        self.writes.append((where, values))


class FakeTreeCache:
    """Tracks lock refcounts and the ordered collective sequence per rank.

    ``dec_lock_ref`` deliberately allows the count to go NEGATIVE instead of
    asserting, so a test can observe an underflow as data rather than as an
    exception raised from inside the code under test.
    """

    def __init__(
        self, rank: int = 0, symmetric: bool = False, barrier=None, ballots=None
    ):
        self.rank = rank
        self.lock_counts = {}
        self.collectives = []
        self.ongoing_prefetch = {}
        self.hicache_storage_pass_prefix_keys = False
        self.req_to_token_pool = FakeReqToTokenPool()
        self.root_node = FakeNode("root")
        self._symmetric = symmetric
        self._barrier = barrier
        # Shared across the rank threads: the participation ballots, so the
        # fake reproduces the real MIN consensus (all-or-none registration).
        self._ballots = ballots if ballots is not None else []
        self.released_aborted = []
        # Configured per test: what the fake init_load_back hands back.
        self.load_back_new_indices = torch.tensor([], dtype=torch.int64)
        self.load_back_node = None

    # -- lock bookkeeping -------------------------------------------------
    def inc_lock_ref(self, node):
        self.lock_counts[node] = self.lock_counts.get(node, 0) + 1
        return None

    def dec_lock_ref(self, node, *args, **kwargs):
        self.lock_counts[node] = self.lock_counts.get(node, 0) - 1
        return None

    def supports_mamba(self):
        return False

    # -- collectives ------------------------------------------------------
    def _enter_collective(self, label: str):
        """Record and, when a barrier is wired up, actually rendezvous.

        A rank that never calls this while its peer does leaves the peer to
        time out -- the real-world symptom of the #607/E defect.
        """
        self.collectives.append(label)
        if self._barrier is not None:
            self._barrier.wait(timeout=_BARRIER_TIMEOUT_S)

    def prefetch_participation_is_collective(self):
        return self._symmetric

    def prefetch_from_storage(
        self, rid, node, tokens, last_hash, prefix_keys, locally_eligible=True
    ):
        if self._symmetric:
            # Mirrors unified_radix_cache.py:2261-2288 -- the vote is entered
            # unconditionally, the local verdict only lowers this rank's ballot,
            # and registration is all-or-none on the MIN consensus.
            self._ballots.append(1 if locally_eligible else 0)
            self._enter_collective("prefetch_participation_vote")
            if min(self._ballots) == 0:
                return
        elif not locally_eligible:
            return
        self.ongoing_prefetch[rid] = SimpleNamespace(node=node)

    def check_prefetch_progress(self, rid):
        if rid not in self.ongoing_prefetch:
            return True
        self._enter_collective("can_terminate_prefetch")
        self._enter_collective("check_prefetch_progress")
        del self.ongoing_prefetch[rid]
        return True

    def pop_prefetch_loaded_tokens(self, rid):
        return 0

    def release_aborted_request(self, rid):
        self.released_aborted.append(rid)

    def init_load_back(self, params):
        return self.load_back_new_indices, self.load_back_node

    def query_storage_hit_length(self, *args, **kwargs):
        return 0


class FakeReq:
    def __init__(self, rid="rid-0", n_tokens=8):
        self.rid = rid
        self.origin_input_ids = list(range(n_tokens))
        self.prefix_indices = torch.tensor([], dtype=torch.int64)
        self.last_node = None
        self.last_host_node = None
        self.req_pool_idx = 0
        self.extra_key = None


class FakeDecodeReq:
    def __init__(self, req, prefix_match):
        self.req = req
        self.prefix_match = prefix_match
        self.hicache_restored_node = None
        self.hicache_restored_kv_indices = None
        self.hicache_restore_status = HiCacheRestoreResult.PENDING
        self.hicache_load_consumer_index = -1


class TransferHarness(DecodeHiCacheTransferMixin):
    def __init__(self, tree_cache):
        self.tree_cache = tree_cache


class PreallocHarness(DecodeHiCachePreallocMixin):
    def __init__(self, tree_cache, enable_decode_hicache=True):
        self.tree_cache = tree_cache
        self.scheduler = SimpleNamespace(enable_decode_hicache=enable_decode_hicache)
        self.transfer_queue = SimpleNamespace(queue=[])


_PM_FIELDS = {f.name for f in dataclasses.fields(DecodePrefixMatch)}


def _prefix_match(last_device_node, host_node=None, l3=0, l1=2, l2=2):
    kwargs = dict(
        prefix_indices=torch.arange(l1, dtype=torch.int64),
        l2_host_hit_length=l2,
        l3_storage_hit_length=l3,
        last_device_node=last_device_node,
        last_host_node=host_node if l3 > 0 else None,
    )
    # `matched_host_node` is part of the #607 fix. Only pass it when the field
    # exists, so that reverting the fix makes these tests fail on the SYMPTOM
    # (lock underflow / collective divergence) rather than on a constructor
    # TypeError.
    if "matched_host_node" in _PM_FIELDS:
        kwargs["matched_host_node"] = host_node
    return DecodePrefixMatch(**kwargs)


def _patch_rematch(monkey_target, new_node, device_indices):
    """Replace match_prefix_for_req with the side effect that matters here:
    it re-points req.last_node (schedule_policy.py:121-137)."""

    def _fake(tree_cache, req, token_ids, cow_mamba=False, include_req=False):
        req.last_node = new_node
        req.prefix_indices = device_indices
        return SimpleNamespace(
            device_indices=device_indices,
            host_hit_length=0,
            best_match_node=new_node,
            last_device_node=new_node,
        )

    return _fake


class TestDefectBRematchLockPairing(unittest.TestCase):
    """B: the admission lock must follow req.last_node across the rematch."""

    def _run_until_failed_restore(self):
        import sglang.srt.disaggregation.decode_hicache_mixin as mod

        tree = FakeTreeCache()
        old_node, new_node = FakeNode("old"), FakeNode("new")
        req = FakeReq()

        # Reproduce DecodePreallocQueue._match_prefix_and_lock (decode.py:537-545):
        # the lock and req.last_node start out on the SAME node.
        req.last_node = old_node
        tree.inc_lock_ref(old_node)

        pm = _prefix_match(old_node, l1=2, l2=2, l3=0)
        dr = FakeDecodeReq(req, pm)

        # The rematch finds a different node and covers too little, so the
        # restore fails (decode_hicache_mixin FAILED branch).
        tree.load_back_new_indices = torch.tensor([], dtype=torch.int64)
        tree.load_back_node = None
        short_cover = torch.arange(1, dtype=torch.int64)

        harness = TransferHarness(tree)
        original = mod.match_prefix_for_req
        mod.match_prefix_for_req = _patch_rematch(mod, new_node, short_cover)
        try:
            queued = harness._try_hicache_queue_load_back(dr)
        finally:
            mod.match_prefix_for_req = original

        self.assertFalse(queued)
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.FAILED)

        # Teardown: release_kv_cache -> cache_finished_req releases req.last_node.
        tree.dec_lock_ref(req.last_node)
        return tree, old_node, new_node

    def test_failed_restore_leaves_no_lock_leak_or_underflow(self):
        tree, old_node, new_node = self._run_until_failed_restore()
        # Pre-fix: old_node == 1 (leaked, still pinning its node/mamba
        # checkpoint) and new_node == -1 (released without an inc).
        self.assertEqual(
            tree.lock_counts.get(old_node, 0),
            0,
            "admission lock leaked on the pre-rematch node",
        )
        self.assertEqual(
            tree.lock_counts.get(new_node, 0),
            0,
            "lock underflow on the post-rematch node",
        )

    def test_prefix_match_tracks_the_locked_node(self):
        """The commit path decs pm.last_device_node, so it must be the locked one."""
        tree, _old, new_node = self._run_until_failed_restore()
        self.assertTrue(
            all(count == 0 for count in tree.lock_counts.values()),
            f"unbalanced locks after failed restore: {tree.lock_counts}",
        )


class TestDefectCDoubleRelease(unittest.TestCase):
    """C: commit transfers restore-lock ownership to req.last_node."""

    def test_commit_then_abort_does_not_release_twice(self):
        tree = FakeTreeCache()
        device_node, restored = FakeNode("device"), FakeNode("restored")
        req = FakeReq()
        req.last_node = device_node

        pm = _prefix_match(device_node, l1=2, l2=2, l3=0)
        dr = FakeDecodeReq(req, pm)

        # State as _try_hicache_queue_load_back leaves it on success.
        tree.inc_lock_ref(device_node)  # admission lock (decode.py:545)
        tree.inc_lock_ref(restored)  # restore lock (mixin inc_lock_ref)
        dr.hicache_restored_node = restored
        dr.hicache_restored_kv_indices = torch.arange(4, dtype=torch.int64)

        harness = TransferHarness(tree)
        harness._commit_hicache_local_restore_to_req(dr)
        self.assertIs(req.last_node, restored)

        # decode.py:1905 -- request already aborted, so the abort handler runs
        # after a successful commit, and then the normal teardown runs too.
        harness._clean_hicache_prefetch_resources(dr)
        tree.dec_lock_ref(req.last_node)

        # Pre-fix: restored == -1 (released by BOTH the abort handler and the
        # request teardown).
        self.assertEqual(
            tree.lock_counts.get(restored, 0),
            0,
            "restore lock released twice after commit + abort",
        )
        self.assertEqual(tree.lock_counts.get(device_node, 0), 0)


class TestDefectERankDivergence(unittest.TestCase):
    """E: rank-local gates must not decide entry into a collective."""

    def _run_two_ranks(self, symmetric):
        """Rank 0 has an L3 hit (backuped), rank 1 does not. Both must walk the
        same collective sequence."""
        barrier = threading.Barrier(2)
        ballots = []
        results = {}
        errors = {}

        def rank_body(rank):
            try:
                tree = FakeTreeCache(
                    rank=rank,
                    symmetric=symmetric,
                    barrier=barrier,
                    ballots=ballots,
                )
                host_node = FakeNode(f"host-{rank}", backuped=(rank == 0))
                device_node = FakeNode(f"device-{rank}")
                req = FakeReq(rid="shared-rid")
                req.last_node = device_node
                tree.inc_lock_ref(device_node)

                # THE DIVERGENCE: l3_storage_hit_length derives from
                # last_host_node.backuped, which is rank-local.
                l3 = 4 if rank == 0 else 0
                pm = _prefix_match(device_node, host_node=host_node, l3=l3)
                dr = FakeDecodeReq(req, pm)

                prealloc = PreallocHarness(tree)
                prealloc._start_hicache_prefetch(req, pm)

                import sglang.srt.disaggregation.decode_hicache_mixin as mod

                tree.load_back_new_indices = torch.arange(4, dtype=torch.int64)
                tree.load_back_node = FakeNode(f"restored-{rank}")
                cover = torch.arange(8, dtype=torch.int64)
                original = mod.match_prefix_for_req
                mod.match_prefix_for_req = _patch_rematch(mod, device_node, cover)
                try:
                    transfer = TransferHarness(tree)
                    transfer._try_hicache_queue_load_back(dr)
                finally:
                    mod.match_prefix_for_req = original

                results[rank] = list(tree.collectives)
            except Exception as exc:  # noqa: BLE001 - reported to the assertions
                errors[rank] = exc
                barrier.abort()

        threads = [threading.Thread(target=rank_body, args=(r,)) for r in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=_BARRIER_TIMEOUT_S * 4)
        for t in threads:
            self.assertFalse(t.is_alive(), "rank thread hung")
        return results, errors

    def test_ranks_enter_the_same_collective_sequence_under_symmetric(self):
        results, errors = self._run_two_ranks(symmetric=True)
        self.assertEqual(
            errors,
            {},
            f"a rank broke the barrier -- collective divergence: {errors}",
        )
        self.assertEqual(len(results), 2, f"a rank produced no result: {results}")
        self.assertEqual(
            results[0],
            results[1],
            "TP ranks walked different collective sequences: "
            f"rank0={results[0]} rank1={results[1]}",
        )
        # The vote must actually have happened on both ranks.
        self.assertIn("prefetch_participation_vote", results[0])

    def test_load_back_gate_is_not_rank_local(self):
        """check_prefetch_progress is called on every rank, hit or no hit."""
        tree = FakeTreeCache(symmetric=False)
        device_node = FakeNode("device")
        req = FakeReq()
        req.last_node = device_node
        tree.inc_lock_ref(device_node)
        # No L3 hit on this rank at all -- the pre-fix gate would skip the call.
        pm = _prefix_match(device_node, host_node=FakeNode("h"), l3=0)
        dr = FakeDecodeReq(req, pm)

        calls = []
        real_check = tree.check_prefetch_progress

        def _spy(rid):
            calls.append(rid)
            return real_check(rid)

        tree.check_prefetch_progress = _spy

        import sglang.srt.disaggregation.decode_hicache_mixin as mod

        tree.load_back_new_indices = torch.arange(4, dtype=torch.int64)
        tree.load_back_node = FakeNode("restored")
        original = mod.match_prefix_for_req
        mod.match_prefix_for_req = _patch_rematch(
            mod, device_node, torch.arange(8, dtype=torch.int64)
        )
        try:
            TransferHarness(tree)._try_hicache_queue_load_back(dr)
        finally:
            mod.match_prefix_for_req = original

        self.assertEqual(
            calls,
            ["rid-0"],
            "check_prefetch_progress was gated out on a rank with no L3 hit",
        )


if __name__ == "__main__":
    unittest.main()
