"""#580 residual: the participation vote never reached ``HiMambaRadixCache``.

WHAT #580 ALREADY CLOSED. The 2026-08-05 06:23 abort
(``op.preamble.length <= op.nbytes. 16 vs 4``) was root-caused to
``check_prefetch_progress`` being entered on a rank-dependent set of requests.
Two repairs shipped: ``Scheduler._drain_prefetch_progress`` moved the ENTRY off
rank-local admission state (8a61a2ec4d), and a ``prefetch_participation_vote``
made prefetch REGISTRATION rank-uniform so that the
``req_id not in self.ongoing_prefetch`` early return is symmetric.

WHAT IT DID NOT CLOSE, WHICH IS THIS FILE. The vote exists in
``UnifiedRadixCache`` and was ported to ``HiRadixCache`` -- whose own comment
says "the #580 mechanism ported from UnifiedRadixCache". ``HiMambaRadixCache``
sits on a different base (``MambaRadixCache``, not ``HiRadixCache``) and was
missed. It has NO vote at all, and its ``prefetch_from_storage`` has four
rank-local early returns that decide whether ``ongoing_prefetch[req_id]`` is
ever set:

  1. ``prefetch_length < self.prefetch_threshold`` or ``prefetch_rate_limited``
  2. host KV alloc fails, then the RETRY sizes itself from
     ``mem_pool_host.available_size()`` -- this rank's own host pool
  3. the retry still returns None
  4. ``mamba_prefetch_alloc`` returns None (host mamba slot exhausted)

2, 3 and 4 are host-memory verdicts, and the host pools on this rig differ per
rank BY CONSTRUCTION -- 359652 / 287722 / 273336 slots, as
``uniform_host_avail_for_backup`` documents while fixing the same class of
defect one tier down. So a rank whose host pool is roomier registers the
prefetch, its peer does not, and at the next ``check_prefetch_progress`` the
registering rank walks alone into the ``can_terminate_prefetch`` MAX reduce
(2 x int32) and the ``min_completed_tokens`` MIN reduce (0-dim int32) while the
peer returns True at the guard without issuing anything. Mismatched-width ops
then meet on one gloo group: the crash class this ticket is named for.

THE ASSERTION THAT MATTERS is not "a vote exists" but "every rank issues the
same collectives". A test that only checked for the presence of a vote would
pass against a vote posted on the wrong group, or posted after an early return.
"""

import types
import unittest

from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: The exact gloo diagnostic from the 2026-08-05 06:23 production log, kept
#: here so the file states which failure it exists to prevent.
PRODUCTION_ABORT_SIGNATURE = "op.preamble.length <= op.nbytes. 16 vs 4"


class _FakeHostPool:
    """A host KV pool with a per-rank capacity -- the divergence source."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.freed = []

    def alloc(self, n: int):
        if n > self.capacity:
            return None
        return list(range(n))

    def available_size(self) -> int:
        return self.capacity

    def free(self, indices) -> None:
        self.freed.append(indices)


class _FakeController:
    def __init__(self, *, capacity: int, rate_limited: bool = False):
        self.mem_pool_host = _FakeHostPool(capacity)
        self.prefetch_tokens_occupied = 0
        self._rate_limited = rate_limited
        self.released = []

    def prefetch_rate_limited(self) -> bool:
        return self._rate_limited

    def prefetch(self, *args, **kwargs):
        return types.SimpleNamespace(host_indices=[0], id=0, pool_transfers=[])

    def append_host_mem_release(self, *args, **kwargs):
        self.released.append((args, kwargs))


def _node():
    return types.SimpleNamespace(
        key=None,
        parent=None,
        protect_host_mamba=lambda: None,
        get_last_hash_value=lambda: None,
        get_prefix_hash_values=lambda parent: None,
    )


def _make_cache(
    *,
    host_capacity: int,
    mamba_ok: bool,
    issued: list,
    tp_world_size=2,
    peer_ok: bool = True,
    symmetric: bool = True,
):
    """The REAL ``prefetch_from_storage`` bound to a minimal carrier.

    Only the collaborators are stubs; the method under test is production code.
    """
    mamba_host_pool = _FakeHostPool(1024)
    cache = types.SimpleNamespace(
        enable_storage=True,
        page_size=1,
        prefetch_threshold=4,
        ongoing_prefetch={},
        cache_controller=_FakeController(capacity=host_capacity),
        mamba_pool_host=mamba_host_pool,
        mamba_host_lru_list=types.SimpleNamespace(
            in_list=lambda node: False,
            remove_node=lambda node: None,
        ),
        tp_world_size=tp_world_size,
        tp_group=object(),
        _hicache_prefetch_symmetric=lambda: symmetric and tp_world_size > 1,
        evict_host=lambda n: 0,
        _protect_host_node=lambda node, protect_mamba=True: None,
        _release_host_node=lambda node, release_mamba=True: None,
        mamba_prefetch_alloc=lambda tokens, last_hash: (
            [
                types.SimpleNamespace(
                    name=PoolName.MAMBA, host_indices=[7], keys=[], hit_policy=None
                )
            ]
            if mamba_ok
            else None
        ),
    )

    def _alloc_with_evict(pool, length, evict_fn):
        return pool.alloc(length)

    cache._alloc_with_evict = _alloc_with_evict

    def _all_reduce(tensor, op, group=None, label="mamba-hicache"):
        """Record every collective this rank issues, with its exact width.

        The PRODUCTION code fills in this rank's own verdict; the stub only
        applies the peer's MIN on top. Overriding the local slot here from the
        test's own view of eligibility would test the stub's opinion instead of
        the method's.
        """
        issued.append((label, int(tensor.numel()), str(tensor.dtype)))
        tensor[-1] = min(int(tensor[-1].item()), 1 if peer_ok else 0)

    cache._all_reduce_prefetch_vote = _all_reduce
    # The release path is REAL, not stubbed: a negative vote must give back
    # both pools, and a leak there would be the fix's own bug.
    cache._release_prefetch_attempt = types.MethodType(
        HiMambaRadixCache._release_prefetch_attempt, cache
    )
    cache.prefetch_from_storage = types.MethodType(
        HiMambaRadixCache.prefetch_from_storage, cache
    )
    return cache


def _run_rank(*, host_capacity: int, mamba_ok: bool = True, peer_ok: bool = True):
    """One rank's prefetch attempt. Returns (collectives issued, registered?)."""
    issued: list = []
    cache = _make_cache(
        host_capacity=host_capacity,
        mamba_ok=mamba_ok,
        issued=issued,
        peer_ok=peer_ok,
    )
    cache.prefetch_from_storage(
        "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
    )
    return issued, ("req-0" in cache.ongoing_prefetch)


class TheMambaCacheVotesOnParticipationTest(unittest.TestCase):
    """The core of the residual: registration must be all-ranks or no-ranks."""

    #: A capacity BELOW ``prefetch_threshold``. Anything between the threshold
    #: and the full request is not a refusal at all: the retry re-sizes the
    #: prefetch from ``available_size()`` and registers a TRUNCATED one, which
    #: the ``min_completed_tokens`` MIN reduce is designed to absorb. Only a
    #: pool under the threshold produces the rank-local RETURN this is about,
    #: and getting that wrong makes the test pass against the live defect.
    STARVED = 2

    def test_a_rank_short_on_host_memory_still_enters_the_vote(self):
        """THE 06:23 SHAPE. Rank B cannot fund the host allocation at all.

        It returns at a rank-local guard having issued NOTHING, while rank A
        registers and later walks into two collectives alone. Asserted as "the
        starved rank issued a collective", not as "both issued the same
        list" -- with no vote anywhere both lists are empty and equal, so the
        list comparison would pass against the very defect it names.
        """
        issued_short, _ = _run_rank(host_capacity=self.STARVED)
        self.assertGreaterEqual(
            len(issued_short),
            1,
            "a rank that cannot fund the prefetch issued no collective at "
            "all, so its peers enter check_prefetch_progress without it -- "
            f"the {PRODUCTION_ABORT_SIGNATURE} class",
        )

    def test_a_rank_short_on_host_MAMBA_slots_still_enters_the_vote(self):
        """The mamba-specific fourth early return, which has no sibling in
        UnifiedRadixCache and is therefore the one a port would most easily
        miss."""
        issued_no_mamba, _ = _run_rank(host_capacity=4096, mamba_ok=False)
        self.assertGreaterEqual(
            len(issued_no_mamba),
            1,
            "a rank that could not allocate a host mamba slot skipped the "
            "vote its peers entered",
        )

    def test_registration_is_all_ranks_or_no_ranks(self):
        """The property the downstream collectives actually depend on."""
        _, registered_roomy = _run_rank(host_capacity=4096, peer_ok=False)
        _, registered_short = _run_rank(host_capacity=self.STARVED)
        self.assertFalse(
            registered_short,
            "the starved rank registered a prefetch it cannot fund",
        )
        self.assertFalse(
            registered_roomy,
            "the roomy rank registered although a peer could not: the vote "
            "must turn one rank's failure into a group-wide refusal, or "
            "check_prefetch_progress is entered by a subset of the group",
        )

    def test_a_group_that_all_agrees_still_registers(self):
        """The vote must not degrade into a blanket refusal: when every rank
        can fund the prefetch, every rank registers and HiCache still works."""
        issued, registered = _run_rank(host_capacity=4096)
        self.assertTrue(
            registered,
            "unanimous consensus failed to register the prefetch, so the "
            "vote has turned HiCache prefetch off rather than symmetrised it",
        )
        self.assertGreaterEqual(
            len(issued), 1, "no vote was issued at all on the healthy path"
        )


class TheVotePayloadIsUniformTest(unittest.TestCase):
    """A vote of a rank-dependent WIDTH would be the same defect again."""

    def test_the_vote_payload_has_the_same_width_on_every_rank(self):
        widths = set()
        for capacity, mamba_ok in ((4096, True), (8, True), (4096, False)):
            issued, _ = _run_rank(host_capacity=capacity, mamba_ok=mamba_ok)
            for _label, numel, dtype in issued:
                widths.add((numel, dtype))
        self.assertLessEqual(
            len(widths),
            1,
            f"the vote is posted with more than one payload shape: {widths}",
        )


class TheSingleRankPathIsUntouchedTest(unittest.TestCase):
    """Off a group there is nothing to diverge from, and the vote must not
    cost a collective."""

    def test_tp_world_size_one_issues_no_vote(self):
        issued: list = []
        cache = _make_cache(
            host_capacity=4096, mamba_ok=True, issued=issued, tp_world_size=1
        )
        cache.prefetch_from_storage(
            "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
        )
        self.assertEqual(issued, [], "a single-rank boot paid for a group collective")
        self.assertIn(
            "req-0",
            cache.ongoing_prefetch,
            "the single-rank path must still register the prefetch",
        )


class ARefusedAttemptGivesBothPoolsBackTest(unittest.TestCase):
    """A vote that leaks is worse than no vote.

    The rank outvoted here is the one that SUCCEEDED locally, so it is holding
    a host KV range and a host mamba slot that nothing downstream will ever
    reclaim: no async op was created, so no completion path runs. The mamba
    slot is the scarcer of the two and has no sibling in UnifiedRadixCache,
    which is exactly why a port could drop it.
    """

    def test_a_negative_vote_frees_host_kv_and_the_mamba_slot(self):
        issued: list = []
        cache = _make_cache(
            host_capacity=4096, mamba_ok=True, issued=issued, peer_ok=False
        )
        cache.prefetch_from_storage(
            "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
        )
        self.assertNotIn("req-0", cache.ongoing_prefetch)
        self.assertTrue(
            cache.cache_controller.mem_pool_host.freed,
            "host KV range was not returned after the group refused",
        )
        self.assertTrue(
            cache.mamba_pool_host.freed,
            "the host MAMBA slot was not returned after the group refused",
        )

    def test_the_occupancy_counter_is_not_charged_for_a_refused_attempt(self):
        issued: list = []
        cache = _make_cache(
            host_capacity=4096, mamba_ok=True, issued=issued, peer_ok=False
        )
        cache.prefetch_from_storage(
            "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
        )
        self.assertEqual(
            cache.cache_controller.prefetch_tokens_occupied,
            0,
            "a refused prefetch still charged prefetch_tokens_occupied, which "
            "feeds prefetch_rate_limited() and would desync the NEXT vote",
        )


class TheEvenTpPathIsUnchangedTest(unittest.TestCase):
    """Stock even-TP HiCache must keep its per-rank behaviour byte-identical,
    including the truncation-retry the symmetric path deliberately skips."""

    def test_no_vote_is_issued_when_the_group_does_not_decide(self):
        issued: list = []
        cache = _make_cache(
            host_capacity=4096, mamba_ok=True, issued=issued, symmetric=False
        )
        cache.prefetch_from_storage(
            "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
        )
        self.assertEqual(issued, [])
        self.assertIn("req-0", cache.ongoing_prefetch)

    def test_the_truncation_retry_still_runs_off_the_symmetric_path(self):
        """capacity 8 < the full 64: the non-symmetric path re-sizes and
        registers a shortened prefetch rather than refusing."""
        issued: list = []
        cache = _make_cache(
            host_capacity=8, mamba_ok=True, issued=issued, symmetric=False
        )
        cache.prefetch_from_storage(
            "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
        )
        self.assertIn(
            "req-0",
            cache.ongoing_prefetch,
            "the even-TP truncation retry stopped registering; that path was "
            "supposed to be untouched",
        )


class AForeignPayloadIsLoudTest(unittest.TestCase):
    """The (tag, -tag) head exists to catch same-width traffic from another
    site. If it cannot fail, it is decoration."""

    def test_a_foreign_payload_raises_a_named_error(self):
        from sglang.srt.mem_cache.hicache_collective import (
            HiCacheCollectiveDesyncError,
        )

        issued: list = []
        cache = _make_cache(host_capacity=4096, mamba_ok=True, issued=issued)

        def _foreign(tensor, op, group=None, label="x"):
            # Another site's same-width payload: the tag head does not survive.
            tensor[0] = 1
            tensor[1] = 1

        cache._all_reduce_prefetch_vote = _foreign
        with self.assertRaises(HiCacheCollectiveDesyncError):
            cache.prefetch_from_storage(
                "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
            )

    def test_the_matching_tag_is_accepted(self):
        issued: list = []
        cache = _make_cache(host_capacity=4096, mamba_ok=True, issued=issued)
        cache.prefetch_from_storage(
            "req-0", _node(), list(range(64)), last_hash=None, prefix_keys=None
        )
        self.assertIn("req-0", cache.ongoing_prefetch)


class TheSchedulerGateIsReleasedTest(unittest.TestCase):
    """Without this predicate the whole fix is unreachable.

    ``Scheduler._prefetch_kvcache`` probes for
    ``prefetch_participation_is_collective`` and, when it is absent or False,
    keeps its own rank-local ``locally_eligible`` early return -- so an
    ineligible rank never calls ``prefetch_from_storage`` and never reaches the
    vote. The vote would then symmetrize a decision its caller already made
    asymmetrically.
    """

    def test_the_cache_advertises_collective_participation(self):
        self.assertTrue(
            hasattr(HiMambaRadixCache, "prefetch_participation_is_collective"),
            "the scheduler will keep its rank-local prefetch gate, leaving the "
            "vote unreachable for exactly the ranks it exists to include",
        )

    def test_prefetch_from_storage_accepts_the_locally_eligible_kwarg(self):
        import inspect

        sig = inspect.signature(HiMambaRadixCache.prefetch_from_storage)
        self.assertIn(
            "locally_eligible",
            sig.parameters,
            "the scheduler passes locally_eligible= when the group decides; "
            "without the parameter that call raises TypeError",
        )

    def test_an_ineligible_rank_still_votes(self):
        """The caller's verdict must lower the ballot, not skip the vote."""
        issued: list = []
        cache = _make_cache(host_capacity=4096, mamba_ok=True, issued=issued)
        cache.prefetch_from_storage(
            "req-0",
            _node(),
            list(range(64)),
            last_hash=None,
            prefix_keys=None,
            locally_eligible=False,
        )
        self.assertGreaterEqual(
            len(issued),
            1,
            "a rank the scheduler judged ineligible skipped the vote",
        )
        self.assertNotIn("req-0", cache.ongoing_prefetch)


class TheGuardDownstreamStillKeysOnRegistrationTest(unittest.TestCase):
    """Why registration symmetry is the property worth enforcing.

    If ``check_prefetch_progress`` stops keying its early return on
    ``ongoing_prefetch``, this whole file is guarding the wrong invariant and
    should be told so rather than staying green.
    """

    def test_check_prefetch_progress_returns_early_on_an_unregistered_req(self):
        import inspect

        src = inspect.getsource(HiMambaRadixCache.check_prefetch_progress)
        self.assertIn("req_id not in self.ongoing_prefetch", src)

    def test_the_two_protected_collectives_are_still_there(self):
        """The MAX reduce in can_terminate_prefetch and the MIN reduce over
        completed tokens are what the vote protects. If they move, the vote's
        justification moves with them."""
        import inspect

        term = inspect.getsource(HiMambaRadixCache.can_terminate_prefetch)
        prog = inspect.getsource(HiMambaRadixCache.check_prefetch_progress)
        self.assertIn("all_reduce", term)
        self.assertIn("all_reduce", prog)


if __name__ == "__main__":
    unittest.main()
