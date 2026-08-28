"""Boot 10's next form: the mamba slot across void-park and re-admission.

THE SYMPTOM, from the boot-10 report: `req.mamba_pool_idx is None` reaching
the `cache_unfinished_req` path on a CARRIED / re-admitted request, where
`mamba_component.py` dereferences it without a guard:

    active_mamba_state_pool(self.cache).copy_from(
        translate(req.mamba_pool_idx.unsqueeze(0)),
        translate(mamba_value_donated),
    )

`None.unsqueeze` raises, so the lifecycle must guarantee the slot rather than
the caching path guarding it -- and that makes the lifecycle a thing worth
pinning.

WHAT THIS FILE ESTABLISHES, AND WHAT IT DOES NOT. It pins the half I could
measure: the void-PARK preserves the slot. It does NOT claim to have found
where the None comes from. The two candidate clearers are named below with
what is known about each, and the one that is measured is measured; the rest
is left open rather than guessed, because a plausible story here would be
indistinguishable from the real one until a boot disagrees.
"""

import types
import unittest

import torch

from sglang.srt.managers import scheduler_pp_mixin as m
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

EXECUTED = 4096
TOTAL = 8422


def _req_with_a_mamba_slot(slot=3):
    req = types.SimpleNamespace(
        rid="carried-with-mamba",
        prefix_indices=torch.arange(EXECUTED, dtype=torch.int64),
        extend_range=m.Range(EXECUTED, TOTAL) if hasattr(m, "Range") else None,
        req_pool_idx=0,
        mamba_pool_idx=torch.tensor([slot]),
        cache_protected_len=EXECUTED,
        is_retracted=False,
        finished_reason=None,
        inflight_middle_chunks=1,
    )
    if req.extend_range is None:
        from sglang.srt.utils.common import Range

        req.extend_range = Range(EXECUTED, TOTAL)
    return req


def _pool():
    """A req_to_token_pool whose mamba release is OBSERVABLE."""
    pool = types.SimpleNamespace(
        req_to_token=torch.arange(TOTAL, dtype=torch.int64).view(1, TOTAL),
        freed_mamba=[],
        freed_req=[],
    )

    def free_mamba_cache(req):
        pool.freed_mamba.append(getattr(req, "rid", None))
        req.mamba_pool_idx = None

    def free(req):
        pool.freed_req.append(getattr(req, "rid", None))

    pool.free_mamba_cache = free_mamba_cache
    pool.free = free
    return pool


class _Allocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.clone())


def _scheduler():
    return types.SimpleNamespace(
        chunked_req=None,
        waiting_queue=[],
        req_to_token_pool=_pool(),
        token_to_kv_pool_allocator=_Allocator(),
        tree_cache=None,
    )


class TheParkKeepsTheMambaSlot(unittest.TestCase):
    """MEASURED: a parked chunk comes out of the park still holding its slot.

    This is the fence. #984 made the void PARK rank 0's members instead of
    retracting them, and the whole point of a park is that the request keeps
    the state a retraction would have thrown away -- pages, prefix, and the
    mamba slot with them. If a future edit makes the park release the slot,
    the re-admitted request reaches `cache_unfinished_req` with None and the
    unguarded `.unsqueeze(0)` raises, which is boot 10's shape.
    """

    def test_the_park_does_not_release_the_mamba_slot(self):
        sched = _scheduler()
        req = _req_with_a_mamba_slot()

        parked = m._park_chunked_prefill_chunk(sched, req, pass_allocated=True)

        self.assertTrue(
            parked,
            "precondition: the park must actually have run -- a no-op park "
            "would make the assertions below vacuous",
        )
        self.assertIsNotNone(
            req.mamba_pool_idx,
            "the void-park released the mamba slot. A parked request is "
            "re-admitted from its own state, and `cache_unfinished_req` "
            "dereferences `req.mamba_pool_idx.unsqueeze(0)` with no guard "
            "(mamba_component.py) -- releasing it here hands that path a None",
        )
        self.assertEqual(
            sched.req_to_token_pool.freed_mamba,
            [],
            "no mamba release may be issued on the park path at all",
        )

    def test_the_park_gives_back_only_the_never_run_tail(self):
        """The park's own contract, kept beside the slot assertion.

        Stated here because the two must not be traded against each other: a
        park that kept the slot but discarded the executed prefix would pass
        the arm above and still cost a re-prefill.
        """
        sched = _scheduler()
        req = _req_with_a_mamba_slot()

        m._park_chunked_prefill_chunk(sched, req, pass_allocated=True)

        self.assertEqual(
            len(req.prefix_indices),
            EXECUTED,
            "the executed prefix must survive the park untouched",
        )
        self.assertEqual(
            req.extend_range.end,
            len(req.prefix_indices),
            "and the range must come out in the parked shape",
        )


class TheCachingPathDereferencesTheSlotUnguarded(unittest.TestCase):
    """Why the lifetime above is load-bearing rather than a nicety.

    Asserted against the source: as long as this dereference is unguarded,
    'the slot is never None here' is a LIFECYCLE obligation, not a local one,
    and every path that can reach re-admission owes it.
    """

    def test_the_deref_is_unguarded_so_the_lifecycle_must_guarantee_it(self):
        import inspect

        from sglang.srt.mem_cache.unified_cache_components import mamba_component

        src = inspect.getsource(mamba_component)
        self.assertIn(
            "req.mamba_pool_idx.unsqueeze(0)",
            src,
            "the dereference this file is written against has moved; "
            "re-check whether the lifecycle obligation still applies here",
        )


class WhatIsNotEstablishedHere(unittest.TestCase):
    """The open half, written down so it is not mistaken for closed.

    The park is measured NOT to clear the slot, so boot 10's None comes from
    somewhere else. Two candidates, neither confirmed by me:

      (a) the VOID_PARK_FINISHED route. That one still reaches
          `_release_voided_request` -> `release_req` -> `reset_for_retract`,
          and `reset_for_retract` clears `mamba_pool_idx` along with
          `last_node` (named in `_park_chunked_prefill_chunk`'s own docstring:
          "that method clears `last_node` and `mamba_pool_idx`"). A request
          disposed of that way must never be re-admitted into
          `cache_unfinished_req` without re-acquiring a slot.

      (b) a re-admission path that does not re-acquire. `init_next_round_input`
          re-matches the prefix and reassigns `prefix_indices` / `last_node`;
          whether anything re-acquires a mamba slot on that path is NOT
          established here.

    Which of the two produced boot 10's None decides the fix, and guessing
    between them is exactly the move that produced this window's recurring
    class -- a predicate assumed where one could have been asked. This arm
    exists to hold the question open in executable form.
    """

    def test_reset_for_retract_is_the_named_clearer(self):
        """Only the documented fact, so the docstring above cannot rot."""
        import inspect

        # CITATION CORRECTED after this arm went red: the sentence naming
        # the clearer lives in the RELEASE helper's docstring, not the
        # park's. Recorded rather than silently repointed -- an arm that
        # cites the wrong function is the same defect class this window
        # keeps producing, and it was caught only by running it.
        src = inspect.getsource(m)
        self.assertIn(
            "clears `last_node` and\n    `mamba_pool_idx`",
            src,
            "the sentence naming reset_for_retract as the clearer of "
            "last_node and mamba_pool_idx is gone from scheduler_pp_mixin; "
            "candidate (a) above must be re-derived from the tree",
        )


if __name__ == "__main__":
    unittest.main()
