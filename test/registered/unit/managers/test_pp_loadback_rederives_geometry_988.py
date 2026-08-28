"""#988: the host load-back re-derives its geometry AT THE MUTATION.

THE INVARIANT IS #965's, NOT A NEW ONE. `extend_range.start ==
len(prefix_indices)` is asserted at the batch boundary
(`schedule_batch.py`, `prepare_for_extend`). The two halves are CO-DERIVED:
the prefix says what is already cached, the extend range says what this pass
will compute, and they meet at one index. Move one without the other and the
next reader sees a request that claims to compute tokens it has already
cached.

THE ONE MOVER THAT DID NOT RE-DERIVE. `PrefillAdder.add_one_req`'s host
load-back grows `req.prefix_indices` in place:

    req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
    prefix_len = len(req.prefix_indices)
    req.cache_protected_len = prefix_len

and, before #988, went on to the budget checks WITHOUT touching
`extend_range`. Every early return below that line therefore left the
PREVIOUS visit's extend range behind a moved prefix. Boot 8 of
window-flip-0828 died on exactly that at 25 s: a `waiting_queue` member that
was also resident in `can_run_list` got visited a second time, grew its
prefix at this line, bailed out at a budget return, and `prepare_for_extend`
read the two halves apart.

WHY THE FIX IS AT THE MUTATION AND THIS FILE TESTS IT THERE. Patching each
early return is how #965 was paid for twice already. Re-deriving immediately
after the prefix moves makes every current AND FUTURE early return inherit a
consistent parked shape for free -- the same `Range(prefix, prefix)` the void
park writes -- while the success paths below overwrite it with the real range
exactly as before. So the arms here drive a bail-out that happens AFTER the
load-back and assert the invariant on the way out, which is the property that
generalises; an arm pinned to one specific return would go stale the moment a
new one is added, which is the failure mode the fix is shaped against.

RED-FIRST, and this one is behavioural rather than symbolic: the arms name no
symbol #988 introduces. `set_extend_range` predates it, and the assertion is
on the co-derived invariant itself. Measured against 4ea93b8009 (pre-#988)
and 8380555e73 -- see the register entry [test-agent flip-0828] of
2026-08-28.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: The first visit cached this much and set its extend range from it.
FIRST_VISIT_PREFIX = 0
#: What the host load-back hands back on the second visit.
LOADED_BACK = 4096
TOTAL = 8422


class _FakeNode:
    def __init__(self):
        self.lock_ref = 0


def _tree_cache(loaded_back=LOADED_BACK):
    """A tree cache that returns `loaded_back` freshly loaded prefix indices."""
    cache = MagicMock()
    cache.init_load_back.return_value = (
        torch.arange(loaded_back, dtype=torch.int64),
        _FakeNode(),
    )
    cache.inc_lock_ref.return_value = IncLockRefResult(0)
    cache.dec_lock_ref.return_value = DecLockRefResult(0)
    cache.full_evictable_size.return_value = 0
    cache.swa_evictable_size.return_value = 0
    cache.evictable_size.return_value = 0
    cache.protected_size.return_value = 0
    cache.full_protected_size.return_value = 0
    cache.swa_protected_size.return_value = 0
    return cache


def _allocator(free=10**6):
    alloc = MagicMock()
    alloc.available_size.return_value = free
    alloc.full_available_size.return_value = free
    alloc.swa_available_size.return_value = free
    return alloc


def _req_awaiting_load_back():
    """A request whose prefix is about to MOVE under an existing extend range.

    The shape a second visit finds: the first visit left a consistent pair
    (prefix 0, extend range starting at 0), and the host load-back is still
    pending, so this pass will grow the prefix.
    """
    req = Req.__new__(Req)
    req.rid = "boot8-loadback"
    fill = list(range(TOTAL))
    req.origin_input_ids = fill
    req.output_ids = []
    req.full_untruncated_fill_ids = fill
    req.prefix_indices = torch.arange(FIRST_VISIT_PREFIX, dtype=torch.int64)
    # The FIRST visit's geometry: co-derived and valid as it stands.
    req.extend_range = Range(FIRST_VISIT_PREFIX, TOTAL)
    req.cache_protected_len = FIRST_VISIT_PREFIX
    req.req_pool_idx = 0
    req.last_node = _FakeNode()
    req.best_match_node = _FakeNode()
    req.host_hit_length = LOADED_BACK
    req.swa_host_hit_length = 0
    req.mamba_host_hit_length = 0
    req.swa_uuid_for_lock = None
    req.is_retracted = False
    req.retracted_stain = False
    req.finished_reason = None
    req.born_spilled = False
    req.born_spilled_deep = False
    req.sampling_params = SimpleNamespace(max_new_tokens=8, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.priority = 0
    req.inflight_middle_chunks = 0
    return req


class TheLoadBackLeavesAConsistentPair(unittest.TestCase):
    """#965's invariant must hold on the way OUT, whichever exit is taken."""

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _adder(self, **kwargs):
        defaults = dict(
            page_size=1,
            tree_cache=_tree_cache(),
            token_to_kv_pool_allocator=_allocator(),
            running_batch=SimpleNamespace(reqs=[]),
            new_token_ratio=1.0,
            # THE BAIL-OUT HAS TO BE ONE THAT COMES AFTER THE LOAD-BACK, and
            # picking it took one wrong turn worth recording. `add_one_req`
            # carries TWO near-identical `rem_chunk_tokens is None and
            # input_tokens >= rem_input_tokens` returns: one BEFORE the
            # `_lock_node` block and one after. Configuring for the input
            # budget hits the FIRST, so the prefix never moves and this file
            # measures nothing -- caught by the precondition assert below,
            # which is exactly what it is there for. The two cannot be
            # separated by budget either: the post-load-back `input_tokens`
            # is SMALLER than the pre-load-back one (the prefix grew), so any
            # budget that trips the second has already tripped the first.
            #
            # The truncation path is reachable only after the load-back: with
            # a chunk budget of 0, `rem_chunk_tokens is None` is False, so the
            # early branch is skipped entirely, and the `else` below computes
            # `trunc_len = 0` and returns OTHER -- with the prefix already
            # moved.
            rem_input_tokens=10000,
            rem_chunk_tokens=0,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )
        defaults.update(kwargs)
        adder = PrefillAdder(**defaults)
        adder.can_run_list.append(SimpleNamespace(rid="an-already-admitted-req"))
        return adder

    def test_a_bail_out_after_the_load_back_leaves_the_halves_together(self):
        """BOOT 8, at 25 s: the prefix moved and the extend range did not.

        The request is not admitted -- that is the point. What must survive
        the refusal is a request whose two halves still agree, because the
        next thing to touch it is `prepare_for_extend`, which asserts exactly
        that.
        """
        adder = self._adder()
        req = _req_awaiting_load_back()

        adder.add_one_req(req, truncation_align_size=None)

        prefix_len = len(req.prefix_indices)
        self.assertEqual(
            prefix_len,
            LOADED_BACK,
            "precondition: the host load-back must actually have moved the "
            "prefix -- if it did not, this arm is not reproducing boot 8 and "
            "its green proves nothing",
        )
        self.assertEqual(
            req.extend_range.start,
            prefix_len,
            "#965's co-derived invariant is broken on the way out of a "
            "budget bail-out: the prefix advanced to %d and extend_range "
            "still starts at %d, so prepare_for_extend will read the two "
            "halves apart -- boot 8's 25s assert"
            % (prefix_len, req.extend_range.start),
        )

    def test_the_parked_shape_is_what_it_re_derives_to(self):
        """Not merely 'consistent' -- the SAME parked shape the void writes.

        An empty range at the prefix means 'nothing is planned yet', which is
        what a refused request should say. A non-empty range would announce
        work no pass is going to do, and the next round's stash would cache a
        chunk that never ran.
        """
        adder = self._adder()
        req = _req_awaiting_load_back()

        adder.add_one_req(req, truncation_align_size=None)

        self.assertEqual(
            (req.extend_range.start, req.extend_range.end),
            (len(req.prefix_indices), len(req.prefix_indices)),
            "a refused load-back must leave the parked shape Range(p, p)",
        )

    def test_the_cached_prefix_is_not_discarded_by_the_re_derivation(self):
        """Kein-Doppel-Prefill: re-deriving geometry must not drop tokens."""
        adder = self._adder()
        req = _req_awaiting_load_back()

        adder.add_one_req(req, truncation_align_size=None)

        self.assertEqual(
            len(req.prefix_indices),
            LOADED_BACK,
            "the loaded-back prefix must survive -- it is cached work, and "
            "shrinking it here would recompute what the host already holds",
        )
        self.assertEqual(
            req.cache_protected_len,
            LOADED_BACK,
            "the protection must track the prefix it protects",
        )


class TheArmCanFail(unittest.TestCase):
    """The pre-#988 behaviour, reconstructed, must drive the arm red.

    Without this, a green above could mean 'the invariant holds' or 'the
    load-back never ran'. The mutant restores exactly the old shape -- move
    the prefix, leave the extend range alone -- and the assertion must catch
    it.
    """

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_a_mover_that_does_not_re_derive_is_caught(self):
        req = _req_awaiting_load_back()
        before = req.extend_range.start

        # The pre-#988 mutation, verbatim: grow the prefix, touch nothing else.
        req.prefix_indices = torch.cat(
            [req.prefix_indices, torch.arange(LOADED_BACK, dtype=torch.int64)]
        )
        req.cache_protected_len = len(req.prefix_indices)

        self.assertEqual(req.extend_range.start, before)
        self.assertNotEqual(
            req.extend_range.start,
            len(req.prefix_indices),
            "the stale shape this file exists to prevent is not even "
            "constructible, so the arms above are not watching for it",
        )


if __name__ == "__main__":
    unittest.main()
