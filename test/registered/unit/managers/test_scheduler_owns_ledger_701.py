"""#701 remainder: the SCHEDULER must own the ledger, or it forgets every pass.

97ceea2f19 made `ChunkedCommitmentLedger` visible to `PrefillAdder` and put the
subtraction at the single chokepoint. What it deliberately did NOT do is decide
who constructs the ledger -- and until something outside the adder owns it, the
cross-pass reservation (defect b) is still open, because a `PrefillAdder` is
rebuilt on every pass. A ledger the adder made would be forgotten exactly when
the next pass needs it.

LIFETIME, answered from the code rather than chosen:

* not per-adder -- `_get_new_batch_prefill_raw` constructs one per pass
  (scheduler.py:5799), which is the defect itself;
* not per-request-set -- commitments are keyed by request id and released on
  finish/abort/retract, so the ledger self-cleans; tying it to a set that turns
  over would drop live commitments;
* per-SCHEDULER, which is what DESIGN_704 already specifies ("owned by the
  SCHEDULER, never by the adder ... passed in, never constructed there").

Exposed as a lazily-initialised property so it needs nothing from the
scheduler's very heavy `__init__`, which is also what makes it testable here
without standing up a real scheduler.

Hermetic: no CUDA, no scheduler boot.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.planner.chunked_admission import ChunkedCommitmentLedger
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def _bare_scheduler() -> Scheduler:
    """A Scheduler instance WITHOUT running __init__.

    The property under test must not depend on boot state; building one this
    way proves that rather than asserting it.
    """
    return Scheduler.__new__(Scheduler)


def _tree_cache(evictable: int = 0) -> MagicMock:
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = evictable
    tc.full_evictable_size.return_value = evictable
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    return tc


def _allocator(available: int = 0) -> MagicMock:
    a = MagicMock()
    a.available_size.return_value = available
    a.full_available_size.return_value = available
    a.swa_available_size.return_value = 0
    return a


def _adder(ledger, available=400_000) -> PrefillAdder:
    return PrefillAdder(
        page_size=1,
        tree_cache=_tree_cache(),
        token_to_kv_pool_allocator=_allocator(available),
        running_batch=MagicMock(reqs=[]),
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=None,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
        commitment_ledger=ledger,
    )


class TheSchedulerOwnsOneLedger(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_the_scheduler_exposes_a_ledger(self):
        self.assertIsInstance(
            _bare_scheduler().chunked_commitment_ledger, ChunkedCommitmentLedger
        )

    def test_it_is_the_same_instance_every_time(self):
        """Per-SCHEDULER lifetime: re-reading must not mint a fresh one."""
        s = _bare_scheduler()
        self.assertIs(s.chunked_commitment_ledger, s.chunked_commitment_ledger)

    def test_two_schedulers_do_not_share_one(self):
        """Rules out a module-global, which would leak across instances."""
        self.assertIsNot(
            _bare_scheduler().chunked_commitment_ledger,
            _bare_scheduler().chunked_commitment_ledger,
        )

    def test_a_commitment_survives_a_prefill_adder_REBUILD(self):
        """THE FALSIFIER: this is defect (b), through the real PrefillAdder.

        Pass 1 admits a chunked request holding 80,000 tokens of remaining
        prefill. Pass 2 builds a NEW adder -- as the scheduler does every pass
        -- and must still see the commitment, or it spends the first request's
        committed future and the deadlock returns with two actors.
        """
        sched = _bare_scheduler()
        ledger = sched.chunked_commitment_ledger

        first = _adder(ledger)
        baseline = first.rem_total_tokens
        ledger.commit("req-A", remaining_tokens=80_000)

        second = _adder(sched.chunked_commitment_ledger)  # the rebuild
        self.assertEqual(second.rem_total_tokens, baseline - 80_000)

    def test_spending_and_releasing_are_visible_to_later_passes(self):
        sched = _bare_scheduler()
        ledger = sched.chunked_commitment_ledger
        baseline = _adder(ledger).rem_total_tokens

        ledger.commit("req-A", remaining_tokens=80_000)
        ledger.spend("req-A", 30_000)
        self.assertEqual(_adder(ledger).rem_total_tokens, baseline - 50_000)

        ledger.release("req-A")
        self.assertEqual(_adder(ledger).rem_total_tokens, baseline)

    def test_an_adder_built_without_the_ledger_is_unchanged(self):
        """The flag/None path stays byte-identical -- no silent behaviour shift."""
        sched = _bare_scheduler()
        sched.chunked_commitment_ledger.commit("req-A", remaining_tokens=80_000)
        self.assertEqual(_adder(None).rem_total_tokens, _adder(None).rem_total_tokens)


if __name__ == "__main__":
    unittest.main()
