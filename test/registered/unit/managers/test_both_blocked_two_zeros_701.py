"""#701: `freed == 0` from the BOTH-BLOCKED relief means TWO different things,
and the first version of the log reported only the alarming one.

``evict_from_tree_cache`` returns 0 in two states:

  * it ran and could not reach anything -- the pathological case, a pool held
    behind something the leaf frontier cannot peel (an in-flight chunked
    request's protected prefix);
  * it was SKIPPED because ``avail >= num_tokens`` already -- see the bare
    ``return 0`` at the tail of its standard-allocator branch. Nothing was
    wrong; there was simply nothing to do.

The live 21:44:37 specimen was the SECOND kind and the log called it the
first: "Eviction delivered NOTHING ... the pool is held by something the
frontier cannot reach". That sends the next reader hunting a phantom locked
chain, and it is the counter-vs-actuator mistake the routine exists to catch,
committed by the instrument itself.

These tests drive the method for real rather than inspecting its source, so a
future edit that collapses the two branches back into one message fails here.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.test_utils import CustomTestCase


def _stub_scheduler(rows_after=139507, chunked=512):
    from sglang.srt.managers import scheduler as scheduler_mod

    stub = SimpleNamespace(
        server_args=SimpleNamespace(chunked_prefill_size=chunked),
        tree_cache=SimpleNamespace(token_to_kv_pool_allocator=object()),
        _both_blocked_evict_at=0.0,
        BOTH_BLOCKED_EVICT_INTERVAL_S=(
            scheduler_mod.Scheduler.BOTH_BLOCKED_EVICT_INTERVAL_S
        ),
        _post_evict_rows=lambda: rows_after,
    )
    return stub


def _both_blocked_decision():
    from sglang.srt.managers.phase_policy import BOTH_BLOCKED

    return SimpleNamespace(reason=f"{BOTH_BLOCKED}: nothing can run")


def _run(stub, avail, freed, pending=0):
    """Drive the relief with a stubbed actuator + availability."""
    from sglang.srt.managers import scheduler as scheduler_mod

    with (
        mock.patch(
            "sglang.srt.mem_cache.common.evict_from_tree_cache", return_value=freed
        ),
        mock.patch(
            "sglang.srt.mem_cache.common.uniform_avail_for_evict", return_value=avail
        ),
    ):
        with mock.patch.object(scheduler_mod.logger, "warning") as warn:
            scheduler_mod.Scheduler._apply_both_blocked_relief(
                stub,
                _both_blocked_decision(),
                SimpleNamespace(now=1000.0, pending_prefill_tokens=pending),
            )
    if not warn.call_args_list:
        return ""
    call = warn.call_args_list[-1]
    fmt, args = call.args[0], call.args[1:]
    try:
        return fmt % args
    except TypeError:
        return fmt + " " + " ".join(str(a) for a in args)


class TestBothBlockedTwoZeros701(CustomTestCase):
    def test_zero_with_sufficient_avail_is_reported_as_SKIPPED(self):
        """The live 21:44:37 case. 139507 rows reachable, 512 wanted, freed 0 --
        because eviction was never needed, not because it failed."""
        msg = _run(_stub_scheduler(), avail=139507, freed=0, pending=0)
        self.assertIn("SKIPPED", msg, msg)
        self.assertNotIn("frontier cannot reach", msg, msg)
        self.assertIn(
            "not the binding resource",
            msg,
            "the benign case must say KV is not what is blocking admission",
        )

    def test_zero_with_insufficient_avail_still_escalates(self):
        """CAN-FAIL. The pathological reading must survive: a fix that always
        reported SKIPPED would pass the test above and fail this one."""
        msg = _run(_stub_scheduler(), avail=0, freed=0)
        self.assertIn("frontier cannot reach", msg, msg)
        self.assertIn("escalate", msg, msg)
        self.assertNotIn("SKIPPED", msg, msg)

    def test_nonzero_freed_reports_the_remedy_ran(self):
        msg = _run(_stub_scheduler(), avail=0, freed=512)
        self.assertIn("actually run", msg, msg)
        self.assertNotIn("SKIPPED", msg, msg)
        self.assertNotIn("frontier cannot reach", msg, msg)

    def test_the_two_zero_branches_differ_in_DIAGNOSIS_not_just_in_numbers(self):
        """The property in one assertion -- and it must not be satisfiable by
        the numbers alone.

        A first version of this test compared the raw strings and PASSED under a
        mutant that gave both zeros the same diagnosis, because the quoted
        availability differed (139507 vs 0). Comparing digit-stripped text is
        what makes it bite: the WORDS have to differ, not the counters embedded
        in them.
        """
        import re

        def diagnosis(msg):
            return re.sub(r"\d+", "N", msg)

        benign = diagnosis(_run(_stub_scheduler(), avail=139507, freed=0))
        bad = diagnosis(_run(_stub_scheduler(), avail=0, freed=0))
        self.assertNotEqual(
            benign,
            bad,
            "freed=0 must not produce one indistinguishable diagnosis; "
            "differing numbers inside an identical sentence is not a diagnosis",
        )

    def test_availability_is_quoted_so_the_claim_is_checkable(self):
        msg = _run(_stub_scheduler(), avail=139507, freed=0)
        self.assertIn("139507", msg)
        self.assertIn("512", msg)


if __name__ == "__main__":
    unittest.main()


class TestPendingDemandGuardsTheClaim701(CustomTestCase):
    """The 22:22:33 specimen: 19004 rows available, 512 wanted, 97922 pending.

    `want` is chunked_prefill_size -- an arbitrary chunk, not what the blocked
    work needs. Concluding "KV is not the binding resource" from `avail >= 512`
    is an over-claim from a partial instrument, which is precisely what this
    routine exists to stop it doing to someone else.
    """

    def test_pending_above_avail_withholds_the_not_binding_claim(self):
        msg = _run(_stub_scheduler(), avail=19004, freed=0, pending=97922)
        self.assertIn("SKIPPED", msg, msg)
        self.assertIn("97922", msg, msg)
        self.assertIn("may well be binding", msg, msg)
        self.assertNotIn("KV is not the binding resource", msg, msg)

    def test_pending_below_avail_still_makes_the_claim(self):
        """CAN-FAIL: the strong claim must survive where it IS warranted. A fix
        that always hedged would pass the test above and fail this one."""
        msg = _run(_stub_scheduler(), avail=139507, freed=0, pending=4096)
        self.assertIn("KV is not the binding resource", msg, msg)
        self.assertNotIn("may well be binding", msg, msg)
