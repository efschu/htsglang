"""#580 after #791b: a grammar-ready request enters the queue AFTER the
memoised prefetch drain and must not reach admission without a verdict.

THE INCIDENT (NF, 2026-10-08, image htsglang:cu130-weg2-rc12z30y9int16-27b-nf,
boots dkrnfint4h6ablxcbar1dauer10080959 and ...10081009): 55 s and 2 min after
SERVING all three D ranks raised in ``Scheduler._prefetch_done_for``::

    RuntimeError: request pdflip-1-8 reached the prefill admission loop without
    a drained HiCache prefetch verdict on a multi-rank boot ...

Both deaths were the same 13528-token chat body (D-IDS sha d389b3d4d3e44eaf,
pdflip-1-8 and after the restart pdflip-0-8). On the restart the D log shows the
request received, then two decode rounds, and only then its intake match --
it waited outside the queue for its grammar to compile (D gets the original
chat body with its tool constraint, P's leg 1 carries none, and D's grammar
cache is per process, so every restart misses again).

THE MECHANISM, black box. On the TP loop the drain no longer runs where #580
placed it ("AFTER the grammar queue has drained"): #791b pulled it forward
into ``_update_uniform_pool_budget`` and memoised it as
``_pass_prefetch_verdicts``. ``_get_new_batch_prefill_raw`` then moves the
grammar-ready requests into ``waiting_queue`` and pops the OLD memo, so the
admission loop meets a request the drain never saw. The #580 guard refuses
it -- correctly: answering with a live ``check_prefetch_progress`` would
enter a collective outside the replicated drain.

THE FIX under test: such a request is pending for this pass -- the answer the
#791b ballot already gives every rid it does not hold -- and is drained and
balloted from the next pass. No live progress check, no collective.

Hermetic: the REAL ``_drain_prefetch_progress`` computes the memo over the
pre-intake queue exactly as the budget site does, the REAL
``_get_new_batch_prefill_raw`` runs from its first line through the grammar
intake and the memo pop, and a sentinel stops it at the next call
(``_take_uniform_head_inputs``). The verdict map the admission loop would read
is the memo object itself (the raw pass pops it, it does not copy it).
"""

import types
import unittest

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)


RID_QUEUED = "pdflip-1-7"
RID_GRAMMAR = "pdflip-1-8"


class _PastTheMemo(Exception):
    """Raised by the stand-in for the call right after the memo pop."""


class _Req:
    def __init__(self, rid):
        self.rid = rid


class _TreeCache:
    """Records every rid the progress check was asked about: each such call
    with a registered prefetch is a collective on the TP group."""

    def __init__(self):
        self.asked = []
        self.ongoing_prefetch = {}

    def drain_retired_prefetch(self):
        pass

    def check_prefetch_progress(self, rid):
        self.asked.append(rid)
        return True


def _holder():
    from flliper.srt.managers.scheduler import Scheduler

    grammar_req = _Req(RID_GRAMMAR)
    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(tp_size=3, pp_size=1),
        server_args=types.SimpleNamespace(enable_flexkv=False),
        enable_hierarchical_cache=False,
        enable_hicache_storage=True,
        waiting_queue=[_Req(RID_QUEUED)],
        tree_cache=_TreeCache(),
        grammar_manager=types.SimpleNamespace(
            has_waiting_grammars=lambda: True,
            get_ready_grammar_requests=lambda: [grammar_req],
        ),
    )
    # the intake's effect on the queue (its prefetch registration is not
    # what this module is about)
    h._add_request_to_queue = lambda req, is_retracted=False: h.waiting_queue.append(req)

    def _stop():
        raise _PastTheMemo()

    h._take_uniform_head_inputs = _stop
    for name in (
        "_drain_prefetch_progress",
        "_prefetch_done_for",
        "_retry_deferred_prefetches",
        "_get_new_batch_prefill_raw",
    ):
        setattr(h, name, types.MethodType(getattr(Scheduler, name), h))
    return h, grammar_req


class GrammarIntakeAfterMemoDrain(unittest.TestCase):
    def test_nf1008_grammar_ready_request_is_pending_not_a_580_stop(self):
        """Pre-fix: the admission loop's ``_prefetch_done_for`` raised the
        #580 RuntimeError for pdflip-1-8 on every TP rank (the NF deaths).
        Post-fix: the request reads not-done this pass, through no live
        progress check, and the ballot agrees."""
        from flliper.srt.managers import prefetch_ballot

        h, grammar_req = _holder()
        # the TP loop's budget site: drain over the PRE-intake queue, memoise
        memo = h._drain_prefetch_progress()
        h._pass_prefetch_verdicts = memo
        h._uniform_prefetch_ballot = {RID_QUEUED: True}
        self.assertNotIn(RID_GRAMMAR, memo)  # the drain never saw it

        with self.assertRaises(_PastTheMemo):
            h._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None, running_batch=None
            )
        # the grammar intake did run, and the memo was consumed
        self.assertIn(grammar_req, h.waiting_queue)
        self.assertNotIn("_pass_prefetch_verdicts", h.__dict__)

        # what the admission loop reads for the late arrival
        self.assertIs(h._prefetch_done_for(grammar_req, memo), False)
        self.assertIs(
            prefetch_ballot.prefetch_done_under_ballot(
                False, RID_GRAMMAR, {RID_QUEUED: True}
            ),
            False,
        )
        # the drained request keeps its drained verdict
        self.assertIs(memo[RID_QUEUED], True)
        # and no rank entered the progress collective for the late arrival
        self.assertEqual(h.tree_cache.asked, [RID_QUEUED])


if __name__ == "__main__":
    unittest.main()
