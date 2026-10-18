# SPDX-License-Identifier: Apache-2.0
"""DQH: a refused D head is re-evaluated on a state change, not every round.

NF int18, 08.10., D log boot_weg2_dkrnfint4h6ablxcbar1dauer10081149_e69f28a7f6_
1008_115011.D.log TP0, 12:13:14-12:17:25Z: 1184 decode rounds, every one with
a full admission evaluation of the refused head weg2-28-97 (1.02 ``WEG2 X-GATE``,
4.30 ``#1427 ARENA-DROP``, 1.00 ``EVICT-FRONTIER-CENSUS`` per round); host gap per
round (``t:`` delta minus ``gpu-ms``) 119.9 ms against 5-9 ms at 12:11 without a
waiting head; in 12:12:54-12:13:14 rounds WITH the evaluation 103.6 ms, WITHOUT
-3.2 ms. Queue head, waiting set and running set were unchanged for minutes.

Driven through the REAL ``Scheduler.get_next_batch_to_run`` elif chain is out of
reach of a unit test; the gate and its fingerprint are the collaborator's and are
pinned here, the wiring by a source check of the one call site.

RED on the base (no module, every round evaluates). GREEN: one evaluation, then
held passes until a replicated input changes or REEVAL_EVERY_PASSES passed.
"""

import inspect
import types
import unittest


def _q(rid):
    return types.SimpleNamespace(rid=rid)


HEAD = "weg2-28-97"


def _fp(H, *, waiting=(HEAD, "weg2-28-98", "weg2-28-100"), running=("weg2-18-77", "weg2-28-101"),
        verdicts=None, continuation=False, hol=None):
    return H.fingerprint(
        hol_state={"rid": HEAD, "passes": 73, "logged": 1} if hol is None else hol,
        waiting=[_q(r) for r in waiting], running=[_q(r) for r in running],
        prefetch_verdicts=verdicts, continuation=continuation)


class DHeadHoldTest(unittest.TestCase):
    def _hold(self, H, cut=True):
        return H.DHeadHold(group_d=True, token_cut=lambda: cut)

    def test_12_13_one_evaluation_then_held_until_reeval(self):
        from sglang.srt.weg2 import d_head_hold as H

        hold = self._hold(H)
        fp = _fp(H)
        self.assertFalse(hold.should_hold(fp), "nothing armed: the pass runs")
        hold.note_pass(fp=fp, built_nothing=True)
        evaluated = 0
        for _ in range(4 * (H.REEVAL_EVERY_PASSES + 1)):
            if not hold.should_hold(_fp(H)):
                evaluated += 1
                hold.note_pass(fp=_fp(H), built_nothing=True)
        # 1245 evaluations in 4m16s became one per REEVAL_EVERY_PASSES+1 rounds
        self.assertEqual(evaluated, 4)

    def test_every_replicated_change_evaluates_at_once(self):
        from sglang.srt.weg2 import d_head_hold as H

        changes = {
            "a running decode finished": _fp(H, running=("weg2-18-77",)),
            "a new arrival": _fp(H, waiting=(HEAD, "weg2-28-98", "weg2-28-100", "weg2-28-111")),
            "a prefetch landed": _fp(H, verdicts={"weg2-28-101": True}),
        }
        for why, changed in changes.items():
            with self.subTest(why):
                hold = self._hold(H)
                hold.note_pass(fp=_fp(H), built_nothing=True)
                self.assertTrue(hold.should_hold(_fp(H)))
                self.assertFalse(hold.should_hold(changed), why)

    def test_never_holds_an_empty_queue_a_continuation_or_a_pass_that_built(self):
        from sglang.srt.weg2 import d_head_hold as H

        hold = self._hold(H)
        self.assertIsNone(_fp(H, waiting=()), "nothing waits: nothing to hold")
        self.assertIsNone(_fp(H, continuation=True), "a chunked continuation must proceed")
        hold.note_pass(fp=_fp(H), built_nothing=False)
        self.assertFalse(hold.should_hold(_fp(H)), "the pass built a batch: nothing armed")

    def test_s42_a_pass_that_built_nothing_holds_without_a_refused_head(self):
        """rc11 14:13: no NO_TOKEN in the pass (the queue was declined before
        add_one_req) and ``_weg2_hol`` still named the head of 14:09, running
        since -- the queue is held on the same rule; the head is only a term."""
        from sglang.srt.weg2 import d_head_hold as H

        for hol in ({"rid": None}, {"rid": "weg2-18-77"}):  # none / admitted and running
            with self.subTest(hol=hol):
                hold = self._hold(H)
                fp = _fp(H, hol=hol)
                self.assertIsNotNone(fp)
                self.assertIsNone(fp[0], "a head that does not wait is no term")
                hold.note_pass(fp=fp, built_nothing=True)
                self.assertTrue(hold.should_hold(_fp(H, hol=hol)))
                self.assertFalse(hold.should_hold(_fp(H, hol={"rid": HEAD})),
                                 "a NO_TOKEN head appearing is a change")

    def test_off_outside_the_form_a_token_cut_and_group_d(self):
        from sglang.srt.weg2 import d_head_hold as H

        for hold in (self._hold(H, cut=False), H.DHeadHold(group_d=False, token_cut=lambda: True)):
            hold.note_pass(fp=_fp(H), built_nothing=True)
            self.assertFalse(hold.should_hold(_fp(H)))

    def test_the_gate_sits_in_the_elif_chain_and_arms_after_the_pass(self):
        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.get_next_batch_to_run)
        # S42: the chain consumes the budget site's DQH decision of the iteration
        # (weg2/head_vote_hold.py) and asks DQH itself only without one
        i_gate = src.index("elif self._weg2_dqh_holds(running_batch):")
        i_pass = src.index("prefill_plan = self.get_new_batch_prefill(running_batch)")
        i_note = src.index("self.weg2_d_head_hold.note_pass(")
        self.assertLess(i_gate, i_pass)
        self.assertLess(i_pass, i_note)
        self.assertIn("self.weg2_d_head_hold.should_hold(", inspect.getsource(Scheduler._weg2_dqh_holds))


if __name__ == "__main__":
    unittest.main()
