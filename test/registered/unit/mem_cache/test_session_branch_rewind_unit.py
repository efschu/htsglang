"""The session-level half of #410: branch and rewind as a token splice.

The runtime tests (``test/registered/unit/managers/test_session_checkpoint.py``)
prove the control plane against a fake controller. This module proves the REAL
``Session`` / ``SessionController`` behaviour the runtime drives, because that
is where a branch could silently corrupt the session it came from.

Both are expressed through ``SessionParams.offset`` -- the token-level splice
point the controller already implements -- rather than a second mechanism for
rewriting history. What is asserted, with the can-fail half of each:

* a rewind moves the next turn's context back to the checkpoint prefix, and
  is consumed ONCE: the turn after it appends normally;
* a client-supplied offset still wins over the server-side one;
* a branch reads the parent's token arrays and never writes them -- the
  splice point forces the copy path, so the parent's context is byte-identical
  before and after the child's first turn. Without the splice the streaming
  append mutates in place, which is exactly the corruption this guards;
* branching or rewinding a session with nothing to continue from is refused.

    python -m pytest test/registered/unit/mem_cache/test_session_branch_rewind_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import FINISH_LENGTH
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session, SessionController
from sglang.test.test_utils import CustomTestCase

VOCAB = 1 << 20


def _recv(rid, input_ids, max_new_tokens=8, offset=None, parent_rid=None):
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", input_ids),
        mm_inputs=None,
        session_params=SimpleNamespace(
            id="s",
            rid=parent_rid,
            offset=offset,
            replace=False,
            drop_previous_output=False,
        ),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        lora_id=None,
        custom_logit_processor=None,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        priority=None,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


class _FakeTreeCache:
    def __init__(self):
        self.released = []

    def release_session(self, session_id):
        self.released.append(session_id)


class _SessionHarness(CustomTestCase):
    def setUp(self):
        self.controller = SessionController(_FakeTreeCache())
        self.session = Session(capacity_of_str_len=0, session_id="s", streaming=True)
        self.controller.sessions["s"] = self.session

    def _turn(self, session, rid, input_ids, output, offset=None, parent_rid=None):
        req = session.create_req(
            _recv(rid, input_ids, offset=offset, parent_rid=parent_rid),
            tokenizer=None,
            vocab_size=VOCAB,
        )
        req.output_ids = list(output)
        if session.streaming:
            # finish_req is the streaming bookkeeping; a non-streaming session
            # keeps its request tree and never calls it -- it only needs the
            # turn to be finished before the next one may continue from it.
            session.finish_req(req)
        else:
            req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
        return req


class TestRewind(_SessionHarness):
    def test_a_rewind_moves_the_next_turns_context_back(self):
        first = self._turn(self.session, "r0", [1, 2, 3], [4, 5])
        self.assertEqual(list(first.origin_input_ids), [1, 2, 3])

        # Checkpoint the conversation at 4 tokens (1,2,3 + the first output).
        self.controller.rewind_to("s", [1, 2, 3, 4])
        self.assertEqual(self.session.pending_rewind_offset, 4)

        second = self._turn(self.session, "r1", [90], [])
        self.assertEqual(list(second.origin_input_ids), [1, 2, 3, 4, 90])

    def test_the_rewind_offset_is_consumed_once(self):
        self._turn(self.session, "r0", [1, 2, 3], [4, 5])
        self.controller.rewind_to("s", [1, 2, 3, 4])
        self._turn(self.session, "r1", [90], [91])
        self.assertIsNone(self.session.pending_rewind_offset)
        third = self._turn(self.session, "r2", [92], [])
        # can-fail half: the turn AFTER the rewind appends, it does not splice.
        self.assertEqual(list(third.origin_input_ids), [1, 2, 3, 4, 90, 91, 92])

    def test_without_a_rewind_the_turn_simply_appends(self):
        self._turn(self.session, "r0", [1, 2, 3], [4, 5])
        second = self._turn(self.session, "r1", [90], [])
        self.assertEqual(list(second.origin_input_ids), [1, 2, 3, 4, 5, 90])

    def test_a_client_offset_wins_over_the_server_side_one(self):
        self._turn(self.session, "r0", [1, 2, 3], [4, 5])
        self.controller.rewind_to("s", [1, 2, 3, 4])
        # A non-streaming session, so the client offset is not refused.
        plain = Session(capacity_of_str_len=0, session_id="p", streaming=False)
        self.controller.sessions["p"] = plain
        self._turn(plain, "p0", [1, 2, 3], [4, 5])
        self.controller.rewind_to("p", [1, 2, 3, 4])
        spliced = self._turn(plain, "p1", [90], [], offset=2, parent_rid="p0")
        self.assertEqual(list(spliced.origin_input_ids), [1, 2, 90])

    def test_rewinding_a_session_with_no_finished_turn_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.controller.rewind_to("s", [1, 2])
        self.assertIn("no completed turn", str(ctx.exception))

    def test_rewinding_an_unknown_session_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.controller.rewind_to("nope", [1, 2])
        self.assertIn("unknown session", str(ctx.exception))


class TestBranch(_SessionHarness):
    def test_a_branch_continues_from_the_checkpoint_prefix(self):
        self._turn(self.session, "r0", [1, 2, 3], [4, 5])
        child_id = self.controller.branch_from(
            parent_session_id="s", checkpoint_tokens=[1, 2, 3, 4], new_session_id="b"
        )
        self.assertEqual(child_id, "b")
        child = self.controller.sessions["b"]
        self.assertEqual(child.pending_rewind_offset, 4)

        branched = self._turn(child, "b0", [90], [])
        self.assertEqual(list(branched.origin_input_ids), [1, 2, 3, 4, 90])

    def test_a_branch_never_mutates_the_parents_token_arrays(self):
        """The corruption this guards against: the streaming append path
        extends the parent's arrays IN PLACE. A branch must take the copy
        path, which the splice point forces."""
        self._turn(self.session, "r0", [1, 2, 3], [4, 5])
        parent_req = next(iter(self.session.req_nodes.values())).req
        before = list(parent_req.origin_input_ids)

        self.controller.branch_from(
            parent_session_id="s", checkpoint_tokens=[1, 2, 3, 4], new_session_id="b"
        )
        self._turn(self.controller.sessions["b"], "b0", [90], [91])
        self.assertEqual(list(parent_req.origin_input_ids), before)

        # ...and the parent still continues from ITS own history.
        parent_next = self._turn(self.session, "r1", [70], [])
        self.assertEqual(list(parent_next.origin_input_ids), [1, 2, 3, 4, 5, 70])

    def test_two_branches_of_one_checkpoint_are_independent(self):
        self._turn(self.session, "r0", [1, 2, 3], [4, 5])
        for name in ("b1", "b2"):
            self.controller.branch_from(
                parent_session_id="s",
                checkpoint_tokens=[1, 2, 3, 4],
                new_session_id=name,
            )
        first = self._turn(self.controller.sessions["b1"], "x0", [90], [])
        second = self._turn(self.controller.sessions["b2"], "y0", [91], [])
        self.assertEqual(list(first.origin_input_ids), [1, 2, 3, 4, 90])
        self.assertEqual(list(second.origin_input_ids), [1, 2, 3, 4, 91])

    def test_a_branch_gets_a_generated_id_when_none_is_given(self):
        self._turn(self.session, "r0", [1, 2, 3], [4, 5])
        child_id = self.controller.branch_from(
            parent_session_id="s", checkpoint_tokens=[1, 2, 3]
        )
        self.assertIn(child_id, self.controller.sessions)
        self.assertNotEqual(child_id, "s")

    def test_branching_onto_an_existing_session_id_is_refused(self):
        self._turn(self.session, "r0", [1, 2, 3], [4, 5])
        with self.assertRaises(ValueError) as ctx:
            self.controller.branch_from(
                parent_session_id="s", checkpoint_tokens=[1, 2, 3], new_session_id="s"
            )
        self.assertIn("already exists", str(ctx.exception))

    def test_branching_from_an_unknown_session_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.controller.branch_from(parent_session_id="nope", checkpoint_tokens=[1])
        self.assertIn("unknown parent session", str(ctx.exception))

    def test_branching_before_the_first_turn_finishes_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.controller.branch_from(parent_session_id="s", checkpoint_tokens=[1])
        self.assertIn("no completed turn", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
