"""retract_decode must never hard-abort the request FCFS says to protect.

WHY THIS FILE EXISTS
--------------------
``ScheduleBatch.retract_decode`` (schedule_batch.py) retracts the
least-preferred requests first (``_get_decode_retraction_order`` sorts by
``(len(output_ids), -len(origin_input_ids))`` descending, and the loop pops
from the tail) -- so under memory pressure, the request that has decoded the
FEWEST tokens so far (the youngest / most recently admitted) is the first
one evicted back to the waiting queue. That is the correct half of FCFS: the
newest, not the oldest, pays first.

The other half is not implemented. Once every other request has been
retracted and exactly one remains, if THAT one still does not fit --
reachable under extreme concurrent pressure once the kv-session-offload
spill budget is exhausted (#236/#242 ``max_spills``: ``try_spill`` returns
False when ``self._free_regions`` is empty, and the scheduler falls back to
stock ``retract_decode``, see scheduler.py's decode-OOM branch) -- the code
does not retract that survivor. It hard-aborts it with
``FINISH_ABORT(..., status_code=HTTPStatus.INTERNAL_SERVER_ERROR)``. Because
every less-preferred (younger) request was already popped out of
``sorted_indices`` before this point, the survivor reaching this branch is,
by construction, the OLDEST / most-progressed request in the batch -- the
one every other line of this same function just finished protecting. A 500
here is the FCFS promise broken at the last possible moment: everyone junior
was sent back to the queue, and the senior request is the one that gets
killed.

Downstream this is not cosmetic: ``FINISH_ABORT`` with
``HTTPStatus.INTERNAL_SERVER_ERROR`` reaches
``TokenizerManager._handle_abort_finish_reason``
(tokenizer_manager.py), which raises
``fastapi.HTTPException(status_code=500, ...)`` for a non-streaming caller,
or yields a terminal abort chunk for a streaming one -- the oldest client in
the system receives a hard failure it did nothing to deserve, instead of the
"try again shortly" a re-queue would give it.

CPU only: builds a bare ``ScheduleBatch`` via ``__new__`` and fakes the
handful of collaborators ``retract_decode`` calls, so this exercises the
real retraction-order and abort-vs-retract decision without a GPU, a model,
or a real memory pool.
"""

import types
import unittest
from http import HTTPStatus

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(rid, num_decoded, max_new_tokens=200, num_prompt_tokens=10):
    """A minimal stand-in for Req: only the fields retract_decode touches."""
    return types.SimpleNamespace(
        rid=rid,
        output_ids=[0] * num_decoded,
        origin_input_ids=[0] * num_prompt_tokens,
        sampling_params=types.SimpleNamespace(max_new_tokens=max_new_tokens),
        to_finish=None,
        priority=None,
        solo_oom_count=0,
    )


def _make_batch(reqs, *, always_short: bool):
    """A ScheduleBatch stand-in with the collaborators retract_decode calls
    faked out, so the test exercises only the retraction-order / abort
    decision, never a real memory pool.

    ``always_short``: check_decode_mem always reports "does not fit", the
    extreme-pressure case this bug lives in -- spilling is exhausted and
    retracting every junior request still is not enough for the senior
    survivor's own next decode step.
    """
    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.reqs = list(reqs)
    batch.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)

    released = []

    def _check_decode_mem(selected_indices=None):
        return not always_short

    def _release_req(idx, remaining_req_count, server_args):
        released.append(idx)

    def _filter_batch(keep_indices):
        batch.reqs = [batch.reqs[i] for i in keep_indices]

    batch.check_decode_mem = _check_decode_mem
    batch.release_req = _release_req
    batch.filter_batch = _filter_batch
    batch._released_indices = released
    return batch


class TestRetractDecodeProtectsTheOldestUnderExtremePressure(unittest.TestCase):
    """The falsifier: old + new session, spill budget exhausted, old survives."""

    def setUp(self):
        server_args = types.SimpleNamespace(retraction_policy="length")
        # old: decoded 100 tokens already -- the senior request, most-progressed.
        self.old_req = _make_req("old-session", num_decoded=100)
        # new: just started decoding -- the junior request, least-progressed.
        self.new_req = _make_req("new-session", num_decoded=1)
        # Extreme pressure: even after retracting every other request, the
        # sole survivor's own next decode step still does not fit (spill
        # budget exhausted -- kv_session_offload.try_spill already declined
        # for lack of a free region before stock retraction ever runs).
        self.batch = _make_batch(
            [self.old_req, self.new_req], always_short=True
        )
        self.server_args = server_args

    def test_the_new_session_is_retracted_first(self):
        """Sanity check on the ALREADY-correct half: junior goes first."""
        retracted_reqs, _, _ = self.batch.retract_decode(self.server_args)
        self.assertIn(self.new_req, retracted_reqs)

    def test_the_old_session_is_not_hard_aborted(self):
        """The falsifier. RED on shipped code: the survivor (old_req, by
        construction the last one standing once every junior request has
        been popped) is hard-aborted with a 500 instead of being retracted
        like everything else in this same function."""
        _, _, reqs_to_abort = self.batch.retract_decode(self.server_args)

        self.assertNotIn(
            self.old_req,
            reqs_to_abort,
            "retract_decode hard-aborted the oldest/most-progressed "
            "request instead of protecting it -- this is the FCFS "
            "violation #273 fixes: every younger request was already "
            "retracted (sent back to the queue) by the time this one is "
            "considered, so aborting it now kills the one request this "
            "function was, up to this line, protecting.",
        )
        self.assertIsNone(
            self.old_req.to_finish,
            "the oldest request must not carry a finish reason at all -- "
            "a re-queued request keeps generating on its next scheduling "
            "turn, an aborted one is dead",
        )

    def test_no_request_gets_a_500_under_extreme_pressure(self):
        """Stronger form of the same falsifier: NOTHING in this scenario
        should reach the client as an internal-server-error abort. Extreme
        transient pressure is backpressure, not a server fault."""
        _, _, reqs_to_abort = self.batch.retract_decode(self.server_args)

        for req in reqs_to_abort:
            finish = req.to_finish
            status = getattr(finish, "status_code", None) if finish else None
            self.assertNotEqual(
                status,
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"{req.rid} was aborted with a 500 under mere pool "
                "pressure; #273 requires this be a re-queue (or, if "
                "truly unfittable, a non-500 status), never a 500",
            )

    def test_pool_state_stays_consistent_after_retraction(self):
        """release_req must be called for every request removed from the
        running batch -- retracted AND (if it ever legitimately happens)
        aborted alike -- or the KV/req-pool bookkeeping leaks a slot."""
        retracted_reqs, _, reqs_to_abort = self.batch.retract_decode(
            self.server_args
        )
        removed = len(retracted_reqs) + len(reqs_to_abort)
        self.assertEqual(
            len(self.batch._released_indices),
            removed,
            "every request removed from the running batch must go "
            "through release_req exactly once, or the token/req pool "
            "and the KV tree end up double-booked or leaked",
        )


class TestRetractDecodeStillAbortsWhenMemoryActuallyRecovers(unittest.TestCase):
    """Not every OOM is the extreme-pressure corner -- pin the ordinary path.

    This is the ordinary, already-tested-in-production shape: retracting the
    junior request is ENOUGH, so the senior one keeps running untouched. No
    abort of any kind is expected here regardless of the #273 fix -- this
    class exists so a fix cannot "solve" the falsifier above by aborting
    more often elsewhere.
    """

    def test_the_survivor_keeps_running_when_it_fits(self):
        server_args = types.SimpleNamespace(retraction_policy="length")
        old_req = _make_req("old-session", num_decoded=100)
        new_req = _make_req("new-session", num_decoded=1)
        batch = _make_batch([old_req, new_req], always_short=False)

        # check_decode_mem: False for the full batch (triggers retraction),
        # True once down to the single survivor (ordinary case, spill/tree
        # eviction was enough).
        calls = {"n": 0}

        def _check_decode_mem(selected_indices=None):
            calls["n"] += 1
            return calls["n"] > 1

        batch.check_decode_mem = _check_decode_mem

        retracted_reqs, _, reqs_to_abort = batch.retract_decode(server_args)

        self.assertIn(new_req, retracted_reqs)
        self.assertEqual(reqs_to_abort, [])
        self.assertIsNone(old_req.to_finish)
        self.assertEqual(batch.reqs, [old_req])


class TestRetractDecodeGivesUpAfterRepeatedSoloOOM(unittest.TestCase):
    """The livelock guard: re-queuing forever is only correct for TRANSIENT
    pressure. A request that keeps losing the solo-OOM race across many
    scheduler iterations is not experiencing ordinary contention -- retrying
    it forever would be a silent infinite loop with zero progress, which is
    worse than the 500 this fix removes. It must eventually fail, cleanly.
    """

    def test_bounded_retries_then_a_clean_non_500_failure(self):
        from sglang.srt.environ import envs

        server_args = types.SimpleNamespace(retraction_policy="length")
        req = _make_req("stuck-session", num_decoded=100)
        max_retries = envs.SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES.get()

        # Simulate the SAME request re-entering retract_decode alone
        # (as if repeatedly re-admitted and immediately solo-OOMing again)
        # across more iterations than the configured retry budget.
        for attempt in range(1, max_retries + 2):
            batch = _make_batch([req], always_short=True)
            retracted_reqs, _, reqs_to_abort = batch.retract_decode(server_args)

            if attempt <= max_retries:
                self.assertEqual(
                    retracted_reqs,
                    [req],
                    f"attempt {attempt} (<= budget {max_retries}) should "
                    "still be a re-queue, not a failure",
                )
                self.assertEqual(reqs_to_abort, [])
                self.assertIsNone(req.to_finish)
            else:
                self.assertEqual(
                    reqs_to_abort,
                    [req],
                    f"attempt {attempt} (> budget {max_retries}) should "
                    "finally fail the request instead of re-queuing it "
                    "forever",
                )
                self.assertIsNotNone(req.to_finish)
                self.assertEqual(
                    req.to_finish.status_code,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "a request that structurally cannot fit gets a clean "
                    "503, never the 500 this bug used to send",
                )


if __name__ == "__main__":
    unittest.main()
