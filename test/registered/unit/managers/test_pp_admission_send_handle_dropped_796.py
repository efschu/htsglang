"""#796: the admission decision was never reaching the wire at all.

THE SIX-BOOT SPECIMEN (2026-08-20, evidence-665-f1/PROGRESS_788_787.log).
Six consecutive instrumented boots of the gapped TP=1/PP=3 shape reached
health and then froze on the first request, zero GPU utilisation on every
card, zero ADMISSION-WEDGE markers. py-spy, identically on the last four:

    PP0  blocked in `_pp_commit_pending_req_work`   -- the chain flush
    PP1  blocked in `_pp_recv_admission_decision`   -- from `_event_loop_pp_
    PP2  blocked in `_pp_recv_admission_decision`      body`'s decision recv

Three successive commits re-ordered that decision send (#791 moved it before
the chain flush, the wraparound receive was made opportunistic, #795 moved
the send before `_pp_launch_batch`). Each fixed a real defect. None changed
the signature, which is the clue this file follows: THE ORDERING OF A
MESSAGE THAT DOES NOT EXIST CANNOT MATTER.

THE DEFECT. `_pp_send_admission_decision` called
`_pp_send_dict_to_next_stage(..., async_send=True)` and DISCARDED the
returned `P2PWork` list. It was the only async channel in
`_event_loop_pp_body` that did so -- `send_req_work`, `send_proxy_work` and
`send_output_work` are all held on the scheduler and committed later. The
decision dict carries no tensors, so it travels entirely as metadata
(`GroupCoordinator._send_tensor_dict_metadata`), whose backing `header` /
`object_tensor` buffers are owned by nothing except the `P2PWork` objects
that were being thrown away. A gloo isend whose work handle and buffers go
out of scope is aborted on destruction rather than delivered, and the
downstream rank's blocking receive then waits for ever on a message that was
never on the wire.

WHY THE EXISTING HARNESS COULD NOT SEE IT -- the methodological point worth
keeping. `test_pp_admission_chain_flush_deadlock_795.py`'s `_RingWire`
deliberately keeps every isend handle and its buffers alive in
`self._inflight` (its own comment: "Kept alive in `self._inflight` rather
than let go out of scope immediately, since nothing here calls `.wait()` on
them any more"). That is a lifetime guarantee PRODUCTION DID NOT HAVE, and
it silently supplied the very property whose absence was the bug. A test
transport may not be more careful with a handle than the code under test is.

WHAT THIS FILE PINS, AND THE HONEST DIVISION OF LABOUR BETWEEN THE ARMS.
  `test_dropped_handle_reproduces_the_six_boot_signature` -- the MECHANISM
    evidence, and the can-fail proof. It drives the in-tree #795 ring
    harness over a transport that drops the handles, and asserts the exact
    py-spy specimen comes back: PP0 in the chain flush, ONE PASS AHEAD of
    PP1 and PP2, both of which sit in this channel's receive. Measured
    5/5 reproductions while writing this file, at 20 and at 500 passes.
  The remaining tests are DETERMINISTIC UNIT assertions on the shipped fix
    -- retention, commit, the last-rank gate, stand-in tolerance -- and are
    deliberately not timing-dependent.

WHY THE GREEN ARM IS A UNIT ASSERTION AND NOT A SECOND RING RUN, STATED
PLAINLY BECAUSE IT COST HALF A SESSION TO ESTABLISH. A ring harness whose
transport RETURNS real handles (so the code under test owns their lifetime)
does not reproduce the wedge on this box even with the handles dropped: with
the extra frames the buffers live a little longer, and gloo pushes the bytes
before the abort. The reproduction is only reliable when the handles are
never returned at all. That makes an integration green arm non-discriminating
-- it would pass with or without the fix -- and a test that cannot fail
proves nothing. So the ring run is used for the RED direction only, where it
is reliable, and the fix itself is pinned by assertions that cannot be
flaky: the handles ARE retained, the commit DOES wait on them and clear, and
the last rank posts NOTHING.
"""

import types
import unittest

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2
LAST = WORLD - 1

#: The dropped-handle arm never completes by construction, so there is no
#: timing window to lose -- this only has to outlast process spawn and gloo
#: init by enough to be unambiguous.
RED_JOIN_TIMEOUT_S = 25.0

#: Measured to wedge on pass 1-2; 20 keeps the suite fast.
RED_PASSES = 20


def _dropping_send_tensor_dict(
    self, tensor_dict, dst=None, all_gather_group=None, async_send=False
):
    """#795's `_RingWire.send_tensor_dict`, minus the `_inflight` retention.

    Everything else is byte-for-byte that method. The handles and their
    backing tensors are function-local and die on return -- which is what
    the pre-#796 `_pp_send_admission_decision` did to the `P2PWork` list
    production's transport handed it.
    """
    import pickle

    import torch
    import torch.distributed as dist

    if dst is None:
        dst = (self.rank_in_group + 1) % self.world_size
    buf = pickle.dumps(tensor_dict)
    size = torch.tensor([len(buf)], dtype=torch.long)
    payload = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
    if async_send:
        dist.isend(size, dst=dst)
        dist.isend(payload, dst=dst)
    else:
        dist.send(size, dst=dst)
        dist.send(payload, dst=dst)
    return []


def _dropping_worker(rank, init_file, out_dir, variant, n_passes):
    """#795's worker, run over the handle-dropping transport.

    THE PATCH MUST BE APPLIED HERE, IN THE CHILD, and that is not a detail.
    `_run` spawns with the "spawn" start method, so each child re-imports
    the module that defines its target and rebuilds every class from source.
    A patch applied in the parent's process reaches the children only when
    it happens at module scope of the parent's `__main__` (which spawn
    re-executes) -- so the same patch written inside a test METHOD silently
    does nothing under pytest, and the ring then runs with #795's own
    retaining `_RingWire` and passes, proving nothing. Measured: that is
    exactly what an earlier draft of this arm did.
    """
    import test_pp_admission_chain_flush_deadlock_795 as ring

    ring._RingWire.send_tensor_dict = _dropping_send_tensor_dict
    return ring._worker(rank, init_file, out_dir, variant, n_passes)


def _run_dropping(n_passes, join_timeout):
    """#795's `_run`, targeting `_dropping_worker`.

    Deliberately a copy of that function's body rather than a parameter
    added to it: this file must not change the behaviour of the harness it
    borrows, and #795's `_run` is pinned by its own four tests.
    """
    import json
    import os
    import tempfile
    import time

    import torch.multiprocessing as mp

    import test_pp_admission_chain_flush_deadlock_795 as ring

    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [
            ctx.Process(
                target=_dropping_worker,
                args=(r, init_file, tmp, "fixed", n_passes),
            )
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        deadline = time.time() + join_timeout
        for p in procs:
            p.join(timeout=max(0.1, deadline - time.time()))
        stuck_ranks = [r for r, p in enumerate(procs) if p.is_alive()]
        stall_report = {r: ring._read_progress(tmp, r) for r in stuck_ranks}
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)

        out = {"stuck_ranks": stuck_ranks, "stall_report": stall_report}
        for r in range(WORLD):
            path = os.path.join(tmp, f"result_r{r}.json")
            if os.path.exists(path):
                with open(path) as f:
                    out[f"result_{r}"] = json.load(f)
            else:
                out[f"result_{r}"] = None
        return out


class _FakeWork:
    def __init__(self):
        self.waited = 0

    def wait(self):
        self.waited += 1


class _FakeP2PWork:
    """`P2PWork`'s only contract, as `_pp_commit_comm_work` uses it."""

    def __init__(self):
        self.work = _FakeWork()


def _decision_holder(is_last_rank: bool, works):
    """Minimal holder for the shipped send/commit pair."""
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    sent = []

    def _fake_send_dict(tensor_dict, async_send=True, msg_type="default", stamp=None):
        sent.append((tensor_dict, async_send, msg_type))
        return list(works)

    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_size=WORLD),
        pp_group=types.SimpleNamespace(
            is_last_rank=is_last_rank, is_first_rank=not is_last_rank
        ),
        _pp_admission_send_work=[],
    )
    h._pp_send_dict_to_next_stage = _fake_send_dict
    for name in (
        "_pp_send_admission_decision",
        "_pp_commit_admission_send_work",
        "_pp_commit_comm_work",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h, sent


class PPAdmissionSendHandleDropped796(unittest.TestCase):
    def test_dropped_handle_reproduces_the_six_boot_signature(self):
        """CAN-FAIL evidence, and the mechanism proof.

        Drives the in-tree #795 ring over a transport that drops the send
        handles, and requires the six-boot py-spy signature back. If this
        ever stops wedging, the claim that a dropped handle is what killed
        those boots has lost its evidence and this file needs re-deriving.
        """
        out = _run_dropping(n_passes=RED_PASSES, join_timeout=RED_JOIN_TIMEOUT_S)

        self.assertEqual(
            out["stuck_ranks"],
            [PP0, PP1, PP2],
            f"dropping the send handle was expected to wedge every rank: {out}",
        )
        stalls = out["stall_report"]
        self.assertIn(
            "chain_flush",
            stalls[PP0],
            f"PP0 should be stuck in the chain flush, as py-spy found it on "
            f"four boots: {stalls}",
        )
        for downstream in (PP1, PP2):
            self.assertIn(
                "decision_recv",
                stalls[downstream],
                f"PP{downstream} should be stuck in the decision receive, as "
                f"py-spy found both downstreams on four boots: {stalls}",
            )
        # The specimen's other half: PP0 is an ITERATION AHEAD of the two
        # ranks waiting on it -- it is flushing a later pass's chain send
        # while they still wait for an earlier pass's decision. That
        # asymmetry is exactly what made this look like an ordering problem
        # for three commits, and it is what a dropped message produces.
        pp0_pass = int(stalls[PP0].split("_", 1)[0][len("pass") :])
        pp1_pass = int(stalls[PP1].split("_", 1)[0][len("pass") :])
        self.assertGreater(
            pp0_pass,
            pp1_pass,
            f"PP0 should be ahead of the ranks blocked on it: {stalls}",
        )

    def test_send_retains_the_work_handle(self):
        """The fix itself: the returned handles must survive the call."""
        from sglang.srt.managers.pp_admission_congruence import PPAdmissionDecision

        works = [_FakeP2PWork(), _FakeP2PWork()]
        h, sent = _decision_holder(is_last_rank=False, works=works)
        h._pp_send_admission_decision(PPAdmissionDecision(mb_id=0, entries=()))
        self.assertEqual(len(sent), 1, "the decision was not sent")
        self.assertTrue(sent[0][1], "the decision must still be sent async")
        self.assertEqual(
            h._pp_admission_send_work,
            works,
            "the send handles were discarded -- the #796 defect",
        )

    def test_commit_waits_on_every_handle_and_clears(self):
        """The other half: a retained handle that is never reaped would just
        trade a lost message for an ever-growing list of live isends."""
        from sglang.srt.managers.pp_admission_congruence import PPAdmissionDecision

        works = [_FakeP2PWork(), _FakeP2PWork()]
        h, _sent = _decision_holder(is_last_rank=False, works=works)
        h._pp_send_admission_decision(PPAdmissionDecision(mb_id=0, entries=()))
        h._pp_commit_admission_send_work()
        for w in works:
            self.assertEqual(w.work.waited, 1, "a retained handle was not reaped")
        self.assertEqual(
            h._pp_admission_send_work, [], "the commit did not clear the list"
        )

    def test_last_rank_posts_nothing(self):
        """The last rank must not emit the ring wraparound: PP0 is never
        required to receive it (it only PEEKS an inbox filled by an output
        receive that is skipped whenever the slot is empty), and one
        unmatched message per pass is the bounded-recv corpse this tree
        already refuses for the proxy under a gapped wire."""
        from sglang.srt.managers.pp_admission_congruence import PPAdmissionDecision

        h, sent = _decision_holder(is_last_rank=True, works=[_FakeP2PWork()])
        h._pp_send_admission_decision(PPAdmissionDecision(mb_id=0, entries=()))
        self.assertEqual(
            sent, [], "the last rank put a message on a wire nobody receives"
        )
        self.assertEqual(h._pp_admission_send_work, [])

    def test_commit_is_a_no_op_without_loop_state(self):
        """#787's stand-in convention: a holder that never ran
        `init_pp_loop_state` must not raise here."""
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        h = types.SimpleNamespace()
        h._pp_commit_admission_send_work = types.MethodType(
            SchedulerPPMixin._pp_commit_admission_send_work, h
        )
        h._pp_commit_admission_send_work()


if __name__ == "__main__":
    unittest.main()
