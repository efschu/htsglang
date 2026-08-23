"""#801: a DEAD PP peer does not wedge a mandatory admission receive -- the
transport already reports it. The wedge shape is a peer that is still ALIVE.

WHY THIS FILE EXISTS AT ALL. #801 was opened to give
`_pp_recv_admission_decision` (scheduler_pp_mixin.py) a dead-peer check,
because it is the one mandatory blocking dict receive in that file with no
readiness gate in front of it, and because the #816 specimen showed PP1 dying
first and the two survivors then sitting in ADMISSION-WEDGE for 119.7 s until
the launcher tore the tree down. The proposed fix was a pre-receive
`os.kill(pid, 0)` probe against an identity the upstream publishes.

THAT FIX WOULD HAVE CLOSED NOTHING, and this file is the measurement that
says so, kept executable so nobody has to take it on trust. The three cases
below pin one table:

    peer state                gloo blocking recv        os.kill(pid, 0)
    ------------------------  ------------------------  ----------------
    exited / SIGKILLed        RAISES, ~1 s              would say dead
    alive but never sends     BLOCKS for ever           says ALIVE
    SIGSTOPped                BLOCKS for ever           says ALIVE

The states a pid probe can detect are exactly the states the transport
already reports, and the states that actually hang are exactly the states a
pid probe cannot see. The intersection of "detectable by pid" and "actually
wedges" is EMPTY, so the probe's coverage is empty. Both directions are
measured here, not just the convenient one, because a file that only showed
the dead peer raising would be equally consistent with "the probe is
redundant" and with "the probe is needed but the harness is unfaithful".

WHY THE ARGUMENT NEEDS CASE 3 TOO. "gloo raises" is only decisive if the
admission receive can ONLY ever block on gloo. `GroupCoordinator.
recv_tensor_dict` has two legs: a metadata phase on the gloo `cpu_group`
(parallel_state.py, `_recv_tensor_dict_metadata` -> `irecv(..., group=self.
cpu_group).wait()`), and a payload phase on the `device_group` (NCCL or
barlink) entered once per entry that is a real `TensorMetadata`. A NCCL peer
death is NOT reported the way a closed TCP pair is -- that asymmetry is the
whole reason `barlink_liveness.py` exists. So the leg matters, and case 3
pins the fact that decides it: the admission decision's wire dict is
primitives only, so the payload leg is never entered for this message kind.
`test_pp_admission_prefix_indices_tensor_796.py` is the sibling that pins the
mechanism keeping it that way (`prefix_indices` is a real tensor on the
sender and crosses as an `int` length).

WHAT THIS FILE DOES NOT CLAIM. It does not say the #816 wedge is explained.
It says the dead-peer explanation is refuted: had a survivor been blocked in
this receive on an exited PP1, it would have raised inside ~1 s with a
traceback, not sat silent for 119.7 s -- and nothing in `_event_loop_pp_body`
catches around that receive, so the traceback would have surfaced. That the
survivors were instead blocked at :4556 on a dead peer was the one premise
COORD-strand16e-801-ring.md marked explicitly unproven ("aus der Aktenlage
geschlossen ... nicht aus einem py-spy-Stack"). It does not survive. The
remaining live shape is a peer that is alive and not progressing, which needs
a statement about PROGRESS rather than about existence.

WHY REAL PROCESSES OVER REAL GLOO. The same reason the sibling files in this
directory give: a mock cannot distinguish "this call would have blocked" from
"this mock has no wire to block on", and the entire question here is which of
those two a real transport does. The wire and holder are the `_RingWire` /
`_make_holder` pattern from test_pp_admission_wraparound_never_blocks.py,
narrowed to two ranks because one directed pair is all these cases need.
"""

import json
import os
import pickle
import signal
import tempfile
import time
import types
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40)

WORLD = 2
SENDER, SURVIVOR = 0, 1

#: How long the survivor is given to notice a peer that is GONE. The measured
#: value is ~1.0 s (the peer's own pre-death sleep dominates it), so this is
#: an order of magnitude of headroom for a loaded box -- the assertion is
#: "raises rather than wedges", never a latency bound.
DEATH_NOTICE_BOUND_S = 20.0

#: How long the survivor must STILL be blocked for in the live-peer arms.
#: Deliberately short: these arms prove a wedge exists, and a wedge that is
#: real at 6 s was real at 1 s. Nothing is proven by waiting longer, and a
#: long bound would just make the file expensive.
WEDGE_OBSERVATION_S = 6.0

#: The peer sends, then waits this long before dying, so the survivor is
#: already inside its second (blocking) receive when the death happens. That
#: is the ordering the #816 specimen implies -- the survivors ran on past
#: PP1's last log line before wedging.
PEER_LIFETIME_AFTER_SEND_S = 1.0


class _RingWire:
    """Real point-to-point tensor-dict transport over gloo, ring-default.

    Same shape as test_pp_admission_wiring_791.py's, mirroring
    ``GroupCoordinator``'s own ``dst=None`` -> next rank / ``src=None`` ->
    previous rank resolution so the receive under test resolves its source
    exactly as production does.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD
        # #796 gates the decision send on `is_last_rank` (the last rank posts
        # nothing, because no peer is required to take it). A holder without
        # these two dies of an AttributeError inside the send -- which closes
        # the socket and makes the DEAD-PEER cases below pass for entirely the
        # wrong reason. Measured, not predicted: that is exactly what the
        # first run of this file did.
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == WORLD - 1

    def send_tensor_dict(
        self, tensor_dict, dst=None, all_gather_group=None, async_send=False
    ):
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        buf = pickle.dumps(tensor_dict)
        size = torch.tensor([len(buf)], dtype=torch.long)
        dist.send(size, dst=dst)
        dist.send(torch.frombuffer(bytearray(buf), dtype=torch.uint8), dst=dst)
        return []

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        size = torch.zeros(1, dtype=torch.long)
        dist.recv(size, src=src)
        buf = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(buf, src=src)
        return pickle.loads(bytes(buf.numpy()))


def _make_holder(rank: int, wire: _RingWire):
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=wire,
        ps=types.SimpleNamespace(pp_size=WORLD, pp_rank=rank),
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        waiting_queue=[],
        tree_cache=None,
    )
    h._pp_boundary_stats = lambda: None
    h._pp_flip_bump_sent = lambda chan: None
    h._pp_flip_bump_attempted = lambda chan: None
    h._pp_flip_bump_consumed = lambda chan: None
    for name in (
        "_pp_send_admission_decision",
        "_pp_recv_admission_decision",
        "_pp_send_dict_to_next_stage",
        "_pp_recv_typed_dict",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _sender_worker(init_file, out_dir, mode):
    from sglang.srt.managers.pp_admission_congruence import PPAdmissionDecision

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=SENDER, world_size=WORLD
    )
    h = _make_holder(SENDER, _RingWire(SENDER))
    # One healthy exchange, so the directed pair is fully established and the
    # survivor is provably past init before anything goes wrong. A wedge on an
    # unestablished pair would be a different (and uninteresting) claim.
    h._pp_send_admission_decision(PPAdmissionDecision(mb_id=0, entries=()))
    with open(os.path.join(out_dir, "sender.pid"), "w") as f:
        f.write(str(os.getpid()))
        f.flush()
        os.fsync(f.fileno())
    time.sleep(PEER_LIFETIME_AFTER_SEND_S)
    if mode == "hard_exit":
        # No teardown on purpose: this models a rank that DIES, not one that
        # leaves. destroy_process_group() would be a polite shutdown and would
        # prove nothing about the case under investigation.
        os._exit(1)
    if mode == "sigkill":
        os.kill(os.getpid(), signal.SIGKILL)
    # mode == "alive_silent" / "sigstop": stay up and send nothing more.
    time.sleep(600)


def _survivor_worker(init_file, out_dir, mode):
    res = {"outcome": None, "elapsed": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=SURVIVOR, world_size=WORLD
        )
        h = _make_holder(SURVIVOR, _RingWire(SURVIVOR))
        h._pp_recv_admission_decision()
        t0 = time.time()
        try:
            # THE CALL UNDER INVESTIGATION -- the shipped mandatory receive,
            # not a stand-in for it. Its peer is dead, dying, or silent.
            h._pp_recv_admission_decision()
            res["outcome"] = "returned"
        except BaseException as exc:  # noqa: BLE001 - the error IS the result
            res["outcome"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        res["elapsed"] = round(time.time() - t0, 2)
    except BaseException as exc:  # noqa: BLE001
        res["outcome"] = f"SETUP {type(exc).__name__}: {str(exc)[:300]}"
    finally:
        with open(os.path.join(out_dir, "survivor.json"), "w") as f:
            json.dump(res, f)


def _run(mode, bound_s):
    """Returns (survivor_still_blocked, result_or_None, pid_probe_said_alive)."""
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [
            ctx.Process(target=_sender_worker, args=(init_file, tmp, mode)),
            ctx.Process(target=_survivor_worker, args=(init_file, tmp, mode)),
        ]
        for p in procs:
            p.start()

        sender_pid = None
        deadline = time.time() + 30.0
        pid_path = os.path.join(tmp, "sender.pid")
        while time.time() < deadline and sender_pid is None:
            if os.path.exists(pid_path):
                try:
                    sender_pid = int(open(pid_path).read())
                except ValueError:
                    pass
            if sender_pid is None:
                time.sleep(0.05)

        if sender_pid is None:
            # The sender writes its pid only AFTER a successful send. No pid
            # means it never got that far, and every assertion below would
            # then be measuring a harness crash while reading as a verdict
            # about peer liveness -- the exact way this file failed first.
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join(timeout=5)
            raise AssertionError(
                f"[{mode}] the sender never completed its healthy send, so "
                f"this run measures a broken harness, not a peer state"
            )

        if mode == "sigstop" and sender_pid:
            time.sleep(PEER_LIFETIME_AFTER_SEND_S)
            os.kill(sender_pid, signal.SIGSTOP)

        procs[1].join(timeout=bound_s)
        still_blocked = procs[1].is_alive()

        # What the REFUTED fix's probe would have concluded at this instant --
        # the same primitive it would have used, read at the moment the
        # verdict matters. `pid_alive` treats PermissionError as "exists",
        # which is why it is called rather than a bare os.kill here.
        from sglang.srt.distributed.device_communicators.barlink_liveness import (
            pid_alive,
        )

        probe_says_alive = pid_alive(sender_pid) if sender_pid else None

        if mode == "sigstop" and sender_pid:
            try:
                os.kill(sender_pid, signal.SIGCONT)
            except OSError:
                pass
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)

        result = None
        path = os.path.join(tmp, "survivor.json")
        if os.path.exists(path):
            with open(path) as f:
                result = json.load(f)
        return still_blocked, result, probe_says_alive


class ADeadPeerIsReportedByTheTransport(unittest.TestCase):
    """The half that makes the pid probe REDUNDANT."""

    def _assert_raised_not_wedged(self, mode):
        blocked, result, _probe = _run(mode, DEATH_NOTICE_BOUND_S)
        self.assertFalse(
            blocked,
            f"[{mode}] the survivor was STILL blocked after "
            f"{DEATH_NOTICE_BOUND_S}s -- if this ever fails, the premise of "
            f"this whole file is wrong and a dead-peer check IS needed",
        )
        self.assertIsNotNone(result, f"[{mode}] the survivor wrote no result")
        self.assertNotEqual(
            result["outcome"],
            "returned",
            f"[{mode}] the receive returned a value from a dead peer: {result}",
        )
        self.assertIn(
            "Connection closed by peer",
            result["outcome"],
            f"[{mode}] expected the gloo transport to report the closed pair, "
            f"got: {result}",
        )

    def test_a_peer_that_exits_is_reported_not_waited_for(self):
        self._assert_raised_not_wedged("hard_exit")

    def test_a_peer_that_is_sigkilled_is_reported_not_waited_for(self):
        self._assert_raised_not_wedged("sigkill")


class ALivePeerWedgesAndThePidProbeCannotSeeIt(unittest.TestCase):
    """The half that makes the pid probe BLIND -- and without which the class
    above would only prove the harness is faithful, not that the probe is
    useless."""

    def _assert_wedged_and_probe_blind(self, mode):
        blocked, result, probe_says_alive = _run(mode, WEDGE_OBSERVATION_S)
        self.assertTrue(
            blocked,
            f"[{mode}] the survivor was NOT blocked at "
            f"{WEDGE_OBSERVATION_S}s; this arm has to reproduce the wedge or "
            f"it discriminates nothing. Result: {result}",
        )
        self.assertTrue(
            probe_says_alive,
            f"[{mode}] os.kill(pid, 0) reported the wedging peer as gone -- "
            f"then a pid probe WOULD cover this shape and the refusal in this "
            f"file's docstring is wrong",
        )

    def test_a_silent_peer_wedges_while_its_pid_reads_alive(self):
        self._assert_wedged_and_probe_blind("alive_silent")

    def test_a_stopped_peer_wedges_while_its_pid_reads_alive(self):
        self._assert_wedged_and_probe_blind("sigstop")


class TheAdmissionWireCannotReachTheDeviceLeg(unittest.TestCase):
    """Case 3: why "gloo reports it" settles the question.

    `GroupCoordinator.recv_tensor_dict` enters its `device_group` payload leg
    once per entry that is a real tensor. A peer death on THAT leg is not
    reported the way a closed TCP pair is. So the argument above holds only
    while the admission decision's wire dict stays primitives-only.
    """

    def _wire_dict(self):
        from sglang.srt.managers.pp_admission_congruence import (
            PPAdmissionDecision,
            PPAdmissionEntry,
        )
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_admission_decision_to_wire,
        )

        decision = PPAdmissionDecision(
            mb_id=3,
            entries=(
                PPAdmissionEntry(
                    rid="r0",
                    prefix_len=128,
                    extend_len=64,
                    admitted=True,
                    retracted=False,
                    retracted_by_rank=None,
                    observed_local=None,
                ),
            ),
        )
        return pp_admission_decision_to_wire(decision)

    def _tensors_in(self, value, path="<root>"):
        found = []
        if isinstance(value, torch.Tensor):
            found.append(path)
        elif isinstance(value, dict):
            for k, v in value.items():
                found.extend(self._tensors_in(v, f"{path}[{k!r}]"))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                found.extend(self._tensors_in(v, f"{path}[{i}]"))
        return found

    def test_the_decision_wire_dict_carries_no_tensor_anywhere(self):
        found = self._tensors_in(self._wire_dict())
        self.assertEqual(
            found,
            [],
            "a tensor on the admission wire would send this receive through "
            "GroupCoordinator.recv_tensor_dict's device_group leg, where a "
            "dead peer is NOT reported the way a closed gloo pair is -- and "
            "this file's refusal of a pid probe would no longer follow. "
            f"Tensors found at: {found}",
        )

    def test_the_detector_can_fail(self):
        """The guard above is only worth its line count if it would notice a
        tensor. Same shape as the sibling files' own can-fail arms."""
        poisoned = self._wire_dict()
        poisoned["__smuggled__"] = {"deep": [torch.zeros(2)]}
        self.assertEqual(
            self._tensors_in(poisoned),
            ["<root>['__smuggled__']['deep'][0]"],
        )


if __name__ == "__main__":
    unittest.main()
