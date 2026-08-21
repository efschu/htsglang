"""#791b: a downstream retraction desynchronised the PP output ring.

THE SPECIMEN (boot instr11, 2026-08-21, evidence-665-f1/SPECIMEN_791_pp_ring_
wedge.txt + pyspy_wedge_instr11.txt). A TP=1/PP=3 boot with the phase flip
survived a 2.5 h idle, then wedged 4m18s into a flip-heavy drive with zero GPU
utilisation on all three cards and eight requests in flight:

    PP0  _pp_commit_send_output_work_and_preprocess_output_tensors
         -> _pp_send_recv_and_preprocess_output_tensors -> _do_recv
         -> _pp_recv_dict_from_prev_stage -> BLOCKING recv_typed_tensor_dict
    PP1  recv_requests -> pp_chain_receiver.recv      (TOP of its next pass)
    PP2  recv_requests -> pp_chain_receiver._advance  (TOP of its next pass)

PP0 waits on PP2's output, PP2 waits on PP1's chain, PP1 waits on PP0's chain.

THE MECHANISM, READ OFF THE BOOT LOG'S LAST FOUR PASSES. Three passes after
the tp_to_pp cutover at 05:00:33 the three ranks stopped agreeing:

    PP0  #788 PP-ADMISSION verdict=ADMIT n_reqs=1 rids=2f5e25a1... prefix_lens=512
    PP1  #791 PP-ADMISSION unhonourable prefix on rank 1: rid=2f5e25a1...
                                                          told=512 local=0
    PP1  #788 PP-ADMISSION verdict=DECLINE n_reqs=0
    PP2  #788 PP-ADMISSION verdict=DECLINE n_reqs=0

A #791 retraction travels DOWNSTREAM only -- `reconcile_pp_admission_decision`
amends the decision for "every remaining downstream rank", and PP0 is upstream
of every rank that can make one. So PP0 kept the batch, launched it, and two
passes later asked the ring for its output, while the last rank's slot for that
microbatch had been empty all along and it had sent nothing. The output ring's
two gates read two DIFFERENT ranks' `mbs` -- `_pp_send_output_to_next_stage`
reads the last rank's, `_do_recv` reads PP0's -- and #753 made them the same
EXPRESSION without being able to make them the same FACT.

THE #796 HYPOTHESIS IS KILLED, and this file pins why. #796 removed the last
rank's admission wraparound on the argument that PP0 was never required to
receive it. That message is `ADMISSION_DECISION_KIND`;
`recv_typed_tensor_dict` (pp_typed_channel.py:136-145) returns ONLY on
`expected_kind`, stashing everything else and looping. So a wraparound could
never have satisfied PP0's `expected_kind="output"` receive, before or after
#796, and restoring it would re-create one unmatched message per pass without
unwedging anything. `test_wraparound_kind_cannot_satisfy_an_output_receive`
holds that.

THE FIX, and why it is the same law twice. PP0's verdict for a slot is
published on the admission decision that already travels 0 -> 1 -> ... -> last
in the same pass (`_PP_OUTPUT_EXPECTED_KEY`), and the last rank honours THAT
rather than re-deriving one from a slot a retraction may have emptied -- if it
has no output for a slot PP0 expects one for, it sends a void. #791's own law
("decide admission on rank 0 and carry the decision") and #753's ("send and
receive must be the SAME QUESTION asked of the SAME batch"), applied to the
one gate pair both left rank-local. The void is not the bounded-recv corpse in
reverse: it exists only on a pass where PP0's own published verdict obliges it
to receive.

THE HAZARD IS REAL DATA, NOT A WIRE TRICK -- the harness note that matters
here. The three earlier files in this lineage had to sabotage the transport to
reproduce their defect; this one does not. `_hazard_local_match` gives PP1 a
COLD cache (`prefix_indices` empty) against a `told` of 512, exactly the boot
log's `told=512 local=0`, and the SHIPPED `_pp_reconcile_incoming_admission`
then produces the retraction on its own. The wire is #795's `_RingWire`
verbatim; nothing about it is made more forgiving, and nothing about it is
made more hostile.

THE SPAWN TRAP, restated because #796 paid for it: `_run` uses the "spawn"
start method, so a patch applied inside a test METHOD reaches no child. The
red arm's neutering therefore happens in `_red_worker`, IN THE CHILD.
"""

import json
import os
import tempfile
import time
import types
import unittest
from collections import deque

import torch.multiprocessing as mp

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2
LAST = WORLD - 1

#: One slot per stage: `pp_async_batch_depth == 0`, the shape boot instr11
#: ran (its py-spy shows the depth-0 call site, scheduler_pp_mixin.py:779).
LOOP = WORLD

#: The pass on which PP1's cache is cold. Late enough that the ring is in
#: steady state (PP0 is two passes ahead and every gate has answered "due" at
#: least once), early enough to leave room for the wedge to form and for the
#: fixed run to recover and keep going.
RETRACT_PASS = 4

#: THE LAST PASS THAT CARRIES WORK, AND IT IS THE RETRACTED ONE. This is not
#: harness convenience; it is the specimen's own shape and the reason the
#: wedge is a wedge. A skipped output send in the MIDDLE of a busy stretch is
#: only a lag: the pair is a FIFO, PP0 does not care which message it gets,
#: and the next pass's output releases it. It becomes a deficit exactly when
#: the pipeline goes idle behind the retraction, so no later message can
#: cover the missing one -- which is what boot instr11's log shows, three
#: DECLINE passes on every rank after `unhonourable prefix on rank 1` and
#: then silence. Measured here too: with work on every pass this same
#: retraction produces a run that completes, and the red arm goes green for
#: the wrong reason.
LAST_BUSY_PASS = RETRACT_PASS

#: Enough passes after RETRACT_PASS that a fixed run has to survive the void
#: AND the passes on either side of it, including the one where PP0 forwards
#: nothing because it absorbed a void.
N_PASSES = 12

#: What PP0 tells downstream about the hazard rid. The boot log's own value.
TOLD = 512

#: Same payload-size reasoning as test_pp_admission_chain_flush_deadlock_795.py:
#: large enough that this environment's gloo backend does not complete the
#: chain flush for free.
PAYLOAD = [b"x" * 65536]

#: Generous bound for spawn + gloo init + N_PASSES of instant work.
GREEN_JOIN_TIMEOUT_S = 60.0

#: The wedged arm never completes by construction, so there is no timing
#: window to lose -- this only has to outlast spawn and gloo init.
RED_JOIN_TIMEOUT_S = 25.0


def _rid_for(pass_idx: int) -> str:
    return f"rid{pass_idx:04d}"


class _StubReq:
    """The three things `_pp_reconcile_incoming_admission` touches on a
    waiting request, and the one thing `_pp_absorb_void_output` does."""

    def __init__(self, rid: str, local_match: int):
        self.rid = rid
        self.prefix_indices = [0] * local_match
        self.retraction_count = 0
        self.reset_calls = 0

    def init_next_round_input(self, tree_cache):
        return None

    def reset_for_retract(self):
        self.retraction_count += 1
        self.reset_calls += 1


class _StubBatch:
    """Only what the two output-ring gates read.

    `contains_last_prefill_chunk=True` makes `_pp_can_skip_output_comm` False
    on its own terms, so the gate's answer does not depend on an env var this
    test has no business setting.
    """

    def __init__(self, reqs):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.forward_mode = ForwardMode.EXTEND
        self.reqs = list(reqs)
        self.contains_last_prefill_chunk = True
        self.return_logprob = False


def _hazard_local_match(rank: int, pass_idx: int) -> int:
    """The specimen's divergence, as data: on RETRACT_PASS, PP1's cache is
    cold (`local=0`) against PP0's `told=512`. Every other rank and pass
    matches, so the ring is otherwise in perfect congruence."""
    if rank == PP1 and pass_idx == RETRACT_PASS:
        return 0
    return TOLD


def _make_holder(rank, wire, chain_group, out_dir):
    """Bind the SHIPPED chain, decision and OUTPUT-RING methods onto one
    holder. Extends test_pp_admission_chain_flush_deadlock_795.py's holder
    with the output ring, which is the channel this file is about."""
    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionCongruenceGuard,
    )
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    ps = types.SimpleNamespace(
        pp_rank=rank,
        pp_size=WORLD,
        tp_size=1,
        attn_tp_rank=0,
        attn_cp_rank=0,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_tp_size=1,
    )
    wire.is_first_rank = rank == PP0
    wire.is_last_rank = rank == LAST

    class _Stream:
        def wait_stream(self, other):
            return None

        def wait_event(self, event):
            return None

    class _Event:
        def record(self, stream=None):
            return None

        def synchronize(self):
            return None

    class _DeviceModule:
        Event = _Event

        @staticmethod
        def current_stream():
            return _Stream()

    class _NullCtx:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    h = types.SimpleNamespace(
        ps=ps,
        pp_group=wire,
        world_group=types.SimpleNamespace(cpu_group=chain_group),
        server_args=types.SimpleNamespace(pp_async_batch_depth=0),
        send_req_work=[],
        send_output_work=[],
        send_proxy_work=[],
        process_input_requests=lambda recv_reqs: None,
        pp_phase_flip_armed=lambda: False,
        pp_flip_service=lambda: None,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        waiting_queue=[],
        tree_cache=None,
        pp_loop_size=LOOP,
        mbs=[None] * LOOP,
        last_mbs=[None] * LOOP,
        mb_metadata=[None] * LOOP,
        last_rank_comm_queue=deque(),
        pp_outputs=None,
        _pp_gapped_wire=False,
        _pp_admission_guard=PPAdmissionCongruenceGuard(),
        _pp_output_expected_by_slot=[False] * LOOP,
        _pp_admission_amended_by_slot=[None] * LOOP,
        _pp_output_expected_incoming=False,
        _pp_admission_send_work=[],
        copy_stream=_Stream(),
        schedule_stream=_Stream(),
        copy_stream_ctx=_NullCtx(),
        device_module=_DeviceModule(),
        req_to_token_pool=None,
    )
    h._pp_flip_bump_sent = lambda chan: None
    h._pp_flip_bump_attempted = lambda chan: None
    h._pp_flip_bump_consumed = lambda chan: None
    h._pp_boundary_stats = lambda: None
    h._pp_prep_batch_result = lambda target, meta, outputs: {"result_for": id(target)}
    for name in (
        # chain channel
        "_pp_forward_and_process_input_requests",
        "_pp_commit_comm_work",
        "_pp_send_pyobj_to_next_stage",
        "_pp_recv_pyobj_from_prev_stage",
        "_pp_commit_pending_req_work",
        # decision channel
        "_pp_send_admission_decision",
        "_pp_recv_admission_decision",
        "_pp_reconcile_incoming_admission",
        "_pp_commit_admission_send_work",
        "_pp_send_dict_to_next_stage",
        "_pp_recv_typed_dict",
        # output ring -- the channel under test
        "_pp_note_output_expectation",
        "_pp_output_expected_for_slot",
        "_pp_void_output_payload",
        "_pp_absorb_void_output",
        "_pp_send_output_to_next_stage",
        "_pp_recv_dict_from_prev_stage",
        "_pp_send_recv_and_preprocess_output_tensors",
        "_pp_commit_send_output_work_and_preprocess_output_tensors",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _worker(rank, init_file, out_dir, n_passes):
    """One rank's driver, mirroring `_event_loop_pp_body` for the three
    channels that closed the ring: request chain, admission decision, output.

    Everything that touches the wire is the shipped method. What the harness
    supplies is only what a scheduler would have supplied anyway: which rids
    PP0 admits, and what each rank's own cache matches for them.
    """
    import torch
    import torch.distributed as dist

    import test_pp_admission_chain_flush_deadlock_795 as ring
    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionDecision,
        PPAdmissionEntry,
    )
    from sglang.srt.managers.scheduler_pp_mixin import _pp_output_exchange_due
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    res = {"rank": rank, "ok": False, "passes": 0, "voids": 0, "requeued": 0}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        chain_group = dist.new_group(ranks=list(range(WORLD)))
        wire = ring._RingWire(rank)
        h = _make_holder(rank, wire, chain_group, out_dir)

        for p in range(n_passes):
            mb_id = p % LOOP
            next_first_rank_mb_id = (mb_id + WORLD) % LOOP
            next_mb_id = (mb_id + 1) % LOOP

            ring._progress(out_dir, rank, f"pass{p}_chain_recv")
            if rank == PP0:
                recv_reqs = list(PAYLOAD)
            else:
                recv_reqs = h._pp_recv_pyobj_from_prev_stage()
            h._pp_forward_and_process_input_requests(recv_reqs)

            busy = p <= LAST_BUSY_PASS
            rid = _rid_for(p)
            if busy:
                h.waiting_queue.append(_StubReq(rid, _hazard_local_match(rank, p)))

            h._pp_output_expected_incoming = False
            amended = None
            if rank != PP0:
                ring._progress(out_dir, rank, f"pass{p}_decision_recv")
                incoming = h._pp_recv_admission_decision()
                effective, amended = h._pp_reconcile_incoming_admission(incoming)
                h._pp_note_output_expectation(
                    mb_id, h._pp_output_expected_incoming, amended
                )
            else:
                effective = {rid: TOLD} if busy else {}

            # The slot this rank will run: exactly the rids it may admit.
            admitted = [r for r in h.waiting_queue if r.rid in effective]
            for req in admitted:
                h.waiting_queue.remove(req)
            h.mbs[mb_id] = _StubBatch(admitted) if admitted else None

            ring._progress(out_dir, rank, f"pass{p}_decision_send")
            if rank == PP0:
                entries = (
                    (PPAdmissionEntry(rid=rid, prefix_len=TOLD, extend_len=1),)
                    if busy
                    else ()
                )
                decision = PPAdmissionDecision(mb_id=mb_id, entries=entries)
                expects_output = _pp_output_exchange_due(h.mbs[mb_id])
                h._pp_note_output_expectation(mb_id, expects_output, None)
                h._pp_send_admission_decision(decision, expects_output=expects_output)
            else:
                fwd = (
                    amended
                    if amended is not None
                    else PPAdmissionDecision(mb_id=mb_id, entries=())
                )
                h._pp_send_admission_decision(
                    fwd, expects_output=h._pp_output_expected_incoming
                )

            # `_pp_launch_batch`'s one effect the output ring depends on.
            if rank == LAST and h.mbs[mb_id] is not None:
                h.last_rank_comm_queue.append(
                    (
                        object(),
                        PPProxyTensors(
                            {"next_token_ids": torch.tensor([p], dtype=torch.int64)}
                        ),
                    )
                )

            ring._progress(out_dir, rank, f"pass{p}_output_exchange")
            before = len(h.waiting_queue)
            npo, nbr, d2h = h._pp_commit_send_output_work_and_preprocess_output_tensors(
                next_first_rank_mb_id, next_mb_id
            )
            if len(h.waiting_queue) > before:
                res["voids"] += 1
                res["requeued"] += len(h.waiting_queue) - before
            h.pp_outputs = npo

            # `_event_loop_pp_body`'s invariant: a non-empty slot means a
            # result arrived for it. A void must have emptied the slot, not
            # left it holding a batch with no result.
            if h.mbs[next_mb_id] is not None and nbr is None:
                raise AssertionError(
                    f"pass {p}: slot {next_mb_id} is occupied but no result "
                    f"arrived -- d2h_event.synchronize() would raise here"
                )
            h.last_mbs[next_mb_id] = h.mbs[next_mb_id]

            ring._progress(out_dir, rank, f"pass{p}_chain_flush")
            h._pp_commit_admission_send_work()
            if rank != LAST:
                h._pp_commit_pending_req_work()
            res["passes"] = p + 1

        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        with open(os.path.join(out_dir, f"result_r{rank}.json"), "w") as f:
            json.dump(res, f)
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:  # noqa: BLE001 - best-effort teardown only
                pass


def _red_worker(rank, init_file, out_dir, n_passes):
    """The SAME driver with ONLY the fix neutered, IN THE CHILD.

    `_pp_output_expected_for_slot` returning False is exactly the pre-fix
    last rank: it re-derives the ring's verdict from its own slot and sends
    nothing when a retraction emptied it. Nothing else is touched -- not the
    transport, not the decision codec, not the retraction.
    """
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    SchedulerPPMixin._pp_output_expected_for_slot = lambda self, mb_id: False
    return _worker(rank, init_file, out_dir, n_passes)


def _run(target, n_passes, join_timeout):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [
            ctx.Process(target=target, args=(r, init_file, tmp, n_passes))
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        deadline = time.time() + join_timeout
        for p in procs:
            p.join(timeout=max(0.1, deadline - time.time()))
        stuck_ranks = [r for r, p in enumerate(procs) if p.is_alive()]

        import test_pp_admission_chain_flush_deadlock_795 as ring

        stall_report = {r: ring._read_progress(tmp, r) for r in stuck_ranks}
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)

        out = {"stuck_ranks": stuck_ranks, "stall_report": stall_report}
        for r in range(WORLD):
            path = os.path.join(tmp, f"result_r{r}.json")
            out[f"result_{r}"] = None
            if os.path.exists(path):
                with open(path) as f:
                    out[f"result_{r}"] = json.load(f)
        return out


class PPOutputRingRetractionWedge791b(unittest.TestCase):
    def test_retraction_wedges_the_ring_without_the_fix(self):
        """RED, and the can-fail proof: neuter ONLY the fix and the boot
        instr11 signature comes back -- PP0 in the output receive, PP1 and
        PP2 both at the top of their next pass in the chain receive."""
        out = _run(_red_worker, N_PASSES, RED_JOIN_TIMEOUT_S)

        self.assertEqual(
            out["stuck_ranks"],
            [PP0, PP1, PP2],
            f"a downstream retraction was expected to close the ring: {out}",
        )
        stalls = out["stall_report"]
        self.assertIn(
            "output_exchange",
            stalls[PP0],
            f"PP0 should be stuck in the output exchange, where py-spy found "
            f"it on boot instr11: {stalls}",
        )
        for downstream in (PP1, PP2):
            self.assertIn(
                "chain_recv",
                stalls[downstream],
                f"PP{downstream} should be stuck at the TOP of its next pass "
                f"in the chain receive, as py-spy found both: {stalls}",
            )
        # The specimen's other half: the two ranks waiting on PP0's chain are
        # AHEAD of it. That inversion of the normal stagger (PP0 leads by
        # pp_size - 1) is what a missing output message produces, and it is
        # why the wedge reads as a ring rather than as a slow rank.
        pp0_pass = int(stalls[PP0].split("_", 1)[0][len("pass") :])
        for downstream in (PP1, PP2):
            other = int(stalls[downstream].split("_", 1)[0][len("pass") :])
            self.assertGreater(
                other,
                pp0_pass,
                f"PP{downstream} should have overtaken the rank it blocks: {stalls}",
            )

    def test_ring_survives_the_retraction_with_the_fix(self):
        """GREEN on the shipped code: every rank completes every pass, the
        void is actually taken, and PP0's requests come back to the queue."""
        out = _run(_worker, N_PASSES, GREEN_JOIN_TIMEOUT_S)

        self.assertEqual(out["stuck_ranks"], [], f"the fixed ring still wedged: {out}")
        for r in range(WORLD):
            res = out[f"result_{r}"]
            self.assertIsNotNone(res, f"rank {r} produced no result: {out}")
            self.assertTrue(res["ok"], f"rank {r} failed: {res}")
            self.assertEqual(res["passes"], N_PASSES, f"rank {r}: {res}")
        self.assertEqual(
            out[f"result_{PP0}"]["voids"],
            1,
            f"exactly one microbatch was retracted, so exactly one void "
            f"should have reached PP0: {out[f'result_{PP0}']}",
        )
        self.assertEqual(
            out[f"result_{PP0}"]["requeued"],
            1,
            f"the voided microbatch's request must come back to the waiting "
            f"queue, not be stranded holding KV: {out[f'result_{PP0}']}",
        )
        for downstream in (PP1, PP2):
            self.assertEqual(
                out[f"result_{downstream}"]["voids"],
                0,
                f"only the first rank ever absorbs a void: "
                f"{out[f'result_{downstream}']}",
            )

    def test_wraparound_kind_cannot_satisfy_an_output_receive(self):
        """The #796 hypothesis, killed at the source.

        The removed wraparound was an `admission_decision` message.
        `recv_typed_tensor_dict` returns only on `expected_kind` and stashes
        everything else, so no number of wraparounds could ever have released
        PP0's `expected_kind="output"` receive. Restoring that send would put
        one unmatched message per pass back on the channel and unwedge
        nothing.
        """
        from sglang.srt.distributed.pp_typed_channel import (
            MSG_TYPE_KEY,
            recv_typed_tensor_dict,
            typed_inbox,
        )
        from sglang.srt.managers.scheduler_pp_mixin import ADMISSION_DECISION_KIND

        delivered = [
            {MSG_TYPE_KEY: ADMISSION_DECISION_KIND, "lap": 1},
            {MSG_TYPE_KEY: ADMISSION_DECISION_KIND, "lap": 2},
            {MSG_TYPE_KEY: "output", "next_token_ids": "the message it wanted"},
        ]

        class _Group:
            rank_in_group = 0
            world_size = WORLD

            def recv_tensor_dict(self, src=None, all_gather_group=None):
                if not delivered:
                    raise AssertionError(
                        "the output receive asked the wire for a FOURTH "
                        "message: every wraparound before it was stashed, "
                        "not returned -- which is the whole point"
                    )
                return delivered.pop(0)

        group = _Group()
        got = recv_typed_tensor_dict(group, "output")
        self.assertEqual(got["next_token_ids"], "the message it wanted")
        self.assertEqual(
            len(typed_inbox(group)[(WORLD - 1, ADMISSION_DECISION_KIND)]),
            2,
            "both wraparounds should have been stashed rather than returned",
        )

    def test_void_payload_is_only_sent_when_the_first_rank_expects_one(self):
        """The law the void must not break: a rank may not post a send no
        peer is required to take. With no published expectation there is no
        void, so an ordinary idle pass puts nothing on the wire."""
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        sent = []

        def _fake_send(tensor_dict, async_send=True, msg_type="default", stamp=None):
            sent.append((tensor_dict, msg_type))
            return []

        h = types.SimpleNamespace(
            pp_group=types.SimpleNamespace(is_last_rank=True),
            _pp_gapped_wire=False,
            pp_loop_size=LOOP,
            _pp_output_expected_by_slot=[False] * LOOP,
            _pp_admission_amended_by_slot=[None] * LOOP,
        )
        h._pp_send_dict_to_next_stage = _fake_send
        for name in (
            "_pp_send_output_to_next_stage",
            "_pp_output_expected_for_slot",
            "_pp_void_output_payload",
            "_pp_note_output_expectation",
        ):
            setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))

        h._pp_send_output_to_next_stage(0, [None] * LOOP, deque(), None)
        self.assertEqual(sent, [], "an idle pass must put nothing on the wire")

        h._pp_note_output_expectation(0, True, None)
        h._pp_send_output_to_next_stage(0, [None] * LOOP, deque(), None)
        self.assertEqual(len(sent), 1, "the void was not sent")
        self.assertEqual(sent[0][1], "output")
        self.assertTrue(sent[0][0]["__pp_void_output__"])


if __name__ == "__main__":
    unittest.main()
