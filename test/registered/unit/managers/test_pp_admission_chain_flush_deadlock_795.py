"""#795: does the #791 admission-decision channel, layered on top of the
already-fixed #788 chain-flush ordering, reproduce the fifth deadlock of the
"a rank must never block on a peer for something not required for THIS
iteration's forward progress" family -- SETTLED BY MEASUREMENT, not by
reasoning about the loop alone.

THE SPECIMEN (boot instr3, 2026-08-20, commit 3df2db37df,
evidence-665-f1/PROGRESS_788_787.log). Health reached, froze on the first
request, zero GPU utilisation, log frozen, zero ADMISSION-WEDGE markers (the
detector is blind to "every rank in a receive"). py-spy:

    PP0  blocked in `_pp_commit_pending_req_work` (scheduler_pp_mixin.py:802)
         -- the request-chain flush.
    PP1  blocked in `_pp_recv_admission_decision` (scheduler_pp_mixin.py:3262)
         from `_event_loop_pp_body:624` -- the FORWARD receive.
    PP2  same as PP1, waiting on PP1.

THE QUESTION THIS FILE SETTLES. `_event_loop_pp_body` already sends this
pass's admission decision (line 729/786) strictly BEFORE the chain flush
(line 802) -- the exact ordering #788's own fix established for the request
channel, and the exact ordering the #791 admission block's own comment
(scheduler_pp_mixin.py:699-720) says it was placed to satisfy, because an
earlier revision that put the send AFTER the flush produced this identical
signature. That earlier bug is already fixed in the code under test. Yet
boot instr3 reproduced the identical signature again, on code that already
has the fix. Two explanations were on the table:

  (A) ADDRESSING/DEMUX: `pp_typed_channel`'s `(src, kind)` inbox does not
      deliver the decision message PP1/PP2 are waiting for into the slot
      their receive reads, despite the sender having genuinely already
      transmitted it.
  (B) STRUCTURAL RING: despite the correct program order, PP0's own
      progress is genuinely gated on PP1/PP2 reaching a point they cannot
      reach without the very decision PP0 has not produced yet.

A PROOF THAT RULES OUT (B) UNDER THE FIXED ORDERING, BEFORE RUNNING ANYTHING.
`_pp_commit_pending_req_work` blocks EVERY iteration K on PP1 performing its
K-th `recv_requests()` call -- so PP0 cannot begin iteration K+1 until PP1
has started iteration K. While PP0's flush(K) is pending, PP0 has therefore
completed at most iteration K-1 in full (including that iteration's decision
send, which -- by the fixed ordering -- happened strictly before flush(K-1)
succeeded), and PP1 has performed at least K-1 `recv_requests()` calls (the
one that unblocked PP0's own flush(K-1)) but not yet K (or flush(K) would
already be satisfied). So PP1 is on iteration exactly K-1 -- the same
iteration whose decision PP0 already sent, in full, before its flush(K-1)
even returned. The message PP1 is blocked wanting was therefore already
transmitted, in program order, before PP1 could possibly have asked for it.
If PP1 is still waiting minutes later, the sender-side program order cannot
be the cause -- something on the DELIVERY side is. This is (A), not (B),
by construction, PROVIDED the two-channel (chain + decision) model captures
the real mechanism. This file exists to find out empirically whether it
does: either it hangs (independent, measured confirmation of (A) beyond the
proof above) or it does not (a NEGATIVE FINDING, honestly reported, meaning
the missing ingredient is a third channel sharing the same demultiplexer --
see the module docstring of test_pp_chain_flush_deadlock_788.py, which
named exactly this shape of gap for the #788 investigation and turned out
to be right: the OUTPUT channel was the missing ring-closing edge there).

METHOD. Two SEPARATE real gloo communicators, one per real production
channel, mirroring the fact that `world_group.cpu_group` (request chain)
and `pp_group`'s device/cpu group pair (admission decision, via
`pp_typed_channel`) are genuinely different `ProcessGroup` objects in
production:
  - chain: shipped `_pp_forward_and_process_input_requests`,
    `_pp_commit_comm_work`, `_pp_send_pyobj_to_next_stage`,
    `_pp_recv_pyobj_from_prev_stage`, `_pp_commit_pending_req_work`, bound
    onto a holder whose `world_group.cpu_group` is a real
    `dist.new_group(...)`. Copied from test_pp_chain_flush_deadlock_788.py.
  - decision: shipped `_pp_send_admission_decision`,
    `_pp_recv_admission_decision`, `_pp_try_recv_admission_decision`,
    `_pp_reconcile_incoming_admission`, `_pp_send_dict_to_next_stage`,
    `_pp_recv_typed_dict`, bound onto the SAME holder, with `pp_group` set
    to `_RingWire` (copied verbatim from
    test_pp_admission_wraparound_never_blocks.py), which uses the default
    WORLD communicator -- a different communicator from the chain group
    above, exactly as production keeps these on different `ProcessGroup`
    instances.

The per-rank driver in `_worker` below reproduces the shipped
`_event_loop_pp_body` order line for line, for every rank, every pass:
    chain_recv        <-> line 602-603 (every rank, every pass)
    chain_stage        <-> line 604 (`_pp_forward_and_process_input_requests`:
                           commits the PREVIOUS pass's send, stages THIS
                           pass's send async, unconditionally)
    decision_recv       <-> line 622-624 (non-first rank only, BLOCKING)
    decision_send       <-> line 720-786 (every rank; PP0 builds+sends;
                           downstream forwards the amended decision)
    chain_flush        <-> line 801-802 (non-last rank only, BLOCKING --
                           the #788-relocated end-of-iteration flush)
Everything else `_event_loop_pp_body` does between these lines
(`get_next_batch_to_run`, proxy tensors, launch_batch, output
send/receive, phase-flip round hook) is deliberately NOT modelled: proxy is
a documented no-op under this run's `_pp_gapped_wire` (scheduler_pp_mixin.py
:3121-3122, :3611-3623), and modelling the output ring or real GPU compute
skew is out of this file's scope for the same reason
test_pp_chain_flush_deadlock_788.py gave for not modelling output either.

CAN-FAIL, MEASURED, AND THE RESULT WAS NEGATIVE -- REPORTED PLAINLY.
`test_ordering_pin_discriminates_broken_send_site` was written expecting
that relocating the decision send to AFTER the chain flush (the pre-#791-fix
placement scheduler_pp_mixin.py:699-720's own comment blames for this exact
signature) would deadlock this two-channel model, proving the harness can
detect the family. MEASURED RESULT: it does NOT deadlock. All three ranks
complete cleanly under the "broken" ordering too. This is not a harness bug;
it is structurally forced by what a two-CHANNEL, two-INDEPENDENT-COMMUNICATOR
model actually contains: `chain_flush` and `decision_send` are two calls on
two disjoint `ProcessGroup`s with no shared state, so swapping their order
WITHIN one rank's own pass cannot change which cross-rank dependencies exist
-- it only changes the wall-clock moment, inside an already-fully-buffered
async send, at which a call that was never going to block anyway returns.
Reordering two calls that don't interact cannot manufacture a new edge in
the wait-for graph. The pre-#791 bug therefore could NOT have been reproduced
by this ordering swap in isolation either, in production -- which means
whatever made the historical swap deadlock in production depended on some
OTHER coupling between the two channels that this reduced model omits. See
`test_fixed_ordering_with_crossing_channel` below, which adds exactly that
coupling and gets a different, positive result.

THE MISSING INGREDIENT, FOUND BY READING THE CODE THE FIRST TWO TESTS
COULDN'T HAVE CAUGHT (scheduler_pp_mixin.py:586-802, pp_crossing_wire.py:
246-277, pp_typed_channel.py, scheduler.py:1523-1636). The mid-forward
CROSSING channel is not modelled by either test above, and it is NOT the
documented proxy no-op this file's original docstring assumed -- gapped-wire
crossings are real, unconditional traffic on this branch, every pass a
non-empty batch runs. Three facts make it the third ingredient:
  1. SHARED DEMULTIPLEXER: `PpGroupLink.__init__` (pp_crossing_wire.py:270)
     stores `self.group = pp_group` and both `send`/`recv` call
     `send_typed_tensor_dict` / `recv_typed_tensor_dict(self.group, ...,
     CROSSING_KIND)` (pp_crossing_wire.py:273-277) -- the identical `group`
     object and the identical `(src, kind)` inbox (pp_typed_channel.py:75-99)
     that `_pp_send_admission_decision` / `_pp_recv_admission_decision` use
     with `ADMISSION_DECISION_KIND`. Two kinds, one demultiplexer, exactly
     the shape pp_typed_channel.py's own module docstring warns is where a
     stashed message of the wrong kind waits for a consumer that never asks.
  2. CROSSING SENDS ARE BLOCKING, UNLIKE EVERYTHING ELSE ON THIS CHANNEL.
     `PpGroupLink.send` calls `send_typed_tensor_dict(..., CROSSING_KIND)`
     with `async_send` left at its default `False` (pp_typed_channel.py:107,
     pp_crossing_wire.py:274) -- a genuine synchronous rendezvous, unlike the
     proxy dict (`async_send=True`, line 689) and the admission decision
     (fire-and-forget, confirmed in this file's own `_RingWire` fidelity
     fix above). A crossing SEND can therefore itself block on the peer's
     matching receive.
  3. POSITION: crossing exchange happens inside `_pp_launch_batch`
     (line 654), which for EVERY rank -- PP0 included -- runs strictly
     BEFORE that rank's own admission-decision send (line 720/786) and, for
     non-first ranks, strictly AFTER that rank's decision_recv (line 622-624)
     and the `get_next_batch_to_run` it gates (line 630-637, which is what
     produces that rank's `cur_batch`, i.e. whether it enters `_pp_launch_
     batch` and its crossings at all). PP0's own decision send is also gated
     behind its own `_pp_launch_batch` in exactly the same way, since it is
     one call earlier in the same `if cur_batch:` sequence (line 653-660 vs
     720-730).

THE RING THIS PRODUCES, TRACED BY HAND BEFORE MEASURING (#753's own
"crossings from BOTH PP1 (after layer 31) and PP2 (after layer 35)",
pp_crossing_wire.py:22-26): on a pass where PP0 owns a layer that needs an
activation crossed in from PP1's span, PP0's `_pp_launch_batch` blocks on a
crossing RECEIVE from PP1. PP1 cannot reach the matching crossing SEND until
PP1 has (a) received and reconciled THIS pass's admission decision (line
622-624, blocking) and (b) derived a non-empty `cur_batch` from it and
entered its own `_pp_launch_batch`. PP0 has not sent this pass's decision
yet -- by fact 3 above, PP0's decision send for this pass is strictly AFTER
the very `_pp_launch_batch` call that is now blocked waiting on PP1. PP0 is
therefore waiting on a message (PP1's crossing) whose sender cannot produce
it without a message (PP0's decision) that PP0 has not sent, and cannot send
without first finishing the wait it is stuck in. `test_fixed_ordering_with_
crossing_channel` below adds exactly this coupling, using the SAME shipped
`send_typed_tensor_dict`/`recv_typed_tensor_dict` functions PpGroupLink uses,
on the SAME `_RingWire` instance the decision channel already shares (one
demultiplexer, as in production) -- and measures whether it deadlocks.
"""

import json
import os
import pickle
import tempfile
import time
import types
import unittest
from collections import deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60)

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2
LAST = WORLD - 1

#: HAND PIN #770: exceeds WORLD so PP0's post-threshold decision sends (the
#: steady-state shape, matching N_ITERS in test_pp_admission_wraparound_
#: never_blocks.py) are exercised, and comfortably more than the "froze on
#: the first request" specimen's single stalled pass -- if the fixed
#: ordering is going to deadlock at all, it has ample room to show it here.
N_PASSES = WORLD + 2

#: Same payload-size reasoning as test_pp_chain_flush_deadlock_788.py's
#: PAYLOAD constant: large enough that this environment's gloo backend
#: (measured there: no eager completion at any size from 8 to 1048576 bytes)
#: genuinely blocks the flush on the peer's recv rather than completing for
#: free.
PAYLOAD = [b"x" * 65536]

#: Generous bound for real process spawn + gloo init + N_PASSES of
#: genuinely instant (no artificial delay anywhere) work under the FIXED
#: ordering. Matches GREEN_JOIN_TIMEOUT_S in
#: test_pp_admission_wraparound_never_blocks.py.
GREEN_JOIN_TIMEOUT_S = 30.0

#: Bound for the CAN-FAIL broken-ordering case. Must be long enough that a
#: real deadlock has unambiguously formed (not merely "still starting up")
#: but short enough to keep the suite fast -- the broken case never
#: completes early by construction (unlike a flaky race), so there is no
#: timing window to lose.
RED_JOIN_TIMEOUT_S = 15.0


class _RingWire:
    """Real point-to-point tensor-dict transport over gloo, ring-default.

    Copied verbatim from test_pp_admission_wraparound_never_blocks.py. Uses
    the default WORLD communicator -- deliberately a DIFFERENT communicator
    from the chain channel's `dist.new_group(...)` below, mirroring
    production's separation between `pp_group` and `world_group.cpu_group`.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD
        # FIDELITY FIX (this file, over the copy in
        # test_pp_admission_wraparound_never_blocks.py): the original
        # version of this class called `.wait()` immediately after every
        # `isend`, even under `async_send=True` -- making EVERY send a
        # synchronous rendezvous regardless of the flag. That is not what
        # production does: `GroupCoordinator.send_tensor_dict` (parallel_
        # state.py:2334-2387) returns `P2PWork` handles WITHOUT waiting on
        # them under `async_send=True`, and `_pp_send_admission_decision`
        # (scheduler_pp_mixin.py) discards the return value entirely --
        # genuinely fire-and-forget, matching its own docstring. Measured
        # while building this file: with the old wait-always behaviour, the
        # two-channel model deadlocked with EVERY rank stuck inside its own
        # `_pp_send_admission_decision` call (a send-side artifact of this
        # transport, not the specimen's signature, since PP0's wraparound
        # receive is deliberately opportunistic/non-blocking and this
        # reduced model has no output channel to incidentally drain PP2's
        # wraparound sends the way production's does -- see
        # `_pp_try_recv_admission_decision`'s own docstring). Kept alive in
        # `self._inflight` rather than let go out of scope immediately,
        # since nothing here calls `.wait()` on them any more.
        self._inflight = []

    def send_tensor_dict(
        self, tensor_dict, dst=None, all_gather_group=None, async_send=False
    ):
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        buf = pickle.dumps(tensor_dict)
        size = torch.tensor([len(buf)], dtype=torch.long)
        payload = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
        if async_send:
            w1 = dist.isend(size, dst=dst)
            w2 = dist.isend(payload, dst=dst)
            self._inflight.append((w1, size))
            self._inflight.append((w2, payload))
        else:
            dist.send(size, dst=dst)
            dist.send(payload, dst=dst)
        return []

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        size = torch.zeros(1, dtype=torch.long)
        dist.recv(size, src=src)
        buf = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(buf, src=src)
        return pickle.loads(bytes(buf.numpy()))


def _make_holder(rank: int, wire: _RingWire, chain_group):
    """Bind the SHIPPED chain AND decision methods onto one holder, merging
    the two existing single-channel test patterns
    (test_pp_chain_flush_deadlock_788.py's chain holder and
    test_pp_admission_wraparound_never_blocks.py's decision holder)."""
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
    pp_group_flags = types.SimpleNamespace(
        is_first_rank=(rank == 0),
        is_last_rank=(rank == LAST),
    )
    # `wire` (the decision channel, _RingWire) needs the is_first_rank/
    # is_last_rank flags too, since `_pp_send_dict_to_next_stage` /
    # `_pp_recv_typed_dict` are bound as methods of the SAME holder and
    # read `self.pp_group.is_first_rank` / `.is_last_rank` -- so `pp_group`
    # itself must carry both the ring-wire transport methods AND these
    # flags. Graft the flags onto the wire instance directly.
    wire.is_first_rank = pp_group_flags.is_first_rank
    wire.is_last_rank = pp_group_flags.is_last_rank

    h = types.SimpleNamespace(
        ps=ps,
        pp_group=wire,
        world_group=types.SimpleNamespace(cpu_group=chain_group),
        send_req_work=[],
        process_input_requests=lambda recv_reqs: None,
        pp_phase_flip_armed=lambda: False,
        pp_flip_service=lambda: None,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        waiting_queue=[],
        tree_cache=None,
    )
    h._pp_flip_bump_sent = lambda chan: None
    # #789 HARNESS REPAIR (interface drift, no assertion touched):
    # _pp_send_dict_to_next_stage now publishes an 'entered the send'
    # count before the post, so a downstream can tell a RENDEZVOUS
    # sender from an idle one. This holder counts nothing, so a no-op
    # restores its previous behaviour rather than changing it.
    h._pp_flip_bump_attempted = lambda chan: None
    h._pp_flip_bump_consumed = lambda chan: None
    h._pp_boundary_stats = lambda: None
    for name in (
        # chain channel (test_pp_chain_flush_deadlock_788.py's set)
        "_pp_forward_and_process_input_requests",
        "_pp_commit_comm_work",
        "_pp_send_pyobj_to_next_stage",
        "_pp_recv_pyobj_from_prev_stage",
        "_pp_commit_pending_req_work",
        # decision channel (test_pp_admission_wraparound_never_blocks.py's set)
        "_pp_send_admission_decision",
        "_pp_recv_admission_decision",
        "_pp_try_recv_admission_decision",
        "_pp_reconcile_incoming_admission",
        "_pp_send_dict_to_next_stage",
        "_pp_recv_typed_dict",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _progress(out_dir, rank, msg):
    """Atomic per-rank progress marker -- copied from
    test_pp_admission_wraparound_never_blocks.py so a timed-out run can
    report per-rank stall location even though a genuinely stuck process
    never reaches its own result-writing `finally` block."""
    path = os.path.join(out_dir, f"progress_r{rank}.txt")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(msg)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _read_progress(out_dir, rank):
    path = os.path.join(out_dir, f"progress_r{rank}.txt")
    if not os.path.exists(path):
        return "<no progress recorded>"
    with open(path) as f:
        return f.read()


def _worker(rank, init_file, out_dir, variant, n_passes):
    """One rank's faithful per-pass driver, mirroring
    `_event_loop_pp_body` (scheduler_pp_mixin.py:586-802) line for line.
    `variant` is "fixed" (shipped ordering: decision send before chain
    flush), "broken" (decision send relocated to AFTER the chain flush --
    the pre-#791-fix placement scheduler_pp_mixin.py:699-720 describes), or
    "fixed_with_crossing" (shipped decision/flush ordering, PLUS the
    mid-`_pp_launch_batch` crossing channel: PP1 and PP2 each block sending
    one crossing to PP0 every pass, positioned exactly where line 654's
    `_pp_launch_batch` sits -- after this rank's own decision_recv/cur_batch
    derivation, strictly before this rank's own decision_send -- using the
    SAME shipped `send_typed_tensor_dict`/`recv_typed_tensor_dict` functions
    `PpGroupLink` uses, on the SAME `wire`/`pp_group` instance the decision
    channel already shares, one demultiplexer, as in production)."""
    from sglang.srt.distributed.pp_typed_channel import (
        CROSSING_KIND,
        recv_typed_tensor_dict,
        send_typed_tensor_dict,
    )
    from sglang.srt.managers.pp_admission_congruence import PPAdmissionDecision
    from sglang.srt.managers.scheduler_pp_mixin import _PP_ADMISSION_PENDING_SENDS_CAP

    broken = variant == "broken"
    with_crossing = variant == "fixed_with_crossing"
    res = {"rank": rank, "ok": False}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        chain_group = dist.new_group(ranks=list(range(WORLD)))
        wire = _RingWire(rank)
        h = _make_holder(rank, wire, chain_group)
        pending_sends = deque()

        for p in range(1, n_passes + 1):
            _progress(out_dir, rank, f"pass{p}_chain_recv")
            if rank == PP0:
                recv_reqs = list(PAYLOAD)
            else:
                recv_reqs = h._pp_recv_pyobj_from_prev_stage()

            _progress(out_dir, rank, f"pass{p}_chain_stage")
            h._pp_forward_and_process_input_requests(recv_reqs)

            amended = None
            if rank != PP0:
                _progress(out_dir, rank, f"pass{p}_decision_recv")
                incoming = h._pp_recv_admission_decision()
                _effective, amended = h._pp_reconcile_incoming_admission(incoming)

            def _send_decision():
                _progress(out_dir, rank, f"pass{p}_decision_send")
                if rank == PP0:
                    decision = PPAdmissionDecision(mb_id=p, entries=())
                    h._pp_send_admission_decision(decision)
                    pending_sends.append(p)
                    while len(pending_sends) > _PP_ADMISSION_PENDING_SENDS_CAP:
                        pending_sends.popleft()
                    if len(pending_sends) >= WORLD:
                        returned = h._pp_try_recv_admission_decision()
                        if returned is not None:
                            pending_sends.popleft()
                else:
                    fwd = amended if amended is not None else PPAdmissionDecision(
                        mb_id=p, entries=()
                    )
                    h._pp_send_admission_decision(fwd)

            def _flush_chain():
                if rank != LAST:
                    _progress(out_dir, rank, f"pass{p}_chain_flush")
                    h._pp_commit_pending_req_work()

            def _do_crossing():
                # `_pp_launch_batch`'s crossing exchange (line 654). PP0
                # receives from BOTH downstream ranks (#753: "crossings from
                # BOTH PP1 (after layer 31) and PP2 (after layer 35)"); PP1
                # and PP2 each send once to PP0. Real payload size, matching
                # this file's own PAYLOAD reasoning, so gloo does not
                # complete it eagerly/bufferedly.
                if rank == PP0:
                    _progress(out_dir, rank, f"pass{p}_crossing_recv_from_1")
                    recv_typed_tensor_dict(wire, CROSSING_KIND, src=PP1)
                    _progress(out_dir, rank, f"pass{p}_crossing_recv_from_2")
                    recv_typed_tensor_dict(wire, CROSSING_KIND, src=PP2)
                else:
                    _progress(out_dir, rank, f"pass{p}_crossing_send")
                    send_typed_tensor_dict(
                        wire, {"x": PAYLOAD[0]}, PP0, CROSSING_KIND
                    )

            if variant == "send_before_launch":
                # THE PROPOSED FIX, measured before touching production:
                # decision content is fully known by this point for every
                # rank (PP0 built it fresh above the loop's decision_recv
                # branch; downstream already reconciled `amended`), so
                # nothing requires deferring the SEND past the
                # crossing-dependent `_pp_launch_batch` the way shipped code
                # does (line 654 before line 720/786). Sending here instead
                # -- strictly between decision_recv/build and crossing --
                # means no rank's crossing wait can ever be blocked behind
                # its own not-yet-sent decision: the send is unconditionally
                # punctual, exactly as the governing rule requires.
                _send_decision()
                _do_crossing()
                _flush_chain()
            elif with_crossing:
                _do_crossing()
                _send_decision()
                _flush_chain()
            elif broken:
                # Pre-#791-fix placement: flush BEFORE the decision send --
                # the ordering scheduler_pp_mixin.py:699-720 says produced
                # this exact specimen.
                _flush_chain()
                _send_decision()
            else:
                # Shipped placement: decision send BEFORE the chain flush.
                _send_decision()
                _flush_chain()

            _progress(out_dir, rank, f"pass{p}_round_hook")

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


def _run(variant, n_passes=N_PASSES, join_timeout=GREEN_JOIN_TIMEOUT_S):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [
            ctx.Process(
                target=_worker, args=(r, init_file, tmp, variant, n_passes)
            )
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        deadline = time.time() + join_timeout
        for p in procs:
            p.join(timeout=max(0.1, deadline - time.time()))
        stuck_ranks = [r for r, p in enumerate(procs) if p.is_alive()]
        stall_report = {
            r: _read_progress(tmp, r) for r in stuck_ranks
        }
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)

        def _load(path, default):
            if not os.path.exists(path):
                return default
            with open(path) as f:
                return json.load(f)

        out = {"stuck_ranks": stuck_ranks, "stall_report": stall_report}
        for r in range(WORLD):
            out[f"result_{r}"] = _load(os.path.join(tmp, f"result_r{r}.json"), None)
        return out


class PPAdmissionChainFlushDeadlock795(unittest.TestCase):
    def test_fixed_ordering_completes(self):
        """GREEN under the SHIPPED ordering, if the two-channel model alone
        is sufficient to explain the specimen. If this instead times out
        with the specimen's own signature (PP0 stuck in a chain_flush
        marker, PP1/PP2 both stuck in a decision_recv marker), that is a
        measured, independent confirmation of hypothesis (A): the fixed
        program order does not save the ring, so the defect must be in
        delivery (the typed-channel demultiplex), not ordering."""
        res = _run("fixed")
        if res["stuck_ranks"]:
            self.fail(
                "SHIPPED ordering deadlocked under the two-channel "
                f"(chain + decision) hermetic model: {res}. Per-rank stall "
                f"markers: {res['stall_report']}. If PP0's marker is a "
                "chain_flush pass and PP1/PP2's markers are both "
                "decision_recv passes, this reproduces the instr3 specimen "
                "signature and is measured confirmation of hypothesis (A) "
                "(addressing/demux defect in pp_typed_channel, not "
                "ordering) -- see the module docstring's pre-run proof for "
                "why (B) is ruled out under this ordering by construction."
            )
        for r in range(WORLD):
            result = res[f"result_{r}"]
            self.assertIsNotNone(result, f"rank {r} produced no result: {res}")
            self.assertTrue(
                result.get("ok"), f"rank {r} failed: {result.get('error')}"
            )

    def test_ordering_pin_discriminates_broken_send_site(self):
        """NEGATIVE FINDING, measured, reported plainly rather than forced
        red. This test was originally written expecting the broken
        (pre-#791-fix) ordering to deadlock this two-channel model. It does
        not: all three ranks complete cleanly, `stuck_ranks == []`. See the
        module docstring's "CAN-FAIL, MEASURED, AND THE RESULT WAS NEGATIVE"
        section for why this is structurally forced (chain and decision are
        two calls on two disjoint communicators with no shared state, so
        swapping their order within one rank's own pass cannot create a new
        cross-rank dependency) rather than a harness defect -- and why it
        means the historical bug this ordering swap is blamed for must have
        depended on a coupling this reduced model omits, which
        `test_fixed_ordering_with_crossing_channel` below supplies and
        measures. This test is kept, asserting the negative result, as the
        record of that finding rather than deleted, per this task's
        instruction to say so plainly instead of manufacturing a hang to
        keep it red."""
        res = _run("broken", join_timeout=RED_JOIN_TIMEOUT_S)
        self.assertEqual(
            res["stuck_ranks"],
            [],
            "MEASURED: the broken (pre-#791-fix) decision-send/chain-flush "
            "ordering does NOT deadlock the two-channel (chain + decision) "
            f"model -- got stuck_ranks={res['stuck_ranks']!r}, "
            f"stall_report={res['stall_report']!r}. If this ever starts "
            "failing (some rank genuinely stuck), that is new information "
            "contradicting this file's own negative-finding measurement and "
            "must be re-investigated, not silenced.",
        )
        for r in range(WORLD):
            result = res[f"result_{r}"]
            self.assertIsNotNone(result, f"rank {r} produced no result: {res}")
            self.assertTrue(
                result.get("ok"), f"rank {r} failed: {result.get('error')}"
            )

    def test_fixed_ordering_with_crossing_channel(self):
        """THE POSITIVE RESULT. Adds the mid-`_pp_launch_batch` CROSSING
        channel (PP1 -> PP0 and PP2 -> PP0, one crossing each per pass with
        a non-empty batch) on top of the SHIPPED (fixed, correct) decision-
        send/chain-flush ordering, using the real `send_typed_tensor_dict`/
        `recv_typed_tensor_dict` functions on the SAME `_RingWire` instance
        the decision channel already shares -- one demultiplexer, as in
        production (pp_crossing_wire.py:270-277). See the module docstring's
        "THE RING THIS PRODUCES" section for the hand-traced mechanism this
        measures: PP0's `_pp_launch_batch` needs a crossing from PP1/PP2,
        whose own `_pp_launch_batch` needs THIS pass's admission decision,
        which PP0 has not sent yet because PP0's decision send is (correctly,
        per #791) positioned AFTER PP0's own `_pp_launch_batch` -- so PP0's
        crossing wait and its downstream peers' decision wait can close a
        ring even though the admission-decision ordering itself is correct.

        If this reproduces the specimen (PP0 stuck in a crossing_recv pass
        marker, PP1/PP2 stuck in a decision_recv pass marker) it is measured
        confirmation that the fifth deadlock is a genuine cross-CHANNEL
        structural ring (a variant of hypothesis (B), one level removed from
        the admission channel itself: PP0's OWN forward progress, not just
        its decision send, is what's entangled) that shares its
        demultiplexer with the admission channel (also implicating (A)'s
        general shape: two kinds, one inbox, is the fragility). If it does
        NOT reproduce, that is reported the same way as the test above --
        plainly, not manufactured."""
        res = _run("fixed_with_crossing", join_timeout=RED_JOIN_TIMEOUT_S)
        if not res["stuck_ranks"]:
            self.fail(
                "MEASURED NEGATIVE: adding the crossing channel on top of "
                "the correct decision-send/chain-flush ordering did NOT "
                f"deadlock this three-channel model either: {res}. The "
                "fifth deadlock's missing ingredient is not (only) the "
                "crossing channel modelled here; see this file's report for "
                "what remains unmodelled (real GPU compute skew between "
                "ranks, or the OUTPUT channel, which also shares this same "
                "demultiplexer)."
            )
        self.assertEqual(
            set(res["stuck_ranks"]),
            {PP0, PP1, PP2},
            f"expected all three ranks stuck once a deadlock forms in this "
            f"ring (partial hangs would themselves be a new, separately "
            f"reportable finding), got: {res}",
        )
        self.assertTrue(
            any(
                m in res["stall_report"].get(PP0, "")
                for m in ("crossing_recv", "chain_flush")
            ),
            f"PP0 expected to be stuck in a crossing_recv (or, if it raced "
            f"past that, a chain_flush) pass marker, was at: "
            f"{res['stall_report'].get(PP0)}",
        )
        for r in (PP1, PP2):
            self.assertIn(
                "decision_recv",
                res["stall_report"].get(r, ""),
                f"rank {r} expected to be stuck in a decision_recv pass, "
                f"was at: {res['stall_report'].get(r)}",
            )

    def test_send_before_launch_fixes_crossing_deadlock(self):
        """THE FIX, MEASURED. Same three-channel model as
        `test_fixed_ordering_with_crossing_channel` above (which deadlocks),
        with exactly one change: each rank's admission-decision send moves
        to occur BEFORE the crossing exchange instead of after -- i.e.
        before `_pp_launch_batch` (line 654) instead of after it, mirroring
        where PP0's decision content is actually finalised (scheduler.py:
        6896, inside `get_next_batch_to_run`, which already runs before
        `_pp_launch_batch`) rather than where it happens to be sent today.
        Nothing about WHAT is computed changes; only when the already-known
        decision is put on the wire. If this passes cleanly across
        N_PASSES, it is measured (not just reasoned) confirmation that
        moving the decision send earlier -- symmetrically, for every rank,
        the same relocation #791 already did across the chain-flush
        boundary, now applied across the crossing/launch_batch boundary --
        removes the ring `test_fixed_ordering_with_crossing_channel` found."""
        res = _run("send_before_launch")
        if res["stuck_ranks"]:
            self.fail(
                "the proposed fix (decision send before crossing exchange) "
                f"still deadlocks this three-channel model: {res}. Per-rank "
                f"stall markers: {res['stall_report']}. This means the "
                "relocation alone is not sufficient and the fix needs "
                "further work before being applied to scheduler_pp_mixin.py."
            )
        for r in range(WORLD):
            result = res[f"result_{r}"]
            self.assertIsNotNone(result, f"rank {r} produced no result: {res}")
            self.assertTrue(
                result.get("ok"), f"rank {r} failed: {result.get('error')}"
            )


if __name__ == "__main__":
    unittest.main()
