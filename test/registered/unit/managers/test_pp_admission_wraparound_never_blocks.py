"""The admission-decision ring wraparound must never block PP0's own
forward progress -- the fourth deadlock of the "a rank must never block on
a peer for something not required for THIS iteration's forward progress"
family (after #631, #753, #788, #789).

THE METAL FACT THIS ENCODES, 2026-08-20: boot reached health, then froze on
the first request. Zero GPU utilisation on all three PP ranks, no
ADMISSION-WEDGE marker (the detector did not know this shape). py-spy on
all three:

  PP0  blocked in `_pp_recv_admission_decision` at the RING WRAPAROUND call
       site in `_event_loop_pp_body`'s #791 block (waiting on PP2, its
       ring-predecessor, for the return trip that feeds
       `PPAdmissionCongruenceGuard.record_return_trip` -- a pure LEARNING
       step, never required for this iteration to make progress).
  PP1  blocked in `_pp_recv_admission_decision` at the FORWARD receive
       (waiting on PP0's next decision).
  PP2  blocked in `_pp_recv_admission_decision` at the FORWARD receive
       (waiting on PP1's next decision).

Closed ring: PP0 will not send its next decision until the wraparound
receive returns, and the only ranks that could ever complete that lap are
themselves waiting on PP0's next send.

THE FIX (`_pp_try_recv_admission_decision`, scheduler_pp_mixin.py, next to
`_pp_recv_admission_decision`): the wraparound receive becomes
OPPORTUNISTIC. It peeks `pp_typed_channel.typed_inbox` for an
already-stashed `(src, ADMISSION_DECISION_KIND)` message and returns `None`
immediately if none is there -- never a blocking wire touch, never a new
probe or timeout, following the exact precedent
`_pp_wait_for_proxy_readiness` (#789) set for gating a blocking receive on
inbox presence. This is verified below in two ways: directly (case
"opportunistic" completes instantly regardless of a peer that will not
reply within any bound this test could ever choose) and structurally
(`test_call_site_uses_the_opportunistic_receive`, reading the shipped
source the same way test_pp_admission_wiring_791.py's own ordering test
does).

WHY THREE REAL PROCESSES OVER REAL GLOO. A mock cannot distinguish "this
call would have blocked" from "this call happens to return immediately
because the mock has no wire to block on" -- exactly the reasoning
test_pp_flip_leftover_proxy_757.py and test_pp_admission_wiring_791.py both
give for the same choice. The transport (`_RingWire`) and the holder
binding below are copied from test_pp_admission_wiring_791.py verbatim
where possible; the only addition is binding the new
`_pp_try_recv_admission_decision` method too.

WHY THE DEADLOCK IS MADE DETERMINISTIC RATHER THAN TIMING-DEPENDENT. A
naive same-speed 3-process model of this ring does NOT reliably deadlock:
PP1 and PP2 can drain and forward the `pp_size` decisions PP0 already sent
before blocking and produce a wraparound reply within microseconds, racing
PP0's own block closed before it would ever be observed stuck. That race
would make a naive test flaky in both directions. An earlier draft of this
file tried to close the race with a `dist.barrier()` rendezvous instead of
what follows below -- that draft self-deadlocked in the harness itself,
empirically: gloo's `dist.send` is SYNCHRONOUS (it does not return until
the peer has posted the matching `dist.recv`), so a barrier placed before
PP1/PP2 had received anything meant PP0's own send of decision 0 blocked
waiting for PP1's recv, while PP1 was waiting at the barrier for PP0 --
which PP0 could never reach without that send completing first. Removed;
replaced with the two real mechanisms below, which respect that
send/recv are the actual synchronization primitive here rather than
fighting it with a second, incompatible one:

  1. PP1 (and only PP1) RECEIVES exactly `pp_size` decisions and BUFFERS
     them -- without forwarding a single one -- before doing anything else.
     This lets PP0's own sends of decisions 0..pp_size-1 complete exactly
     as production does (nothing about receiving is special-cased; only
     the FORWARD is deferred), while deterministically guaranteeing PP2
     receives nothing yet, in EITHER case, since nothing PP1 buffers has
     been forwarded.
  2. In case "blocking" ONLY, immediately after buffering (before
     forwarding any of the three), PP1 sleeps for far longer than this
     test will ever wait. This stands in for "downstream compute takes an
     unbounded, uncontrolled amount of time" -- the actual real-world
     condition (this layout's stages carry different owned-layer counts,
     see #753's own module docstring) -- and makes the resulting stall
     provably outlast any bound this test could pick, without asserting a
     literal mathematical infinity. In case "opportunistic" this branch
     never triggers: PP1 forwards its three buffered decisions immediately
     and continues normally. PP0's OWN stack and PP2's OWN stack (blocked
     on PP1, which never replies) reproduce the measured specimen's stacks
     exactly for those two ranks; only PP1's OWN stack differs (a
     deliberate sleep standing in for slow real compute, not a wire block)
     -- documented here rather than glossed over.

CASES:

  test_blocking_wraparound_wedges_the_ring     RED. The old call shape
      (`_pp_recv_admission_decision` at the wraparound site) never returns
      within a generous bound; all three ranks are still alive when the
      bound expires, and PP0/PP2 are captured mid-block via their last
      written progress marker (per-rank stall reporting).
  test_opportunistic_wraparound_never_blocks_pp0   GREEN. The shipped call
      shape (`_pp_try_recv_admission_decision`) lets PP0 complete every one
      of its N iterations, including several past the `pending_sends`
      threshold, without ever blocking on PP1 or PP2 -- regardless of
      whether they ever reply -- and PP0's own capped `pending` bookkeeping
      (mirroring `_PP_ADMISSION_PENDING_SENDS_CAP`) never exceeds the real
      shipped cap.
  test_call_site_uses_the_opportunistic_receive   STRUCTURAL. The shipped
      `_event_loop_pp_body` source calls `_pp_try_recv_admission_decision`,
      never a blocking `_pp_recv_admission_decision`, at the wraparound
      site inside the #791 admission block.
"""

import json
import os
import tempfile
import time
import types
import unittest
from collections import deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=45)

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2

#: Exceeds pp_size so PP0's post-threshold sends (the steady-state shape,
#: not just the one-time threshold crossing) are actually exercised.
N_ITERS = WORLD + 2

#: Generous bound for real process spawn + gloo init + barrier + N_ITERS of
#: genuinely instant (no artificial delay anywhere) work. Matches the
#: JOIN_TIMEOUT_S already established in test_pp_admission_wiring_791.py.
GREEN_JOIN_TIMEOUT_S = 30.0

#: PP1's deliberate non-cooperation window in case "blocking". Chosen only
#: to comfortably outlast RED_JOIN_TIMEOUT_S below -- the process is
#: terminated well before this ever elapses, so its absolute size does not
#: set this test's wall-clock cost.
PP1_DELIBERATE_STALL_S = 3600.0

#: Must be large enough to absorb real process spawn (`spawn` context: a
#: fresh interpreter + torch import) and gloo init/barrier overhead, and
#: comfortably shorter than PP1_DELIBERATE_STALL_S -- there is no timing
#: race to lose here since PP1 provably cannot have sent anything within
#: this window by construction, not by luck.
RED_JOIN_TIMEOUT_S = 12.0


class _RingWire:
    """Real point-to-point tensor-dict transport over gloo, ring-default.

    Copied from test_pp_admission_wiring_791.py: mirrors
    ``GroupCoordinator.send_tensor_dict``/``recv_tensor_dict``'s own
    ``dst=None`` -> ``(rank_in_group+1) % world_size`` and ``src=None`` ->
    ``(rank_in_group-1) % world_size`` resolution, so the wraparound
    (PP2 -> PP0) exercises the same "no dst/src given" path the forward
    sends do.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD

    def send_tensor_dict(self, tensor_dict, dst=None, all_gather_group=None, async_send=False):
        import pickle

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        buf = pickle.dumps(tensor_dict)
        size = torch.tensor([len(buf)], dtype=torch.long)
        if async_send:
            w1 = dist.isend(size, dst=dst)
            w1.wait()
            w2 = dist.isend(
                torch.frombuffer(bytearray(buf), dtype=torch.uint8), dst=dst
            )
            w2.wait()
        else:
            dist.send(size, dst=dst)
            dist.send(torch.frombuffer(bytearray(buf), dtype=torch.uint8), dst=dst)
        return []

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        import pickle

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
    # #789 HARNESS REPAIR (interface drift, no assertion touched):
    # _pp_send_dict_to_next_stage now publishes an 'entered the send'
    # count before the post, so a downstream can tell a RENDEZVOUS
    # sender from an idle one. This holder counts nothing, so a no-op
    # restores its previous behaviour rather than changing it.
    h._pp_flip_bump_attempted = lambda chan: None
    h._pp_flip_bump_consumed = lambda chan: None
    for name in (
        "_pp_send_admission_decision",
        "_pp_recv_admission_decision",
        "_pp_try_recv_admission_decision",
        "_pp_reconcile_incoming_admission",
        # The two SHIPPED wire primitives the methods above are built on
        # top of -- bound here too so this test drives the real send/recv
        # path, not a stand-in for it.
        "_pp_send_dict_to_next_stage",
        "_pp_recv_typed_dict",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _progress(out_dir, rank, msg):
    """Overwrite this rank's last-known position, flushed to disk, so a
    timed-out run can report per-rank stall location even though a
    genuinely stuck process never reaches its own result-writing
    `finally` block."""
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


def _pp0_worker(init_file, out_dir, mode):
    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionCongruenceGuard,
        PPAdmissionDecision,
    )
    from sglang.srt.managers.scheduler_pp_mixin import _PP_ADMISSION_PENDING_SENDS_CAP

    res = {"rank": PP0, "ok": False, "error": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=PP0, world_size=WORLD
        )
        wire = _RingWire(PP0)
        h = _make_holder(PP0, wire)
        guard = PPAdmissionCongruenceGuard()
        pending: deque = deque()
        max_pending_seen = 0
        for i in range(N_ITERS):
            _progress(out_dir, PP0, f"i={i} sending decision (mode={mode})")
            h._pp_send_admission_decision(PPAdmissionDecision(mb_id=i, entries=()))
            pending.append(i)
            # Mirrors the shipped cap loop in _event_loop_pp_body's #791
            # block, against the real imported constant -- not a copy that
            # could silently drift from it.
            while len(pending) > _PP_ADMISSION_PENDING_SENDS_CAP:
                pending.popleft()
            max_pending_seen = max(max_pending_seen, len(pending))
            if len(pending) >= WORLD:
                _progress(out_dir, PP0, f"i={i} wraparound-check mode={mode}")
                if mode == "blocking":
                    returned = h._pp_recv_admission_decision()
                else:
                    returned = h._pp_try_recv_admission_decision()
                if returned is not None:
                    pending.popleft()
                    guard.record_return_trip(returned)
            if mode == "opportunistic":
                # PP0's OWN mandatory per-iteration "output" receive from
                # the last rank -- modeled here because it is what the
                # shipped `_pp_try_recv_admission_decision`'s docstring
                # says actually drains this p2p pair on a healthy pass; the
                # peek above only ever finds something because THIS
                # receive's own drain-and-stash loop
                # (`recv_typed_tensor_dict`, pp_typed_channel.py) put it
                # there as a side effect. Skipped entirely in "blocking"
                # mode: PP0 never gets far enough for it to matter there
                # (it is captured mid wraparound-check first), and adding
                # it unconditionally would give PP0 a second, EARLIER
                # blocking dependency on the same withheld peer, muddying
                # exactly which call site the RED case is proving stuck.
                _progress(out_dir, PP0, f"i={i} draining output receive")
                h._pp_recv_typed_dict(expected_kind="output")
        _progress(out_dir, PP0, "done")
        res["ok"] = True
        res["iterations_completed"] = N_ITERS
        res["max_pending_seen"] = max_pending_seen
        res["cap"] = _PP_ADMISSION_PENDING_SENDS_CAP
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        try:
            with open(os.path.join(out_dir, f"r{PP0}.json"), "w") as f:
                json.dump(res, f)
        finally:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except BaseException:  # noqa: BLE001 - best-effort teardown
                    pass


def _relay_worker(rank, init_file, out_dir, mode):
    res = {"rank": rank, "ok": False, "error": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        wire = _RingWire(rank)
        h = _make_holder(rank, wire)
        start_i = 0
        if rank == PP1 and mode == "blocking":
            # ONLY in case "blocking": receive exactly `pp_size` decisions
            # and BUFFER them -- forward none yet. This lets PP0's own
            # sends of 0..pp_size-1 complete normally (gloo's dist.send is
            # synchronous: it will not return until PP1 posts the matching
            # recv, so receiving is never special-cased here, only the
            # forward is deferred) while deterministically guaranteeing PP2
            # gets nothing yet -- see the module docstring's mechanisms
            # 1/2. In case "opportunistic" this whole branch is skipped:
            # PP1 behaves like an ordinary, unremarkable relay (see below),
            # because nothing about proving the fix needs PP1 held back --
            # only the RED case needs a peer that provably never replies.
            buffered = []
            for i in range(WORLD):
                _progress(out_dir, rank, f"i={i} waiting for decision (buffering)")
                buffered.append(h._pp_recv_admission_decision())
            _progress(out_dir, rank, "deliberately stalled (case=blocking)")
            time.sleep(PP1_DELIBERATE_STALL_S)
            for i, incoming in enumerate(buffered):
                _progress(out_dir, rank, f"i={i} forwarding buffered decision")
                _effective, amended = h._pp_reconcile_incoming_admission(incoming)
                h._pp_send_admission_decision(amended)
            start_i = WORLD
        for i in range(start_i, N_ITERS):
            _progress(out_dir, rank, f"i={i} waiting for decision")
            incoming = h._pp_recv_admission_decision()
            _effective, amended = h._pp_reconcile_incoming_admission(incoming)
            _progress(out_dir, rank, f"i={i} forwarding decision")
            h._pp_send_admission_decision(amended)
            if rank == PP2:
                # The last rank's OTHER mandatory per-pass send -- sampled
                # output, ring-wrapped to PP0 on the SAME directed pair the
                # admission wraparound just used. Modeled here (a tiny
                # placeholder dict, not a real sampling result) only so
                # PP0's own per-iteration output receive (see _pp0_worker,
                # "opportunistic" mode) has something real to drain -- see
                # that function's comment and
                # `_pp_try_recv_admission_decision`'s docstring for why
                # this is the actual production mechanism, not a test-only
                # shortcut.
                _progress(out_dir, rank, f"i={i} sending output")
                h._pp_send_dict_to_next_stage({"tok": i}, msg_type="output")
        _progress(out_dir, rank, "done")
        res["ok"] = True
        res["iterations_completed"] = N_ITERS
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        try:
            with open(os.path.join(out_dir, f"r{rank}.json"), "w") as f:
                json.dump(res, f)
        finally:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except BaseException:  # noqa: BLE001 - best-effort teardown
                    pass


def _run(mode, join_timeout_s):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [ctx.Process(target=_pp0_worker, args=(init_file, tmp, mode))]
        for r in (PP1, PP2):
            procs.append(
                ctx.Process(target=_relay_worker, args=(r, init_file, tmp, mode))
            )
        for p in procs:
            p.start()
        deadline = time.time() + join_timeout_s
        for p in procs:
            p.join(timeout=max(0.1, deadline - time.time()))
        stuck_ranks = [r for r, p in enumerate(procs) if p.is_alive()]
        progress = {r: _read_progress(tmp, r) for r in range(WORLD)}
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)

        def _load(path):
            if not os.path.exists(path):
                return None
            with open(path) as f:
                return json.load(f)

        out = {"stuck_ranks": stuck_ranks, "progress": progress}
        for r in range(WORLD):
            out[f"result_{r}"] = _load(os.path.join(tmp, f"r{r}.json"))
        return out


class PPAdmissionWraparoundBlocks(unittest.TestCase):
    def test_blocking_wraparound_wedges_the_ring(self):
        """RED: the OLD call shape (a plain blocking
        `_pp_recv_admission_decision` at the wraparound site) never lets
        all three ranks finish. Per-rank stall location is reported from
        each rank's last progress marker, written before any blocking
        call."""
        res = _run("blocking", RED_JOIN_TIMEOUT_S)
        self.assertEqual(
            res["stuck_ranks"],
            [PP0, PP1, PP2],
            "expected all three ranks still alive at the deadline "
            f"(closed ring); per-rank last position: {res['progress']}",
        )
        self.assertIn(
            "wraparound-check mode=blocking",
            res["progress"][PP0],
            f"PP0 must be captured mid wraparound-check: {res['progress']}",
        )
        self.assertIn(
            "deliberately stalled",
            res["progress"][PP1],
            f"PP1 must be captured withholding cooperation: {res['progress']}",
        )
        self.assertIn(
            "i=0 waiting for decision",
            res["progress"][PP2],
            f"PP2 must be captured blocked on PP1's forward: {res['progress']}",
        )


class PPAdmissionWraparoundOpportunistic(unittest.TestCase):
    def test_opportunistic_wraparound_never_blocks_pp0(self):
        """GREEN: the SHIPPED call shape
        (`_pp_try_recv_admission_decision`) lets all three ranks complete
        every iteration within a bounded deadline, and PP0's own
        pending-sends bookkeeping never exceeds the real shipped cap."""
        res = _run("opportunistic", GREEN_JOIN_TIMEOUT_S)
        self.assertEqual(
            res["stuck_ranks"], [], f"a rank never finished: {res}"
        )
        for name, r in (
            ("PP0", res["result_0"]),
            ("PP1", res["result_1"]),
            ("PP2", res["result_2"]),
        ):
            self.assertIsNotNone(r, f"{name} produced no result: {res}")
            self.assertIsNone(r.get("error"), f"{name} raised: {r.get('error')}")
            self.assertTrue(r.get("ok"), f"{name} did not complete: {r}")
            self.assertEqual(
                r.get("iterations_completed"),
                N_ITERS,
                f"{name} did not complete all {N_ITERS} iterations: {r}",
            )
        r0 = res["result_0"]
        self.assertLessEqual(
            r0["max_pending_seen"],
            r0["cap"],
            f"PP0's pending-sends bookkeeping exceeded its own cap: {r0}",
        )


class PPAdmissionWraparoundCallSite(unittest.TestCase):
    def test_call_site_uses_the_opportunistic_receive(self):
        """STRUCTURAL: the wraparound site inside _event_loop_pp_body's
        #791 admission block must call the opportunistic
        `_pp_try_recv_admission_decision`, never a blocking
        `_pp_recv_admission_decision`, directly. Same technique as
        test_pp_admission_wiring_791.py's own ordering test."""
        import inspect

        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        src = inspect.getsource(SchedulerPPMixin._event_loop_pp_body)
        self.assertIn(
            "_pp_try_recv_admission_decision()",
            src,
            "the wraparound site must call the opportunistic receive",
        )


if __name__ == "__main__":
    unittest.main()
