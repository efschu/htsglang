"""#973: the PP ring commit wait must be BOUNDED and LOUD, never silent.

THE DEFECT. ``scheduler_pp_mixin._pp_commit_comm_work`` flushed this rank's
outstanding p2p sends with a naked ``p2p_work.work.wait()`` -- no deadline, on
a gloo ``cpu_group`` whose own timeout is two hours. Boot 2 of window-flip-0828
(commit 8c27b5f0c2) died exactly there: PP0 and PP1 both blocked in gloo
``UnboundBuffer::waitSend`` under this function (PP1 arriving via
``_pp_commit_pending_req_work``) while PP2 sat in ``_do_recv`` -- a closed
three-arc cycle that stayed SILENT for 10+ minutes until the boot was killed by
hand. Specimen: ``/spinning/evidence-665-f1/SPECIMEN-2026-08-28T1135Z-flip0828-
boot2-RING-WAIT-WEDGE.txt``.

WHAT IS ASSERTED HERE, on real 3-process gloo (no CUDA, no serving):

  Arm 1a  the PRE-#973 shape (a verbatim copy of the naked loop, bound onto
          the holder) HANGS on an unpaired send -- and the test can TELL a
          hang from a raise, because a hung child writes no result and its
          progress marker is still parked at "committing".
  Arm 1b  the SHIPPED ``_pp_commit_comm_work`` raises ``RingCommitTimeout``
          on the same wire, inside its budget, carrying the peer statement.
  Arm 2   healthy paired traffic completes with no timeout and the handle
          list still cleared -- zero behaviour change.
  Arm 3   the shipped code with the bound NEUTERED via the documented escape
          hatch (``SGLANG_PP_RING_COMMIT_BUDGET_S=0``) HANGS again. That is
          the can-fail proof: Arm 1b passes because of the bound, not because
          of the harness.

MEASURED PREMISE OF THE UNPAIRED SHAPE (probe, 2026-08-28, 2-process gloo,
CVD=""): an isend whose peer posts no matching irecv blocks in ``wait()`` at
EVERY payload size tried -- 1, 1024, 262144, 8388608 and 67108864 float32
elements all returned ``completed=False`` against a 3 s parked bound. gloo does
not buffer it away, so a 1-element tensor reproduces the wedge and the test
needs no large-payload trickery.

SPAWN TRAP (the reason every holder is built inside the child): the workers run
under ``multiprocessing`` "spawn", which re-imports this module fresh in each
child. Anything bound in the parent's test method never reaches a child.
"""

import json
import os
import sys
import tempfile
import time
import types
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120)

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2

#: Bound used by the arms under test. Far below the shipped 120 s default so
#: the suite stays quick; the DEFAULT's rationale lives on
#: DEFAULT_RING_COMMIT_BUDGET_S in scheduler_pp_mixin.py.
BUDGET_S = 3.0

#: How long the peer ranks stay alive in the unpaired arms. MUST exceed the
#: parent's RED join timeout: if a peer exited first, its socket would close
#: and the blocked sender would get a TRANSPORT failure ("Connection closed by
#: peer") instead of staying wedged -- which would quietly turn the red arms
#: into a different test.
PEER_ALIVE_S = 12.0

#: Generous, for arms expected to finish.
GREEN_JOIN_TIMEOUT_S = 30.0
#: Short, for arms that wedge BY CONSTRUCTION -- they have no timing window to
#: lose, and the whole point is that they never finish.
RED_JOIN_TIMEOUT_S = 8.0


def _progress(out_dir, rank, msg):
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


def _write_result(out_dir, rank, res):
    path = os.path.join(out_dir, f"result_r{rank}.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(res, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _naked_commit(self, work):
    """VERBATIM the pre-#973 body of ``_pp_commit_comm_work``.

    Kept here rather than obtained by reverting the tree, so the red arm can
    run against the shipped tree with no checkout gymnastics -- and so the
    thing this fix replaced stays visible next to what replaced it.
    """
    for p2p_work in work:
        p2p_work.work.wait()
    work.clear()


def _build_holder(rank, out_dir):
    """A bare namespace carrying the SHIPPED methods under test.

    Only the attributes ``_pp_commit_comm_work`` actually touches are
    supplied, which is also a check that the new code does not quietly grow a
    dependency on the whole Scheduler.
    """
    from sglang.srt.managers.phase_flip_counters import CHAN_REQ, PhaseFlipCounters
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    counters = PhaseFlipCounters(
        n_ranks=WORLD,
        rank=rank,
        directory=os.path.join(out_dir, "counters"),
        instance="t973",
    )
    holder = types.SimpleNamespace(
        pp_flip_counters=counters,
        pp_rank=rank,
        pp_size=WORLD,
        _pp_blocked_recv_arm=None,
    )
    holder._pp_flip_ring = lambda: (rank, WORLD)
    # DELIBERATELY the MINIMUM set, and the minimum is the point: a holder
    # binds one method at a time, so if `_pp_commit_comm_work` ever resolves a
    # helper name off `self` again, this arm goes red exactly as
    # test_pp_admission_send_handle_dropped_796 did. The #973 helpers are
    # module-level functions precisely so they need no entry here.
    for name in (
        "_pp_commit_comm_work",
        "_pp_flip_upstream",
        "_pp_flip_downstream",
    ):
        setattr(holder, name, types.MethodType(getattr(SchedulerPPMixin, name), holder))
    # This rank has posted one CHAN_REQ message that its downstream has not
    # consumed -- the exact imbalance the peer statement must name.
    counters.bump_sent(CHAN_REQ)
    return holder, CHAN_REQ


def _worker(rank, init_file, out_dir, variant):
    res = {"rank": rank, "ok": False, "error": None, "raised": None, "elapsed": None}
    try:
        if variant == "escape_hatch":
            os.environ["SGLANG_PP_RING_COMMIT_BUDGET_S"] = "0"
        else:
            os.environ["SGLANG_PP_RING_COMMIT_BUDGET_S"] = str(BUDGET_S)

        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        from sglang.srt.distributed.parallel_state import P2PWork

        _progress(out_dir, rank, "initialised")

        if rank == PP0:
            holder, _chan = _build_holder(rank, out_dir)
            payload = torch.ones(1024, dtype=torch.float32)
            work = dist.isend(payload, dst=PP1, tag=973)
            handles = [P2PWork(work=work, payload=payload)]
            # Identity is how `_pp_commit_channel_of` names the wire, so the
            # list must BE the attribute, not merely equal it.
            holder.send_req_work = handles

            _progress(out_dir, rank, "committing")
            started = time.monotonic()
            try:
                if variant == "naked":
                    _naked_commit(holder, handles)
                else:
                    holder._pp_commit_comm_work(handles)
                res["raised"] = None
                res["cleared"] = handles == []
            except BaseException as exc:  # noqa: BLE001 - the raise IS the result
                res["raised"] = type(exc).__name__
                res["message"] = str(exc)[:1500]
            res["elapsed"] = time.monotonic() - started
            _progress(out_dir, rank, "returned")

        elif rank == PP1:
            if variant == "healthy":
                buf = torch.zeros(1024, dtype=torch.float32)
                dist.recv(buf, src=PP0, tag=973)
                res["received_sum"] = float(buf.sum().item())
            else:
                # Deliberately NO matching irecv: this is the unpaired shape.
                # Staying alive is load-bearing -- see PEER_ALIVE_S.
                _progress(out_dir, rank, "withholding_recv")
                time.sleep(PEER_ALIVE_S)
        else:
            _progress(out_dir, rank, "bystander")
            if variant != "healthy":
                time.sleep(PEER_ALIVE_S)

        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        _write_result(out_dir, rank, res)
        # HARD EXIT, deliberately. In the bounded-timeout arm a ParkedWait
        # daemon thread is STILL inside gloo's blocking wait by design (that
        # is what keeps the pair intact), so an orderly destroy_process_group
        # can block here. The result file is already fsynced, and this suite
        # parses results rather than exit codes, so nothing downstream reads
        # the status this discards.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _run(variant, join_timeout):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [
            ctx.Process(target=_worker, args=(r, init_file, tmp, variant))
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        deadline = time.time() + join_timeout
        for p in procs:
            p.join(timeout=max(0.1, deadline - time.time()))
        stuck = [r for r, p in enumerate(procs) if p.is_alive()]
        out = {
            "stuck_ranks": stuck,
            "stall_report": {r: _read_progress(tmp, r) for r in stuck},
        }
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        for r in range(WORLD):
            path = os.path.join(tmp, f"result_r{r}.json")
            if os.path.exists(path):
                with open(path) as f:
                    out[f"result_{r}"] = json.load(f)
            else:
                out[f"result_{r}"] = None
        return out


class RingCommitBounded973(unittest.TestCase):
    def setUp(self):
        if not dist.is_gloo_available():  # pragma: no cover - env guard
            self.skipTest("gloo backend unavailable")

    def _assert_wedged_in_commit(self, res):
        """PP0 is STILL WAITING in the commit -- not raised, not finished.

        The peers are expected to be alive too and that is not a wedge: they
        sleep PEER_ALIVE_S (12 s) by construction, which is longer than the
        RED window (8 s), precisely so the blocked sender cannot be rescued by
        a peer's socket closing. So the arm asserts each rank's stall REASON
        rather than a bare count -- a peer parked anywhere other than its own
        designed sleep would mean the arm reproduced some other failure.
        """
        self.assertIn(
            PP0,
            res["stuck_ranks"],
            f"PP0 was expected to still be wedged in the commit: {res}",
        )
        # The "still waiting" vs "raised" discriminator this arm turns on: a
        # rank that raised would have fsynced its result and advanced its
        # progress marker to "returned".
        self.assertIsNone(
            res["result_0"],
            f"PP0 produced a result, so it did NOT hang -- arm is invalid: {res}",
        )
        self.assertEqual(
            res["stall_report"].get(PP0),
            "committing",
            f"PP0 wedged somewhere other than the commit: {res}",
        )
        for peer, expected in ((PP1, "withholding_recv"), (PP2, "bystander")):
            if peer in res["stuck_ranks"]:
                self.assertEqual(
                    res["stall_report"].get(peer),
                    expected,
                    f"rank {peer} stalled somewhere other than its designed "
                    f"sleep, so this arm is not the shape it claims: {res}",
                )

    def test_arm1a_pre_973_naked_commit_hangs_on_an_unpaired_send(self):
        """RED: the shape that shipped. It must be seen HANGING, not raising."""
        res = _run("naked", RED_JOIN_TIMEOUT_S)
        self._assert_wedged_in_commit(res)

    def test_arm1b_bounded_commit_raises_ring_commit_timeout_with_peer_statement(self):
        """GREEN: same wire, same unpaired shape, now loud and inside budget."""
        res = _run("bounded", GREEN_JOIN_TIMEOUT_S)
        self.assertEqual(res["stuck_ranks"], [], f"nothing should wedge now: {res}")
        r0 = res["result_0"]
        self.assertIsNotNone(r0, f"PP0 produced no result: {res}")
        self.assertEqual(
            r0.get("raised"),
            "RingCommitTimeout",
            f"expected a named RingCommitTimeout, got {r0}",
        )
        msg = r0.get("message", "")
        self.assertIn("#973 RING COMMIT TIMEOUT", msg)
        self.assertIn("Peer statement:", msg)
        # The wire is named by identity against send_req_work.
        self.assertIn("req/send_req_work", msg)
        # The peer statement must name the SILENT HOP, not merely that a
        # deadline passed.
        self.assertIn("downstream rank 1", msg)
        self.assertIn("has NOT taken this send off the wire", msg)
        self.assertIn("SGLANG_PP_RING_COMMIT_BUDGET_S", msg)
        # Inside its budget, with room for process/gloo latency -- and NOT
        # instant, which would mean something other than the bound fired.
        self.assertGreaterEqual(r0["elapsed"], BUDGET_S * 0.5, f"{r0}")
        self.assertLess(r0["elapsed"], BUDGET_S + 10.0, f"{r0}")

    def test_arm2_healthy_paired_traffic_is_unchanged_and_never_times_out(self):
        """Zero behaviour change on the path a healthy boot actually takes."""
        res = _run("healthy", GREEN_JOIN_TIMEOUT_S)
        self.assertEqual(res["stuck_ranks"], [], f"healthy traffic wedged: {res}")
        for r in range(WORLD):
            result = res[f"result_{r}"]
            self.assertIsNotNone(result, f"rank {r} produced no result: {res}")
            self.assertTrue(result.get("ok"), f"rank {r} failed: {result.get('error')}")
        r0 = res["result_0"]
        self.assertIsNone(r0.get("raised"), f"a healthy commit raised: {r0}")
        self.assertTrue(r0.get("cleared"), f"the handle list was not cleared: {r0}")
        # The receiver got the real payload, so the commit committed something.
        self.assertEqual(res["result_1"].get("received_sum"), 1024.0, f"{res}")
        # A healthy commit is nowhere near the bound; this also pins the
        # ParkedWait thread hop as cheap on the hot path.
        self.assertLess(r0["elapsed"], BUDGET_S, f"healthy commit was slow: {r0}")

    def test_the_pending_req_commit_path_reaches_the_bound(self):
        """REACHABILITY of the OTHER arc of boot 2's cycle.

        PP0 wedged in ``_pp_commit_comm_work`` directly; PP1 got there through
        ``_pp_commit_pending_req_work`` (:3270 in the specimen's stack). A
        bound is only proven for a defect path once that path is shown to
        REACH it, so this drives the real ``_pp_commit_pending_req_work`` and
        asserts the #973 error comes out of it -- rather than reading the one
        delegating line and calling it covered.

        In-process and transport-free on purpose: the wire is already proven
        by the gloo arms above, and what is in question here is only the call
        graph.
        """
        import types as _types

        from sglang.srt.managers.scheduler_pp_mixin import (
            RingCommitTimeout,
            SchedulerPPMixin,
        )

        class _NeverCompletingWork:
            def wait(self, *args, **kwargs):
                time.sleep(3600)

        class _Handle:
            def __init__(self):
                self.work = _NeverCompletingWork()

        os.environ["SGLANG_PP_RING_COMMIT_BUDGET_S"] = "1.0"
        try:
            holder = _types.SimpleNamespace(
                pp_flip_counters=None, pp_rank=1, pp_size=WORLD
            )
            holder.send_req_work = [_Handle()]
            for name in (
                "_pp_commit_comm_work",
                "_pp_commit_pending_req_work",
            ):
                setattr(
                    holder,
                    name,
                    _types.MethodType(getattr(SchedulerPPMixin, name), holder),
                )
            started = time.monotonic()
            with self.assertRaises(RingCommitTimeout) as caught:
                holder._pp_commit_pending_req_work()
            elapsed = time.monotonic() - started
        finally:
            os.environ.pop("SGLANG_PP_RING_COMMIT_BUDGET_S", None)

        msg = str(caught.exception)
        self.assertIn("#973 RING COMMIT TIMEOUT", msg)
        self.assertIn("req/send_req_work", msg)
        # Counters are absent on a non-flip boot; the statement must SAY so
        # rather than silently omitting the peer half.
        self.assertIn("no phase-flip counters on this boot", msg)
        self.assertLess(elapsed, 30.0, "the bound did not fire promptly")

    def test_arm3_neutering_the_bound_restores_the_hang(self):
        """CAN-FAIL PROOF: with the bound off, arm 1b's shape wedges again."""
        res = _run("escape_hatch", RED_JOIN_TIMEOUT_S)
        self._assert_wedged_in_commit(res)


if __name__ == "__main__":
    unittest.main()
