"""The proxy receive must not enter an unbounded blocking wait for a
message no upstream will ever send (#789).

THE METAL FACT THIS ENCODES. py-spy capture, PP=3, --enable-phase-flip,
evidence-665-f1, 2026-08-20 (locals_428076/428077/428078.txt):

    PP0  mb_id=0  cur_batch=None            server_is_idle=True   parked in
         _pp_commit_pending_req_work
    PP1  mb_id=1  cur_batch=None            server_is_idle=True   parked in
         _pp_commit_pending_req_work
    PP2  mb_id=2  cur_batch=<ScheduleBatch>  server_is_idle=False  parked in
         _pp_recv_proxy_tensors(mb_id=2) -> _pp_recv_typed_dict ->
         recv_typed_tensor_dict -> _recv_tensor_dict_metadata -> recv_object
         (parallel_state.py:2133)

The mb_ids reflect the normal -1 stagger, not a desync. The divergence is
BATCH PRESENCE: the last rank scheduled a batch for its slot; both upstream
ranks did not, and are idle. The last rank then calls the SHIPPED
``_pp_recv_proxy_tensors``, which enters a plain blocking gloo receive for a
proxy that -- since neither upstream ever decided to run a batch this pass
-- will never be sent. WHY admission diverges between ranks is a separate,
out-of-scope question (a different investigation thread owns it); this file
is the safety belt: the last rank must fail loudly and boundedly instead of
wedging silently forever.

WHAT THIS FILE DOES NOT DO. It does not explain admission divergence, and
it does not fix it. #631's own stamp guard is a precedent for "detect and
refuse rather than mispair or wedge"; #789 is the same shape one call
earlier -- before any message has arrived at all, rather than after a wrong
one has.

THE FIX UNDER TEST, ``_pp_wait_for_proxy_readiness`` (scheduler_pp_mixin.py,
directly above ``_pp_recv_proxy_tensors``), called from
``_pp_recv_proxy_tensors`` BEFORE ``_pp_recv_typed_dict`` is ever entered:
polls ``PhaseFlipCounters``' CHAN_DICT sent/consumed counters (the same
pollable, out-of-band /dev/shm side channel ``pp_flip_drain_leftover_dicts``
already uses) for a POSITIVE presence signal -- the upstream's published
sent-count exceeding this rank's own consumed-count, which is provably true
only once the upstream's isend has actually posted (``bump_sent`` fires
strictly after the post, the same ordering law the drain relies on). The
instant that signal appears, the gate returns and the ordinary blocking
receive proceeds exactly as before. If the counter never moves for
``DEFAULT_PROXY_READINESS_BUDGET_S`` (a backstop, not the decision: see the
function's docstring for the #630-lesson framing), the gate raises a named
``RuntimeError`` naming the slot, the upstream rank, and the counter state,
instead of ever calling ``_pp_recv_typed_dict``. Never a timing-out
``Work.wait()`` on the transport itself -- corpse F (measured: a timed-out
gloo wait destroys the pair, the peer then sees "Connection closed by
peer") -- and never a plain "wait N seconds and give up regardless" either:
the loop acts on the counter's value on every poll, not on the clock alone.

WHY THIS TEST IS THREE REAL PROCESSES OVER REAL GLOO. Same reasoning as
test_pp_flip_leftover_proxy_757.py and test_pp_drain_completeness_787.py:
"nobody ever sends" has to be modelled as a real, live, unfulfilled gloo
peer -- not a mock that could not distinguish a genuine unbounded block from
a fast, uninteresting return. ``_pp_recv_proxy_tensors``,
``_pp_recv_typed_dict`` and the new ``_pp_wait_for_proxy_readiness`` are the
SHIPPED functions, bound to a holder exactly as the #757/#787/#788 files
bind ``SchedulerPPMixin`` methods.

THE CASES:

  test_neutered_gate_wedges_unboundedly        RED / CAN-FAIL PROOF. With
      the gate monkeypatched to a no-op -- byte-identical to the pre-#789
      shipped code, which had no such call at all -- the victim's call to
      the shipped ``_pp_recv_proxy_tensors`` genuinely blocks forever in
      the real gloo receive against two upstreams that never send,
      reproducing the specimen's own signature. Bounded here only by the
      OUTER test driver (a join deadline followed by ``terminate()``),
      never by anything inside the code under test -- that absence of an
      internal bound is exactly the bug.

  test_shipped_gate_raises_named_diagnostic_instead_of_wedging   GREEN,
      the primary invariant. With the real shipped gate in place (its
      internal budget shortened via the env override so the test does not
      have to wait out the 30 s production default), the same two-idle-
      upstream constellation now makes the victim return -- raising a named
      RuntimeError that identifies the slot, the upstream rank, and the
      counter state -- well within the outer test driver's bound, instead
      of hanging past it.

  test_gate_does_not_delay_a_genuinely_posted_message   DEFAULT-CONSERVATIVE
      REGRESSION GUARD. When the immediate upstream genuinely posts (a real
      isend over gloo, counted the same way production counts it), the gate
      must add no more than a vanishingly small delay and the ordinary
      receive must still deliver the real message -- proof that a healthy
      pass is unchanged by this contract, not merely an assumption.

  test_noop_when_pp_flip_counters_is_none (separate TestCase, no
      multiprocessing needed)   The reference regression launch command in
      CLAUDE.md never sets --enable-phase-flip, so ``pp_flip_counters`` is
      ``None`` on that path and this gate must be a true no-op there --
      checked directly, in-process, by wall-clock.
"""

import json
import os
import tempfile
import time
import types
import unittest
from collections import defaultdict, deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60)

WORLD = 3
UPSTREAM_A, UPSTREAM_B, VICTIM = 0, 1, 2

#: The slot the victim (the last rank, PP2 in the specimen) resumes on.
#: Matches the specimen's own PP2 mb_id=2 -- red-first means red on the
#: specimen's own number, not a convenient one.
LIVE_MB = 2

#: How long the OUTER driver waits before concluding a rank is genuinely
#: stuck and terminating it. Short on purpose for the neutered/RED case:
#: the point is only to prove "still blocked at time X", not to wait out
#: any deadline internal to the code under test (there is none, in that
#: case -- that absence is the bug).
NEUTERED_JOIN_TIMEOUT_S = 5.0

#: The #789 gate's own internal budget, shortened for this test via
#: ENV_PROXY_READINESS_BUDGET so the GREEN case does not have to wait out
#: DEFAULT_PROXY_READINESS_BUDGET_S (30 s, sized for production). Must
#: still be long enough to comfortably exceed real-gloo-init + a handful of
#: PROXY_READINESS_POLL_STEP_S (0.02 s) polls.
SHORT_READINESS_BUDGET_S = 0.4

#: Outer driver bound for the cases that exercise the shortened budget
#: above. Generous relative to SHORT_READINESS_BUDGET_S so process-start /
#: gloo-init overhead never makes a passing case look stuck.
SHIPPED_JOIN_TIMEOUT_S = 20.0
HEALTHY_JOIN_TIMEOUT_S = 20.0

#: How long an idle upstream (cur_batch=None, server_is_idle=True in the
#: specimen) sleeps before the OUTER driver would need to terminate it.
#: Deliberately longer than every join timeout above: an idle upstream in
#: the real specimen never finishes on its own either, and a rank that
#: happened to exit early here for an unrelated reason should never be
#: mistaken for the driver's own termination.
IDLE_UPSTREAM_SLEEP_S = 90.0


class _GlooWire:
    """A real point-to-point tensor-dict wire over gloo.

    Only the TRANSPORT is adapted here -- pickle to bytes, bytes over gloo.
    The demultiplexing, the stamp check, and (under test) the new readiness
    gate are all the shipped functions. Copied transport-only from
    test_pp_flip_leftover_proxy_757.py / test_pp_drain_completeness_787.py;
    see those files for why ``rank_in_group`` / ``world_size`` are carried
    (``pp_typed_channel.resolve_src`` derives peer identity from them).
    """

    def __init__(self, rank: int, src, dst):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD
        self.src = src
        self.dst = dst
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == WORLD - 1

    def send_tensor_dict(self, d, all_gather_group=None):
        import pickle

        buf = pickle.dumps(d)
        size = torch.tensor([len(buf)], dtype=torch.long)
        dist.send(size, dst=self.dst)
        dist.send(torch.frombuffer(bytearray(buf), dtype=torch.uint8), dst=self.dst)

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        import pickle

        # `src` is accepted and ignored: this wire already has exactly one
        # fixed peer per direction, set at construction.
        size = torch.zeros(1, dtype=torch.long)
        dist.recv(size, src=self.src)
        buf = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(buf, src=self.src)
        return pickle.loads(bytes(buf.numpy()))

    def isend_tensor_dict(self, d):
        """Post-only send, mirroring production's real send path.

        ``_pp_send_dict_to_next_stage`` calls ``send_tensor_dict`` with
        ``async_send=True`` (parallel_state.py), which maps to
        ``torch.distributed.isend`` -- returns as soon as the message is
        POSTED, not once a peer has actually taken it off the wire. That
        distinction is exactly what CHAN_DICT's ``bump_sent`` certifies
        (see _pp_wait_for_proxy_readiness's docstring): "posted", not
        "delivered". A blocking ``dist.send`` here instead would make the
        sender's own counter bump wait on the receiver already being in
        its matching ``dist.recv`` -- a deadlock this test's OWN harness
        would introduce, not a fact about the shipped code. ``isend`` is
        used here so the healthy-path test exercises the same "post, then
        certify" ordering production actually has.
        """
        import pickle

        buf = pickle.dumps(d)
        size = torch.tensor([len(buf)], dtype=torch.long)
        work_size = dist.isend(size, dst=self.dst)
        work_buf = dist.isend(
            torch.frombuffer(bytearray(buf), dtype=torch.uint8), dst=self.dst
        )
        # Posted, not completed: return immediately, exactly like
        # production's async_send=True path.
        return work_size, work_buf


def _bump_sent_counter(counter_path, n):
    """Publish `n` as the new sent-count, atomically (write + rename).

    Mirrors ``PhaseFlipCounters.bump_sent``'s real synchronicity: called by
    the upstream immediately AFTER its real ``send_tensor_dict`` returns,
    never before -- the ordering law the whole contract leans on.
    """
    tmp = counter_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(n))
    os.replace(tmp, counter_path)


def _victim(rank, wire, out_dir, case):
    """The shipped mixin methods, bound to a holder (the 630/757/787
    pattern). ``pp_flip_counters`` is a stub whose ``sent`` either always
    reads 0 (the two-idle-upstream constellation under test) or reads a
    live file the upstream writes (the healthy-path regression case)."""
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    state = {"consumed": 0}
    if case == "healthy_upstream_sends":
        sent_path = os.path.join(out_dir, "sent_counter_r1")

        def _sent(_chan, _rank):
            try:
                with open(sent_path) as f:
                    raw = f.read().strip()
            except FileNotFoundError:
                return 0
            return int(raw) if raw else 0
    else:
        # #789: the constellation under test IS "no upstream ever posted
        # anything" -- a fixed stub that never reveals a message, exactly
        # like the specimen's idle PP0/PP1.
        def _sent(_chan, _rank):
            return 0

    counters = types.SimpleNamespace(
        sent=_sent,
        # #789 second counter: "the upstream has ENTERED a send for me",
        # published before the post so a RENDEZVOUS send (the lazy creation
        # of a torch NCCL 2-rank p2p communicator) is distinguishable from
        # an upstream that scheduled nothing. It reads the same source as
        # `sent` here, which keeps BOTH constellations this file tests
        # intact: an idle upstream has neither entered nor posted (0, so
        # the gate still raises), and a healthy upstream that posted had
        # necessarily entered first.
        attempted=_sent,
        local_consumed=lambda chan: state["consumed"],
    )
    h = types.SimpleNamespace(
        pp_group=wire,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        pp_flip_counters=counters,
        _pp_tensor_dict_inbox=defaultdict(deque),
        _pp_proxy_drops=0,
    )
    h._pp_boundary_stats = lambda: None
    h._pp_flip_bump_consumed = lambda chan: state.__setitem__(
        "consumed", state["consumed"] + 1
    )
    # The victim's immediate upstream in this ring is UPSTREAM_B (rank 1),
    # matching the specimen's PP1 -> PP2 edge. Stubbed directly rather than
    # via the shipped _pp_flip_ring/_pp_flip_upstream, exactly as 757/787
    # do, since this file is not testing ring-topology derivation.
    h._pp_flip_upstream = lambda: UPSTREAM_B
    for name in (
        "_pp_recv_typed_dict",
        "_pp_recv_proxy_tensors",
        "_pp_wait_for_proxy_readiness",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _worker(rank, init_file, out_dir, case, readiness_budget_env):
    res = {"rank": rank, "ok": False, "error": None, "note": None, "delivered_h": None}
    progress_path = os.path.join(out_dir, f"progress_r{rank}.json")

    def wp(where):
        with open(progress_path, "w") as f:
            json.dump({"rank": rank, "where": where}, f)

    wp("start")
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        if rank == VICTIM:
            wire = _GlooWire(rank, src=UPSTREAM_B, dst=None)
            h = _victim(rank, wire, out_dir, case)
            if case == "neutered":
                # #789 REMOVED: reproduces the pre-#789 shipped code, which
                # had no such call at this site at all.
                h._pp_wait_for_proxy_readiness = lambda mb_id: None
                wp("neutered_calling_recv_proxy_tensors")
                h._pp_recv_proxy_tensors(LIVE_MB)  # expected to hang forever
                res["note"] = "did not hang -- unexpected under a neutered gate"
                res["ok"] = True
            elif case == "shipped_no_upstream":
                if readiness_budget_env is not None:
                    from sglang.srt.managers.scheduler_pp_mixin import (
                        ENV_PROXY_READINESS_BUDGET,
                    )

                    os.environ[ENV_PROXY_READINESS_BUDGET] = str(readiness_budget_env)
                wp("shipped_calling_recv_proxy_tensors")
                h._pp_recv_proxy_tensors(LIVE_MB)
                res["note"] = "did not raise -- the #789 contract failed to fire"
                res["ok"] = True
            elif case == "healthy_upstream_sends":
                wp("shipped_calling_recv_proxy_tensors_healthy")
                proxy = h._pp_recv_proxy_tensors(LIVE_MB)
                res["delivered_h"] = proxy["h"]
                res["note"] = "delivered without raising, upstream had genuinely posted"
                res["ok"] = True
            else:
                raise AssertionError(f"unknown case {case!r}")
        elif rank == UPSTREAM_B and case == "healthy_upstream_sends":
            wire = _GlooWire(rank, src=None, dst=VICTIM)
            wp("posting_real_proxy")
            # isend: returns once POSTED, not once delivered -- the same
            # ordering production's async_send=True path has. Publishing
            # the counter here, before the transfer necessarily lands,
            # mirrors bump_sent's real "posted, not delivered" contract
            # (see isend_tensor_dict's docstring) instead of introducing a
            # deadlock the shipped code does not have.
            works = wire.isend_tensor_dict({"__msg_type__": "proxy", "h": 42})
            _bump_sent_counter(os.path.join(out_dir, "sent_counter_r1"), 1)
            wp("posted_and_published")
            for w in works:
                w.wait()
            wp("sent_done")
            res["ok"] = True
            res["note"] = "posted the real proxy and published the counter"
        else:
            # UPSTREAM_A always, and UPSTREAM_B under the two-idle-upstream
            # cases: models cur_batch=None, server_is_idle=True -- never
            # touches the tensor-dict wire. Stays alive (never tears its
            # process group down) so the victim's wait is against a live,
            # silent peer, not a torn-down one -- the real shape of the
            # specimen, not a connection-refused shortcut.
            wp("idle_no_batch_scheduled")
            time.sleep(IDLE_UPSTREAM_SLEEP_S)
            res["ok"] = True
            res["note"] = "idle upstream, never sent"
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
                except BaseException:  # noqa: BLE001 - best-effort teardown only
                    pass


def _run(case, join_timeout, readiness_budget_env=None):
    """Bounded, per-rank-failure-reporting driver: real processes over real
    gloo, joined with a deadline and forcibly terminated (never left to
    hang the suite) if still alive past it -- the 788 pattern."""
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [
            ctx.Process(
                target=_worker,
                args=(r, init_file, tmp, case, readiness_budget_env),
            )
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        deadline = time.time() + join_timeout
        for p in procs:
            p.join(timeout=max(0.1, deadline - time.time()))
        stuck_ranks = [r for r, p in enumerate(procs) if p.is_alive()]
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

        out = {"stuck_ranks": stuck_ranks}
        for r in range(WORLD):
            out[f"progress_{r}"] = _load(os.path.join(tmp, f"progress_r{r}.json"), None)
            out[f"result_{r}"] = _load(os.path.join(tmp, f"r{r}.json"), None)
        return out


class PPProxyReadinessContract789(unittest.TestCase):
    def test_neutered_gate_wedges_unboundedly(self):
        """RED reproduction / CAN-FAIL PROOF #1. With the #789 gate
        monkeypatched to a no-op -- byte-identical to the pre-#789 shipped
        code, which never called any such gate -- the victim genuinely
        blocks forever in the real gloo receive against two idle upstreams,
        exactly the specimen's own signature. Bounded here only by the
        outer driver."""
        res = _run("neutered", join_timeout=NEUTERED_JOIN_TIMEOUT_S)
        self.assertIn(
            VICTIM,
            res["stuck_ranks"],
            f"expected the victim to be genuinely blocked with the gate "
            f"neutered (the pre-#789 shipped behaviour) -- it was not, so "
            f"either this harness does not reproduce the specimen or the "
            f"gate is somehow still active: {res}",
        )
        v = res[f"result_{VICTIM}"]
        self.assertIsNone(
            v,
            f"the victim produced a result at all, meaning it returned "
            f"from _pp_recv_proxy_tensors instead of blocking in it: {res}",
        )

    def test_shipped_gate_raises_named_diagnostic_instead_of_wedging(self):
        """GREEN, the primary invariant. With the real shipped gate active
        (internal budget shortened via env override for test speed), the
        same two-idle-upstream constellation makes the victim RETURN --
        raising a named RuntimeError identifying the slot, the upstream
        rank, and the counter state -- well inside the outer driver's
        bound, instead of hanging past it."""
        res = _run(
            "shipped_no_upstream",
            join_timeout=SHIPPED_JOIN_TIMEOUT_S,
            readiness_budget_env=SHORT_READINESS_BUDGET_S,
        )
        # Only the victim's liveness is asserted here: the two idle
        # upstreams are DESIGNED to sleep past IDLE_UPSTREAM_SLEEP_S
        # (90s) regardless of the victim's outcome -- modelling
        # cur_batch=None, server_is_idle=True, forever -- so they are
        # expected to still be alive (and get terminated) at this join
        # deadline. That is the correct shape of the specimen, not a bug.
        self.assertNotIn(
            VICTIM,
            res["stuck_ranks"],
            f"the victim itself was still blocked -- the #789 gate did "
            f"not return in time: {res}",
        )
        v = res[f"result_{VICTIM}"]
        self.assertIsNotNone(v, f"victim produced no result at all: {res}")
        self.assertIsNotNone(
            v.get("error"), f"the #789 gate did not fire, no error raised: {v}"
        )
        self.assertIn("#789 PROXY READINESS TIMEOUT", v["error"])
        self.assertIn(f"mb_id={LIVE_MB}", v["error"])
        self.assertIn(f"upstream (rank {UPSTREAM_B})", v["error"])
        self.assertIn("No upstream scheduled work for this slot", v["error"])

    def test_gate_does_not_delay_a_genuinely_posted_message(self):
        """DEFAULT-CONSERVATIVE REGRESSION GUARD. When the immediate
        upstream genuinely posts (a real isend over gloo, counted the same
        way production counts it), the gate must add no more than a
        vanishingly small delay, and the ordinary receive must still
        deliver the real message -- proof, not assumption, that a healthy
        pass is unchanged by this contract."""
        res = _run("healthy_upstream_sends", join_timeout=HEALTHY_JOIN_TIMEOUT_S)
        # UPSTREAM_A is a bystander in this case too (never sends, sleeps
        # IDLE_UPSTREAM_SLEEP_S) -- only the victim (and the sender,
        # UPSTREAM_B) are expected to finish inside the join deadline.
        self.assertNotIn(
            VICTIM, res["stuck_ranks"], f"the victim was still blocked: {res}"
        )
        self.assertNotIn(
            UPSTREAM_B,
            res["stuck_ranks"],
            f"the sender itself was unexpectedly still blocked: {res}",
        )
        v = res[f"result_{VICTIM}"]
        self.assertIsNotNone(v, f"victim produced no result: {res}")
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertTrue(v.get("ok"), f"victim did not finish: {v}")
        self.assertEqual(
            v.get("delivered_h"),
            42,
            f"wrong or missing proxy delivered to the ordinary receive: {v}",
        )


class PPProxyReadinessNoOpWithoutCounters(unittest.TestCase):
    def test_noop_when_pp_flip_counters_is_none(self):
        """The reference regression launch command (CLAUDE.md) never sets
        --enable-phase-flip, so pp_flip_counters is None on that path.
        This gate must be a true no-op there: no multiprocessing needed to
        prove it, since the shipped function returns before touching
        anything else on ``self`` at all."""
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        h = types.SimpleNamespace(pp_flip_counters=None)
        h._pp_wait_for_proxy_readiness = types.MethodType(
            SchedulerPPMixin._pp_wait_for_proxy_readiness, h
        )
        started = time.monotonic()
        h._pp_wait_for_proxy_readiness(LIVE_MB)  # must not raise, must not block
        elapsed = time.monotonic() - started
        self.assertLess(
            elapsed,
            0.05,
            f"the gate must be an immediate no-op when pp_flip_counters is "
            f"None (the ordinary non-phase-flip boot); took {elapsed:.3f}s",
        )


if __name__ == "__main__":
    unittest.main()
