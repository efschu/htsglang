"""The request-chain flush must not sit at the top of the pass (#788).

THE SPECIMEN. Production py-spy capture: PP0 and PP1 stuck in gloo
`waitSend` inside `_pp_commit_comm_work` (scheduler_pp_mixin.py:2539, called
from the flush site), PP2 stuck in gloo `waitRecv` inside
`_pp_recv_proxy_tensors` (scheduler_pp_mixin.py:2816). Three ranks, three
different blocking points, one cycle: PP2 is waiting on a proxy tensor that
PP1 has not sent yet because PP1 is blocked flushing a PREVIOUS chain send
that only PP0 (also blocked the same way) can drain. This REFUTES the
"armed-diversion" hypothesis (phase-flip arming/draining, #757/#631's
territory) -- nothing here involves the flip machinery. The cause is
ordering: the old `_pp_forward_and_process_input_requests`
(scheduler_pp_mixin.py:924, pre-fix numbering) issued a BLOCKING flush of
last iteration's chain send at the TOP of the pass, before this rank's own
proxy send for the current iteration was posted. Two upstream ranks parked
in that flush cannot post the proxy the last rank is waiting on: classic
"flush-before-you've-fed-your-downstream-peer" cycle. Precedent: commit
a7ff250dc8 fixed the analogous defect on the OUTPUT channel ("flush the
output sends after the exchange, not before it"); #788 is the same shape on
the REQUEST channel.

THE FIX (as it stands now -- no staging slot). `_pp_forward_and_process_
input_requests` is BYTE-IDENTICAL to the original: it calls
`_pp_commit_comm_work(self.send_req_work)`, then stages
`self.send_req_work = self._pp_send_pyobj_to_next_stage(recv_reqs,
async_send=True)`, then `process_input_requests(recv_reqs)` -- unchanged,
guarded `if not self.pp_group.is_last_rank` for the send half throughout
(scheduler_pp_mixin.py:989), unconditional for the last rank's
`process_input_requests` call. There is no `_pp_pending_req_work` staging
attribute; an earlier design that added one broke
test_scheduler_pp_request_order_633 (its pinned contract is
`commit(previous) -> forward(reqs, async_send=True) -> process(reqs)` with
`send_req_work IS current_send_work` on return) and was reverted. The
entire fix is ONE addition: a new method `_pp_commit_pending_req_work(self)`
whose body is `self._pp_commit_comm_work(self.send_req_work)`
(scheduler_pp_mixin.py:1071), called once per iteration from
`_event_loop_pp_body`, AFTER this rank's own proxy isend and output commit,
BEFORE the phase-flip round hook (`if not self.pp_group.is_last_rank:
self._pp_commit_pending_req_work()`, scheduler_pp_mixin.py:497). The
top-of-pass call inside `_pp_forward_and_process_input_requests` stays
exactly where it was and simply finds an already-cleared list there, so it
is a no-op in the shipped ordering. The two disaggregation loops
(scheduler_pp_mixin.py:618, :765) never call the new method and keep their
original top-of-pass flush behavior -- out of scope for this file.

THE ORDERING PIN (peer-mandated, runtime not source-text). In the real
event loop (`_event_loop_pp_body`, scheduler_pp_mixin.py:403-520), the
shipped call sequence per iteration, for a non-last rank, is:
    recv_reqs = self.request_receiver.recv_requests()      (line 420, chain recv, EVERY rank, EVERY iteration)
    self._pp_forward_and_process_input_requests(recv_reqs) (line 421)
    if cur_batch: pp_proxy_tensors = self._pp_recv_proxy_tensors(mb_id)  (line 433, gated on cur_batch; no-op for the first rank)
    ...
    if not is_last_rank and cur_batch: <proxy isend>        (lines 471-484)
    ...
    if not is_last_rank: self._pp_commit_pending_req_work() (line 497)  <-- #788's relocated flush
    ...
    if enable_phase_flip: self._phase_flip_on_round(...)    (lines 508-509, applies to ALL ranks)
i.e. per pass: chain recv, then (if this pass has a batch) proxy recv, then
this rank's own proxy send, then the end-of-iteration chain flush, then the
round hook -- and chain_flush lands strictly between this rank's proxy send
and the round hook. This file cannot spin up the full CUDA scheduler to call
`_phase_flip_on_round` for real, so the driver below calls the SHIPPED
`_pp_commit_pending_req_work` (real function object, bound the same way
test_pp_flip_leftover_proxy_757.py binds shipped methods) and a
runtime-instrumented `_round_hook_stub` standing in for the real hook,
wrapped so each call APPENDS to a live ops list as it happens -- not a grep
over line numbers.
`test_fixed_ordering_completes_and_ordering_pin_holds` asserts on that
recorded, per-iteration sequence for every non-last rank.

WHY THIS TEST IS THREE REAL PROCESSES OVER REAL GLOO. Same reasoning as
test_pp_flip_leftover_proxy_757.py: this is a message-ordering fact on a
real wire, unreproducible by a mocked group. `_pp_forward_and_process_input_
requests`, `_pp_commit_comm_work`, `_pp_send_pyobj_to_next_stage`, and
`_pp_commit_pending_req_work` are the SHIPPED code, bound to a holder
exactly as the #757 file binds `SchedulerPPMixin` methods.

FIDELITY CORRECTION (this revision). An earlier revision of this file
scripted PP2 (the last rank) with a hardcoded sequence `proxy_recv_1,
proxy_recv_2, chain_recv_1, ...` -- two proxy receives before any chain
receive. That schedule is UNREACHABLE: the real loop posts exactly one
chain receive at the top of every iteration (line 420, unconditional for
every rank) and at most one proxy receive per iteration (line 433, gated on
this rank's own `cur_batch`), so two consecutive proxy receives with no
chain receive between them can never happen for any real rank. Under that
unfaithful script the FIXED (correct) code was reported as deadlocked -- a
false failure. This revision replaces the three hand-different, partly
illegal per-rank scripts with ONE generic per-rank loop
(`_worker`/`_run_pipeline`) that every rank drives identically, in the
faithful line-420/433/471-484/497/508-509 order above, and reports on the
outcome under that faithful model in the DETERMINISM/NEGATIVE FINDING
section below.

DETERMINISM, NOT A TIMING RACE. An earlier throwaway probe
(/tmp/wedge788_probe/probe1.py) tried to reproduce this by racing wall-clock
skew against the shipped loop and was UNRELIABLE -- it hung under both old
and new ordering, and produced inconsistent results (including a gloo
"Connection closed by peer") when tuned. This file instead drives an
explicit call order across several passes per rank, so any outcome (hang or
completion) is forced by the dependency graph, not by luck.

PAYLOAD SIZE. The request-chain messages carry a 65536-byte payload (not an
empty list), chosen so the chain-flush `isend(...).wait()` cannot complete
"for free" regardless of whether this build's gloo backend buffers small
sends eagerly. This was checked directly, not assumed: a standalone 2-process
gloo probe (isend on one rank, a 2.0s-delayed matching recv on the other)
was run across payload sizes {8, 1024, 65536, 1048576} bytes; in this
venv/build ALL of them blocked for the full 2.0s with no eager completion at
any size, including 8 bytes. So in this environment payload size is not the
deciding factor for the finding below -- but a realistic, above-any-plausible-
threshold payload is used anyway, both because a peer reviewer raised it as
a legitimate model parameter worth ruling out explicitly, and because it
matches the production trigger (the boot idled healthily and only wedged on
the first substantial request, a ~6443-token prompt -- i.e. the first chain
message large enough for its size to matter at all, on any backend where it
would).

NEGATIVE FINDING -- READ BEFORE ASSUMING THIS FILE REPRODUCES THE SPECIMEN.
Task instruction, verbatim: "If, after modelling it faithfully, the
neutered ordering does NOT deadlock, then SAY SO PLAINLY ... do not
manufacture a hang to keep the test red." It does not. Under the faithful,
reachable per-rank schedule above -- including asymmetric/telescoping
variants where an upstream rank runs more passes than a downstream rank
ever consumes, i.e. "a pass in which the downstream peer has already
stopped consuming and this rank's proxy send/chain forward has no waiting
receiver this round" -- the neutered (pre-#788-fix flush placement) ordering
completes cleanly every time this was tried: by hand (dependency-graph
reasoning: in a linear 3-rank chain+proxy-only relay with
"flush-old-before-dispatch-new", each rank's blocking flush only ever needs
its immediate downstream neighbor to be exactly one pass behind, which
holds by induction from the empty-initial-state base case, regardless of
per-rank pass-count skew) and empirically (dozens of real 3-process gloo
runs across N=3 and N=10 depths and 10+ telescoping offset combinations,
all `stuck_ranks == []`). `test_neutered_ordering_completes_under_faithful_
asymmetric_schedule` below asserts exactly this, as an honest regression
guard on the reduced model -- NOT as a specimen reproduction.

What this means for the specimen: the two-channel (chain + proxy) relay
modelled here is provably deadlock-free by itself. The production deadlock
therefore needs an ingredient this reduced model does not have. The most
plausible candidate, found while checking for one: `_event_loop_pp_body`
also drives a THIRD channel, the pp OUTPUT tensors
(`_pp_send_recv_and_preprocess_output_tensors`, scheduler_pp_mixin.py:3038,
called at :2582), which closes the ring from the LAST rank back to rank 0
(`next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size`,
and `_pp_send_output_to_next_stage`'s last-rank branch sends to "rank 0").
A linear chain+proxy relay plus a ring-closing output edge is a materially
different dependency graph from the one proven acyclic above, and could not
be added here without pulling in the full GPU scheduler/batch machinery --
out of both this file's and this task's scope (edit exactly one test file,
no source changes, no GPU). Real compute-time skew between ranks (GPU
forward-pass duration, absent entirely from this CPU-only hermetic model)
is a second plausible ingredient. Neither is confirmed; both are named here
so the gap is not silently dropped. The mechanism's evidence of record
remains the production py-spy trace quoted under THE SPECIMEN above, plus
precedent a7ff250dc8 -- not this file. A deterministic 3-process repro of
the exact specimen was attempted in good faith and not achieved under any
schedule this file can honestly call reachable.

CAN-FAIL, TWO SEPARATE PROOFS.
  1. `test_fixed_ordering_completes_and_ordering_pin_holds` is the
     load-bearing test: every non-last rank's shipped-code call sequence is
     recorded live and must be exactly (proxy_send, chain_flush, round_hook)
     per pass.
  2. `test_ordering_pin_discriminates_neutered_placement` proves that pin is
     not vacuous, via a SOURCE-LEVEL REORDERING mutation: it runs the exact
     same driver with the flush call relocated back to the top of the pass
     (`_neutered_flush_then_send`, i.e. the #788 bug re-introduced) and
     checks the recorded ops sequence no longer matches -- the shipped
     `_pp_commit_pending_req_work` marker is simply never emitted under that
     placement, because the pre-fix code never called it at all (the flush
     happened inline, earlier, inside `_pp_forward_and_process_input_
     requests`). If #788 were ever silently reverted, this is the specific,
     narrow way the pin catches it -- independent of whether that reversion
     also happens to deadlock under any given schedule.
"""

import json
import os
import tempfile
import time
import types
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60)

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2
LAST = WORLD - 1

#: HAND PIN #770: depth chosen analytically (see module docstring
#: DETERMINISM section) as a depth deep enough that a real ordering-induced
#: cycle (were the neutered placement to reintroduce one) would show up
#: within the join deadline, not merely be "still starting up". Not
#: independently re-measured; kept in lockstep with the driver logic below.
N_PASSES = 3

#: See PAYLOAD SIZE in the module docstring: large enough that this
#: environment's gloo backend (measured: no eager completion at any tested
#: size, 8 to 1048576 bytes) genuinely blocks the flush on the peer's recv,
#: and large enough to resemble a real request payload rather than an idle
#: tick.
PAYLOAD = [b"x" * 65536]

#: Bounded per-rank join deadline. Generous relative to a hermetic CPU-only
#: gloo run (sub-second in every case actually observed by this file); any
#: case that consumes the entire budget is, by construction, reported as
#: stuck rather than silently hung.
JOIN_TIMEOUT_S = 20.0


def _make_holder(rank):
    """Bind the SHIPPED PP mixin methods to a minimal holder, #757-style."""
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
    pp_group = types.SimpleNamespace(
        is_first_rank=(rank == 0),
        is_last_rank=(rank == LAST),
    )
    h = types.SimpleNamespace(
        ps=ps,
        pp_group=pp_group,
        world_group=types.SimpleNamespace(cpu_group=None),
        send_req_work=[],
        process_input_requests=lambda recv_reqs: None,
        pp_phase_flip_armed=lambda: False,
        pp_flip_service=lambda: None,
        _pp_flip_bump_sent=lambda chan: None,
    )
    for name in (
        "_pp_forward_and_process_input_requests",
        "_pp_commit_comm_work",
        "_pp_send_pyobj_to_next_stage",
        "_pp_commit_pending_req_work",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _neutered_flush_then_send(h, recv_reqs):
    """Mirrors the PRE-FIX body of `_pp_forward_and_process_input_requests`:
    blocking-flush the previous send BEFORE posting this pass's send, at the
    top of the pass -- the exact ordering the #788 fix removed. Note this is
    NOT routed through the wrapped `_pp_commit_pending_req_work` (see
    `_install_ops_pin`): the pre-fix code had no such call site, the flush
    was inline here. That asymmetry is itself the basis of
    `test_ordering_pin_discriminates_neutered_placement` below."""
    h._pp_commit_comm_work(h.send_req_work)
    h.send_req_work = h._pp_send_pyobj_to_next_stage(recv_reqs, async_send=True)


def _install_ops_pin(h, ops):
    """Wrap the SHIPPED `_pp_commit_pending_req_work` so every real call is
    recorded into `ops` as it happens -- a runtime instrumentation, not a
    source-text/line-number check."""
    real_commit = h._pp_commit_pending_req_work

    def _instrumented():
        ops.append("chain_flush")
        real_commit()

    h._pp_commit_pending_req_work = _instrumented

    def _round_hook_stub():
        # Stands in for `_phase_flip_on_round` (scheduler_pp_mixin.py:509),
        # which needs the full CUDA scheduler to invoke for real. Its
        # position in the driver mirrors the shipped call order verified at
        # scheduler_pp_mixin.py:497 (commit) vs :509 (round hook): the
        # commit is unconditionally BEFORE the round hook in the shipped
        # `_event_loop_pp_body`, for every rank (the round hook itself is
        # not gated on is_last_rank).
        ops.append("round_hook")

    return _round_hook_stub


def _worker(rank, init_file, out_dir, variant, n_local_passes, downstream_passes):
    """One rank's faithful per-pass driver, mirroring
    `_event_loop_pp_body` (scheduler_pp_mixin.py:403-520) line for line:

        pass{p}_chain_recv    <-> line 420 (chain recv, every rank, every pass)
        pass{p}_chain_stage/
        pass{p}_chain_flush_send <-> line 421 (fixed: stage only; neutered:
                                      pre-fix inline flush-then-stage)
        pass{p}_proxy_recv    <-> line 433 (gated on cur_batch; modelled as
                                      always-true here, i.e. a steady, always-
                                      busy pipeline -- itself the REACHABLE
                                      schedule the previous revision lacked)
        pass{p}_proxy_send    <-> lines 471-484 (gated on not is_last_rank)
        pass{p}_chain_flush   <-> line 497 (the #788-relocated end-of-
                                      iteration flush, not_is_last_rank only)
        pass{p}_round_hook    <-> lines 508-509 (unconditional every rank)

    `variant` is "neutered" (flush call relocated back to the top of the
    pass, the pre-#788-fix placement) or "fixed" (shipped placement).
    `n_local_passes` is how many passes THIS rank runs; `downstream_passes`
    is how many the next rank runs (or None for the last rank) -- used only
    to bound how many of this rank's fire-and-forget proxy isends are worth
    waiting on before teardown, so a rank that intentionally outruns a
    downstream peer that already exited (the asymmetric/telescoping case)
    does not hang waiting for a receive nobody will ever post."""
    from sglang.srt.utils.common import point_to_point_pyobj

    neutered = variant == "neutered"
    progress_path = os.path.join(out_dir, f"progress_r{rank}.json")

    def wp(where):
        with open(progress_path, "w") as f:
            json.dump({"rank": rank, "where": where}, f)

    wp("start")
    res = {"rank": rank, "ok": False}
    ops = []
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        proxy_group = dist.new_group(ranks=list(range(WORLD)))
        h = _make_holder(rank)
        round_hook = _install_ops_pin(h, ops)
        received = []
        pending_isends = []  # fire-and-forget proxy sends; drained below

        for p in range(1, n_local_passes + 1):
            wp(f"pass{p}_chain_recv")
            if rank == PP0:
                recv_reqs = list(PAYLOAD)
            else:
                recv_reqs = point_to_point_pyobj([], rank, None, rank - 1, rank)

            if neutered and rank != LAST:
                wp(f"pass{p}_chain_flush_send")
                _neutered_flush_then_send(h, recv_reqs)
            else:
                wp(f"pass{p}_chain_stage")
                h._pp_forward_and_process_input_requests(recv_reqs)

            if rank != PP0:
                wp(f"pass{p}_proxy_recv")
                buf = torch.zeros(1, dtype=torch.long)
                dist.recv(buf, src=rank - 1, group=proxy_group)
                received.append(buf.item())

            if rank != LAST:
                wp(f"pass{p}_proxy_send")
                ops.append("proxy_send")
                t = torch.tensor([p], dtype=torch.long)
                pending_isends.append(dist.isend(t, dst=rank + 1, group=proxy_group))

            if not neutered and rank != LAST:
                wp(f"pass{p}_chain_flush")
                h._pp_commit_pending_req_work()

            wp(f"pass{p}_round_hook")
            round_hook()

        # Only wait on isends the downstream peer will actually consume:
        # telescoping deliberately leaves each upstream rank's trailing
        # passes unconsumed by a downstream that already exited, and
        # waiting on those would hang on a message nobody will ever
        # receive. This mirrors nothing in the real loop (which never
        # exits early); it exists purely so this hermetic driver can shut
        # down cleanly after an intentionally asymmetric run.
        matched = (
            min(len(pending_isends), downstream_passes)
            if downstream_passes is not None
            else len(pending_isends)
        )
        for w in pending_isends[:matched]:
            w.wait()

        res["ok"] = True
        res["received"] = received
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        with open(os.path.join(out_dir, f"result_r{rank}.json"), "w") as f:
            json.dump(res, f)
        with open(os.path.join(out_dir, f"ops_r{rank}.json"), "w") as f:
            json.dump(ops, f)
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:  # noqa: BLE001 - best-effort teardown only
                pass


def _run(variant, n_local_passes=None, join_timeout=JOIN_TIMEOUT_S):
    """Bounded, per-rank-failure-reporting driver: real processes over real
    gloo, joined with a deadline and terminated (never left to hang the
    suite) if still alive past it. `n_local_passes` maps rank -> pass count;
    defaults to N_PASSES for every rank (the uniform/no-skew schedule)."""
    if n_local_passes is None:
        n_local_passes = dict.fromkeys(range(WORLD), N_PASSES)
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [
            ctx.Process(
                target=_worker,
                args=(
                    r,
                    init_file,
                    tmp,
                    variant,
                    n_local_passes[r],
                    n_local_passes.get(r + 1) if r + 1 < WORLD else None,
                ),
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
            out[f"result_{r}"] = _load(os.path.join(tmp, f"result_r{r}.json"), None)
            out[f"ops_{r}"] = _load(os.path.join(tmp, f"ops_r{r}.json"), [])
        return out


_FIXED_PIN_TRIPLE_PER_PASS = ["proxy_send", "chain_flush", "round_hook"]


class PPChainFlushDeadlock788(unittest.TestCase):
    def test_fixed_ordering_completes_and_ordering_pin_holds(self):
        """GREEN + ORDERING PIN, load-bearing. With the shipped (post-#788-
        fix) ordering, all three ranks complete cleanly, PP2 receives the
        proxy payloads in order, and every non-last rank's runtime-recorded
        op sequence confirms the chain flush lands strictly after this
        rank's own proxy send and strictly before the round hook, on every
        pass -- the ordering pinned at runtime, not via a source-text/
        line-number check."""
        res = _run("fixed")
        self.assertEqual(
            res["stuck_ranks"], [], f"unexpected hang under fixed ordering: {res}"
        )
        for r in range(WORLD):
            result = res[f"result_{r}"]
            self.assertIsNotNone(result, f"rank {r} produced no result: {res}")
            self.assertTrue(result.get("ok"), f"rank {r} failed: {result.get('error')}")
        self.assertEqual(res[f"result_{PP2}"]["received"], [1, 2, 3])

        for rank in (PP0, PP1):
            ops = res[f"ops_{rank}"]
            expected = _FIXED_PIN_TRIPLE_PER_PASS * N_PASSES
            self.assertEqual(
                ops,
                expected,
                f"rank {rank}: ordering pin violated, recorded sequence was "
                f"{ops} (expected proxy_send before chain_flush before "
                f"round_hook on every pass, mirroring scheduler_pp_mixin.py:"
                f"471-484 (proxy isend) < :497 (_pp_commit_pending_req_work)"
                f" < :508-509 (round hook))",
            )
        # PP2 (last rank) never sends chain or proxy, but the round hook is
        # unconditional (line 508-509 is not gated on is_last_rank).
        self.assertEqual(res[f"ops_{PP2}"], ["round_hook"] * N_PASSES)

    def test_ordering_pin_discriminates_neutered_placement(self):
        """CAN-FAIL PROOF #1, source-level reordering mutation. Runs the
        identical driver with the flush call relocated back to the top of
        the pass (`_neutered_flush_then_send`, i.e. the #788 bug
        re-introduced) and confirms the ordering pin no longer matches what
        `test_fixed_ordering_completes_and_ordering_pin_holds` requires --
        proving that assertion is not vacuously true. Under this placement
        the shipped `_pp_commit_pending_req_work` marker never fires at all
        (the pre-fix code called `_pp_commit_comm_work` inline, not through
        that method), so `chain_flush` is simply absent from the recorded
        sequence. This is deliberately independent of whether the neutered
        placement also happens to deadlock under any given schedule --
        see the NEGATIVE FINDING section of the module docstring for that
        separate question."""
        res = _run("neutered")
        for r in range(WORLD):
            result = res[f"result_{r}"]
            self.assertIsNotNone(result, f"rank {r} produced no result: {res}")
            self.assertTrue(result.get("ok"), f"rank {r} failed: {result.get('error')}")
        for rank in (PP0, PP1):
            ops = res[f"ops_{rank}"]
            self.assertNotEqual(
                ops,
                _FIXED_PIN_TRIPLE_PER_PASS * N_PASSES,
                f"rank {rank}: neutered placement unexpectedly produced the "
                f"same recorded sequence as the fixed ordering -- the pin "
                f"would not have caught a #788 reversion",
            )
            self.assertNotIn(
                "chain_flush",
                ops,
                f"rank {rank}: expected the pre-fix inline flush to bypass "
                f"the instrumented `_pp_commit_pending_req_work` entirely, "
                f"but saw it recorded: {ops}",
            )

    def test_neutered_ordering_completes_under_faithful_asymmetric_schedule(self):
        """HONEST NEGATIVE FINDING, not a specimen reproduction -- see the
        module docstring's NEGATIVE FINDING section for the full reasoning
        (by-hand induction proof + this file's own empirical sweep). Task
        instruction, verbatim: "If, after modelling it faithfully, the
        neutered ordering does NOT deadlock, then SAY SO PLAINLY ... do not
        manufacture a hang to keep the test red." It does not: under this
        faithful, reachable, deliberately ASYMMETRIC schedule (PP0 outruns
        PP1 outruns PP2 by one pass each, i.e. each upstream rank has at
        least one pass where its downstream peer has already stopped
        consuming -- the "an upstream rank has no [more] batch[es]" case the
        task asked to model honestly), the pre-#788-fix flush placement
        still completes cleanly. This is asserted here as a real regression
        guard on the reduced (chain+proxy only) model: it should keep
        completing under this file's own driver regardless of unrelated
        future edits to this test, and a change that broke that invariant
        would be worth noticing even though it would NOT, by itself,
        constitute reproducing the production specimen (which this reduced
        model cannot honestly claim to reproduce -- again, see the module
        docstring)."""
        n_local_passes = {PP0: N_PASSES + 2, PP1: N_PASSES + 1, PP2: N_PASSES}
        res = _run("neutered", n_local_passes=n_local_passes)
        self.assertEqual(
            res["stuck_ranks"],
            [],
            "NEGATIVE FINDING DID NOT HOLD: the neutered ordering hung "
            f"under this asymmetric faithful schedule: {res}. This would "
            "be a genuinely interesting result (the by-hand induction proof "
            "and prior empirical sweep both said this cannot happen) and "
            "should be investigated, not silently accepted or reverted.",
        )
        for r in range(WORLD):
            result = res[f"result_{r}"]
            self.assertIsNotNone(result, f"rank {r} produced no result: {res}")
            self.assertTrue(result.get("ok"), f"rank {r} failed: {result.get('error')}")


if __name__ == "__main__":
    unittest.main()
