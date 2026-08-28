"""#801-H3: the state is REACHABLE, and it is already closed -- FORWARDS.

THE ORDER WAS "build the reverse wire, red repro first". The repro was run
before anything was built and it answered a different question than the one
asked. This file records the measurement; no wire is built.

WHAT WAS ASKED FOR. `_pp_send_output_to_next_stage`'s intermediate hop names
its own residual: "the case where this rank received NOTHING because its OWN
slot was empty while the successor's was not. No rank publishes its per-slot
expectation to its PREDECESSOR, so this end cannot see that disagreement --
closing it is a wire against the ring direction and is a separate posting."
That posting is H3, and `send_tensor_dict`/`recv_tensor_dict` being ring-
directional makes it a genuinely new collective pair on a ring that has
already buried four (bounded-recv, CORPSE R, CORPSE S, the 1716 void loop).

THE REGISTER'S RECIPE, EXECUTED (REGISTER_OPEN_876.txt 2026-08-27, "#801
REPRO-REZEPT (abgeleitet, NICHT ausgefuehrt)"): a void PP0 absorbs whose
forward payload is None empties PP0's slot and sends nothing on; PP1's slot
is untouched, PP1 receives unconditionally, wedge.

  DRIVEN, AND RED -- against the PRE-#801 relay rule. Restoring #797's
  `pp_void_relay_stop_rank` default over the 3-process gloo ring reproduces
  the predicted signature exactly: PP0 pass7_chain_flush, PP1
  pass6_output_exchange, PP2 pass7_chain_recv. With the shipped default the
  same hazard completes all twelve passes.

  AND ALREADY IN THE TREE. Rebuilt from the recipe's own four facts, the
  hazard converges on test_pp_void_send_contract_801.py's verbatim -- every
  prefix told as 0 (the zero-offer branch honours an offer of 0 without a
  lookup, so nothing is retracted anywhere) and one rank not holding the
  request. `TheRingSurvivesAVoidWithNoRetraction` is that red/green pair.
  The recipe was a correct derivation against the pin it was written for
  (a516b3750b, #792+#926 and NOT #801); at a pin carrying 49f39b1068 it
  describes a closed defect. None of its arms is duplicated here.

  IT ALSO IS NOT H3. The recipe's wedge is a rank whose OWN slot is FULL and
  whose message never arrives. H3 is the inverse.

H3 IS REACHABLE, AND THE RETRACTION PATH IS NOT HOW. Two findings, and their
order is the whole content of this file.

  (1) NOT THROUGH A RETRACTION. `pp_pass_should_void` voids a rank's WHOLE
      pass on any retraction of its own, `_pp_send_admission_decision`
      forwards `pass_voided` OR-ed and never cleared, and
      `reconcile_pp_admission_decision` passes an already-retracted entry
      through verbatim. Emptiness therefore spreads only FROM a rank TOWARDS
      the last one and cannot skip one, so the ranks with an empty slot are
      exactly `k..last`. The H3 antecedent asks for the inverse ordering and
      is unrepresentable on this path -- pinned exhaustively over rings of
      2..8 and measured over every retraction position the ring can produce.
      This is why the earlier attempts, which drove cold-cache hazards, could
      not reach it.

  (2) THROUGH A LOST REQUEST ON AN INTERMEDIATE RANK -- the live D1 shape,
      "rid 10df51e9 verschwindet auf PP1 nach Chunk 2 aus allen vier Orten"
      with `#944 UNRESOLVED told=... local=UNKNOWN`. The zero offer that
      `UNRESOLVED_DEFER_CAP`'s escape produces is honoured WITHOUT a lookup,
      so a rank that no longer holds the request retracts NOTHING and still
      admits NOTHING. Its slot is empty; its successor, which does hold the
      request, admits and has an unconditional receive posted. Own slot
      empty, successor's full: H3, reached with no retraction anywhere, and
      `test_a_request_lost_on_an_intermediate_rank_wedges_the_successor`
      measures the wedge on real gloo -- PP2 parked in the output exchange
      while PP0 and PP1 overtake it.

THE CLOSURE IS #951, AND IT RUNS THE OTHER WAY DOWN THE RING. The successor
does not have to be told what its predecessor EXPECTS; it is told what its
predecessor DID. `_pp_send_admission_decision` carries `launched=self.mbs
[mb_id] is not None` (scheduler_pp_mixin.py:2457), overwritten by every hop,
`_pp_recv_admission_decision` reads it into `_pp_upstream_launched_incoming`,
and `pp_upstream_void_pending` refuses a pass whose upstream posted nothing.
The successor gives up its receive instead of the predecessor acquiring an
obligation to send -- the same disagreement, resolved in the direction the
ring already runs. `test_the_upstream_launched_posting_closes_it` is the
green arm: the identical hazard, the shipped guard wired in, twelve passes.

VERDICT: NO REVERSE WIRE. The state H3 names is real, and a fifth collective
pair against the ring direction is not what closes it -- #951 already does,
forwards. What remains is that the comment quoting the residual predates its
own fix and still reads as an open posting; that staleness is what generated
this order, and it is corrected at the site rather than left to generate the
order again.

HARNESS. The ring, the wire and every method under test are 791b's and #795's
verbatim. What this file adds is the hazard SHAPE and POSITION as parameters
(791b fixes both), the per-pass occupancy trace that turns reachability into a
measurement, and #951's posting as a switch -- 791b's driver predates #951 and
models neither half of it, which is precisely what lets the red arm reach the
H3 state at all. The spawn trap is 791b's: `_run` uses the "spawn" start
method, so everything a child must see travels through the environment.
"""

import json
import os
import tempfile
import time
import types
import unittest
from collections import deque

import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=180)

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2
LAST = WORLD - 1
LOOP = WORLD

#: 791b's values, kept identical so the two files' runs stay comparable.
RETRACT_PASS = 4
LAST_BUSY_PASS = RETRACT_PASS
N_PASSES = 12
TOLD = 512
PAYLOAD = [b"x" * 65536]

JOIN_TIMEOUT_S = 60.0

#: The wedged arm never completes by construction, so there is no timing
#: window to lose -- this only has to outlast spawn and gloo init.
RED_JOIN_TIMEOUT_S = 25.0

#: 791b's hazard: one rank's cache is cold on `RETRACT_PASS`, so it RETRACTS
#: and the retraction is NAMED on the wire.
MODE_COLD = "cold"

#: The recipe's hazard: on `RETRACT_PASS` every prefix is told as 0 and the
#: hazard rank no longer holds the request. `reconcile_pp_admission_decision`
#: honours a zero offer without a lookup, so nothing is retracted anywhere and
#: the rank still has no batch -- a void naming no retracting rank.
MODE_LOST_RID = "lost_rid"

MODE_ENV = "SGLANG_801_H3_MODE"
HAZARD_RANK_ENV = "SGLANG_801_H3_HAZARD_RANK"

#: Whether the harness carries #951's `launched` posting and the guard that
#: reads it. 791b's driver predates both and models neither, which is exactly
#: what lets the red arm here reach the H3 state at all.
GUARD_951_ENV = "SGLANG_801_H3_GUARD_951"


def _rid_for(pass_idx: int) -> str:
    return f"rid{pass_idx:04d}"


class _StubReq:
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
    def __init__(self, reqs):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.forward_mode = ForwardMode.EXTEND
        self.reqs = list(reqs)
        self.contains_last_prefill_chunk = True
        self.return_logprob = False


def _make_holder(rank, wire, chain_group):
    """791b's holder verbatim: the shipped chain, decision and output-ring
    methods bound onto one stand-in."""
    from sglang.srt.managers.pp_admission_congruence import PPAdmissionCongruenceGuard
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
        "_pp_forward_and_process_input_requests",
        "_pp_commit_comm_work",
        "_pp_send_pyobj_to_next_stage",
        "_pp_recv_pyobj_from_prev_stage",
        "_pp_commit_pending_req_work",
        "_pp_send_admission_decision",
        "_pp_recv_admission_decision",
        "_pp_reconcile_incoming_admission",
        "_pp_commit_admission_send_work",
        "_pp_send_dict_to_next_stage",
        "_pp_recv_typed_dict",
        "_pp_note_output_expectation",
        "_pp_output_expected_for_slot",
        "_pp_void_output_payload",
        "_pp_absorb_void_output",
        "_pp_send_output_to_next_stage",
        "_pp_wait_for_dict_readiness",
        "_pp_recv_dict_from_prev_stage",
        "_pp_send_recv_and_preprocess_output_tensors",
        "_pp_commit_send_output_work_and_preprocess_output_tensors",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _worker(rank, init_file, out_dir, n_passes):
    """791b's driver with the hazard as a parameter and a slot-occupancy
    trace. Everything that touches the wire is shipped code."""
    import torch
    import torch.distributed as dist

    import test_pp_admission_chain_flush_deadlock_795 as ring
    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionDecision,
        PPAdmissionEntry,
    )
    from sglang.srt.managers.scheduler_pp_mixin import (
        _pp_output_exchange_due,
        pp_upstream_void_pending,
    )
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    mode = os.environ.get(MODE_ENV, MODE_COLD)
    hazard_rank = int(os.environ.get(HAZARD_RANK_ENV, PP1))
    guard_951 = os.environ.get(GUARD_951_ENV) == "1"

    def _local_match(pass_idx):
        if mode == MODE_COLD and rank == hazard_rank and pass_idx == RETRACT_PASS:
            return 0
        return TOLD

    def _told_for(pass_idx):
        # The zero offer is the recipe's door into a void that names no
        # retraction: honoured without a lookup, so a rank that no longer
        # holds the request retracts NOTHING and still has no batch.
        if mode == MODE_LOST_RID and pass_idx == RETRACT_PASS:
            return 0
        return TOLD

    def _rank_holds_request(pass_idx):
        return not (
            mode == MODE_LOST_RID and pass_idx == RETRACT_PASS and rank == hazard_rank
        )

    res = {
        "rank": rank,
        "ok": False,
        "passes": 0,
        "voids": 0,
        "mode": mode,
        "hazard_rank": hazard_rank,
        "trace": [],
    }
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        chain_group = dist.new_group(ranks=list(range(WORLD)))
        wire = ring._RingWire(rank)
        h = _make_holder(rank, wire, chain_group)

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
            told = _told_for(p)
            if busy and _rank_holds_request(p):
                h.waiting_queue.append(_StubReq(rid, _local_match(p)))

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
                effective = {rid: told} if busy else {}

            # #951's guard, at the moment scheduler.py asks it: BEFORE the
            # batch is built, so a pass whose upstream posted nothing is
            # REFUSED rather than unwound. `_pp_upstream_launched_incoming`
            # was written by the shipped `_pp_recv_admission_decision` above.
            refused = guard_951 and pp_upstream_void_pending(h)
            admitted = (
                [] if refused else [r for r in h.waiting_queue if r.rid in effective]
            )
            for req in admitted:
                h.waiting_queue.remove(req)
            h.mbs[mb_id] = _StubBatch(admitted) if admitted else None
            entry_refused = refused

            ring._progress(out_dir, rank, f"pass{p}_decision_send")
            if rank == PP0:
                entries = (
                    (PPAdmissionEntry(rid=rid, prefix_len=told, extend_len=1),)
                    if busy
                    else ()
                )
                decision = PPAdmissionDecision(mb_id=mb_id, entries=entries)
                expects_output = _pp_output_exchange_due(h.mbs[mb_id])
                h._pp_note_output_expectation(mb_id, expects_output, None)
                h._pp_send_admission_decision(
                    decision,
                    expects_output=expects_output,
                    **({"launched": h.mbs[mb_id] is not None} if guard_951 else {}),
                )
            else:
                fwd = (
                    amended
                    if amended is not None
                    else PPAdmissionDecision(mb_id=mb_id, entries=())
                )
                h._pp_send_admission_decision(
                    fwd,
                    expects_output=h._pp_output_expected_incoming,
                    # #951: overwritten by every hop -- it is THIS rank's own
                    # slot, not an OR-ed ring fact like `pass_voided`.
                    **({"launched": h.mbs[mb_id] is not None} if guard_951 else {}),
                )

            if rank == LAST and h.mbs[mb_id] is not None:
                h.last_rank_comm_queue.append(
                    (
                        object(),
                        PPProxyTensors(
                            {"next_token_ids": torch.tensor([p], dtype=torch.int64)}
                        ),
                    )
                )

            # THE OCCUPANCY BOTH OUTPUT-RING GATES READ, recorded BEFORE the
            # exchange. `recv_slot_occupied` is what this rank's receive gate
            # tests (`target = mbs[next_mb_id]`), i.e. whether this rank has an
            # unconditional receive posted for this generation. Comparing it
            # ACROSS ranks within one pass IS the H3 reachability question.
            entry = {
                "pass": p,
                "mb_id": mb_id,
                "recv_slot": next_mb_id,
                "recv_slot_occupied": h.mbs[next_mb_id] is not None,
                "own_slot_occupied": h.mbs[mb_id] is not None,
                "has_pp_outputs": bool(h.pp_outputs),
                "expects_output": h._pp_output_expected_for_slot(next_first_rank_mb_id),
                "refused_951": entry_refused,
            }

            ring._progress(out_dir, rank, f"pass{p}_output_exchange")
            before = len(h.waiting_queue)
            npo, nbr, d2h = h._pp_commit_send_output_work_and_preprocess_output_tensors(
                next_first_rank_mb_id, next_mb_id
            )
            if len(h.waiting_queue) > before:
                res["voids"] += 1
            entry["forwarded"] = npo is not None
            res["trace"].append(entry)
            h.pp_outputs = npo

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

        # LEAVE TOGETHER, and this is not decoration. The last rank skips
        # `_pp_commit_pending_req_work`, so ranks 0 and 1 reach the end of the
        # run first; without a rendezvous they tear their process groups down
        # while the last rank is still flushing pass 11 and it dies of
        # `Connection closed by peer` -- a harness teardown race that reads
        # exactly like a ring defect in the trace. Measured on the (cold, PP2)
        # position, which is a hazard 791b does not drive.
        dist.barrier(group=chain_group)
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


def _run(
    mode, hazard_rank, n_passes=N_PASSES, join_timeout=JOIN_TIMEOUT_S, guard_951=False
):
    ctx = mp.get_context("spawn")
    previous = {
        k: os.environ.get(k) for k in (MODE_ENV, HAZARD_RANK_ENV, GUARD_951_ENV)
    }
    os.environ[MODE_ENV] = mode
    os.environ[HAZARD_RANK_ENV] = str(hazard_rank)
    if guard_951:
        os.environ[GUARD_951_ENV] = "1"
    else:
        os.environ.pop(GUARD_951_ENV, None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            init_file = os.path.join(tmp, "pg_init")
            procs = [
                ctx.Process(target=_worker, args=(r, init_file, tmp, n_passes))
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
    finally:
        for key, prev in previous.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


class TheRetractionPathCannotReachTheH3State(unittest.TestCase):
    """Real gloo rings, every retraction position, one question."""

    #: Every hazard that goes through a NAMED retraction, plus the recipe's
    #: void that names none: a cold cache on the first downstream rank, a cold
    #: cache on the last rank (a position 791b does not drive), and the last
    #: rank not holding the request.
    POSITIONS = ((MODE_COLD, PP1), (MODE_COLD, PP2), (MODE_LOST_RID, LAST))

    def _traces(self):
        runs = {}
        for mode, hazard in self.POSITIONS:
            out = _run(mode, hazard)
            self.assertEqual(
                out["stuck_ranks"],
                [],
                f"({mode}, {hazard}) wedged, so it cannot answer a "
                f"reachability question: {out}",
            )
            for r in range(WORLD):
                res = out[f"result_{r}"]
                self.assertIsNotNone(res, f"({mode}, {hazard}) rank {r}: no result")
                self.assertTrue(res["ok"], f"({mode}, {hazard}) rank {r}: {res}")
                self.assertEqual(
                    len(res["trace"]),
                    N_PASSES,
                    f"({mode}, {hazard}) rank {r} traced "
                    f"{len(res['trace'])}/{N_PASSES} passes -- an incomplete "
                    f"trace cannot answer a reachability question: {res}",
                )
            runs[(mode, hazard)] = out
        return runs

    def test_a_retraction_never_leaves_a_rank_empty_while_its_successor_is_full(self):
        """WHERE H3 IS *NOT*, and why two attempts hunting it here missed.

        The residual the intermediate hop's comment names is "this rank
        received NOTHING because its OWN slot was empty while the successor's
        was not". Across every pass of every retraction-driven hazard, the
        occupancy of the slot both gates read is monotone downstream: some
        prefix of the ranks holds a batch and the rest do not, never the
        inverse. A retraction can only ever spread emptiness AWAY from rank 0.

        The state IS reachable -- by a path with no retraction in it at all,
        which the next class drives. Keeping this arm is what stops the search
        being repeated on this path a third time.
        """
        for key, out in self._traces().items():
            by_pass = {}
            for r in range(WORLD):
                for e in out[f"result_{r}"]["trace"]:
                    by_pass.setdefault(e["pass"], {})[r] = e["recv_slot_occupied"]
            for p, occ in sorted(by_pass.items()):
                vector = [occ[r] for r in range(WORLD)]
                for r in range(WORLD - 1):
                    self.assertFalse(
                        (not vector[r]) and vector[r + 1],
                        f"{key} pass {p}: PP{r} has an empty slot while PP"
                        f"{r + 1} has a full one. A retraction reached the H3 "
                        f"state, which the monotonicity pin says it cannot -- "
                        f"one of the two is wrong: {vector}",
                    )

    def test_the_recipes_hazard_produces_the_relay_it_was_written_against(self):
        """THE RECIPE'S OWN STATE, REACHED -- so the negative above is not a
        hazard that failed to fire.

        A reachability probe that measured nothing would report the same
        monotone vectors as one that measured everything, which is the
        indicator law applied to this file's own instrument. The recipe's
        void does travel: PP0 and PP1 both held a launched batch for the
        diverged slot and both absorb it, and the last rank -- its source --
        does not. The red/green pair for that relay lives in
        test_pp_void_send_contract_801.py and is deliberately not repeated.
        """
        out = _run(MODE_LOST_RID, LAST)
        self.assertEqual(out["stuck_ranks"], [], f"{out}")
        self.assertEqual(
            [out[f"result_{r}"]["voids"] for r in range(WORLD)],
            [1, 1, 0],
            f"the recipe's void must reach PP0 and PP1 and stop one hop short "
            f"of its own source: {out}",
        )

    def test_the_cold_cache_hazard_never_enters_the_no_retraction_branch(self):
        """WHY THE FIRST ATTEMPT'S RED ARM STAYED GREEN, pinned rather than
        retold, so it is not paid for a third time.

        The register records two honest stops on #801, the first because a
        red arm built on 791b's cold-cache hazard did not go red. The reason
        is structural: a cold cache makes the rank RETRACT, the retraction is
        NAMED on the wire, and `pp_first_retracting_rank` therefore answers a
        rank rather than `None` -- the branch under test is never entered.
        Only the rank that retracted, and the ranks upstream of it, ever see
        the void.
        """
        for hazard in (PP1, PP2):
            out = _run(MODE_COLD, hazard)
            self.assertEqual(out["stuck_ranks"], [], f"hazard {hazard}: {out}")
            voids = [out[f"result_{r}"]["voids"] for r in range(WORLD)]
            self.assertEqual(
                voids,
                [1 if r < hazard else 0 for r in range(WORLD)],
                f"a named retraction at PP{hazard} must stop the relay there, "
                f"so exactly ranks 0..{hazard - 1} absorb: {out}",
            )


class TheH3StateIsReachedByALostRequestAndClosedForwards(unittest.TestCase):
    """THE ANSWER TO THE REVERSE-WIRE ORDER, red arm first.

    The hazard is the live D1 shape and not an invention: a request that has
    gone from an INTERMEDIATE rank's four lookup places, met by the zero offer
    `UNRESOLVED_DEFER_CAP`'s escape produces. `reconcile_pp_admission_decision`
    honours `told <= 0` without a lookup -- "executable whether or not the rank
    has ever heard of the rid" -- so nothing is retracted anywhere, the rank
    that lost the request admits nothing, and the successor that still holds it
    admits and posts its receive. That is the H3 antecedent, with no retraction
    in it, which is why the retraction-driven searches could not find it.
    """

    def test_a_request_lost_on_an_intermediate_rank_wedges_the_successor(self):
        """RED, over real gloo, with #951's posting absent -- which is 791b's
        driver as written, predating #951 and modelling neither half of it.

        MEASURED SIGNATURE: PP0 pass8_chain_flush, PP1 pass7_chain_flush, PP2
        pass6_output_exchange. PP2 is the parked rank: its slot for the
        generation is full, so its receive is unconditional, and PP1 -- whose
        own slot was empty, so `_do_recv` early-returned and `pp_outputs`
        stayed None -- has nothing to forward. The two ranks it blocks have
        overtaken it, which is what makes this a ring and not a slow rank.
        """
        out = _run(MODE_LOST_RID, PP1, join_timeout=RED_JOIN_TIMEOUT_S)

        self.assertEqual(
            out["stuck_ranks"],
            [PP0, PP1, PP2],
            f"the H3 state was expected to close the ring: {out}",
        )
        stalls = out["stall_report"]
        self.assertIn(
            "output_exchange",
            stalls[PP2],
            f"PP2 holds the full slot whose message never comes, so it is the "
            f"rank that must be parked in the output exchange: {stalls}",
        )
        pp2_pass = int(stalls[PP2].split("_", 1)[0][len("pass") :])
        for other in (PP0, PP1):
            self.assertGreater(
                int(stalls[other].split("_", 1)[0][len("pass") :]),
                pp2_pass,
                f"PP{other} should have overtaken the rank it blocks: {stalls}",
            )

    def test_the_upstream_launched_posting_closes_it(self):
        """GREEN, same hazard, #951 wired in -- and the reason no wire against
        the ring direction is needed.

        `launched=self.mbs[mb_id] is not None` rides the admission decision
        that already travels 0 -> 1 -> ... -> last and is overwritten by every
        hop, so PP2 is told by PP1 itself that PP1 posted nothing, and
        `pp_upstream_void_pending` refuses PP2's pass before it is built. The
        successor gives up its receive rather than the predecessor acquiring
        an obligation to send: the same disagreement H3 names, resolved in the
        direction the ring already runs.

        This is also the reachability proof #951 needed. A compensator is only
        established once it is shown to be REACHED FROM the defect path, and
        the arm above is that path driven to a wedge without it.
        """
        out = _run(MODE_LOST_RID, PP1, guard_951=True)

        self.assertEqual(out["stuck_ranks"], [], f"the ring still wedged: {out}")
        for r in range(WORLD):
            res = out[f"result_{r}"]
            self.assertIsNotNone(res, f"rank {r} produced no result: {out}")
            self.assertTrue(res["ok"], f"rank {r} failed: {res}")
            self.assertEqual(res["passes"], N_PASSES, f"rank {r}: {res}")

        # The guard is what did it, and it fired on the rank the hazard names.
        pp1 = out[f"result_{PP1}"]
        pp2 = out[f"result_{PP2}"]
        self.assertFalse(
            any(e["refused_951"] for e in pp1["trace"][:RETRACT_PASS]),
            f"PP1's upstream launched on every pass before the hazard, so "
            f"nothing may be refused there: {pp1['trace']}",
        )
        self.assertTrue(
            any(e["refused_951"] for e in pp2["trace"]),
            f"PP2 must have refused the pass whose upstream posted nothing -- "
            f"without that refusal it holds a full slot and parks: "
            f"{pp2['trace']}",
        )

    def test_the_guard_is_inert_on_every_ring_that_cannot_have_the_problem(self):
        """The dangerous half of the fix, and it is a `False` in three shapes.

        The first rank never receives an admission decision, so its
        `_pp_upstream_launched_incoming` is False on EVERY pass -- a guard
        without the `is_first_rank` term would void every pass on PP0 and
        serve nothing at all. `pp_size <= 1` has no upstream hop, and a gapped
        set has no stage-boundary proxy for the void to protect.
        """
        from sglang.srt.managers.scheduler_pp_mixin import pp_upstream_void_pending

        def _h(pp_size, is_first, gapped, launched):
            return types.SimpleNamespace(
                ps=types.SimpleNamespace(pp_size=pp_size),
                pp_group=types.SimpleNamespace(is_first_rank=is_first),
                _pp_gapped_wire=gapped,
                _pp_upstream_launched_incoming=launched,
            )

        self.assertTrue(pp_upstream_void_pending(_h(WORLD, False, False, False)))
        self.assertFalse(pp_upstream_void_pending(_h(WORLD, False, False, True)))
        self.assertFalse(
            pp_upstream_void_pending(_h(WORLD, True, False, False)),
            "the first rank is never told, so it must never be refused",
        )
        self.assertFalse(pp_upstream_void_pending(_h(1, False, False, False)))
        self.assertFalse(pp_upstream_void_pending(_h(WORLD, False, True, False)))


class TheForwardContractIsTotalOverTheReachableStates(unittest.TestCase):
    """Pure functions, exhaustive over the ring sizes rather than sampled."""

    def test_pass_voidness_is_monotone_downstream(self):
        """THE PREMISE THE UNREACHABILITY RESTS ON, taken at the source.

        `_pp_send_admission_decision` forwards `pass_voided` OR-ed and never
        cleared (scheduler_pp_mixin.py:2451-2456),
        `_pp_recv_admission_decision` reads it back, and
        `pp_pass_should_void` returns True on it unconditionally. Once ANY
        rank voids, every rank after it voids whatever its own local verdict
        is, so an empty slot can only spread FROM a rank TOWARDS the last one
        and can never skip one. That is what makes the H3 state
        unrepresentable rather than merely unobserved.
        """
        from sglang.srt.managers.pp_admission_congruence import (
            PPAdmissionDecision,
            PPAdmissionEntry,
        )
        from sglang.srt.managers.scheduler_pp_mixin import pp_pass_should_void

        def _decision(retracted_by):
            return PPAdmissionDecision(
                mb_id=0,
                entries=(
                    PPAdmissionEntry(
                        rid="rid",
                        prefix_len=TOLD,
                        extend_len=1,
                        admitted=retracted_by is None,
                        retracted=retracted_by is not None,
                        retracted_by_rank=retracted_by,
                    ),
                ),
            )

        for pp_size in range(2, 9):
            for retracting in range(1, pp_size):
                voided = False
                empty = []
                for rank in range(pp_size):
                    amended = _decision(retracting if rank == retracting else None)
                    voided = pp_pass_should_void(amended, rank, voided)
                    empty.append(voided)
                self.assertEqual(
                    empty,
                    [r >= retracting for r in range(pp_size)],
                    f"pp_size={pp_size}, retraction at {retracting}: the "
                    f"voided ranks must be exactly {retracting}..last",
                )
                for r in range(pp_size - 1):
                    self.assertFalse(
                        empty[r] and not empty[r + 1],
                        f"pp_size={pp_size}, retraction at {retracting}: PP{r} "
                        f"is empty while PP{r + 1} is full -- the H3 state is "
                        f"representable after all: {empty}",
                    )

    def test_the_void_reaches_exactly_the_ranks_that_posted_a_receive(self):
        """THE COVERAGE THEOREM the reverse-wire question turns on.

        Given monotone emptiness, the ranks with a posted receive for a voided
        generation are exactly `0..k-1` for the first retracting rank `k`, and
        `0..pp_size-2` when no rank retracted (every rank but the void's own
        source still ran the pass). `pp_void_relay_stop_rank` must name
        exactly that boundary: one rank short and the successor parks for
        ever, one rank long and an unmatched message is left on the channel --
        #796's law, and the bounded-recv corpse from the sender's side.

        Over every ring size 2..8, where the rule is otherwise pinned on three
        sampled sizes (test_pp_void_send_contract_801.py's
        `TheRelayStopRankIsARingPosition`).
        """
        from sglang.srt.managers.scheduler_pp_mixin import pp_void_relay_stop_rank

        for pp_size in range(2, 9):
            self.assertEqual(
                pp_void_relay_stop_rank(None, pp_size),
                pp_size - 1,
                f"no retraction named at pp_size={pp_size}: every rank but the "
                f"last still ran the pass, so the void travels furthest",
            )
            for k in range(1, pp_size):
                self.assertEqual(
                    pp_void_relay_stop_rank(k, pp_size),
                    k,
                    f"pp_size={pp_size}, first retracting rank {k}: ranks "
                    f"0..{k - 1} hold a launched batch and ranks {k}.. do not",
                )

    def test_the_coverage_theorem_fails_under_each_boundary_mutant(self):
        """THE CAN-FAIL PROOF FOR THE THEOREM ITSELF, on the hazard direction.

        Three mutants, each a plausible edit at this boundary: #797's `None`
        (stop here), one rank short, one rank long. A theorem no mutant breaks
        is a tautology, and this family has already shipped one of those.
        """
        from sglang.srt.managers.scheduler_pp_mixin import pp_void_relay_stop_rank

        def _mutant_pre_801(first, pp_size):
            return None if first is None else int(first)

        def _mutant_one_short(first, pp_size):
            base = pp_void_relay_stop_rank(first, pp_size)
            return None if base is None else max(base - 1, 0)

        def _mutant_one_long(first, pp_size):
            base = pp_void_relay_stop_rank(first, pp_size)
            return None if base is None else base + 1

        def _holds(fn):
            for pp_size in range(2, 9):
                if fn(None, pp_size) != pp_size - 1:
                    return False
                for k in range(1, pp_size):
                    if fn(k, pp_size) != k:
                        return False
            return True

        self.assertTrue(_holds(pp_void_relay_stop_rank), "the shipped rule")
        for name, mutant in (
            ("pre-#801 stop-here", _mutant_pre_801),
            ("one rank short", _mutant_one_short),
            ("one rank long", _mutant_one_long),
        ):
            self.assertFalse(
                _holds(mutant),
                f"mutant '{name}' survived the coverage theorem -- the theorem "
                f"does not constrain this boundary",
            )


if __name__ == "__main__":
    unittest.main()
