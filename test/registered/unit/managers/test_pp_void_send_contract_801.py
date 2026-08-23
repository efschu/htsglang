"""#801: a void that stops at PP0 parks every rank between PP0 and its origin.

THE SPECIMEN (wedge_802f_1712, evidence-665-f1, PP=3 with --enable-phase-flip,
py-spy of all three ranks; quoted in `_pp_wait_for_dict_readiness`' own
docstring):

    PP0  _pp_commit_comm_work <- _pp_commit_pending_req_work
    PP1  _pp_recv_dict_from_prev_stage <- _do_recv    BLOCKED HERE
    PP2  PpChainReceiver.recv <- recv_requests

PP1 sits in the intermediate output receive, which is UNCONDITIONAL once its
own slot is non-empty, while the send that would satisfy it is CONDITIONAL on
state PP1 does not publish and PP0 cannot see.

THE CHANNEL ASYMMETRY, AND WHERE #797 LEFT IT OPEN. `_pp_send_output_to_next_
stage` has three send arms. The last rank's real-output arm and its #791b void
arm are matched by construction -- both consult `_pp_output_expected_for_slot`,
PP0's published verdict for the slot. The intermediate arm is not: it forwards
on `if pp_outputs:`. #797 made a void reach that arm by republishing it out of
`_do_recv`, but only through `pp_void_forward_payload`, which answered "stop
here" for BOTH states `pp_first_retracting_rank` cannot distinguish:

    (a) the void carries a decision that names no retracting rank
    (b) the void carries no decision at all

Both are states in which every intermediate rank still RAN the pass and still
has a receive posted. Reading them as "stop" is what parks PP1.

THE FIX IS A DEFAULT, NOT A BRANCH. `pp_void_relay_stop_rank` turns the two
absences into a ring position -- `pp_size - 1`, one hop short of the void's own
source, which is the only rank whose slot was empty -- so the ring becomes a
strict relay: a non-last rank that took one message off this wire for a
generation puts exactly one back on it, void included. The named-retraction
stop is unchanged; a hop past `first_retracting_rank - 1` would still be an
unmatched message, which is #796's law and is separately pinned below.

THE HAZARD IS REAL DATA, NOT A WIRE TRICK, and it is a divergence this corpus
already measured. On the hazard pass the LAST rank's waiting queue does not
hold the request the others hold -- the `#new-seq 1 vs 3` shape of the 0516
specimen (COORD-strand16f-801-build.md B.10.3), where rank-local admission on
diverged queue heads leaves one rank with an empty slot. Every prefix is told
as 0, so the SHIPPED `reconcile_pp_admission_decision` retracts nothing and the
void it produces names no retracting rank: state (a), reached without touching
the transport, the codec or the retraction path.

WHY REAL PROCESSES OVER A MOCK. The defect is that a blocking gloo receive is
entered with nothing coming. A mock returns; only a real peer fails to send.
The transport is #795's `_RingWire` verbatim and the holder is
test_pp_output_ring_retraction_wedge_791b's, unchanged -- every method that
touches the wire, the decision codec, the void absorption and the relay rule is
shipped code.

CASES:
  TheRelayStopRankIsARingPosition          STRUCTURAL, pure
  AVoidWithNoNamedRetractionStillTravels   RED before #801, pure
  TheNamedRetractionStopIsUnchanged        GREEN, pure -- #796's law kept
  TheRingSurvivesAVoidWithNoRetraction     RED before #801, 3 real processes
  TheRelayCannotSpin                       REVERSE CORPSE -- the 1716 shape
"""

import json
import os
import tempfile
import time
import types
import unittest

import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120)

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2
LAST = WORLD - 1
LOOP = WORLD

#: Every prefix is told as zero, so no rank can ever be short of it and
#: `reconcile_pp_admission_decision` retracts nothing anywhere in this file.
#: That is deliberate: the void this file is about is the one produced WITHOUT
#: a retraction, which is precisely the state #797 could not name.
TOLD = 0

#: The pass on which the LAST rank's waiting queue diverges from the others'.
#: Late enough that the ring is in steady state, early enough to leave passes
#: behind it.
HAZARD_PASS = 4

#: THE LAST PASS THAT CARRIES WORK, AND IT IS THE HAZARD ONE -- the same
#: reasoning test_pp_output_ring_retraction_wedge_791b.py records for its own
#: constant, and it is not harness convenience. A missing output in the MIDDLE
#: of a busy stretch is only a lag: the pair is a FIFO and the next pass's
#: message releases the receiver. It becomes a permanent deficit exactly when
#: the pipeline goes idle behind it.
LAST_BUSY_PASS = HAZARD_PASS

N_PASSES = 12

PAYLOAD = [b"x" * 65536]

GREEN_JOIN_TIMEOUT_S = 60.0

#: The wedged arm never completes by construction, so there is no timing
#: window to lose -- this only has to outlast spawn and gloo init.
RED_JOIN_TIMEOUT_S = 25.0


def _rid_for(pass_idx: int) -> str:
    return f"rid{pass_idx:04d}"


def _last_rank_holds_the_request(rank: int, pass_idx: int) -> bool:
    """The divergence, as data: on HAZARD_PASS the last rank's waiting queue
    does not hold the request PP0 and PP1 hold. Everywhere else the three
    queues agree."""
    return not (rank == LAST and pass_idx == HAZARD_PASS)


def _worker(rank, init_file, out_dir, n_passes):
    """One rank's driver over the three channels that close the ring:
    request chain, admission decision, output.

    Structurally test_pp_output_ring_retraction_wedge_791b.py's worker; what
    differs is only the hazard -- a diverged waiting queue instead of a cold
    prefix cache -- and the bookkeeping of voids absorbed against voids
    forwarded, which is what `TheRelayCannotSpin` reads.
    """
    import test_pp_admission_chain_flush_deadlock_795 as ring
    import test_pp_output_ring_retraction_wedge_791b as ring791b
    import torch
    import torch.distributed as dist

    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionDecision,
        PPAdmissionEntry,
    )
    from sglang.srt.managers.scheduler_pp_mixin import _pp_output_exchange_due
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    res = {
        "rank": rank,
        "ok": False,
        "error": None,
        "passes": 0,
        "voids_absorbed": 0,
        "voids_forwarded": 0,
    }
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        chain_group = dist.new_group(ranks=list(range(WORLD)))
        wire = ring._RingWire(rank)
        h = ring791b._make_holder(rank, wire, chain_group, out_dir)

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
            if busy and _last_rank_holds_the_request(rank, p):
                h.waiting_queue.append(ring791b._StubReq(rid, TOLD))

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

            admitted = [r for r in h.waiting_queue if r.rid in effective]
            for req in admitted:
                h.waiting_queue.remove(req)
            h.mbs[mb_id] = ring791b._StubBatch(admitted) if admitted else None

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
            slot_before = h.mbs[next_mb_id]
            npo, nbr, d2h = h._pp_commit_send_output_work_and_preprocess_output_tensors(
                next_first_rank_mb_id, next_mb_id
            )
            # A void is the one outcome that empties an occupied slot without
            # producing a result. That is `_pp_absorb_void_output`'s contract
            # and it is how this driver counts one, without reaching into the
            # method's internals.
            if slot_before is not None and h.mbs[next_mb_id] is None and nbr is None:
                res["voids_absorbed"] += 1
                if npo is not None:
                    res["voids_forwarded"] += 1
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


def _pre_801_worker(rank, init_file, out_dir, n_passes):
    """The SAME driver with ONLY the #801 default neutered, IN THE CHILD.

    THE SPAWN TRAP, restated because this lineage has paid for it twice
    (#796, then #797): `_run` uses the "spawn" start method, so a patch
    applied inside a test METHOD reaches no child. It has to happen here.

    The replacement is the pre-#801 rule verbatim -- a stop position only
    when a retraction is named, and `None`, meaning "stop here", otherwise.
    Nothing else is touched: not the transport, not the codec, not the
    absorption, not the last rank's void arm.
    """
    from sglang.srt.managers import scheduler_pp_mixin as m

    m.pp_void_relay_stop_rank = lambda first, pp_size: (
        None if first is None else int(first)
    )
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
            if os.path.exists(path):
                with open(path) as f:
                    out[r] = json.load(f)
        return out


class TheRelayStopRankIsARingPosition(unittest.TestCase):
    """STRUCTURAL. The decision is a pure function of two numbers, so a
    mutant that disables it cannot hide behind a source-level grep -- the
    #823 lesson (`prefetch_ballot.advance_mismatch_streak`) applied here."""

    def test_a_named_retraction_is_the_stop_position(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_void_relay_stop_rank

        for first in (1, 2, 5):
            self.assertEqual(pp_void_relay_stop_rank(first, WORLD), first)

    def test_no_named_retraction_stops_one_hop_short_of_the_source(self):
        """The void's source is the LAST rank -- its own slot is what was
        empty -- so it is the one rank that will not receive."""
        from sglang.srt.managers.scheduler_pp_mixin import pp_void_relay_stop_rank

        self.assertEqual(pp_void_relay_stop_rank(None, WORLD), WORLD - 1)
        self.assertEqual(pp_void_relay_stop_rank(None, 2), 1)
        self.assertEqual(pp_void_relay_stop_rank(None, 1), 0)

    def test_an_unknown_ring_size_keeps_the_pre_801_behaviour(self):
        """No production Scheduler is in that state; a stand-in that lacks
        `ps.pp_size` gets the old answer rather than a guessed ring."""
        from sglang.srt.managers.scheduler_pp_mixin import pp_void_relay_stop_rank

        self.assertIsNone(pp_void_relay_stop_rank(None, None))
        self.assertEqual(pp_void_relay_stop_rank(2, None), 2)


def _holder(pp_rank, is_last, pp_size=WORLD):
    return types.SimpleNamespace(
        pp_group=types.SimpleNamespace(is_last_rank=is_last),
        ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size),
    )


def _void_without_retraction():
    """A void carrying a real decision in which nothing is retracted -- the
    state `_pp_send_output_to_next_stage`'s void arm produces whenever the
    last rank simply has no batch for a slot PP0 expects an output for."""
    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionDecision,
        PPAdmissionEntry,
    )
    from sglang.srt.managers.scheduler_pp_mixin import (
        _PP_VOID_OUTPUT_KEY,
        pp_admission_decision_to_wire,
    )

    payload = {_PP_VOID_OUTPUT_KEY: True}
    payload.update(
        pp_admission_decision_to_wire(
            PPAdmissionDecision(
                mb_id=0,
                entries=(PPAdmissionEntry(rid="a" * 32, prefix_len=0, extend_len=1),),
            )
        )
    )
    return payload


class AVoidWithNoNamedRetractionStillTravels(unittest.TestCase):
    """RED before #801. Both arms returned None, and None on a ring is a
    parked successor."""

    def test_a_void_naming_no_retraction_reaches_every_rank_that_launched(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            _PP_VOID_OUTPUT_KEY,
            pp_admission_decision_from_wire,
            pp_first_retracting_rank,
            pp_void_forward_payload,
        )

        void = _void_without_retraction()
        # The premise, pinned: this really is the state #797 could not name.
        self.assertIsNone(
            pp_first_retracting_rank(pp_admission_decision_from_wire(void))
        )

        fwd = pp_void_forward_payload(_holder(PP0, False), void)
        self.assertIsNotNone(fwd, "PP1 would block for ever without this hop")
        self.assertTrue(fwd[_PP_VOID_OUTPUT_KEY])
        # And it stops one hop short of its own source, which does not receive.
        self.assertIsNone(pp_void_forward_payload(_holder(PP1, False), fwd))

    def test_a_void_carrying_no_decision_at_all_travels_the_same_way(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            _PP_VOID_OUTPUT_KEY,
            pp_void_forward_payload,
        )

        bare = {_PP_VOID_OUTPUT_KEY: True}
        fwd = pp_void_forward_payload(_holder(PP0, False), bare)
        self.assertIsNotNone(fwd, "an absent decision is not evidence of a stop")
        self.assertTrue(fwd[_PP_VOID_OUTPUT_KEY])
        self.assertIsNone(pp_void_forward_payload(_holder(PP1, False), fwd))

    def test_a_stand_in_without_a_ring_size_is_unchanged(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            _PP_VOID_OUTPUT_KEY,
            pp_void_forward_payload,
        )

        h = types.SimpleNamespace(
            pp_group=types.SimpleNamespace(is_last_rank=False),
            ps=types.SimpleNamespace(pp_rank=PP0),
        )
        self.assertIsNone(
            pp_void_forward_payload(h, {_PP_VOID_OUTPUT_KEY: True}),
        )


class TheNamedRetractionStopIsUnchanged(unittest.TestCase):
    """GREEN, and it is #796's law: a rank must not post a send no peer is
    required to take. Ranks from the retraction down have empty slots and
    receives that early-return, so the stop is still a stop."""

    def _void(self, retracted_by):
        from sglang.srt.managers.pp_admission_congruence import (
            PPAdmissionDecision,
            PPAdmissionEntry,
        )
        from sglang.srt.managers.scheduler_pp_mixin import (
            _PP_VOID_OUTPUT_KEY,
            pp_admission_decision_to_wire,
        )

        payload = {_PP_VOID_OUTPUT_KEY: True}
        payload.update(
            pp_admission_decision_to_wire(
                PPAdmissionDecision(
                    mb_id=0,
                    entries=(
                        PPAdmissionEntry(
                            rid="b" * 32,
                            prefix_len=512,
                            extend_len=1,
                            admitted=False,
                            retracted=True,
                            retracted_by_rank=retracted_by,
                            observed_local=0,
                        ),
                    ),
                )
            )
        )
        return payload

    def test_a_retraction_on_the_first_downstream_rank_still_stops_at_zero(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_void_forward_payload

        self.assertIsNone(pp_void_forward_payload(_holder(PP0, False), self._void(1)))

    def test_a_retraction_further_down_still_reaches_the_ranks_that_ran(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_void_forward_payload

        fwd = pp_void_forward_payload(_holder(PP0, False), self._void(2))
        self.assertIsNotNone(fwd)
        self.assertIsNone(pp_void_forward_payload(_holder(PP1, False), fwd))

    def test_the_last_rank_never_forwards(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_void_forward_payload

        self.assertIsNone(pp_void_forward_payload(_holder(LAST, True), self._void(2)))
        self.assertIsNone(
            pp_void_forward_payload(_holder(LAST, True), _void_without_retraction())
        )

    def test_the_last_rank_refuses_even_a_stop_position_beyond_the_ring(self):
        """THE ONE STATE IN WHICH THE RANK IDENTITY IS NOT REDUNDANT, and it
        is a measured hole rather than a hypothetical: with the #801
        arithmetic in place, a mutant that deletes `is_last_rank` from
        `pp_void_forward_payload` survives every other arm in this file --
        `rank + 1 >= stop` already answers for the last rank whenever the
        stop position lies inside the ring. It stops answering when the
        carried decision names a rank that does NOT, which a stale or
        cross-epoch decision can do (the #795 mispair family is exactly
        decisions outliving their generation).

        A last rank forwarding there would post a message no peer takes --
        #796's law, and the corpse `_pp_send_dict_to_next_stage` already
        refuses for the proxy. So the identity check stays, and this arm is
        what makes it reachable instead of merely present (the #824 lesson:
        measuring presence is not measuring reachability).
        """
        from sglang.srt.managers.scheduler_pp_mixin import pp_void_forward_payload

        beyond = self._void(WORLD + 2)
        self.assertIsNone(
            pp_void_forward_payload(_holder(PP0, True, pp_size=1), beyond),
            "the last rank forwarded a void because a stale decision named a "
            "stop position outside its own ring",
        )


class TheRingSurvivesAVoidWithNoRetraction(unittest.TestCase):
    """Three real processes over real gloo. The RED arm is the pre-#801 rule
    and nothing else."""

    def test_the_ring_completes(self):
        res = _run(_worker, N_PASSES, GREEN_JOIN_TIMEOUT_S)
        self.assertEqual(
            res["stuck_ranks"],
            [],
            f"ranks still running at the deadline: {res.get('stall_report')}",
        )
        for r in range(WORLD):
            row = res.get(r, {})
            self.assertIsNone(row.get("error"), f"rank {r} failed: {row.get('error')}")
            self.assertTrue(row.get("ok"), f"rank {r} did not finish: {row}")
            self.assertEqual(row.get("passes"), N_PASSES, f"rank {r}: {row}")

    def test_pre_801_the_intermediate_rank_parks_in_the_output_receive(self):
        res = _run(_pre_801_worker, N_PASSES, RED_JOIN_TIMEOUT_S)
        self.assertTrue(
            res["stuck_ranks"],
            "the pre-#801 rule completed the run -- the hazard did not "
            f"produce a void that stops at PP0: {res}",
        )
        stalls = res.get("stall_report", {})
        self.assertIn(
            PP1,
            res["stuck_ranks"],
            f"PP1 is the rank the specimen shows parked; stalls={stalls}",
        )
        self.assertIn(
            "output_exchange",
            stalls.get(PP1, ""),
            f"PP1 stalled somewhere other than the output exchange: {stalls}",
        )


class TheRelayCannotSpin(unittest.TestCase):
    """REVERSE CORPSE -- the 1716 void storm, and this fix may not revive it.

    On boot 1716 PP1 logged 2353 `#797d own pass voided on slot` and 2353
    `#798 pass voided on slot` against THREE real retractions -- 4709 voids
    on 3, factor 1570, where a healthy boot runs 5 on 5. A relay that
    re-derived a void per pass instead of per received message would be that
    shape again: a hang traded for a spin, which is still a dead server and
    is harder to see because the rank looks busy.

    The invariant that makes it impossible is one-in-one-out, and this reads
    it as a ratio rather than trusting the argument.
    """

    def test_every_forwarded_void_answers_exactly_one_absorbed_void(self):
        res = _run(_worker, N_PASSES, GREEN_JOIN_TIMEOUT_S)
        self.assertEqual(res["stuck_ranks"], [], f"{res}")
        for r in range(WORLD):
            row = res.get(r, {})
            absorbed = row.get("voids_absorbed", 0)
            forwarded = row.get("voids_forwarded", 0)
            self.assertLessEqual(
                forwarded,
                absorbed,
                f"rank {r} put more voids on the wire than it took off it "
                f"({forwarded} > {absorbed}) -- the 1716 spin shape: {row}",
            )
            self.assertLessEqual(
                absorbed,
                1,
                f"rank {r} absorbed {absorbed} voids for ONE diverged pass "
                f"over {N_PASSES} passes -- a void that regenerates itself "
                f"is the 1716 storm, not a relay: {row}",
            )

    def test_the_void_is_absorbed_by_the_ranks_that_ran_and_by_no_others(self):
        """The relay's reach, measured: PP0 and PP1 held a launched batch for
        the diverged slot, the last rank is the void's own source."""
        res = _run(_worker, N_PASSES, GREEN_JOIN_TIMEOUT_S)
        self.assertEqual(res["stuck_ranks"], [], f"{res}")
        self.assertEqual(res.get(PP0, {}).get("voids_absorbed"), 1, f"{res.get(PP0)}")
        self.assertEqual(res.get(PP0, {}).get("voids_forwarded"), 1, f"{res.get(PP0)}")
        self.assertEqual(res.get(PP1, {}).get("voids_absorbed"), 1, f"{res.get(PP1)}")
        self.assertEqual(res.get(PP1, {}).get("voids_forwarded"), 0, f"{res.get(PP1)}")
        self.assertEqual(res.get(LAST, {}).get("voids_absorbed"), 0, f"{res.get(LAST)}")


if __name__ == "__main__":
    unittest.main()
