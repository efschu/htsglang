"""#944 FALSIFIERS: three real gloo processes, deadline-bounded.

WHY REAL PROCESSES. The claim under test is about a GROUP -- "if one rank
cannot locate a request, the whole ring defers, together, a bounded number of
times, and then resolves". Every word of that is about what more than one
process does with a value that crossed a wire, and an in-process stand-in
cannot distinguish a genuine cross-rank agreement from a convenient no-op. The
ring wire here is the one from test_pp_admission_wiring_791.py: real gloo
point-to-point over the SHIPPED `_pp_send_admission_decision` /
`_pp_recv_admission_decision` / `_pp_reconcile_incoming_admission`, with
`dst`/`src` left at `None` so the ring's own resolution formula runs.

THE THREE FALSIFIERS, and each names what would make it fail.

  (a) A MISS ON ONE RANK.  PP1's request has left every place the lookup knows
      about. Before the fix its miss was a measured `local=0`, the pass voided,
      and the offer was re-made unchanged -- the wedge. After it: PP1 reports
      UNRESOLVED (not a shortfall), the report reaches the last rank verbatim,
      PP0 counts the round, and within `UNRESOLVED_DEFER_CAP` rounds the group
      admits the request. FAILS IF: the miss arrives as a measurement, the
      populations are pooled, or the drive never terminates.

  (b) UNRESOLVED EVERYWHERE, CAP EXHAUSTED.  Nothing ever resolves on any rank.
      This must end in a LOUD, NAMED refusal and a served pass -- never a hang,
      never silence, never an unbounded re-offer. FAILS IF: the run does not
      terminate inside the deadline, or terminates quietly.

  (c) THE DANGER-DIRECTION MUTANT.  Someone turns `observed_local=None` back
      into `0` -- the single edit that reinstates the class. Under the mutant
      PP0 learns a prefix floor from a number NO RANK MEASURED. This falsifier
      asserts the shipped code does not, and that the mutant does, so it is a
      can-fail proof and not a hopeful assertion. FAILS IF: the mutant changes
      nothing, i.e. this file is not measuring what it claims.

DEADLINE-BOUNDED THROUGHOUT: the wedge this ticket is about presents as a hang,
and a test that could hang cannot report one. Every join carries a deadline and
a rank still alive at it is a RESULT (`stuck_ranks`), not a timeout of the
suite.
"""

import json
import logging
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

RID = "rid-944-unresolvable"
PP0_PREFIX = 4096
EXTEND = 64

#: Every rank must be past its work well inside this, or the "never a hang"
#: half of falsifier (b) is not being measured.
JOIN_TIMEOUT_S = 60.0


class _RingWire:
    """Real gloo point-to-point, ring-default -- same shape as
    test_pp_admission_wiring_791.py's, deliberately: `dst=None` /
    `src=None` are resolved with `GroupCoordinator`'s own formula so the
    forwarding path under test is the production one."""

    def __init__(self, rank: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == WORLD - 1
        self.sends = 0

    def send_tensor_dict(
        self, tensor_dict, dst=None, all_gather_group=None, async_send=False
    ):
        import pickle

        self.sends += 1
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        buf = pickle.dumps(tensor_dict)
        size = torch.tensor([len(buf)], dtype=torch.long)
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


class _FakeReq:
    def __init__(self, rid: str, local_match_len: int):
        self.rid = rid
        self.prefix_indices = torch.arange(local_match_len, dtype=torch.int64)

    def init_next_round_input(self, tree_cache):
        pass


def _make_holder(rank: int, wire: _RingWire, waiting_queue):
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=wire,
        ps=types.SimpleNamespace(pp_size=WORLD, pp_rank=rank),
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        waiting_queue=waiting_queue,
        # THE THREE LOOKUPS THAT PRECEDE THE RUNNING BATCH, all empty on
        # purpose: this holder IS the exhausted lookup chain. Left explicit
        # rather than absent so a getattr default cannot quietly stand in for
        # a real miss.
        chunked_req=None,
        running_batch=None,
        tree_cache=None,
    )
    h._pp_boundary_stats = lambda: None
    h._pp_flip_bump_sent = lambda chan: None
    h._pp_flip_bump_consumed = lambda chan: None
    h._pp_flip_bump_attempted = lambda chan: None
    for name in (
        "_pp_send_admission_decision",
        "_pp_recv_admission_decision",
        "_pp_reconcile_incoming_admission",
        "_pp_send_dict_to_next_stage",
        "_pp_recv_typed_dict",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


class _Catcher(logging.Handler):
    def __init__(self, level):
        super().__init__(level=level)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _worker(rank, init_file, out_dir, scenario, mutant):
    """One PP rank. Drives `rounds` full ring laps of ONE rid, where PP1 (and,
    under scenario 'all', every rank) can never resolve it."""
    from sglang.srt.managers import pp_admission_congruence as pac
    from sglang.srt.managers import scheduler_pp_mixin as spm_mod

    res = {"rank": rank, "ok": False, "error": None}
    warn = _Catcher(logging.WARNING)
    err = _Catcher(logging.ERROR)
    logging.getLogger("sglang.srt.managers.scheduler_pp_mixin").addHandler(warn)
    logging.getLogger("sglang.srt.managers.pp_admission_congruence").addHandler(err)

    if mutant:
        # (c) THE ONE EDIT THAT PUTS THE CLASS BACK: the miss goes out in the
        # pre-#944 shape -- a measured `observed_local=0` and no flag saying it
        # was never measured. That is the whole defect, restored in two
        # attribute writes, and it is REACHABLE: it is what reverting the
        # UNRESOLVED branch produces.
        #
        # Everything else still runs -- the sentinel, the fourth lookup, the
        # cap, the told=0 hoist -- so what this arm isolates is exactly the
        # difference between "a miss that says so" and "a miss that reads as a
        # measurement of nothing".
        _real = pac.reconcile_pp_admission_decision

        def _mutated(decision, local_match_lens, **kw):
            effective, amended = _real(decision, local_match_lens, **kw)
            from dataclasses import replace as _replace

            entries = tuple(
                _replace(e, observed_local=0, unresolved=False) if e.unresolved else e
                for e in amended.entries
            )
            return effective, pac.PPAdmissionDecision(
                mb_id=amended.mb_id, entries=entries
            )

        pac.reconcile_pp_admission_decision = _mutated
        spm_mod.reconcile_pp_admission_decision = _mutated

    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        wire = _RingWire(rank)
        # Scenario 'one': only PP1 has lost the request. PP2 holds the whole
        # prefix, so a bug that pooled "unresolved" with "short" would show up
        # as PP2 re-deriving a verdict it is not entitled to.
        if scenario == "one":
            queues = {PP0: [], PP1: [], PP2: [_FakeReq(RID, PP0_PREFIX)]}
        else:
            queues = {PP0: [], PP1: [], PP2: []}
        h = _make_holder(rank, wire, waiting_queue=list(queues[rank]))

        rounds = pac.UNRESOLVED_DEFER_CAP + 3
        res["tolds"], res["served_round"] = [], None

        if rank == PP0:
            guard = pac.PPAdmissionCongruenceGuard()
            h._pp_admission_guard = guard
            for i in range(rounds):
                told = guard.prefix_len_for(RID, PP0_PREFIX)
                res["tolds"].append(told)
                decision = pac.PPAdmissionDecision(
                    mb_id=0,
                    entries=(
                        pac.PPAdmissionEntry(
                            rid=RID,
                            prefix_len=told,
                            extend_len=EXTEND + (PP0_PREFIX - told),
                            admitted=True,
                        ),
                    ),
                )
                h._pp_send_admission_decision(decision)
                # THE RETURN TRIP RIDES THE OUTPUT CHANNEL, NOT THE RING.
                # #796 deleted the wraparound edge -- the last rank posts no
                # send, because PP0 issues no blocking receive for one -- so a
                # test that closed the ring here would hang forever on a
                # message production never emits. The learning travels on the
                # output message instead (`pp_output_payload_with_return_trip`
                # -> `pp_absorb_admission_return`), and BOTH of those, plus the
                # wire codec they use, are the shipped functions. Only the
                # transport under them stands in for the output ring.
                msg = wire.recv_tensor_dict(src=PP2)
                # Decoded once for observation, then handed to the shipped
                # absorber -- which POPS the payload, so the order matters and
                # the copy is not decoration.
                returned = spm_mod.pp_admission_decision_from_wire(dict(msg))
                self_took = spm_mod.pp_absorb_admission_return(h, msg)
                if not self_took:
                    raise AssertionError(
                        "the shipped absorber refused the return trip; the "
                        "guard never learns and nothing below means anything"
                    )
                entry = returned.entries[0]
                res.setdefault("wire", []).append(
                    {
                        "told": told,
                        "retracted": bool(entry.retracted),
                        "unresolved": bool(entry.unresolved),
                        "observed_local": entry.observed_local,
                        "by_rank": entry.retracted_by_rank,
                        # SAMPLED PER ROUND, NOT AT THE END, and that is not
                        # tidiness: `record_return_trip` CLEARS the floor on
                        # the round that finally serves the rid, so an
                        # end-of-drive read shows None whether or not a floor
                        # was ever invented. The claim is "no floor is EVER
                        # learned from a miss", so every round has to be
                        # looked at.
                        "floor": guard.learned_floor(RID),
                        "rounds": guard.unresolved_rounds(RID),
                    }
                )
                if entry.admitted and not entry.retracted:
                    res["served_round"] = i
                    break
            res["floor"] = guard.learned_floor(RID)
            res["unresolved_rounds"] = guard.unresolved_rounds(RID)
            res["escalations"] = [r.getMessage() for r in err.records]
        else:
            for _ in range(rounds):
                incoming = h._pp_recv_admission_decision()
                effective, amended = h._pp_reconcile_incoming_admission(incoming)
                res.setdefault("effective", []).append(sorted(effective))
                res.setdefault("wire", []).append(
                    {
                        "told": incoming.entries[0].prefix_len,
                        "retracted": bool(amended.entries[0].retracted),
                        "unresolved": bool(amended.entries[0].unresolved),
                        "observed_local": amended.entries[0].observed_local,
                        "by_rank": amended.entries[0].retracted_by_rank,
                    }
                )
                if rank == PP2:
                    # THE LAST RANK POSTS NO RING SEND (#796: a rank must not
                    # post a send no peer is required to take), so the shipped
                    # `_pp_send_admission_decision` is a deliberate no-op here
                    # and closing the ring through it would hang PP0 forever.
                    # Production carries the chain-reconciled decision home on
                    # the OUTPUT message instead; that payload builder is
                    # shipped code and is used verbatim.
                    h._pp_admission_amended_by_slot = {0: amended}
                    wire.send_tensor_dict(
                        spm_mod.pp_output_payload_with_return_trip(h, {}, 0),
                        dst=PP0,
                    )
                else:
                    h._pp_send_admission_decision(amended)
                if amended.entries[0].admitted and not amended.entries[0].retracted:
                    break
        res["warnings"] = [r.getMessage() for r in warn.records]
        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:2000]
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


def _run(scenario, mutant=False):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [
            ctx.Process(target=_worker, args=(r, init_file, tmp, scenario, mutant))
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        deadline = time.time() + JOIN_TIMEOUT_S
        for p in procs:
            p.join(timeout=max(0.1, deadline - time.time()))
        stuck = [r for r, p in enumerate(procs) if p.is_alive()]
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

        out = {"stuck_ranks": stuck}
        for r in range(WORLD):
            out[f"result_{r}"] = _load(os.path.join(tmp, f"r{r}.json"))
        return out


def _require_clean(case, res):
    case.assertEqual(
        res["stuck_ranks"],
        [],
        f"a rank never finished -- THIS IS THE WEDGE, measured: {res}",
    )
    for r in range(WORLD):
        got = res[f"result_{r}"]
        case.assertIsNotNone(got, f"PP{r} produced no result: {res}")
        case.assertIsNone(got.get("error"), f"PP{r} raised: {got.get('error')}")
        case.assertTrue(got.get("ok"), f"PP{r} did not complete: {got}")
    return [res[f"result_{r}"] for r in range(WORLD)]


class FalsifierAMissOnOneRank(unittest.TestCase):
    def test_one_ranks_miss_defers_the_group_and_then_resolves(self):
        res = _run("one")
        r0, r1, r2 = _require_clean(self, res)

        self.assertIsNotNone(
            r0["served_round"],
            f"the group never got the request through: tolds={r0['tolds']} "
            f"wire={r0.get('wire')} -- an unbounded defer is the #858 shape",
        )

        # THE MISS TRAVELS AS A MISS. PP1 could not locate the rid; what
        # reaches PP0 says so, and carries no number.
        first = r0["wire"][0]
        self.assertTrue(first["unresolved"], f"the miss lost its identity: {first}")
        self.assertIsNone(
            first["observed_local"],
            "a rank that measured nothing must put nothing on the wire",
        )
        self.assertEqual(first["by_rank"], PP1)

        # ...AND IT IS NOT POOLED WITH A SHORTFALL. The warning PP1 emits is
        # the #944 one, never the #791 one -- two populations, two lines.
        self.assertTrue(
            any("#944 PP-ADMISSION UNRESOLVED" in w for w in r1["warnings"]),
            f"PP1 must report the miss as a miss: {r1['warnings']}",
        )
        self.assertFalse(
            any("#791 PP-ADMISSION unhonourable" in w for w in r1["warnings"]),
            f"a lookup miss must never be reported as a measured shortfall: "
            f"{r1['warnings']}",
        )

        # PP2 holds the whole prefix and is downstream of a retraction, so it
        # must pass the entry through without an opinion.
        self.assertFalse(
            any("PP-ADMISSION" in w for w in r2["warnings"]),
            f"PP2 must not re-derive an already-retracted entry: {r2['warnings']}",
        )

        # NO FLOOR WAS INVENTED, ON ANY ROUND. The clamp that terminates the
        # ordinary shortfall is fed by measurement; there was none to feed it.
        self.assertEqual(
            [w["floor"] for w in r0["wire"]],
            [None] * len(r0["wire"]),
            f"a prefix floor learned from a lookup miss is the class itself: "
            f"{r0['wire']}",
        )
        # The count, on the other hand, must move -- that is the population
        # the cap bounds, and a defer nobody counts is unbounded.
        self.assertEqual(r0["wire"][0]["rounds"], 1)

        # The group admitted it, and every rank agreed on the round it ran.
        served_told = r0["tolds"][r0["served_round"]]
        self.assertEqual(
            r1["effective"][-1],
            [RID],
            f"the rank that could not find it must be the rank that admits it "
            f"once the offer is honourable (told={served_told})",
        )


class FalsifierBCapExhaustionIsLoudNeverAHang(unittest.TestCase):
    def test_unresolved_on_every_rank_ends_in_a_named_refusal(self):
        res = _run("all")
        r0, _r1, _r2 = _require_clean(self, res)

        self.assertIsNotNone(
            r0["served_round"],
            f"cap exhaustion must END the loop, not extend it: {r0['tolds']}",
        )
        self.assertLessEqual(
            r0["served_round"],
            len(r0["tolds"]) - 1,
        )
        self.assertEqual(
            r0["tolds"][-1],
            0,
            f"the terminator is told=0, the only offer honourable without a "
            f"measurement: {r0['tolds']}",
        )

        # LOUD, and it names what the next reader needs.
        self.assertEqual(
            len(r0["escalations"]),
            1,
            f"exactly one refusal -- silence is the wedge and a line per pass "
            f"is the 2106-line log: {r0['escalations']}",
        )
        msg = r0["escalations"][0]
        self.assertIn(RID, msg)
        for place in ("waiting queue", "chunked_req", "slot", "running batch"):
            self.assertIn(
                place, msg, f"the refusal must name every place searched: {msg}"
            )

    def test_the_defer_count_is_bounded_by_the_cap(self):
        from sglang.srt.managers.pp_admission_congruence import UNRESOLVED_DEFER_CAP

        res = _run("all")
        r0, _, _ = _require_clean(self, res)
        self.assertEqual(
            r0["served_round"],
            UNRESOLVED_DEFER_CAP,
            f"the bound must be the cap, exactly, not 'eventually': {r0['tolds']}",
        )


class FalsifierCTheDangerDirectionMutant(unittest.TestCase):
    """CAN-FAIL PROOF. Falsifier (a) asserts PP0 learns no floor from a miss.
    That assertion is only worth something if some reachable edit would break
    it -- otherwise it passes for a reason unrelated to the fix."""

    def test_turning_the_miss_back_into_a_zero_is_caught(self):
        clean = _run("one")
        mutated = _run("one", mutant=True)

        c0 = clean["result_0"]
        m0 = mutated["result_0"]
        self.assertIsNotNone(m0, f"the mutant run produced nothing: {mutated}")
        self.assertIsNone(m0.get("error"), f"mutant harness broke: {m0.get('error')}")

        c_floors = [w["floor"] for w in c0["wire"]]
        m_floors = [w["floor"] for w in m0["wire"]]
        self.assertEqual(
            c_floors,
            [None] * len(c_floors),
            "shipped: no floor is learned from a miss, on any round",
        )
        self.assertIn(
            0,
            m_floors,
            "MUTANT NOT CAUGHT. Rewriting the miss as `observed_local=0` is "
            "the single edit that reinstates the class, and it must show up "
            "as PP0 learning a prefix floor of 0 -- a clamp derived from a "
            "number no rank measured. If no round shows one, the falsifier "
            f"above is passing for some other reason: {m_floors}",
        )
        self.assertEqual(
            [w["rounds"] for w in m0["wire"]],
            [0] * len(m_floors),
            "and the mutant must also lose the COUNT, since the population it "
            "erases is the one the cap bounds -- which is why the two halves "
            "(the field and the cap) had to land together",
        )
        # AND THE MUTANT STILL TERMINATES, which is the trap. The false 0 IS a
        # clamp: it drops the next offer to told=0 and the pass runs, one round
        # sooner than the honest bound does. That is why this defect survived
        # #797c and #798 -- it looks like it is working. Only the PROVENANCE of
        # the clamp distinguishes them, so only the provenance can be asserted.
        self.assertIsNotNone(m0["served_round"])
        self.assertLess(
            m0["served_round"],
            c0["served_round"],
            "if the mutant were merely slower this would be a performance "
            "note; it is faster, and wrong, which is what makes 'it terminated' "
            "worthless as evidence here",
        )


if __name__ == "__main__":
    unittest.main()
