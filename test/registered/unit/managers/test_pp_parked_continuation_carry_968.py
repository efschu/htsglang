"""#968: the parked chunked continuation rides the #791b return lap home.

THE SPECIMEN -- window-flip-0828 boot 3, pin 3c470fb8f1, Arm A, ~2 minutes,
514 refusals and a dead server.

After a #797 void, PP1 re-homed a #971-parked chunked continuation
(rid `eaa82bcc...`, executed 4096 of 8422) into `self.chunked_req`
(`pp_rehome_refused_chunked_req`, scheduler.py's refusal exit). That request
is then in NO queue on ANY rank -- `add_chunked_req` re-admits it from
`self.chunked_req` directly -- so PP0 could not see it and kept naming two
OTHER rids per decision, drawn from a queue tail the void's requeue kept
rotating. PP1's `add_chunked_req` meanwhile appended its continuation to
`can_run_list` unconditionally (schedule_policy.py:1521), so PP1's batch and
PP0's decision could not agree:

    514 UNEXECUTABLE on PP1 = 191 "the decision names 2 request(s) and this
    rank's admission loop reached only 2; missing rid(s)=..."
                            + 323 "this rank admitted rid(s)=eaa82bcc...,
    which the decision does not name"

    -> 512 voids -> the #801-spin guard killed the boot.

Measured against the same slice: PP0 built 518 ADMIT decisions and NAMED
`eaa82bcc` in exactly 2 of them.

WHY A SEAT FIX IS NOT THE FIX (determination, pinned here so it is not
rebuilt). Even with a correctly priced uniform cap, PP1's continuation would
still be an UN-NAMED EXTRA in its batch, and the `extra` raise is as correct
as the `missing` one -- an unnamed request breaks the cross-stage tensor
pairing exactly as a dropped one does. The follower's continuation can never
legally run until PP0 NAMES it. That is the one junction, and both halves of
the boot-3 defect collapse onto it.

WHAT THIS HARNESS DRIVES FOR REAL -- the whole lap, not a hand-called absorb.
A falsifier that supplies the delivery cannot test the delivery (#944's
lesson), so the fact is put on the wire by the shipped sender, relayed by the
shipped relay, and taken off by the shipped absorber:

    PP1  `_pp_send_admission_decision`      (real, shipped)
      -> PP2  `_pp_recv_admission_decision` (real, shipped)
      -> PP2  `pp_output_payload_with_return_trip`  (real, shipped)
      -> PP0  `pp_absorb_admission_return`  (real, shipped)
      -> PP0  `pp_parked_continuation_priority`     (real, shipped)
      -> PP0  `build_pp_admission_decision`         (real, shipped)

Only the two transports are stood in (a captured dict instead of gloo) and
only the 900-line `_get_new_batch_prefill_raw` is stood in for the membership
verdict -- `TheHarnessMatchesProduction` below pins that block's shape
against the shipped source, so the stand-in cannot drift from what it claims
to represent.
"""

import inspect
import types
import unittest
from array import array

import torch

from sglang.srt.managers import scheduler_pp_mixin as m
from sglang.srt.managers.pp_admission_congruence import (
    PPAdmissionCongruenceGuard,
    PPAdmissionDecision,
    build_pp_admission_decision,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2
MB_ID = 1

# The specimen's own rids and geometry, kept verbatim so a reader can find
# them in the boot-3 slice.
RID_PARKED = "eaa82bcc718a4ff8bfa8cdd43e737155"
RID_A = "c121b2f051de40729e9dd59855e58378"
RID_B = "e423ea58bb7e46fc8a3be75b26ca786a"
EXECUTED = 4096
TOTAL = 8422
# PP0's uniform seat count in the specimen: every decision named 2.
CAP = 2


def _req(rid, *, prefix_len, extend_len):
    """A real `Req` in the state the adder leaves it in.

    `Req.__new__` rather than a stand-in class: `build_pp_admission_decision`
    reads `extend_range` through the shipped `_executed_extent`, and
    `prefix_indices` must be a real TENSOR (#796 -- the shipped readers take
    `len()` of it and must never put it in a boolean context). A stub would
    answer the question with the harness's arithmetic instead of the
    product's.
    """
    req = Req.__new__(Req)
    req.rid = rid
    fill = list(range(TOTAL))
    req.origin_input_ids = array("q", fill)
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", fill)
    req.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)
    req.extend_range = Range(prefix_len, prefix_len + extend_len)
    req.req_pool_idx = 0
    req.inflight_middle_chunks = 1
    req.cache_protected_len = prefix_len
    return req


def _parked_req(rid=RID_PARKED, executed=EXECUTED):
    """The #971-parked continuation: `extend_range(prefix, prefix)`.

    This is `add_chunked_req`'s #679 park shape verbatim
    (schedule_policy.py:1399) -- the range ENDS where the prefix ends, so the
    request has run 4096 tokens and has no chunk in flight. That equality is
    exactly what `pp_parked_continuation_fact` keys on.
    """
    return _req(rid, prefix_len=executed, extend_len=0)


def _holder(rank, *, chunked_req=None, is_last=None):
    """A scheduler stand-in carrying exactly what the lap reads.

    The #630/#757/#795/#797/#971 pattern this directory uses throughout: a
    bare `SimpleNamespace` with the SHIPPED methods bound on via
    `types.MethodType`, so the code under test is the shipped code.
    """
    if is_last is None:
        is_last = rank == WORLD - 1
    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=rank, pp_size=WORLD),
        pp_group=types.SimpleNamespace(is_first_rank=(rank == 0), is_last_rank=is_last),
        chunked_req=chunked_req,
        waiting_queue=[],
        pp_loop_size=WORLD,
        _pp_admission_guard=None,
        _pp_admission_send_work=[],
        _pp_admission_amended_by_slot=[None] * WORLD,
        _pp_launched_chain_by_slot=[None] * WORLD,
        sent=[],
    )
    h._pp_note_launched_chain = types.MethodType(
        m.SchedulerPPMixin._pp_note_launched_chain, h
    )
    # The transport, and the ONLY thing stood in on the send side: the
    # shipped method builds the dict and hands it to this.
    h._pp_send_dict_to_next_stage = lambda td, async_send=True, msg_type=None: (
        h.sent.append(dict(td)) or []
    )
    h._pp_send_admission_decision = types.MethodType(
        m.SchedulerPPMixin._pp_send_admission_decision, h
    )
    return h


def _receiver(rank, *, chunked_req=None, incoming=None):
    """A rank about to take a decision message off the wire."""
    h = _holder(rank, chunked_req=chunked_req)
    h._pp_recv_typed_dict = lambda expected_kind=None: dict(incoming or {})
    h._pp_recv_admission_decision = types.MethodType(
        m.SchedulerPPMixin._pp_recv_admission_decision, h
    )
    return h


def _empty_decision(mb_id=MB_ID):
    return PPAdmissionDecision(mb_id=mb_id, entries=())


def _drive_lap(parked_on_pp1=True):
    """Drive ONE full ring lap and return PP0 after it has absorbed.

    Every step below is the shipped function. Nothing in this harness writes
    a parked fact onto a message or into a table by hand -- that is the whole
    point (#944: a falsifier that supplies the delivery cannot test it).
    """
    # 1. PP1 holds the parked continuation and sends its decision downstream.
    pp1 = _holder(PP1, chunked_req=_parked_req() if parked_on_pp1 else None)
    pp1._pp_send_admission_decision(
        _empty_decision(), expects_output=True, launched=True
    )
    assert pp1.sent, "the shipped sender put nothing on the wire"
    downstream = pp1.sent[-1]

    # 2. PP2 (the last rank) takes it off the wire and relays.
    pp2 = _receiver(PP2, incoming=downstream)
    pp2._pp_recv_admission_decision()

    # 3. PP2 builds the slot's void output -- the #791b return trip.
    home = m.pp_output_payload_with_return_trip(
        pp2, {"__pp_void_output__": True}, MB_ID
    )
    # Snapshotted BEFORE step 4, because the shipped absorber POPS the key it
    # consumes -- what is left of that dict becomes a `PPProxyTensors` and is
    # forwarded on, and a tuple of triples left in it would slice to nonsense
    # rather than raise. Asserting on the post-absorb dict would therefore be
    # asserting that the absorber failed to clean up after itself.
    on_the_wire = dict(home)

    # 4. PP0 absorbs it.
    pp0 = _holder(PP0)
    m.pp_absorb_admission_return(pp0, home)
    return pp0, pp1, pp2, downstream, on_the_wire, home


def _pp0_pass(pp0, *, queue_rids, cap=CAP):
    """PP0's next pass: reorder, admit up to `cap`, build the decision.

    The admission LOOP itself is stood in (it is 900 lines and needs a live
    pool); what it is stood in for is a seat-limited walk of the queue in
    order, which is what the shipped loop does once
    `pp_parked_continuation_priority` has run. Both the reorder and the
    decision build are the shipped functions.
    """
    pp0.waiting_queue = [
        _req(rid, prefix_len=EXECUTED if rid == RID_PARKED else 0, extend_len=64)
        for rid in queue_rids
    ]
    moved = m.pp_parked_continuation_priority(pp0)
    can_run_list = pp0.waiting_queue[:cap]
    guard = PPAdmissionCongruenceGuard()
    decision = build_pp_admission_decision(
        MB_ID,
        can_run_list,
        pp_size=WORLD,
        guard=guard,
        require_executed_geometry=True,
    )
    return decision, moved


def _membership_verdict(named_rids, admitted_rids):
    """The shipped check's verdict, in the shipped ORDER: missing, then extra.

    Stands in for scheduler.py's `#791 CORE: EVERY NAMED REQUEST, OR NONE OF
    THEM` block, which cannot be driven hermetically (it sits inside
    `_get_new_batch_prefill_raw`). `TheHarnessMatchesProduction` pins that
    block's shape against the shipped source, so this cannot drift.
    """
    missing = sorted(r for r in named_rids if r not in admitted_rids)
    if missing:
        return "missing", missing
    extra = sorted(r for r in admitted_rids if r not in named_rids)
    if extra:
        return "extra", extra
    return "ok", []


def _follower_batch(named_rids, *, parked_rid=RID_PARKED, cap=CAP):
    """What PP1 actually builds: its parked continuation, then named rids.

    `add_chunked_req` appends the continuation UNCONDITIONALLY and PP1 cannot
    decline it (schedule_policy.py:1521); the loop then admits named rids
    until the seat cap is reached. This is the boot-3 shape verbatim.
    """
    admitted = [parked_rid]
    for rid in named_rids:
        if len(admitted) >= cap:
            break
        if rid == parked_rid:
            continue
        admitted.append(rid)
    return admitted


class Arm1TheBoot3Loop(unittest.TestCase):
    """The loop, and its end."""

    def test_the_fact_reaches_pp0_over_the_real_lap(self):
        pp0, _pp1, _pp2, downstream, on_the_wire, home = _drive_lap()

        self.assertIn(
            m._PP_PARKED_CONTINUATION_KEY,
            downstream,
            "PP1's shipped decision send must put its parked fact on the "
            "message that already travels downstream every pass -- the "
            "middle rank has no other way onto the lap, because the output "
            "goes home in ONE hop from the last rank",
        )
        self.assertIn(
            m._PP_PARKED_CONTINUATION_KEY,
            on_the_wire,
            "the last rank must RELAY what it learned, not only what it "
            "holds: PP1 parks, PP2 originates the home-bound message, PP0 "
            "is the only consumer",
        )
        self.assertEqual(
            m.pp_parked_continuation_carry(pp0),
            {RID_PARKED: (EXECUTED, PP1)},
            "PP0 must know the rid, the executed extent, and which rank holds it",
        )

    def test_pp0_names_it_within_one_pass_and_the_follower_builds(self):
        pp0, _pp1, _pp2, _downstream, _wire, _home = _drive_lap()

        # The specimen's queue: the parked rid rotated to the TAIL, two
        # foreign rids ahead of it, and only CAP seats.
        decision, moved = _pp0_pass(pp0, queue_rids=[RID_A, RID_B, RID_PARKED])

        self.assertEqual(moved, (RID_PARKED,))
        named = [e.rid for e in decision.entries]
        self.assertIn(
            RID_PARKED,
            named,
            "PP0 must NAME the parked continuation -- until it does, the "
            "follower's occupant is an unnamed extra and the pass cannot "
            "be built by anyone",
        )
        told = {e.rid: e.prefix_len for e in decision.entries}[RID_PARKED]
        self.assertEqual(
            told,
            EXECUTED,
            "told must report the extent the holder ACTUALLY executed: a "
            "value derived from anything else names a pass no rank ran",
        )

        verdict, offenders = _membership_verdict(named, _follower_batch(named))
        self.assertEqual(
            (verdict, offenders),
            ("ok", []),
            "with the continuation named, the follower's add_chunked_req "
            "occupant IS a named member: no extra, no missing",
        )

    def test_without_the_carry_the_rid_is_never_named_and_the_pass_refuses(self):
        """The BEFORE shape, pinned: an empty table reproduces boot 3."""
        pp0 = _holder(PP0)
        self.assertEqual(m.pp_parked_continuation_carry(pp0), {})

        refusals = []
        for _ in range(8):
            decision, moved = _pp0_pass(pp0, queue_rids=[RID_A, RID_B, RID_PARKED])
            named = [e.rid for e in decision.entries]
            self.assertEqual(moved, ())
            self.assertNotIn(RID_PARKED, named)
            refusals.append(_membership_verdict(named, _follower_batch(named))[0])

        # BOTH shipped forms are the boot-3 shape, and which one fires is a
        # property of where the seat cap cuts, not of the defect: with two
        # seats and the continuation occupying one, a named rid is also
        # dropped, so `missing` raises first (the slice: 191 missing, 323
        # extra, 514 together). What must NEVER appear is "ok".
        self.assertTrue(
            set(refusals) <= {"missing", "extra"},
            f"unexpected verdicts: {sorted(set(refusals))}",
        )
        self.assertNotIn(
            "ok",
            refusals,
            "every pass must refuse while the rid is unnamed -- that is the "
            "514-refusal loop, and it is why the fact has to travel",
        )


class Arm2Legacy(unittest.TestCase):
    """A message with no stamp is the pre-#968 message, byte for byte."""

    def test_no_parked_continuation_means_no_key_anywhere(self):
        _pp0, pp1, _pp2, downstream, on_the_wire, home = _drive_lap(parked_on_pp1=False)
        self.assertNotIn(m._PP_PARKED_CONTINUATION_KEY, downstream)
        self.assertNotIn(m._PP_PARKED_CONTINUATION_KEY, on_the_wire)
        self.assertIsNone(m.pp_parked_continuation_fact(pp1))

    def test_payload_is_returned_unchanged_when_there_is_nothing_to_say(self):
        pp2 = _holder(PP2)
        payload = {"__pp_void_output__": True}
        out = m.pp_output_payload_with_return_trip(pp2, payload, MB_ID)
        self.assertIs(
            out,
            payload,
            "the shipped builder returns the caller's own dict untouched "
            "when no decision, no chain and no parked fact exists",
        )

    def test_absorb_of_a_legacy_message_is_a_no_op(self):
        pp0 = _holder(PP0)
        self.assertFalse(
            m.pp_absorb_admission_return(pp0, {"__pp_void_output__": True})
        )
        self.assertEqual(m.pp_parked_continuation_carry(pp0), {})

    def test_a_live_chunk_is_not_a_parked_chunk(self):
        """`end > len(prefix_indices)` is a chunk IN FLIGHT and must not be
        reported: it is already a named member of a live pass."""
        live = _req(RID_PARKED, prefix_len=EXECUTED, extend_len=512)
        self.assertIsNone(m.pp_parked_continuation_fact(_holder(PP1, chunked_req=live)))

    def test_a_torn_down_request_is_not_reported(self):
        """`reset_for_retract` sets `extend_range = None` -- boot instr19's
        producer. Reporting it would re-offer a geometry nobody holds."""
        req = _parked_req()
        req.extend_range = None
        self.assertIsNone(m.pp_parked_continuation_fact(_holder(PP1, chunked_req=req)))


class Arm3NoDoublePrefill(unittest.TestCase):
    """The parked 4096 must ADVANCE. Nothing here may recompute it."""

    def test_the_carried_extent_is_never_reduced(self):
        pp0, pp1, _pp2, _downstream, _wire, _home = _drive_lap()
        self.assertEqual(m.pp_parked_continuation_carry(pp0)[RID_PARKED][0], EXECUTED)

        before = int(pp1.chunked_req.extend_range.end)
        before_prefix = len(pp1.chunked_req.prefix_indices)
        decision, _moved = _pp0_pass(pp0, queue_rids=[RID_A, RID_B, RID_PARKED])
        told = {e.rid: e.prefix_len for e in decision.entries}[RID_PARKED]

        self.assertEqual(told, EXECUTED)
        self.assertGreaterEqual(
            told,
            before_prefix,
            "a told BELOW the executed prefix would recompute tokens the "
            "holder already ran -- the one-chunk law's straight breach",
        )
        self.assertEqual(int(pp1.chunked_req.extend_range.end), before)
        self.assertEqual(len(pp1.chunked_req.prefix_indices), before_prefix)

    def test_the_holder_still_holds_it_after_the_lap(self):
        _pp0, pp1, _pp2, _downstream, _wire, _home = _drive_lap()
        self.assertIsNotNone(
            pp1.chunked_req,
            "the carry REPORTS; discarding the continuation anywhere on this "
            "path is the #961 producer shape",
        )
        self.assertEqual(pp1.chunked_req.rid, RID_PARKED)


class Arm4CanFail(unittest.TestCase):
    """Neuter the absorption alone and Arm 1 must go red again."""

    def test_blinding_the_absorb_restores_the_boot3_loop(self):
        original = getattr(m, "pp_note_parked_continuation", None)
        self.assertIsNotNone(
            original,
            "#968's absorption must be a module-global in "
            "`scheduler_pp_mixin`, so a can-fail proof can neuter THAT ONE "
            "step without reverting the harness with it",
        )
        m.pp_note_parked_continuation = lambda holder, message: 0
        try:
            pp0, _pp1, _pp2, _downstream, _wire, _home = _drive_lap()
            self.assertEqual(
                m.pp_parked_continuation_carry(pp0),
                {},
                "with the absorb neutered the table must stay empty",
            )
            decision, moved = _pp0_pass(pp0, queue_rids=[RID_A, RID_B, RID_PARKED])
            named = [e.rid for e in decision.entries]
            self.assertEqual(moved, ())
            self.assertNotIn(RID_PARKED, named)
            self.assertIn(
                _membership_verdict(named, _follower_batch(named))[0],
                ("missing", "extra"),
                "the refusal must come back -- if it does not, this arm is "
                "proving nothing",
            )
        finally:
            m.pp_note_parked_continuation = original

    def test_blinding_the_priority_leaves_the_rid_unnamed(self):
        original = m.pp_parked_continuation_priority
        m.pp_parked_continuation_priority = lambda scheduler: ()
        try:
            pp0, _pp1, _pp2, _downstream, _wire, _home = _drive_lap()
            self.assertEqual(
                m.pp_parked_continuation_carry(pp0), {RID_PARKED: (EXECUTED, PP1)}
            )
            decision, _moved = _pp0_pass(pp0, queue_rids=[RID_A, RID_B, RID_PARKED])
            self.assertNotIn(
                RID_PARKED,
                [e.rid for e in decision.entries],
                "absorbing without acting is not a fix: the acting half is "
                "what puts the rid in front of a seat",
            )
        finally:
            m.pp_parked_continuation_priority = original


class Arm5Staleness(unittest.TestCase):
    """Latest wins; a served or finished rid leaves."""

    def test_a_newer_stamp_overwrites_an_older_one(self):
        pp0 = _holder(PP0)
        m.pp_note_parked_continuation(
            pp0, {m._PP_PARKED_CONTINUATION_KEY: ((RID_PARKED, 4096, PP1),)}
        )
        m.pp_note_parked_continuation(
            pp0, {m._PP_PARKED_CONTINUATION_KEY: ((RID_PARKED, 8192, PP1),)}
        )
        self.assertEqual(m.pp_parked_continuation_carry(pp0), {RID_PARKED: (8192, PP1)})

    def test_a_rank_reporting_nothing_withdraws_its_own_claim(self):
        """Absence on a lap is a POSITIVE statement by the holder. A table
        that only grew would have PP0 naming rids nobody waits to build."""
        pp0 = _holder(PP0)
        m.pp_note_parked_continuation(
            pp0,
            {
                m._PP_PARKED_CONTINUATION_KEY: (
                    (RID_PARKED, 4096, PP1),
                    (RID_B, 512, PP2),
                )
            },
        )
        # PP1 still reports; PP2 has moved on and no longer names RID_B.
        m.pp_note_parked_continuation(
            pp0, {m._PP_PARKED_CONTINUATION_KEY: ((RID_PARKED, 4096, PP1),)}
        )
        self.assertEqual(m.pp_parked_continuation_carry(pp0), {RID_PARKED: (4096, PP1)})

    def test_the_holder_withdraws_its_claim_as_the_message_passes(self):
        pp1_holding = _holder(PP1, chunked_req=_parked_req())
        out = {}
        m.pp_parked_continuation_stamp(pp1_holding, {}, out)
        self.assertEqual(
            out[m._PP_PARKED_CONTINUATION_KEY], ((RID_PARKED, EXECUTED, PP1),)
        )

        pp1_empty = _holder(PP1, chunked_req=None)
        out2 = {}
        m.pp_parked_continuation_stamp(pp1_empty, out, out2)
        self.assertNotIn(
            m._PP_PARKED_CONTINUATION_KEY,
            out2,
            "a rank that no longer holds a continuation must drop its own "
            "stale claim as it relays",
        )

    def test_a_relay_preserves_a_foreign_claim(self):
        pp2 = _holder(PP2, chunked_req=None)
        incoming = {m._PP_PARKED_CONTINUATION_KEY: ((RID_PARKED, EXECUTED, PP1),)}
        out = {}
        m.pp_parked_continuation_stamp(pp2, incoming, out)
        self.assertEqual(
            out[m._PP_PARKED_CONTINUATION_KEY],
            ((RID_PARKED, EXECUTED, PP1),),
            "PP2 must relay PP1's fact -- overwriting per hop would lose it "
            "one hop before its only consumer",
        )

    def test_naming_a_rid_clears_it(self):
        pp0 = _holder(PP0)
        m.pp_note_parked_continuation(
            pp0, {m._PP_PARKED_CONTINUATION_KEY: ((RID_PARKED, EXECUTED, PP1),)}
        )
        self.assertTrue(m.pp_clear_parked_continuation(pp0, RID_PARKED))
        self.assertEqual(m.pp_parked_continuation_carry(pp0), {})
        self.assertFalse(m.pp_clear_parked_continuation(pp0, RID_PARKED))

    def test_malformed_entries_are_dropped_not_raised(self):
        """This sits on the output path, where raising turns one defect into
        two on the path least able to afford it."""
        pp0 = _holder(PP0)
        m.pp_note_parked_continuation(
            pp0,
            {m._PP_PARKED_CONTINUATION_KEY: (("bad",), (RID_PARKED, EXECUTED, PP1))},
        )
        self.assertEqual(
            m.pp_parked_continuation_carry(pp0), {RID_PARKED: (EXECUTED, PP1)}
        )
        self.assertEqual(m.pp_parked_continuation_facts_from_wire(None), ())
        self.assertEqual(m.pp_parked_continuation_facts_from_wire({}), ())


class TheHarnessMatchesProduction(unittest.TestCase):
    """Fidelity guards: the stand-ins must not drift from what they stand in
    for."""

    def test_the_membership_check_still_raises_missing_before_extra(self):
        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        self.assertIn("missing = [rid for rid in scheduled_extents", src)
        self.assertIn("extra = [rid for rid in admitted_rids", src)
        self.assertLess(
            src.index("missing = [rid for rid in scheduled_extents"),
            src.index("extra = [rid for rid in admitted_rids"),
            "`_membership_verdict` reports `missing` first because the "
            "shipped block raises it first; if that order moved, this "
            "harness is lying",
        )

    def test_pp0_acts_on_the_carry_inside_the_shipped_admission(self):
        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        self.assertIn("pp_parked_continuation_priority", src)
        self.assertLess(
            src.index("pp_parked_continuation_priority"),
            src.index("for req in self.waiting_queue:"),
            "the reorder must run BEFORE the admission loop walks the "
            "queue, or it reorders nothing that pass",
        )

    def test_every_new_helper_is_module_level(self):
        """The holder lesson, measured three times in this family (#973,
        #974, #978): anything reachable from the event loop must be a
        module-level function, never a `self.`-method."""
        for name in (
            "pp_parked_continuation_fact",
            "pp_parked_continuation_facts_from_wire",
            "pp_parked_continuation_stamp",
            "pp_note_parked_continuation",
            "pp_parked_continuation_carry",
            "pp_clear_parked_continuation",
            "pp_parked_continuation_priority",
        ):
            fn = getattr(m, name, None)
            self.assertTrue(
                inspect.isfunction(fn),
                f"{name} must be a module-level function on "
                f"scheduler_pp_mixin, not a method",
            )
            self.assertFalse(
                hasattr(m.SchedulerPPMixin, name),
                f"{name} must not also exist as a mixin method",
            )


if __name__ == "__main__":
    unittest.main()
