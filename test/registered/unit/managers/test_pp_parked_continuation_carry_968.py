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

    WHAT THIS FUNCTION DOES NOT PROVE, recorded after boot 4 rather than
    left for the next reader to rediscover (#968 self-audit, #968b). The
    line below builds `pp0.waiting_queue` WITH the parked rid already in it.
    That was written as scene-setting and it is in fact the ONE precondition
    the whole defect was about: PP0's copy of the continuation had been
    displaced out of the single `chunked_req` field by the cross-slot void
    restore and was in no queue at all, so the shipped reorder had nothing
    to move and returned `()` for 400+ consecutive passes. This harness
    supplied by hand the delivery it was measuring -- the #944 falsifier
    trap, and the arms below are valid against carry links 1-3 (mint,
    forward, absorb) and prove nothing about link 4 reaching a real
    production queue.

    THE QUEUE MEMBERSHIP IS NOW ESTABLISHED BY PRODUCTION CODE, and
    `test_pp_continuation_cross_slot_rehome_968b.py` drives that for real
    through the shipped `_pp_absorb_void_output` / `_pp_void_own_batch`:
    `pp_rehome_displaced_chunked_req` queues a continuation displaced by
    another slot's restore, and `pp_requeue_cleared_chunked_carry` queues
    one whose reset-shape carry is cleared. This module keeps its
    hand-built queue deliberately -- it is the unit test of the reorder and
    the decision build, given the queue -- and the other module is the
    integration proof that the queue is populated at all.

    THE SAME LIMIT WITH ITS CITATIONS, so the next reader can check the claim
    instead of taking it. This file's own header says it four hundred lines
    up -- "That request is then in NO queue on ANY rank -- `add_chunked_req`
    re-admits it from `self.chunked_req` directly -- so PP0 could not see
    it" -- and the shipped code says it twice more: `pp_rehome_refused_chunked_req`
    (scheduler_pp_mixin.py:2563-2565) "The chunked continuation is the one
    request that is never in the waiting queue, which is exactly why it is
    the one that goes missing", and the void's own disposal (:798-802) is
    what keeps it out. The ACT's docstring (:1757-1759) claims the opposite
    ("requests already in PP0's own `waiting_queue` -- the void's requeue put
    them there"); that claim is the one that is wrong.

    So: arms driven through this helper measure the REORDER MECHANISM given
    queue membership -- that `pp_parked_continuation_priority` rotates a
    named rid to the front, that the seat walk then admits it, and that
    `build_pp_admission_decision` names it. They do NOT measure that a parked
    continuation reaches a seat in production, and no assertion in them may
    be read that way. `Arm6ActIsANoOpOnTheRealDefectPath` pins what the act
    does when the queue membership is absent, without this helper.
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
    """The loop, and its end -- links 1 to 4 of five, and only those.

    HONESTLY BOUNDED 2026-08-28 (window-flip-0828 #968 self-audit). The lap
    itself -- MINT on PP1, FORWARD over the shipped sender/relay/absorber,
    ABSORB into PP0's carry table -- is driven for real here and the arms
    that assert on it stand. The FIFTH link, PP0 acting on what it absorbed,
    is reached in these arms only through `_pp0_pass`, which supplies the
    queue membership production never supplies (see that helper's docstring).
    The arms below therefore carry their scope in their own docstrings, and
    the unbounded reading -- "so a parked continuation gets named" -- is
    retracted. `Arm6ActIsANoOpOnTheRealDefectPath` states what actually
    happens on the defect path.
    """

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

    def test_a_carried_rid_THAT_IS_QUEUED_is_reordered_named_and_built(self):
        """Reorder -> seat -> naming, GIVEN queue membership. Not delivery.

        Renamed 2026-08-28 from `test_pp0_names_it_within_one_pass_and_the_
        follower_builds`, whose name asserted the unconditional delivery this
        arm does not measure: `_pp0_pass` puts `RID_PARKED` into
        `pp0.waiting_queue` itself. What is genuinely pinned here is that
        once a carried rid IS queued, the shipped reorder moves it to the
        front, the seat walk admits it inside `CAP`, the shipped
        `build_pp_admission_decision` names it, `told` reports the executed
        extent rather than a re-derived one, and the follower's
        `add_chunked_req` occupant is then a NAMED member. Every one of those
        is a real property of the shipped functions; none of them is evidence
        that a parked continuation ever reaches `waiting_queue`.
        """
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
        """The BEFORE shape, pinned: an empty carry table refuses every pass.

        SCOPE, tightened 2026-08-28. This reproduces the boot-3 REFUSAL
        ARITHMETIC (missing/extra, never "ok") from an empty carry table, and
        that much is real. It does not reproduce boot 3's full geometry: the
        specimen's rid was not merely un-carried, it was in no queue at all,
        whereas `_pp0_pass` queues it here. Read this as "an empty table is
        sufficient for the refusal loop", not as "an empty table is the only
        way to get there" -- the self-audit found a second, live way.
        """
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
        """Can-fail for the REORDER, under the same queue-membership premise.

        SCOPE, added 2026-08-28: this proves the reorder is load-bearing for
        a QUEUED carried rid -- neuter it and the tail rid never reaches a
        seat. It does not prove the reorder is load-bearing on the defect
        path, where the rid is not in the queue and the reorder is already a
        no-op with or without neutering (`Arm6ActIsANoOpOnTheRealDefectPath`).
        """
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


class Arm6ActIsANoOpOnTheRealDefectPath(unittest.TestCase):
    """What the ACT does when the continuation is where production leaves it.

    THE HONEST STATEMENT of the #968 self-audit (window-flip-0828,
    2026-08-28), pinned as a test so it cannot be re-forgotten. Links 1-4 of
    the carry deliver: the fact is minted on PP1, travels the shipped wire,
    and lands in PP0's carry table. Link 5 -- `pp_parked_continuation_
    priority` -- then permutes `scheduler.waiting_queue` and NOTHING ELSE
    (scheduler_pp_mixin.py:1785-1793, three silent `return ()` at :1784,
    :1787, :1792). A parked continuation is never in that queue: the void's
    disposal keeps it out (:798-802), `pp_void_keeps_request` skips the
    requeues (:2626-2628), and `pp_rehome_refused_chunked_req` says it in
    words (:2563-2565). So on the path that actually failed, the act returns
    the empty tuple and the rid stays unnamed.

    These arms are GREEN BEFORE AND AFTER the #968b fix by design -- that fix
    re-homes the continuation at the cross-slot restore sites and leaves the
    act unchanged. They are the regression fence around the retracted claim:
    if someone later "fixes" the act by teaching it to admit out of the carry
    table, these go red and the DESIGN_679 §4 rule-1 violation (a second
    admission authority) is caught at the desk instead of at a boot.
    """

    def _pp0_with_carry(self):
        pp0, _pp1, _pp2, _downstream, _wire, _home = _drive_lap()
        self.assertEqual(
            m.pp_parked_continuation_carry(pp0),
            {RID_PARKED: (EXECUTED, PP1)},
            "precondition: links 1-4 must have delivered the fact",
        )
        return pp0

    def test_the_fact_arrives_and_the_act_still_cannot_name_it(self):
        """The production shape: carry table full, continuation NOT queued."""
        pp0 = self._pp0_with_carry()
        # Production: PP0's queue holds OTHER requests. The continuation is
        # in none of them -- it lives in a `chunked_req` slot, or, under the
        # cross-slot defect #968b repairs, in nothing at all.
        pp0.waiting_queue = [
            _req(rid, prefix_len=0, extend_len=64) for rid in (RID_A, RID_B)
        ]
        before = list(pp0.waiting_queue)

        moved = m.pp_parked_continuation_priority(pp0)

        self.assertEqual(
            moved,
            (),
            "the act can only reorder what is IN the queue; the parked "
            "continuation is not, so it names nothing -- this is the "
            "structural no-op the self-audit found, not a bug in this test",
        )
        self.assertEqual(
            [r.rid for r in pp0.waiting_queue],
            [r.rid for r in before],
            "and it must leave the queue untouched when it names nothing",
        )

        guard = PPAdmissionCongruenceGuard()
        decision = build_pp_admission_decision(
            MB_ID,
            pp0.waiting_queue[:CAP],
            pp_size=WORLD,
            guard=guard,
            require_executed_geometry=True,
        )
        named = [e.rid for e in decision.entries]
        self.assertNotIn(
            RID_PARKED,
            named,
            "unnamed, exactly as in the 407-extra specimen: PP1's "
            "add_chunked_req occupant is an extra PP0 never named",
        )
        self.assertIn(
            _membership_verdict(named, _follower_batch(named))[0],
            ("missing", "extra"),
            "and so the pass refuses -- the boot-3/boot-4 signature",
        )

    def test_an_empty_queue_takes_the_silent_early_return(self):
        """queue=0 on 259 of PP0's 468 passes in the boot-4 specimen."""
        pp0 = self._pp0_with_carry()
        pp0.waiting_queue = []
        self.assertEqual(
            m.pp_parked_continuation_priority(pp0),
            (),
            "empty queue takes the `if not queue: return ()` branch "
            "(scheduler_pp_mixin.py:1787) -- silently, which is why the "
            "boot-4 log carried zero '#968' lines while the chain ran",
        )
        self.assertEqual(pp0.waiting_queue, [])

    def test_the_act_never_admits_out_of_the_carry_table(self):
        """DESIGN_679 §4 rule 1, fenced: reorder only, never a second adder.

        The fix for the real defect is upstream (re-home the continuation so
        an ordinary authority owns it again), never here. If a future edit
        makes the act synthesise a queue member out of the carry table, the
        rid appears without ever having been queued -- caught here.
        """
        pp0 = self._pp0_with_carry()
        pp0.waiting_queue = [
            _req(rid, prefix_len=0, extend_len=64) for rid in (RID_A, RID_B)
        ]
        m.pp_parked_continuation_priority(pp0)
        self.assertEqual(
            sorted(r.rid for r in pp0.waiting_queue),
            sorted((RID_A, RID_B)),
            "the act must not have ADDED anything to the queue -- it "
            "reorders a set it does not own",
        )


if __name__ == "__main__":
    unittest.main()
