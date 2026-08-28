"""#968b: a chunked continuation displaced from the ONE `chunked_req` field.

THE SPECIMEN -- window-flip-0828 boot 4, pin b07029698d, Arm A, ~2 minutes,
514 refusals (407 extra + 107 missing) and a dead server.

#968 built a four-link carry so PP0 would learn what a follower has parked:
MINT the fact, FORWARD it on the message that already travels, ABSORB it on
PP0, ACT on it by moving the rid to the front of the admission queue. Links
one to three were PROVEN on metal (the absorb ran 514 times). The fourth was
structurally void, and this module drives the two reasons why:

  (a) `pp_parked_continuation_priority` permutes ONLY `waiting_queue`. A
      parked chunked continuation is never there -- this file says so in
      three places (`_park_chunked_prefill_chunk` :798-802,
      `pp_void_keeps_request` :2611-2615, `pp_rehome_refused_chunked_req`
      :2563-2564), and `pp_request_locations` names the same structural fact
      as the fourth instance of the compensator-reachability class. So the
      ACT could never reach the object it exists for, and its docstring
      cited two line ranges that contain no requeue at all.

  (b) PP0's OWN copy of the parked rid was dropped before the ACT could see
      it. `self.chunked_req` is ONE field and the void restore writes it
      once per slot, back to back, with no `get_next_batch_to_run` in
      between (`_pp_absorb_void_output` says so itself at :8296-8315). The
      LAST slot wins; any other slot's continuation is displaced out of the
      only place it lives, and the next pass's ring snapshot overwrites its
      last reference. Live witness in the boot-4 slice, line 408: the
      reset-shape clear drops `4077b704`, one of the 107 rids the very next
      refusal reports as MISSING.

THE CLASS, and it is not new: "a chunked continuation can leave all four
`pp_request_locations` places". #971 fixed the refusal exit. These are the
same class at exits two, three, four and five -- the two per-slot restores
(`_pp_absorb_void_output`, `_pp_void_own_batch`) and the two reset-shape
clears beside them.

WHAT THIS HARNESS DRIVES FOR REAL. The displacement is produced by calling
the SHIPPED `_pp_absorb_void_output` / `_pp_void_own_batch` twice, once per
slot, in the order production calls them, against the SHIPPED per-slot ring
written by the SHIPPED `_pp_note_chunked_req_before_admission`. Nothing here
hands a request to a queue or a table by hand to then find it there -- that
is the #944 falsifier trap the #968 self-audit caught itself in (its
`_pp0_pass` built `pp0.waiting_queue` WITH the parked rid in it, establishing
by hand the precondition production never met). The assertions read
`pp_request_locations`, the canonical four-place reader, so "the request
still exists somewhere" is answered by the product's own enumeration rather
than by this file's idea of where to look.
"""

import inspect
import logging
import types
import unittest
from array import array

import torch

from sglang.srt.managers import scheduler_pp_mixin as m
from sglang.srt.managers.scheduler_pp_mixin import (
    _PP_VOID_OUTPUT_KEY,
    SchedulerPPMixin,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORLD = 3
SLOT_A, SLOT_B = 0, 1

# The boot-4 slice's own rids, kept verbatim so a reader can find them.
RID_X = "1fa3f8087b3c4a1fa0b1b2c3d4e5f607"  # the parked continuation, 407x extra
RID_Y = "4077b7040b0a4c6d8e9f0a1b2c3d4e5f"  # the reset-shape clear's victim
RID_A = "c121b2f051de40729e9dd59855e58378"
RID_B = "e423ea58bb7e46fc8a3be75b26ca786a"

EXECUTED_X = 4096
EXECUTED_Y = 2048
TOTAL = 8422


def _req(rid, *, prefix_len, extend_len):
    """A real `Req`, because the shipped readers take `len(prefix_indices)`.

    `Req.__new__` rather than a stub class: `#796` forbids `prefix_indices`
    ever reaching a boolean context, and a stub would answer the geometry
    question with this file's arithmetic instead of the product's.
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
    req.is_retracted = False
    req.finished_reason = None
    return req


def _parked(rid, executed):
    """The parked shape verbatim: `extend_range.end == len(prefix_indices)`.

    `add_chunked_req`'s #679 branch and `_park_chunked_prefill_chunk` both
    restore exactly this, and it is what `pp_parked_continuation_fact` keys
    on. The request has EXECUTED `executed` tokens and has no chunk in
    flight -- which is why displacing it discards real work.
    """
    return _req(rid, prefix_len=executed, extend_len=0)


def _reset_shape(rid, executed, *, is_retracted=False):
    """`reset_for_retract`'s shape, or the belt case the shipped comment names.

    `_pp_absorb_void_output`'s own comment says `extend_range` is merely the
    field the next reader touches first and that "a request could in
    principle reach the reset shape by a path that does not set the flag".
    That is the case here, and it is the one the clear used to lose.
    """
    req = _parked(rid, executed)
    req.extend_range = None
    req.is_retracted = is_retracted
    return req


class _StubPool:
    def __init__(self, rows=1, cols=TOTAL):
        self.req_to_token = torch.arange(rows * cols, dtype=torch.int64).view(
            rows, cols
        )
        self.freed_req = []

    def free(self, req):
        self.freed_req.append(req)


class _StubAllocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.clone())


def _scheduler(*, waiting=(), running_reqs=()):
    """A holder carrying exactly what the two void sites read.

    The `#630/#757/#795/#797/#971` pattern this directory uses throughout: a
    bare `SimpleNamespace` with the SHIPPED methods bound on, so the code
    under test is the shipped code and only the pools are stood in.
    """
    running = (
        types.SimpleNamespace(reqs=list(running_reqs), is_prefill_only=False)
        if running_reqs
        else None
    )
    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=0, pp_size=WORLD),
        pp_group=types.SimpleNamespace(is_last_rank=False, is_first_rank=True),
        chunked_req=None,
        waiting_queue=list(waiting),
        running_batch=running,
        running_mbs=[None] * WORLD,
        mbs=[None] * WORLD,
        mb_metadata=[None] * WORLD,
        pp_loop_size=WORLD,
        req_to_token_pool=_StubPool(),
        token_to_kv_pool_allocator=_StubAllocator(),
        tree_cache=None,
        _pp_admission_guard=None,
        _pp_chunked_req_before_by_slot=[None] * WORLD,
        _pp_launched_chain_by_slot=[None] * WORLD,
        _pp_idle_void_suppress_log=False,
        _pp_parked_continuations={},
    )
    for name in (
        "_pp_absorb_void_output",
        "_pp_void_own_batch",
        "_pp_note_chunked_req_before_admission",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _snapshot_slot(h, slot, req):
    """One pass's top-of-pass snapshot for `slot`, through the shipped writer."""
    h.chunked_req = req
    h._pp_note_chunked_req_before_admission(slot)


def _batch(*reqs):
    return types.SimpleNamespace(reqs=list(reqs))


def _absorb(h, slot):
    """One void OUTPUT for `slot`, through the shipped absorber."""
    return h._pp_absorb_void_output(
        slot, {_PP_VOID_OUTPUT_KEY: True}, h.mbs, h.mb_metadata
    )


def _next_pass_ring_write(h, slot):
    """The next pass that visits `slot` overwrites its snapshot.

    THE RING IS NOT A HOME, and this line is what proves it. Production
    writes this on EVERY pass immediately before `get_next_batch_to_run`
    (`_event_loop_pp_body` :2925), so a request whose only remaining
    reference is a stale slot entry has at most `pp_size` passes to live.
    Asserting reachability before this line would credit the ring with a
    permanence it does not have.
    """
    h._pp_note_chunked_req_before_admission(slot)


def _located(h):
    return set(m.pp_request_locations(h))


class _ResetsTheLogGate(unittest.TestCase):
    """#968b instruments are rate-limited (first occurrence + every Nth).

    The counter is a MODULE-LEVEL diagnostic, exactly as production needs it
    -- a per-holder counter would reset on every scheduler rebuild and turn
    "first occurrence" into "every pass". The consequence for tests is that
    one test consumes another's first occurrence, so each log arm resets the
    gate rather than asserting against whatever the previous test left.
    """

    def setUp(self):
        m._PP_968_LOG_COUNTS.clear()


class Arm1CrossSlotDisplacement(_ResetsTheLogGate):
    """Two slots, one field, back-to-back restores: the last slot wins."""

    def _two_slot_void(self):
        h = _scheduler()
        x = _parked(RID_X, EXECUTED_X)
        y = _parked(RID_Y, EXECUTED_Y)
        # Two passes admitted, each carrying its own continuation.
        _snapshot_slot(h, SLOT_A, x)
        _snapshot_slot(h, SLOT_B, y)
        # The round moved `chunked_req` on (the adder took it), which is the
        # state the restore exists to undo.
        h.chunked_req = None
        # X is slot A's carried chunk and NOT a member of slot B's batch --
        # so slot B's disposal loop never sees it and never re-queues it.
        # That is the 407-extra shape: the ONLY thing that touches X is slot
        # B's restore, which overwrites the field it lives in.
        h.mbs[SLOT_A] = _batch(x, _req(RID_A, prefix_len=0, extend_len=64))
        h.mbs[SLOT_B] = _batch(y, _req(RID_B, prefix_len=0, extend_len=64))
        self.assertTrue(_absorb(h, SLOT_A))
        self.assertIs(h.chunked_req, x, "slot A's restore must re-home X")
        self.assertTrue(_absorb(h, SLOT_B))
        self.assertIs(h.chunked_req, y, "slot B's restore takes the one field")
        return h, x, y

    def test_the_displaced_continuation_is_still_reachable(self):
        h, x, _y = self._two_slot_void()
        _next_pass_ring_write(h, SLOT_A)

        self.assertIn(
            RID_X,
            _located(h),
            "X was displaced out of `self.chunked_req` by slot B's restore "
            "and its stale ring entry has now been overwritten by the next "
            "pass -- it is in NONE of the four `pp_request_locations` places. "
            "That is boot 4's 407 EXTRA: PP1 keeps building it, PP0 has no "
            "copy left to name",
        )
        self.assertIn(
            RID_X,
            {getattr(r, "rid", None) for r in h.waiting_queue},
            "the waiting queue is the only home left once the one "
            "`chunked_req` field is taken: `add_chunked_req` re-admits from "
            "`self.chunked_req` and nothing else, so a displaced "
            "continuation that is not queued can never be admitted again",
        )
        self.assertIn(
            id(x),
            [id(r) for r in h.waiting_queue],
            "the SAME object, not a copy -- the prefix, `last_node` and "
            "`prefix_indices` are the 4096 tokens already executed. Position "
            "is deliberately NOT asserted: the disposal loop appends its own "
            "released requests after the re-home, and the admission order is "
            "`pp_parked_continuation_priority`'s job, not this site's",
        )

    def test_the_prefix_is_never_discarded(self):
        """Kein-Doppel-Prefill: the re-home may not reset or truncate."""
        h, x, _y = self._two_slot_void()
        _next_pass_ring_write(h, SLOT_A)

        self.assertEqual(len(x.prefix_indices), EXECUTED_X)
        self.assertIsNotNone(x.extend_range)
        self.assertEqual(int(x.extend_range.end), EXECUTED_X)
        self.assertFalse(x.is_retracted)
        self.assertNotIn(
            x,
            h.req_to_token_pool.freed_req,
            "a displaced continuation is RE-HOMED, never released: its pages "
            "hold the tokens it already ran",
        )

    def test_the_act_can_now_reach_it(self):
        """The whole point: with X queued, link four is no longer void."""
        h, _x, _y = self._two_slot_void()
        _next_pass_ring_write(h, SLOT_A)
        m.pp_note_parked_continuation(
            h, {m._PP_PARKED_CONTINUATION_KEY: ((RID_X, EXECUTED_X, 1),)}
        )

        moved = m.pp_parked_continuation_priority(h)
        self.assertEqual(
            moved,
            (RID_X,),
            "the ACT permutes `waiting_queue`; until the re-home puts the "
            "displaced continuation there, it permutes a queue the request "
            "is not in and returns the empty tuple for ever",
        )
        self.assertEqual(
            h.waiting_queue[0].rid,
            RID_X,
            "the ACT moves the carried rid to the FRONT, which is what puts "
            "it in front of a seat in the ordinary admission loop",
        )

    def test_the_displacement_is_logged_with_the_slot(self):
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "INFO") as cm:
            self._two_slot_void()
        line = "\n".join(cm.output)
        self.assertIn("#968b REHOME-ON-DISPLACE", line)
        self.assertIn(RID_X, line)
        self.assertIn("from-slot=", line)


class Arm2ResetShapeClear(_ResetsTheLogGate):
    """The clear's own justification, made true instead of asserted.

    The shipped comment said the cleared request "is re-admitted from the
    waiting queue by the ordinary path". For a chunked continuation that is
    FALSE by this same file's :798-802, and boot 4 line 408 is the witness:
    the clear dropped `4077b704`, which the next refusal reported MISSING.
    """

    def _clear_pass(self, *, is_retracted=False, already_queued=False):
        z = _reset_shape(RID_Y, EXECUTED_Y, is_retracted=is_retracted)
        h = _scheduler(waiting=[z] if already_queued else [])
        _snapshot_slot(h, SLOT_B, z)
        h.chunked_req = None
        h.mbs[SLOT_B] = _batch(_req(RID_A, prefix_len=0, extend_len=64))
        self.assertTrue(_absorb(h, SLOT_B))
        return h, z

    def test_a_cleared_carry_is_queued_not_dropped(self):
        h, z = self._clear_pass()
        self.assertIsNone(
            h.chunked_req,
            "the field must still be cleared -- the next pass's "
            "`get_next_batch_to_run` dereferences `extend_range.end` with no "
            "guard of its own (boot instr19)",
        )
        self.assertIn(
            RID_Y,
            _located(h),
            "clearing the FIELD is right; dropping the REQUEST is the loss. "
            "Boot 4 line 408 cleared 4077b704 and the next refusal reported "
            "it missing",
        )
        self.assertIn(id(z), [id(r) for r in h.waiting_queue])

    def test_a_retracted_carry_is_queued_too(self):
        """`retract_decode` re-queues what it retracts; so must this."""
        h, z = self._clear_pass(is_retracted=True)
        self.assertIsNone(h.chunked_req)
        self.assertIn(id(z), [id(r) for r in h.waiting_queue])

    def test_an_already_queued_carry_is_not_duplicated(self):
        """The DOCUMENTED producer already re-queued it at :8411.

        A blind append would put the request in the batch twice, which is
        the hazard `_park_chunked_prefill_chunk` :798-802 warns about. The
        claim is made true by CHECKING reachability, not by asserting it.
        """
        h, z = self._clear_pass(already_queued=True)
        self.assertEqual(
            [r.rid for r in h.waiting_queue].count(RID_Y),
            1,
            "reachable already -- appending again duplicates the request",
        )
        self.assertIs(h.waiting_queue[0], z)

    def test_a_finished_request_is_not_re_queued(self):
        z = _reset_shape(RID_Y, EXECUTED_Y)
        z.finished_reason = object()
        h = _scheduler()
        _snapshot_slot(h, SLOT_B, z)
        h.chunked_req = None
        h.mbs[SLOT_B] = _batch(_req(RID_A, prefix_len=0, extend_len=64))
        _absorb(h, SLOT_B)
        self.assertIsNone(h.chunked_req)
        self.assertNotIn(
            RID_Y,
            {getattr(r, "rid", None) for r in h.waiting_queue},
            "a finished request has nothing left to run; queueing it would "
            "hand the admission loop a member it must immediately drop",
        )

    def test_the_requeue_is_logged(self):
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "INFO") as cm:
            self._clear_pass()
        line = "\n".join(cm.output)
        self.assertIn("#968b REQUEUE-ON-CLEAR", line)
        self.assertIn(RID_Y, line)


class Arm3TwinSiteOwnVoid(_ResetsTheLogGate):
    """`_pp_void_own_batch` carries the identical eight lines. Same treatment.

    The sibling sweep is part of the fix and not a follow-up: a class fixed
    at one of its two exits is a class that returns through the other.
    """

    def test_cross_slot_displacement_on_the_own_void(self):
        h = _scheduler()
        x = _parked(RID_X, EXECUTED_X)
        y = _parked(RID_Y, EXECUTED_Y)
        _snapshot_slot(h, SLOT_A, x)
        _snapshot_slot(h, SLOT_B, y)
        h.chunked_req = None
        h.mbs[SLOT_A] = _batch(x, _req(RID_A, prefix_len=0, extend_len=64))
        h.mbs[SLOT_B] = _batch(y, _req(RID_B, prefix_len=0, extend_len=64))

        self.assertTrue(h._pp_void_own_batch(SLOT_A))
        self.assertIs(h.chunked_req, x)
        self.assertTrue(h._pp_void_own_batch(SLOT_B))
        self.assertIs(h.chunked_req, y)
        _next_pass_ring_write(h, SLOT_A)

        self.assertIn(RID_X, _located(h))
        self.assertIn(id(x), [id(r) for r in h.waiting_queue])
        self.assertEqual(len(x.prefix_indices), EXECUTED_X)

    def test_reset_shape_clear_on_the_own_void(self):
        z = _reset_shape(RID_Y, EXECUTED_Y)
        h = _scheduler()
        _snapshot_slot(h, SLOT_B, z)
        h.chunked_req = None
        h.mbs[SLOT_B] = _batch(_req(RID_A, prefix_len=0, extend_len=64))

        self.assertTrue(h._pp_void_own_batch(SLOT_B))
        self.assertIsNone(h.chunked_req)
        self.assertIn(RID_Y, _located(h))
        self.assertIn(id(z), [id(r) for r in h.waiting_queue])


class Arm4ActInstrument(_ResetsTheLogGate):
    """The line that would have named boot 4 in ONE pass.

    Boot 4 logged ZERO lines containing `#968`. That zero was read as "the
    chain did not run" and it meant "the chain has no instrument" -- the
    #962a blind-probe class, inherited. Every link now prints, and the
    NO-OBSERVATION case prints most loudly: a carry that names rids with
    nothing in front of a seat is the exact state that spent 514 passes.
    """

    def test_carry_with_no_reachable_rid_prints_front_zero(self):
        h = _scheduler(waiting=[_req(RID_A, prefix_len=0, extend_len=64)])
        m.pp_note_parked_continuation(
            h, {m._PP_PARKED_CONTINUATION_KEY: ((RID_X, EXECUTED_X, 1),)}
        )
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "INFO") as cm:
            moved = m.pp_parked_continuation_priority(h)
        self.assertEqual(moved, ())
        line = "\n".join(cm.output)
        self.assertIn("#968 ACT", line)
        self.assertIn("carry=1", line)
        self.assertIn("front=0", line)
        self.assertIn(
            RID_X,
            line,
            "the line must NAME the rid it could not reach -- a count alone "
            "sends the next reader back to the slice to find out which",
        )

    def test_an_empty_queue_still_prints_the_zero_case(self):
        h = _scheduler()
        m.pp_note_parked_continuation(
            h, {m._PP_PARKED_CONTINUATION_KEY: ((RID_X, EXECUTED_X, 1),)}
        )
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "INFO") as cm:
            self.assertEqual(m.pp_parked_continuation_priority(h), ())
        joined = "\n".join(cm.output)
        self.assertIn("#968 ACT", joined)
        self.assertIn("queue=0", joined)
        self.assertIn("front=0", joined)

    def test_a_successful_act_prints_the_rids_it_moved(self):
        h = _scheduler(
            waiting=[
                _req(RID_A, prefix_len=0, extend_len=64),
                _parked(RID_X, EXECUTED_X),
            ]
        )
        m.pp_note_parked_continuation(
            h, {m._PP_PARKED_CONTINUATION_KEY: ((RID_X, EXECUTED_X, 1),)}
        )
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "INFO") as cm:
            self.assertEqual(m.pp_parked_continuation_priority(h), (RID_X,))
        joined = "\n".join(cm.output)
        self.assertIn("#968 ACT", joined)
        self.assertIn("front=1", joined)


class Arm4bChainInstruments(_ResetsTheLogGate):
    """Every link of the carry prints, and both park actuators name a route.

    Boot 4's route was UNDECIDABLE from the log: whether the continuation
    entered the slot via the #971 refusal re-home or via the #797b
    retraction restore could not be told, because both actuators are silent.
    """

    def test_mint_prints_the_fact_it_minted(self):
        h = _scheduler()
        h.chunked_req = _parked(RID_X, EXECUTED_X)
        h.ps.pp_rank = 1
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "INFO") as cm:
            fact = m.pp_parked_continuation_fact(h)
        self.assertEqual(fact, (RID_X, EXECUTED_X, 1))
        joined = "\n".join(cm.output)
        self.assertIn("#968 MINT", joined)
        self.assertIn(RID_X, joined)

    def test_mint_prints_the_none_case_when_a_chunked_req_exists(self):
        """A live chunk is NOT parked, and the silence about that was the
        ambiguity: nothing distinguished "no chunked request" from "a chunked
        request that does not qualify"."""
        h = _scheduler()
        h.chunked_req = _req(RID_X, prefix_len=EXECUTED_X, extend_len=512)
        h.ps.pp_rank = 1
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "INFO") as cm:
            self.assertIsNone(m.pp_parked_continuation_fact(h))
        joined = "\n".join(cm.output)
        self.assertIn("#968 MINT none", joined)
        self.assertIn("end=", joined)
        self.assertIn("prefix=", joined)

    def test_forward_prints_what_it_put_on_the_wire(self):
        h = _scheduler()
        h.chunked_req = _parked(RID_X, EXECUTED_X)
        h.ps.pp_rank = 1
        out = {}
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "INFO") as cm:
            m.pp_parked_continuation_stamp(h, {}, out)
        self.assertIn(m._PP_PARKED_CONTINUATION_KEY, out)
        self.assertIn("#968 FORWARD", "\n".join(cm.output))

    def test_absorb_prints_what_it_took_off(self):
        h = _scheduler()
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "INFO") as cm:
            n = m.pp_note_parked_continuation(
                h, {m._PP_PARKED_CONTINUATION_KEY: ((RID_X, EXECUTED_X, 1),)}
            )
        self.assertEqual(n, 1)
        joined = "\n".join(cm.output)
        self.assertIn("#968 ABSORB", joined)
        self.assertIn(RID_X, joined)

    def test_the_park_actuator_names_its_route(self):
        h = _scheduler()
        req = _req(RID_X, prefix_len=EXECUTED_X, extend_len=512)
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "INFO") as cm:
            self.assertTrue(m._park_chunked_prefill_chunk(h, req))
        joined = "\n".join(cm.output)
        self.assertIn("#968 PARK", joined)
        self.assertIn(RID_X, joined)

    def test_the_rehome_actuator_names_its_route(self):
        h = _scheduler()
        req = _req(RID_X, prefix_len=EXECUTED_X, extend_len=512)
        _snapshot_slot(h, SLOT_B, req)
        h.chunked_req = None
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "INFO") as cm:
            m.pp_rehome_refused_chunked_req(h, SLOT_B)
        joined = "\n".join(cm.output)
        self.assertIn("#971 REHOME-ON-REFUSAL", joined)
        self.assertIn(RID_X, joined)

    def test_the_void_output_line_names_rids_not_only_counts(self):
        """514 of 514 boot-4 void lines were rid-free; a count cannot be
        joined to a refusal."""
        h = _scheduler()
        a = _req(RID_A, prefix_len=0, extend_len=64)
        _snapshot_slot(h, SLOT_B, None)
        h.mbs[SLOT_B] = _batch(a)
        with self.assertLogs("sglang.srt.managers.scheduler_pp_mixin", "WARNING") as cm:
            _absorb(h, SLOT_B)
        joined = "\n".join(cm.output)
        self.assertIn("#791b PP-ADMISSION void output", joined)
        self.assertIn(RID_A, joined)


class Arm5LegacyUntouched(_ResetsTheLogGate):
    """A healthy pass shape must be byte-identical to the shipped behaviour."""

    def test_a_single_slot_void_queues_nothing_extra(self):
        h = _scheduler()
        x = _parked(RID_X, EXECUTED_X)
        _snapshot_slot(h, SLOT_A, x)
        h.chunked_req = x  # nothing displaced it: the field already holds it
        h.mbs[SLOT_A] = _batch(x)

        self.assertTrue(_absorb(h, SLOT_A))
        self.assertIs(h.chunked_req, x)
        self.assertEqual(
            h.waiting_queue,
            [],
            "the continuation is KEPT as `chunked_req` -- queueing it here "
            "would put it in the batch twice (:798-802)",
        )

    def test_no_carried_chunk_at_all_is_a_no_op(self):
        h = _scheduler()
        _snapshot_slot(h, SLOT_A, None)
        h.mbs[SLOT_A] = _batch(_req(RID_A, prefix_len=0, extend_len=64))

        self.assertTrue(_absorb(h, SLOT_A))
        self.assertIsNone(h.chunked_req)
        self.assertEqual(
            [r.rid for r in h.waiting_queue],
            [RID_A],
            "only the ordinary disposal re-queue, exactly as before #968b",
        )

    def test_a_resident_request_is_never_queued_by_the_rehome(self):
        """instr20: nothing in flight is rewritten."""
        x = _parked(RID_X, EXECUTED_X)
        h = _scheduler(running_reqs=[x])
        y = _parked(RID_Y, EXECUTED_Y)
        _snapshot_slot(h, SLOT_A, x)
        _snapshot_slot(h, SLOT_B, y)
        h.chunked_req = None
        h.mbs[SLOT_A] = _batch(x)
        h.mbs[SLOT_B] = _batch(y)
        _absorb(h, SLOT_A)
        _absorb(h, SLOT_B)

        self.assertNotIn(
            RID_X,
            {getattr(r, "rid", None) for r in h.waiting_queue},
            "a request that is decoding from its pages must not be queued "
            "for admission as well -- that is the double-admission #797 "
            "names, and it is already reachable in `running_batch`",
        )
        self.assertIn(RID_X, _located(h))

    def test_the_ordinary_pass_prints_no_968b_line(self):
        h = _scheduler()
        _snapshot_slot(h, SLOT_A, None)
        h.mbs[SLOT_A] = _batch(_req(RID_A, prefix_len=0, extend_len=64))
        logger = logging.getLogger("sglang.srt.managers.scheduler_pp_mixin")
        with self.assertLogs(logger, "INFO") as cm:
            logger.info("#968b probe anchor")
            _absorb(h, SLOT_A)
        self.assertNotIn("#968b REHOME-ON-DISPLACE", "\n".join(cm.output))
        self.assertNotIn("#968b REQUEUE-ON-CLEAR", "\n".join(cm.output))


class Arm6CanFail(_ResetsTheLogGate):
    """Neuter each new step alone; the arms it carries must go red again."""

    def test_blinding_the_displace_rehome_loses_the_continuation(self):
        original = getattr(m, "pp_rehome_displaced_chunked_req", None)
        self.assertIsNotNone(
            original,
            "the re-home must be a module-level function on "
            "`scheduler_pp_mixin`, so a can-fail proof can neuter THAT ONE "
            "step without reverting the harness with it (the holder lesson, "
            "measured three times in this family)",
        )
        m.pp_rehome_displaced_chunked_req = lambda *a, **k: None
        try:
            h = _scheduler()
            x = _parked(RID_X, EXECUTED_X)
            y = _parked(RID_Y, EXECUTED_Y)
            _snapshot_slot(h, SLOT_A, x)
            _snapshot_slot(h, SLOT_B, y)
            h.chunked_req = None
            h.mbs[SLOT_A] = _batch(x)
            h.mbs[SLOT_B] = _batch(y)
            _absorb(h, SLOT_A)
            _absorb(h, SLOT_B)
            _next_pass_ring_write(h, SLOT_A)
            self.assertNotIn(
                RID_X,
                _located(h),
                "with the re-home neutered the boot-4 loss must come back -- "
                "if it does not, this arm proves nothing",
            )
        finally:
            m.pp_rehome_displaced_chunked_req = original

    def test_blinding_the_clear_requeue_loses_the_carry(self):
        original = getattr(m, "pp_requeue_cleared_chunked_carry", None)
        self.assertIsNotNone(original)
        m.pp_requeue_cleared_chunked_carry = lambda *a, **k: False
        try:
            z = _reset_shape(RID_Y, EXECUTED_Y)
            h = _scheduler()
            _snapshot_slot(h, SLOT_B, z)
            h.chunked_req = None
            h.mbs[SLOT_B] = _batch(_req(RID_A, prefix_len=0, extend_len=64))
            _absorb(h, SLOT_B)
            _next_pass_ring_write(h, SLOT_B)
            self.assertNotIn(RID_Y, _located(h))
        finally:
            m.pp_requeue_cleared_chunked_carry = original

    def test_blinding_the_rehome_breaks_the_twin_site_too(self):
        original = getattr(m, "pp_rehome_displaced_chunked_req", None)
        m.pp_rehome_displaced_chunked_req = lambda *a, **k: None
        try:
            h = _scheduler()
            x = _parked(RID_X, EXECUTED_X)
            y = _parked(RID_Y, EXECUTED_Y)
            _snapshot_slot(h, SLOT_A, x)
            _snapshot_slot(h, SLOT_B, y)
            h.chunked_req = None
            h.mbs[SLOT_A] = _batch(x)
            h.mbs[SLOT_B] = _batch(y)
            h._pp_void_own_batch(SLOT_A)
            h._pp_void_own_batch(SLOT_B)
            _next_pass_ring_write(h, SLOT_A)
            self.assertNotIn(RID_X, _located(h))
        finally:
            m.pp_rehome_displaced_chunked_req = original


class TheFixMatchesWhatItClaims(unittest.TestCase):
    """Fidelity guards: the docstrings must cite lines that exist."""

    def test_every_new_helper_is_module_level(self):
        for name in (
            "pp_rehome_displaced_chunked_req",
            "pp_requeue_cleared_chunked_carry",
            "pp_chunked_req_is_reachable",
        ):
            fn = getattr(m, name, None)
            self.assertTrue(
                inspect.isfunction(fn),
                f"{name} must be a module-level function on "
                f"scheduler_pp_mixin, not a method",
            )
            self.assertFalse(hasattr(m.SchedulerPPMixin, name))

    def test_the_act_docstring_no_longer_cites_the_dead_line_ranges(self):
        """`:6791-6793` is the reconcile function and `:8104-8108` is a
        #951/#978 comment block. Neither contains a requeue, and the ACT's
        premise rested on both."""
        doc = m.pp_parked_continuation_priority.__doc__ or ""
        # The dead citations are KEPT, as a named retraction rather than
        # erased: a reader who greps for `:6791-6793` must land on the line
        # that says it was wrong, not on nothing at all.
        self.assertIn("DEAD", doc)
        self.assertIn(":7099", doc)
        self.assertIn(":8411", doc)
        self.assertIn(
            "pp_rehome_displaced_chunked_req",
            doc,
            "the ACT must cite the site that actually puts a displaced "
            "continuation in the queue it permutes",
        )
        self.assertIn("pp_requeue_cleared_chunked_carry", doc)

    def test_both_restore_sites_call_the_rehome(self):
        for fn in (
            SchedulerPPMixin._pp_absorb_void_output,
            SchedulerPPMixin._pp_void_own_batch,
        ):
            src = inspect.getsource(fn)
            self.assertIn("pp_rehome_displaced_chunked_req", src)
            self.assertIn("pp_requeue_cleared_chunked_carry", src)
            self.assertLess(
                src.index("pp_rehome_displaced_chunked_req"),
                src.index("self.chunked_req = chunked_before"),
                "the re-home must run BEFORE the overwrite, or it re-homes "
                "the value that just displaced the one it was meant to save",
            )


if __name__ == "__main__":
    unittest.main()
