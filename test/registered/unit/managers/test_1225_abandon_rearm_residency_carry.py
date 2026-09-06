"""#1225 -- AN ABANDONED ARM MUST NOT LOSE THE ROWS ITS LEDGER NAMED.

MEASURED, boot 12 of this campaign, 4 of 4 runs across two tips
(/spinning/evidence-665-f1/boot_855_weg1b12g1f3r2_82ce0477f0_0906_132827.log
and ..._weg1b12g1f3r3_..., plus boot_855_weg1b12g1_1aafcd3b59_0906_103408.log
runs 1 and 5).  The sequence, byte-identical every time::

    cutover 1 COMMITTED (binding=3)
    cutover 2 COMMITTED (binding=4)
    arm 3            -- at-arm census: live_reqs=1 on all three ranks
    FLIP ABANDONED (pool too small for the live set) x3
    re-arm
    pre-cutover census: live_reqs=0 on all three ranks
    PP0  #1040 REQ-POOL REBOUND (binding=5, outgoing free=8/8, escapees=0)
    PP1  ReqPoolRebindRefused: #1040 REQ-POOL 1 of 8 rows are still held in
         the OUTGOING request pool at the cutover
         (free=7, rows=[1], rids=[], unnamed=1)
    PP2  identical

THE ROOT THIS FILE PINS.  ``_armed_residents`` is the #1202 armed-window
residency ledger: it exists precisely so that a request VISIBLE AT ARM and
INVISIBLE at the release is still retracted, and its row therefore still
returned before ``rebind_req_pool_for_cutover`` censuses the outgoing pool.
The ledger is scoped to an ARM.  The fact it records -- which request object
holds which request-pool row -- is scoped to the request-pool BINDING.

An abandon ends the arm WITHOUT a cutover.  No pool is rebound, no
``binding_tag`` is re-minted, and every row keeps the owner it had.  Yet both
abandon exits discarded the ledger outright, so a request that went invisible
between the abandon and the next cutover was in NO ledger: not in the
discarded one, and not in the fresh one the re-arm built from the
already-invisible live set.  ``cutover_resident_set`` then had nothing to
carry, the row was never retracted, and the census refused it as an escapee
that no rid names.

That is boot 9's shape re-entered through a different door, and it is why the
row is reported ``rids=[] unnamed=1``: on that rank the request was never in
the ledger the release reconciles against.

WHAT THE RANK DIVERGENCE IS AND IS NOT.  The defect is RANK-UNIFORM -- both
abandon exits and the arm entry run on decider and follower alike
(``_enter_armed_state`` docstring: "The state entry every arm performs,
decider and follower alike").  Only the WINDOW in which a request stays
visible differs between PP0 and its followers, because PP0 holds ``mbs[slot]``
for a microbatch until that microbatch's output has come back around the ring
(``on_round`` docstring).  So the decider survives the same defect the
followers die of.  ``resident_slot_entries`` -- the at-arm census figure the
boot postmortems read as "the followers' residency ledger" -- is NOT a ledger
at all: it is a local variable of ``_pool_census`` with no consumer, and it
walks ONE container while the residency authority walks six.  The last test in
this file pins that, so the instrument is never again read as a ledger.

THE DANGEROUS DIRECTION IS THE CARRY ITSELF.  Retracting too little stops the
boot loudly at the rebind; retracting too much frees a row its current owner
still holds and two requests then share one ``req_to_token`` row, silently.
So the carry is gated on the request-pool BINDING TAG, which ``clear()``
re-mints at every rebind: a carry may only ever be resumed into the same pool
generation it was parked from.  A carry that survives a cutover is dropped,
not applied.

Hermetic: CPU tensors, real ``ReqToTokenPool``, real
``rebind_req_pool_for_cutover``, no accelerator, no scheduler, no GPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import ast
import pathlib
import types
import unittest

from sglang.srt.managers import phase_flip_runtime as pfr
from sglang.srt.managers import phase_req_pool_binding as prpb
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.test_utils import CustomTestCase

_REPO = pathlib.Path(__file__).resolve().parents[4]
_RUNTIME_SRC = (
    _REPO / "python" / "sglang" / "srt" / "managers" / "phase_flip_runtime.py"
)


def _require(name):
    """The named carry, or a FAILURE that says what is missing.

    An ``AssertionError`` rather than an ``AttributeError``: the absence of the
    carry IS the defect under test, so it must read as a failed property and
    not as a broken test file.
    """
    fn = getattr(pfr.PhaseFlipRuntime, name, None)
    if fn is None:
        raise AssertionError(
            f"PhaseFlipRuntime has no {name!r}: an abandoned arm still drops "
            "the armed-window residency ledger outright, so a request that "
            "goes invisible between the abandon and the next cutover is in no "
            "ledger and its request-pool row is refused as an escapee (#1225)"
        )
    return fn


def _req(rid):
    return types.SimpleNamespace(
        rid=rid,
        req_pool_idx=None,
        req_pool_binding=None,
        inflight_middle_chunks=0,
        kv_committed_len=0,
    )


def _batch(reqs):
    return types.SimpleNamespace(reqs=list(reqs))


def _pool(size=8, ctx=8):
    return ReqToTokenPool(
        size=size, max_context_len=ctx, device="cpu", enable_memory_saver=False
    )


def _scheduler(pool, tp_pool=None, **containers):
    """A scheduler stand-in carrying exactly the attributes the walks read."""
    sched = types.SimpleNamespace(
        req_to_token_pool=pool,
        running_mbs=[],
        last_mbs=[],
        mbs=[],
        running_batch=None,
        last_batch=None,
        chunked_req=None,
        waiting_queue=[],
        tp_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(req_to_token_pool=pool)
        ),
        phase_flip_stacks=types.SimpleNamespace(
            tp_worker=types.SimpleNamespace(
                model_runner=types.SimpleNamespace(
                    req_to_token_pool=tp_pool if tp_pool is not None else _pool()
                )
            )
        ),
    )
    for k, v in containers.items():
        setattr(sched, k, v)
    return sched


def _runtime(sched):
    """The minimum ``self`` the carry methods read, and nothing more.

    Deliberately NOT a constructed ``PhaseFlipRuntime``: the methods under
    test are called unbound against this stand-in, so the code exercised is
    the real code while the fixture stays hermetic.  The WIRING -- that the
    two abandon exits and the arm entry actually call them -- is pinned
    separately by the AST test below, which is the half that catches a helper
    written and never reached.
    """
    return types.SimpleNamespace(
        _census_scheduler=sched,
        _armed_residents={},
        _carried_residents={},
        _carried_residents_binding=None,
    )


def _quiesce(sched):
    """The measured gap: every container the authority walks goes empty.

    This is what the 16 s between the abandon and the fatal cutover did to PP1
    and PP2 -- the request left every enumerable structure while its
    request-pool row stayed allocated.
    """
    sched.running_mbs = [None]
    sched.last_mbs = [None]
    sched.mbs = [None]
    sched.running_batch = None
    sched.last_batch = None
    sched.chunked_req = None


class TestAnAbandonedArmKeepsTheRowsItsLedgerNamed(CustomTestCase):
    def test_the_row_named_at_the_abandoned_arm_is_retracted_at_the_next_cutover(
        self,
    ):
        """The killer, end to end, through the real pool and the real refusal.

        Falls back to the pre-#1225 behaviour (drop the ledger) when the carry
        is absent, so at the parent commit this fails with boot 12's own
        ``ReqPoolRebindRefused``, not with a missing name.
        """
        park = getattr(pfr.PhaseFlipRuntime, "_park_armed_residents", None)
        resume = getattr(pfr.PhaseFlipRuntime, "_resume_armed_residents", None)

        pp_pool, tp_pool = _pool(), _pool()
        r = _req("weg1b12")
        pp_pool.alloc([r])
        self.assertEqual(r.req_pool_idx, 1, "the boot's own row id")
        sched = _scheduler(pp_pool, tp_pool=tp_pool, running_mbs=[_batch([r])])
        rt = _runtime(sched)

        # ARM 3.  live_reqs=1 on every rank; the ledger names the request.
        pfr.note_armed_residents(rt._armed_residents, sched)
        self.assertEqual(len(rt._armed_residents), 1)

        # FLIP ABANDONED (pool too small for the live set).  No cutover
        # happened, so the pool is not rebound and row 1 keeps its owner.
        if park is None:
            rt._armed_residents = {}  # the pre-#1225 exit
        else:
            park(rt, "pool too small for the live set")

        # The 16 s gap: the request leaves every container, its row stays out.
        _quiesce(sched)
        self.assertEqual(list(pfr._live_reqs(sched)), [])
        self.assertNotIn(1, pp_pool.free_slots, "row 1 is still held")

        # RE-ARM.  The fresh ledger can only see what is visible NOW, which is
        # nothing -- unless the carry survived the abandon.
        if resume is None:
            rt._armed_residents = {}  # the pre-#1225 arm entry
        else:
            resume(rt)
        pfr.note_armed_residents(rt._armed_residents, sched)

        # THE CUTOVER.
        reqs, report = pfr.cutover_resident_set(sched, rt._armed_residents)
        self.assertEqual(
            [id(x) for x in reqs],
            [id(r)],
            "the abandon dropped the ledger, so the cutover retracts nothing "
            "and row 1 stays held (#1225)",
        )
        self.assertEqual(report["carried_from_arm"], 1)

        # Retract it the way the seam does, then take the real refusal.
        pp_pool.free(r)
        prpb.rebind_req_pool_for_cutover(sched, "tp")
        census = prpb.census_outgoing_req_pool(pp_pool, pfr._live_reqs(sched))
        self.assertEqual(census.escapees, 0)
        self.assertEqual(census.unnamed, 0)

    def test_the_pre_fix_shape_is_exactly_the_boot_refusal(self):
        """The control: WITHOUT the carry the refusal text reproduces.

        Pins that the harness reaches the measured failure rather than some
        other one -- ``free=7, rows=[1], rids=[], unnamed=1`` is quoted from
        the boot log.
        """
        pp_pool, tp_pool = _pool(), _pool()
        r = _req("weg1b12")
        pp_pool.alloc([r])
        sched = _scheduler(pp_pool, tp_pool=tp_pool, running_mbs=[_batch([r])])

        snapshot = {}
        pfr.note_armed_residents(snapshot, sched)
        snapshot = {}  # the abandon, pre-#1225
        _quiesce(sched)

        reqs, _ = pfr.cutover_resident_set(sched, snapshot)
        self.assertEqual(reqs, [], "nothing to retract, which is the defect")

        with self.assertRaises(prpb.ReqPoolRebindRefused) as caught:
            prpb.rebind_req_pool_for_cutover(sched, "tp")
        text = str(caught.exception)
        self.assertIn("1 of 8 rows are still held", text)
        self.assertIn("free=7", text)
        self.assertIn("rows=[1]", text)
        self.assertIn("rids=[]", text)
        self.assertIn("unnamed=1", text)

    def test_a_carry_whose_pool_binding_moved_is_dropped_not_applied(self):
        """THE DANGEROUS DIRECTION.

        A carry parked under one request-pool generation must never be resumed
        into another.  ``ReqToTokenPool.clear()`` re-mints ``binding_tag`` at
        every rebind, and both pools hold the same number of rows, so a stale
        carry names a row that is IN RANGE and belongs to somebody else.
        Applying it would free that row under its live owner -- the silent
        corruption ``cutover_resident_set`` is written to refuse.
        """
        park = _require("_park_armed_residents")
        resume = _require("_resume_armed_residents")

        pool = _pool()
        r = _req("stale")
        pool.alloc([r])
        sched = _scheduler(pool, running_mbs=[_batch([r])])
        rt = _runtime(sched)

        pfr.note_armed_residents(rt._armed_residents, sched)
        park(rt, "pool too small for the live set")
        self.assertEqual(
            len(rt._carried_residents), 1, "the row was held, so it is carried"
        )

        # A cutover happens: the pool is cleared and its binding re-minted.
        before = pool.binding_tag
        pool.clear()
        self.assertNotEqual(pool.binding_tag, before)
        _quiesce(sched)

        resume(rt)
        self.assertEqual(
            rt._armed_residents,
            {},
            "a carry from a previous pool generation was resumed; it names a "
            "row this pool has already re-minted and retracting it would free "
            "a row somebody else owns (#1225 danger direction)",
        )

    def test_a_row_returned_by_its_other_namer_is_not_carried(self):
        """The carry is NARROWED at the park, never widened.

        ``ReqToTokenPool.free_slot`` names the case: "every row handed out is
        named in exactly two places -- the request that holds it and, for a
        streaming session, the ``SessionSlot`` that parked it between turns".
        ``pool.free(req)`` nulls ``req_pool_idx`` and is therefore already
        caught by the rowless arm; the case this guard exists for is the
        OTHER namer returning the row while the request object still names it.
        Carrying that entry would hand a freed row to the retraction, and the
        pool calls the second return "the row was returned twice".
        """
        park = _require("_park_armed_residents")

        pool = _pool()
        kept, parked = _req("kept"), _req("parked-by-session-slot")
        pool.alloc([kept, parked])
        sched = _scheduler(pool, running_mbs=[_batch([kept, parked])])
        rt = _runtime(sched)
        pfr.note_armed_residents(rt._armed_residents, sched)
        self.assertEqual(len(rt._armed_residents), 2)

        # The session slot returns the row; the request object is untouched
        # and still names it. This is the #1208(b) shape.
        pool.free_slot(parked.req_pool_idx, owner="session slot")
        self.assertIsNotNone(parked.req_pool_idx, "the request still names it")

        park(rt, "pool too small for the live set")

        carried = list(rt._carried_residents.values())
        self.assertEqual(
            [x.rid for x in carried],
            ["kept"],
            "a row already back in the free list must not be carried: the "
            "retraction would return it a second time",
        )

    def test_the_committed_cutover_still_retires_ledger_and_carry(self):
        """The #746 M5 property must survive the fix.

        A commit rebinds the pool and re-mints its ids, so neither the ledger
        nor the carry may outlive it -- a surviving strong reference would pin
        a stale object naming a row the next phase may legally re-mint.
        """
        src = _RUNTIME_SRC.read_text()
        tree = ast.parse(src)
        body = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_execute_body":
                body = node
                break
        self.assertIsNotNone(body, "_execute_body not found")
        cleared = set()
        for node in ast.walk(body):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                    ):
                        cleared.add(tgt.attr)
        self.assertIn("_armed_residents", cleared)
        self.assertIn(
            "_carried_residents",
            cleared,
            "the commit retires the ledger but not the carry, so a parked "
            "carry can outlive the pool generation it was taken from",
        )


class TestBothAbandonExitsAreWiredToTheCarry(CustomTestCase):
    """Written-and-never-reached is the failure this class exists for.

    Both abandon exits are the same class.  The one measured on metal 4/4 is
    the affordability exit ("pool too small for the live set"); the
    park-deadline exit in ``_abandon_parked_flip`` is its sibling and was
    never reached in these boots (the park-deadline string is 0 in every
    run).  A fix that repairs only the exit that happened to fire leaves the
    sibling live, so both are pinned here BY SHAPE rather than by line number.
    """

    def _assignments_and_calls(self, funcname):
        tree = ast.parse(_RUNTIME_SRC.read_text())
        target = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == funcname:
                target = node
                break
        self.assertIsNotNone(target, f"{funcname} not found")
        assigned, called = set(), set()
        for node in ast.walk(target):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                    ):
                        assigned.add(tgt.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
        return assigned, called

    def test_the_park_deadline_abandon_parks_the_ledger(self):
        assigned, called = self._assignments_and_calls("_abandon_parked_flip")
        self.assertIn(
            "_park_armed_residents",
            called,
            "the park-deadline abandon still drops the ledger outright",
        )
        self.assertNotIn("_armed_residents", assigned)

    def test_the_arm_entry_resumes_a_carry_before_it_notes_residents(self):
        assigned, called = self._assignments_and_calls("_enter_armed_state")
        self.assertIn(
            "_resume_armed_residents",
            called,
            "a fresh arm still starts from an empty ledger, so a carry parked "
            "by the abandon can never reach the cutover",
        )
        self.assertNotIn("_armed_residents", assigned)

    def test_the_affordability_abandon_parks_the_ledger(self):
        """The exit measured on metal, 4 of 4 runs across two tips.

        ``_execute_body`` also carries the COMMITTED exit, which must keep
        clearing.  So this asserts the park call is present, not that no
        assignment exists.
        """
        _, called = self._assignments_and_calls("_execute_body")
        self.assertIn(
            "_park_armed_residents",
            called,
            "the 'pool too small for the live set' abandon -- the exit that "
            "fired on every rank in every failing run -- still drops the "
            "ledger outright",
        )


class TestTheAtArmCensusFigureIsAnInstrumentNotALedger(CustomTestCase):
    """INDICATOR LAW, pinned so the figure is never read as a ledger again.

    The boot postmortems read ``resident_slot_entries=1`` on the decider and
    ``0`` on both followers, at 6 of 6 arms across three boots and two tips,
    as "the followers' residency ledger never names a live request".  It is
    not a ledger: it is a local of ``_pool_census`` with no consumer, and the
    divergence is fully explained by walk WIDTH -- it counts entries in
    ``running_mbs`` alone while the residency authority walks six containers.
    A request reachable only through ``mbs[slot]`` -- which is where a
    follower's planned batch sits -- is counted by one and not the other, on
    the same rank, with no ledger involved.
    """

    def test_the_authority_sees_a_request_the_census_figure_does_not(self):
        pool = _pool()
        r = _req("mbs-only")
        pool.alloc([r])
        sched = _scheduler(pool, mbs=[_batch([r])], running_mbs=[None])

        self.assertEqual(
            [x.rid for x in pfr._live_reqs(sched)],
            ["mbs-only"],
            "the residency authority walks mbs (#1202)",
        )
        narrow = sum(
            len(getattr(mb, "reqs", []) or [])
            for mb in (getattr(sched, "running_mbs", []) or [])
            if mb is not None
        )
        self.assertEqual(
            narrow,
            0,
            "resident_slot_entries counts running_mbs entries only, so "
            "live_reqs=1 with resident_slot_entries=0 is an instrument-width "
            "reading and not a divergence between two books",
        )

    def test_resident_slot_entries_has_no_consumer(self):
        """It is a local of ``_pool_census`` and is read only by its logger."""
        tree = ast.parse(_RUNTIME_SRC.read_text())
        owners = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == "resident_slot_entries":
                    owners.add(node.name)
        self.assertEqual(
            owners,
            {"_pool_census"},
            "resident_slot_entries is referenced outside the census; it is an "
            "instrument and nothing may branch on it",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                self.assertNotEqual(
                    node.attr,
                    "resident_slot_entries",
                    "resident_slot_entries became an attribute; a census "
                    "local that grows a consumer is a second book",
                )


if __name__ == "__main__":
    unittest.main()
