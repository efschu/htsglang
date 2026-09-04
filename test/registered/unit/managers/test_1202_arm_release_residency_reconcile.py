"""#1202 -- THE RELEASE MUST RETRACT THE SET THE ARM APPROVED.

MEASURED, boot 9 of this campaign
(/spinning/evidence-665-f1/boot_855_weg1b9_1116175f6d_0904_164023.log):

    log:1883 PP0 at-arm pp_to_tp: cur_slot_reqs=1 resident_reqs=1 resident_slots=[1]
    log:1886 PP1 at-arm pp_to_tp: cur_slot_reqs=1 resident_reqs=0 resident_slots=[]
    log:1888 PP2 at-arm pp_to_tp: cur_slot_reqs=1 resident_reqs=0 resident_slots=[]

``cur_slot_reqs`` IS ``len(_live_reqs(scheduler))``, so at 16:45:03 the flip's
residency authority saw ONE request on all three ranks. One second later the
release runs its own ``list(_live_reqs(scheduler))``:

    log:2203 PP0 RESIDENTS RELEASED: 1 request(s) retracted
    log:2210 PP1 RESIDENTS RELEASED: 0 request(s) retracted
    log:2213 PP2 RESIDENTS RELEASED: 0 request(s) retracted

and the rows stayed behind:

    log:2189/:2194 (PP1+PP2) #938 PROTECTED RESIDUE AT DROP: 67 row(s) still locked
    log:2305/:2351 ReqPoolRebindRefused: 1 of 8 rows are still held in the
                   OUTGOING request pool at the cutover (free=7, rids=[], rows=[])

The same authority, the same rank, two answers one second apart, and NOTHING
RECONCILES THE TWO. That temporal gap is the root; the container gap (``mbs``
is not a route of the authority) is a real second hole but it is not what the
at-arm reading of 1 can be explained by.

WHAT THIS FILE PINS, and it is the temporal property rather than the container
property:

* a resident VISIBLE AT ARM and INVISIBLE one moment later is still retracted;
* a follower does not enter the cutover holding a row -- the end state, proved
  through the real ``ReqToTokenPool`` and the real
  ``rebind_req_pool_for_cutover`` refusal that killed boot 9;
* the DANGEROUS DIRECTION is refused: a snapshot member whose row has since
  been freed, or handed to a request that is live now, is NOT retracted a
  second time. Under-retraction stops the boot loudly at the rebind;
  OVER-retraction frees a row somebody else owns, which corrupts silently.
  The asymmetry is why the reconciliation is a filter and not a union;
* ``mbs`` is a route of the ONE authority (#1202 half 2);
* the refusal can name the rows it counts (#1202 half 3): ``rids=[]`` at
  ``escapees=1`` is a guard that fires correctly and tells the operator
  nothing;
* the citation of ``phase_flip_resident_carry`` names a module that exists.

Hermetic: CPU tensors, no accelerator, no scheduler.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

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
_CARRY_MODULE = (
    _REPO / "python" / "sglang" / "srt" / "managers" / "phase_flip_resident_carry.py"
)


def _require(name):
    """The named reconciliation, or a FAILURE that says what is missing.

    Deliberately an ``AssertionError`` rather than an ``AttributeError``: the
    absence of the reconciliation is the defect under test, so it must read as
    a failed property and not as a broken test file.
    """
    fn = getattr(pfr, name, None)
    if fn is None:
        raise AssertionError(
            f"phase_flip_runtime has no {name!r}: the release still enumerates "
            "the resident set at its own instant and nothing reconciles it "
            "with the set the arm approved (#1202)"
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


def _quiesce(sched):
    """The measured gap: every container the authority walks goes empty.

    This is what one second of boot 9 did to PP1 and PP2 -- the request left
    every enumerable structure while its request-pool row stayed allocated.
    """
    sched.running_mbs = [None]
    sched.last_mbs = [None]
    sched.mbs = [None]
    sched.running_batch = None
    sched.last_batch = None
    sched.chunked_req = None


class TestTheArmAndTheReleaseMustAgreeOnTheResidentSet(CustomTestCase):
    def test_a_resident_seen_at_arm_and_gone_at_release_is_still_retracted(self):
        note = _require("note_armed_residents")
        resolve = _require("cutover_resident_set")

        pool = _pool()
        r = _req("boot9")
        pool.alloc([r])
        sched = _scheduler(pool, running_mbs=[_batch([r])])

        # THE ARM INSTANT. This is log:1883's cur_slot_reqs=1.
        self.assertEqual([id(x) for x in pfr._live_reqs(sched)], [id(r)])
        snapshot = {}
        note(snapshot, sched)

        # ONE SECOND LATER. This is log:2210's "0 request(s) retracted".
        _quiesce(sched)
        self.assertEqual(list(pfr._live_reqs(sched)), [])

        reqs, report = resolve(sched, snapshot)
        self.assertEqual(
            [id(x) for x in reqs],
            [id(r)],
            "the release enumerated at its own instant and lost the resident "
            "the arm approved",
        )
        self.assertEqual(report["carried_from_arm"], 1)
        self.assertEqual(report["live_now"], 0)

    def test_the_follower_does_not_enter_the_cutover_holding_a_row(self):
        """End state, through the real pool and the real refusal.

        Falls back to the pre-#1202 authority when the reconciliation is
        absent, so at the parent commit this test fails with boot 9's own
        ``ReqPoolRebindRefused`` rather than with a missing name.
        """
        resolve = getattr(
            pfr,
            "cutover_resident_set",
            lambda s, snap: (list(pfr._live_reqs(s)), {}),
        )
        note = getattr(pfr, "note_armed_residents", lambda snap, s: snap)

        pp_pool, tp_pool = _pool(), _pool()
        r = _req("boot9")
        pp_pool.alloc([r])
        sched = _scheduler(pp_pool, tp_pool=tp_pool, running_mbs=[_batch([r])])

        snapshot = {}
        note(snapshot, sched)
        _quiesce(sched)

        # The seam's retraction, modelled at exactly the width that matters:
        # every enumerated resident returns its row.
        reqs, _ = resolve(sched, snapshot)
        for q in reqs:
            pp_pool.free_slot(q.req_pool_idx, owner="cutover")
            q.req_pool_idx = None

        # log:2305 / log:2351 -- this raised on PP1 and PP2 and killed boot 9.
        prpb.rebind_req_pool_for_cutover(sched, "tp")
        self.assertEqual(len(pp_pool.free_slots), pp_pool.size)


class TestTheReconciliationRefusesTheDangerousDirection(CustomTestCase):
    """Over-retraction corrupts; under-retraction stops loudly. Only one of
    the two may be traded away, and it is not this one."""

    def test_a_row_already_returned_is_not_retracted_a_second_time(self):
        note = _require("note_armed_residents")
        resolve = _require("cutover_resident_set")

        pool = _pool()
        r = _req("finished-cleanly")
        pool.alloc([r])
        sched = _scheduler(pool, running_mbs=[_batch([r])])
        snapshot = {}
        note(snapshot, sched)

        # It finished between arm and release: the row is back in the pool.
        pool.free_slot(r.req_pool_idx, owner="request")
        _quiesce(sched)

        reqs, report = resolve(sched, snapshot)
        self.assertEqual(list(reqs), [])
        self.assertEqual(report["skipped_row_free"], 1)

    def test_a_row_reallocated_to_a_live_request_is_not_taken_from_it(self):
        note = _require("note_armed_residents")
        resolve = _require("cutover_resident_set")

        pool = _pool()
        old = _req("old")
        pool.alloc([old])
        stale_row = old.req_pool_idx
        sched = _scheduler(pool, running_mbs=[_batch([old])])
        snapshot = {}
        note(snapshot, sched)

        # The row goes back and is handed to a request that IS live now, while
        # the stale object still names it.
        # `prepend=True` puts the row back at the HEAD, which is what makes
        # the next allocation reuse this very row -- the case under test.
        pool.free_slot(stale_row, owner="request", prepend=True)
        new = _req("new")
        pool.alloc([new])
        self.assertEqual(new.req_pool_idx, stale_row)
        sched.running_mbs = [_batch([new])]
        sched.last_mbs = [None]
        sched.mbs = [None]

        reqs, report = resolve(sched, snapshot)
        self.assertEqual(
            [id(x) for x in reqs],
            [id(new)],
            "the stale snapshot member was retracted and would have freed a "
            "row its new owner still holds",
        )
        self.assertEqual(report["skipped_row_reallocated"], 1)


class TestTheOneAuthorityWalksEveryResidencyRoute(CustomTestCase):
    def test_mbs_is_a_route_of_the_authority(self):
        pool = _pool()
        r = _req("in-mbs")
        pool.alloc([r])
        sched = _scheduler(pool, mbs=[_batch([r])])
        self.assertEqual(
            [id(x) for x in pfr._live_reqs(sched)],
            [id(r)],
            "`mbs` is written unconditionally on all three planning paths "
            "(scheduler_pp_mixin.py:4670/:4689/:4701) and is not a route of "
            "the residency authority",
        )

    def test_the_alias_between_mbs_and_running_mbs_costs_nothing(self):
        pool = _pool()
        r = _req("aliased")
        pool.alloc([r])
        batch = _batch([r])
        sched = _scheduler(pool, mbs=[batch], running_mbs=[batch])
        self.assertEqual(len(pfr._live_reqs(sched)), 1)

    def test_the_rebind_census_names_requests_through_the_same_authority(self):
        pool = _pool()
        r = _req("in-running-mbs")
        pool.alloc([r])
        sched = _scheduler(pool, running_mbs=[_batch([r])])
        self.assertEqual(
            [x.rid for x in prpb._live_reqs(sched)],
            ["in-running-mbs"],
            "the rebind census keeps its own two-container walk, so its "
            "refusal cannot name what the flip's authority can see",
        )


class TestTheRefusalNamesTheRowsItCounts(CustomTestCase):
    def test_held_rows_come_from_the_pools_own_arithmetic(self):
        pool = _pool()
        r = _req("unnamed")
        pool.alloc([r])
        # No request list at all -- exactly boot 9's rids=[] situation.
        census = prpb.census_outgoing_req_pool(pool, ())
        self.assertEqual(census.escapees, 1)
        self.assertEqual(
            tuple(census.rows),
            (r.req_pool_idx,),
            "the census counted an escapee it could not point at, although "
            "the pool's own free list names the row exactly",
        )

    def test_a_named_row_still_carries_its_rid(self):
        pool = _pool()
        r = _req("named")
        pool.alloc([r])
        census = prpb.census_outgoing_req_pool(pool, [r])
        self.assertEqual(tuple(census.rows), (r.req_pool_idx,))
        self.assertEqual(tuple(census.rids), ("named",))


class TestTheCitationNamesAModuleThatExists(CustomTestCase):
    def test_no_present_tense_citation_of_a_deleted_module(self):
        if _CARRY_MODULE.exists():
            self.skipTest("phase_flip_resident_carry.py exists at this tree")
        lines = _RUNTIME_SRC.read_text().splitlines()
        offenders = []
        for i, line in enumerate(lines):
            if "phase_flip_resident_carry" not in line:
                continue
            window = "\n".join(lines[max(0, i - 6) : i + 7])
            if "deleted" not in window and "#969" not in window:
                offenders.append(f"{_RUNTIME_SRC.name}:{i + 1}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "the module does not exist at this tree and these citations do not say so",
        )


if __name__ == "__main__":
    unittest.main()
