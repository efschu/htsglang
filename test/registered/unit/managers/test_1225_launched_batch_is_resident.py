"""#1225 -- A LAUNCHED-BUT-UNDELIVERED MICROBATCH IS RESIDENT, AND ITS ROWS ARE OWNED.

MEASURED, boot weg1b12s1f4 on the S1 tip 488aa516b2 (base + S0 bus + S1
fixes, NO G1), log
/spinning/evidence-665-f1/boot_855_weg1b12s1f4_488aa516b2_0906_152449.log.
14 flip legs, ``flip_abandoned=0``, then at the 15th cutover (pp_to_tp,
generation=15) under agent load, in ONE second, at ONE seam::

    15:35:23 PP0  POOL CENSUS at-arm pp_to_tp: ... live_reqs=6 resident_slot_entries=6
    15:35:23 PP1  POOL CENSUS at-arm pp_to_tp: ... live_reqs=0 resident_slot_entries=0
    15:35:24 PP0  RESIDENTS RELEASED: 6 request(s) retracted, 6 live reference(s)
                  retired, tree dropped returning 84240 row(s)
    15:35:24 PP1  RESIDENTS RELEASED: 0 request(s) retracted, 0 live reference(s)
                  retired, tree dropped returning 0 row(s)
    15:35:35 PP0  #1040 REQ-POOL REBOUND ... (binding=17, outgoing free=8/8, escapees=0)
    15:35:35 PP1  ReqPoolRebindRefused: #1040 REQ-POOL 6 of 8 rows are still held
                  in the OUTGOING request pool at the cutover
                  (free=2, rows=[1,2,3,4,5,6], rids=[], unnamed=6)

The six rows are the six requests PP0 retracted at that seam (``#969AD RETRACT
site=readmit_seam_residents rank=0``, n=8..13): ``78fa9a0a``, ``5d37a09c``,
``a1957186``, ``56f3bb6c``, ``2b1f6c76``, ``171d5363``. Each has a LAUNCH
record on PP1 (``#969N ADMIT slot=N fwd_ct=...`` and/or ``#631 PROXY-SEND``)
and NO delivery record before the arm -- the last of them verbatim::

    15:35:17 PP1  #631 ROW-DELIVER d34 planned: batch=set
    15:35:18 PP1  #631 PROXY-SEND t34 stamp=(2, 34, 2107, 14, 296,
                                             ('171d5363', 81920, 84027))
    (no ROW-DELIVER d35, no delivery, 5 s of silence, then the arm)

THE DEFECT. A microbatch this rank LAUNCHED and has not yet had returned is
reachable only through ``mbs[slot]``, and ``scheduler_pp_mixin.py:5159-5177``
records in the tree's own words that nothing keeps it there: "the held batch's
last reference is destroyed at the next visit to this slot ... #1009 owns
making it true, and no line in this arm establishes it." So on a follower the
launched batch's requests fall out of every container the residency authority
walks, the cutover's retraction has nothing to retract, and their request-pool
rows are refused as escapees no rid names.

#1173 FIXED EXACTLY HALF OF THIS. It made ``arm()`` DEFER while
``_pp_launched_pending`` is non-empty -- on the REQUEST-ORIGIN rank. Its own
comment (``phase_flip_runtime.py:860-868``) then names the survivors and
dismisses them: "(a) a FOLLOWER, which takes the arm as an order and may
legitimately still hold an in-flight slot ... Both drain on their own; neither
is the launch-and-arm race." **The metal above refutes "both drain on their
own"**: the follower's in-flight slots do not drain, they are overwritten, and
their requests are stranded. The premise is the same one #1173 repaired; it
was left unreduced on every rank but PP0.

THE FIX PINNED HERE. The route is added to the ONE AUTHORITY, never to the
consumers -- the same rule ``_live_reqs`` already applied for ``last_mbs``
(W30) and ``mbs`` (#1202). It reuses the EXISTING ``_pp_launched_pending``
lifecycle rather than inventing a second notion of "outstanding": the batch is
retained at the same site that marks the slot pending, and released at the
same site that discards it, so the two can never disagree about what is
outstanding.

DANGER DIRECTION. Widening the authority widens the RETRACTION set, and
over-retraction frees a row its owner still holds. Two things bound it and
both are pinned below: a DELIVERED batch must never remain in the register
(else its requests are retracted twice), and the walk must dedup by identity
against the six containers (else a request resident in both is double-counted
by every consumer that sums them).

Hermetic: CPU tensors, real ``ReqToTokenPool``, real
``rebind_req_pool_for_cutover``, no accelerator, no GPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import ast
import pathlib
import types
import unittest

from sglang.srt.managers import phase_flip_runtime as pfr
from sglang.srt.managers import phase_req_pool_binding as prpb
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.test_utils import CustomTestCase

_REPO = pathlib.Path(__file__).resolve().parents[4]
_MIXIN_SRC = (
    _REPO / "python" / "sglang" / "srt" / "managers" / "scheduler_pp_mixin.py"
)
_RUNTIME_SRC = (
    _REPO / "python" / "sglang" / "srt" / "managers" / "phase_flip_runtime.py"
)

LAUNCHED = "_pp_launched_batches"


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
    sched = types.SimpleNamespace(
        req_to_token_pool=pool,
        running_mbs=[None, None, None],
        last_mbs=[None, None, None],
        mbs=[None, None, None],
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


class TestALaunchedUndeliveredBatchIsResident(CustomTestCase):
    def test_the_authority_sees_a_launched_batch_no_container_holds(self):
        """The follower shape at 15:35:23: six owners, ``live_reqs=0``."""
        pool = _pool()
        reqs = [
            _req(r)
            for r in (
                "78fa9a0a",
                "5d37a09c",
                "a1957186",
                "56f3bb6c",
                "2b1f6c76",
                "171d5363",
            )
        ]
        pool.alloc(reqs)
        self.assertEqual([r.req_pool_idx for r in reqs], [1, 2, 3, 4, 5, 6])

        # Launched on slot 2 and never returned; the slot has since been
        # revisited, so `mbs[2]` no longer names it -- the measured shape.
        sched = _scheduler(pool)
        setattr(sched, LAUNCHED, {2: _batch(reqs)})

        self.assertEqual(
            [r.rid for r in pfr._live_reqs(sched)],
            [r.rid for r in reqs],
            "the residency authority does not walk the launched-but-undelivered "
            "register, so six requests that own rows 1-6 read as live_reqs=0 "
            "and the cutover retracts nothing (#1225 / #1009)",
        )

    def test_the_follower_does_not_enter_the_cutover_holding_six_rows(self):
        """End state, through the real pool and the real refusal."""
        pp_pool, tp_pool = _pool(), _pool()
        reqs = [_req(f"r{i}") for i in range(6)]
        pp_pool.alloc(reqs)
        sched = _scheduler(pp_pool, tp_pool=tp_pool)
        setattr(sched, LAUNCHED, {2: _batch(reqs)})

        # The seam retracts what the authority names, then rebinds.
        for r in pfr._live_reqs(sched):
            pp_pool.free(r)
        setattr(sched, LAUNCHED, {})
        prpb.rebind_req_pool_for_cutover(sched, "tp")
        census = prpb.census_outgoing_req_pool(pp_pool, pfr._live_reqs(sched))
        self.assertEqual(census.escapees, 0)
        self.assertEqual(census.unnamed, 0)

    def test_the_pre_fix_shape_is_exactly_the_boot_refusal(self):
        """Control: with the register unwalked the refusal reproduces verbatim."""
        pp_pool, tp_pool = _pool(), _pool()
        reqs = [_req(f"r{i}") for i in range(6)]
        pp_pool.alloc(reqs)
        sched = _scheduler(pp_pool, tp_pool=tp_pool)  # register absent

        self.assertEqual(list(pfr._live_reqs(sched)), [])
        with self.assertRaises(prpb.ReqPoolRebindRefused) as caught:
            prpb.rebind_req_pool_for_cutover(sched, "tp")
        text = str(caught.exception)
        self.assertIn("6 of 8 rows are still held", text)
        self.assertIn("free=2", text)
        self.assertIn("rows=[1, 2, 3, 4, 5, 6]", text)
        self.assertIn("rids=[]", text)
        self.assertIn("unnamed=6", text)

    def test_a_request_in_both_the_register_and_a_container_is_counted_once(self):
        """DANGER DIRECTION 1: dedup by identity.

        A request resident in a slot AND still marked launched must appear
        once. Every consumer that sums the authority would otherwise
        double-bill it, and the retraction would free its row twice.
        """
        pool = _pool()
        r = _req("both")
        pool.alloc([r])
        b = _batch([r])
        sched = _scheduler(pool, running_mbs=[b, None, None])
        setattr(sched, LAUNCHED, {0: b})
        self.assertEqual([x.rid for x in pfr._live_reqs(sched)], ["both"])

    def test_a_delivered_batch_is_not_retained(self):
        """DANGER DIRECTION 2: the register empties where the slot delivers.

        Pinned BY SHAPE at the two sites that maintain ``_pp_launched_pending``:
        the batch must be retained where the slot is marked pending and
        released where it is discarded, so the two notions of "outstanding"
        cannot drift apart.
        """
        tree = ast.parse(_MIXIN_SRC.read_text())
        retains = []
        releases = []
        for node in ast.walk(tree):
            # `self._pp_launched_batches[mb_id] = <batch>` -- the RETAIN.
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Attribute)
                        and tgt.value.attr == LAUNCHED
                    ):
                        retains.append(node.lineno)
            # `...(self, "_pp_launched_batches", {}).pop(slot, None)` -- the RELEASE.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop"
                and LAUNCHED in ast.dump(node.func.value)
            ):
                releases.append(node.lineno)
        self.assertTrue(
            retains,
            "the launch site marks the slot pending but never retains the "
            "batch, so the authority has nothing to walk and the rows strand "
            "exactly as they did on the metal",
        )
        self.assertTrue(
            releases,
            "the delivery site discards the pending slot but never releases "
            "the batch: a delivered batch stays in the register and its "
            "requests would be retracted a second time, which frees a row its "
            "owner still holds (over-retraction, the silent direction)",
        )

    def test_the_register_is_reset_when_the_ring_is_rebuilt(self):
        """A rebuilt ring owes nothing; a surviving entry would name a slot
        of the previous topology."""
        src = _MIXIN_SRC.read_text()
        self.assertIn(
            f"holder.{LAUNCHED}",
            src,
            "the ring rebuild resets _pp_launched_pending but not the batch "
            "register, so entries outlive the topology they were taken from",
        )

    def test_the_route_is_in_the_authority_not_in_a_consumer(self):
        """``_live_reqs`` is the one authority; consumers never grow routes."""
        tree = ast.parse(_RUNTIME_SRC.read_text())
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_live_reqs":
                fn = node
                break
        self.assertIsNotNone(fn, "_live_reqs not found")
        self.assertIn(
            LAUNCHED,
            ast.dump(fn),
            "the launched-but-undelivered route was added somewhere other than "
            "the residency authority",
        )


if __name__ == "__main__":
    unittest.main()
