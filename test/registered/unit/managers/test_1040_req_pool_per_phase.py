"""#1040 CUT 1 -- per-phase ownership of the REQUEST-INDEX space.

The flip aliased both stacks' ``req_to_token`` (and the mamba index map) to ONE
tensor at boot, and never reassigned ``scheduler.req_to_token_pool`` at the
cutover.  Consequence, in the equivalence census' words: every row id the TP
phase used was minted by the PP allocator.

This suite pins the cut:

* the two ALIAS ASSIGNMENTS are gone from ``phase_flip_boot`` and the three
  SHAPE CHECKS around them stay (they are what keeps the shape-caching
  consumers correct across the rebind -- ``hisparse_coordinator`` and
  ``overlap_utils`` read ``req_to_token.shape`` once, at construction);
* the cutover REBINDS ``scheduler.req_to_token_pool`` to the incoming phase's
  own pool, unconditionally, and RAISES when it cannot;
* a ``Req`` carrying a row minted under one binding cannot re-present that row
  to another binding (in-range wrong-row write, silent, is the failure mode --
  both pools have the same row count, so no device assert catches it);
* the outgoing pool is CENSUSED at every cutover -- emitted always, raising on
  a non-zero escapee count (#919 on the request axis, never measured before).

The tests are hermetic: CPU tensors, fake runners, no accelerator.
"""

from __future__ import annotations

import pathlib
import types
import unittest

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.test_utils import CustomTestCase

_REPO = pathlib.Path(__file__).resolve().parents[4]
_BOOT = _REPO / "python" / "sglang" / "srt" / "managers" / "phase_flip_boot.py"


def _fake_req(idx=None, binding=None, chunked=0, committed=0):
    return types.SimpleNamespace(
        rid="r-test",
        req_pool_idx=idx,
        req_pool_binding=binding,
        inflight_middle_chunks=chunked,
        kv_committed_len=committed,
    )


def _pool(size=4, ctx=8):
    return ReqToTokenPool(
        size=size, max_context_len=ctx, device="cpu", enable_memory_saver=False
    )


def _scheduler(pp_pool, tp_pool, *, rebind_flag=False):
    """Minimal stand-in carrying exactly the attributes the rebind walks."""
    pp_runner = types.SimpleNamespace(req_to_token_pool=pp_pool)
    tp_runner = types.SimpleNamespace(req_to_token_pool=tp_pool)
    return types.SimpleNamespace(
        req_to_token_pool=pp_pool,
        tp_worker=types.SimpleNamespace(model_runner=pp_runner),
        phase_flip_stacks=types.SimpleNamespace(
            tp_worker=types.SimpleNamespace(model_runner=tp_runner)
        ),
        server_args=types.SimpleNamespace(phase_flip_rebind_hicache=rebind_flag),
        running_batch=None,
        waiting_queue=[],
    )


class TestTheBootNoLongerAliasesTheRequestIndexSpace(CustomTestCase):
    """T4 / C1.1: the assignments are deleted, the three shape checks remain."""

    def setUp(self):
        self.src = _BOOT.read_text()

    def test_the_req_to_token_alias_assignment_is_gone(self):
        self.assertNotIn(
            "tp_req_pool.req_to_token = pp_req_pool.req_to_token",
            self.src,
            "the flip must not point the TP stack at the PP request->token "
            "tensor; each phase owns its own request-index space (#1040)",
        )

    def test_the_mamba_index_map_alias_assignment_is_gone(self):
        self.assertNotIn(
            "tp_req_pool.req_index_to_mamba_index_mapping = pp_map",
            self.src,
            "the mamba index map is the same alias in the other coordinate "
            "system and goes with it (#1040)",
        )

    def test_the_three_shape_checks_survive_the_deletion(self):
        # They are load-bearing AFTER the deletion, not before: three consumers
        # cache req_to_token's SHAPE at construction (hisparse_coordinator,
        # overlap_utils x2) and are correct only while the two phases agree.
        for needle in (
            "req_to_token shapes diverge between stacks",
            "mamba index mapping shapes diverge",
            "mamba slot spaces diverge",
        ):
            self.assertIn(needle, self.src, f"shape check '{needle}' was removed")

    def test_the_dead_5a_premise_is_not_still_asserted(self):
        self.assertNotIn(
            "both stacks\n            # must read the same request->token rows",
            self.src,
        )


class TestTheCutoverRebindsTheSchedulerRequestPool(CustomTestCase):
    """T1 / C1.2: the scheduler follows the incoming phase's own pool."""

    def test_pp_to_tp_rebinds_to_the_tp_pool(self):
        from sglang.srt.managers.phase_req_pool_binding import (
            rebind_req_pool_for_cutover,
        )

        pp, tp = _pool(), _pool()
        sched = _scheduler(pp, tp)
        rebind_req_pool_for_cutover(sched, "tp")
        self.assertIs(sched.req_to_token_pool, tp)
        self.assertIsNot(sched.req_to_token_pool, pp)
        self.assertIs(sched.req_to_token_pool.req_to_token, tp.req_to_token)
        self.assertIsNot(sched.req_to_token_pool.req_to_token, pp.req_to_token)

    def test_tp_to_pp_rebinds_back(self):
        from sglang.srt.managers.phase_req_pool_binding import (
            rebind_req_pool_for_cutover,
        )

        pp, tp = _pool(), _pool()
        sched = _scheduler(pp, tp)
        sched.req_to_token_pool = tp
        rebind_req_pool_for_cutover(sched, "pp")
        self.assertIs(sched.req_to_token_pool, pp)

    def test_the_rebind_is_not_gated_on_the_hicache_flag(self):
        # DANGER DIRECTION.  `--phase-flip-rebind-hicache` is default-OFF; a
        # req-pool rebind inside that gate would leave the aliases deleted and
        # the pools never swapped -- the TP phase reading a tensor nobody
        # writes.  Both flag states must rebind identically.
        from sglang.srt.managers.phase_req_pool_binding import (
            rebind_req_pool_for_cutover,
        )

        for flag in (False, True):
            pp, tp = _pool(), _pool()
            sched = _scheduler(pp, tp, rebind_flag=flag)
            rebind_req_pool_for_cutover(sched, "tp")
            self.assertIs(sched.req_to_token_pool, tp, f"flag={flag}")

    def test_a_missing_incoming_pool_raises_rather_than_returning_none(self):
        from sglang.srt.managers.phase_req_pool_binding import (
            ReqPoolRebindRefused,
            rebind_req_pool_for_cutover,
        )

        pp = _pool()
        sched = _scheduler(pp, None)
        with self.assertRaises(ReqPoolRebindRefused):
            rebind_req_pool_for_cutover(sched, "tp")

    def test_the_incoming_pool_is_cleared_to_a_fresh_boot_state(self):
        from sglang.srt.managers.phase_req_pool_binding import (
            rebind_req_pool_for_cutover,
        )

        pp, tp = _pool(), _pool()
        # dirty the incoming pool the way a previous phase would have
        tp.alloc([_fake_req()])
        tp.req_to_token[1, 0] = 7
        sched = _scheduler(pp, tp)
        rebind_req_pool_for_cutover(sched, "tp")
        self.assertEqual(len(tp.free_slots), tp.size)
        self.assertEqual(int(tp.req_to_token.sum().item()), 0)


class TestTheWrongRowGuard(CustomTestCase):
    """T2 / C1.3: a row minted under binding A cannot be reused under B."""

    def test_alloc_stamps_the_binding_on_the_request(self):
        pool = _pool()
        req = _fake_req()
        pool.alloc([req])
        self.assertEqual(req.req_pool_binding, pool.binding_tag)

    def test_reusing_a_row_from_another_binding_raises(self):
        a, b = _pool(), _pool()
        self.assertNotEqual(a.binding_tag, b.binding_tag)
        req = _fake_req()
        a.alloc([req])
        # The request now presents a row id that is IN RANGE for b (both pools
        # have the same row count) but names a different phase's row.
        req.inflight_middle_chunks = 1
        with self.assertRaises(ValueError):
            b.alloc([req])

    def test_reusing_a_row_from_the_same_binding_is_allowed(self):
        pool = _pool()
        req = _fake_req()
        pool.alloc([req])
        req.inflight_middle_chunks = 1
        self.assertIsNotNone(pool.alloc([req]))

    def test_clear_mints_a_new_binding_tag(self):
        pool = _pool()
        before = pool.binding_tag
        pool.clear()
        self.assertNotEqual(pool.binding_tag, before)

    def test_a_row_carried_across_a_clear_is_refused(self):
        pool = _pool()
        req = _fake_req()
        pool.alloc([req])
        pool.clear()
        req.inflight_middle_chunks = 1
        with self.assertRaises(ValueError):
            pool.alloc([req])


class TestTheRequestAxisCensus(CustomTestCase):
    """T3 / C1.4: #919 on the request axis. Emitted always, raises non-zero."""

    def test_a_fully_free_outgoing_pool_reports_zero_escapees(self):
        from sglang.srt.managers.phase_req_pool_binding import (
            census_outgoing_req_pool,
        )

        pool = _pool()
        census = census_outgoing_req_pool(pool, reqs=())
        self.assertEqual(census.escapees, 0)
        self.assertEqual(census.free, pool.size)
        self.assertEqual(census.size, pool.size)

    def test_an_unfreed_row_is_counted_and_its_rid_named(self):
        from sglang.srt.managers.phase_req_pool_binding import (
            census_outgoing_req_pool,
        )

        pool = _pool()
        req = _fake_req()
        req.rid = "escapee-1"
        pool.alloc([req])
        census = census_outgoing_req_pool(pool, reqs=(req,))
        self.assertEqual(census.escapees, 1)
        self.assertIn("escapee-1", census.rids)
        self.assertIn(req.req_pool_idx, census.rows)

    def test_the_cutover_raises_when_the_outgoing_pool_is_not_empty(self):
        from sglang.srt.managers.phase_req_pool_binding import (
            ReqPoolRebindRefused,
            rebind_req_pool_for_cutover,
        )

        pp, tp = _pool(), _pool()
        sched = _scheduler(pp, tp)
        pp.alloc([_fake_req()])
        with self.assertRaises(ReqPoolRebindRefused):
            rebind_req_pool_for_cutover(sched, "tp")

    def test_the_census_line_carries_every_named_term(self):
        # Indicator law: an unmeasured zero is not a zero, so the line is
        # emitted on EVERY cutover and carries its own denominator.
        from sglang.srt.managers import phase_req_pool_binding as mod

        self.assertIn("#1040 REQ-POOL REBOUND", mod.REBIND_LOG_FORMAT)
        for term in ("binding=", "rows=", "outgoing free=", "escapees=", "rids="):
            self.assertIn(term, mod.REBIND_LOG_FORMAT)


class TestTheDefaultPathIsUnchanged(CustomTestCase):
    """T5: without a flip there is no second pool and nothing to rebind."""

    def test_a_lone_pool_allocates_and_frees_exactly_as_before(self):
        pool = _pool(size=3)
        reqs = [_fake_req(), _fake_req(), _fake_req()]
        idx = pool.alloc(reqs)
        self.assertEqual(sorted(idx), [1, 2, 3])
        self.assertEqual(pool.available_size(), 0)
        for r in reqs:
            pool.free(r)
        self.assertEqual(pool.available_size(), 3)

    def test_alloc_returns_none_when_the_pool_is_exhausted(self):
        pool = _pool(size=1)
        pool.alloc([_fake_req()])
        self.assertIsNone(pool.alloc([_fake_req()]))


class TestTheOnlyPoolObjectCacherFollowsTheRebind(CustomTestCase):
    """C1.5: kv_session_offload cached the pool OBJECT off the scheduler."""

    def test_kv_session_offload_reads_the_scheduler_at_use(self):
        src = (
            _REPO
            / "python"
            / "sglang"
            / "srt"
            / "managers"
            / "kv_session_offload.py"
        ).read_text()
        self.assertNotIn(
            "self.req_to_token_pool = scheduler.req_to_token_pool",
            src,
            "caching the pool OBJECT survives the cutover and writes through "
            "the outgoing phase's tensor (#1040 C1.5)",
        )


if __name__ == "__main__":
    unittest.main()
