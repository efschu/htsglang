"""#968 STARVATION UMBAU -- tests for the four-point rebuild of 2026-09-01.

THE ROOT THESE PIN (two-sided traversal, trainA/B/B2 @ d2b78d38d8): a PP
follower could not EXECUTE a forwarded prefix it did not locally hold; the
old equality clause refused, the refusal voided the pass rank-locally, PP0
stayed blocked in its ring recv with a launched microbatch, and the group
starved until a bound killed it (#1071 / PpChainRecvStalled / deadman
STARVED-ADMISSION). #1065 (write-side line) proved the store HAD the bytes
-- the underpriced WAIT and the #915 sole-occupant ban were the feeders.

Four points, each with its danger direction pinned:

  P1  `execute_scheduled_prefix`: truncate down / materialise up to EXACTLY
      the scheduled prefix, waiting bounded and length-priced for the store
      read. DANGER DIRECTION: a shortfall silently degrading to a short
      prefix (today's defect returning as a fallback) -- pinned as
      RuntimeError-or-nothing. The instr20 direction (growth PAST the
      schedule) is pinned clamped.
  P2  dynamic recv-abort provider: PP0 holding an ARMED flip must not
      outlive its own park deadline inside `recv_object` (the #977 form).
  P3  W30 arm guard: the seam-transport premise must consult LIVE store
      presence, not retract stamps alone; the boot-static
      `seam_readmit_available` knob is deleted.
  P4  feeders: the #915 rate cap may throttle the crowd, never ban the sole
      occupant; the prefetch-timeout pricing must cover a 16k readmit at the
      measured store rate (~0.32 s/KiToken best case, #1065).
"""

from __future__ import annotations

import time
import types
import unittest

import torch

import sglang.srt.managers.pp_admission_congruence as congruence
from sglang.srt.managers.pp_admission_congruence import (
    PPScheduleRefused,
    execute_scheduled_prefix,
    schedule_refusal_reason,
    store_read_bound_s,
)


class _Node:
    pass


class _Req:
    def __init__(self, prefix_len, total=845, rid="rid968"):
        self.rid = rid
        self.prefix_indices = torch.arange(prefix_len)
        self.cache_protected_len = prefix_len
        self.host_hit_length = 0
        self.best_match_node = _Node()
        self.last_node = _Node()
        self.full_untruncated_fill_ids = list(range(total))
        self.mamba_loadback_anchor_adopted = False

    def init_next_round_input(self, tree_cache=None):
        # Rank-local re-match stand-in: state unchanged between polls.
        pass


class _ServingTree:
    """Fake tree cache whose host tier serves ``serve`` tokens per load-back
    (or ``serve(call_no)`` when callable)."""

    enable_storage = False

    def __init__(self, serve):
        self.serve = serve
        self.calls = 0

    def init_load_back(self, params):
        self.calls += 1
        n = self.serve(self.calls) if callable(self.serve) else self.serve
        start = 10_000 + self.calls * 1_000
        return torch.arange(start, start + n), _Node()


class ExecuteScheduledPrefix968(unittest.TestCase):
    """P1: the materialisation station."""

    def setUp(self):
        self._base = congruence.MATERIALISE_BASE_S
        self._max = congruence.MATERIALISE_MAX_S
        congruence.MATERIALISE_BASE_S = 0.0
        congruence.MATERIALISE_MAX_S = 0.15

    def tearDown(self):
        congruence.MATERIALISE_BASE_S = self._base
        congruence.MATERIALISE_MAX_S = self._max

    def test_deficit_is_materialised_to_exactly_the_schedule(self):
        req = _Req(prefix_len=0)
        loaded = execute_scheduled_prefix(req, _ServingTree(512), 512)
        self.assertEqual(loaded, 512)
        self.assertEqual(len(req.prefix_indices), 512)
        self.assertEqual(req.cache_protected_len, 512)

    def test_overserve_is_clamped_the_instr20_direction(self):
        req = _Req(prefix_len=0)
        execute_scheduled_prefix(req, _ServingTree(700), 512)
        self.assertEqual(len(req.prefix_indices), 512)

    def test_local_surplus_is_truncated_to_the_schedule(self):
        req = _Req(prefix_len=512)
        tree = _ServingTree(0)
        loaded = execute_scheduled_prefix(req, tree, 100)
        self.assertEqual(loaded, 0)
        self.assertEqual(len(req.prefix_indices), 100)
        self.assertEqual(req.cache_protected_len, 100)
        self.assertEqual(tree.calls, 0, "truncation must not touch the store")

    def test_shortfall_is_a_loud_group_stop_never_a_short_prefix(self):
        """THE DANGER DIRECTION: today's defect (proceed at a short prefix /
        void the pass) must never return as a fallback. A store that cannot
        serve the deficit within the bound is a RuntimeError -- and NOT a
        PPScheduleRefused, whose void disposal is what left PP0 starving."""
        req = _Req(prefix_len=0)
        with self.assertRaises(RuntimeError) as ctx:
            execute_scheduled_prefix(req, _ServingTree(0), 512)
        msg = str(ctx.exception)
        self.assertIn("#968 PREFIX MATERIALISATION SHORTFALL", msg)
        self.assertIn("RAENGE-NIE-UNEINS", msg)
        self.assertNotIsInstance(ctx.exception, PPScheduleRefused)

    def test_partial_service_across_the_wait_completes(self):
        """#1065: the store delivers over time; the bounded wait accumulates
        partial reads instead of dying on the first one."""
        req = _Req(prefix_len=0)
        tree = _ServingTree(lambda call: 200 if call == 1 else 312)
        loaded = execute_scheduled_prefix(req, tree, 512)
        self.assertEqual(loaded, 512)
        self.assertEqual(len(req.prefix_indices), 512)
        self.assertGreaterEqual(tree.calls, 2)

    def test_anchor_off_extent_is_fatal_not_clamped(self):
        """S1/FIX-3: a GDN anchor adopted at a depth the schedule does not
        name may be neither clamped under nor taken -- fatal, named."""

        class _AnchorTree(_ServingTree):
            def init_load_back(self, params):
                params.req.mamba_loadback_anchor_adopted = True
                return super().init_load_back(params)

        req = _Req(prefix_len=0)
        with self.assertRaises(RuntimeError) as ctx:
            execute_scheduled_prefix(req, _AnchorTree(700), 512)
        self.assertIn("GDN anchor", str(ctx.exception))

    def test_no_op_when_local_already_matches(self):
        req = _Req(prefix_len=512)
        tree = _ServingTree(0)
        self.assertEqual(execute_scheduled_prefix(req, tree, 512), 0)
        self.assertEqual(tree.calls, 0)

    def test_bound_is_length_priced_and_capped(self):
        congruence.MATERIALISE_BASE_S = 2.0
        congruence.MATERIALISE_MAX_S = 45.0
        base = store_read_bound_s(0)
        self.assertAlmostEqual(base, 2.0)
        self.assertGreater(store_read_bound_s(16 * 1024), base)
        self.assertLessEqual(store_read_bound_s(10_000_000), 45.0)


class RefusalReason968(unittest.TestCase):
    """P1: the equality clause is deleted, the other two clauses survive."""

    def test_prefix_mismatch_is_no_longer_a_refusal_input(self):
        import inspect

        params = inspect.signature(schedule_refusal_reason).parameters
        self.assertNotIn("local_prefix_len", params)

    def test_fill_overrun_still_refused(self):
        reason = schedule_refusal_reason(
            rid="r",
            scheduled_prefix_len=0,
            scheduled_extend_len=1000,
            local_fill_len=845,
        )
        self.assertIsNotNone(reason)
        self.assertIn("845", reason)

    def test_negative_extend_still_refused(self):
        self.assertIsNotNone(
            schedule_refusal_reason(
                rid="r",
                scheduled_prefix_len=0,
                scheduled_extend_len=-1,
                local_fill_len=845,
            )
        )

    def test_executable_geometry_has_no_reason(self):
        self.assertIsNone(
            schedule_refusal_reason(
                rid="r",
                scheduled_prefix_len=0,
                scheduled_extend_len=512,
                local_fill_len=845,
            )
        )


class StorePresenceRefuter968(unittest.TestCase):
    """P3: the premise consults live tier presence, refuters only."""

    def _scheduler(self, children, avail, size, storage_on):
        pool = types.SimpleNamespace(size=size)
        pool.available_size = lambda: avail
        cc = types.SimpleNamespace(mem_pool_host=pool)
        root = types.SimpleNamespace(children=children)
        tree = types.SimpleNamespace(
            root_node=root, cache_controller=cc, enable_storage=storage_on
        )
        return types.SimpleNamespace(tree_cache=tree)

    def test_all_tiers_refute_stands_the_premise_down(self):
        from sglang.srt.managers.phase_purity import seam_store_presence_refuted

        refuted, detail = seam_store_presence_refuted(
            self._scheduler(children={}, avail=100, size=100, storage_on=False)
        )
        self.assertTrue(refuted)
        self.assertIn("tree_empty=True", detail)

    def test_host_content_blocks_refutation(self):
        from sglang.srt.managers.phase_purity import seam_store_presence_refuted

        refuted, _ = seam_store_presence_refuted(
            self._scheduler(children={}, avail=50, size=100, storage_on=False)
        )
        self.assertFalse(refuted)

    def test_storage_on_blocks_refutation(self):
        from sglang.srt.managers.phase_purity import seam_store_presence_refuted

        refuted, _ = seam_store_presence_refuted(
            self._scheduler(children={}, avail=100, size=100, storage_on=True)
        )
        self.assertFalse(refuted)

    def test_unreadable_probes_never_refute(self):
        """Denominator law: unknown is not empty."""
        from sglang.srt.managers.phase_purity import seam_store_presence_refuted

        refuted, detail = seam_store_presence_refuted(
            types.SimpleNamespace(tree_cache=None)
        )
        self.assertFalse(refuted)
        self.assertIn("None", detail)

    def test_premise_with_stamps_but_empty_tiers_is_refused(self):
        """RED-FIRST against the pre-#968 premise: retract stamps alone said
        True while every tier was provably empty -- the W30 arm then flipped
        into a layout that could only re-prefill cold (four 2026-09-01 logs:
        0 premise refusals beside total re-prefill)."""
        from sglang.srt.managers import phase_purity

        req = types.SimpleNamespace(
            cached_prompt_tokens_at_retract=4096, cache_protected_len=0
        )
        setattr(req, phase_purity.SEAM_READMIT_ATTR, 1)
        setattr(req, phase_purity.SEAM_GRANT_CONSUMED_ATTR, False)
        sched = self._scheduler(children={}, avail=100, size=100, storage_on=False)
        sched.waiting_queue = [req]
        self.assertFalse(phase_purity.seam_transport_premise_holds(sched))

    def test_premise_with_stamps_and_host_content_still_holds(self):
        from sglang.srt.managers import phase_purity

        req = types.SimpleNamespace(
            cached_prompt_tokens_at_retract=4096, cache_protected_len=0
        )
        setattr(req, phase_purity.SEAM_READMIT_ATTR, 1)
        setattr(req, phase_purity.SEAM_GRANT_CONSUMED_ATTR, False)
        sched = self._scheduler(children={}, avail=50, size=100, storage_on=False)
        sched.waiting_queue = [req]
        self.assertTrue(phase_purity.seam_transport_premise_holds(sched))

    def test_the_static_knob_is_gone(self):
        """Upstream-minimal: the dead boot-static `seam_readmit_available`
        knob must not grow back (its hardcoded True made the W30 refusal
        unreachable for its whole life)."""
        from sglang.srt.managers.phase_policy import PhasePolicyConfig

        self.assertNotIn(
            "seam_readmit_available",
            {f.name for f in __import__("dataclasses").fields(PhasePolicyConfig)},
        )


class RecvAbortProvider968(unittest.TestCase):
    """P2: dynamic abort bound for the parked recv."""

    def tearDown(self):
        from sglang.srt.distributed.pp_object_recv import set_recv_abort_provider

        set_recv_abort_provider(None)

    def test_merge_semantics_tighter_positive_bound_wins(self):
        from sglang.srt.distributed import pp_object_recv as m

        m.set_recv_abort_provider(None)
        self.assertEqual(m.effective_abort_after_s(0.0), 0.0)
        self.assertEqual(m.effective_abort_after_s(7.0), 7.0)
        m.set_recv_abort_provider(lambda: 60.0)
        self.assertEqual(m.effective_abort_after_s(0.0), 60.0)
        self.assertEqual(m.effective_abort_after_s(7.0), 7.0)
        m.set_recv_abort_provider(lambda: 3.0)
        self.assertEqual(m.effective_abort_after_s(7.0), 3.0)

    def test_provider_error_reads_as_no_bound(self):
        from sglang.srt.distributed import pp_object_recv as m

        def boom() -> float:
            raise ValueError("provider broke")

        m.set_recv_abort_provider(boom)
        self.assertEqual(m.effective_abort_after_s(0.0), 0.0)
        self.assertEqual(m.effective_abort_after_s(9.0), 9.0)

    def test_flip_hold_bound_is_deadline_plus_slack_only_while_armed(self):
        from sglang.srt.managers.phase_flip_runtime import (
            DEFAULT_FLIP_HOLD_RECV_SLACK_S,
            pp0_flip_hold_recv_bound_s,
        )

        self.assertEqual(pp0_flip_hold_recv_bound_s(None), 0.0)
        idle = types.SimpleNamespace(
            _pending=None, _armed_at=None, _park_deadline_s=30.0
        )
        self.assertEqual(pp0_flip_hold_recv_bound_s(idle), 0.0)
        unarmed_clock = types.SimpleNamespace(
            _pending="pp_to_tp", _armed_at=None, _park_deadline_s=30.0
        )
        self.assertEqual(pp0_flip_hold_recv_bound_s(unarmed_clock), 0.0)
        armed = types.SimpleNamespace(
            _pending="pp_to_tp",
            _armed_at=time.monotonic(),
            _park_deadline_s=30.0,
        )
        self.assertEqual(
            pp0_flip_hold_recv_bound_s(armed),
            30.0 + DEFAULT_FLIP_HOLD_RECV_SLACK_S,
        )


class Feeders968(unittest.TestCase):
    """P4: the two feeders named by #1065."""

    def test_rate_cap_floor_restores_one_request_prefetchability(self):
        """#1065: half of the measured TP-phase pool (30518 -> 15259) sat
        below one 16k readmit, structurally banning the second readmit
        prefetch. The producer floor lifts the budget to one request's
        worth; retention-scale pools keep the plain half; tiny pools are
        never over-claimed past 90%."""
        from sglang.srt.managers.cache_controller import (
            PREFETCH_CAP_FLOOR_TOKENS,
            prefetch_capacity_limit_for,
        )

        self.assertGreaterEqual(PREFETCH_CAP_FLOOR_TOKENS, 16384)
        self.assertGreaterEqual(prefetch_capacity_limit_for(30518), 16384)
        self.assertEqual(prefetch_capacity_limit_for(1_000_000), 500_000)
        self.assertLessEqual(prefetch_capacity_limit_for(8192), int(0.9 * 8192))
        self.assertEqual(prefetch_capacity_limit_for(0), 0)

    def test_timeout_pricing_covers_a_16k_readmit_at_measured_rate(self):
        """#1065: measured best-case store rate ~0.32 s/KiToken; the linear
        default must price at or above it and the max must not clip a 16k
        readmit."""
        from sglang.srt.mem_cache.hicache_storage import PrefetchTimeoutConfig

        cfg = PrefetchTimeoutConfig()
        self.assertGreaterEqual(cfg.per_ki_token, 0.32)
        self.assertGreaterEqual(cfg.max, cfg.base + cfg.per_ki_token * 16.0)


if __name__ == "__main__":
    unittest.main()
