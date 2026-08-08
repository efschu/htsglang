# SPDX-License-Identifier: Apache-2.0
"""#631 slice 5.3, the scheduler flip protocol: hermetic tests (CPU-only).

The load-bearing gates, mapped to DESIGN_631 5.3 and the operator's 5.3
acceptance pins:

* PIN-4 SCHEDULER-LEVEL REPLAY: aborts arriving between arm and cutover go
  through the REAL Scheduler.abort_request router into the REAL
  AbortDeferralWindow while REAL PhaseFlipRuntime threads flip through the
  REAL production cutover; the deferred aborts apply IN ORDER, strictly
  AFTER cutover, on every rank, and the window ends inactive. (The full
  event loop needs cards; this is the hermetic maximum -- every flip-owned
  code object in the chain is the production one.)
* CUTOVER COMPLETENESS CAN-FAIL: verify_flip_cutover runs as the
  cutover's last step; each red arm reverts EXACTLY ONE rebuilt reference
  (a cached group handle, the ps topology, the model_worker) and the
  checker must fail loudly naming it -- proving a missed rebuild can
  never survive silently.
* layer-map derivation is a pure replicated function with red arms
  (non-partitioning stage bounds, out-of-range ordinals), and reproduces
  both the even split and the pinned 32/16/16 -> 8/4/4 recipe geometry.
* the event-loop wrapper re-dispatches per phase on PhaseFlipLoopExit and
  returns on normal loop return; maybe_sleep_on_idle keeps ticking rounds
  while a flip is pending (the #297 parked-loop lesson).
"""

import dataclasses
import os
import threading
import unittest
from types import SimpleNamespace

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.distributed.utils import set_cp_token_ratios
from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP
from sglang.srt.layers.dcp.reshard_plan import KvReshardError
from sglang.srt.managers.phase_flip_runtime import (
    PHASE_PP,
    PHASE_TP,
    PhaseFlipLoopExit,
    PhaseFlipRuntime,
    build_gdn_flip_guard,
    build_production_flip_cutover,
    derive_pp_full_attn_layer_map,
    verify_flip_cutover,
)
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_phase_flip_runtime import (  # noqa: E402  (sibling harness)
    _BarrierMinChannel,
    _MailboxExchange,
    _make_layout_pools,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# Qwen3.6-27B full-attention geometry: 64 layers, every 4th is full attn.
FULL_IDS = list(range(3, 64, 4))
N_HIDDEN = 64
VEC = (30, 17, 17)
MAP_625 = ((0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))


class TestLayerMapDerivation(CustomTestCase):
    def setUp(self):
        self._saved_env = os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)

    def tearDown(self):
        os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
        if self._saved_env is not None:
            os.environ["SGLANG_PP_LAYER_PARTITION"] = self._saved_env

    def test_recipe_split_gives_8_4_4(self):
        """The measured #625 recipe (32/16/16 layers) owns 8/4/4 of the 16
        full-attention ordinals -- the design section 2 numbers."""
        os.environ["SGLANG_PP_LAYER_PARTITION"] = "32,16,16"
        layer_map = derive_pp_full_attn_layer_map(FULL_IDS, N_HIDDEN, 3)
        self.assertEqual(layer_map, MAP_625)

    def test_even_split_covers_exactly_once(self):
        layer_map = derive_pp_full_attn_layer_map(FULL_IDS, N_HIDDEN, 3)
        flat = sorted(o for stage in layer_map for o in stage)
        self.assertEqual(flat, list(range(16)))
        self.assertEqual(len(layer_map), 3)

    def test_can_fail_unsorted_ids_refused(self):
        with self.assertRaisesRegex(KvReshardError, "ascending"):
            derive_pp_full_attn_layer_map([7, 3, 11], N_HIDDEN, 3)

    def test_can_fail_out_of_range_ids_refused(self):
        with self.assertRaisesRegex(KvReshardError, "outside"):
            derive_pp_full_attn_layer_map([3, 7, 64], N_HIDDEN, 3)

    def test_can_fail_bad_partition_env_refused(self):
        os.environ["SGLANG_PP_LAYER_PARTITION"] = "32,16,15"  # sums to 63
        with self.assertRaises(ValueError):
            derive_pp_full_attn_layer_map(FULL_IDS, N_HIDDEN, 3)


class _SentinelGroup:
    def __init__(self, name):
        self.name = name
        self.cpu_group = SimpleNamespace(kind=f"{name}.cpu")
        self.device_group = SimpleNamespace(kind=f"{name}.device")


def _boot_ps(rank):
    return ParallelState(
        tp_rank=0,
        tp_size=1,
        pp_rank=rank,
        pp_size=3,
        dp_rank=None,
        dp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        attn_dp_rank=0,
        attn_dp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        moe_dp_rank=None,
        moe_dp_size=1,
        gpu_id=rank,
    )


class _StubScheduler:
    """Attribute shell driven through the REAL Scheduler methods and the
    REAL cutover/verify/window/runtime objects."""

    def __init__(self, rank):
        self.rank = rank
        self.ps = _boot_ps(rank)
        self.max_running_requests = 4
        self.tp_worker = SimpleNamespace(name=f"pp_worker[{rank}]")
        self.model_worker = self.tp_worker
        self.phase_flip_stacks = SimpleNamespace(
            tp_worker=SimpleNamespace(name=f"tp_worker[{rank}]"),
            vector=VEC,
            refill=lambda direction: self.log.append(("refill", direction)),
        )
        from sglang.srt.managers.phase_flip_runtime import AbortDeferralWindow

        self.phase_flip_abort_window = AbortDeferralWindow()
        self.phase_flip_active_stack = PHASE_PP
        self.phase_flip_runtime = None
        self.running_batch = SimpleNamespace(reqs=[])
        self.log = []
        # ancillary attrs the cutover touches
        self.tp_group = None
        self.tp_cpu_group = None
        self.attn_tp_group = None
        self.attn_tp_cpu_group = None
        self.pp_group = None
        self.dp_tp_group = None
        # step-4b component holders (rebuilt via dataclasses.replace, so
        # they must be real dataclasses carrying the touched fields)
        import dataclasses as _dc

        @_dc.dataclass
        class _ReceiverStub:
            ps: object
            tp_group: object = None
            tp_cpu_group: object = None
            attn_tp_group: object = None
            attn_tp_cpu_group: object = None

        @_dc.dataclass
        class _PsHolderStub:
            ps: object

        self.request_receiver = _ReceiverStub(ps=self.ps)
        self.output_streamer = _PsHolderStub(ps=self.ps)
        self.load_inquirer = _PsHolderStub(ps=self.ps)

    def init_pp_loop_state(self):
        self.log.append(("init_pp_loop_state", self.ps.pp_size))

    def _abort_request_now(self, recv_req):
        # Record the active stack AT APPLY TIME: a deferred abort must run
        # only after the cutover rebuilds switched the phase (pin 4).
        self.log.append(
            ("abort_applied", recv_req.rid, self.phase_flip_active_stack)
        )


class _ParallelStatePatch:
    GLOBALS = (
        "_FLIP_TP",
        "_FLIP_DCP",
        "_FLIP_PP",
        "_TP",
        "_ATTN_TP",
        "_DCP",
        "_PP",
        "_WORLD",
        "_PHASE_FLIP_TP_ACTIVE",
    )

    def __enter__(self):
        self.saved = {g: getattr(parallel_state, g) for g in self.GLOBALS}
        self.flip_tp = _SentinelGroup("flip_tp")
        self.flip_dcp = _SentinelGroup("flip_dcp")
        self.flip_pp = _SentinelGroup("flip_pp")
        self.tp = _SentinelGroup("tp")
        self.attn_tp = _SentinelGroup("attn_tp")
        self.dcp = _SentinelGroup("dcp")
        self.pp = _SentinelGroup("pp")
        world = SimpleNamespace(
            world_size=3, rank_in_group=0, cpu_group=SimpleNamespace()
        )
        parallel_state._FLIP_TP = self.flip_tp
        parallel_state._FLIP_DCP = self.flip_dcp
        parallel_state._FLIP_PP = self.flip_pp
        parallel_state._TP = self.tp
        parallel_state._ATTN_TP = self.attn_tp
        parallel_state._DCP = self.dcp
        parallel_state._PP = self.pp
        parallel_state._WORLD = world
        parallel_state._PHASE_FLIP_TP_ACTIVE = False
        return self

    def __exit__(self, *exc):
        for g, v in self.saved.items():
            setattr(parallel_state, g, v)
        return False


class _PublishedArgsPatch:
    """Publish a stub ServerArgs (override() recorder) for the cutover."""

    def __enter__(self):
        try:
            self.saved = get_server_args()
        except ValueError:
            self.saved = None
        self.overrides = []
        stub = SimpleNamespace(
            override=lambda reason, **kw: self.overrides.append((reason, kw))
        )
        get_context().set_server_args(stub)
        return self

    def __exit__(self, *exc):
        get_context().set_server_args(self.saved)
        return False


class TestProductionCutover(CustomTestCase):
    def tearDown(self):
        set_cp_token_ratios(None)

    def test_cutover_rebuild_list_and_return_trip(self):
        with _ParallelStatePatch() as psp, _PublishedArgsPatch() as pap:
            sched = _StubScheduler(0)
            cutover = build_production_flip_cutover(sched)

            cutover(PP_TO_TP)
            self.assertEqual(sched.ps.tp_size, 3)
            self.assertEqual(sched.ps.pp_size, 1)
            self.assertEqual(sched.ps.attn_tp_size, 3)
            self.assertIs(sched.tp_group, psp.flip_tp)
            self.assertIs(sched.tp_cpu_group, psp.flip_tp.cpu_group)
            self.assertIs(sched.attn_tp_group, psp.flip_tp)
            self.assertIs(sched.pp_group, psp.flip_pp)
            self.assertIs(sched.dp_tp_group, psp.flip_tp)
            self.assertIs(sched.model_worker, sched.phase_flip_stacks.tp_worker)
            self.assertEqual(sched.phase_flip_active_stack, PHASE_TP)
            self.assertIn(("init_pp_loop_state", 1), sched.log)
            self.assertEqual(
                pap.overrides[-1][1]["pp_max_micro_batch_size"], 4
            )

            cutover(TP_TO_PP)
            self.assertEqual(sched.ps.tp_size, 1)
            self.assertEqual(sched.ps.pp_size, 3)
            self.assertIs(sched.tp_group, psp.tp)
            self.assertIs(sched.attn_tp_group, psp.attn_tp)
            self.assertIs(sched.pp_group, psp.pp)
            self.assertIs(sched.model_worker, sched.tp_worker)
            self.assertEqual(sched.phase_flip_active_stack, PHASE_PP)
            self.assertIn(("init_pp_loop_state", 3), sched.log)
            self.assertEqual(
                pap.overrides[-1][1]["pp_max_micro_batch_size"], 1
            )

    def test_can_fail_each_single_stale_reference_is_caught(self):
        """The coordinator's completeness pin: revert ONE rebuilt item
        after a good cutover; verify_flip_cutover must fail red NAMING it."""
        with _ParallelStatePatch() as psp, _PublishedArgsPatch():
            sched = _StubScheduler(0)
            cutover = build_production_flip_cutover(sched)
            cutover(PP_TO_TP)
            verify_flip_cutover(sched, tp_phase=True)  # green baseline

            saboteurs = {
                "tp_group": lambda: setattr(sched, "tp_group", psp.tp),
                "attn_tp_group": lambda: setattr(
                    sched, "attn_tp_group", psp.attn_tp
                ),
                "pp_group": lambda: setattr(sched, "pp_group", psp.pp),
                "ps topology": lambda: setattr(
                    sched, "ps", dataclasses.replace(sched.ps, pp_size=3)
                ),
                "model_worker": lambda: setattr(
                    sched, "model_worker", sched.tp_worker
                ),
                "abort window": lambda: sched.phase_flip_abort_window.activate(),
            }
            for name, sabotage in saboteurs.items():
                saved = (
                    sched.tp_group,
                    sched.attn_tp_group,
                    sched.pp_group,
                    sched.ps,
                    sched.model_worker,
                )
                sabotage()
                with self.assertRaisesRegex(
                    KvReshardError, "CUTOVER INCOMPLETE", msg=name
                ):
                    verify_flip_cutover(sched, tp_phase=True)
                (
                    sched.tp_group,
                    sched.attn_tp_group,
                    sched.pp_group,
                    sched.ps,
                    sched.model_worker,
                ) = saved
                sched.phase_flip_abort_window.deactivate_and_drain()
                verify_flip_cutover(sched, tp_phase=True)  # green again

    def test_can_fail_stale_routing_flag_is_caught(self):
        with _ParallelStatePatch(), _PublishedArgsPatch():
            sched = _StubScheduler(0)
            build_production_flip_cutover(sched)(PP_TO_TP)
            parallel_state._PHASE_FLIP_TP_ACTIVE = False  # sabotage routing
            with self.assertRaisesRegex(KvReshardError, "CUTOVER INCOMPLETE"):
                verify_flip_cutover(sched, tp_phase=True)


class TestPin4SchedulerLevelReplay(CustomTestCase):
    """Aborts between arm and cutover: REAL router -> REAL window -> REAL
    runtime flip threads -> REAL production cutover -> ordered drain."""

    def tearDown(self):
        set_cp_token_ratios(None)

    def test_deferred_aborts_apply_in_order_after_cutover_on_all_ranks(self):
        from sglang.srt.managers.scheduler import Scheduler

        n = len(VEC)
        with _ParallelStatePatch(), _PublishedArgsPatch():
            _, live, _, pp_views, _, tp_views = _make_layout_pools(
                MAP_625, list(VEC), num_slots=64
            )
            channel = _BarrierMinChannel(n)
            mailbox = _MailboxExchange(n)
            scheds = [_StubScheduler(r) for r in range(n)]
            for r, sched in enumerate(scheds):
                sched.phase_flip_runtime = PhaseFlipRuntime(
                    n_ranks=n,
                    rank=r,
                    layer_map=MAP_625,
                    n_layers=16,
                    tp_vector=VEC,
                    boot_phase=PHASE_PP,
                    consensus_interval=2,
                    collective_min=channel.channel_for(r),
                    exchange=mailbox.exchange_for(r),
                    pp_pool_view=pp_views[r],
                    tp_pool_view=tp_views[r],
                    live_slots_fn=lambda live=live: live,
                    ready_fn=lambda: True,
                    cutover_fn=self._logging_cutover(sched),
                    pre_cutover_fns=(
                        build_gdn_flip_guard(sched),
                        sched.phase_flip_stacks.refill,
                    ),
                )

            errors = [None] * n

            def _rank(r):
                sched = scheds[r]
                try:
                    ok, msg = Scheduler.arm_phase_flip(
                        sched, PP_TO_TP, source=f"pin4-rank{r}"
                    )
                    assert ok, msg
                    # Aborts arrive while the flip is armed: the REAL
                    # router must defer them, in order.
                    Scheduler.abort_request(
                        sched, SimpleNamespace(rid=f"req-A{r}")
                    )
                    Scheduler.abort_request(
                        sched, SimpleNamespace(rid=f"req-B{r}")
                    )
                    for _ in range(4):
                        sched.phase_flip_runtime.on_round()
                except BaseException as e:  # noqa: BLE001
                    errors[r] = e

            threads = [
                threading.Thread(target=_rank, args=(r,)) for r in range(n)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)
            self.assertFalse(
                [t for t in threads if t.is_alive()], "flip hang"
            )
            for r, e in enumerate(errors):
                self.assertIsNone(e, f"rank {r}: {e!r}")

            for r, sched in enumerate(scheds):
                self.assertEqual(sched.phase_flip_runtime.phase, PHASE_TP)
                self.assertEqual(sched.phase_flip_runtime.completed, 1)
                self.assertFalse(sched.phase_flip_abort_window.active)
                self.assertIs(
                    sched.model_worker, sched.phase_flip_stacks.tp_worker
                )
                # Order: the arena refill runs first (pre-cutover seam);
                # both aborts drain in submission order, and each applied
                # with the TP stack already active -- the drain is the
                # cutover's LAST rebuild step, so apply-time phase == tp
                # proves "after the rebuilds", not merely "eventually".
                events = [
                    ev
                    for ev in sched.log
                    if ev[0] in ("refill", "abort_applied")
                ]
                self.assertEqual(
                    events,
                    [
                        ("refill", PP_TO_TP),
                        ("abort_applied", f"req-A{r}", PHASE_TP),
                        ("abort_applied", f"req-B{r}", PHASE_TP),
                    ],
                )
                # And the wrapper's marker confirms cutover completed.
                self.assertIn(("cutover_done", PP_TO_TP), sched.log)

    def _logging_cutover(self, sched):
        production = build_production_flip_cutover(sched)

        def _cutover(direction):
            production(direction)
            sched.log.append(("cutover_done", direction))

        return _cutover

    def test_abort_refused_arm_drains_immediately(self):
        """A refused arm must not leave the window active (stuck deferral
        starves every later abort)."""
        from sglang.srt.managers.scheduler import Scheduler

        sched = _StubScheduler(0)
        sched.phase_flip_runtime = SimpleNamespace(
            arm=lambda direction, source: (False, "guards"),
            pending=None,
        )
        ok, _ = Scheduler.arm_phase_flip(sched, PP_TO_TP, source="t")
        self.assertFalse(ok)
        self.assertFalse(sched.phase_flip_abort_window.active)
        Scheduler.abort_request(sched, SimpleNamespace(rid="direct"))
        self.assertIn(("abort_applied", "direct", PHASE_PP), sched.log)

    def test_gdn_guard_refuses_live_linear_state(self):
        """5.3 placeholder honesty: live requests hold GDN state; until
        the 5.3b mover lands the flip must refuse loudly, never truncate."""
        sched = _StubScheduler(0)
        sched.running_batch = SimpleNamespace(reqs=[object()])
        with self.assertRaisesRegex(KvReshardError, "GDN"):
            build_gdn_flip_guard(sched)(PP_TO_TP)
        sched.running_batch = SimpleNamespace(reqs=[])
        build_gdn_flip_guard(sched)(PP_TO_TP)  # empty flip allowed


class TestEventLoopRedispatch(CustomTestCase):
    def test_wrapper_redispatches_per_phase_then_returns(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler import run_phase_flip_event_loops

        calls = []
        sched = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.NULL,
            enable_pdmux=False,
            enable_overlap=True,
            phase_flip_active_stack=PHASE_PP,
        )

        def _pp_loop():
            calls.append("pp")
            sched.phase_flip_active_stack = PHASE_TP  # what cutover does
            raise PhaseFlipLoopExit(PP_TO_TP)

        def _overlap_loop():
            calls.append("overlap")
            if len(calls) < 4:
                sched.phase_flip_active_stack = PHASE_PP
                raise PhaseFlipLoopExit(TP_TO_PP)
            return None  # shutdown

        def _pp_loop_second():
            calls.append("pp")
            sched.phase_flip_active_stack = PHASE_TP
            raise PhaseFlipLoopExit(PP_TO_TP)

        sched.event_loop_pp = _pp_loop
        sched.event_loop_overlap = _overlap_loop
        sched.event_loop_normal = lambda: calls.append("normal")
        run_phase_flip_event_loops(sched)
        self.assertEqual(calls, ["pp", "overlap", "pp", "overlap"])

    def test_wrapper_refuses_disaggregation(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler import run_phase_flip_event_loops

        sched = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            enable_pdmux=False,
        )
        with self.assertRaises(AssertionError):
            run_phase_flip_event_loops(sched)


class TestPhaseFlipRpc(CustomTestCase):
    """The /phase_flip control plane (5.5 ladder rung 2 prerequisite):
    the REAL Scheduler.handle_phase_flip on stubs -- flag off refuses
    naming the flag, pre-round arming reports not-built, an armed verdict
    passes through, an arm exception is wrapped, never raised."""

    def _handle(self, sched, direction="pp_to_tp"):
        from sglang.srt.managers.io_struct import PhaseFlipReqInput
        from sglang.srt.managers.scheduler import Scheduler

        return Scheduler.handle_phase_flip(
            sched, PhaseFlipReqInput(direction=direction)
        )

    def test_flag_off_refuses_naming_the_flag(self):
        sched = SimpleNamespace(
            server_args=SimpleNamespace(enable_phase_flip=False)
        )
        out = self._handle(sched)
        self.assertFalse(out.success)
        self.assertIn("--enable-phase-flip", out.message)

    def test_pre_round_arm_reports_not_built(self):
        from sglang.srt.managers.scheduler import Scheduler

        sched = _StubScheduler(0)
        sched.server_args = SimpleNamespace(enable_phase_flip=True)
        sched.arm_phase_flip = lambda d, source: Scheduler.arm_phase_flip(
            sched, d, source
        )
        out = self._handle(sched)
        self.assertFalse(out.success)
        self.assertIn("not built yet", out.message)

    def test_armed_verdict_passes_through(self):
        sched = SimpleNamespace(
            server_args=SimpleNamespace(enable_phase_flip=True),
            arm_phase_flip=lambda d, source: (True, f"armed {d} ({source})"),
        )
        out = self._handle(sched, "tp_to_pp")
        self.assertTrue(out.success)
        self.assertIn("armed tp_to_pp (rpc)", out.message)

    def test_arm_exception_is_wrapped_not_raised(self):
        def _boom(d, source):
            raise RuntimeError("wiring hole")

        sched = SimpleNamespace(
            server_args=SimpleNamespace(enable_phase_flip=True),
            arm_phase_flip=_boom,
        )
        out = self._handle(sched)
        self.assertFalse(out.success)
        self.assertIn("wiring hole", out.message)


class TestSleepOnIdleSkip(CustomTestCase):
    def test_pending_flip_keeps_loop_ticking(self):
        """A parked loop never reaches a consensus boundary (#297 lesson):
        with a pending flip the sleeper must return early. The stub lacks
        every attribute AFTER the flip check -- falling through would
        AttributeError, which is the red proof the early return fired."""
        from sglang.srt.managers.scheduler import Scheduler

        sched = SimpleNamespace(
            kv_reshard_runtime=None,
            phase_flip_runtime=SimpleNamespace(pending=PP_TO_TP),
        )
        self.assertIsNone(Scheduler.maybe_sleep_on_idle(sched))


if __name__ == "__main__":
    unittest.main()
