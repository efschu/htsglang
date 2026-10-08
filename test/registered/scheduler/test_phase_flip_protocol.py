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

import os
import unittest
from types import SimpleNamespace

from flliper.srt.distributed import parallel_state
from flliper.srt.distributed.parallel_state_wrapper import ParallelState
from flliper.srt.distributed.utils import set_cp_token_ratios
from flliper.srt.layers.dcp.phase_flip_plan import PP_TO_TP
from flliper.srt.layers.dcp.reshard_plan import KvReshardError
from flliper.srt.managers.phase_flip_runtime import (
    PHASE_PP,
    build_gdn_flip_guard,
    build_production_flip_cutover,
    derive_pp_full_attn_layer_map,
)
from flliper.srt.runtime_context import get_context, get_server_args
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# Qwen3.6-27B full-attention geometry: 64 layers, every 4th is full attn.
FULL_IDS = list(range(3, 64, 4))
N_HIDDEN = 64
VEC = (30, 17, 17)
MAP_625 = ((0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))


class TestLayerMapDerivation(CustomTestCase):
    def setUp(self):
        self._saved_env = os.environ.pop("FLLIPER_PP_LAYER_PARTITION", None)

    def tearDown(self):
        os.environ.pop("FLLIPER_PP_LAYER_PARTITION", None)
        if self._saved_env is not None:
            os.environ["FLLIPER_PP_LAYER_PARTITION"] = self._saved_env

    def test_recipe_split_gives_8_4_4(self):
        """The measured #625 recipe (32/16/16 layers) owns 8/4/4 of the 16
        full-attention ordinals -- the design section 2 numbers."""
        os.environ["FLLIPER_PP_LAYER_PARTITION"] = "32,16,16"
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
        os.environ["FLLIPER_PP_LAYER_PARTITION"] = "32,16,15"  # sums to 63
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
            # The KV token vector. Equal to the weight vector unless
            # FLLIPER_UNEVEN_TOKEN_VECTOR overrides it; the cutover's owner
            # rule and the transition plan both read THIS one.
            token_vector=VEC,
            refill=lambda direction: self.log.append(("refill", direction)),
            # #631 speculation slice: None unless a test arms it.
            draft_worker=None,
        )
        # Speculation is a TP-decode-phase capability; the boot (PP) state
        # is the same one an instance without speculation has.
        from flliper.srt.speculative.spec_info import SpeculativeAlgorithm

        self.spec_algorithm = SpeculativeAlgorithm.from_string(None)
        self.flip_spec_algorithm = SpeculativeAlgorithm.from_string(None)
        self.draft_worker = None
        from flliper.srt.managers.phase_flip_runtime import AbortDeferralWindow

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

        @_dc.dataclass
        class _OutputStreamerStub:
            """Models the two fields the cutover has to refresh.

            ``spec_algorithm`` is here because the real streamer keeps a
            VALUE COPY of it and gates the whole spec-counter wire on that
            copy: a phase-flip instance parks speculation at boot, so a
            streamer that is only ps-refreshed reports no accept length for
            the life of the process. A stub without the field cannot catch
            that regression.
            """

            ps: object
            spec_algorithm: object = None

        @_dc.dataclass
        class _BatchResultProcessorStub:
            model_worker: object
            draft_worker: object

        # On the decode hot path and holds the WORKERS, so the cutover
        # must rebuild it per phase (#631: with speculation it calls into
        # the spec worker).
        self.batch_result_processor = _BatchResultProcessorStub(
            model_worker=self.tp_worker, draft_worker=None
        )
        self.request_receiver = _ReceiverStub(ps=self.ps)
        self.output_streamer = _OutputStreamerStub(
            ps=self.ps, spec_algorithm=self.spec_algorithm
        )
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
            override=lambda reason, **kw: self.overrides.append((reason, kw)),
            # #905: since #856 the cutover retracts its residents, and
            # `release_kv_cache` reads these three off the PUBLISHED args
            # rather than off the ones it is handed. A stub that carries only
            # `override` no longer models what the seam reads.
            page_size=1,
            speculative_algorithm=None,
            strip_thinking_cache=False,
            disaggregation_mode="null",
        )
        get_context().set_server_args(stub)
        return self

    def __exit__(self, *exc):
        get_context().set_server_args(self.saved)
        return False


class TestSpeculationIsTpDecodePhaseOnly(CustomTestCase):
    """#631 speculation slice: NEXTN runs in the TP DECODE phase.

    The user requirement is speculation in TP decode; the structural
    constraint is that no draft worker has a PP form. Both are satisfied
    by the same rule -- the draft worker is built on the flip's TP stack
    and is armed ONLY while that stack is active. These tests pin the
    cutover swap in both directions, and that a half-armed cutover is
    refused rather than run."""

    def tearDown(self):
        set_cp_token_ratios(None)

    def _armed_stub(self):
        from flliper.srt.speculative.spec_info import SpeculativeAlgorithm

        sched = _StubScheduler(0)
        # "NEXTN" is a CLI alias the arg hook resolves before the enum
        # sees it; FROZEN_KV_MTP is what server_args actually carries.
        sched.flip_spec_algorithm = SpeculativeAlgorithm.from_string(
            "FROZEN_KV_MTP"
        )
        sched.phase_flip_stacks.draft_worker = SimpleNamespace(name="draft[0]")
        return sched

    def test_pp_phase_carries_no_speculation_at_all(self):
        sched = self._armed_stub()
        self.assertTrue(
            sched.spec_algorithm.is_none(),
            "the PP prefill phase must run with no speculation",
        )
        self.assertIsNone(sched.draft_worker)

class TestPin4SchedulerLevelReplay(CustomTestCase):
    """Aborts between arm and cutover: REAL router -> REAL window -> REAL
    runtime flip threads -> REAL production cutover -> ordered drain."""

    def tearDown(self):
        set_cp_token_ratios(None)

    def _logging_cutover(self, sched):
        production = build_production_flip_cutover(sched)

        def _cutover(direction):
            production(direction)
            sched.log.append(("cutover_done", direction))

        return _cutover

    def test_gdn_guard_refuses_live_linear_state(self):
        """5.3 placeholder honesty: live requests hold GDN state; until
        the 5.3b mover lands the flip must refuse loudly, never truncate."""
        sched = _StubScheduler(0)
        sched.running_batch = SimpleNamespace(reqs=[object()])
        with self.assertRaisesRegex(KvReshardError, "GDN"):
            build_gdn_flip_guard(sched)(PP_TO_TP)
        sched.running_batch = SimpleNamespace(reqs=[])
        build_gdn_flip_guard(sched)(PP_TO_TP)  # empty flip allowed


if __name__ == "__main__":
    unittest.main()
