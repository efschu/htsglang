"""CPU tests for the adaptive graph-memory offload layer (#93).

Covers mode resolution, manager bookkeeping (tagging, pause/resume ordering,
zero-on-resume, no-OOM reserve check, segment-isolation audit), controller
integration (build order, swap path, forced-swap stress knob), and the
high-accept (k=4/5) ladder profile. All CUDA interaction is mocked; the
GPU-side behavior is exercised in the T93 GPU validation phase.
"""

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.speculative.adaptive_graph_memory as agm
from sglang.srt.speculative.adaptive_graph_memory import (
    AdaptiveGraphMemoryManager,
    resolve_adaptive_graph_memory_mode,
)
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveStepSlot,
    HIGH_ACCEPT_ADAPTIVE_CONFIG,
    resolve_candidate_steps_from_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _server_args(**overrides):
    base = dict(
        speculative_adaptive=True,
        speculative_adaptive_graph_memory="auto",
        device="cuda",
        attention_backend="flashinfer",
        decode_attention_backend=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeAdapter:
    """Records region/pause/resume calls; never touches CUDA."""

    def __init__(self):
        self.calls = []
        self.region_tags = []

    @contextlib.contextmanager
    def region(self, tag, enable_cpu_backup=False):
        self.region_tags.append(tag)
        self.calls.append(("region", tag))
        yield

    def pause(self, tag):
        self.calls.append(("pause", tag))

    def resume(self, tag):
        self.calls.append(("resume", tag))


@contextlib.contextmanager
def _mock_cuda(free_bytes=1 << 40, snapshots=None):
    """Neutralize every torch.cuda call the manager makes (no GPU touch)."""
    snap_iter = iter(snapshots) if snapshots is not None else None

    def _snap():
        if snap_iter is None:
            return {}
        try:
            return next(snap_iter)
        except StopIteration:
            return {}

    with (
        mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None),
        mock.patch.object(torch.cuda, "empty_cache", lambda *a, **k: None),
        mock.patch.object(
            torch.cuda, "mem_get_info", lambda *a, **k: (free_bytes, 1 << 41)
        ),
        mock.patch.object(agm, "_snapshot_segment_addrs", _snap),
    ):
        yield


def _offload_manager():
    """Manager in offload mode with a FakeAdapter installed."""
    with mock.patch.dict(
        os.environ, {"LD_PRELOAD": "/x/torch_memory_saver_hook_mode_preload.so"}
    ):
        with mock.patch.object(
            AdaptiveGraphMemoryManager,
            "__init__",
            _offload_init,
        ):
            return AdaptiveGraphMemoryManager()


def _offload_init(self):
    # Mirrors the real __init__ minus the TorchMemorySaverAdapter creation.
    self.mode = "offload"
    self._states = {}
    self._paused = set()
    self._resumed_tag = None
    self._build_tag = None
    self._pre_build_segments = None
    self._finalized = False
    self._swap_ordinal = 0
    self.last_swap_ms = None
    self._tp_cpu_group = None
    self._adapter = FakeAdapter()
    agm._ACTIVE_MANAGER = self


class TestModeResolution(unittest.TestCase):
    def test_not_adaptive_is_resident(self):
        args = _server_args(speculative_adaptive=False)
        self.assertEqual(resolve_adaptive_graph_memory_mode(args), "resident")

    def test_explicit_resident(self):
        args = _server_args(speculative_adaptive_graph_memory="resident")
        self.assertEqual(resolve_adaptive_graph_memory_mode(args), "resident")

    def test_auto_with_prereqs_is_offload(self):
        # torch_memory_saver is installed in the dev venv.
        with mock.patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": ""}):
            self.assertEqual(
                resolve_adaptive_graph_memory_mode(_server_args()), "offload"
            )

    def test_auto_degrades_on_non_flashinfer(self):
        args = _server_args(attention_backend="triton")
        self.assertEqual(resolve_adaptive_graph_memory_mode(args), "resident")

    def test_auto_degrades_on_non_cuda_device(self):
        args = _server_args(device="cpu")
        self.assertEqual(resolve_adaptive_graph_memory_mode(args), "resident")

    def test_auto_degrades_on_expandable_segments(self):
        with mock.patch.dict(
            os.environ,
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        ):
            self.assertEqual(
                resolve_adaptive_graph_memory_mode(_server_args()), "resident"
            )

    def test_explicit_offload_with_missing_prereq_raises(self):
        args = _server_args(
            speculative_adaptive_graph_memory="offload",
            attention_backend="triton",
        )
        with self.assertRaisesRegex(ValueError, "flashinfer required"):
            resolve_adaptive_graph_memory_mode(args)

    def test_invalid_mode_raises(self):
        args = _server_args(speculative_adaptive_graph_memory="banana")
        with self.assertRaises(ValueError):
            resolve_adaptive_graph_memory_mode(args)


class TestManagerResident(unittest.TestCase):
    def test_everything_is_a_noop(self):
        mgr = AdaptiveGraphMemoryManager(mode="resident")
        with mgr.build_state(2):
            t = torch.zeros(1024)
            mgr.note_tensor(t)
        self.assertEqual(mgr._states, {})
        mgr.pause_after_build(2)
        mgr.finalize_boot(3)
        mgr.ensure_active(2)  # must not require CUDA / adapter
        self.assertEqual(mgr.swap_count, 0)
        self.assertFalse(mgr.is_paused_tensor(t))


class TestManagerOffloadGuard(unittest.TestCase):
    def test_missing_ld_preload_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "LD_PRELOAD"):
                AdaptiveGraphMemoryManager(mode="offload")


class TestManagerOffload(unittest.TestCase):
    def _build_two_states(self, mgr):
        tensors = {}
        with _mock_cuda():
            for steps in (2, 1):
                with mgr.build_state(steps):
                    with agm.tagged_state_alloc():
                        t = torch.full((1024,), 7, dtype=torch.int32)
                    agm.note_state_tensor(t)
                    tensors[steps] = t
                mgr.pause_after_build(steps)
        return tensors

    def test_build_tags_and_pause_after_build(self):
        mgr = _offload_manager()
        tensors = self._build_two_states(mgr)
        self.assertEqual(
            mgr._adapter.region_tags,
            ["adaptive_state_k2", "adaptive_state_k1"],
        )
        self.assertEqual(
            [c for c in mgr._adapter.calls if c[0] == "pause"],
            [("pause", "adaptive_state_k2"), ("pause", "adaptive_state_k1")],
        )
        self.assertTrue(mgr.is_paused_tensor(tensors[1]))
        self.assertTrue(mgr.is_paused_tensor(tensors[2]))

    def test_swap_pauses_old_before_resuming_new_and_zeroes(self):
        mgr = _offload_manager()
        tensors = self._build_two_states(mgr)
        with _mock_cuda():
            mgr.finalize_boot(initial_steps=3)  # baseline: nothing mapped
            mgr._adapter.calls.clear()

            mgr.ensure_active(2)
            self.assertEqual(mgr._adapter.calls, [("resume", "adaptive_state_k2")])
            # zero-on-resume restores the boot content contract
            self.assertTrue(torch.all(tensors[2] == 0))
            self.assertFalse(mgr.is_paused_tensor(tensors[2]))
            self.assertTrue(mgr.is_paused_tensor(tensors[1]))

            mgr._adapter.calls.clear()
            mgr.ensure_active(1)
            self.assertEqual(
                mgr._adapter.calls,
                [("pause", "adaptive_state_k2"), ("resume", "adaptive_state_k1")],
            )
            self.assertTrue(torch.all(tensors[1] == 0))
            self.assertIsNotNone(mgr.last_swap_ms)

            # Back to an untagged baseline: outgoing state paused, no resume.
            mgr._adapter.calls.clear()
            mgr.ensure_active(3)
            self.assertEqual(mgr._adapter.calls, [("pause", "adaptive_state_k1")])
            self.assertEqual(mgr.swap_count, 3)

    def test_ensure_active_is_idempotent(self):
        mgr = _offload_manager()
        self._build_two_states(mgr)
        with _mock_cuda():
            mgr.finalize_boot(initial_steps=2)
            n = mgr.swap_count
            mgr._adapter.calls.clear()
            mgr.ensure_active(2)
            self.assertEqual(mgr._adapter.calls, [])
            self.assertEqual(mgr.swap_count, n)

    def test_finalize_boot_no_oom_guarantee(self):
        mgr = _offload_manager()
        self._build_two_states(mgr)
        # 1024 * int32 = 4096 bytes per state; free memory below that fails.
        with _mock_cuda(free_bytes=1024):
            with self.assertRaisesRegex(RuntimeError, "k-swap could OOM"):
                mgr.finalize_boot(initial_steps=3)

    def test_audit_detects_cross_tag_segment(self):
        mgr = _offload_manager()
        tensors = self._build_two_states(mgr)
        rec1 = mgr._states["adaptive_state_k1"]
        rec2 = mgr._states["adaptive_state_k2"]
        # Fabricate: k2's build window owns the segment holding k1's tensor.
        ptr = tensors[1].data_ptr()
        rec2.segment_ranges = [(ptr - 16, ptr + 16)]
        rec1.segment_ranges = []
        live = {ptr - 16: 32}
        with _mock_cuda(snapshots=[live, live, live]):
            with self.assertRaisesRegex(RuntimeError, "unmap another"):
                mgr.finalize_boot(initial_steps=3)

    def test_audit_ignores_recycled_segments(self):
        mgr = _offload_manager()
        tensors = self._build_two_states(mgr)
        rec2 = mgr._states["adaptive_state_k2"]
        ptr = tensors[1].data_ptr()
        # Same overlap, but the segment no longer exists (recycled VA range).
        rec2.segment_ranges = [(ptr - 16, ptr + 16)]
        mgr._states["adaptive_state_k1"].segment_ranges = []
        with _mock_cuda(snapshots=[{}]):
            mgr.finalize_boot(initial_steps=3)  # must not raise


class _BuildRecordingWorker:
    """Minimal AdaptiveSpecWorker capturing build order."""

    speculative_num_steps = 3

    def __init__(self):
        self.build_order = []
        self.applied = []

    def build_adaptive_runtime_state(
        self, speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs=None
    ):
        self.build_order.append(speculative_num_steps)
        return SpecRuntimeState(
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            draft_attn_backend=SimpleNamespace(),
            cuda_graph_runner=None,
            target_attn_backend=SimpleNamespace(),
            target_graph_runner=None,
            draft_extend_attn_backend=None,
            cuda_graph_runner_for_draft_extend=None,
        )

    def apply_runtime_state(self, state):
        self.applied.append(state.speculative_num_steps)
        self.speculative_num_steps = state.speculative_num_steps


def _baseline_state(steps):
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=SimpleNamespace(),
        cuda_graph_runner=None,
        target_attn_backend=SimpleNamespace(),
        target_graph_runner=None,
        draft_extend_attn_backend=None,
        cuda_graph_runner_for_draft_extend=None,
    )


class TestControllerIntegration(unittest.TestCase):
    def _controller(self, mode):
        worker = _BuildRecordingWorker()
        # FROZEN default -> candidate union [1, 2, 3] (no step-0 states).
        # No server_args on the worker -> graph memory resolves to resident.
        controller = AdaptiveController(worker, algorithm="FROZEN_KV_MTP")
        self.assertEqual(controller.graph_memory.mode, "resident")
        if mode == "offload":
            controller.graph_memory = _offload_manager()
        controller.register(_baseline_state(3))
        return worker, controller

    def test_worker_without_server_args_resolves_resident(self):
        worker, controller = self._controller("resident")
        with _mock_cuda():
            controller.init_states(cuda_graph_bs=None)
        # Resident keeps the historical ascending build order.
        self.assertEqual(worker.build_order, [1, 2])

    def test_offload_builds_largest_first_and_finalizes(self):
        worker, controller = self._controller("offload")
        with _mock_cuda():
            controller.init_states(cuda_graph_bs=None)
        self.assertEqual(worker.build_order, [2, 1])
        # Candidate states were paused after build; baseline stays active.
        self.assertEqual(
            {t for t in controller.graph_memory._paused},
            set(),  # no tensors noted by the stub builds -> nothing pauseable
        )
        self.assertEqual(worker.applied, [3])

    def test_activation_goes_through_graph_memory(self):
        worker, controller = self._controller("offload")
        with _mock_cuda():
            controller.init_states(cuda_graph_bs=None)
        seen = []
        controller.graph_memory.ensure_active = lambda steps: seen.append(steps)
        with _mock_cuda():
            controller._activate(2)
        self.assertEqual(seen, [2])
        self.assertEqual(worker.applied, [3, 2])

    def test_forced_swap_knob_cycles_candidates(self):
        worker, controller = self._controller("resident")
        with _mock_cuda():
            controller.init_states(cuda_graph_bs=None)
        with mock.patch.dict(
            os.environ, {"SGLANG_ADAPTIVE_FORCE_SWAP_INTERVAL": "2"}
        ):
            for _ in range(8):
                controller.on_verify_complete([3, 3], batch_size=1)
        # candidates [1,2,3]; start 3 -> 1 -> 2 -> 3 -> 1 (every 2nd call)
        self.assertEqual(worker.applied, [3, 1, 2, 3, 1])

    def test_forced_swap_disabled_by_default(self):
        worker, controller = self._controller("resident")
        with _mock_cuda():
            controller.init_states(cuda_graph_bs=None)
        self.assertFalse(controller._maybe_forced_swap())


class TestHighAcceptProfile(unittest.TestCase):
    def test_profile_resolves_by_name(self):
        self.assertEqual(
            resolve_candidate_steps_from_config("high-accept"), [1, 2, 3, 4, 5]
        )

    def test_default_name_matches_algorithm_default(self):
        self.assertEqual(
            resolve_candidate_steps_from_config("default", algorithm="FROZEN_KV_MTP"),
            resolve_candidate_steps_from_config(None, algorithm="FROZEN_KV_MTP"),
        )

    def test_no_step_zero_anywhere(self):
        # Keeps the profile valid for FROZEN_KV_MTP (hard-rejects step < 1).
        for entry in HIGH_ACCEPT_ADAPTIVE_CONFIG.values():
            self.assertNotIn(0, entry["candidate_steps"])

    def test_ladder_climbs_to_five_on_sustained_high_accept(self):
        slot = AdaptiveStepSlot(
            initial_steps=3, cfg={**HIGH_ACCEPT_ADAPTIVE_CONFIG["1"]}
        )
        reached = set()
        for _ in range(60):
            slot.update([slot.current_steps])  # every draft fully accepted
            reached.add(slot.current_steps)
        self.assertEqual(slot.current_steps, 5)
        self.assertIn(4, reached)  # climbed through the k=4 rung

    def test_up_hysteresis_blocks_borderline_climb(self):
        slot = AdaptiveStepSlot(
            initial_steps=3, cfg={**HIGH_ACCEPT_ADAPTIVE_CONFIG["1"]}
        )
        # Accept length hovering just above the no-hysteresis threshold
        # (2.5 < 2.6 <= 2.75): without up_hysteresis=0.25 this would climb.
        for _ in range(60):
            slot.update([2.6])
        self.assertEqual(slot.current_steps, 3)


if __name__ == "__main__":
    unittest.main()
