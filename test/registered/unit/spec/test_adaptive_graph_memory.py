"""CPU tests for the adaptive graph-memory offload layer (#93).

Covers mode resolution, manager bookkeeping (tagging, pause/resume ordering,
zero-on-resume, no-OOM reserve check, segment-isolation audit), controller
integration (build order, swap path, forced-swap stress knob), and the
high-accept (k=4/5) ladder profile. All CUDA interaction is mocked; the
GPU-side behavior is exercised in the T93 GPU validation phase.
"""

import contextlib
import itertools
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

    @contextlib.contextmanager
    def region_config(self, tag, enable_cpu_backup=False):
        # The manager routes tagged allocations through per-tag MemPools and
        # only uses the config (tag interception) part of the adapter.
        self.region_tags.append(tag)
        self.calls.append(("region_config", tag))
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

    @contextlib.contextmanager
    def _fake_use_mem_pool(pool):
        yield

    with (
        mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None),
        mock.patch.object(torch.cuda, "empty_cache", lambda *a, **k: None),
        mock.patch.object(
            torch.cuda, "mem_get_info", lambda *a, **k: (free_bytes, 1 << 41)
        ),
        mock.patch.object(torch.cuda, "MemPool", lambda *a, **k: object()),
        mock.patch.object(torch.cuda, "use_mem_pool", _fake_use_mem_pool),
        mock.patch.object(agm, "_snapshot_segment_addrs", _snap),
    ):
        yield


def _offload_manager(mode="offload"):
    """Manager in an offload mode with a FakeAdapter installed."""
    with mock.patch.dict(
        os.environ, {"LD_PRELOAD": "/x/torch_memory_saver_hook_mode_preload.so"}
    ):
        with mock.patch.object(
            AdaptiveGraphMemoryManager,
            "__init__",
            lambda self: _offload_init(self, mode=mode),
        ):
            return AdaptiveGraphMemoryManager()


def _offload_init(self, mode="offload"):
    # Mirrors the real __init__ minus the TorchMemorySaverAdapter creation.
    self.mode = mode
    self._states = {}
    self._pools = {}
    self._capture_pools = {}
    self._paused = set()
    self._resumed_tag = None
    self._build_tag = None
    self._in_capture_region = False
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
        n = agm.MIN_TAGGED_BYTES // 4  # exactly at the tagging threshold
        with _mock_cuda():
            for steps in (2, 1):
                with mgr.build_state(steps):
                    with agm.tagged_state_alloc(nbytes=n * 4):
                        t = torch.full((n,), 7, dtype=torch.int32)
                    agm.note_state_tensor(t)
                    tensors[steps] = t
                mgr.pause_after_build(steps)
        return tensors

    def test_size_gate_keeps_small_allocations_resident(self):
        # Regression for the 5-state boot crash: a sub-2MiB allocation
        # tagged into the TMS pool can be served from a paused tag's
        # segment tail -> illegal access. The size gate must keep small
        # allocations out of the tagged region AND out of the zero list.
        mgr = _offload_manager()
        with _mock_cuda():
            with mgr.build_state(3):
                with agm.tagged_state_alloc(nbytes=1024):
                    small = torch.full((256,), 5, dtype=torch.int32)
                agm.note_state_tensor(small)
                big_nbytes = agm.MIN_TAGGED_BYTES
                with agm.tagged_state_alloc(nbytes=big_nbytes):
                    big = torch.full(
                        (big_nbytes // 4,), 5, dtype=torch.int32
                    )
                agm.note_state_tensor(big)
        rec = mgr._states["adaptive_state_k3"]
        self.assertEqual(len(rec.tensors), 1)
        self.assertIs(rec.tensors[0], big)
        # Small tensor was never routed through the adapter region.
        self.assertEqual(mgr._adapter.region_tags, ["adaptive_state_k3"])
        with _mock_cuda():
            mgr.finalize_boot(initial_steps=2)
            mgr.ensure_active(3)
        self.assertTrue(torch.all(big == 0))  # zeroed on resume
        self.assertTrue(torch.all(small == 5))  # untouched, resident

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
            with self.assertRaisesRegex(RuntimeError, "Increase the graph/KV reserve"):
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


class _FakeWrapper:
    """Mimics a flashinfer wrapper's int-workspace surface."""

    def __init__(self, nbytes=8 << 20):
        self._float_workspace_buffer = torch.zeros(16, dtype=torch.uint8)
        self._int_workspace_buffer = torch.full(
            (nbytes,), 7, dtype=torch.uint8
        )
        self.resets = []

    def reset_workspace_buffer(self, float_ws, int_ws):
        self.resets.append((float_ws, int_ws))
        self._float_workspace_buffer = float_ws
        self._int_workspace_buffer = int_ws


@contextlib.contextmanager
def _mock_capture_graph():
    """Mock torch.cuda.graph + graph_pool_handle, recording capture calls."""
    calls = []
    counter = itertools.count()

    @contextlib.contextmanager
    def _fake_graph(cuda_graph, pool=None, stream=None, **kwargs):
        calls.append(("graph", cuda_graph, pool, stream))
        yield

    with (
        mock.patch.object(torch.cuda, "graph", _fake_graph),
        mock.patch.object(
            torch.cuda, "graph_pool_handle", lambda: ("pool", next(counter))
        ),
    ):
        yield calls


class TestStage2ModeResolution(unittest.TestCase):
    def test_explicit_offload_scratch(self):
        with mock.patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": ""}):
            args = _server_args(
                speculative_adaptive_graph_memory="offload-scratch"
            )
            self.assertEqual(
                resolve_adaptive_graph_memory_mode(args), "offload-scratch"
            )

    def test_auto_degrades_to_scratch_on_non_full_decode_backend(self):
        with mock.patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": ""}):
            args = _server_args(cuda_graph_backend_decode="breakable")
            self.assertEqual(
                resolve_adaptive_graph_memory_mode(args), "offload-scratch"
            )

    def test_auto_degrades_to_scratch_on_memory_saver_cuda_graph(self):
        with mock.patch.dict(
            os.environ,
            {
                "PYTORCH_CUDA_ALLOC_CONF": "",
                "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "1",
            },
        ):
            self.assertEqual(
                resolve_adaptive_graph_memory_mode(_server_args()),
                "offload-scratch",
            )

    def test_explicit_offload_with_non_full_backend_raises(self):
        with mock.patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": ""}):
            args = _server_args(
                speculative_adaptive_graph_memory="offload",
                cuda_graph_backend_decode="breakable",
            )
            with self.assertRaisesRegex(ValueError, "offload-scratch"):
                resolve_adaptive_graph_memory_mode(args)

    def test_explicit_scratch_with_missing_base_prereq_raises(self):
        args = _server_args(
            speculative_adaptive_graph_memory="offload-scratch",
            attention_backend="triton",
        )
        with self.assertRaisesRegex(ValueError, "flashinfer required"):
            resolve_adaptive_graph_memory_mode(args)

    def test_decode_backend_read_from_cuda_graph_config(self):
        with mock.patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": ""}):
            cfg = SimpleNamespace(decode=SimpleNamespace(backend="breakable"))
            args = _server_args(cuda_graph_config=cfg)
            self.assertEqual(
                resolve_adaptive_graph_memory_mode(args), "offload-scratch"
            )


class TestStage2CapturePools(unittest.TestCase):
    def test_pool_override_is_per_tag_and_scoped_to_builds(self):
        mgr = _offload_manager()
        with _mock_cuda(), _mock_capture_graph():
            self.assertIsNone(agm.capture_pool_override())
            with mgr.build_state(2):
                p2 = agm.capture_pool_override()
                self.assertIsNotNone(p2)
                # Stable within the build.
                self.assertEqual(agm.capture_pool_override(), p2)
            with mgr.build_state(1):
                p1 = agm.capture_pool_override()
            self.assertNotEqual(p1, p2)
            self.assertIsNone(agm.capture_pool_override())

    def test_scratch_mode_has_no_pool_override(self):
        mgr = _offload_manager(mode="offload-scratch")
        with _mock_cuda(), _mock_capture_graph():
            with mgr.build_state(2):
                self.assertIsNone(agm.capture_pool_override())
                self.assertFalse(agm.in_capture_offload_build())
                self.assertTrue(agm.in_offload_build())

    def test_capture_graph_routes_through_private_pool_and_region(self):
        mgr = _offload_manager()
        with _mock_cuda(), _mock_capture_graph() as calls:
            with mgr.build_state(2):
                pool = agm.capture_pool_override()
                graph = object()
                with agm.capture_graph_ctx(
                    None, cuda_graph=graph, pool=pool, stream="s"
                ):
                    self.assertTrue(mgr._in_capture_region)
                    # Wrap sites firing inside the capture must not nest
                    # region_config (non-reentrant) -- passthrough instead.
                    before = list(mgr._adapter.region_tags)
                    with agm.tagged_state_alloc(nbytes=4 << 20):
                        pass
                    self.assertEqual(mgr._adapter.region_tags, before)
                self.assertFalse(mgr._in_capture_region)
            self.assertEqual(
                [c for c in calls if c[0] == "graph"],
                [("graph", graph, pool, "s")],
            )
            # The capture body ran inside the tag's region_config.
            self.assertIn(
                ("region_config", "adaptive_state_k2"), mgr._adapter.calls
            )

    def test_capture_graph_defers_to_default_ctx_outside_builds(self):
        mgr = _offload_manager()
        seen = []

        @contextlib.contextmanager
        def default_ctx(cuda_graph=None, pool=None, stream=None):
            seen.append((cuda_graph, pool, stream))
            yield

        with _mock_cuda(), _mock_capture_graph() as calls:
            with agm.capture_graph_ctx(
                default_ctx, cuda_graph="g", pool="global", stream="s"
            ):
                pass
        self.assertEqual(seen, [("g", "global", "s")])
        self.assertEqual(calls, [])
        self.assertEqual(mgr._adapter.calls, [])

    def test_capture_graph_asserts_on_foreign_pool(self):
        mgr = _offload_manager()
        with _mock_cuda(), _mock_capture_graph():
            with mgr.build_state(2):
                agm.capture_pool_override()
                with self.assertRaisesRegex(AssertionError, "private pool"):
                    with agm.capture_graph_ctx(
                        None, cuda_graph="g", pool="global", stream=None
                    ):
                        pass

    def test_capture_pool_only_state_swaps_and_reserve_checks(self):
        # A state whose entire footprint is the capture pool (no noted
        # tensors) must still pause after build, count for the reserve
        # check, and pause/resume on swaps.
        mgr = _offload_manager()
        with _mock_cuda(), _mock_capture_graph():
            with mgr.build_state(2):
                pool = agm.capture_pool_override()
                with agm.capture_graph_ctx(
                    None, cuda_graph="g", pool=pool, stream=None
                ):
                    pass
            mgr.pause_after_build(2)
            self.assertIn(
                ("pause", "adaptive_state_k2"), mgr._adapter.calls
            )
            mgr.finalize_boot(initial_steps=3)
            mgr._adapter.calls.clear()
            mgr.ensure_active(2)
            self.assertEqual(
                mgr._adapter.calls, [("resume", "adaptive_state_k2")]
            )
            mgr._adapter.calls.clear()
            mgr.ensure_active(3)
            self.assertEqual(
                mgr._adapter.calls, [("pause", "adaptive_state_k2")]
            )

    def test_paused_bytes_measured_and_drives_reserve_check(self):
        mgr = _offload_manager()
        n = agm.MIN_TAGGED_BYTES // 4
        free_seq = iter(
            [
                (1 << 30, 1 << 41),  # before pause
                ((1 << 30) + (64 << 20), 1 << 41),  # after: 64 MiB released
            ]
        )
        with _mock_cuda(), _mock_capture_graph():
            with mgr.build_state(2):
                with agm.tagged_state_alloc(nbytes=n * 4):
                    t = torch.zeros(n, dtype=torch.int32)
                agm.note_state_tensor(t)
            with mock.patch.object(
                torch.cuda, "mem_get_info", lambda *a, **k: next(free_seq)
            ):
                mgr.pause_after_build(2)
        rec = mgr._states["adaptive_state_k2"]
        self.assertEqual(rec.paused_bytes, 64 << 20)
        self.assertEqual(rec.footprint_bytes, 64 << 20)
        # Reserve check now needs >= the MEASURED footprint, not just the
        # noted tensors.
        with _mock_cuda(free_bytes=32 << 20):
            with self.assertRaisesRegex(RuntimeError, "Increase the graph/KV reserve"):
                mgr.finalize_boot(initial_steps=3)

    def test_note_kinds_are_itemized(self):
        mgr = _offload_manager()
        n = agm.MIN_TAGGED_BYTES
        with _mock_cuda():
            with mgr.build_state(2):
                with agm.tagged_state_alloc(nbytes=n):
                    a = torch.zeros(n, dtype=torch.uint8)
                agm.note_state_tensor(a)
                with agm.tagged_state_alloc(nbytes=2 * n):
                    b = torch.zeros(2 * n, dtype=torch.uint8)
                agm.note_state_tensor(b, kind="int_ws")
        rec = mgr._states["adaptive_state_k2"]
        self.assertEqual(rec.kind_nbytes("scratch"), n)
        self.assertEqual(rec.kind_nbytes("int_ws"), 2 * n)


class TestStage2IntWorkspaceTagging(unittest.TestCase):
    def _helper(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            _tag_adaptive_int_workspace,
        )

        return _tag_adaptive_int_workspace

    def test_retag_inside_stage2_build(self):
        tag_fn = self._helper()
        mgr = _offload_manager()
        with _mock_cuda(), _mock_capture_graph():
            with mgr.build_state(2):
                w = _FakeWrapper()
                old = w._int_workspace_buffer
                self.assertIs(tag_fn(w), w)
        self.assertEqual(len(w.resets), 1)
        self.assertIsNot(w._int_workspace_buffer, old)
        # Fresh int workspace starts zeroed (boot contract) and is noted
        # for zero-on-resume.
        self.assertTrue(torch.all(w._int_workspace_buffer == 0))
        rec = mgr._states["adaptive_state_k2"]
        self.assertEqual(rec.kind_nbytes("int_ws"), 8 << 20)

    def test_noop_outside_build_and_in_scratch_mode(self):
        tag_fn = self._helper()
        # Outside any build scope.
        mgr = _offload_manager()
        w = _FakeWrapper()
        tag_fn(w)
        self.assertEqual(w.resets, [])
        # Inside a Stage-1 (offload-scratch) build scope.
        mgr = _offload_manager(mode="offload-scratch")
        with _mock_cuda():
            with mgr.build_state(2):
                tag_fn(w)
        self.assertEqual(w.resets, [])
        self.assertEqual(mgr._states["adaptive_state_k2"].tensors, [])

    def test_share_key_reuses_one_buffer_across_buckets(self):
        tag_fn = self._helper()
        mgr = _offload_manager()
        with _mock_cuda(), _mock_capture_graph():
            with mgr.build_state(2):
                w1 = tag_fn(_FakeWrapper(), share_key=("decode_cg", 1, 0))
                w2 = tag_fn(_FakeWrapper(), share_key=("decode_cg", 1, 0))
                w3 = tag_fn(_FakeWrapper(), share_key=("decode_cg", 2, 0))
        # Same slot -> same physical workspace; different slot -> private.
        self.assertEqual(
            w1._int_workspace_buffer.data_ptr(),
            w2._int_workspace_buffer.data_ptr(),
        )
        self.assertNotEqual(
            w1._int_workspace_buffer.data_ptr(),
            w3._int_workspace_buffer.data_ptr(),
        )
        rec = mgr._states["adaptive_state_k2"]
        # Noted once per distinct buffer, not per wrapper.
        self.assertEqual(rec.kind_nbytes("int_ws"), 2 * (8 << 20))
        self.assertEqual(len(rec.tensors), 2)

    def test_serving_margin_blocks_thin_configs(self):
        mgr = _offload_manager()
        n = agm.MIN_TAGGED_BYTES // 4
        with _mock_cuda():
            with mgr.build_state(2):
                with agm.tagged_state_alloc(nbytes=n * 4):
                    t = torch.zeros(n, dtype=torch.int32)
                agm.note_state_tensor(t)
            mgr.pause_after_build(2)
        rec = mgr._states["adaptive_state_k2"]
        rec.paused_bytes = 100 << 20
        # Free covers the mapped state but NOT state + serving margin
        # (default 512 MiB): must fail fast at boot, not OOM at runtime.
        with _mock_cuda(free_bytes=200 << 20):
            with self.assertRaisesRegex(
                RuntimeError, "serving transient margin"
            ):
                mgr.finalize_boot(initial_steps=3)
        # With margin honored it finalizes.
        with _mock_cuda(free_bytes=(100 + 512 + 1) << 20):
            mgr.finalize_boot(initial_steps=3)

    def test_small_int_workspace_stays_resident(self):
        tag_fn = self._helper()
        mgr = _offload_manager()
        with _mock_cuda(), _mock_capture_graph():
            with mgr.build_state(2):
                w = _FakeWrapper(nbytes=1 << 20)  # below MIN_TAGGED_BYTES
                tag_fn(w)
        self.assertEqual(w.resets, [])
        self.assertEqual(mgr._states["adaptive_state_k2"].tensors, [])


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
        # Stage-1 anti-flap: each rung is held >= min_dwell_rounds (default 64)
        # before the next climb, so reaching 5 takes warmup + two dwells.
        for _ in range(2 * slot.min_dwell_rounds + slot.warmup_batches + 20):
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
