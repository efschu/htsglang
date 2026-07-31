# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#274 slice C1: the de-globalization primitive, and the falsifiers for the
process globals that made concurrency unsafe.

Every test here follows the same shape, because that is the shape of the bug
class: build the situation in which TWO lanes read or write one process-wide
thing AT THE SAME TIME, show what the old (shared) form would have produced,
and show that the lane-scoped form does not produce it.

CPU only -- these are properties of the scoping mechanism, not of any kernel.
"""

from __future__ import annotations

import os
import threading
import unittest

from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    scoped_tp_partition_ratios,
    set_tp_partition_ratios,
)
from sglang.srt.runtime_context import (
    current_lane_id,
    get_context,
    get_parallel,
    lane_scope,
    reset_context,
)


class _Rendezvous:
    """Two threads that are provably INSIDE their scopes at the same time.

    A sequential test cannot falsify a concurrency bug: it would pass just as
    happily against the broken shared form. The barrier makes the overlap a
    property of the test rather than of the scheduler's whims.
    """

    def __init__(self, n=2):
        self.barrier = threading.Barrier(n, timeout=10)
        self.results = {}
        self.errors = []

    def run(self, targets):
        threads = [
            threading.Thread(target=self._wrap, args=(name, fn))
            for name, fn in targets.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
        if self.errors:
            raise self.errors[0]
        return self.results

    def _wrap(self, name, fn):
        try:
            self.results[name] = fn(self.barrier)
        except BaseException as exc:  # noqa: BLE001 - surfaced by run()
            self.errors.append(exc)
            try:
                self.barrier.abort()
            except Exception:
                pass


class TestServerArgsOverlay(unittest.TestCase):
    """The named slice-C blocker: slice B SWAPPED the process config around
    each lane tick, so a concurrent serving forward would read the lane's."""

    def setUp(self):
        reset_context()
        self.serving = object()
        self.lane = object()
        get_context().set_server_args(self.serving)

    def tearDown(self):
        reset_context()

    def test_lane_scope_does_not_touch_the_process_slot(self):
        ctx = get_context()
        with lane_scope(0, self.lane):
            self.assertIs(ctx.server_args, self.lane)
            # THE point of the overlay: the slot itself is untouched, so a
            # thread without the scope keeps reading the serving config.
            self.assertIs(ctx._server_args, self.serving)
        self.assertIs(ctx.server_args, self.serving)

    def test_two_threads_read_different_configs_at_the_same_time(self):
        ctx = get_context()
        rv = _Rendezvous()

        def lane_thread(barrier):
            with lane_scope(0, self.lane):
                barrier.wait()  # serving thread is reading right now
                barrier.wait()
                return ctx.server_args

        def serving_thread(barrier):
            barrier.wait()
            seen = ctx.server_args  # lane scope is active on the other thread
            barrier.wait()
            return seen

        out = rv.run({"lane": lane_thread, "serving": serving_thread})
        self.assertIs(out["lane"], self.lane)
        # Under slice B's set_server_args swap this assertion FAILS: the
        # serving thread would read the lane's config.
        self.assertIs(out["serving"], self.serving)

    def test_lane_id_is_the_key_for_per_lane_resources(self):
        self.assertIsNone(current_lane_id())
        with lane_scope(1, self.lane):
            self.assertEqual(current_lane_id(), 1)
        self.assertIsNone(current_lane_id())

    def test_a_fresh_thread_inherits_the_serving_config(self):
        ctx = get_context()
        seen = {}

        def body():
            seen["args"] = ctx.server_args
            seen["lane"] = current_lane_id()

        t = threading.Thread(target=body)
        t.start()
        t.join()
        self.assertIs(seen["args"], self.serving)
        self.assertIsNone(seen["lane"])


class TestParallelOverrideIsolation(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_two_threads_hold_different_geometries(self):
        parallel = get_parallel()
        rv = _Rendezvous()

        def lane_thread(barrier):
            with parallel.override(tp_size=1, tp_rank=0):
                barrier.wait()
                barrier.wait()
                return parallel._overrides.get("tp_size")

        def serving_thread(barrier):
            with parallel.override(tp_size=3, tp_rank=0):
                barrier.wait()
                seen = parallel._overrides.get("tp_size")
                barrier.wait()
                return seen

        out = rv.run({"lane": lane_thread, "serving": serving_thread})
        self.assertEqual(out["lane"], 1)
        # The lane's tp_size=1 must not be what the serving group's
        # collectives size themselves against.
        self.assertEqual(out["serving"], 3)

    def test_nesting_still_restores_within_one_thread(self):
        parallel = get_parallel()
        with parallel.override(tp_size=3):
            with parallel.override(tp_size=1):
                self.assertEqual(parallel._overrides["tp_size"], 1)
            self.assertEqual(parallel._overrides["tp_size"], 3)
        self.assertNotIn("tp_size", parallel._overrides)


class TestPartitionRatioOverlay(unittest.TestCase):
    def setUp(self):
        set_tp_partition_ratios([2, 1, 1], {"mlp": [6, 1, 1]})

    def tearDown(self):
        set_tp_partition_ratios(None)
        reset_context()

    def test_process_plan_is_visible_and_unchanged_by_a_scope(self):
        self.assertEqual(get_tp_partition_ratios(), [2, 1, 1])
        self.assertEqual(get_tp_partition_ratios("mlp"), [6, 1, 1])
        with scoped_tp_partition_ratios([2, 2], {"mlp": [6, 2]}):
            self.assertEqual(get_tp_partition_ratios(), [2, 2])
            self.assertEqual(get_tp_partition_ratios("mlp"), [6, 2])
        self.assertEqual(get_tp_partition_ratios(), [2, 1, 1])
        self.assertEqual(get_tp_partition_ratios("mlp"), [6, 1, 1])

    def test_lane_vector_is_invisible_to_a_concurrent_thread(self):
        rv = _Rendezvous()

        def lane_thread(barrier):
            with scoped_tp_partition_ratios([2, 2], {"mlp": [6, 2]}):
                barrier.wait()
                barrier.wait()
                return get_tp_partition_ratios(), get_tp_partition_ratios("mlp")

        def serving_thread(barrier):
            barrier.wait()
            seen = get_tp_partition_ratios(), get_tp_partition_ratios("mlp")
            barrier.wait()
            return seen

        out = rv.run({"lane": lane_thread, "serving": serving_thread})
        self.assertEqual(out["lane"], ([2, 2], [6, 2]))
        # The falsifier that matters: a 2-entry vector installed globally
        # while the serving group has tp_size 3 does not raise -- the
        # len(ratios) == tp_size discriminator silently falls back to the
        # EVEN split, i.e. wrong units, no error.
        self.assertEqual(out["serving"], ([2, 1, 1], [6, 1, 1]))

    def test_none_is_a_meaningful_overlay_value(self):
        # "no plan" (even split) must be distinguishable from "no overlay".
        with scoped_tp_partition_ratios(None):
            self.assertIsNone(get_tp_partition_ratios())
        self.assertEqual(get_tp_partition_ratios(), [2, 1, 1])


class TestPerLaneGraphPool(unittest.TestCase):
    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_each_lane_gets_its_own_pool_handle(self):
        from sglang.srt.model_executor.runner_utils.pool import (
            get_global_graph_memory_pool,
            get_or_create_global_graph_memory_pool,
        )

        class _FakeDeviceModule:
            def __init__(self):
                self.n = 0

            def graph_pool_handle(self):
                self.n += 1
                return f"pool-{self.n}"

        dm = _FakeDeviceModule()
        serving = get_or_create_global_graph_memory_pool(dm)
        self.assertEqual(get_or_create_global_graph_memory_pool(dm), serving)
        with lane_scope(0, None):
            lane = get_or_create_global_graph_memory_pool(dm)
            self.assertEqual(get_global_graph_memory_pool(), lane)
        # Two graphs sharing a pool share the buffers their intermediates
        # live in; replaying them at the same time is the corruption.
        self.assertNotEqual(lane, serving)
        self.assertEqual(get_global_graph_memory_pool(), serving)


class TestPerLaneDequantWorkspace(unittest.TestCase):
    """Geteilte-Puffer-Familie, falsifier first.

    The GGUF dequant workspace is ONE reused buffer per (device, dtype), and
    its safety argument is explicitly sequential: dequant -> GEMM pairs run
    back-to-back on one stream. Two lanes forwarding at once are two streams
    with interleaved pairs over one buffer -- lane B's dequant overwrites the
    weight lane A's GEMM is still reading. Silent wrong numbers, no crash.
    """

    def tearDown(self):
        reset_context()

    def test_the_key_separates_lanes(self):
        import torch

        from sglang.srt.layers.quantization.gguf import _dequant_ws_key

        device = torch.device("cpu")
        serving = _dequant_ws_key(torch.float16, device)
        with lane_scope(0, None):
            lane0 = _dequant_ws_key(torch.float16, device)
        with lane_scope(1, None):
            lane1 = _dequant_ws_key(torch.float16, device)
        self.assertNotEqual(serving, lane0)
        self.assertNotEqual(lane0, lane1)
        # Same lane, same key: within a lane the sequential premise holds
        # again, which is what makes one buffer per lane correct.
        with lane_scope(0, None):
            self.assertEqual(_dequant_ws_key(torch.float16, device), lane0)

    def test_shared_form_would_alias(self):
        import torch

        from sglang.srt.layers.quantization.gguf import _dequant_ws_key

        # The pre-slice-C key, reconstructed: device + dtype only.
        device = torch.device("cpu")
        old_key = (device, torch.float16)
        with lane_scope(0, None):
            self.assertNotEqual(_dequant_ws_key(torch.float16, device), old_key)
        with lane_scope(1, None):
            self.assertNotEqual(_dequant_ws_key(torch.float16, device), old_key)


class TestPerLaneInputBufferPool(unittest.TestCase):
    """Geteilte-Puffer-Familie, the CUDA-graph INPUT buffers (#274 slice D2).

    ``share_input_buffer`` coalesces every graph runner's ``input_ids`` /
    ``positions`` / ``out_cache_loc`` by (name, numel, dtype, device), and its
    safety argument is the same sequential one: the buffers are filled
    immediately before each replay and the forwards are mutually exclusive.

    A concurrent lane is the case that breaks it, and it is not a hypothetical
    collision: the lane's breakable-prefill tier ladder tops out at the serving
    group's ``chunked_prefill_size``, so the two runners request an IDENTICAL
    key and both graphs are captured against one address. The loser's
    ``store_kvcache`` then reads slot ids belonging to the other group's pool
    -- the ``index >= 0 && index < size_limit`` device assert of DESIGN_121
    §12.7.
    """

    def setUp(self):
        from sglang.srt.model_executor.input_buffers import (
            _forward_input_buffer_pool,
        )

        _forward_input_buffer_pool.clear()

    def tearDown(self):
        from sglang.srt.model_executor.input_buffers import (
            _forward_input_buffer_pool,
        )

        _forward_input_buffer_pool.clear()
        os.environ.pop("SGLANG_LANE_SHARED_INPUT_BUFFERS", None)
        reset_context()

    @staticmethod
    def _out_cache_loc():
        import torch

        # The measured vehicle's shape: 2048 x int64, the serving group's
        # chunked_prefill_size and the top of the lane's thinned tier ladder.
        return torch.zeros(2048, dtype=torch.int64)

    def test_a_concurrent_lane_does_not_land_on_the_serving_buffer(self):
        from sglang.srt.model_executor.input_buffers import share_input_buffer

        serving = share_input_buffer("out_cache_loc", self._out_cache_loc())
        with lane_scope(0, None):
            lane0 = share_input_buffer("out_cache_loc", self._out_cache_loc())
        with lane_scope(1, None):
            lane1 = share_input_buffer("out_cache_loc", self._out_cache_loc())

        self.assertNotEqual(serving.data_ptr(), lane0.data_ptr())
        self.assertNotEqual(lane0.data_ptr(), lane1.data_ptr())

    def test_within_one_scope_the_sharing_is_unchanged(self):
        from sglang.srt.model_executor.input_buffers import share_input_buffer

        # The default path: the serving group's own runners still coalesce,
        # which is the whole point of the pool.
        first = share_input_buffer("out_cache_loc", self._out_cache_loc())
        second = share_input_buffer("out_cache_loc", self._out_cache_loc())
        self.assertEqual(first.data_ptr(), second.data_ptr())

        with lane_scope(0, None):
            a = share_input_buffer("out_cache_loc", self._out_cache_loc())
            b = share_input_buffer("out_cache_loc", self._out_cache_loc())
        self.assertEqual(a.data_ptr(), b.data_ptr())

    def test_the_escape_hatch_reproduces_the_defect(self):
        from sglang.srt.model_executor.input_buffers import share_input_buffer

        os.environ["SGLANG_LANE_SHARED_INPUT_BUFFERS"] = "1"
        serving = share_input_buffer("out_cache_loc", self._out_cache_loc())
        with lane_scope(0, None):
            lane0 = share_input_buffer("out_cache_loc", self._out_cache_loc())
        # This IS the bug: one address, two concurrent writers.
        self.assertEqual(serving.data_ptr(), lane0.data_ptr())

    def test_a_serial_lane_keeps_sharing(self):
        from sglang.srt.model_executor.input_buffers import share_input_buffer

        # ``DualGroupLane.scope_lane_id`` is None in serial mode, so the serial
        # path stays byte-for-byte the slice-B path -- the gate slice C was not
        # allowed to move.
        serving = share_input_buffer("out_cache_loc", self._out_cache_loc())
        with lane_scope(None, None):
            serial_lane = share_input_buffer("out_cache_loc", self._out_cache_loc())
        self.assertEqual(serving.data_ptr(), serial_lane.data_ptr())


class TestPerLaneAttentionWorkspace(unittest.TestCase):
    """Geteilte-Puffer-Familie, the flashinfer FLOAT WORKSPACE (#274 slice D3).

    Two defects, one family, found together because the second is only
    reachable once the first is fixed.

    C1 -- ``RuntimeContext.get_buffer`` was keyed by NAME alone, and every
    production caller of it is an attention workspace (flashinfer,
    flashinfer-MLA, trtllm-MLA, trtllm-MHA, DSA, musa-flashattention). A
    concurrent lane builds a second set of backends in the same process, under
    ``lane_scope(lane_id)``, and was handed the serving group's 384 MiB
    scratch: one buffer holding live split-KV partials, two threads, two
    streams. Nothing asserts -- the loser reads the winner's partials and the
    attention output is silently wrong.

    C2 -- ``zero_flashinfer_workspaces()`` runs at the SERVING group's request
    finish and zeroed every registered workspace, because ``_WORKSPACE_BUFFERS``
    was keyed by ``id()`` and knew no lane. Even with C1 fixed, a lane's own
    workspace was wiped mid-forward by a foreign thread.
    """

    def setUp(self):
        from sglang.srt.layers.attention.flashinfer_backend import _WORKSPACE_BUFFERS

        _WORKSPACE_BUFFERS.clear()
        reset_context()

    def tearDown(self):
        from sglang.srt.layers.attention.flashinfer_backend import _WORKSPACE_BUFFERS

        _WORKSPACE_BUFFERS.clear()
        os.environ.pop("SGLANG_LANE_SHARED_ATTN_WORKSPACE", None)
        reset_context()

    @staticmethod
    def _workspace():
        import torch

        # Shape is irrelevant to the keying; the production one is
        # SGLANG_FLASHINFER_WORKSPACE_SIZE bytes of uint8.
        return torch.empty(1024, dtype=torch.uint8)

    def _alloc(self):
        from sglang.srt.runtime_context import get_buffer

        return get_buffer("flashinfer_workspace", self._workspace)

    # -- C1: the allocation ------------------------------------------------

    def test_a_concurrent_lane_gets_its_own_attention_workspace(self):
        serving = self._alloc()
        with lane_scope(0, None):
            lane0 = self._alloc()
        with lane_scope(1, None):
            lane1 = self._alloc()

        self.assertNotEqual(serving.data_ptr(), lane0.data_ptr())
        self.assertNotEqual(lane0.data_ptr(), lane1.data_ptr())

    def test_within_one_scope_the_sharing_is_unchanged(self):
        # The default path: one named buffer per name, which is what every
        # backend in the serving group relies on.
        first = self._alloc()
        second = self._alloc()
        self.assertEqual(first.data_ptr(), second.data_ptr())

        with lane_scope(0, None):
            a = self._alloc()
            b = self._alloc()
        self.assertEqual(a.data_ptr(), b.data_ptr())

    def test_a_serial_lane_keeps_sharing(self):
        # ``DualGroupLane.scope_lane_id`` is None in serial mode: the lane and
        # the serving group never run at the same time, so sharing is sound and
        # keeping it is what makes serial byte-for-byte the slice-B mode.
        serving = self._alloc()
        with lane_scope(None, None):
            serial_lane = self._alloc()
        self.assertEqual(serving.data_ptr(), serial_lane.data_ptr())

    def test_the_escape_hatch_reproduces_the_defect(self):
        os.environ["SGLANG_LANE_SHARED_ATTN_WORKSPACE"] = "1"
        serving = self._alloc()
        with lane_scope(0, None):
            lane0 = self._alloc()
        # This IS the bug: one scratch, two concurrent forwards.
        self.assertEqual(serving.data_ptr(), lane0.data_ptr())

    def test_distinct_names_stay_distinct_within_a_lane(self):
        from sglang.srt.runtime_context import get_buffer

        with lane_scope(0, None):
            ws = get_buffer("flashinfer_workspace", self._workspace)
            other = get_buffer("some_other_workspace", self._workspace)
        self.assertNotEqual(ws.data_ptr(), other.data_ptr())

    # -- C2: the zeroing ---------------------------------------------------

    def test_serving_zeroing_leaves_a_concurrent_lanes_workspace_alone(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            register_flashinfer_workspace_buffer,
            zero_flashinfer_workspaces,
        )

        serving_ws = self._workspace()
        register_flashinfer_workspace_buffer(serving_ws)
        serving_ws.fill_(0xCD)
        lane_ws = self._workspace()
        with lane_scope(0, None):
            register_flashinfer_workspace_buffer(lane_ws)
        lane_ws.fill_(0xAB)  # the lane's in-flight split-KV partials

        # The scheduler thread, a serving request just finished.
        self.assertEqual(zero_flashinfer_workspaces(), 1)
        self.assertTrue(bool((serving_ws == 0).all()))
        self.assertTrue(bool((lane_ws == 0xAB).all()))

    def test_a_lane_zeroes_its_own_and_only_its_own(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            register_flashinfer_workspace_buffer,
            zero_flashinfer_workspaces,
        )

        serving_ws = self._workspace()
        register_flashinfer_workspace_buffer(serving_ws)
        serving_ws.fill_(0xCD)
        lane_ws = self._workspace()
        with lane_scope(0, None):
            register_flashinfer_workspace_buffer(lane_ws)
            lane_ws.fill_(0xAB)
            # DualGroupLane._finish: the lane's own job boundary, which is
            # where the #50 boot contract is restored for the lane.
            self.assertEqual(zero_flashinfer_workspaces(), 1)

        self.assertTrue(bool((lane_ws == 0).all()))
        self.assertTrue(bool((serving_ws == 0xCD).all()))

    def test_the_escape_hatch_reproduces_the_zeroing_defect(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            register_flashinfer_workspace_buffer,
            zero_flashinfer_workspaces,
        )

        os.environ["SGLANG_LANE_SHARED_ATTN_WORKSPACE"] = "1"
        lane_ws = self._workspace()
        with lane_scope(0, None):
            register_flashinfer_workspace_buffer(lane_ws)
        lane_ws.fill_(0xAB)

        # This IS the bug: the serving group's request finish wipes the lane's
        # live scratch, with no assert anywhere.
        self.assertEqual(zero_flashinfer_workspaces(), 1)
        self.assertTrue(bool((lane_ws == 0).all()))

    def test_a_serial_lane_is_zeroed_by_the_serving_group(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            register_flashinfer_workspace_buffer,
            zero_flashinfer_workspaces,
        )

        # Serial mode shares the workspace on purpose, so the serving group's
        # request finish must keep reaching it -- the #50 contract is unchanged
        # in the mode that must not move.
        ws = self._workspace()
        with lane_scope(None, None):
            register_flashinfer_workspace_buffer(ws)
        ws.fill_(0xAB)
        self.assertEqual(zero_flashinfer_workspaces(), 1)
        self.assertTrue(bool((ws == 0).all()))

    def test_zeroing_an_empty_bucket_is_zero_not_a_key_error(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            zero_flashinfer_workspaces,
        )

        with lane_scope(3, None):
            self.assertEqual(zero_flashinfer_workspaces(), 0)


class TestCaptureModeIsolation(unittest.TestCase):
    def test_capture_on_one_thread_is_not_capture_on_another(self):
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
            model_capture_mode,
        )

        rv = _Rendezvous()

        def capturing(barrier):
            with model_capture_mode():
                barrier.wait()
                barrier.wait()
                return get_is_capture_mode()

        def forwarding(barrier):
            barrier.wait()
            seen = get_is_capture_mode()
            barrier.wait()
            return seen

        out = rv.run({"capture": capturing, "forward": forwarding})
        self.assertTrue(out["capture"])
        # A lane capturing its graphs must not make the serving group's
        # concurrent forward take capture-time branches.
        self.assertFalse(out["forward"])


class TestForwardContextIsolation(unittest.TestCase):
    """The bug the first concurrent boot found, as a falsifier.

    ``get_attn_backend()`` resolves through a per-forward context that was a
    plain module global, with a docstring saying exactly what would happen
    if worker threads ever shared a process. They now do. The measured
    failure: the lane published its GDN backend, and the serving group's
    draft extend -- forwarding on the scheduler thread at that same moment --
    read it, so a full-attention call landed in ``GDNAttnBackend.forward_
    extend`` and asserted on a positional mismatch (``mixed_qkv``). The whole
    server died on rank 0 and took the group with it via gloo.
    """

    def tearDown(self):
        from sglang.srt.model_executor.forward_context import set_forward_context

        set_forward_context(None)

    def test_two_threads_see_their_own_attention_backend(self):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
            get_attn_backend,
        )

        serving_backend = object()
        lane_backend = object()
        rv = _Rendezvous()

        def lane_thread(barrier):
            with forward_context(ForwardContext(attn_backend=lane_backend)):
                barrier.wait()
                barrier.wait()
                return get_attn_backend()

        def serving_thread(barrier):
            with forward_context(ForwardContext(attn_backend=serving_backend)):
                barrier.wait()
                seen = get_attn_backend()
                barrier.wait()
                return seen

        out = rv.run({"lane": lane_thread, "serving": serving_thread})
        self.assertIs(out["lane"], lane_backend)
        self.assertIs(out["serving"], serving_backend)

    def test_a_fresh_thread_has_no_forward_context(self):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
            has_forward_context,
        )

        seen = {}

        def body():
            seen["has"] = has_forward_context()

        with forward_context(ForwardContext(attn_backend=object())):
            t = threading.Thread(target=body)
            t.start()
            t.join()
        # Not inherited: a lane thread must publish its own, and reading an
        # unpublished context asserts instead of silently using the other
        # lane's backend.
        self.assertFalse(seen["has"])

    def test_graph_window_flags_do_not_leak_across_threads(self):
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (  # noqa: E501
            enable_breakable_cuda_graph,
            is_in_breakable_cuda_graph,
        )

        rv = _Rendezvous()

        def in_graph(barrier):
            with enable_breakable_cuda_graph():
                barrier.wait()
                barrier.wait()
                return is_in_breakable_cuda_graph()

        def outside(barrier):
            barrier.wait()
            seen = is_in_breakable_cuda_graph()
            barrier.wait()
            return seen

        out = rv.run({"in": in_graph, "out": outside})
        self.assertTrue(out["in"])
        # get_is_capture_mode() reads this flag, so a leak makes the serving
        # group take capture-time branches during a lane graph window.
        self.assertFalse(out["out"])

    def test_dcp_guard_toggle_does_not_leak_across_threads(self):
        from sglang.srt.layers.dcp.collective_guard import (
            guard_enabled,
            set_guard_enabled,
        )

        rv = _Rendezvous()

        def lane_replay(barrier):
            set_guard_enabled(False)  # what a decode-graph replay does
            barrier.wait()
            barrier.wait()
            return guard_enabled()

        def serving_forward(barrier):
            barrier.wait()
            seen = guard_enabled()
            barrier.wait()
            return seen

        out = rv.run({"lane": lane_replay, "serving": serving_forward})
        self.assertFalse(out["lane"])
        # If this leaked, rank 0 would skip a handshake the other ranks still
        # perform -- the rank-divergence hang the guard exists to catch,
        # caused by the guard.
        self.assertTrue(out["serving"])


class TestSpeedDial(unittest.TestCase):
    """Speed through sacrifice as one explicit regulator."""

    def _args(self, dial, budget=1600, requests=4):
        class _A:
            dual_group_lane_speed_dial = dial
            dual_group_lane_budget_mib = budget
            dual_group_lane_max_requests = requests

        return _A()

    def test_unset_is_exactly_the_configured_capacity(self):
        from sglang.srt.model_executor.dual_group_lane import resolve_speed_dial

        self.assertEqual(resolve_speed_dial(self._args(None)), (1600, 4))

    def test_dial_only_reduces_and_is_monotone(self):
        from sglang.srt.model_executor.dual_group_lane import resolve_speed_dial

        seen = [resolve_speed_dial(self._args(d)) for d in (0.0, 0.25, 0.5, 1.0)]
        self.assertEqual(seen[0], (1600, 4))
        self.assertEqual(seen[-1], (200, 1))
        budgets = [b for b, _ in seen]
        self.assertEqual(budgets, sorted(budgets, reverse=True))

    def test_out_of_range_is_refused(self):
        from sglang.srt.model_executor.dual_group_lane import resolve_speed_dial

        with self.assertRaises(ValueError):
            resolve_speed_dial(self._args(1.5))


class TestLaneSpecBudget(unittest.TestCase):
    """Speculation on the lane: ONE budget, split -- not two that share a name.

    Both lane runners read --dual-group-lane-budget-mib, so without an
    explicit split the NEXTN head claims a second full budget and the lane's
    capacity post silently doubles. That is the exact failure resource
    principle 2 forbids ("no VRAM duplicated that does not have to be, and
    never as a side effect").
    """

    def _args(self, budget=1600):
        class _A:
            dual_group_lane_budget_mib = budget

        return _A()

    def _cfg(self, layers, full_ids=None, nextn=None):
        """A model config shaped like the real ones.

        ``full_ids`` is what a hybrid text config exposes and what actually
        pays per token; ``nextn`` is the NEXTN declaration. A REAL draft
        config carries the target's ``num_hidden_layers`` AND the target's
        ``full_attention_layer_ids`` -- only ``num_nextn_predict_layers``
        distinguishes it -- so the head fixtures below reproduce that rather
        than the friendlier "draft reports 1 layer" that the pre-r8 tests
        assumed.
        """

        class _Text:
            full_attention_layer_ids = list(full_ids or [])

        class _Hf:
            num_nextn_predict_layers = nextn

            @staticmethod
            def get_text_config():
                return _Text()

        class _C:
            num_hidden_layers = layers
            num_nextn_predict_layers = nextn
            hf_config = _Hf()

        return _C()

    def _qwen35(self):
        """The rig vehicle: 64 layers, every fourth one full attention."""
        return self._cfg(64, full_ids=list(range(3, 64, 4)))

    def _qwen35_head(self):
        """Its NEXTN head: same checkpoint-derived config, one predict layer."""
        return self._cfg(64, full_ids=list(range(3, 64, 4)), nextn=1)

    def test_the_split_conserves_the_operators_budget(self):
        from sglang.srt.model_executor.dual_group_lane import split_lane_budget

        target, draft = split_lane_budget(
            self._args(1600), self._qwen35(), self._qwen35_head()
        )
        self.assertEqual(target + draft, 1600)

    def test_the_head_gets_its_KV_LAYER_share_not_a_quarter(self):
        from sglang.srt.model_executor.dual_group_lane import split_lane_budget

        _, draft = split_lane_budget(
            self._args(1600), self._qwen35(), self._qwen35_head()
        )
        # 16 full-attention layers against the head's one: 1/17 of 1600,
        # rounded up. The pre-r8 rule returned 400 here -- the ratio of
        # num_hidden_layers is 64/64 on a real draft config, so the clamp
        # decided, and ~325 of those MiB were then thrown away by the token
        # cap instead of going back to the lane target.
        self.assertEqual(draft, 95)

    def test_a_head_that_declares_no_nextn_layer_is_not_guessed_to_be_one(self):
        from sglang.srt.model_executor.dual_group_lane import split_lane_budget

        # Same config WITHOUT the NEXTN declaration: it is then indistinguishable
        # from the target and must be charged like it, not silently shrunk.
        _, draft = split_lane_budget(
            self._args(1600),
            self._qwen35(),
            self._cfg(64, full_ids=list(range(3, 64, 4))),
        )
        self.assertEqual(draft, 800)

    def test_a_dense_target_is_counted_by_its_plain_layers(self):
        from sglang.srt.model_executor.dual_group_lane import split_lane_budget

        # No hybrid split: every layer bears KV, so 1 of 64 + 1.
        _, draft = split_lane_budget(
            self._args(1600), self._cfg(64), self._cfg(64, nextn=1)
        )
        self.assertEqual(draft, 25)

    def test_the_head_pool_stays_allocatable_on_a_tiny_budget(self):
        from sglang.srt.model_executor.dual_group_lane import split_lane_budget

        target, draft = split_lane_budget(
            self._args(8), self._cfg(4096), self._cfg(4096, nextn=1)
        )
        self.assertGreaterEqual(draft, 1)
        self.assertEqual(target + draft, 8)

    def test_the_head_can_never_crowd_out_the_target(self):
        from sglang.srt.model_executor.dual_group_lane import split_lane_budget

        # A pathological ratio (the head deeper than the target) must not
        # starve the lane target the head exists to serve.
        target, draft = split_lane_budget(
            self._args(1600), self._cfg(1), self._cfg(4, nextn=4)
        )
        self.assertLessEqual(draft, 800)
        self.assertGreaterEqual(target, draft)

    def test_the_ledger_names_both_posts_and_their_layer_counts(self):
        from sglang.srt.model_executor.dual_group_lane import (
            LaneBudgetSplit,
            split_lane_budget,
        )

        target, draft = split_lane_budget(
            self._args(1600), self._qwen35(), self._qwen35_head()
        )
        line = LaneBudgetSplit(1600, target, draft, 16, 1).ledger()
        for token in ("1600", str(target), str(draft), "16", "1/17"):
            self.assertIn(token, line)


class TestLaneKvBearingLayerCount(unittest.TestCase):
    """The layer count that decides the split is the KV-BEARING one.

    Three ways to get it wrong, one test each, because each of them was a
    real reading of a real config: the plain layer count (counts GDN layers
    that hold no KV), the draft config's layer count (it is the target's),
    and the draft config's full-attention ids (also the target's).
    """

    def _cfg(self, layers, full_ids=None, nextn=None):
        return TestLaneSpecBudget._cfg(TestLaneSpecBudget(), layers, full_ids, nextn)

    def test_a_hybrid_target_counts_only_its_full_attention_layers(self):
        from sglang.srt.model_executor.dual_group_lane import kv_bearing_layer_count

        cfg = self._cfg(64, full_ids=list(range(3, 64, 4)))
        self.assertEqual(kv_bearing_layer_count(cfg), 16)

    def test_a_head_is_counted_by_its_nextn_declaration(self):
        from sglang.srt.model_executor.dual_group_lane import kv_bearing_layer_count

        # The inherited 64 layers and the inherited 16 full-attention ids are
        # both wrong for the head, and both are present on the object.
        cfg = self._cfg(64, full_ids=list(range(3, 64, 4)), nextn=1)
        self.assertEqual(kv_bearing_layer_count(cfg, is_head=True), 1)
        self.assertEqual(kv_bearing_layer_count(cfg), 16)

    def test_a_dense_config_falls_through_to_its_layer_count(self):
        from sglang.srt.model_executor.dual_group_lane import kv_bearing_layer_count

        self.assertEqual(kv_bearing_layer_count(self._cfg(32)), 32)

    def test_an_empty_config_never_returns_zero(self):
        from sglang.srt.model_executor.dual_group_lane import kv_bearing_layer_count

        class _Bare:
            pass

        # A zero here divides by zero one frame later (that is how the head's
        # first pool sizing died), so the floor is part of the contract.
        self.assertGreaterEqual(kv_bearing_layer_count(_Bare()), 1)


class TestLaneLending(unittest.TestCase):
    """Elastic occupancy stage 2: threshold, hysteresis, reclaim."""

    class _FakeLane:
        lane_id = 0

        def __init__(self, idle_s=99.0, busy=False):
            self._idle = idle_s
            self.has_work = busy

            class _R:
                gpu_id = 0

            self.runner = _R()

        @property
        def idle_seconds(self):
            return self._idle

    def _lending(self, lane, **kw):
        from sglang.srt.model_executor.dual_group_lane import LaneLending

        return LaneLending(lane, lend_mib=64, threshold_s=5.0, **kw)

    def test_nothing_is_lent_before_the_amortization_threshold(self):
        # Refused on the threshold, i.e. BEFORE any device allocation --
        # which is also why this test needs no CUDA.
        lending = self._lending(self._FakeLane(idle_s=1.0))
        self.assertFalse(lending.maybe_lend())
        self.assertFalse(lending.is_lent)

    def test_a_busy_lane_never_lends(self):
        lane = self._FakeLane(idle_s=99.0, busy=True)
        lending = self._lending(lane)
        self.assertFalse(lending.maybe_lend())

    def test_reclaim_is_never_refused_only_counted(self):
        lane = self._FakeLane()
        lending = self._lending(lane)
        # Simulate a lend without touching CUDA.
        lending._borrowed = object()
        import time as _t

        lending._lent_at = _t.monotonic()
        lending.on_lane_work_arrived()
        self.assertFalse(lending.is_lent)
        self.assertEqual(lending.reclaim_events, 1)
        # Below the minimum hold the reclaim still happens -- the guarantee
        # is unconditional; the flap is recorded instead of paid for.
        self.assertEqual(lending.refused_min_hold, 1)

    def test_stats_carry_the_two_mandatory_numbers(self):
        lane = self._FakeLane()
        lending = self._lending(lane)
        stats = lending.stats()
        self.assertIn("threshold_s", stats)  # amortization threshold
        self.assertIn("reclaim", stats)  # reclaim latency


class TestPerLaneGraphSharedOutput(unittest.TestCase):
    """Geteilte-Puffer-Familie, third member: the shared LOGITS buffer.

    Slice C keyed the graph memory POOL by lane (shared intermediates) and the
    GGUF dequant workspace by lane, and left ``GraphSharedOutput`` -- the
    next-token-logits destination every captured decode graph writes into --
    process-wide. Same argument, same failure: a concurrent lane replaying its
    decode graph while the serving group replays its own gives two writers one
    address, and the logits the loser reads are the winner's.
    """

    def setUp(self):
        reset_context()
        from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput

        self._saved = dict(GraphSharedOutput._lane_shared)
        GraphSharedOutput._lane_shared.clear()

    def tearDown(self):
        from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput

        GraphSharedOutput._lane_shared.clear()
        GraphSharedOutput._lane_shared.update(self._saved)
        reset_context()

    class _FakeRunner:
        """Enough of a ModelRunner for ``create_for_model_runner``."""

        device = "cpu"

        def __init__(self, rows=8):
            from sglang.srt.model_executor.cuda_graph_config import (
                default_cuda_graph_config,
            )

            class _A:
                pass

            self.server_args = _A()
            self.server_args.cuda_graph_config = default_cuda_graph_config()
            self.server_args.cuda_graph_config.decode.bs = [1, 2]
            self._rows = rows

        def max_decode_logits_rows(self):
            return self._rows

    def test_each_lane_gets_its_own_logits_buffer(self):
        from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput

        runner = self._FakeRunner()
        serving = GraphSharedOutput.create_for_model_runner(runner)
        self.assertIsNotNone(serving)
        # Same scope, same buffer: within one lane the graphs are serial with
        # each other, which is what makes ONE buffer correct there.
        self.assertIs(GraphSharedOutput.create_for_model_runner(runner), serving)

        with lane_scope(0, None):
            lane0 = GraphSharedOutput.create_for_model_runner(runner)
        with lane_scope(1, None):
            lane1 = GraphSharedOutput.create_for_model_runner(runner)

        self.assertIsNot(lane0, serving)
        self.assertIsNot(lane1, serving)
        self.assertIsNot(lane0, lane1)

    def test_the_buffers_are_distinct_storage_not_just_distinct_objects(self):
        from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput

        runner = self._FakeRunner()
        serving = GraphSharedOutput.create_for_model_runner(runner)
        with lane_scope(0, None):
            lane0 = GraphSharedOutput.create_for_model_runner(runner)
        # The aliasing that matters is at the TENSOR, not the holder.
        a = serving.get_logits_buffer(32, rows=4)
        b = lane0.get_logits_buffer(32, rows=4)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        a.fill_(1.0)
        b.fill_(2.0)
        self.assertEqual(float(a[0, 0]), 1.0)

    def test_the_serving_group_is_unchanged(self):
        """The default path must keep exactly one buffer (backward compat)."""
        from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput

        runner = self._FakeRunner()
        first = GraphSharedOutput.create_for_model_runner(runner)
        second = GraphSharedOutput.create_for_model_runner(self._FakeRunner())
        self.assertIs(first, second)
        self.assertEqual(list(GraphSharedOutput._lane_shared), [None])


class TestLaneVocabShellSelection(unittest.TestCase):
    """Contract 5: type is not a unique selector for the target's vocab shells.

    A multimodal target carries more than one vocab-parallel table, and
    ``modules()`` order decides which one "the first of the right type" is.
    Picking the companion tower's is silent at bring-up and fails one forward
    later inside a fused kernel (measured: a cutlass signature dump from
    ``pre_fc_norm_embedding``, width 1152 against the head's 5120).
    """

    def _shells(self, widths):
        import torch.nn as nn

        from sglang.srt.model_executor.dual_group_lane import (
            LaneLmHeadShell,
            LaneVocabEmbeddingShell,
        )

        class _Part(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.embedding_dim = dim
                self.qweight = None

        target = nn.Module()
        for i, w in enumerate(widths):
            target.add_module(f"embed{i}", LaneVocabEmbeddingShell([_Part(w)]))
            target.add_module(f"head{i}", LaneLmHeadShell([_Part(w)]))
        return target

    def test_the_language_width_is_chosen_not_the_first_registered(self):
        from sglang.srt.model_executor.dual_group_lane import (
            _find_lane_vocab_shells,
        )

        # Companion tower registered FIRST -- the order that made round 2 fail.
        target = self._shells([1152, 5120])
        embed, head = _find_lane_vocab_shells(target, 5120)
        self.assertEqual(embed.embedding_dim, 5120)
        self.assertEqual(head.embedding_dim, 5120)

    def test_a_single_shell_is_taken_as_is(self):
        """Unimodal targets must not need the discriminator at all."""
        from sglang.srt.model_executor.dual_group_lane import (
            _find_lane_vocab_shells,
        )

        target = self._shells([5120])
        embed, head = _find_lane_vocab_shells(target, 5120)
        self.assertIsNotNone(embed)
        self.assertIsNotNone(head)

    def test_an_ambiguous_target_is_refused_loudly(self):
        from sglang.srt.model_executor.dual_group_lane import (
            _find_lane_vocab_shells,
        )

        target = self._shells([5120, 5120])
        with self.assertRaises(ValueError) as ctx:
            _find_lane_vocab_shells(target, 5120)
        self.assertIn("5120", str(ctx.exception))

    def test_no_matching_width_is_refused_rather_than_guessed(self):
        from sglang.srt.model_executor.dual_group_lane import (
            _find_lane_vocab_shells,
        )

        target = self._shells([1152, 2048])
        with self.assertRaises(ValueError):
            _find_lane_vocab_shells(target, 5120)


class TestLaneTargetVsHeadClassification(unittest.TestCase):
    """Contracts 6 and 7: the two lane runners need OPPOSITE answers to
    "is this a draft model?", and both carry ``is_draft_worker=True``.

    Spelled out as ``is_draft_worker and not is_dual_group_lane``, the
    exemption covered the head too, so the head inherited the target's
    64-layer geometry: its full-attention call dispatched into the GDN
    backend (contract 6), and once that was fixed its KV pool had an empty
    full-attention layer map (contract 7). One property, asked everywhere.
    """

    class _Runner:
        def __init__(self, lane, draft):
            self.is_draft_worker = True
            self.is_dual_group_lane = lane
            self.is_dual_group_lane_draft = draft

        @property
        def is_dual_group_lane_target(self):
            from sglang.srt.model_executor.model_runner import ModelRunner

            return ModelRunner.is_dual_group_lane_target.fget(self)

    def test_the_lane_target_is_exempt(self):
        self.assertTrue(self._Runner(lane=True, draft=False).is_dual_group_lane_target)

    def test_the_lane_head_is_not_exempt(self):
        self.assertFalse(self._Runner(lane=True, draft=True).is_dual_group_lane_target)

    def test_an_ordinary_draft_worker_is_not_exempt(self):
        self.assertFalse(
            self._Runner(lane=False, draft=False).is_dual_group_lane_target
        )

    def test_the_exemption_sites_all_ask_the_property(self):
        """The three sites must not respell the condition and drift apart."""
        import inspect

        from sglang.srt.layers.attention import attention_registry
        from sglang.srt.model_executor import model_runner, model_runner_kv_cache_mixin

        for mod in (attention_registry, model_runner, model_runner_kv_cache_mixin):
            src = inspect.getsource(mod)
            # No site may test the lane flag directly against is_draft_worker.
            self.assertNotIn(
                'and not getattr(self, "is_dual_group_lane", False)',
                src,
                f"{mod.__name__} respells the lane-target condition",
            )
            self.assertNotIn(
                'and not getattr(runner, "is_dual_group_lane", False)',
                src,
                f"{mod.__name__} respells the lane-target condition",
            )


class TestLaneSpecDispatch(unittest.TestCase):
    """The serial tick and the concurrent worker must dispatch the SAME way.

    Round 1 wired the speculative round into the concurrent worker only, so a
    serial lane built the NEXTN head and then never asked it for a proposal.
    Serial is the default mode, so the chain was unreachable by default.
    """

    def test_both_step_paths_route_a_spec_lane_to_the_spec_round(self):
        import inspect

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        for fn in (DualGroupLane.tick, DualGroupLane._step_locked_scope):
            src = inspect.getsource(fn)
            self.assertIn("_prefill", src, f"{fn.__name__} lost the prefill arm")
            self.assertIn(
                "_job_spec_on", src, f"{fn.__name__} does not consult _job_spec_on"
            )
            self.assertIn(
                "_spec_round", src, f"{fn.__name__} never runs a speculative round"
            )
            self.assertIn(
                "_decode_step", src, f"{fn.__name__} lost the plain decode arm"
            )
        # The dispatch asks _job_spec_on, so the guarantee this test protects
        # only holds if that helper is still anchored on the server flag.
        self.assertIn(
            "spec_active",
            inspect.getsource(DualGroupLane._job_spec_on),
            "_job_spec_on no longer consults spec_active",
        )

    def test_job_spec_override_defaults_to_the_server_flag(self):
        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        lane = DualGroupLane.__new__(DualGroupLane)
        lane.draft_runner = object()
        # Absent key: the flag decides, so the default path is untouched.
        self.assertTrue(lane._job_spec_on({}))
        self.assertTrue(lane._job_spec_on({"spec": None}))
        # Explicit opt-out: the gate's reference side.
        self.assertFalse(lane._job_spec_on({"spec": False}))
        # No head assembled: no job may speculate.
        lane.draft_runner = None
        self.assertFalse(lane._job_spec_on({"spec": True}))


class _StubBatch:
    """Just enough ScheduleBatch for the accept rule: the strategy under test
    owns the token bookkeeping, not the allocation."""

    device = "cpu"
    input_ids = None

    def prepare_for_decode(self):
        pass


def _stub_out(token_id, vocab=8):
    import torch

    out = type("LogitsOutput", (), {})()
    out.next_token_logits = torch.zeros(1, vocab)
    out.next_token_logits[0][token_id] = 1.0
    out.hidden_states = torch.zeros(1, 4)
    return out


class TestLaneVerifyStrategy(unittest.TestCase):
    """#274 rounds 3+4: which verify forward the lane uses, and why.

    The batched continued extend advances the target's RECURRENT state over
    rejected candidates with no way to restore it, so it stops tracking the
    lane's own no-spec continuation. TARGET_VERIFY parks a state per draft
    step and commits the accepted one; sequential decode sidesteps the problem
    at one forward per token. The default must be a coherent one.
    """

    def _lane(self):
        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        return DualGroupLane.__new__(DualGroupLane)

    def test_default_is_the_coherent_strategy(self):
        import inspect

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        # Round 5 made TARGET_VERIFY coherent; round 6 captured it and made it
        # dominant (35.9 vs 75-96 ms per round); rounds 7a/7b kept every byte
        # gate green on it. The default was promoted at the R7b merge -- this
        # assertion flipped with it, on purpose: the pair is the record that
        # the change was chosen rather than drifted into. seqdecode stays one
        # explicit word away as the fallback.
        self.assertEqual(self._lane()._verify_mode({}), "target_verify")
        doc = inspect.getdoc(DualGroupLane._verify_mode) or ""
        self.assertIn("default since R7b", doc)

    def test_target_verify_stays_reachable_but_only_on_request(self):
        self.assertEqual(
            self._lane()._verify_mode({"verify": "target_verify"}), "target_verify"
        )

    def test_batched_extend_stays_reachable_but_only_on_request(self):
        self.assertEqual(self._lane()._verify_mode({"verify": "extend"}), "extend")

    def test_unknown_strategy_is_refused_rather_than_guessed(self):
        import torch

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        lane = DualGroupLane.__new__(DualGroupLane)
        job = {
            "_batch": _StubBatch(),
            "_req": None,
            "_req_pool_idx": 0,
            "_kv_len": 4,
            "_next": torch.tensor([1]),
            "verify": "fastest",
        }
        with self.assertRaises(ValueError):
            lane._verify(job, [2, 3, 4])

    def test_the_defect_is_named_where_the_batched_path_lives(self):
        import inspect

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        doc = inspect.getdoc(DualGroupLane._verify) or ""
        # Not prose-policing: this is the one place a future reader decides
        # whether to "optimise" the default back to the fast, wrong path.
        self.assertIn("TARGET_VERIFY", doc)
        self.assertIn("recurrent", doc.lower())

    def test_the_measured_boundary_of_target_verify_is_written_down(self):
        """What is proven about TARGET_VERIFY, at the code that runs it.

        The mode is coherent since round 5, so the docstring no longer has to
        carry a defect -- it has to carry the reason the mode is STILL not the
        default, which is a number and not an opinion. A future reader who
        only sees "TARGET_VERIFY is the right mode" would promote it and
        measure nothing; the break-even has to be findable from the same
        docstring, together with the condition that changes it.
        """
        import inspect

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        doc = inspect.getdoc(DualGroupLane._verify_by_target_verify) or ""
        # Round 6 captured the forward and moved the number: 4.78 -> 2.22. The
        # docstring has to carry the CURRENT one, or a future reader promotes
        # the flag against a break-even that no longer exists.
        self.assertIn("2.22", doc)
        self.assertIn("graph capture", doc)
        # And the root cause, so nobody re-derives it from the symptom.
        self.assertIn("local_row_split", doc)

    def test_seqdecode_emits_accepted_prefix_then_the_targets_own_token(self):
        """The accept rule itself, with the forwards stubbed out.

        Round shape: candidates are [last, *proposals]; the target's answer
        after candidate i is compared against proposal i.
        """
        import torch

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        class _Fake(DualGroupLane):
            def __init__(self, target_says):
                self._says = list(target_says)

            def _timed_forward_raw(self, batch, capture_mode=None):
                return _stub_out(self._says.pop(0)), 1.0

        # The target says 3 after the last token, and proposal 0 was 3, so it
        # is accepted. Then it says 7 where the proposal was 5: rejected, and
        # the target's own 7 is what the round emits.
        lane = _Fake([3, 7])
        job = {"_batch": _StubBatch(), "_next": torch.tensor([1])}
        emitted, n_accept, _ = lane._verify_by_decode(job, [3, 5, 6], n_cached=10)
        self.assertEqual(emitted, [3, 7])
        self.assertEqual(n_accept, 1)
        # KV advanced by exactly the emitted count, not by the whole block.
        self.assertEqual(job["_kv_len"], 10 + 2)

    def test_seqdecode_rejecting_everything_still_emits_one_token(self):
        import torch

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        class _Fake(DualGroupLane):
            def _timed_forward_raw(self, batch, capture_mode=None):
                return _stub_out(4), 1.0

        lane = _Fake.__new__(_Fake)
        job = {"_batch": _StubBatch(), "_next": torch.tensor([1])}
        emitted, n_accept, _ = lane._verify_by_decode(job, [2, 2, 2], n_cached=5)
        self.assertEqual(n_accept, 0)
        self.assertEqual(emitted, [4])
        self.assertEqual(job["_kv_len"], 6)


class TestLaneChainVerifyInput(unittest.TestCase):
    """#274 round 4: the TARGET_VERIFY input the lane hands to the target.

    The mask layout is a CONTRACT with ``build_tree_kernel_efficient`` /
    ``EagleVerifyInput.generate_attn_arg_prefill``; getting it wrong is silent
    (the forward runs, the logits are just attended over the wrong keys), so
    it is pinned here rather than discovered on the rig.
    """

    def test_chain_mask_is_prefix_visible_and_lower_triangular(self):
        from sglang.srt.model_executor.dual_group_lane import lane_chain_verify_mask

        n_cached, d = 5, 3
        mask = lane_chain_verify_mask(n_cached, d).view(d, n_cached + d)
        self.assertEqual(mask.numel(), n_cached * d + d * d)
        self.assertTrue(bool(mask[:, :n_cached].all()))
        block = mask[:, n_cached:]
        self.assertEqual(
            block.tolist(),
            [[True, False, False], [True, True, False], [True, True, True]],
        )

    def test_verify_input_describes_the_chain(self):
        from sglang.srt.model_executor.dual_group_lane import (
            build_lane_chain_verify_input,
        )

        vi = build_lane_chain_verify_input([11, 12, 13, 14], n_cached=7)
        self.assertEqual(vi.draft_token_num, 4)
        self.assertEqual(vi.topk, 1)
        self.assertEqual(vi.spec_steps, 3)
        self.assertEqual(vi.positions.tolist(), [7, 8, 9, 10])
        self.assertEqual(vi.draft_token.tolist(), [11, 12, 13, 14])
        # A chain: every node's successor is the next one, none has a sibling.
        self.assertEqual(vi.retrieve_next_token.tolist(), [[1, 2, 3, -1]])
        self.assertEqual(vi.retrieve_next_sibling.tolist(), [[-1, -1, -1, -1]])
        self.assertEqual(vi.custom_mask.numel(), 7 * 4 + 4 * 4)


class _StubVerifyBatch:
    """A ScheduleBatch stand-in with only the fields a verify round writes."""

    def __init__(self, n_cached, pool_width=64):
        import torch

        self.device = "cpu"
        self.input_ids = None
        self.spec_info = None
        self.forward_mode = None
        self.capture_hidden_mode = None
        self.out_cache_loc = None
        self.seq_lens_sum = None
        self.seq_lens = torch.tensor([n_cached])
        self.seq_lens_cpu = torch.tensor([n_cached])
        self.orig_seq_lens = torch.tensor([n_cached])
        self.freed = []

        outer = self

        class _Alloc:
            def alloc(self, n):
                return torch.arange(100, 100 + n, dtype=torch.int64)

            def free(self, idx):
                outer.freed.append(idx.tolist())

        class _Pool:
            req_to_token = torch.zeros((2, pool_width), dtype=torch.int32)

        self.token_to_kv_pool_allocator = _Alloc()
        self.req_to_token_pool = _Pool()


class _StubReq:
    decode_batch_idx = 0
    kv_committed_len = 0
    kv_allocated_len = 0


class TestLaneTargetVerifyRound(unittest.TestCase):
    """#274 round 4: the accept rule and the bookkeeping of one TARGET_VERIFY.

    The forward and the state commit are stubbed; what is under test is that
    the round emits the accepted prefix plus the target's own token, frees
    exactly the rejected slots, and ALWAYS commits the recurrent state of the
    last accepted step -- skipping that commit reproduces the round-1 defect.
    """

    def _lane(self, target_says, n_cached=10):
        import torch

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        calls = []

        class _Backend:
            def update_mamba_state_after_mtp_verify(self, **kw):
                calls.append(kw)

        class _Args:
            page_size = 1

        class _Runner:
            attn_backend = _Backend()
            server_args = _Args()
            model = None

        class _Fake(DualGroupLane):
            def _verify_state_buffers(self, draft_token_num):
                return None

            def _timed_forward_raw(self, batch, capture_mode=None):
                out = type("LogitsOutput", (), {})()
                out.hidden_states = torch.zeros(len(target_says), 4)
                return out, 2.5

            def _candidate_logits(self, hidden_states):
                logits = torch.zeros(len(target_says), 32)
                for row, tok in enumerate(target_says):
                    logits[row][tok] = 1.0
                return logits

        lane = _Fake.__new__(_Fake)
        lane.runner = _Runner()
        job = {
            "_batch": _StubVerifyBatch(n_cached),
            "_req": _StubReq(),
            "_req_pool_idx": 0,
            "_kv_len": n_cached,
            "_next": torch.tensor([1]),
        }
        return lane, job, calls

    def test_partial_accept_frees_only_the_rejected_slots(self):
        # Candidates [1, 3, 5, 6]. The target says 3 after the first (proposal
        # 0 accepted) and 7 where proposal 1 was 5 (rejected).
        lane, job, calls = self._lane([3, 7, 0, 0])
        emitted, n_accept, ms = lane._verify_by_target_verify(job, [3, 5, 6], 10)
        self.assertEqual(emitted, [3, 7])
        self.assertEqual(n_accept, 1)
        self.assertEqual(ms, 2.5)
        self.assertEqual(job["_kv_len"], 12)
        self.assertEqual(job["_batch"].freed, [[102, 103]])
        self.assertEqual(job["_batch"].seq_lens.tolist(), [12])
        self.assertEqual(job["_req"].kv_committed_len, 12)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["last_correct_step_indices"].tolist(), [n_accept])

    def test_full_accept_frees_nothing_and_commits_the_last_step(self):
        lane, job, calls = self._lane([3, 5, 6, 9])
        emitted, n_accept, _ = lane._verify_by_target_verify(job, [3, 5, 6], 10)
        self.assertEqual(emitted, [3, 5, 6, 9])
        self.assertEqual(n_accept, 3)
        self.assertEqual(job["_batch"].freed, [])
        self.assertEqual(job["_kv_len"], 14)
        self.assertEqual(calls[0]["last_correct_step_indices"].tolist(), [3])

    def test_the_verify_leaves_no_spec_input_on_the_batch(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        lane, job, _ = self._lane([9, 0, 0, 0])
        lane._verify_by_target_verify(job, [3, 5, 6], 10)
        # A stale EagleVerifyInput on the ScheduleBatch would put the NEXT
        # forward (a plain decode, or the next round's prefill) into the
        # verify plan with the previous round's mask.
        self.assertIsNone(job["_batch"].spec_info)
        self.assertEqual(job["_batch"].forward_mode, ForwardMode.DECODE)

    def test_per_candidate_rows_are_taken_only_when_the_count_matches(self):
        """Contract 8's failure mode, made loud.

        ``next_token_logits`` is per-candidate under TARGET_VERIFY and per
        REQUEST everywhere else. Reading it positionally when it is the latter
        is silent and wrong, so the row count decides which source is used.
        """
        import torch

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        rebuilt = []

        class _Fake(DualGroupLane):
            def _candidate_logits(self, hidden_states):
                rebuilt.append(hidden_states.shape)
                return torch.zeros(hidden_states.shape[0], 8)

        lane = _Fake.__new__(_Fake)
        out = type("LogitsOutput", (), {})()
        out.hidden_states = torch.zeros(4, 4)

        out.next_token_logits = torch.zeros(4, 8)
        out.next_token_logits[2][5] = 1.0
        self.assertEqual(lane._verify_predictions(out, 4).tolist(), [0, 0, 5, 0])
        self.assertEqual(rebuilt, [])

        # One row per REQUEST (the shape that made round 2's accept length
        # exactly 1.000): rebuild instead of indexing rows that are not there.
        out.next_token_logits = torch.zeros(1, 8)
        lane._verify_predictions(out, 4)
        self.assertEqual(rebuilt, [torch.Size([4, 4])])

    def test_candidate_slots_are_published_into_req_to_token(self):
        lane, job, _ = self._lane([3, 7, 0, 0])
        lane._verify_by_target_verify(job, [3, 5, 6], 10)
        row = job["_batch"].req_to_token_pool.req_to_token[0]
        self.assertEqual(row[10:14].tolist(), [100, 101, 102, 103])


class TestLaneVerifyGraphEntry(unittest.TestCase):
    """#274 round 6: the lane's verify as a SECOND capture entry.

    The lane's plain decode entry is the thing under protection here. It has
    been byte-green for five rounds, and every plausible way to give the verify
    a graph either deletes it (un-clearing ``speculative_algorithm`` on the args
    view re-shapes the whole runner to TARGET_VERIFY) or re-records it (a FULL
    hidden-mode batch reaching ``recapture_if_needed`` tears down every captured
    decode graph and captures them again). These tests pin the narrow path
    between the two.
    """

    def _runner(self, *, verify_tokens=(4,), captured=None, max_bs=1, ntpb=1):
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        r = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        r.capture_forward_mode = ForwardMode.DECODE
        r.capture_hidden_mode = CaptureHiddenMode.NULL
        r.num_tokens_per_bs = ntpb
        r.max_bs = max_bs
        r._lane_verify_tokens = tuple(verify_tokens)
        r._lane_verify_active = None
        r._lane_verify_captured = (
            frozenset(verify_tokens) if captured is None else frozenset(captured)
        )
        r._lane_draft_capture = False
        r._lane_draft_captured = False
        r._wl_block_graph = False
        r._sess_block_graph = False
        return r

    def _batch(self, *, bs=1, tokens=4, verify=True):
        import torch

        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        fb = type("FB", (), {})()
        fb.batch_size = bs
        fb.forward_mode = ForwardMode.TARGET_VERIFY if verify else ForwardMode.DECODE
        fb.input_ids = torch.zeros(tokens, dtype=torch.int64)
        fb.replace_embeds = None
        return fb

    def test_the_args_view_still_clears_the_algorithm(self):
        """The opening is targeted, and this is what "targeted" means.

        Round 6's brief was to stop the lane args view clearing
        ``speculative_algorithm`` for the CAPTURE branch. It is cleared here as
        before, and the capture branch is opened by a field of its own instead
        -- because un-clearing it is not a capture-only change: it flips
        ``decode_num_tokens_per_bs`` to K+1 and ``capture_forward_mode`` to
        TARGET_VERIFY for the WHOLE runner, which does not add the verify entry,
        it replaces the decode entry with it. The lane still runs plain decode
        steps (every no-spec job, and every job with speculation off), so that
        entry has to survive.
        """
        from types import SimpleNamespace

        from sglang.srt.model_executor.dual_group_lane import _lane_server_args_view

        args = SimpleNamespace(
            speculative_algorithm="NEXTN",
            speculative_draft_model_path="p",
            speculative_cross_algorithm=True,
            dual_group_lane_budget_mib=1600,
            dual_group_lane_max_requests=1,
            dual_group_lane_speed_dial=None,
            dual_group_lane_eager=False,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(bs=[16, 32, 64], max_bs=64)
            ),
        )
        view = _lane_server_args_view(args)
        self.assertIsNone(view.speculative_algorithm)

    def test_the_shape_scope_swaps_and_restores(self):
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )

        r = self._runner()
        with r.lane_verify_shape():
            self.assertTrue(r._lane_verify_active)
            self.assertEqual(r.capture_forward_mode, ForwardMode.TARGET_VERIFY)
            self.assertEqual(r.num_tokens_per_bs, 4)
            self.assertEqual(r.capture_hidden_mode, CaptureHiddenMode.FULL)
        self.assertIsNone(r._lane_verify_active)
        self.assertEqual(r.capture_forward_mode, ForwardMode.DECODE)
        self.assertEqual(r.num_tokens_per_bs, 1)
        self.assertEqual(r.capture_hidden_mode, CaptureHiddenMode.NULL)

    def test_the_shape_scope_restores_on_a_raising_forward(self):
        """A verify that throws must not leave the decode path mis-shaped.

        Without the restore the next plain decode step would compute
        ``raw_num_token = bs * 4`` and look up a graph key that does not exist
        -- a KeyError one forward away from the real error, which is how a
        one-line failure becomes an unreadable one.
        """
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        r = self._runner()
        with self.assertRaises(RuntimeError):
            with r.lane_verify_shape():
                raise RuntimeError("verify blew up")
        self.assertIsNone(r._lane_verify_active)
        self.assertEqual(r.capture_forward_mode, ForwardMode.DECODE)
        self.assertEqual(r.num_tokens_per_bs, 1)

    def test_the_two_entries_are_told_apart_by_the_variant_only(self):
        """Both entries key on ShapeKey.size 1; the variant is the whole key.

        ``_capture_graph_size`` returns the padded BS for a non-ragged decode
        runner, and the lane's verify is bs 1 exactly like its decode graph. If
        the variant did not differ, the second capture would overwrite the first
        under the same key and the lane's no-spec path would silently start
        replaying a TARGET_VERIFY graph.
        """
        r = self._runner()
        self.assertIsNone(r._wl_variant_label(None))
        with r.lane_verify_shape():
            self.assertEqual(r._wl_variant_label(None), r.LANE_VERIFY_VARIANT + "4")
        self.assertIsNone(r._wl_variant_label(None))

    def test_admission_is_shape_exact_and_never_pads(self):
        r = self._runner()
        with r.lane_verify_shape():
            self.assertTrue(r.can_run_graph(self._batch()))
            # A shorter chain is NOT padded into the captured entry: padding a
            # verify would change which candidate rows exist.
            self.assertFalse(r.can_run_graph(self._batch(tokens=3)))
            self.assertFalse(r.can_run_graph(self._batch(tokens=8)))
            self.assertFalse(r.can_run_graph(self._batch(bs=2)))
            self.assertFalse(r.can_run_graph(self._batch(verify=False)))

    def test_nothing_captured_means_eager_not_a_key_error(self):
        r = self._runner(captured=())
        with r.lane_verify_shape():
            self.assertFalse(r.can_run_graph(self._batch()))

    def test_the_lane_scope_yields_false_without_a_captured_entry(self):
        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        lane = DualGroupLane.__new__(DualGroupLane)
        lane.runner = type("R", (), {"decode_cuda_graph_runner": None})()
        with lane._verify_graph_scope(4) as captured:
            self.assertFalse(captured)

    def test_the_lane_scope_refuses_a_chain_of_another_length(self):
        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        lane = DualGroupLane.__new__(DualGroupLane)
        lane.runner = type(
            "R", (), {"decode_cuda_graph_runner": self._runner(verify_tokens=(4,))}
        )()
        with lane._verify_graph_scope(5) as captured:
            self.assertFalse(captured)
        with lane._verify_graph_scope(4) as captured:
            self.assertTrue(captured)

    def test_the_capture_stand_in_carries_a_real_chain_mask(self):
        """flashinfer picks its mask mode by BUFFER PRESENCE, not by content.

        ``_create_prefill_wrappers`` binds ``custom_mask_buf`` only when the
        capture-time ``spec_info`` carries a ``custom_mask``. A stand-in without
        one records a causal kernel, and the live chain-masked round then
        replays it -- a wrong answer, not a crash. So the stand-in is built by
        the same builder the live round uses.
        """
        import inspect

        from sglang.srt.model_executor.dual_group_lane import (
            build_lane_chain_verify_input,
        )
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        src = inspect.getsource(DecodeCudaGraphRunner._lane_verify_spec_info)
        self.assertIn("build_lane_chain_verify_input", src)
        spec = build_lane_chain_verify_input([0] * 4, 7)
        self.assertIsNotNone(spec.custom_mask)
        self.assertEqual(spec.custom_mask.numel(), 4 * (7 + 4))
        self.assertEqual(spec.draft_token_num, 4)

    def test_the_widths_are_cut_once_for_the_widest_entry(self):
        """One sizing for both entries, and it is a PRODUCT, not a max.

        The mamba backend reads its per-slot verify width back out of the pair
        it is handed as ``max_num_tokens // max_bs``, so ``max(max_bs, K+1)``
        would give the GDN verify a query-start-loc ladder of step 1 while the
        forward runs K+1 rows per slot. Re-cutting the buffers per entry is the
        other tempting shortcut and is worse: the decode graphs already hold
        pointers into the first cut.
        """
        import inspect

        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        src = inspect.getsource(DecodeCudaGraphRunner.__init__)
        self.assertIn(
            "self.max_num_token = self.max_bs * max(",
            src,
        )
        # and the token count reaches the GGUF dispatch, so a replay cannot
        # pick a different kernel than the capture recorded (round 5's defect
        # lived in exactly that <= 8-row window).
        self.assertIn("_register_gguf_decode_buckets(self._lane_verify_tokens", src)

    def test_the_shared_logits_buffer_grows_for_the_lane_only(self):
        from types import SimpleNamespace
        from unittest import mock

        from sglang.srt.model_executor import model_runner as mr_mod
        from sglang.srt.model_executor.model_runner import ModelRunner

        mr = ModelRunner.__new__(ModelRunner)
        mr.server_args = SimpleNamespace(max_speculative_num_draft_tokens=None)
        mr.decode_num_tokens_per_bs = lambda **_kw: 1
        with mock.patch.object(
            mr_mod, "get_batch_sizes_to_capture", return_value=([1], [])
        ):
            mr.dual_group_lane_verify_tokens = (4,)
            self.assertEqual(ModelRunner.max_decode_logits_rows(mr), 4)
            # Nothing else grows: without the lane field this is the row count
            # it always was, so the serving group's shared buffer is untouched.
            mr.dual_group_lane_verify_tokens = None
            self.assertEqual(ModelRunner.max_decode_logits_rows(mr), 1)


class TestLaneHeadGraphEntry(unittest.TestCase):
    """#274 round 7a: the lane's NEXTN HEAD as a captured entry.

    Round 6 left this as a NAMED GAP with a reason: the generic decode capture
    builds ``spec_info=None`` and an MTP forward dereferences it. These tests
    pin the branch that closes it, and pin that it stays confined to the head's
    own runner -- the lane target's plain decode entry is byte-green over six
    rounds and this round is not allowed to reach it.
    """

    def _head_runner(self, *, captured=True, max_bs=1, hidden=8):
        import torch

        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        r = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        r.capture_forward_mode = ForwardMode.DECODE
        r.capture_hidden_mode = CaptureHiddenMode.NULL
        r.num_tokens_per_bs = 1
        r.max_bs = max_bs
        r.max_num_token = max_bs
        r._lane_verify_tokens = None
        r._lane_verify_active = None
        r._lane_verify_captured = frozenset()
        r._lane_draft_capture = True
        r._lane_draft_captured = captured
        r._lane_draft_hidden = torch.zeros((max_bs, hidden))
        r._wl_block_graph = False
        r._sess_block_graph = False
        return r

    def _draft_batch(self, *, bs=1, hidden_rows=1, hidden=8, decode=True):
        import torch

        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        fb = type("FB", (), {})()
        fb.batch_size = bs
        fb.forward_mode = ForwardMode.DECODE if decode else ForwardMode.TARGET_VERIFY
        fb.input_ids = torch.zeros(bs, dtype=torch.int64)
        fb.replace_embeds = None
        fb.spec_info = type("SI", (), {})()
        fb.spec_info.hidden_states = (
            None
            if hidden_rows is None
            else torch.arange(hidden_rows * hidden, dtype=torch.float).view(
                hidden_rows, hidden
            )
        )
        return fb

    def test_the_capture_stand_in_is_a_real_draft_input(self):
        """The whole of the round-6 gap, in one predicate.

        Without this branch ``get_spec_info`` falls through to None and the MTP
        forward dies on ``spec_info.hidden_states``. With it the capture sees a
        real ``EagleDraftInput`` whose hidden states are the STATIC buffer the
        replay copies into -- a graph input, not a per-round object.
        """
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        r = self._head_runner()
        spec = r.get_spec_info(1)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.capture_hidden_mode, CaptureHiddenMode.LAST)
        self.assertEqual(spec.hidden_states.data_ptr(), r._lane_draft_hidden.data_ptr())

    def test_the_hidden_states_are_copied_not_aliased(self):
        """The replay writes THROUGH to the captured address.

        Aliasing (rebinding ``spec_info.hidden_states`` to the live tensor)
        would leave the captured graph reading its own stale buffer -- the
        classic shared-buffer failure of this feature (#274 D2/D3), which does
        not announce itself.
        """
        r = self._head_runner()
        fb = self._draft_batch()
        before = r._lane_draft_hidden.data_ptr()
        r._lane_draft_load_hidden(fb)
        self.assertEqual(r._lane_draft_hidden.data_ptr(), before)
        self.assertTrue(
            bool((r._lane_draft_hidden[0] == fb.spec_info.hidden_states[0]).all())
        )

    def test_admission_needs_a_captured_graph_and_hidden_states(self):
        r = self._head_runner()
        self.assertTrue(r.can_run_graph(self._draft_batch()))
        # No hidden states -> nothing to copy -> eager, not a stale replay.
        self.assertFalse(r.can_run_graph(self._draft_batch(hidden_rows=None)))
        # Not a decode -> not this entry.
        self.assertFalse(r.can_run_graph(self._draft_batch(decode=False)))
        # Nothing captured yet (the eager warmup, or --no-...-head-graph).
        self.assertFalse(
            self._head_runner(captured=False).can_run_graph(self._draft_batch())
        )

    def test_the_head_entry_carries_its_own_variant(self):
        r = self._head_runner()
        self.assertEqual(r._wl_variant_label(None), r.LANE_DRAFT_VARIANT)

    def test_capture_and_replay_derive_the_same_variant(self):
        """The two sides of the graph key, and boot 2 of round 7a is why.

        Replay derives its label from ``_wl_variant_label``; the plain capture
        loop passes the LoRA variant straight through, which is None without
        LoRA. Under the head that recorded ``variant_label=None`` and looked up
        ``variant_label='lanedraft'`` -- a KeyError on the first head forward,
        AFTER a boot that logged a successful capture. A label mismatch is
        invisible at capture time by construction, so the invariant is pinned
        here instead.
        """
        import inspect

        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        src = inspect.getsource(DecodeCudaGraphRunner._capture_one_stream)
        self.assertIn("capture_label = (", src)
        self.assertIn("self.LANE_DRAFT_VARIANT", src)
        r = self._head_runner()
        capture_label = r.LANE_DRAFT_VARIANT if r._lane_draft_capture else None
        self.assertEqual(capture_label, r._wl_variant_label(None))

    def test_the_per_job_eager_falsifier_restores(self):
        """``head_graph: false`` is a per-JOB gate and restores unconditionally.

        Per job, not per process, for the same reason as round 6's
        ``verify_graph``: the replay arm and the eager arm of the byte gate have
        to come from ONE boot.
        """
        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        lane = DualGroupLane.__new__(DualGroupLane)
        runner = self._head_runner()
        lane.draft_runner = type("R", (), {"decode_cuda_graph_runner": runner})()
        with lane._head_graph_scope({"head_graph": False}):
            self.assertFalse(runner._lane_draft_captured)
        self.assertTrue(runner._lane_draft_captured)
        with self.assertRaises(RuntimeError):
            with lane._head_graph_scope({"head_graph": False}):
                raise RuntimeError("head blew up")
        self.assertTrue(runner._lane_draft_captured)
        # Absent key: nothing is touched at all.
        with lane._head_graph_scope({}):
            self.assertTrue(runner._lane_draft_captured)

    def test_only_the_decode_phase_is_given_back_to_the_head(self):
        """PREFILL stays disabled, and that is a decision, not an omission.

        The head's extend shapes follow the target's prompt (no fixed ladder)
        and its prefill is one forward per JOB against K per ROUND -- the time
        is not there, and a capture would be another VRAM post for it.
        """
        from types import SimpleNamespace

        from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
        from sglang.srt.model_executor.dual_group_lane import (
            _disable_graph_phases,
            _enable_decode_graph_phase,
        )

        args = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=Backend.FULL),
                prefill=SimpleNamespace(backend=Backend.FULL),
            )
        )
        _disable_graph_phases(args)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)
        _enable_decode_graph_phase(args)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)
        # The legacy booleans move WITH the phase config: contract 4 is what
        # happens when the two readers disagree.
        self.assertFalse(args.disable_cuda_graph)
        self.assertFalse(args.disable_decode_cuda_graph)
        self.assertTrue(args.disable_prefill_cuda_graph)
        del Phase

    def test_the_capture_flag_is_set_after_the_pool_not_before(self):
        """ORDERING, and it cost boot 1 of round 7a to learn.

        ``alloc_memory_pool`` re-inits the whole block of graph-runner fields
        that ``dual_group_lane_draft_capture`` lives in (the same block that
        holds ``dual_group_lane_verify_tokens``), and it runs inside the head's
        scoped bring-up. A flag written before that call is silently gone by
        the time the capture reads it -- and the failure is not a missing flag
        but round 6's named gap verbatim, ``'NoneType' object has no attribute
        'hidden_states'``, i.e. it looks like the fix was never made.
        """
        import inspect

        from sglang.srt.model_executor import dual_group_lane as dgl

        src = inspect.getsource(dgl._finish_lane_draft_runner_scoped)
        pool = src.index("alloc_memory_pool")
        flag = src.index("dual_group_lane_draft_capture")
        graphs = src.index("init_cuda_graphs")
        self.assertLess(pool, flag, "capture flag set before the pool re-init")
        self.assertLess(flag, graphs, "capture flag set after the capture")
        # And the outer function must NOT set it (that is the trap).
        outer = inspect.getsource(dgl._finish_lane_draft_runner)
        self.assertNotIn(
            "draft_runner.dual_group_lane_draft_capture = True",
            outer,
            "the flag is set where alloc_memory_pool will undo it",
        )


class TestLaneSpecRungLadder(unittest.TestCase):
    """#274 round 7a: K as a ladder of PRE-CAPTURED entries.

    The contract under test is that a rung change is a graph-key flip and
    never a re-capture: capture happens once, at boot, for every configured
    rung, and the runtime only ever selects among them.
    """

    def _runner(self, rungs=(2, 3, 4), captured=None):
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        r = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        r.capture_forward_mode = ForwardMode.DECODE
        r.capture_hidden_mode = CaptureHiddenMode.NULL
        r.num_tokens_per_bs = 1
        r.max_bs = 1
        r._lane_verify_tokens = tuple(rungs)
        r._lane_verify_active = None
        r._lane_verify_captured = frozenset(rungs if captured is None else captured)
        r._lane_draft_capture = False
        r._lane_draft_captured = False
        r._wl_block_graph = False
        r._sess_block_graph = False
        return r

    def _batch(self, tokens):
        import torch

        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        fb = type("FB", (), {})()
        fb.batch_size = 1
        fb.forward_mode = ForwardMode.TARGET_VERIFY
        fb.input_ids = torch.zeros(tokens, dtype=torch.int64)
        fb.replace_embeds = None
        return fb

    def test_every_rung_gets_its_own_graph_key(self):
        """One label for the whole ladder would let the last capture win.

        Every rung is bs 1 and every rung is TARGET_VERIFY, so ``ShapeKey.size``
        is 1 for all of them; the variant is the entire discriminator, exactly
        as it was between the verify and the plain decode entry in round 6.
        """
        r = self._runner()
        labels = set()
        for rung in r._lane_verify_tokens:
            with r.lane_verify_shape(rung):
                labels.add(r._wl_variant_label(None))
        self.assertEqual(len(labels), len(r._lane_verify_tokens))

    def test_a_rung_flip_never_recaptures(self):
        """Selecting a rung touches state, not the backend.

        The property is structural: ``lane_verify_shape`` swaps four scalars
        and nothing else, and the capture entry point is only reachable from
        ``_capture_one_stream``. If a flip could re-record, the lane's rounds
        would pay a capture at every policy switch.
        """
        import inspect

        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        src = inspect.getsource(DecodeCudaGraphRunner.lane_verify_shape)
        for forbidden in ("capture_one_shape", "self.capture(", "cleanup"):
            self.assertNotIn(forbidden, src)
        # And every rung is recorded up front, in ONE pass over the ladder.
        cap = inspect.getsource(DecodeCudaGraphRunner._capture_one_stream)
        self.assertIn("for rung in sorted(self._lane_verify_tokens", cap)

    def test_admission_is_per_rung_and_against_what_was_captured(self):
        r = self._runner(rungs=(2, 4), captured=(4,))
        with r.lane_verify_shape(4):
            self.assertTrue(r.can_run_graph(self._batch(4)))
            self.assertFalse(r.can_run_graph(self._batch(2)))
        with r.lane_verify_shape(2):
            # Configured but NOT recorded (a thinned ladder, or a capture that
            # raised): eager, never the neighbour's graph.
            self.assertFalse(r.can_run_graph(self._batch(2)))

    def test_the_shape_scope_refuses_a_rung_off_the_ladder(self):
        r = self._runner(rungs=(2, 4))
        with self.assertRaises(AssertionError):
            with r.lane_verify_shape(3):
                pass

    def test_the_lane_scope_picks_the_rung_it_is_asked_for(self):
        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        lane = DualGroupLane.__new__(DualGroupLane)
        runner = self._runner(rungs=(2, 4))
        lane.runner = type("R", (), {"decode_cuda_graph_runner": runner})()
        for d in (2, 4):
            with lane._verify_graph_scope(d) as captured:
                self.assertTrue(captured)
                self.assertEqual(runner._lane_verify_active, d)
        with lane._verify_graph_scope(3) as captured:
            self.assertFalse(captured)

    def test_k_zero_is_the_plain_decode_entry_not_a_one_row_verify(self):
        """K=0 costs no graph, because it already has one.

        The ladder's cheapest rung is the lane's existing no-spec decode entry.
        Recording a one-row TARGET_VERIFY for it would be a second graph AND a
        different kernel for the same work.
        """
        from types import SimpleNamespace

        from sglang.srt.model_executor.dual_group_lane import resolve_lane_spec_rungs

        args = SimpleNamespace(
            dual_group_lane_spec_rungs="0,1,3", dual_group_lane_spec_steps=3
        )
        self.assertEqual(resolve_lane_spec_rungs(args), (0, 1, 3))
        verify_rungs = tuple(k + 1 for k in resolve_lane_spec_rungs(args) if k >= 1)
        self.assertEqual(verify_rungs, (2, 4))

    def test_the_gdn_verify_stride_follows_the_batch_not_the_boot(self):
        """Ladder defect 1, and it was SILENT: no assert, just wrong tokens.

        ``MambaAttnBackendBase.init_cuda_graph_state`` derives ONE verify
        stride from ``max_num_tokens // max_bs``. That is the widest rung once
        a ladder exists, so the narrow rungs' captured GDN verify advanced the
        recurrent state over 4 rows where the batch had 2. Measured: with the
        ladder on, K=1 diverged from its own eager arm at index 1 while K=3 --
        whose stride happened to match -- stayed byte-green.

        The eager path has always read ``spec_info.draft_token_num``; the graph
        path now asks the same question.
        """
        import torch

        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            MambaAttnBackendBase,
        )

        b = MambaAttnBackendBase.__new__(MambaAttnBackendBase)
        b.device = torch.device("cpu")
        b._cuda_graph_max_bs = 1
        b.cached_cuda_graph_verify_query_start_loc = torch.tensor(
            [0, 4], dtype=torch.int32
        )
        b._verify_query_start_loc_by_rows = {
            4: b.cached_cuda_graph_verify_query_start_loc
        }
        self.assertEqual(list(b._verify_query_start_loc(4)), [0, 4])
        self.assertEqual(list(b._verify_query_start_loc(2)), [0, 2])
        self.assertEqual(list(b._verify_query_start_loc(3)), [0, 3])
        # Unknown / absent row count keeps the boot-time buffer, so a caller
        # that never had a draft_token_num behaves exactly as before.
        self.assertEqual(list(b._verify_query_start_loc(None)), [0, 4])
        # And the cache is a cache: the same rung is not rebuilt.
        self.assertIs(b._verify_query_start_loc(2), b._verify_query_start_loc(2))

    def test_the_verify_wrapper_key_separates_the_rungs(self):
        """Ladder defect 2, and it was LOUD but late.

        ``prefill_cuda_graph_metadata`` was keyed by bs alone, so every rung of
        the ladder (all bs=1) overwrote the previous rung's flashinfer
        wrappers. flashinfer latches ``_max_total_num_rows`` on a wrapper's
        first plan, so the surviving rung's capacity became everyone's:
        "the total number of rows in qo_indptr 3 ... cannot exceed ... 2".
        """
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        key = FlashInferAttnBackend._verify_cg_key
        b = object()
        si4 = type("SI", (), {"draft_token_num": 4})()
        si2 = type("SI", (), {"draft_token_num": 2})()
        tv = ForwardMode.TARGET_VERIFY
        self.assertNotEqual(key(b, 1, tv, si4), key(b, 1, tv, si2))
        self.assertEqual(key(b, 1, tv, si4), (1, 4))
        # Everything that is not a verify keeps the plain bs key, so no other
        # deployment's dict changes shape.
        self.assertEqual(key(b, 1, ForwardMode.DECODE, si4), 1)
        self.assertEqual(key(b, 3, tv, None), 3)

    def test_an_unset_ladder_is_the_pre_ladder_shape(self):
        """Opt-in, because every rung is another graph pool on the lane's card."""
        from types import SimpleNamespace

        from sglang.srt.model_executor.dual_group_lane import resolve_lane_spec_rungs

        args = SimpleNamespace(
            dual_group_lane_spec_rungs=None, dual_group_lane_spec_steps=3
        )
        self.assertEqual(resolve_lane_spec_rungs(args), (3,))


class TestLaneSpecPolicy(unittest.TestCase):
    """#274 round 7a: the adaptive-K policy, on the CPU.

    Everything it decides is arithmetic over numbers the lane hands it, so the
    hysteresis, the break-evens and the extrapolation can be argued about here
    rather than at the cost of a boot.
    """

    def _policy(self, **kw):
        from sglang.srt.model_executor.lane_spec_policy import LaneSpecPolicy

        kw.setdefault("rungs", (0, 1, 2, 3))
        kw.setdefault("adaptive", True)
        return LaneSpecPolicy(**kw)

    def test_the_rung_list_parses_and_rejects_junk(self):
        from sglang.srt.model_executor.lane_spec_policy import parse_lane_spec_rungs

        self.assertEqual(parse_lane_spec_rungs("0,1,2,3"), (0, 1, 2, 3))
        self.assertEqual(parse_lane_spec_rungs(" 3 , 1 ,3"), (1, 3))
        self.assertIsNone(parse_lane_spec_rungs(None))
        self.assertIsNone(parse_lane_spec_rungs(""))
        with self.assertRaises(ValueError):
            parse_lane_spec_rungs("1,x")
        with self.assertRaises(ValueError):
            parse_lane_spec_rungs("1,-2")

    def test_a_static_policy_answers_one_number(self):
        p = self._policy(adaptive=False, default_rung=3)
        for _ in range(10):
            self.assertEqual(p.choose(), 3)
            p.observe(3, 36.0, 2)

    def test_a_pin_beats_everything_and_does_not_become_the_resting_state(self):
        p = self._policy()
        self.assertEqual(p.choose({"rung": 1}), 1)
        self.assertEqual(p.current, 1)
        # Off-ladder pins are honoured (the lane falls back to the eager
        # verify) but must not park the policy on a rung it cannot replay.
        self.assertEqual(p.choose({"rung": 7}), 7)
        self.assertEqual(p.current, 1)

    def test_the_probe_phase_visits_every_rung_before_comparing(self):
        """Break-evens come from THIS boot, so every rung has to be measured.

        A policy that compared a measured rung against an unmeasured one would
        be comparing a measurement with a constant, which is the thing round 7a
        exists to stop doing.
        """
        p = self._policy(probe_rounds=2)
        seen = []
        for _ in range(8):
            k = p.choose()
            seen.append(k)
            p.observe(k, 16.0 + 6.0 * k, 1 + k // 2)
        self.assertEqual(set(seen[:8]) & {0, 1, 2, 3}, {0, 1, 2, 3})
        self.assertEqual(p.stats()["reason"] in ("probe", "hold", "switch"), True)

    def test_accept_is_read_per_position_so_saturation_is_visible(self):
        """PER-POSITION, because saturation is the thing that has to be seen.

        A head whose first proposal is usually right and whose third never is
        has the SAME mean accept length as one that degrades evenly, and only
        the per-position view tells them apart -- which is exactly the case the
        marginal criterion exists to catch (prose saturates after a position or
        two, so every further row is pure cost while the average still creeps
        up).
        """
        p = self._policy(probe_rounds=0, accept_ema=1.0)
        # A chain of 3 that accepts 2 and is rejected at the third: positions
        # 0 and 1 hit, position 2 evaluated and missed.
        for _ in range(5):
            p.observe(3, 36.0, 3)
        self.assertEqual(p.position_accept(0), 0.999)
        self.assertEqual(p.position_accept(1), 0.999)
        self.assertEqual(p.position_accept(2), 0.0)
        # The reach probability collapses at the saturated position; the mean
        # accept length does not show that at all.
        self.assertAlmostEqual(p.reach_probability(2), 0.998, places=3)
        self.assertAlmostEqual(p.reach_probability(3), 0.0, places=6)
        self.assertEqual(p.predicted_accept(0), 1.0)

    def test_the_criterion_is_marginal_not_average(self):
        """The rule, on the boot-5 numbers, and why the average is wrong.

        Costs 16.16 / 24.24 / 27.99 / 33.64 ms at K = 0 / 1 / 2 / 3. On
        `squares` the first proposal was accepted ~40 % of the time, and the
        AVERAGE ms/token ranks K=1 best. The MARGIN asks whether the first
        row's 8.1 ms buys more than 0.40 x 16.2 = 6.5 ms of decode step -- it
        does not, and the measured table agrees (K=0 16.19 vs K=1 17.27).
        """
        p = self._policy(probe_rounds=0, cost_ema=1.0, accept_ema=1.0)
        for k, ms in ((0, 16.16), (1, 24.24), (2, 27.99), (3, 33.64)):
            p.observe(k, ms, 1)
        # The K=0 step is NOT on the verify line: 8.1 ms against 3.7 and 5.7.
        self.assertAlmostEqual(p.marginal_cost(1), 8.08, places=2)
        self.assertAlmostEqual(p.marginal_cost(2), 3.75, places=2)
        self.assertAlmostEqual(p.t_decode, 16.16, places=3)
        p._pos_reached = {0: 1.0, 1: 1.0, 2: 1.0}
        p._pos_hits = {0: 0.40, 1: 0.0, 2: 0.0}
        self.assertEqual(p.marginal_depth(), 0)
        # Raise the FIRST position past its own margin and the chain grows by
        # exactly one: the second position still never lands, so depth stops
        # there. That is the whole difference from an average criterion, which
        # would keep rewarding a rung whose extra rows never pay.
        p._pos_hits = {0: 0.66, 1: 0.0, 2: 0.0}
        self.assertEqual(p.marginal_depth(), 1)
        # And a chain that does NOT saturate keeps growing on the same costs --
        # but only while the COMPOUND reach still pays: at a flat 0.66 the
        # third row is reached 0.29 of the time, worth 4.6 ms against its
        # 5.7 ms, so the margin stops at 2 without any saturation at all.
        p._pos_hits = {0: 0.66, 1: 0.66, 2: 0.66}
        self.assertEqual(p.marginal_depth(), 2)

    def test_the_flip_point_rounds_down_to_an_available_rung(self):
        """{0, 1, 3} with a flip at depth 2 answers 1, not 3.

        Rounding up would run two chain steps the margin already rejected.
        """
        from sglang.srt.model_executor.lane_spec_policy import LaneSpecPolicy

        p = LaneSpecPolicy((0, 1, 3), adaptive=True)
        self.assertEqual(p._rung_at_or_below(2, (0, 1, 3)), 1)
        self.assertEqual(p._rung_at_or_below(3, (0, 1, 3)), 3)
        self.assertEqual(p._rung_at_or_below(0, (0, 1, 3)), 0)
        # A ladder whose shortest rung is above the flip falls back to that
        # shortest rung rather than to nothing.
        self.assertEqual(p._rung_at_or_below(0, (1, 3)), 1)

    def test_the_cost_of_an_unvisited_rung_is_fitted_over_verify_rungs_only(self):
        """Affine in the row count, and K=0 is deliberately not on the line.

        K=0 is the plain DECODE graph, not a one-row verify; including it in
        the slope spreads the expensive first step over the whole ladder and
        makes every rung look equally priced.
        """
        p = self._policy(probe_rounds=0, cost_ema=1.0)
        p.observe(0, 16.0, 1)
        p.observe(1, 24.0, 1)
        p.observe(3, 36.0, 1)
        # rows 2 -> 24, rows 4 -> 36  =>  6 ms per chain step over the VERIFY
        # rungs, and the K=0 point does not drag the slope.
        self.assertAlmostEqual(p.marginal_cost(2), 6.0, places=6)
        self.assertAlmostEqual(p.predicted_round_ms(2), 30.0, places=6)

    def test_hysteresis_damps_a_flapping_challenger(self):
        p = self._policy(probe_rounds=0, hysteresis=3, cost_ema=1.0, accept_ema=1.0)
        p.current = 3
        for k, ms in ((0, 16.0), (1, 22.0), (2, 26.0), (3, 30.0)):
            p.observe(k, ms, 1)
        # First position accepted 80 % of the time, second never: the margin
        # flips at depth 1 and stays there.
        p._pos_reached = {0: 1.0, 1: 1.0}
        p._pos_hits = {0: 0.80, 1: 0.0}
        self.assertEqual(p.marginal_depth(), 1)
        self.assertEqual(p.choose(), 3, "switched on the first look")
        self.assertEqual(p.choose(), 3)
        self.assertEqual(p.choose(), 1, "never switched despite a stable winner")
        self.assertEqual(p.stats()["switches"], 1)

    def test_a_gain_inside_the_margin_does_not_extend_the_chain(self):
        """A row whose gain only ties its cost is not worth the transition."""
        p = self._policy(
            rungs=(0, 1),
            probe_rounds=0,
            hysteresis=1,
            cost_ema=1.0,
            accept_ema=1.0,
            margin=0.2,
        )
        p.observe(0, 16.0, 1)
        p.observe(1, 24.0, 1)
        p._pos_reached = {0: 1.0}
        # 8.0 ms cost; 0.52 x 16 = 8.32 ms gain -- better, but inside the 20 %
        # margin, so the chain stays at 0.
        p._pos_hits = {0: 0.52}
        self.assertEqual(p.marginal_depth(), 0)
        p._pos_hits = {0: 0.70}
        self.assertEqual(p.marginal_depth(), 1)

    def test_the_k_zero_rung_is_cost_evidence_only(self):
        """A plain decode step is not a rejected proposal.

        Folding K=0 rounds into the acceptance counters as failures would drag
        the estimate down for a reason that has nothing to do with the head.
        """
        p = self._policy(probe_rounds=0, accept_ema=1.0)
        p.observe(1, 22.0, 2)
        before = (dict(p._pos_hits), dict(p._pos_reached))
        for _ in range(5):
            p.observe(0, 16.0, 1)
        self.assertEqual((dict(p._pos_hits), dict(p._pos_reached)), before)
        self.assertEqual(p.stats()["round_n"][0], 5)

    def test_a_falsifier_round_is_not_priced_into_the_policy(self):
        """The byte gates must not teach the policy what a round costs.

        The gates run the same rungs with the graphs switched OFF, and an eager
        verify costs 68 ms against the captured 21 ms. Those rounds landed in
        the per-rung cost EMA and the policy then believed K=1 cost 77 ms per
        round -- measured, round 7a boot 6: ``round_ms {0: 16.1, 1: 77.1,
        2: 75.2, 3: 52.7}`` against measured graph costs of 24 / 28 / 34, so it
        pinned itself to K=0 for a reason that had nothing to do with content.
        """
        import inspect

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        src = inspect.getsource(DualGroupLane._spec_step)
        self.assertIn('job.get("verify_graph") is False', src)
        self.assertIn('job.get("head_graph") is False', src)
        # And the observe call is on the other side of that branch.
        skip = src.index('job.get("verify_graph") is False')
        obs = src.index("self.spec_policy.observe")
        self.assertLess(skip, obs, "the falsifier check comes after observe()")

    def test_the_policy_takes_a_context_for_round_7b(self):
        """The turn-routing seam, pinned so the next round does not move it.

        Round 7b routes by TURN (first request one algorithm, multiturn
        another), and an algorithm need not offer every rung. That restriction
        arrives as a ctx, not as a second decision site next to this one.
        """
        p = self._policy(probe_rounds=0)
        self.assertEqual(p.candidate_rungs({"rungs": [0, 1]}), (0, 1))
        self.assertEqual(p.candidate_rungs(None), (0, 1, 2, 3))
        # An empty intersection falls back to the full ladder rather than
        # returning nothing to choose from.
        self.assertEqual(p.candidate_rungs({"rungs": [9]}), (0, 1, 2, 3))


class TestLaneDraftRollback(unittest.TestCase):
    """#274 round 7b posten 0: the head's sequence follows the target's.

    ``_propose`` advances the head by K per round and the verify commits
    ``n_accept + 1``. Nothing put the difference back, so from the second
    round on the head answered about a position the sequence was not at, over
    a KV cache still holding every rejected proposal -- measured on the rig as
    a lag of 179-224 positions over a 192-token job. These are the arithmetic
    of the fix, on the CPU.
    """

    def _lane_and_batch(self, start=10, steps=3):
        import torch

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        class _Fake(DualGroupLane):
            def __init__(self):
                self.spec_steps = steps

        lane = _Fake()
        batch_d = _StubVerifyBatch(start + steps)
        req = _StubReq()
        req.decode_batch_idx = steps
        req.req_pool_idx = 0
        batch_d.reqs = [req]
        batch_d.req_to_token_pool.req_to_token[0, : start + steps] = torch.arange(
            500, 500 + start + steps, dtype=torch.int32
        )
        job = {
            "_batch_d": batch_d,
            "_round_start": start,
            "_rung": steps,
            "_kv_len_draft": start + steps,
        }
        return lane, batch_d, job, req

    def test_a_rejected_chain_gives_its_slots_and_its_positions_back(self):
        lane, batch_d, job, req = self._lane_and_batch(start=10, steps=3)
        # Nothing accepted: the round commits 1 token, the head ran 3.
        lane._rollback_draft(job, n_accept=0)
        self.assertEqual(int(batch_d.seq_lens[0]), 11)
        self.assertEqual(int(batch_d.seq_lens_cpu[0]), 11)
        self.assertEqual(job["_kv_len_draft"], 11)
        self.assertEqual(req.kv_committed_len, 11)
        # Exactly the two rejected proposals' slots came back.
        self.assertEqual(batch_d.freed, [[511, 512]])

    def test_a_partly_accepted_chain_keeps_the_accepted_positions(self):
        lane, batch_d, job, req = self._lane_and_batch(start=10, steps=3)
        lane._rollback_draft(job, n_accept=1)
        self.assertEqual(int(batch_d.seq_lens[0]), 12)
        self.assertEqual(batch_d.freed, [[512]])

    def test_a_fully_accepted_chain_needs_no_truncation(self):
        """kept == K: the head is exactly where the target is, nothing to do."""
        lane, batch_d, job, req = self._lane_and_batch(start=10, steps=3)
        lane._rollback_draft(job, n_accept=2)
        self.assertEqual(int(batch_d.seq_lens[0]), 13)
        self.assertEqual(batch_d.freed, [])

    def test_the_per_job_falsifier_leaves_the_defect_in_place(self):
        """Both arms have to come from ONE boot, or the gate carries variance."""
        lane, batch_d, job, req = self._lane_and_batch(start=10, steps=3)
        job["draft_rollback"] = False
        lane._rollback_draft(job, n_accept=0)
        self.assertEqual(int(batch_d.seq_lens[0]), 13)
        self.assertEqual(batch_d.freed, [])

    def test_full_accept_runs_the_one_missing_head_forward(self):
        """kept == K + 1: the bonus token has no head position yet.

        The head ran K forwards, the round committed K + 1 tokens, so exactly
        one position is missing and one head forward fills it -- against the
        TARGET's hidden state of the row before it, the same pairing every
        other chain step uses.
        """
        import torch

        lane, batch_d, job, req = self._lane_and_batch(start=10, steps=3)
        seen = {}

        def _fwd(b, hidden):
            seen["hidden"] = hidden
            seen["input_ids"] = b.input_ids

        lane._draft_forward = _fwd
        batch_d.prepare_for_decode = lambda: batch_d.seq_lens.add_(1)
        job["_verify_last_token"] = torch.tensor([77], dtype=torch.int64)
        job["_verify_hidden"] = torch.zeros((1, 4))
        job["_kv_len_draft"] = 13
        lane._rollback_draft(job, n_accept=3)
        self.assertEqual(int(seen["input_ids"][0]), 77)
        self.assertEqual(job["_kv_len_draft"], 14)
        self.assertEqual(int(batch_d.seq_lens[0]), 14)
        self.assertEqual(batch_d.freed, [])

    def test_full_accept_without_a_stashed_row_does_not_guess(self):
        """The seqdecode bridge produces no row block, so it stashes nothing."""
        lane, batch_d, job, req = self._lane_and_batch(start=10, steps=3)
        lane._draft_forward = lambda *a, **k: self.fail("must not forward")
        lane._rollback_draft(job, n_accept=3)
        self.assertEqual(int(batch_d.seq_lens[0]), 13)


class TestLaneDraftReseed(unittest.TestCase):
    """#274 round 7c posten 2: the accepted positions carry TARGET hidden.

    After round 7b's rollback fix, one structural difference between the lane's
    chain and the serving group's was left: ``_propose`` writes the head's KV
    for the accepted positions with the HEAD's own hidden state, because that
    is all it has while it speculates, while
    ``EagleWorkerV2._draft_extend_for_decode`` re-runs the head over that same
    block against the TARGET's hidden. These are the arithmetic and the
    pairing of closing it, on the CPU.
    """

    def _lane_and_batch(self, start=10, steps=3, rows=3):
        import torch

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        class _Fake(DualGroupLane):
            def __init__(self):
                self.spec_steps = steps
                self.draft_runner = None

        lane = _Fake()
        batch_d = _StubVerifyBatch(start + steps)
        req = _StubReq()
        req.decode_batch_idx = steps
        req.req_pool_idx = 0
        batch_d.reqs = [req]
        batch_d.req_to_token_pool.req_to_token[0, : start + steps] = torch.arange(
            500, 500 + start + steps, dtype=torch.int32
        )
        batch_d.prepare_for_decode = lambda: batch_d.seq_lens.add_(1)
        job = {
            "_batch_d": batch_d,
            "_round_start": start,
            "_rung": steps,
            "_kv_len_draft": start + steps,
            # cand = [committed token] + proposals; the verify's row block is
            # one shorter, since row j predicts candidate j + 1.
            "_verify_tokens": torch.tensor([40, 41, 42, 43], dtype=torch.int64),
            "_verify_rows": torch.arange(rows * 4, dtype=torch.float32).reshape(
                rows, 4
            ),
        }
        seen = []
        lane._draft_forward = lambda b, hidden: seen.append(
            (int(b.input_ids[0]), hidden.clone())
        )
        return lane, batch_d, job, req, seen

    def test_nothing_accepted_re_seeds_nothing(self):
        """Position ``start`` was ALREADY written against the target's hidden.

        Its input pair is the previous round's committed token and the
        previous round's target row -- the lane has always had that one right,
        so a round that accepts no proposal has nothing to re-seed and must not
        pay a forward for it.
        """
        lane, batch_d, job, req, seen = self._lane_and_batch()
        lane._rollback_draft(job, n_accept=0)
        self.assertEqual(seen, [])
        self.assertEqual(int(batch_d.seq_lens[0]), 11)
        self.assertEqual(batch_d.freed, [[511, 512]])
        self.assertEqual(job["_kv_len_draft"], 11)

    def test_one_accepted_position_is_re_run_against_the_target_row(self):
        lane, batch_d, job, req, seen = self._lane_and_batch()
        lane._rollback_draft(job, n_accept=1)
        # The head speculated 3 positions; all of them past ``start`` go back,
        # then exactly the accepted one is re-run.
        self.assertEqual(batch_d.freed, [[511, 512]])
        self.assertEqual(len(seen), 1)
        token, hidden = seen[0]
        # Candidate 1 is proposal 0, the one the verify accepted...
        self.assertEqual(token, 41)
        # ...paired with the target row that PREDICTED it, i.e. row 0.
        self.assertTrue(bool((hidden[0] == job["_verify_rows"][0]).all()))
        # And the head ends exactly where the target does.
        self.assertEqual(int(batch_d.seq_lens[0]), 12)
        self.assertEqual(job["_kv_len_draft"], 12)
        self.assertEqual(job["_reseed_forwards"], 1)

    def test_the_pairing_is_the_same_one_the_chain_uses_everywhere(self):
        """Token ``cand[j]`` against target row ``j - 1``, for every j."""
        lane, batch_d, job, req, seen = self._lane_and_batch()
        lane._rollback_draft(job, n_accept=2)
        self.assertEqual([t for t, _ in seen], [41, 42])
        for j, (_, hidden) in enumerate(seen, start=1):
            self.assertTrue(bool((hidden[0] == job["_verify_rows"][j - 1]).all()))
        self.assertEqual(int(batch_d.seq_lens[0]), 13)

    def test_full_accept_needs_no_separate_catch_up_any_more(self):
        """kept == K + 1 is the same rule, not a special case.

        Round 7b filled the bonus token's missing head position with a
        dedicated branch. The re-seed subsumes it: it re-runs positions
        ``1 .. kept-1``, and at full accept that is exactly K positions, the
        last of which IS the bonus token's.
        """
        lane, batch_d, job, req, seen = self._lane_and_batch()
        lane._rollback_draft(job, n_accept=3)
        self.assertEqual([t for t, _ in seen], [41, 42, 43])
        self.assertEqual(int(batch_d.seq_lens[0]), 14)
        self.assertEqual(job["_kv_len_draft"], 14)

    def test_the_per_job_falsifier_leaves_the_heads_own_hidden_in_place(self):
        """Both arms out of ONE boot, on a content-driven quantity."""
        lane, batch_d, job, req, seen = self._lane_and_batch()
        job["draft_reseed"] = False
        lane._rollback_draft(job, n_accept=1)
        self.assertEqual(seen, [])
        self.assertEqual(int(batch_d.seq_lens[0]), 12)
        self.assertEqual(batch_d.freed, [[512]])

    def test_a_bridge_that_stashes_no_rows_keeps_the_old_behaviour(self):
        """``_verify_by_decode`` produces no candidate row block.

        It must not be made to guess one: no rows means no re-seed and the
        round-7b truncation stands, which is also what keeps the bridge a
        usable fallback rather than a second, subtly different chain.
        """
        lane, batch_d, job, req, seen = self._lane_and_batch()
        job["_verify_rows"] = None
        job["_verify_tokens"] = None
        lane._rollback_draft(job, n_accept=1)
        self.assertEqual(seen, [])
        self.assertEqual(int(batch_d.seq_lens[0]), 12)

    def test_the_rollback_falsifier_still_disables_everything(self):
        lane, batch_d, job, req, seen = self._lane_and_batch()
        job["draft_rollback"] = False
        lane._rollback_draft(job, n_accept=1)
        self.assertEqual(seen, [])
        self.assertEqual(int(batch_d.seq_lens[0]), 13)
        self.assertEqual(batch_d.freed, [])


class TestAcceptPositionProbe(unittest.TestCase):
    """#274 round 7b posten 0: the serving group's per-position counter.

    The whole point of the probe is that it produces the SAME quantity the
    lane's policy reports, so the two curves can be laid next to each other.
    That equivalence is asserted here rather than assumed.
    """

    def setUp(self):
        from sglang.srt.speculative import accept_position_probe

        accept_position_probe.reset()

    def test_a_greedy_chain_only_evaluates_up_to_the_first_rejection(self):
        from sglang.srt.speculative import accept_position_probe as probe

        # accept_len 1 == bonus token only, i.e. proposal 0 rejected.
        probe.record_accept_lens([1, 1, 2, 4], num_proposals=3)
        snap = probe.snapshot()
        # Position 0 was evaluated by all four rounds, 2 of them accepted it.
        self.assertEqual(snap["position_reached"][0], 4)
        self.assertEqual(snap["position_hits"][0], 2)
        # Position 1 only by the two rounds that got past position 0.
        self.assertEqual(snap["position_reached"][1], 2)
        # Position 2 only by the accept_len 4 round, which accepted it.
        self.assertEqual(snap["position_reached"][2], 1)
        self.assertEqual(snap["position_hits"][2], 1)
        self.assertEqual(snap["accept_len_mean"], 2.0)

    def test_it_agrees_with_the_lane_policys_definition(self):
        """Same rounds, same evaluated/accepted bookkeeping in both counters.

        The lane EMAs (it decides from the number, so old content must stop
        voting) and the probe counts raw, so the two cannot be compared as
        rates. What CAN be compared -- and what makes the two curves the same
        quantity -- is the per-round rule: which positions a greedy chain
        evaluates, and which of those it accepts. Driven one round at a time
        with the EMA weight at 1.0, the policy's rate IS that round's hit flag.
        """
        from sglang.srt.model_executor.lane_spec_policy import LaneSpecPolicy
        from sglang.srt.speculative import accept_position_probe as probe

        for accept_len in (1, 2, 3, 4):
            probe.reset()
            probe.record_accept_lens([accept_len], num_proposals=3)
            pol = LaneSpecPolicy((3,), adaptive=False, accept_ema=1.0)
            pol.observe(3, 20.0, accept_len)
            snap = probe.snapshot()
            for j in range(3):
                evaluated = snap["position_reached"].get(j, 0) == 1
                self.assertEqual(
                    evaluated,
                    pol.position_accept(j) is not None,
                    f"position {j} evaluated disagrees at accept_len {accept_len}",
                )
                if evaluated:
                    # The policy clamps a rate of 1 to 0.999 on purpose (an
                    # unbounded chain must never look free to the margin), so
                    # the comparison is of the DECISION, not of the last digit.
                    self.assertAlmostEqual(
                        snap["position_hits"].get(j, 0),
                        pol.position_accept(j),
                        places=2,
                        msg=f"position {j} hit disagrees at accept_len {accept_len}",
                    )

    def test_the_probe_is_off_unless_asked_for(self):
        import os

        from sglang.srt.speculative import accept_position_probe as probe

        saved = os.environ.pop("SGLANG_ACCEPT_POSITION_PROBE", None)
        try:
            self.assertFalse(probe.probe_enabled())
            os.environ["SGLANG_ACCEPT_POSITION_PROBE"] = "1"
            self.assertTrue(probe.probe_enabled())
        finally:
            os.environ.pop("SGLANG_ACCEPT_POSITION_PROBE", None)
            if saved is not None:
                os.environ["SGLANG_ACCEPT_POSITION_PROBE"] = saved


class TestLaneDrafterQuant(unittest.TestCase):
    """#274 round 7c, nachtrag 13g: the drafter's precision is a CHOICE.

    The rule under test is "similar to, or slightly better than, the target":
    a drafter coarser than its target proposes tokens the target rejects, and
    one far finer buys accept the target's own coarseness already discarded --
    while costing the KV a coarse target needs most. So the DEFAULT is a band
    above the target and the recommendation is the highest step of that band
    that fits the card. The rest of the ladder stays reachable; what is banded
    is the default, never the choice.
    """

    # The rig's DFLASH drafter, read off the checkpoint header rather than
    # estimated -- the NEXTN lesson (2684 MiB where 120 was assumed).
    PARAMS = 1_730_213_120
    DENSE = 62_720  # 1-D norms; no format quantises them

    def test_the_band_for_a_q3_target_is_q4_to_q6(self):
        from sglang.srt.model_executor.lane_spec_policy import drafter_quant_band

        self.assertEqual(drafter_quant_band("Q3_K_M"), ("q4_k_m", "q5_k_m", "q6_k"))
        # Spelling is the caller's, not the ladder's: this is read off a
        # checkpoint config or a CLI flag.
        self.assertEqual(drafter_quant_band("q3_k_m"), drafter_quant_band("Q3_K_M"))

    def test_the_band_follows_the_target_up_the_ladder(self):
        from sglang.srt.model_executor.lane_spec_policy import drafter_quant_band

        self.assertEqual(drafter_quant_band("fp8_e4m3"), ("bf16",))
        self.assertEqual(drafter_quant_band("bf16"), ())

    def test_an_unknown_target_gets_no_band_rather_than_a_guess(self):
        """A recommendation with no evidence under it is worse than none."""
        from sglang.srt.model_executor.lane_spec_policy import (
            choose_drafter_quant,
            drafter_quant_band,
        )

        self.assertEqual(drafter_quant_band(None), ())
        quant, why = choose_drafter_quant(None, 8000.0, self.PARAMS, self.DENSE)
        self.assertIsNone(quant)
        self.assertIn("no default band", why)

    def test_the_footprint_matches_the_checkpoint_at_bf16(self):
        """The ladder's top step must reproduce the measured 3300 MiB."""
        from sglang.srt.model_executor.lane_spec_policy import drafter_weight_mib

        self.assertAlmostEqual(
            drafter_weight_mib(self.PARAMS, "bf16", self.DENSE), 3300.1, places=1
        )

    def test_a_roomy_card_gets_the_top_of_the_band_not_the_top_of_the_ladder(self):
        """Q6, not Q8 and not BF16: the band is what the default respects."""
        from sglang.srt.model_executor.lane_spec_policy import choose_drafter_quant

        quant, why = choose_drafter_quant(
            "Q3_K_M", budget_mib=2600.0, params=self.PARAMS, dense_params=self.DENSE
        )
        self.assertEqual(quant, "q6_k")
        self.assertIn("q6_k", why)

    def test_a_tight_card_steps_down_inside_the_band(self):
        from sglang.srt.model_executor.lane_spec_policy import choose_drafter_quant

        # Room for Q4 and Q5 but not Q6 (1354 MiB).
        quant, _ = choose_drafter_quant(
            "Q3_K_M", budget_mib=1200.0, params=self.PARAMS, dense_params=self.DENSE
        )
        self.assertEqual(quant, "q5_k_m")
        # Room for Q4 only (1000 MiB).
        quant, _ = choose_drafter_quant(
            "Q3_K_M", budget_mib=1050.0, params=self.PARAMS, dense_params=self.DENSE
        )
        self.assertEqual(quant, "q4_k_m")

    def test_overhead_is_subtracted_before_the_band_is_walked(self):
        """KV, graphs and dequant scratch are not weights, and they are real."""
        from sglang.srt.model_executor.lane_spec_policy import choose_drafter_quant

        roomy = dict(target_quant="Q3_K_M", params=self.PARAMS, dense_params=self.DENSE)
        # 1400 MiB holds Q6 (1354). Take 100 MiB of graphs off the top and it
        # does not any more, so the band steps down to Q5 (1134).
        self.assertEqual(choose_drafter_quant(budget_mib=1400.0, **roomy)[0], "q6_k")
        self.assertEqual(
            choose_drafter_quant(budget_mib=1400.0, overhead_mib=100.0, **roomy)[0],
            "q5_k_m",
        )

    def test_a_card_that_cannot_hold_the_band_says_so_instead_of_dropping_below(self):
        """Going under the band is a decision, not a sizing fallback.

        Reaching for Q3 or Q2 to make something fit means accepting a drafter
        coarser than its own target -- that is a policy judgement about
        expected accept, and a helper that made it silently would hide exactly
        the trade the operator asked to control.
        """
        from sglang.srt.model_executor.lane_spec_policy import choose_drafter_quant

        quant, why = choose_drafter_quant(
            "Q3_K_M", budget_mib=400.0, params=self.PARAMS, dense_params=self.DENSE
        )
        self.assertIsNone(quant)
        self.assertIn("q4_k_m", why)
        self.assertIn("400", why)

    def test_the_policy_carries_the_choice_and_reports_it(self):
        from sglang.srt.model_executor.lane_spec_policy import (
            LANE_DRAFTER_DFLASH,
            LANE_DRAFTER_NEXTN,
            LaneDrafterPolicy,
        )

        pol = LaneDrafterPolicy(
            available=(LANE_DRAFTER_NEXTN, LANE_DRAFTER_DFLASH),
            drafter_quant={LANE_DRAFTER_DFLASH: "Q6_K"},
            # An algorithm change is a plan flip and needs the full window;
            # this test is about the precision field, not about hysteresis.
            hysteresis=1,
        )
        d = pol.decide({"content": "code"})
        self.assertEqual(d.algorithm, LANE_DRAFTER_DFLASH)
        self.assertEqual(d.quant, "Q6_K")
        self.assertEqual(d.as_dict()["quant"], "Q6_K")
        # An unset drafter reports None -- "whatever the checkpoint is", never
        # a claim about a precision nobody chose.
        self.assertIsNone(pol.decide({"content": "prose"}).quant)
        self.assertEqual(pol.stats()["drafter_quant"], {LANE_DRAFTER_DFLASH: "Q6_K"})

    def test_the_policy_rejects_a_quant_for_a_drafter_it_does_not_have(self):
        from sglang.srt.model_executor.lane_spec_policy import (
            LANE_DRAFTER_DFLASH,
            LANE_DRAFTER_NEXTN,
            LaneDrafterPolicy,
        )

        with self.assertRaises(ValueError):
            LaneDrafterPolicy(
                available=(LANE_DRAFTER_NEXTN,),
                drafter_quant={LANE_DRAFTER_DFLASH: "q6_k"},
            )
        with self.assertRaises(ValueError):
            LaneDrafterPolicy(
                available=(LANE_DRAFTER_NEXTN,),
                drafter_quant={LANE_DRAFTER_NEXTN: "q9_ultra"},
            )


class TestLaneDrafterPolicy(unittest.TestCase):
    """#274 round 7b posten 2: deterministic turn routing (nachtrag 13c-13e).

    One object, three actuators, no learning. Every rule below is a sentence
    from the nachtrag turned into a comparison, which is why it can be argued
    about without a boot.
    """

    def _policy(self, **kw):
        from sglang.srt.model_executor.lane_spec_policy import LaneDrafterPolicy

        kw.setdefault("available", ("nextn", "dflash"))
        kw.setdefault("ctx_gate_tokens", 8192)
        kw.setdefault("hysteresis", 1)
        return LaneDrafterPolicy(**kw)

    def test_first_request_in_a_short_context_prefers_dflash(self):
        p = self._policy()
        d = p.decide({"turn_index": 0, "context_len": 512})
        self.assertEqual(d.preferred, "dflash")
        self.assertEqual(d.algorithm, "dflash")

    def test_multiturn_goes_to_nextn(self):
        p = self._policy()
        d = p.decide({"turn_index": 3, "context_len": 512})
        self.assertEqual(d.preferred, "nextn")
        self.assertIn("first request", d.reason)

    def test_code_content_keeps_dflash_past_the_first_turn(self):
        p = self._policy()
        d = p.decide({"turn_index": 7, "context_len": 512, "content": "code"})
        self.assertEqual(d.preferred, "dflash")

    def test_prose_is_a_hard_veto_even_on_the_first_request(self):
        """13d correction: DFLASH is measured very poor on prose."""
        p = self._policy()
        d = p.decide({"turn_index": 0, "context_len": 128, "content": "prose"})
        self.assertEqual(d.preferred, "nextn")
        self.assertEqual(d.reason, "prose content")

    def test_the_context_bound_is_read_not_written_down(self):
        """The gate is whatever the caller derived from the drafter config."""
        p = self._policy(ctx_gate_tokens=4096)
        self.assertEqual(
            p.decide({"turn_index": 0, "context_len": 4095}).preferred, "dflash"
        )
        self.assertEqual(
            p.decide({"turn_index": 0, "context_len": 4096}).preferred, "nextn"
        )
        # A drafter that declares nothing usable lifts the gate entirely.
        q = self._policy(ctx_gate_tokens=None)
        self.assertEqual(
            q.decide({"turn_index": 0, "context_len": 10**6}).preferred, "dflash"
        )

    def test_a_load_peak_takes_the_cheaper_drafter(self):
        p = self._policy(load_threshold=0.8)
        self.assertEqual(
            p.decide({"turn_index": 0, "context_len": 128, "load": 0.5}).preferred,
            "dflash",
        )
        d = p.decide({"turn_index": 0, "context_len": 128, "load": 0.9})
        self.assertEqual(d.preferred, "nextn")
        self.assertIn("load>=", d.reason)

    def test_a_protected_request_under_load_says_so(self):
        """Nachtrag 5: the protected class never pays for another's drafter."""
        p = self._policy(load_threshold=0.8)
        d = p.decide(
            {"turn_index": 0, "context_len": 128, "load": 0.95, "protected": True}
        )
        self.assertIn("protected class", d.reason)

    def test_fixed_load_policy_never_moves(self):
        p = self._policy(load_policy="fixed", default_algorithm="nextn")
        d = p.decide({"turn_index": 0, "context_len": 128, "load": 0.99})
        self.assertEqual(d.preferred, "nextn")
        self.assertEqual(d.reason, "load-policy fixed")

    def test_an_unknown_load_policy_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            self._policy(load_policy="sometimes")

    def test_the_accept_guard_is_the_net(self):
        p = self._policy(accept_guard_rounds=4, accept_guard_floor=0.25)
        for _ in range(4):
            p.observe("dflash", position0_accepted=False)
        d = p.decide({"turn_index": 0, "context_len": 128})
        self.assertEqual(d.preferred, "nextn")
        self.assertIn("accept guard", d.reason)

    def test_the_guard_does_not_fire_on_a_healthy_drafter(self):
        p = self._policy(accept_guard_rounds=4, accept_guard_floor=0.25)
        for i in range(8):
            p.observe("dflash", position0_accepted=i % 4 != 0)
        self.assertFalse(p.guarded("dflash"))
        self.assertEqual(
            p.decide({"turn_index": 0, "context_len": 128}).preferred, "dflash"
        )

    def test_hysteresis_needs_a_whole_window_before_it_flips(self):
        p = self._policy(hysteresis=3, default_algorithm="nextn")
        short = {"turn_index": 0, "context_len": 128}
        self.assertEqual(p.decide(short).algorithm, "nextn")
        self.assertEqual(p.decide(short).algorithm, "nextn")
        self.assertEqual(p.decide(short).algorithm, "dflash")
        self.assertEqual(p.stats()["switches"], 1)

    def test_a_preference_for_a_lane_that_does_not_exist_is_recorded_not_taken(self):
        """Today's state: NEXTN only, and the policy still says what it wanted.

        This is what makes the routing measurable before R7c builds the second
        lane -- and what stops "not built" looking like "not preferred".
        """
        p = self._policy(available=("nextn",))
        d = p.decide({"turn_index": 0, "context_len": 128})
        self.assertEqual(d.preferred, "dflash")
        self.assertEqual(d.algorithm, "nextn")
        self.assertIn("lane not built", d.reason)

    def test_topk_is_carried_and_not_wired(self):
        """#141 lives in the same object; nothing reads the field yet."""
        p = self._policy()
        self.assertEqual(p.decide({"turn_index": 0, "context_len": 1}).topk, 1)
        self.assertEqual(
            p.decide({"turn_index": 0, "context_len": 1, "topk": 4}).topk, 4
        )

    def test_stats_report_the_decision_mix(self):
        p = self._policy()
        p.decide({"turn_index": 0, "context_len": 128})
        p.decide({"turn_index": 2, "context_len": 128})
        st = p.stats()
        self.assertEqual(st["decisions"], 2)
        self.assertEqual(st["ctx_gate_tokens"], 8192)
        self.assertEqual(sum(st["reasons"].values()), 2)


if __name__ == "__main__":
    unittest.main()
