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

    def _cfg(self, layers):
        class _C:
            num_hidden_layers = layers

        return _C()

    def test_the_split_conserves_the_operators_budget(self):
        from sglang.srt.model_executor.dual_group_lane import split_lane_budget

        target, draft = split_lane_budget(
            self._args(1600), self._cfg(64), self._cfg(1)
        )
        self.assertEqual(target + draft, 1600)

    def test_the_head_gets_its_layer_share_not_a_second_budget(self):
        from sglang.srt.model_executor.dual_group_lane import split_lane_budget

        _, draft = split_lane_budget(self._args(1600), self._cfg(64), self._cfg(1))
        # One layer of 64 -> 25 MiB by ratio, lifted to the 64 MiB floor.
        self.assertEqual(draft, 64)

    def test_a_floor_keeps_the_head_pool_viable(self):
        from sglang.srt.model_executor.dual_group_lane import split_lane_budget

        _, draft = split_lane_budget(self._args(1600), self._cfg(4096), self._cfg(1))
        self.assertGreaterEqual(draft, 64)

    def test_the_head_can_never_take_more_than_a_quarter(self):
        from sglang.srt.model_executor.dual_group_lane import split_lane_budget

        # A pathological ratio (half the layers) must not starve the target.
        target, draft = split_lane_budget(
            self._args(1600), self._cfg(2), self._cfg(1)
        )
        self.assertLessEqual(draft, 400)
        self.assertGreater(target, draft)


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
                "spec_active", src, f"{fn.__name__} does not consult spec_active"
            )
            self.assertIn(
                "_spec_step", src, f"{fn.__name__} never runs a speculative round"
            )
            self.assertIn(
                "_decode_step", src, f"{fn.__name__} lost the plain decode arm"
            )


if __name__ == "__main__":
    unittest.main()
