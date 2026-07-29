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

        target, draft = split_lane_budget(self._args(1600), self._cfg(64), self._cfg(1))
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
        target, draft = split_lane_budget(self._args(1600), self._cfg(2), self._cfg(1))
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
                "_job_spec_on", src, f"{fn.__name__} does not consult _job_spec_on"
            )
            self.assertIn(
                "_spec_step", src, f"{fn.__name__} never runs a speculative round"
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
        # dominant (35.9 vs 75-96 ms per round). The default still reads
        # seqdecode HERE because promoting it is a merge decision, not a thing
        # a measurement branch does to itself -- the recommendation and its
        # gate evidence live in _verify_mode's docstring. When that merge
        # happens this assertion flips with it, on purpose: the pair is the
        # record that the change was chosen rather than drifted into.
        self.assertEqual(self._lane()._verify_mode({}), "seqdecode")
        doc = inspect.getdoc(DualGroupLane._verify_mode) or ""
        self.assertIn("Recommendation on record", doc)

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

    def _runner(self, *, verify_tokens=4, captured=True, max_bs=1, ntpb=1):
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
        r._lane_verify_tokens = verify_tokens
        r._lane_verify_active = False
        r._lane_verify_captured = captured
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
        self.assertFalse(r._lane_verify_active)
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
        self.assertFalse(r._lane_verify_active)
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
            self.assertEqual(r._wl_variant_label(None), r.LANE_VERIFY_VARIANT)
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
        r = self._runner(captured=False)
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
            "R", (), {"decode_cuda_graph_runner": self._runner(verify_tokens=4)}
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
        self.assertIn("_register_gguf_decode_buckets([self._lane_verify_tokens]", src)

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
            mr.dual_group_lane_verify_tokens = 4
            self.assertEqual(ModelRunner.max_decode_logits_rows(mr), 4)
            # Nothing else grows: without the lane field this is the row count
            # it always was, so the serving group's shared buffer is untouched.
            mr.dual_group_lane_verify_tokens = None
            self.assertEqual(ModelRunner.max_decode_logits_rows(mr), 1)


if __name__ == "__main__":
    unittest.main()
