# SPDX-License-Identifier: Apache-2.0
"""#631 phase-flip TP-stack boot: hermetic contract tests (CPU-only).

The load-bearing gates, mapped to DESIGN_631 sections 3.3/3.6a/5.2:

* OPERATOR PIN 3 (row byte-compatibility is TESTED, not asserted): both
  layouts' full-attention KV row schemas are derived from the real
  Qwen3.6-27B config constants THROUGH the real functions
  (ModelConfig.get_num_kv_heads / get_total_num_kv_heads and the
  uneven_dcp_kv_replicated predicate) and must be byte-equal; the can-fail
  arm proves the checker fails red the moment the weighted-DCP head
  replication rule stops applying (head-sharded TP pool);
* OPERATOR PIN 2, STRUCTURAL: only the TP stack may capture decode CUDA
  graphs -- the PP stack's capture entry points REFUSE (poison gates), and
  the is_phase_flip_pp_stack predicate is exactly scoped (default boots
  and the TP stack itself are never caught);
* group routing: with phase-flip TP routing active the module-level group
  getters return the flip set (forward-time collectives reach groups
  through these getters, not the contextvar -- routing to the primary
  tp=1 group would be a silent no-op all-reduce); activating the route
  without built flip groups is REFUSED, not no-oped;
* server-args derivation (the #470 REACH discipline): the TP stack's args
  copy differs from the boot args in EXACTLY the declared field set;
* snapshot/free/bind/refill: the boot-order weights choreography restores
  bit-exact values and strides, and PhaseFlipStacks.refill dispatches the
  correct image per direction.
"""

import dataclasses
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.utils import (
    set_cp_token_ratios,
    set_tp_partition_ratios,
    uneven_dcp_kv_replicated,
)
from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP
from sglang.srt.managers.phase_flip_boot import (
    TP_STACK_OVERRIDDEN_FIELDS,
    PhaseFlipBootError,
    PhaseFlipStacks,
    assert_row_schema_compatible,
    checkpoint_param_dict,
    derive_tp_stack_server_args,
    snapshot_and_free,
)
from sglang.srt.model_executor.weights_arena import (
    allocate_arena,
    arena_image,
    arena_refill,
    bind_arena_views,
    image_from_tensors,
    pack_into_arena,
    plan_arena_layout,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


# Real Qwen3.6-27B constants (models-cache config.json, text_config;
# also DESIGN_631 section 0/2: 64 layers, 16 full-attention, measured
# K = V = exactly 1 KiB/token/layer at fp8).
QWEN36_27B = dict(
    model_type="qwen3_5_text",
    architectures=["Qwen3_5ForCausalLM"],
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    hidden_size=5120,
    num_hidden_layers=64,
)
FLIP_VECTOR = [30, 17, 17]
KV_DTYPE = torch.float8_e4m3fn  # kv-cache-dtype fp8_e4m3 in the recipe


def _model_config():
    cfg = ModelConfig.__new__(ModelConfig)
    hf = SimpleNamespace(**QWEN36_27B)
    cfg.hf_config = hf
    cfg.hf_text_config = hf
    return cfg


def _pool_stub(head_num, head_dim, dtype=KV_DTYPE, rows=4):
    k = torch.zeros(rows, head_num, head_dim, dtype=dtype)
    return SimpleNamespace(full_kv_pool=SimpleNamespace(k_buffer=[k]))


def _clear_plan():
    set_tp_partition_ratios(None, families=None)
    set_cp_token_ratios(None)


class TestPin3RowSchema(CustomTestCase):
    """Operator pin 3: PP and TP full-attn KV rows are the same bytes."""

    def tearDown(self):
        _clear_plan()

    def test_real_config_row_schemas_are_byte_equal(self):
        cfg = _model_config()
        # PP layout: tp=1 per stage, dcp off -> the per-rank shard IS the
        # full head set. Computed BEFORE any plan install, like the boot.
        pp_heads = cfg.get_num_kv_heads(1)
        # TP layout: weighted uneven DCP replicates KV heads -- the pool
        # stores get_total_num_kv_heads() ONLY because the predicate holds.
        set_tp_partition_ratios(list(FLIP_VECTOR), families=None)
        self.assertTrue(uneven_dcp_kv_replicated(len(FLIP_VECTOR)))
        tp_heads = cfg.get_total_num_kv_heads()

        self.assertEqual(pp_heads, 4)
        self.assertEqual(tp_heads, 4)
        pp_pool = _pool_stub(pp_heads, cfg.hf_text_config.head_dim)
        tp_pool = _pool_stub(tp_heads, cfg.hf_text_config.head_dim)
        assert_row_schema_compatible(pp_pool, tp_pool)  # no raise

        # The measured absolute: K row = 4 heads x 256 x 1 B = exactly
        # 1 KiB/token/layer (section 2's log-derived cell size).
        k0 = pp_pool.full_kv_pool.k_buffer[0]
        self.assertEqual(k0[0].numel() * k0.element_size(), 1024)

    def test_can_fail_head_sharded_tp_pool_is_refused(self):
        """Red arm: without the replication rule the TP pool would be
        head-sharded (max(1, 4//3) = 1 head) and the checker MUST fail."""
        cfg = _model_config()
        _clear_plan()
        self.assertFalse(uneven_dcp_kv_replicated(len(FLIP_VECTOR)))
        sharded_heads = cfg.get_num_kv_heads(len(FLIP_VECTOR))
        self.assertEqual(sharded_heads, 1)
        pp_pool = _pool_stub(cfg.get_num_kv_heads(1), 256)
        tp_pool = _pool_stub(sharded_heads, 256)
        with self.assertRaisesRegex(PhaseFlipBootError, "DIVERGE"):
            assert_row_schema_compatible(pp_pool, tp_pool)

    def test_can_fail_dtype_divergence_is_refused(self):
        pp_pool = _pool_stub(4, 256, dtype=KV_DTYPE)
        tp_pool = _pool_stub(4, 256, dtype=torch.bfloat16)
        with self.assertRaisesRegex(PhaseFlipBootError, "DIVERGE"):
            assert_row_schema_compatible(pp_pool, tp_pool)


class _StubRunner:
    """Attribute shell for exercising real unbound ModelRunner methods."""

    def __new__(cls, enable_flip, is_draft=False, is_tp_stack=False):
        from sglang.srt.model_executor.model_runner import ModelRunner

        r = ModelRunner.__new__(ModelRunner)
        r.server_args = SimpleNamespace(enable_phase_flip=enable_flip)
        r.is_draft_worker = is_draft
        r.is_phase_flip_tp_stack = is_tp_stack
        return r


class TestPin2StructuralGraphAsymmetry(CustomTestCase):
    """Operator pin 2: the PP stack cannot capture ANY graph set."""

    def test_pp_stack_predicate_scope(self):
        self.assertTrue(_StubRunner(True).is_phase_flip_pp_stack)
        # Default boots, drafts, and the TP stack itself are never caught.
        self.assertFalse(_StubRunner(False).is_phase_flip_pp_stack)
        self.assertFalse(_StubRunner(True, is_draft=True).is_phase_flip_pp_stack)
        self.assertFalse(
            _StubRunner(
                True, is_draft=True, is_tp_stack=True
            ).is_phase_flip_pp_stack
        )

    def test_decode_capture_poisoned_for_pp_stack(self):
        r = _StubRunner(True)
        with self.assertRaisesRegex(RuntimeError, "pin 2"):
            r.init_decode_cuda_graph()

    def test_prefill_capture_poisoned_for_pp_stack(self):
        r = _StubRunner(True)
        with self.assertRaisesRegex(RuntimeError, "pin 2"):
            r.init_prefill_cuda_graph()

    def test_poison_gates_are_flag_scoped_not_unconditional(self):
        """Green arm: a non-flip runner passes both gates (and exits at
        the next early-return we arrange), proving the poison is the
        flag, not the method."""
        r = _StubRunner(False)
        r.is_generation = False
        self.assertIsNone(r.init_decode_cuda_graph())
        self.assertIsNone(r.decode_cuda_graph_runner)
        r2 = _StubRunner(False)
        r2.is_weightless_head = True
        r2.is_weightless_worker = False
        r2.eager_runner = sentinel = object()
        self.assertIsNone(r2.init_prefill_cuda_graph())
        self.assertIs(r2.prefill_cuda_graph_runner, sentinel)

    def test_tp_stack_runner_requires_draft_gates(self):
        """The TP stack must ride the is_draft_worker secondary-runner
        gates; constructing it without them is refused (mirrors the
        dual-group-lane guard)."""
        from sglang.srt.model_executor.model_runner import ModelRunner

        with self.assertRaisesRegex(ValueError, "is_draft_worker=True"):
            ModelRunner(
                model_config=None,
                mem_fraction_static=0.5,
                gpu_id=0,
                tp_rank=0,
                tp_size=3,
                moe_ep_rank=0,
                moe_ep_size=1,
                pp_rank=0,
                pp_size=1,
                nccl_port=1,
                server_args=None,
                is_draft_worker=False,
                is_phase_flip_tp_stack=True,
            )


class _SentinelGroup:
    """Stands in for a GroupCoordinator behind the module getters."""


class TestFlipGroupRouting(CustomTestCase):
    _GLOBALS = (
        "_FLIP_TP",
        "_FLIP_DCP",
        "_FLIP_PP",
        "_TP",
        "_ATTN_TP",
        "_DCP",
        "_PP",
        "_PHASE_FLIP_TP_ACTIVE",
    )

    def setUp(self):
        self._saved = {g: getattr(parallel_state, g) for g in self._GLOBALS}

    def tearDown(self):
        for g, v in self._saved.items():
            setattr(parallel_state, g, v)

    def test_activation_without_groups_is_refused(self):
        parallel_state._FLIP_TP = None
        parallel_state._PHASE_FLIP_TP_ACTIVE = False
        with self.assertRaisesRegex(RuntimeError, "never called"):
            parallel_state.set_phase_flip_tp_active(True)
        self.assertFalse(parallel_state.phase_flip_tp_routing_active())

    def test_getters_route_to_flip_set_when_active(self):
        flip_tp, flip_dcp, flip_pp = (
            _SentinelGroup(),
            _SentinelGroup(),
            _SentinelGroup(),
        )
        primary_tp, primary_attn, primary_dcp, primary_pp = (
            _SentinelGroup(),
            _SentinelGroup(),
            _SentinelGroup(),
            _SentinelGroup(),
        )
        parallel_state._FLIP_TP = flip_tp
        parallel_state._FLIP_DCP = flip_dcp
        parallel_state._FLIP_PP = flip_pp
        parallel_state._TP = primary_tp
        parallel_state._ATTN_TP = primary_attn
        parallel_state._DCP = primary_dcp
        parallel_state._PP = primary_pp

        parallel_state.set_phase_flip_tp_active(True)
        self.assertIs(parallel_state.get_tp_group(), flip_tp)
        self.assertIs(parallel_state.get_attn_tp_group(), flip_tp)
        self.assertIs(parallel_state.get_dcp_group(), flip_dcp)
        self.assertIs(parallel_state.get_pp_group(), flip_pp)

        parallel_state.set_phase_flip_tp_active(False)
        self.assertIs(parallel_state.get_tp_group(), primary_tp)
        self.assertIs(parallel_state.get_attn_tp_group(), primary_attn)
        self.assertIs(parallel_state.get_dcp_group(), primary_dcp)
        self.assertIs(parallel_state.get_pp_group(), primary_pp)


def _flip_args(**kwargs):
    defaults = dict(
        enable_phase_flip=True,
        phase_flip_tp_vector="30,17,17",
        pp_size=3,
    )
    defaults.update(kwargs)
    return ServerArgs(model_path="dummy", **defaults)


class TestDeriveTpStackServerArgs(CustomTestCase):
    def test_geometry_rotation(self):
        tp_args = derive_tp_stack_server_args(_flip_args())
        self.assertEqual(tp_args.tp_size, 3)
        self.assertEqual(tp_args.pp_size, 1)
        self.assertEqual(tp_args.dcp_size, 3)
        self.assertEqual(tp_args.rank_tp_ratio, [30, 17, 17])
        self.assertIsNone(tp_args.pp_layer_ratio)
        self.assertIsNone(tp_args.pp_stage_ratio)
        self.assertFalse(tp_args.enable_phase_flip)
        self.assertIsNone(tp_args.phase_flip_tp_vector)

    def test_reach_exactly_the_declared_fields(self):
        """#470 REACH: the copy differs from the boot args ONLY within
        TP_STACK_OVERRIDDEN_FIELDS -- no silent drive-by field. Fields
        whose boot value already equals the override (the pp ratios
        default to None) legitimately show no diff, so the pin is
        subset-plus-core, and a second arm proves a SET pp ratio is
        cleared."""
        src = _flip_args()
        tp_args = derive_tp_stack_server_args(src)
        changed = set()
        for f in dataclasses.fields(ServerArgs):
            if getattr(src, f.name) != getattr(tp_args, f.name):
                changed.add(f.name)
        self.assertLessEqual(changed, set(TP_STACK_OVERRIDDEN_FIELDS))
        self.assertLessEqual(
            {
                "tp_size",
                "pp_size",
                "rank_tp_ratio",
                "dcp_size",
                "enable_phase_flip",
                "phase_flip_tp_vector",
            },
            changed,
        )

        src2 = _flip_args(pp_layer_ratio="32,16,16")
        tp_args2 = derive_tp_stack_server_args(src2)
        self.assertIsNone(tp_args2.pp_layer_ratio)
        self.assertEqual(src2.pp_layer_ratio, "32,16,16")

    def test_source_args_are_untouched(self):
        src = _flip_args()
        derive_tp_stack_server_args(src)
        self.assertEqual(src.tp_size, 1)
        self.assertEqual(src.pp_size, 3)
        self.assertTrue(src.enable_phase_flip)
        self.assertEqual(src.phase_flip_tp_vector, "30,17,17")

    def test_vector_length_mismatch_refused(self):
        with self.assertRaisesRegex(PhaseFlipBootError, "pp_size"):
            derive_tp_stack_server_args(_flip_args(pp_size=2))


def _make_module(seed):
    g = torch.Generator().manual_seed(seed)
    mod = torch.nn.Module()
    base = torch.randint(-128, 127, (6, 8), generator=g, dtype=torch.int8)
    mod.weight = torch.nn.Parameter(base.t(), requires_grad=False)  # strided
    mod.scale = torch.nn.Parameter(
        torch.randn(6, generator=g, dtype=torch.float32), requires_grad=False
    )
    mod.marlin_workspace = torch.nn.Parameter(
        torch.zeros(4, dtype=torch.int32), requires_grad=False
    )
    return mod


class TestSnapshotFreeBindRefill(CustomTestCase):
    def test_checkpoint_param_dict_excludes_workspace_family(self):
        named = checkpoint_param_dict(_make_module(11))
        self.assertIn("weight", named)
        self.assertIn("scale", named)
        self.assertNotIn("marlin_workspace", named)

    def test_snapshot_free_bind_refill_round_trip(self):
        mod = _make_module(13)
        named = checkpoint_param_dict(mod)
        originals = {n: p.detach().clone() for n, p in named.items()}
        layout = plan_arena_layout(named)

        image = snapshot_and_free(named, layout, pin=False)
        # Freed: every checkpoint param is a 0-sized placeholder, the
        # excluded workspace is untouched.
        for n, p in named.items():
            self.assertEqual(p.data.numel(), 0, n)
        self.assertEqual(mod.marlin_workspace.numel(), 4)

        arena = allocate_arena(layout.total_bytes, "cpu")
        views = bind_arena_views(layout, arena, rebind=list(named.items()))
        arena_refill(arena, layout, image)
        for n in originals:
            self.assertEqual(named[n].data.stride(), originals[n].stride(), n)
            self.assertTrue(
                torch.equal(named[n].data.to(torch.float32),
                            originals[n].to(torch.float32)),
                n,
            )
            # Parameter object identity survived (GDN capture contract)
            # and the data now LIVES at the arena view's address.
            self.assertEqual(
                named[n].data.untyped_storage().data_ptr(),
                views[n].untyped_storage().data_ptr(),
                n,
            )
        self.assertIs(mod.weight, named["weight"])
        self.assertIs(mod.scale, named["scale"])

    def test_image_from_tensors_matches_packed_slots(self):
        mod = _make_module(17)
        named = checkpoint_param_dict(mod)
        layout = plan_arena_layout(named)
        direct = image_from_tensors(named, layout, pin=False)
        arena = allocate_arena(layout.total_bytes, "cpu")
        arena.zero_()  # deterministic gap bytes for the comparison
        pack_into_arena(named, layout, arena)
        packed = arena_image(arena, layout, pin=False)
        self.assertTrue(torch.equal(direct, packed))

    def test_refill_refuses_corrupted_direct_image(self):
        mod = _make_module(19)
        named = checkpoint_param_dict(mod)
        layout = plan_arena_layout(named)
        image = image_from_tensors(named, layout, pin=False)
        image[0] ^= 0xFF
        arena = allocate_arena(layout.total_bytes, "cpu")
        with self.assertRaisesRegex(Exception, "checksum"):
            arena_refill(arena, layout, image)


class TestPhaseFlipStacksRefill(CustomTestCase):
    def _stacks(self):
        named_a = checkpoint_param_dict(_make_module(23))
        named_b = checkpoint_param_dict(_make_module(29))
        layout_a = plan_arena_layout(named_a)
        layout_b = plan_arena_layout(named_b)
        arena = allocate_arena(
            max(layout_a.total_bytes, layout_b.total_bytes), "cpu"
        )
        image_a = image_from_tensors(named_a, layout_a, pin=False)
        image_b = image_from_tensors(named_b, layout_b, pin=False)
        return (
            PhaseFlipStacks(
                tp_worker=None,
                arena=arena,
                layout_pp=layout_a,
                layout_tp=layout_b,
                image_pp=image_a,
                image_tp=image_b,
                vector=(30, 17, 17),
            ),
            named_a,
            named_b,
        )

    def test_refill_dispatches_per_direction(self):
        stacks, named_a, named_b = self._stacks()
        stacks.refill(PP_TO_TP)
        views_b = bind_arena_views(stacks.layout_tp, stacks.arena, rebind=())
        for n, t in named_b.items():
            self.assertTrue(
                torch.equal(views_b[n].to(torch.float32), t.to(torch.float32)), n
            )
        stacks.refill(TP_TO_PP)
        views_a = bind_arena_views(stacks.layout_pp, stacks.arena, rebind=())
        for n, t in named_a.items():
            self.assertTrue(
                torch.equal(views_a[n].to(torch.float32), t.to(torch.float32)), n
            )

    def test_unknown_direction_refused(self):
        stacks, _, _ = self._stacks()
        with self.assertRaisesRegex(PhaseFlipBootError, "direction"):
            stacks.refill("sideways")


if __name__ == "__main__":
    unittest.main()
