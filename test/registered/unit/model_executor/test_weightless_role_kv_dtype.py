"""Per-ROLE KV-cache precision on the weightless-KV fast lane (#127).

The lane's two roles spend their VRAM on different things -- the head rank on
weights plus a KV token-shard, a weightless worker on nothing but a KV
token-shard -- so ``--weightless-kv-worker-cache-dtype`` lets the workers
store KV at a lower precision than the group-wide ``--kv-cache-dtype``.

These are the FALSIFIERS for the format boundary. Two roles holding the same
logical slot space in two different physical formats is exactly the shape of
bug that does not announce itself: nothing in the host<->device transfer path
or the cross-rank collective re-derives an element type, so a wrong dtype
reinterprets bytes and returns plausible garbage instead of raising. Each test
below pins one edge of that boundary:

* the resolver only ever moves a WORKER off the group spec (never the head,
  never the default path),
* the cross-rank K/V wire dtype is the model COMPUTE dtype and must never be
  wired to the KV storage dtype (that is the one place bytes really do cross
  roles),
* the host overflow tier's byte stride must equal its device pool's, checked
  rather than assumed,
* the capacity arithmetic says out loud what a role-split actually buys, so
  nobody reads a doubled worker pool as a doubled context,
* and the per-token cell size charges the kv-head count the lane really
  allocates.

CPU only: every rule under test is a pure function of the configuration.
"""

import inspect
import unittest

import torch

from sglang.srt.distributed.utils import set_weightless_kv_head_rank
from sglang.srt.layers.dcp.role_kv_dtype import (
    LOSSY_KV_CACHE_DTYPE_SPECS,
    WORKER_KV_CACHE_DTYPE_CHOICES,
    effective_kv_cache_dtype_spec,
    even_modulo_global_capacity,
    host_tier_stride_mismatch,
    worker_dtype_is_role_split,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_args(**kwargs):
    """ServerArgs with model_path='dummy' short-circuits __post_init__, so the
    weightless handler can be exercised in isolation."""
    return ServerArgs(model_path="dummy", **kwargs)


class TestRoleSpecResolution(CustomTestCase):
    """Who gets moved off the group-wide spec, and who never does."""

    def test_default_path_is_the_identity(self):
        # No lane, no flag -> every rank resolves the group spec. This is the
        # byte-identity guarantee for every existing boot.
        for spec in ("auto", "fp8_e5m2", "bf16"):
            for is_worker in (False, True):
                self.assertEqual(
                    effective_kv_cache_dtype_spec(spec, None, is_worker),
                    spec,
                    msg=f"spec={spec} is_worker={is_worker}",
                )

    def test_head_rank_never_takes_the_worker_spec(self):
        # The head keeps its own precision by construction -- that IS the
        # feature. A head that silently followed the worker flag would make
        # the split meaningless and would change the head-local paths.
        self.assertEqual(
            effective_kv_cache_dtype_spec("bf16", "fp8_e5m2", False), "bf16"
        )

    def test_worker_takes_the_override(self):
        self.assertEqual(
            effective_kv_cache_dtype_spec("bf16", "fp8_e5m2", True), "fp8_e5m2"
        )

    def test_auto_means_inherit_not_redetect(self):
        # "auto" on the worker flag must mean "inherit the group spec", NOT
        # "run the checkpoint auto-detection independently on this rank" --
        # the latter could land the two roles on different formats without
        # anyone asking for a split.
        self.assertEqual(
            effective_kv_cache_dtype_spec("fp8_e5m2", "auto", True), "fp8_e5m2"
        )
        self.assertEqual(effective_kv_cache_dtype_spec("auto", "auto", True), "auto")

    def test_role_split_predicate(self):
        self.assertFalse(worker_dtype_is_role_split("bf16", None))
        self.assertFalse(worker_dtype_is_role_split("bf16", "auto"))
        self.assertFalse(worker_dtype_is_role_split("bf16", "bf16"))
        # Same dtype spelled two ways is not a split.
        self.assertFalse(worker_dtype_is_role_split("bf16", "bfloat16"))
        self.assertTrue(worker_dtype_is_role_split("bf16", "fp8_e5m2"))
        self.assertTrue(worker_dtype_is_role_split("fp8_e5m2", "bf16"))

    def test_lossy_specs_are_labelled_and_not_reachable_by_default(self):
        # Quality-last: the lossy formats must be named as such, and the
        # default (None) must not resolve to one of them.
        self.assertIn("fp8_e5m2", LOSSY_KV_CACHE_DTYPE_SPECS)
        self.assertIn("fp8_e4m3", LOSSY_KV_CACHE_DTYPE_SPECS)
        self.assertNotIn("auto", LOSSY_KV_CACHE_DTYPE_SPECS)
        self.assertNotIn("bf16", LOSSY_KV_CACHE_DTYPE_SPECS)
        self.assertEqual(
            effective_kv_cache_dtype_spec("bf16", None, True), "bf16"
        )


class TestServerArgsValidation(CustomTestCase):
    """Fail at arg-parse, not after the pools are built."""

    def test_flag_requires_the_lane(self):
        args = make_args(weightless_kv_worker_cache_dtype="fp8_e5m2")
        with self.assertRaises(ValueError) as ctx:
            args._handle_weightless_kv_fastlane()
        self.assertIn("--weightless-kv-fastlane", str(ctx.exception))

    def test_fp4_is_refused_on_the_worker(self):
        # fp4 needs a second (scale) buffer in the cell and its own pool
        # class; refusing it is a scope statement, not a capability claim.
        self.assertNotIn("fp4_e2m1", WORKER_KV_CACHE_DTYPE_CHOICES)
        args = make_args(
            weightless_kv_fastlane=True,
            tp_size=2,
            dcp_size=2,
            weightless_kv_worker_cache_dtype="fp4_e2m1",
        )
        with self.assertRaises(ValueError) as ctx:
            args._handle_weightless_kv_fastlane()
        self.assertIn("fp4_e2m1", str(ctx.exception))

    def test_scale_file_is_refused_with_a_role_split(self):
        # KV scales live on model weights; a meta-device worker has none, so
        # head and workers would quantize against different scales.
        args = make_args(
            weightless_kv_fastlane=True,
            tp_size=2,
            dcp_size=2,
            kv_cache_dtype="bf16",
            weightless_kv_worker_cache_dtype="fp8_e4m3",
            quantization_param_path="/tmp/scales.json",
        )
        with self.assertRaises(ValueError) as ctx:
            args._handle_weightless_kv_fastlane()
        self.assertIn("quantization-param-path", str(ctx.exception))

    def test_scale_file_is_fine_without_a_role_split(self):
        # Inheriting (the default) is the pre-existing group-wide behaviour
        # and must stay reachable with a scale file.
        args = make_args(
            weightless_kv_fastlane=True,
            tp_size=2,
            dcp_size=2,
            kv_cache_dtype="fp8_e4m3",
            weightless_kv_worker_cache_dtype="auto",
            quantization_param_path="/tmp/scales.json",
        )
        args._handle_weightless_kv_fastlane()  # must not raise

    def test_valid_role_split_parses(self):
        args = make_args(
            weightless_kv_fastlane=True,
            tp_size=3,
            dcp_size=3,
            kv_cache_dtype="bf16",
            weightless_kv_worker_cache_dtype="fp8_e5m2",
        )
        args._handle_weightless_kv_fastlane()  # must not raise


class TestCapacityArithmetic(CustomTestCase):
    """What a role split actually buys -- stated, not assumed."""

    def test_even_modulo_capacity_is_dcp_times_the_minimum(self):
        total, binding, stranded = even_modulo_global_capacity([100, 400, 400], 3)
        self.assertEqual(total, 300)
        self.assertEqual(binding, 0)
        self.assertEqual(stranded, [0, 300, 300])

    def test_worker_precision_buys_nothing_when_the_head_binds(self):
        # THE headline caveat. Head = rank 0 (weights eat its card), workers
        # already have the bigger KV budget. Halving the workers' per-token
        # bytes doubles THEIR capacity and changes the group total by zero,
        # because the even-modulo slot space is rank-uniform and the group
        # takes the MIN. Anyone expecting "+2x context from worker fp8" has
        # to check the binding rank first; this test is why the boot log
        # prints it.
        bf16 = [100, 400, 400]
        worker_fp8 = [100, 800, 800]
        self.assertEqual(
            even_modulo_global_capacity(bf16, 3)[0],
            even_modulo_global_capacity(worker_fp8, 3)[0],
        )

    def test_worker_precision_pays_when_a_worker_binds(self):
        # The configuration the feature is FOR: small worker cards, a head
        # with slack. Now the min moves and the whole group gains.
        bf16 = [400, 100, 100]
        worker_fp8 = [400, 200, 200]
        self.assertEqual(even_modulo_global_capacity(bf16, 3)[0], 300)
        self.assertEqual(even_modulo_global_capacity(worker_fp8, 3)[0], 600)
        self.assertEqual(even_modulo_global_capacity(worker_fp8, 3)[1], 1)

    def test_capacity_rejects_a_malformed_vector(self):
        with self.assertRaises(ValueError):
            even_modulo_global_capacity([100, 200], 3)
        with self.assertRaises(ValueError):
            even_modulo_global_capacity([], 1)
        with self.assertRaises(ValueError):
            even_modulo_global_capacity([100], 0)


class TestHostTierStrideGuard(CustomTestCase):
    """The host overflow tier must never reinterpret KV bytes."""

    def test_matching_tiers_pass(self):
        self.assertIsNone(
            host_tier_stride_mismatch(
                device_store_itemsize=1,
                device_head_num=4,
                device_head_dim=128,
                host_itemsize=1,
                host_token_stride_size=4 * 128 * 1,
            )
        )

    def test_itemsize_mismatch_is_named_not_silent(self):
        msg = host_tier_stride_mismatch(
            device_store_itemsize=1,  # fp8 device pool (uint8-backed)
            device_head_num=4,
            device_head_dim=128,
            host_itemsize=2,  # bf16 host tier -- the corruption case
            host_token_stride_size=4 * 128 * 2,
        )
        self.assertIsNotNone(msg)
        self.assertIn("itemsize", msg)

    def test_stride_mismatch_is_named_not_silent(self):
        # Same element size, wrong row width (e.g. a tier built for a
        # different kv-head count) -- equally silent without this check.
        msg = host_tier_stride_mismatch(
            device_store_itemsize=1,
            device_head_num=4,
            device_head_dim=128,
            host_itemsize=1,
            host_token_stride_size=2 * 128 * 1,
        )
        self.assertIsNotNone(msg)
        self.assertIn("token stride", msg)


class TestWireDtypeIsNotStorageDtype(CustomTestCase):
    """The one place KV bytes really do cross the role boundary.

    Head and workers exchange the new token's K/V through
    ``cp_all_gather_heads_uneven``, whose payload dtype is ``_wl_dtype``. That
    must stay the model COMPUTE dtype: it is what the head actually projects,
    and both sides must agree on it for the padded all-gather. Wiring it to
    the KV STORAGE dtype would (a) desync the collective the moment the two
    roles store different formats and (b) quantize the value twice. Pinned at
    source level because the failure mode is an NCCL fault or silent garbage,
    not an exception any unit test could provoke on CPU.
    """

    def test_wl_dtype_follows_the_compute_dtype(self):
        from sglang.srt.layers.attention import flashinfer_backend

        src = inspect.getsource(flashinfer_backend.FlashInferAttnBackend.__init__)
        self.assertIn("self._wl_dtype = mc.dtype", src)
        self.assertNotIn("self._wl_dtype = model_runner.kv_cache_dtype", src)
        self.assertNotIn("self._wl_dtype = self.kv_cache_dtype", src)


class TestRoleDtypeReachesTheResolver(CustomTestCase):
    """The role override must be an INPUT to the one existing resolver.

    ``configure_kv_cache_dtype`` is the single place a spec string becomes a
    torch dtype (it also handles HIP remapping and checkpoint auto-detection).
    If any branch there kept reading ``server_args.kv_cache_dtype`` directly,
    a worker would silently fall back to the group spec for that format only
    -- a per-format hole rather than a clean refusal.
    """

    def test_no_branch_bypasses_the_resolved_spec(self):
        from sglang.srt.model_executor import model_runner as mr_mod

        src = inspect.getsource(mr_mod.ModelRunner.configure_kv_cache_dtype)
        self.assertIn("effective_kv_cache_dtype_spec(", src)
        # A DISPATCH branch is one that compares the spec against a concrete
        # format literal. Reads that merely pass the group spec to the
        # resolver, or compare it to the resolved spec for the log line, are
        # fine -- they cannot route a rank to the wrong format.
        formats = ("auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1")
        branch_reads = [
            line
            for line in src.splitlines()
            if "self.server_args.kv_cache_dtype" in line
            and ("if " in line or "elif " in line)
            and any(f'"{fmt}"' in line for fmt in formats)
        ]
        self.assertEqual(branch_reads, [], msg=f"unresolved branches: {branch_reads}")
        # And the resolved spec is what every branch keys off.
        self.assertIn('if kv_spec == "auto":', src)
        self.assertIn('elif kv_spec == "fp8_e5m2":', src)


class TestCellSizeMatchesTheAllocatedPool(CustomTestCase):
    """The profiled per-token cost must charge what the lane really allocates.

    On the weightless lane every rank stores the FULL ``total_num_kv_heads``
    per slot (the head projects them all and broadcasts). The lane runs with
    ``rank_tp_ratio=None``, so the pre-existing ``uneven_dcp_kv_replicated()``
    trigger is False and the cell used to be charged the per-rank
    ``get_num_kv_heads(tp)`` share instead -- under-charging by the head ratio
    and inflating ``max_total_num_tokens`` by the same factor, i.e. sizing the
    pool past the rank's own budget.
    """

    def setUp(self):
        set_weightless_kv_head_rank(None)

    def tearDown(self):
        set_weightless_kv_head_rank(None)

    def _cell_size(self, *, kv_itemsize, total_kv_heads, sharded_kv_heads):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
        from sglang.srt.runtime_context import get_parallel

        mr = MagicMock()
        mr.use_mla_backend = False
        mr.num_effective_layers = 4
        mr.start_layer = 0
        mr.end_layer = 4
        mc = SimpleNamespace()
        mc.head_dim = 128
        mc.v_head_dim = 128
        mc.get_num_kv_heads = lambda tp_size: sharded_kv_heads
        mc.get_total_num_kv_heads = lambda: total_kv_heads
        mc.hf_config = SimpleNamespace(architectures=["Qwen3NextForCausalLM"])
        mc.hf_text_config = SimpleNamespace(architectures=["Qwen3NextForCausalLM"])
        mr.model_config = mc

        with (
            patch("torch._utils._element_size", return_value=kv_itemsize),
            get_parallel().override(attn_tp_size=3, attn_dcp_size=3),
        ):
            return DefaultPoolConfigurator._compute_cell_size(
                DefaultPoolConfigurator.__new__(DefaultPoolConfigurator), mr, 4
            )

    def test_stock_path_charges_the_per_rank_shard(self):
        # Lane off -> unchanged (this is the byte-identity guard for every
        # non-weightless deployment).
        cell = self._cell_size(kv_itemsize=2, total_kv_heads=4, sharded_kv_heads=1)
        self.assertEqual(cell, 1 * (128 + 128) * 4 * 2)

    def test_weightless_lane_charges_the_full_kv_heads(self):
        set_weightless_kv_head_rank(0)
        cell = self._cell_size(kv_itemsize=2, total_kv_heads=4, sharded_kv_heads=1)
        self.assertEqual(cell, 4 * (128 + 128) * 4 * 2)

    def test_fp8_halves_the_cell_with_no_extra_scale_term(self):
        # fp8 is a pure 2x slot win: unlike fp4 it adds no scale buffer to the
        # cell. This is the arithmetic behind "the worker pool doubles".
        set_weightless_kv_head_rank(0)
        bf16 = self._cell_size(kv_itemsize=2, total_kv_heads=4, sharded_kv_heads=1)
        fp8 = self._cell_size(kv_itemsize=1, total_kv_heads=4, sharded_kv_heads=1)
        self.assertEqual(bf16, 2 * fp8)


class TestPoolDtypeIsUint8Backed(CustomTestCase):
    """fp8 pools are physically uint8; the logical dtype is a view.

    Both the write (`cast then .view(store_dtype)`) and the read
    (`.view(self.dtype)`) sides depend on this, and so does every byte-stride
    transfer. Pinning it here means a torch change that made fp8 indexable
    (and someone dropping the indirection) shows up as a red unit test rather
    than as a wrong-stride host transfer.
    """

    def test_fp8_store_dtype_is_uint8_and_half_the_width(self):
        self.assertEqual(torch.float8_e5m2.itemsize, 1)
        self.assertEqual(torch.float8_e4m3fn.itemsize, 1)
        self.assertEqual(torch.bfloat16.itemsize, 2)
        self.assertEqual(torch.uint8.itemsize, torch.float8_e5m2.itemsize)


if __name__ == "__main__":
    unittest.main()
