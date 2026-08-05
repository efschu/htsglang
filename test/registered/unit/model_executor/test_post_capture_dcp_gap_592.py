"""#592: why post-capture KV sizing is off under DCP, pinned.

``post_capture_kv_sizing_planned()`` requires ``dcp_size == 1``, so on this
rig (dcp_size=3 in production) the post-capture measure-and-resize never runs
and the ``_profile_available_bytes`` subtraction behind the same flag is
unreachable. The question this file answers is WHICH KIND of restriction that
is, because the answer decides whether anyone should try to lift it.

IT IS AN IMPLEMENTATION GAP, NOT A HARD CORRECTNESS LIMIT. The gate arrived
with the feature commit itself (upstream 2ad9a243f5) and carried no comment.
Three properties that a resize under DCP would need are already true:

  * graph safety comes from ADDRESS STABILITY, not from a frozen size -- the
    VMM owner remaps physical pages behind a fixed VA reservation, and
    ``store_bound_rows`` is the bound a captured graph baked in;
  * the identical post-capture backing mechanism already runs under uneven DCP
    on the #330 vram-dial lane;
  * the context C stays group-consistent because the resize goes through
    ``_apply_token_constraints``, which min-reduces over the world group on the
    uneven-DCP lane, and the weighted owner rule does not depend on C at all.

What is missing is ONE translation, and that is what makes the gate load
bearing today: on the weighted lane a rank's physical row count is
``dcp_compact_pool_rows(C, cp_S, ratio_r)``, not C. Every pool CONSTRUCTION
site routes through ``_dcp_token_sharded_pool_rows`` for that reason; the
resize path hands ``config.max_total_num_tokens`` to ``_finalize_backing_tokens``
as a row count. The tests below pin the gate, pin the asymmetry, and record the
two lemmas a lift may build on -- as assertions rather than as prose, so a
future attempt starts from checkable evidence.
"""

import inspect
import types
import unittest
from unittest import mock

from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _planned_args(**over):
    """A ServerArgs-shaped stub on which every OTHER condition of
    ``post_capture_kv_sizing_planned`` is satisfied, so a False answer can only
    come from the condition under test."""
    from sglang.srt.server_args import Backend

    base = dict(
        device="cuda",
        dcp_size=1,
        use_mla_backend=False,
        prefill_only_disable_kv_cache=False,
        enable_memory_saver=False,
        enable_dp_attention=False,
        disaggregation_mode="null",
        cuda_graph_config=types.SimpleNamespace(
            prefill=types.SimpleNamespace(backend=Backend.FULL),
            decode=types.SimpleNamespace(backend=Backend.FULL),
        ),
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _planned(args) -> bool:
    from sglang.srt.server_args import ServerArgs

    with mock.patch.dict("os.environ", {"SGLANG_ENABLE_POST_CAPTURE_KV_SIZING": "1"}):
        return ServerArgs.post_capture_kv_sizing_planned(args)


class TestTheGateIsWhereTheDocumentationSaysItIs(unittest.TestCase):
    def test_dcp_pins_the_flag_false(self):
        self.assertFalse(_planned(_planned_args(dcp_size=3)))

    def test_and_it_is_the_dcp_condition_doing_it(self):
        """The can-fail half: with the SAME stub at dcp_size=1 the answer is
        True, so a future edit that drops the condition cannot pass both."""
        self.assertTrue(_planned(_planned_args(dcp_size=1)))

    def test_the_env_switch_still_dominates(self):
        from sglang.srt.server_args import ServerArgs

        with mock.patch.dict(
            "os.environ", {"SGLANG_ENABLE_POST_CAPTURE_KV_SIZING": "0"}
        ):
            self.assertFalse(
                ServerArgs.post_capture_kv_sizing_planned(_planned_args(dcp_size=1))
            )

    def test_the_gate_carries_its_reason(self):
        """#592 exists because this condition had no comment for a year. The
        reason is now next to it and must stay there."""
        from sglang.srt import server_args as sa

        src = inspect.getsource(sa.ServerArgs.post_capture_kv_sizing_planned)
        self.assertIn("#592", src)
        self.assertIn("dcp_compact_pool_rows", src)
        self.assertIn("_dcp_token_sharded_pool_rows", src)


class TestTheMissingTranslation(unittest.TestCase):
    """The falsifier: is the resize path safe to run under DCP as written?
    No -- and this is exactly why, in one number."""

    #: The production shape on this rig: dcp_size 3 with a weighted vector.
    RATIOS = (5, 4, 4)
    CONTEXT = 65536

    def test_the_row_count_a_rank_needs_is_not_the_global_context(self):
        cp_S = sum(self.RATIOS)
        for rank, ratio in enumerate(self.RATIOS):
            rows = dcp_compact_pool_rows(self.CONTEXT, cp_S, ratio)
            self.assertNotEqual(
                rows,
                self.CONTEXT,
                f"rank {rank}: if rows equalled C the gate would be pointless",
            )
            self.assertLess(rows, self.CONTEXT)

    def test_finalize_backing_passes_the_global_context_untranslated(self):
        """Behavioural, not just textual: the real method is invoked against a
        recording stub, so the pin survives a refactor of the source text."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        seen = []
        stub = types.SimpleNamespace(
            _finalize_backing_tokens=lambda n: seen.append(int(n))
        )
        config = types.SimpleNamespace(max_total_num_tokens=self.CONTEXT)
        MHATokenToKVPool.finalize_backing(stub, config)
        self.assertEqual(seen, [self.CONTEXT])

    def test_construction_translates_where_the_resize_does_not(self):
        """The asymmetry itself. Five construction sites go through the
        translation; the resize path has none -- that is the whole gap."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.model_executor import model_runner_kv_cache_mixin as mx

        mixin_src = inspect.getsource(mx)
        self.assertGreaterEqual(
            mixin_src.count("self._dcp_token_sharded_pool_rows("),
            5,
            "the construction sites stopped translating the row count",
        )
        # The CALL form, not the bare name: both sites now MENTION the helper
        # in prose (that is the #592 documentation), and a test that could not
        # tell a reference from a call would go green on a comment.
        resize_src = inspect.getsource(
            mx.ModelRunnerKVCacheMixin.post_capture_resize_kv_pool
        )
        self.assertNotIn("_dcp_token_sharded_pool_rows(", resize_src)
        self.assertNotIn(
            "_dcp_token_sharded_pool_rows(",
            inspect.getsource(MHATokenToKVPool.finalize_backing),
        )

    def test_the_resize_primitive_names_the_assumption(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        src = inspect.getsource(MHATokenToKVPool.finalize_backing)
        self.assertIn("#592", src)
        self.assertIn("dcp_size == 1", src)


class TestTheLemmasALiftMayBuildOn(unittest.TestCase):
    """Recorded as assertions so a future lift can check them instead of
    trusting this file's prose."""

    def test_the_row_count_is_monotone_in_the_context(self):
        """A shrinking C can only shrink the rows, which is what keeps a
        post-capture resize inside the VA reservation a graph baked in."""
        cp_S, ratio = 13, 5
        rows = [dcp_compact_pool_rows(c, cp_S, ratio) for c in range(0, 40000, 997)]
        self.assertEqual(rows, sorted(rows))

    def test_the_post_capture_context_can_only_shrink(self):
        """`cap_tokens=self.max_total_num_tokens` is what bounds it; together
        with monotonicity above, a translated resize can never ask for more
        rows than the boot reserved."""
        from sglang.srt.model_executor import model_runner_kv_cache_mixin as mx

        src = inspect.getsource(mx.ModelRunnerKVCacheMixin.post_capture_resize_kv_pool)
        self.assertIn("cap_tokens=self.max_total_num_tokens", src)

    def test_the_weighted_owner_rule_does_not_depend_on_the_context(self):
        """So a C change post-capture cannot invalidate the owner mapping --
        only the row COUNT follows C, not who owns which slot."""
        from sglang.srt.layers.dcp.owner import dcp_weighted_owner_bounds

        params = list(inspect.signature(dcp_weighted_owner_bounds).parameters)
        self.assertEqual(params, ["dcp_size", "dcp_rank"])

    def test_the_context_is_agreed_group_wide_on_the_resize_path(self):
        """The resize measures a rank-local budget but does NOT decide C
        rank-locally: it routes through the min-reduce in
        _apply_token_constraints."""
        from sglang.srt.model_executor import model_runner_kv_cache_mixin as mx

        resize = inspect.getsource(
            mx.ModelRunnerKVCacheMixin.post_capture_resize_kv_pool
        )
        self.assertIn("_config_from_budget", resize)
        budget = inspect.getsource(mx.ModelRunnerKVCacheMixin._config_from_budget)
        self.assertIn("_apply_token_constraints", budget)
        constraints = inspect.getsource(
            mx.ModelRunnerKVCacheMixin._apply_token_constraints
        )
        self.assertIn("ReduceOp.MIN", constraints)
        self.assertIn("uneven_dcp_active", constraints)

    def test_backing_changes_are_graph_safe_by_address_stability(self):
        """Not by the size being frozen -- which is the property that makes a
        lift conceivable at all."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        src = inspect.getsource(MHATokenToKVPool.runtime_set_backing_tokens)
        self.assertIn("Addresses never move", src)


if __name__ == "__main__":
    unittest.main()
