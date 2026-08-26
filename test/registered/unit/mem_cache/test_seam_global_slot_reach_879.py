"""#879: global slots handed to a physically-indexed pool -- a PIN, and the asymmetry was not one.

I raised this in #875: `Req.load_kv_cache` (schedule_batch.py:1806) hands
`req_to_token[..., :seqlen-1]` -- GLOBAL cache slots -- to `load_cpu_copy`, which
indexes PHYSICAL buffer rows, and `HybridLinearKVPool` forwards them untouched.
I flagged it as "bigger than the carry". It is real, and it is a pin.

Q1 -- THE SIBLING ASYMMETRY IS VOID, AND FOR A STRONGER REASON THAN I FOUND.
I argued in #875 that "one pool family translates on this path and the other does
not". The operator then measured the thing neither of us had checked:
`UnifiedSWAKVPool` IS NEVER CONSTRUCTED ON THIS RIG. One construction site
(unified_memory_pool.py:1325), behind `enable_unified_memory` which defaults
False (server_args.py:1371) and is False in this boot, behind
`assert self.is_hybrid_swa` (model_runner_kv_cache_mixin.py:2809) which this
GDN/Mamba-hybrid checkpoint does not satisfy, and behind a help text excluding
speculative decoding which this rig runs on every boot.

So the comparison was against a pool nobody instantiates. A family that is never
built cannot be evidence that another family is missing something. My own reason
-- that the two translate different axes (`virtual_to_physical`, a lazily-bound
page layer, versus the DCP owner rule) -- is true and is the weaker argument; the
pool not existing here is the stronger one, and both point the same way.

WHAT THAT KILLS: the asymmetry must never be the reason anyone touches this path.
Copying the unified pool's translation into the hybrid path would install a
virtual->physical mapping where no virtual layer exists -- the fix that copies
the wrong side.

WHAT SURVIVES, AND IT IS THE WHOLE TICKET: `Req.load_kv_cache`
(schedule_batch.py:1806) hands GLOBAL slots to `load_cpu_copy`, which indexes
PHYSICALLY. That is a property of THIS rig's LIVE path -- `HybridLinearKVPool`
wrapping an `MHATokenToKVPool` -- measured at file:line and independent of any
other pool.

Q2 -- IS THE PATH REACHABLE UNDER TP WITH dcp_size > 1? Almost never, and the
remaining window is narrow enough to name.
  * `schedule_batch.py:2005` and `disaggregation/decode.py:736` are gated on
    `disaggregation_mode == "decode"`. This rig boots `disaggregation_mode='null'`
    and the tree says so itself at phase_flip_runtime.py:1545 -- "the decode-disagg
    host copy (unreachable here)".
  * The seam path is the only other one. `copy_state=True` is set at
    phase_flip_runtime.py:1548 "and nowhere else", i.e. a copy is taken ONLY at a
    flip -- so it is taken in the source phase and restored after the cutover,
    which is always cross-layout, which #861c's layout refusal declines BEFORE
    `load_cpu_copy` is reached.
  * THE ONE SURVIVING WINDOW: a flip ABANDONED after `retract_all(copy_state=True)`
    has run. The restore then happens in the SAME phase, the layouts match, the
    #861c guard passes, and `load_cpu_copy` runs with that phase's slots. In TP
    those are global slots against a compact pool.
Whether abandonment can occur after the copy is taken is the one thing I could
NOT settle at the desk. It is the next check and it is cheap.

Q3 -- DOES TP PRODUCE IN-RANGE-BUT-WRONG-ROW, so that #783b's guard catches
nothing? BOTH modes exist, and the silent one is bounded.

`dcp_global_context_slots` (layers/dcp/owner.py:230-233) settles the geometry for
the WEIGHTED lane, which is this rig's: "``max_total_num_tokens`` is ALREADY the
global context budget C -- the allocator index space is C and each rank stores
its ``ratio_r / S`` share." So `req_to_token` holds slots in [0, C) while the
pool has about C * ratio_r / S physical rows.

  * Most slots EXCEED the physical row count, so `check_cpu_copy_rows` fires.
    Loud. The guard works, and it works for the majority of cases.
  * Slots BELOW the compact row count are in-range and map to the WRONG row:
    global L belongs at `(L // cp_S) * cp_ratio + (L % cp_S - cp_lo)`, not at L.
    There the guard is blind.
  * But the guard tests the MIN and MAX of the whole vector, so it raises if ANY
    index is out of range. Only a request whose ENTIRE context sits inside the
    low compact window slips through silently -- short requests at low slot ids.

So the guard is neither useless nor a proof of safety. It suggests more safety
than it delivers, which is the uncomfortable form -- but the silent subset is
confined to short contexts inside one already-narrow window.

VERDICT: PIN. Unreachable through the disagg path on this rig (mode is null),
unreachable through the ordinary seam path (#861c refuses cross-layout first),
reachable only through an abandoned flip, and even there loud for all but short
contexts. No fix is applied: the correct fix is the DCP owner rule on this path,
and applying it while the path is unreachable would be a change nothing could
observe -- with a real chance of copying the wrong side, which Q1 shows was
already the tempting move.

Hermetic: source inspection and integer arithmetic. No CUDA, no device.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest

from sglang.test.test_utils import CustomTestCase


class TestQ1TheSiblingIsNeverConstructedHere(CustomTestCase):
    """The asymmetry is void. Asserted on the GATES, because that is the strong
    reason; the different-axes argument is kept below as the weaker second."""

    def test_the_unified_pool_has_exactly_one_construction_site(self):
        import ast
        from pathlib import Path

        import sglang.srt.mem_cache.unified_memory_pool as ump

        tree = ast.parse(Path(ump.__file__).read_text())
        sites = sum(
            1
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "UnifiedSWAKVPool"
        )
        self.assertEqual(1, sites, "UnifiedSWAKVPool gained a construction site")

    def test_its_flag_defaults_off(self):
        from sglang.srt.server_args import ServerArgs

        self.assertFalse(
            ServerArgs.__dataclass_fields__["enable_unified_memory"].default
        )

    def test_its_builder_asserts_a_model_class_this_rig_is_not(self):
        """`assert self.is_hybrid_swa` -- this checkpoint is a GDN/Mamba hybrid,
        not an SWA hybrid, so the builder refuses before the pool exists."""
        import inspect

        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        src = inspect.getsource(ModelRunnerKVCacheMixin._init_unified_swa_pools)
        self.assertIn("assert self.is_hybrid_swa", src)

    def test_the_weaker_second_reason_also_holds(self):
        """The load-bearing fact. If this ever gains cp_S/cp_ratio/cp_lo the
        asymmetry becomes real evidence and Q1 must be redone."""
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool

        src = inspect.getsource(UnifiedSWAKVPool._virt_tokens_to_phys_tokens)
        self.assertIn("virtual_to_physical", src)
        for owner_term in ("cp_S", "cp_ratio", "cp_lo", "dcp_weighted"):
            self.assertNotIn(
                owner_term,
                src,
                f"the unified translation now mentions {owner_term}; it would be "
                f"the DCP owner rule after all and #879's Q1 answer flips",
            )

    def test_the_unified_translation_is_about_a_virtual_page_layer(self):
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool

        doc = UnifiedSWAKVPool._virt_tokens_to_phys_tokens.__doc__ or ""
        self.assertIn("Virtual TOKEN ids", doc)
        self.assertIn("Unbound pages", doc)

    def test_the_hybrid_pool_translates_the_mamba_ids_and_says_why(self):
        """Not an omission: a different id space, named at the site."""
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        src = inspect.getsource(HybridLinearKVPool.get_cpu_copy)
        self.assertIn("_mamba_translate", src)
        self.assertIn("PHYSICAL ids", src)


class TestQ2Reachability(CustomTestCase):
    def test_the_disagg_callers_are_gated_on_decode_mode(self):
        """Both non-seam callers sit behind `disaggregation_mode == "decode"`.
        Asserted on the source of the enclosing function so a moved gate shows
        up here rather than silently widening the reach."""
        from sglang.srt.managers import schedule_batch as sb

        src = inspect.getsource(sb.release_req)
        self.assertIn('disaggregation_mode == "decode"', src)
        self.assertIn("offload_kv_cache", src)

    def test_the_seam_copy_is_taken_only_at_a_flip(self):
        """`copy_state=True` at exactly one site. If a second appears, a copy
        can be taken without a flip and the whole reachability answer moves."""
        import ast
        from pathlib import Path

        import sglang.srt.managers.phase_flip_runtime as pfr

        # BY AST, NOT BY TEXT -- and this test was written by text first and
        # caught itself: line 1545 is a COMMENT quoting `copy_state=True` while
        # line 1548 is the argument. Fourth instance on this branch of a name in
        # prose read as a use, and the first one my own test found.
        tree = ast.parse(Path(pfr.__file__).read_text())
        sites = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if (
                    kw.arg == "copy_state"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    sites += 1
        self.assertEqual(
            1, sites, "copy_state=True is passed at more than one call site now"
        )

    def test_the_restore_is_guarded_by_the_layout_refusal_first(self):
        """The #861c guard runs BEFORE `load_kv_cache` inside
        `restore_seam_state` -- which is why the ordinary cross-layout path
        never reaches the pool. Asserted on ORDER, not on presence."""
        from sglang.srt.managers import schedule_batch as sb

        src = inspect.getsource(sb.restore_seam_state)
        self.assertLess(
            src.index("kv_drifted"),
            src.index("req.load_kv_cache("),
            "the layout check no longer precedes the restore; the ordinary "
            "cross-layout path would reach load_cpu_copy again",
        )


class TestQ3TheGuardCatchesTheLoudMajorityOnly(CustomTestCase):
    """The geometry, as arithmetic, from the weighted lane's own statement."""

    C = 100_000  # global context budget == the allocator index space
    VECTOR = (29, 19, 16)

    def _compact_rows(self, rank):
        S = sum(self.VECTOR)
        return (self.C // S + 1) * self.VECTOR[rank]

    def test_the_pool_is_far_smaller_than_the_allocator_index_space(self):
        for r in range(3):
            self.assertLess(self._compact_rows(r), self.C // 2)

    def test_most_global_slots_are_OUT_OF_RANGE_hence_loud(self):
        for r in range(3):
            rows = self._compact_rows(r)
            out_of_range = self.C - rows
            self.assertGreater(
                out_of_range / self.C,
                0.5,
                "the majority of global slots must exceed the physical rows, "
                "or check_cpu_copy_rows stops being the loud majority case",
            )

    def test_a_low_slot_is_IN_RANGE_but_maps_to_the_wrong_row(self):
        """The silent subset, demonstrated rather than asserted: a slot below
        the row count is accepted by the bound and belongs somewhere else."""
        from sglang.srt.layers.dcp.owner import dcp_weighted_read_slots
        import torch

        S = sum(self.VECTOR)
        cp_lo = self.VECTOR[0]  # rank 1's lower bound
        cp_hi = cp_lo + self.VECTOR[1]
        loc = torch.tensor([S + cp_lo + 3], dtype=torch.int64)
        compact, owned = dcp_weighted_read_slots(loc, S, cp_lo, cp_hi, self.VECTOR[1])
        self.assertTrue(bool(owned[0]))
        self.assertNotEqual(
            int(loc[0]),
            int(compact[0]),
            "if the global slot equalled its compact row there would be no "
            "wrong-row case at all",
        )
        self.assertLess(int(compact[0]), self._compact_rows(1))

    def test_the_row_guard_tests_the_extremes_of_the_whole_vector(self):
        """Why the silent case is confined to short contexts: one out-of-range
        entry anywhere raises for the entire call."""
        from sglang.srt.mem_cache.memory_pool import check_cpu_copy_rows

        src = inspect.getsource(check_cpu_copy_rows)
        self.assertIn("indices.min()", src)
        self.assertIn("indices.max()", src)


if __name__ == "__main__":
    unittest.main()
