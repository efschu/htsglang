"""#747 seam parity: one grid rule, two cache lineages.

`MambaRadixCache` honours `--mamba-checkpoint-interval`; the unified tree's
mamba component does not know the flag exists. That divergence is why
`server_args.py`'s refusal was written, and why its stated reason names a class
(`HiMambaRadixCache`) that has no construction site: the two lineages drifted
and nobody could see it. See NOTE_747.

So the decisions live in ONE place -- `mem_cache/mamba_ckpt_utils.py`, pure
integer arithmetic -- and both lineages call it. These tests pin the rules
against the `mamba_radix_cache.py` lines they mirror, so a future edit to
either lineage that changes an anchor decision has to change this file too.

THE ONE GENUINELY NEW DECISION is the eviction protection, because the premise
behind it changes when a host tier exists:

* device-only: an evicted anchor is a DEAD anchor, so `MambaRadixCache` spares
  the deepest anchors of every path -- "losing the deepest one silently moves
  the resume point of identical requests and re-introduces run-to-run drift"
  (`mamba_radix_cache.py:1092-1097`);
* host tier present: an evicted anchor still matches and loads back
  (`mamba_component.py:71-74`, `:139-144`), so it is not lost and the
  protection is not needed to keep the resume point stable.

That relaxation is deliberate and is implemented as an explicit branch, never
as a silent difference -- both directions are pinned below.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.mamba_ckpt_utils import (
    floor_to_interval,
    is_on_interval,
    is_resume_candidate,
    protect_deepest_anchors,
)
from sglang.test.test_utils import CustomTestCase

#: The cadence #747 documents: 8192 tokens = 16 x chunked_prefill_size(512).
CADENCE = 8192
CHUNK = 512


class TestTheEvictionProtectionBranch(CustomTestCase):
    """The new decision. Both directions pinned."""

    def test_device_only_keeps_todays_protection(self):
        self.assertTrue(protect_deepest_anchors(CADENCE, host_tier_present=False))

    def test_a_host_tier_relaxes_it(self):
        self.assertFalse(protect_deepest_anchors(CADENCE, host_tier_present=True))

    def test_without_an_interval_a_device_only_pool_still_protects(self):
        """#767 CORRECTED THE PREMISE THIS TEST ENCODED.

        It used to assert `protect = interval is not None` -- "without the grid
        there are no anchors to protect". There are: the no_buffer path donates
        a checkpoint for every finished request at `cache_len = len(token_ids)`,
        grid or no grid, and evict_mamba's own docstring says losing the deepest
        one "silently moves the resume point of identical requests and
        re-introduces run-to-run drift". Measured on the short-prompt gate at 48
        slots: 1 distinct over 10 probes idle, 2 over 20, 7 under 4-way load.

        The HOST-TIER half of the branch is unchanged and still asserted below:
        a spilled anchor stays matchable and reloads, so it may go."""
        self.assertTrue(protect_deepest_anchors(None, host_tier_present=False))
        self.assertFalse(protect_deepest_anchors(None, host_tier_present=True))

    def test_the_relaxation_is_documented_where_it_happens(self):
        """A silent relaxation is the failure mode; the reason must sit at the
        decision, not only in a note."""
        src = inspect.getsource(protect_deepest_anchors)
        low = src.lower()
        self.assertIn("host", low)
        self.assertIn("load", low)
        self.assertTrue(
            "drift" in low or "resume point" in low,
            "must say WHAT the protection preserves",
        )


class TestRetentionRefusesOffGrid(CustomTestCase):
    """Both retention sites REFUSE an off-grid length rather than floor it --
    `mamba_radix_cache.py:654` and `:796`. Rounding the key down would pair a
    deeper state with a shorter key, which is silent corruption."""

    def test_on_grid_lengths_are_retainable(self):
        for k in range(1, 6):
            with self.subTest(pos=k * CADENCE):
                self.assertTrue(is_on_interval(k * CADENCE, CADENCE))

    def test_off_grid_lengths_are_not(self):
        for pos in (1, CADENCE - 1, CADENCE + 1, 3 * CADENCE + 7):
            with self.subTest(pos=pos):
                self.assertFalse(is_on_interval(pos, CADENCE))

    def test_flooring_is_NOT_the_retention_rule(self):
        """Guards against a future 'fix' that floors instead of refusing."""
        pos = CADENCE + 7
        self.assertEqual(floor_to_interval(pos, CADENCE), CADENCE)
        self.assertFalse(is_on_interval(pos, CADENCE))

    def test_the_interval_is_a_multiple_of_the_prefill_chunk(self):
        """#747 cadence contract: anchors can only exist at chunk boundaries,
        so the grid must be a multiple of the chunk size."""
        self.assertEqual(CADENCE % CHUNK, 0)
        self.assertEqual(CADENCE // CHUNK, 16)


class TestTheGridIsOffByDefault(CustomTestCase):
    def test_interval_none_accepts_every_position(self):
        for pos in (0, 1, 7, 512, 8191, 8192):
            with self.subTest(pos=pos):
                self.assertTrue(is_on_interval(pos, None))

    def test_interval_none_floors_to_identity(self):
        for pos in (0, 1, 7, 8191):
            with self.subTest(pos=pos):
                self.assertEqual(floor_to_interval(pos, None), pos)


class TestBothLineagesCallTheSameRule(CustomTestCase):
    """Seam parity by construction: neither lineage may re-implement the rule.
    This is the pin that would have caught the original drift."""

    def test_mamba_radix_cache_imports_the_shared_rules(self):
        from sglang.srt.mem_cache import mamba_radix_cache

        src = inspect.getsource(mamba_radix_cache)
        self.assertIn("from sglang.srt.mem_cache.mamba_ckpt_utils import", src)

    def test_neither_lineage_reimplements_the_modulo(self):
        """`pos % interval == 0` must appear only in the shared helper."""
        from sglang.srt.mem_cache import mamba_ckpt_utils, mamba_radix_cache
        from sglang.srt.mem_cache.unified_cache_components import mamba_component

        for mod in (mamba_radix_cache, mamba_component):
            with self.subTest(module=mod.__name__):
                src = inspect.getsource(mod)
                self.assertNotIn("% self.mamba_checkpoint_interval", src)
        self.assertIn("% interval", inspect.getsource(mamba_ckpt_utils))

    def test_the_device_lineage_calls_the_eviction_rule(self):
        """Pin the CALL inside the evicting function, not the import -- an
        import alone satisfied two pins earlier today."""
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

        src = inspect.getsource(MambaRadixCache.evict_mamba)
        self.assertIn("protect_deepest_anchors(", src)

    def test_the_unified_branching_seam_uses_the_checkpoint_grid(self):
        """mamba_radix_cache.py:1600 prefers the interval over the chunk size;
        the unified component must make the same choice, or a configured grid
        is silently ignored on that lineage -- the #747 defect itself."""
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        src = inspect.getsource(MambaComponent.finalize_match_result)
        self.assertIn("mamba_checkpoint_interval", src)
        self.assertIn("floor_to_interval(", src)


class TestTheMatchSeamAnchorRule(CustomTestCase):
    """Seam 3 (match gating): a node is a resume anchor iff it CARRIES a
    state AND SITS on the grid -- the compound decision both walks make
    (`mamba_radix_cache.py:1521`/`:1542`; unified validator). One pure
    function, both directions pinned."""

    def test_on_grid_device_anchor_is_a_candidate(self):
        self.assertTrue(is_resume_candidate(CADENCE, CADENCE, has_device_value=True))

    def test_off_grid_device_anchor_is_refused(self):
        """The gate itself: a state at a non-multiple position must never be
        a resume point, or resume points follow the traffic-dependent split."""
        for depth in (1, CHUNK, CADENCE - 1, CADENCE + 1, 3 * CADENCE + CHUNK):
            with self.subTest(depth=depth):
                self.assertFalse(
                    is_resume_candidate(depth, CADENCE, has_device_value=True)
                )

    def test_no_state_is_never_a_candidate(self):
        for depth in (0, CADENCE):
            with self.subTest(depth=depth):
                self.assertFalse(
                    is_resume_candidate(depth, CADENCE, has_device_value=False)
                )

    def test_host_backup_counts_only_when_host_matches_are_allowed(self):
        """HiCache seam: evicted + host-backed is a valid anchor for the
        consensus walk (device_only=False) but NOT for the device walk --
        mirrors `create_match_validator`'s two variants."""
        kw = dict(has_device_value=False, has_host_value=True)
        self.assertFalse(is_resume_candidate(CADENCE, CADENCE, device_only=True, **kw))
        self.assertTrue(is_resume_candidate(CADENCE, CADENCE, device_only=False, **kw))

    def test_off_grid_host_anchor_is_refused_too(self):
        """The grid gates HOST anchors as well: storage populated by a
        non-interval run must not re-introduce off-grid resume points."""
        self.assertFalse(
            is_resume_candidate(
                CADENCE + CHUNK,
                CADENCE,
                has_device_value=False,
                has_host_value=True,
                device_only=False,
            )
        )

    def test_no_interval_is_presence_only(self):
        """Unset stays byte-identical: without a grid the rule degenerates to
        exactly the presence test both walks used before #747."""
        for depth in (0, 1, 7, CADENCE - 1):
            with self.subTest(depth=depth):
                self.assertTrue(
                    is_resume_candidate(depth, None, has_device_value=True)
                )
                self.assertFalse(
                    is_resume_candidate(depth, None, has_device_value=False)
                )


class TestBothWalksCallTheAnchorRule(CustomTestCase):
    """Seam parity by construction, matching the eviction seam's pattern:
    the compound decision must not be re-implemented per lineage."""

    def test_the_device_walk_calls_it(self):
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

        src = inspect.getsource(MambaRadixCache._match_prefix_helper)
        self.assertIn("is_resume_candidate(", src)

    def test_the_unified_validator_calls_it(self):
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        src = inspect.getsource(MambaComponent.create_match_validator)
        self.assertIn("is_resume_candidate(", src)

    def test_the_unified_walk_tracks_absolute_depth(self):
        """The unified walk never knew a node's absolute token depth -- that
        is WHY its component could not gate on the grid. Pin the depth
        accumulator so a refactor cannot silently drop it."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache._match_prefix_helper)
        self.assertIn("cum_tokens", src)


class TestUnifiedValidatorBehaviour(CustomTestCase):
    """The REAL validator off `MambaComponent` (the 639b test's technique),
    driven through both grid directions."""

    @staticmethod
    def _component(interval):
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        comp = object.__new__(MambaComponent)
        comp.mamba_checkpoint_interval = interval
        return comp

    @staticmethod
    def _node(value=None, host_value=None):
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            ComponentType,
        )

        return SimpleNamespace(
            component_data={
                ComponentType.MAMBA: SimpleNamespace(
                    value=value, host_value=host_value
                )
            }
        )

    def test_device_validator_gates_on_the_grid(self):
        v = self._component(CADENCE).create_match_validator(match_device_only=True)
        anchored = self._node(value="mamba-state")
        self.assertTrue(v(anchored, CADENCE))
        self.assertFalse(v(anchored, CADENCE - 1))
        self.assertFalse(v(self._node(), CADENCE))

    def test_consensus_validator_accepts_host_backed_on_grid_anchors(self):
        v = self._component(CADENCE).create_match_validator(match_device_only=False)
        spilled = self._node(host_value="host-backup")
        self.assertTrue(v(spilled, CADENCE))
        self.assertFalse(v(spilled, CADENCE + CHUNK))

    def test_device_validator_never_accepts_host_only_nodes(self):
        v = self._component(CADENCE).create_match_validator(match_device_only=True)
        self.assertFalse(v(self._node(host_value="host-backup"), CADENCE))

    def test_no_interval_keeps_the_old_presence_semantics(self):
        v = self._component(None).create_match_validator(match_device_only=True)
        self.assertTrue(v(self._node(value="mamba-state"), 7))
        self.assertFalse(v(self._node(), 7))


class TestRetentionAndCacheLenSeams(CustomTestCase):
    """Seams 4-5: `prepare_for_caching_req` is where the unified tree decides
    the retained length (the device lineage's :626-640 extra_buffer arm and
    :652-659/:795-809 no_buffer arms in one method)."""

    def _prepare_src(self):
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        return inspect.getsource(MambaComponent.prepare_for_caching_req)

    def test_prepare_gates_the_cache_len_on_the_grid(self):
        src = self._prepare_src()
        self.assertIn("is_on_interval(", src)
        self.assertIn("mamba_checkpoint_interval", src)

    def test_prepare_refuses_and_never_floors(self):
        """Rounding the retained key down while donating a deeper state would
        pair state and key at different positions -- the corruption the
        device lineage's comment at :654-656 names. The retention seam must
        refuse (cache nothing), never floor."""
        self.assertNotIn("floor_to_interval(", self._prepare_src())

    def test_the_insert_backstop_gates_other_producers(self):
        """`prepare_for_caching_req` is not the only insert producer (session
        restore paths build InsertParams too). The commit site refuses an
        off-grid leaf so no producer can plant an off-grid anchor."""
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        src = inspect.getsource(MambaComponent.commit_insert_component_data)
        self.assertIn("is_on_interval(", src)
        self.assertIn("mamba_exist", src)


class TestStrictResumeMirror(CustomTestCase):
    """`SGLANG_MAMBA_CKPT_STRICT_RESUME` (`mamba_radix_cache.py:1590-1601`)
    must not become a silently-inert env on the unified lineage -- the #742
    defect class. It is mirrored where the chunk sums are exact."""

    def test_the_component_reads_the_flag_at_init(self):
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        src = inspect.getsource(MambaComponent.__init__)
        self.assertIn("SGLANG_MAMBA_CKPT_STRICT_RESUME", src)
        self.assertIn("mamba_checkpoint_interval", src)

    def test_finalize_zeroes_through_the_shared_helper(self):
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        src = inspect.getsource(MambaComponent.finalize_match_result)
        self.assertIn("strict_resume", src)
        self.assertIn("zero_match_result(", src)
        self.assertIn("floor_to_interval(", src)


if __name__ == "__main__":
    unittest.main()
