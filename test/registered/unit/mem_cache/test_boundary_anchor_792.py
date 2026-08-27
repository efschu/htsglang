# Copyright 2023-2024 SGLang Team
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
"""#792/#869b: the boundary anchor must be ACKED before the cutover drops it.

WHAT #869b TURNED OUT TO BE. The composite boot refused every HiCache
read-through hit with ``why=MambaComponent:absent``. The cause is
``--mamba-checkpoint-interval 8192`` acting as a STORE REFUSAL, not a resume
policy: ``prepare_for_caching_req`` declines the anchor whenever the request
end misses the grid, and the tree is inserted CHUNK-WISE so no walk ever
reaches past ``chunked_prefill_size`` (4096 measured). Every reachable node
therefore sits below the first grid point and the tree is anchor-free.

THE REPAIR IS THE FLAG, NOT A REWRITE. ``interval=None`` already "degenerates
to the pure presence test, byte-identical to the pre-#747 behaviour"
(``is_resume_candidate``) -- which is upstream's semantics: a state per radix
node, host-backed states matchable, no grid. So the fix is to stop passing the
interval, and this file pins the two things that must hold in that world.

1. THE GEFAHRRICHTUNG SURVIVES THE PRESENCE TEST. Presence is exactly "there is
   a proven state at this node", so dropping the grid never serves a hit
   without one. Pinned below in both directions, including the host/device
   distinction that must not be relaxed along with it.

2. THE ANCHOR MUST OUTLIVE THE CUTOVER. #924 traced the second releaser:
   ``_drop_tree`` -> ``drop_prefix_tree_returning_rows`` -> ``tree.evict`` ->
   ``_evict_component_and_detach_lru`` -> ``mamba_component.evict_component``
   -> ``_free_mamba_value``. The flip evicts the prefix tree INCLUDING its
   mamba values, and ``reset_tree()`` runs on the line right after the
   retraction that donates them. So a boundary anchor is only worth anything if
   its host write-through is ACKED first -- otherwise it dies in the very
   cutover it was built for.

Hermetic: predicate level plus source-level wiring pins. No CUDA, no pools.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import inspect
import unittest

from sglang.srt.mem_cache.mamba_ckpt_utils import (
    RESUME_REFUSAL_ABSENT,
    is_resume_candidate,
    resume_refusal_reason,
)
from sglang.test.test_utils import CustomTestCase


class TestPresenceIsStillTheGefahrrichtung(CustomTestCase):
    """Without a grid, "is this a valid anchor" is "does a state exist here".
    That is the whole safety property, and it must not soften."""

    def test_no_state_is_never_an_anchor_without_a_grid(self):
        """The #767 direction. If this ever passes, a match resumes from a
        recurrent state that does not exist."""
        for depth in (0, 4, 45, 49, 2215, 4096):
            with self.subTest(depth=depth):
                self.assertFalse(
                    is_resume_candidate(depth, None, has_device_value=False)
                )
                self.assertEqual(
                    resume_refusal_reason(depth, None, has_device_value=False),
                    RESUME_REFUSAL_ABSENT,
                )

    def test_a_host_only_state_is_refused_on_a_device_walk(self):
        """Dropping the grid must not drop the device/host distinction: a
        device-only walk resuming from a host copy reads a slot that holds
        someone else's state."""
        self.assertFalse(
            is_resume_candidate(
                45, None, has_device_value=False, has_host_value=True, device_only=True
            )
        )

    def test_a_host_backed_state_matches_on_the_consensus_walk(self):
        """The upstream property this repair restores: evicted-but-backuped is
        still a match, and triggers load_back."""
        self.assertTrue(
            is_resume_candidate(
                45, None, has_device_value=False, has_host_value=True, device_only=False
            )
        )

    def test_a_resident_state_matches_at_any_position(self):
        """The depths the 0827 census actually reached. Under interval 8192
        every one of these was refused; under presence every one is an anchor,
        which is what ends the recompute loop."""
        for depth in (4, 45, 49, 2215, 4096):
            with self.subTest(depth=depth):
                self.assertTrue(
                    is_resume_candidate(depth, None, has_device_value=True)
                )


class TestTheAnchorIsAckedBeforeTheDrop(CustomTestCase):
    """#792 at the one boundary the flip seam owns."""

    def test_the_retract_forces_the_anchor_into_the_canonical_store(self):
        """Retraction already donates the state (``release_req`` ->
        ``cache_finished_req``); the stamp is what stops the hit-count
        write-through heuristic from leaving it device-only."""
        from sglang.srt.managers import phase_flip_runtime

        src = inspect.getsource(phase_flip_runtime.build_cutover_release)
        self.assertIn("FORCE_HOST_WRITE_THROUGH_ATTR", src)

    def test_a_writeback_fence_runs_between_the_retract_and_the_drop(self):
        """The ordering #924 makes load-bearing. The fence must sit in the
        RETRACT callable -- `release_residents_for_cutover` calls
        `reset_tree()` immediately after it returns, and that is what frees the
        mamba values."""
        from sglang.srt.managers import phase_flip_runtime

        src = inspect.getsource(
            phase_flip_runtime.PhaseFlipRuntime._release_residents_for_cutover
        )
        self.assertIn("maybe_flip_writeback", src)

    def test_the_fence_cannot_take_the_cutover_down(self):
        """Past the no-return point a raise kills the flip, and a lost anchor
        only costs a recompute -- so this one is best-effort by design."""
        from sglang.srt.managers import phase_flip_runtime

        src = inspect.getsource(
            phase_flip_runtime.PhaseFlipRuntime._release_residents_for_cutover
        )
        fence_at = src.index("maybe_flip_writeback")
        self.assertIn("except Exception", src[fence_at - 2000 : fence_at + 2000])

    def test_the_order_is_retract_then_reset(self):
        """The property the fence placement depends on."""
        from sglang.srt.managers import phase_flip_runtime

        src = inspect.getsource(phase_flip_runtime.release_residents_for_cutover)
        self.assertLess(src.index("retract(reqs)"), src.index("reset_tree()"))


if __name__ == "__main__":
    unittest.main()
