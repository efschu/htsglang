# SPDX-License-Identifier: Apache-2.0
"""#450b: who owns a row of the verify intermediate caches.

THE CLAIM UNDER TEST
``GDNAttnBackend.verify_intermediate_state_indices`` is
``arange(req_to_token_pool.size)`` sliced ``[:batch_size]``
(``layers/attention/linear/gdn_backend.py:337``, ``:566``, ``:604``), and
``MambaMixer2.forward`` builds ``arange(num_decodes)``
(``layers/attention/mamba/mamba.py:669``). Both are BATCH POSITIONS, not mamba
slots, and the commit reads them back by position too -- ``src_idx = pid_req``
in ``_fused_conv_window_scatter_with_mask_kernel``
(``mamba/mamba_state_scatter_triton.py:344``). The claim was that two lanes
verifying concurrently therefore collide on those rows, independently of the
conv window #444/#450a made private.

THE MECHANISM IS REAL, THE PRECONDITION IS NOT
``TestBatchPositionRowsCollideWhenTheCacheIsShared`` builds the collision and
measures it: two batches interleaved as verify(A), verify(B), commit(A),
commit(B) over ONE intermediate cache, and A commits B's window. Position
indexing gives both batches row 0, so the second verify overwrites the first's
un-committed intermediates. Nothing about the conv window prevents it -- the
private buffer of #444/#450a covers the *persistent* pool row, not the
intermediate cache.

But the two things that verify concurrently in this fork do not share that
cache. The #274 dual-group lane is a SECOND ``ModelRunner``, built without a
``req_to_token_pool`` argument and told to allocate its own
(``model_executor/dual_group_lane.py:_build_lane_under_scope``); ``ModelRunner``
only reuses a pool that was handed to it (``if self.req_to_token_pool is None``
in ``model_runner_kv_cache_mixin.py``), and the lane's args view even gives it
its own slot count (``_lane_server_args_view``: ``max_mamba_cache_size =
lane_requests + 1``). Its NEXTN head is held to the same rule in as many words
(``:5807``: "it must NOT share the target's req_to_token_pool"). Each runner
also builds its own attention backend, so ``verify_intermediate_state_indices``
and the #444 private conv window are per-runner objects as well. Within ONE
runner the target and its draft workers do share the pool, but they run in
sequence on one stream, so only one verify is ever in flight against it.

So the row space is position-keyed AND collision-free, for a reason that has
nothing to do with what the rows mean. That is a precondition, not a property,
and this file pins it:
``TestDisjointCachesSurviveTheSameInterleaving`` runs the identical schedule on
per-runner caches and shows it is clean;
``TestEachRunnerOwnsItsVerifyRowSpace`` pins the source-level facts the
precondition rests on;
``TestTheDetectorCanFail`` points both instruments at the wrong configuration
and shows they report it.

WHY NOT SLOT-KEYED ROWS
Keying the intermediate caches by mamba slot would make the rows self-owning
and remove the precondition. It does not fit: the caches are sized
``mamba_spec_state_size = max_num_reqs``
(``model_runner_kv_cache_mixin.py:3074``) while the slot space is
``max_mamba_cache_size``, and the runtime derives ``max_num_reqs =
max_mamba_cache_size // mamba_ratio`` (``:122``) -- slots outnumber spec rows by
the mamba ratio, so slot ids index the cache out of bounds. Making them fit
means growing ``intermediate_ssm`` + ``intermediate_conv_window`` by that same
ratio, which is the largest speculative scratch in the pool. A per-lane offset
partition has the same cost problem in miniature and buys nothing while the
caches are already per-runner. The smallest correct answer is therefore: keep
position keying, and pin the ownership rule that makes it correct.
"""

import inspect
import unittest
from typing import List, Optional

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DIM = 4
KERNEL_WIDTH = 4
WIN = KERNEL_WIDTH - 1
DRAFT_TOKENS = 3
BATCH = 2
POOL_ROWS = 8
SPEC_ROWS = 4
DTYPE = torch.float32


def _reference_conv_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
) -> torch.Tensor:
    """Same transcription as ``test_mamba2_conv_verify_private_window_450``:
    STEP 2 in-place window store plus the ``SAVE_INTERMEDIATE`` per-step
    window (``mamba/causal_conv1d_triton.py:705-752``, ``:930-960``)."""
    bs, dim, seqlen = x.shape
    win = conv_state.shape[-1]
    width = weight.shape[-1]
    out = torch.empty_like(x)
    for b in range(bs):
        row = int(conv_state_indices[b])
        slot = int(intermediate_state_indices[b])
        full = torch.cat([conv_state[row], x[b]], dim=-1)
        for t in range(seqlen):
            acc = torch.zeros(dim, dtype=x.dtype)
            for j in range(width):
                acc = acc + weight[:, j] * full[:, t + j]
            out[b, :, t] = acc
            intermediate_conv_window[slot, t] = full[:, t + 1 : t + 1 + win]
        conv_state[row] = full[:, seqlen:]
    return out


def _reference_commit(
    conv_states: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    dst_indices: torch.Tensor,
    step_indices: torch.Tensor,
) -> None:
    """``_fused_conv_window_scatter_with_mask_kernel``: destination row from
    ``dst_indices_raw``, SOURCE row from the request POSITION (``src_idx =
    pid_req``), ``step < 0`` skipped."""
    for pos in range(step_indices.shape[0]):
        step = int(step_indices[pos])
        if step < 0:
            continue
        conv_states[int(dst_indices[pos])] = intermediate_conv_window[pos, step]


class _VerifyLane:
    """One runner's verify, split at the seam the commit sits on.

    Models exactly the production shape after #444/#450a: the conv runs on a
    request-private window (per backend instance, hence per runner) while the
    intermediate window cache is the pool's, and the rows into BOTH are batch
    positions.
    """

    def __init__(
        self,
        name: str,
        conv_states: torch.Tensor,
        intermediate: torch.Tensor,
        slots: torch.Tensor,
        seed: int,
    ):
        self.name = name
        self.conv_states = conv_states
        self.intermediate = intermediate
        self.slots = slots
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(BATCH, DIM, DRAFT_TOKENS, generator=gen, dtype=DTYPE)
        self.weight = torch.randn(DIM, KERNEL_WIDTH, generator=gen, dtype=DTYPE)
        # The #444 / #450a private conv window: allocated by this runner's
        # attention backend, so never shared with another runner.
        self.private_window = torch.zeros(SPEC_ROWS, DIM, WIN, dtype=DTYPE)
        self.rows = torch.arange(BATCH, dtype=torch.int32)
        self.expected: Optional[List[torch.Tensor]] = None

    def verify(self) -> None:
        self.private_window.index_copy_(
            0,
            self.rows.to(torch.int64),
            self.conv_states.index_select(0, self.slots.to(torch.int64)),
        )
        _reference_conv_update(
            self.x,
            self.private_window,
            self.weight,
            conv_state_indices=self.rows,
            intermediate_conv_window=self.intermediate,
            intermediate_state_indices=self.rows,
        )
        # What an uncorrupted commit must land, recorded at the moment this
        # lane's own verify wrote it.
        self.expected = [
            self.intermediate[int(self.rows[pos])].clone() for pos in range(BATCH)
        ]

    def commit(self, accepted: torch.Tensor) -> None:
        _reference_commit(self.conv_states, self.intermediate, self.slots, accepted)

    def committed_matches_own_verify(self, accepted: torch.Tensor) -> bool:
        for pos in range(BATCH):
            want = self.expected[pos][int(accepted[pos])]
            got = self.conv_states[int(self.slots[pos])]
            if not torch.equal(want, got):
                return False
        return True


def _two_lanes(shared_cache: bool):
    """Two runners mid-verify. ``shared_cache`` is the counterfactual: the one
    thing that would have to be true for the claimed collision to exist."""
    gen = torch.Generator().manual_seed(11)
    pool_a = torch.randn(POOL_ROWS, DIM, WIN, generator=gen, dtype=DTYPE)
    pool_b = torch.randn(POOL_ROWS, DIM, WIN, generator=gen, dtype=DTYPE)
    cache_a = torch.zeros(SPEC_ROWS, DRAFT_TOKENS, DIM, WIN, dtype=DTYPE)
    cache_b = cache_a if shared_cache else torch.zeros_like(cache_a)
    lane_a = _VerifyLane("A", pool_a, cache_a, torch.tensor([5, 2]), seed=1)
    lane_b = _VerifyLane("B", pool_b, cache_b, torch.tensor([6, 3]), seed=2)
    return lane_a, lane_b


def _interleave(lane_a: _VerifyLane, lane_b: _VerifyLane, accepted: torch.Tensor):
    """The schedule that makes the row question observable: B's verify lands
    inside A's verify->commit seam, which is precisely the window #444 showed
    is not empty for a concurrent second worker."""
    lane_a.verify()
    lane_b.verify()
    lane_a.commit(accepted)
    lane_b.commit(accepted)


class TestBatchPositionRowsCollideWhenTheCacheIsShared(CustomTestCase):
    """The mechanism, measured. Position-keyed rows are not self-owning: two
    batches both start at row 0, so an interleaved verify destroys the other's
    un-committed intermediates."""

    def test_lane_a_commits_lane_b_window(self):
        accepted = torch.tensor([1, DRAFT_TOKENS - 1], dtype=torch.int64)
        lane_a, lane_b = _two_lanes(shared_cache=True)
        _interleave(lane_a, lane_b, accepted)

        self.assertFalse(
            lane_a.committed_matches_own_verify(accepted),
            "a shared intermediate cache did NOT corrupt the first lane -- the "
            "collision model is wrong and the rest of this file proves nothing",
        )
        # And specifically: what A committed is what B's verify wrote.
        for pos in range(BATCH):
            self.assertTrue(
                torch.equal(
                    lane_a.conv_states[int(lane_a.slots[pos])],
                    lane_b.expected[pos][int(accepted[pos])],
                ),
                f"row {pos}: A's committed state is neither its own nor B's",
            )

    def test_the_rows_are_positions_not_slots(self):
        """Why the collision exists at all: both lanes address row 0 first,
        although they own disjoint mamba slots."""
        lane_a, lane_b = _two_lanes(shared_cache=True)
        self.assertTrue(torch.equal(lane_a.rows, lane_b.rows))
        self.assertEqual(
            set(int(s) for s in lane_a.slots) & set(int(s) for s in lane_b.slots),
            set(),
            "the fixture's lanes must own disjoint slots, or the collision "
            "could be blamed on slot reuse instead of row keying",
        )

    def test_the_private_conv_window_does_not_help(self):
        """#444/#450a made the persistent pool row private; the collision is on
        the intermediate cache and survives it untouched."""
        accepted = torch.tensor([0, 0], dtype=torch.int64)
        lane_a, lane_b = _two_lanes(shared_cache=True)
        self.assertIsNot(lane_a.private_window, lane_b.private_window)
        before = lane_a.conv_states.clone()
        lane_a.verify()
        lane_b.verify()
        # The private window did its job: A's pool is untouched at the seam.
        self.assertTrue(torch.equal(lane_a.conv_states, before))
        # And the collision is still there once the commit runs.
        lane_a.commit(accepted)
        self.assertFalse(lane_a.committed_matches_own_verify(accepted))


class TestDisjointCachesSurviveTheSameInterleaving(CustomTestCase):
    """Production's configuration: one intermediate cache per runner. The
    identical schedule is clean."""

    def test_each_lane_commits_its_own_window(self):
        for accepted in (
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([1, DRAFT_TOKENS - 1], dtype=torch.int64),
            torch.tensor([DRAFT_TOKENS - 1, 0], dtype=torch.int64),
        ):
            with self.subTest(accepted=accepted.tolist()):
                lane_a, lane_b = _two_lanes(shared_cache=False)
                _interleave(lane_a, lane_b, accepted)
                self.assertTrue(lane_a.committed_matches_own_verify(accepted))
                self.assertTrue(lane_b.committed_matches_own_verify(accepted))

    def test_a_rejected_step_still_lands_nothing(self):
        """``step < 0`` (nothing accepted) must leave the pool row alone even
        while the other lane is mid-verify."""
        lane_a, lane_b = _two_lanes(shared_cache=False)
        before = lane_a.conv_states.clone()
        lane_a.verify()
        lane_b.verify()
        lane_a.commit(torch.tensor([-1, -1], dtype=torch.int64))
        self.assertTrue(torch.equal(lane_a.conv_states, before))


class TestEachRunnerOwnsItsVerifyRowSpace(CustomTestCase):
    """The precondition, pinned at its sources. Each of these is the reason a
    concurrent second verifier cannot reach the first's rows; if one goes red,
    position-keyed rows stop being safe and #450b turns from refuted into
    real."""

    def test_lane_runner_is_not_handed_the_host_pool(self):
        from sglang.srt.model_executor import dual_group_lane

        src = inspect.getsource(dual_group_lane._build_lane_under_scope)
        self.assertIn("ModelRunner(", src)
        self.assertNotIn("req_to_token_pool", src)
        self.assertIn("runner.alloc_memory_pool()", src)

    def test_runner_only_reuses_a_pool_it_was_handed(self):
        from sglang.srt.model_executor import model_runner_kv_cache_mixin

        src = inspect.getsource(model_runner_kv_cache_mixin)
        self.assertIn("if self.req_to_token_pool is None:", src)

    def test_lane_gets_its_own_mamba_slot_count(self):
        from sglang.srt.model_executor import dual_group_lane

        src = inspect.getsource(dual_group_lane._lane_server_args_view)
        self.assertIn("view.max_mamba_cache_size = lane_requests + 1", src)

    def test_verify_rows_and_private_window_are_per_backend_instance(self):
        """Neither the row tensor nor the #444/#450a private conv window may
        live at class or module scope -- that would reintroduce the sharing
        the separate pools are there to prevent."""
        from sglang.srt.layers.attention import hybrid_linear_attn_backend as hlab
        from sglang.srt.layers.attention.linear import gdn_backend as gdn

        for owner, names in (
            (
                gdn.GDNAttnBackend,
                ("verify_intermediate_state_indices", "_verify_conv_scratch"),
            ),
            (hlab.Mamba2AttnBackend, ("verify_conv_window",)),
        ):
            for name in names:
                with self.subTest(owner=owner.__name__, attr=name):
                    self.assertFalse(
                        hasattr(owner, name),
                        f"{owner.__name__}.{name} exists on the CLASS, so every "
                        "runner in the process shares one",
                    )
                    self.assertIn(
                        f"self.{name}",
                        inspect.getsource(owner.__init__),
                        f"{owner.__name__}.__init__ does not bind {name} per "
                        "instance",
                    )


class TestTheDetectorCanFail(CustomTestCase):
    """Both instruments pointed at the configuration they must reject.

    Without this the two suites above are unfalsifiable: a
    ``committed_matches_own_verify`` that always returned True would make
    ``TestDisjointCaches...`` green, and a class-attribute check that never
    looked at anything would make ``TestEachRunnerOwns...`` green.
    """

    def test_collision_detector_reports_clean_on_disjoint_caches(self):
        accepted = torch.tensor([1, DRAFT_TOKENS - 1], dtype=torch.int64)
        lane_a, lane_b = _two_lanes(shared_cache=False)
        _interleave(lane_a, lane_b, accepted)
        self.assertTrue(lane_a.committed_matches_own_verify(accepted))

    def test_collision_detector_reports_dirty_on_a_shared_cache(self):
        accepted = torch.tensor([1, DRAFT_TOKENS - 1], dtype=torch.int64)
        lane_a, lane_b = _two_lanes(shared_cache=True)
        _interleave(lane_a, lane_b, accepted)
        self.assertFalse(lane_a.committed_matches_own_verify(accepted))

    def test_ownership_check_rejects_a_class_level_buffer(self):
        class _Shared:
            verify_conv_window = torch.zeros(1)

            def __init__(self):
                pass

        self.assertTrue(hasattr(_Shared, "verify_conv_window"))
        self.assertNotIn("self.verify_conv_window", inspect.getsource(_Shared.__init__))


if __name__ == "__main__":
    unittest.main()
