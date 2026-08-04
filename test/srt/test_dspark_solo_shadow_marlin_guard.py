# SPDX-License-Identifier: Apache-2.0
"""#470: a solo SHADOW rank must not be refused for the draft host's MoE backend.

Falsifier for a comment-vs-code contradiction found on hardware during the
2026-08-04 DSV4F window. `_refuse_unsupported_speculative_moe_backend` says in
its own docstring:

    Deliberately per-rank: a heterogeneous group (this rig is sm120 + 2x sm86)
    has different answers on different ranks, and only the ranks that actually
    build draft weights are affected -- a solo SHADOW builds on the ``meta``
    device and never reaches a kernel.

The predicate that follows tests only `torch.cuda.is_available()`,
`backend.is_marlin()` and SM support. It has no shadow check at all, and
`build_draft_tp_worker` calls it unconditionally. So on this rig
(5090 = cuda:0 = SM120, two 3080s = SM86) the documented configuration

    --speculative-draft-placement solo --speculative-draft-gpu 0
    --speculative-moe-runner-backend marlin

is unreachable: the two 3080 shadow ranks raise ValueError during init and the
server never comes up. The guard's own error message recommends exactly those
flags, so it refuses the fix it prescribes.

Observed: Boot B of TICKET_470, arm 470_b_dspark, both 3080 ranks raising
"marlin is not runnable on this rank's GPU (compute capability 8.6 ...)".

This test fails before the fix and passes after it.
"""

from __future__ import annotations

import unittest
from unittest import mock

from sglang.srt.speculative.draft_worker_common import (
    _refuse_unsupported_speculative_moe_backend,
)


class _Backend:
    def __init__(self, marlin: bool):
        self._marlin = marlin

    def is_marlin(self) -> bool:
        return self._marlin


class _Args:
    """Only the fields the guard reads."""

    def __init__(self, placement, solo_rank):
        self.speculative_draft_placement = placement
        self._solo_rank = solo_rank

    def speculative_draft_solo_rank(self):
        if self._solo_rank is None:
            raise AttributeError("no solo rank resolvable")
        return self._solo_rank


def _run_guard(*, placement, solo_rank, tp_rank, sm_ok):
    """Call the guard as an SM86 card with the marlin backend selected."""
    mod = "sglang.srt.speculative.draft_worker_common"
    with mock.patch(f"{mod}.torch") as t, mock.patch(
        "sglang.srt.layers.moe.utils.get_speculative_moe_runner_backend",
        return_value=_Backend(True),
    ), mock.patch(
        "sglang.srt.utils.common.is_sm90_supported", return_value=sm_ok
    ), mock.patch(
        "sglang.srt.utils.common.is_sm120_supported", return_value=sm_ok
    ):
        t.cuda.is_available.return_value = True
        t.cuda.get_device_capability.return_value = (8, 6)
        t.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        _refuse_unsupported_speculative_moe_backend(
            server_args=_Args(placement, solo_rank),
            algo_label="DSPARK",
            tp_rank=tp_rank,
        )


class TestSoloShadowMarlinGuard(unittest.TestCase):
    def test_solo_shadow_rank_is_exempt(self):
        """The documented heterogeneous config must be reachable.

        tp_rank=1 is a 3080 shadow; the draft lives solo on rank 0. It builds
        on ``meta`` and never reaches a Marlin kernel, so refusing it makes
        the guard's own recommended configuration impossible.
        """
        _run_guard(placement="solo", solo_rank=0, tp_rank=1, sm_ok=False)

    def test_solo_host_on_incapable_card_still_refused(self):
        """The guard must still bite where it matters -- can-fail arm.

        If the solo host itself is the SM86 card, the draft really would die
        in process_weights_after_loading, so the refusal must stand.
        """
        with self.assertRaises(ValueError) as ctx:
            _run_guard(placement="solo", solo_rank=1, tp_rank=1, sm_ok=False)
        self.assertIn("marlin is not runnable", str(ctx.exception))

    def test_non_solo_placement_still_refused(self):
        """Without solo placement every rank builds real draft weights."""
        with self.assertRaises(ValueError) as ctx:
            _run_guard(placement=None, solo_rank=None, tp_rank=1, sm_ok=False)
        self.assertIn("marlin is not runnable", str(ctx.exception))

    def test_capable_card_never_refused(self):
        """Sanity: an SM120 host is fine either way."""
        _run_guard(placement="solo", solo_rank=0, tp_rank=0, sm_ok=True)


if __name__ == "__main__":
    unittest.main()
