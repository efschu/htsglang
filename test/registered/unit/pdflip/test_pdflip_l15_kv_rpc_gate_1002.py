# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-KVRPC-GATE (LCWAKE2 finding, confirmed on the N3l D log): the
L1.5 fence-tail block runs ONLY in the resume RPC that carries kv_cache.

A D wake is TWO resume RPCs (N3l D log 02:29:42, TP0): first
"PDFLIP-GROUP-FENCE resume tags=['weights_6', ..., 'weights']" -> L15-DECIDE +
L15-RESTORE, THEN "PDFLIP-GROUP-FENCE resume tags=['kv_cache', 'cuda_graph']"
-> a second L15-DECIDE + L15-RESTORE, and the hold-aware restore
(PDFLIP-WAKE-RESTORE) sits only in the kv RPC. The fence tail of the WEIGHTS
RPC ran with no stashed manifest yet, so _l15_fence_manifest fell through to
load_for_wake, which READS AND UNLINKS the per-rank manifest file -- the kv
RPC's restore then finds no file, takes the plain restore (pools cleared,
keep set cleared, no re-reservation) while the sleep's reset_keep left the
tree reduced to the held chains: the tree names unreserved rows (F2-class
garbage prefix hit / #924-class aliasing), group-uniform, wake "succeeds".
N3l never got that far (it died at the sleep), N3n would have.
"""

from __future__ import annotations

import pathlib

from flliper.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager as W,
)

_WU = (pathlib.Path(__file__).resolve().parents[4] / "python" / "flliper" /
       "srt" / "managers" / "scheduler_components" / "weight_updater.py")


def test_gate_is_kv_rpc_only():
    assert W._l15_wake_rpc(["kv_cache", "cuda_graph"]) is True
    assert W._l15_wake_rpc({"kv_cache"}) is True
    assert W._l15_wake_rpc(["weights_6", "weights_0", "weights",
                            "weights_draft"]) is False
    assert W._l15_wake_rpc([]) is False
    assert W._l15_wake_rpc(None) is False


def test_fence_tail_block_is_gated_by_the_kv_rpc():
    src = _WU.read_text()
    assert ('if pdflip_memory_saver_on and self._pdflip_group_name() == "D" '
            'and self._l15_wake_rpc(tags):') in src, (
        "the L15 fence-tail block must only run in the kv_cache resume RPC")
