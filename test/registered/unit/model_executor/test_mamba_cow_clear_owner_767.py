# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#767: the deferred mamba clear/COW must run on the flip's TP TARGET.

THE DROPPED CLEAR. The phase-flip TP stack's TARGET worker is built with
``is_draft_worker=True`` (phase_flip_boot.py:710 -- deliberate, to inherit
draft-style init), and ``_maybe_execute_deferred_mamba_cow_and_clear``
early-returned on that raw flag. Every TP-phase extend therefore dropped
its pending clear/COW indices on the floor (metal 2026-08-19, boot
fix767t3: probe prefills logged ``SKIP: draft=True mode=1 clear=[5]`` and
fresh requests answered with the WARMUP requests' content -- banana,
4+5=9, the black test image -- out of uncleaned slots).

The truthful predicate exists: ``is_draft_model_runner`` (model_runner.py
:517) is False for the flip TP target and True for the real draft runner,
whose skip stays correct -- the real draft shares the target's pool and
runs AFTER the target wrote fresh state, so a re-clear there would zero
it.

Red-first against the raw-flag guard.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch


class _Pool:
    """Minimal HybridReqToTokenPool stand-in that records state ops."""

    def __init__(self):
        self.cleared = []
        self.cowed = []
        self.mamba_ckpt_pool = None
        self.mamba_pool = SimpleNamespace(
            clear_slots=lambda idx: self.cleared.append(idx.tolist()),
            copy_from=lambda src, dst: self.cowed.append(
                (src.tolist(), dst.tolist())
            ),
        )

    def translate_mamba_indices(self, idx):
        return idx


def _run(is_draft_worker, is_draft_model_runner):
    from sglang.srt.mem_cache import memory_pool
    from sglang.srt.model_executor.model_runner import ModelRunner

    pool = _Pool()
    # The executor type-checks the pool; register the stand-in.
    pool.__class__ = type(
        "_HybridPoolStandIn", (memory_pool.HybridReqToTokenPool,), {}
    )

    mode = SimpleNamespace(
        is_extend=lambda **kw: True,
        is_target_verify=lambda: False,
        is_draft_extend_v2=lambda: False,
    )
    batch = SimpleNamespace(
        forward_mode=mode,
        batch_size=1,
        mamba_clear_indices=torch.tensor([5]),
        mamba_cow_src_indices=None,
        mamba_cow_dst_indices=None,
        extend_prefix_lens_cpu=[0],
    )
    runner = SimpleNamespace(
        req_to_token_pool=pool,
        is_draft_worker=is_draft_worker,
        is_draft_model_runner=is_draft_model_runner,
    )
    ModelRunner._maybe_execute_deferred_mamba_cow_and_clear(runner, batch)
    return pool


class TestTheClearRunsOnTheFlipTpTarget(unittest.TestCase):
    def test_the_flip_tp_target_executes_its_pending_clear(self):
        # is_draft_worker=True but NOT the real draft runner: the flip TP
        # target. Its pending clear must execute.
        pool = _run(is_draft_worker=True, is_draft_model_runner=False)
        self.assertEqual(pool.cleared, [[5]])

    def test_a_plain_target_still_executes(self):
        pool = _run(is_draft_worker=False, is_draft_model_runner=False)
        self.assertEqual(pool.cleared, [[5]])

    def test_the_real_draft_runner_still_skips(self):
        # Shared pool, runs after the target wrote fresh state: a re-clear
        # here would zero it. The skip must survive the fix.
        pool = _run(is_draft_worker=True, is_draft_model_runner=True)
        self.assertEqual(pool.cleared, [])


if __name__ == "__main__":
    unittest.main()
