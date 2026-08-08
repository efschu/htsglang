# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#622: falsifier for the first-token cross-rank sync in the base sampler.

THE BUG THIS PINS. Upstream's sampler skips the cross-rank token-id sync by
default; correctness rests on "the last all-reduce, the last lm_head matmul,
and all sampling kernels" being cross-rank deterministic. On this fork those
assumptions are violated three ways at once: mixed GPU architectures (near-tie
argmax flips), uneven-TP shard geometry (per-rank reduction order), and
per-rank sampling RNG at temperature > 0. The farm proof (2026-08-08, rounds
20260808T0538..0545): ranks read DIFFERENT first tokens for the same request
under barlink AND under NCCL, at temperature 0.7 AND 0.0, always surfacing at
output length 1 with a genuine EOS id (248046) on exactly one side — the only
single-token flip that changes batch membership, which then wedges the group
tens of thousands of replays later (#622/#649).

RED-FIRST. ``test_default_no_sync_leaves_divergence_visible`` demonstrates the
unfixed behavior (upstream default: divergent stays divergent). The fix makes
``maybe_sync_sampled_tokens`` broadcast rank 0's tokens by default under
tp > 1; ``test_fork_default_broadcasts_rank0`` is red on the unfixed tree
(function absent / default off) and green on the fixed one.
"""

import json
import os
import tempfile
import unittest

import torch
import torch.multiprocessing as mp

WORLD = 3


def _worker(rank: int, init_file: str, out_dir: str, mode: str) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
    )
    # Divergent per-rank samples for the same 4 requests: rank 0's row is
    # authoritative; rank 2 disagrees at position 1 (an "EOS on one rank"
    # first-token flip, the farm-proven injury).
    tokens = torch.tensor([11, 22, 33, 44], dtype=torch.int64)
    if rank == 2:
        tokens = torch.tensor([11, 248046, 33, 44], dtype=torch.int64)

    if mode == "fixed":
        from sglang.srt.layers.sampler import maybe_sync_sampled_tokens

        maybe_sync_sampled_tokens(tokens, group=dist.group.WORLD, src=0)
    # mode == "default": upstream default path — no sync at all.

    with open(os.path.join(out_dir, f"tokens_rank{rank}.json"), "w") as f:
        json.dump(tokens.tolist(), f)
    dist.destroy_process_group()


def _run(mode: str):
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        mp.spawn(_worker, args=(init_file, tmp, mode), nprocs=WORLD, join=True)
        out = []
        for r in range(WORLD):
            with open(os.path.join(tmp, f"tokens_rank{r}.json")) as f:
                out.append(json.load(f))
        return out


class TestSamplerTokenSync622(unittest.TestCase):
    def test_default_no_sync_leaves_divergence_visible(self):
        """The bug, demonstrated: without sync, rank 2 keeps its EOS flip."""
        toks = _run("default")
        self.assertNotEqual(toks[0], toks[2])
        self.assertEqual(toks[2][1], 248046)

    def test_fork_default_broadcasts_rank0(self):
        """The fix: every rank ends with rank 0's authoritative tokens."""
        toks = _run("fixed")
        self.assertEqual(toks[0], [11, 22, 33, 44])
        self.assertEqual(toks[1], toks[0])
        self.assertEqual(toks[2], toks[0])


if __name__ == "__main__":
    unittest.main()
