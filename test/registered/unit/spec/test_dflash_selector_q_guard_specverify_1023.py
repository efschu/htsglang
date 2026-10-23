"""27b-specverify-1023: the DFlash2 selector's sampling verify must stay EXACT
whatever q the draft hands over.

Boot dkr27bint4arbar1int4ar10231552 (rc35i4ar2, ctx 131072): the draft read
undefined RoPE cos/sin after every wake (accept_len 1.03) and long SAMPLED
answers carried foreign tokens / degenerated, while greedy probes stayed clean.
A bad draft may only lower acceptance; it must never change the output
distribution. The classic verify (``u * q(x) < p(x)``, residual ``relu(p - q)``)
holds that only for a valid q row: with a NaN row it rejects every draft AND
removes every selector candidate from the output -- the target's top token
never comes out once the draft listed it.

Everything runs on CPU through the REAL Triton kernels (TRITON_INTERPRET=1,
set before the first triton import of this process): the selector walk
(kernels/ops/speculative/dflash.py) and the chain verify
(speculative/reject_sampling.py via dspark_accept.accept_sampling), driven
through ``DFlashWorkerV2._selector_sampling_accept``.
"""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import unittest
from types import SimpleNamespace

import torch

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=240, suite="base-a-test-cpu")

V, BLOCK, K = 64, 4, 8
GAMMA = BLOCK - 1


def _sampling_info(bs, temperature=1.0):
    return SimpleNamespace(
        temperatures=torch.full((bs, 1), float(temperature)),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        is_all_greedy=False,
    )


def _worker():
    from flliper.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    w = DFlashWorkerV2.__new__(DFlashWorkerV2)
    # Only what _selector_sampling_accept (and the guard's report) reads.
    for name, value in (
        ("_draft_probs_buf", None),
        ("_q_guard_stats", None),
        ("_q_guard_rounds", 0),
        ("_q_guard_last", None),
        ("tp_rank", 0),
    ):
        object.__setattr__(w, name, value)
    return w


def _draft_input():
    from flliper.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

    return DFlashDraftInputV2.create_idle_input(torch.device("cpu"))


def _first_token_hist(*, mode, bs, seed, guard):
    """Emitted first-token histogram over ``bs`` independent rows of one verify.

    Every row has the same target distribution p0 (row 0 of the block). The draft
    lists the target's top token among its slot-0 candidates; ``mode`` picks the
    selector scores: "valid" (random finite), "nan" (all NaN), "inf" (one +inf).
    """
    from flliper.kernels.ops.speculative.dflash import selector_walk_triton

    g = torch.Generator().manual_seed(seed)
    logit_row = torch.randn(V, generator=g) * 2.0
    p0 = torch.softmax(logit_row, dim=-1)
    top = int(p0.argmax())
    logits = logit_row.repeat(bs * BLOCK, 1)
    cand_ids = torch.stack(
        [
            torch.stack([torch.randperm(V, generator=g)[:K] for _ in range(GAMMA)])
            for _ in range(bs)
        ]
    )
    slot0 = cand_ids[:, 0, :]
    has_top = (slot0 == top).any(dim=-1)
    slot0[~has_top, 0] = top
    scores = torch.randn(bs, GAMMA, K, K, generator=g) * 3.0
    if mode == "nan":
        scores.fill_(float("nan"))
    elif mode == "inf":
        scores[:, :, :, 0] = float("inf")
    tokens, q_rows = selector_walk_triton(
        candidate_ids=cand_ids,
        scores=scores,
        uniforms=torch.rand(bs, GAMMA, generator=g),
        temperatures=torch.ones(bs),
        greedy_mask=torch.zeros(bs, dtype=torch.bool),
    )
    candidates = torch.cat([torch.full((bs, 1), 5, dtype=torch.int64), tokens], dim=1)
    os.environ["FLLIPER_DFLASH_Q_GUARD"] = "1" if guard else "0"
    try:
        w = _worker()
        torch.manual_seed(seed + 1)
        accept_len, bonus = w._selector_sampling_accept(
            candidates=candidates,
            next_token_logits=logits,
            candidate_ids=cand_ids,
            q_rows=q_rows,
            sampling_info=_sampling_info(bs),
            draft_input=_draft_input(),
        )
    finally:
        os.environ.pop("FLLIPER_DFLASH_Q_GUARD", None)
    first = torch.where(accept_len >= 1, candidates[:, 1], bonus)
    hist = torch.bincount(first, minlength=V).float() / bs
    return hist, p0, top, accept_len, w


class TestSelectorQGuardExactness(unittest.TestCase):
    BS = 768

    def _tv(self, hist, p0):
        return float(0.5 * (hist - p0).abs().sum())

    def test_valid_q_exact_and_unchanged(self):
        hist, p0, top, acc, _ = _first_token_hist(mode="valid", bs=self.BS, seed=11, guard=True)
        # sampling noise at bs=768, V=64: TV ~0.07
        self.assertLess(self._tv(hist, p0), 0.12)
        self.assertGreater(float((acc >= 1).float().mean()), 0.0)

    def test_nan_q_defect_without_guard(self):
        """The defect itself (documents WHY the guard exists): unchecked NaN q
        never emits the target's top token."""
        hist, p0, top, acc, _ = _first_token_hist(mode="nan", bs=self.BS, seed=12, guard=False)
        self.assertGreater(float(p0[top]), 0.2)
        self.assertEqual(float(hist[top]), 0.0)
        self.assertGreater(self._tv(hist, p0), 0.2)

    def test_nan_q_exact_with_guard(self):
        hist, p0, top, acc, w = _first_token_hist(mode="nan", bs=self.BS, seed=12, guard=True)
        self.assertLess(self._tv(hist, p0), 0.12)
        self.assertGreater(float(hist[top]), 0.5 * float(p0[top]))
        stats = dict(zip(("slots", "fallback", "nonfinite"), w._q_guard_stats.tolist()[:3]))
        self.assertEqual(stats["slots"], self.BS * GAMMA)
        self.assertEqual(stats["fallback"], self.BS * GAMMA)
        self.assertEqual(stats["nonfinite"], self.BS * GAMMA)

    def test_single_inf_score_exact_with_guard(self):
        hist, p0, top, _, _ = _first_token_hist(mode="inf", bs=self.BS, seed=13, guard=True)
        self.assertLess(self._tv(hist, p0), 0.12)


class TestSelectorQGuardScatter(unittest.TestCase):
    def _run(self, q, ids, x):
        from flliper.srt.speculative.dflash_utils import (
            SELECTOR_Q_GUARD_STATS,
            selector_q_guard_scatter,
        )

        dp = torch.zeros(1, q.shape[1], V)
        stats = torch.zeros(len(SELECTOR_Q_GUARD_STATS), dtype=torch.int64)
        fb = selector_q_guard_scatter(
            draft_probs=dp, candidate_ids=ids, q_rows=q, draft_tokens=x, stats=stats
        )
        return dp, fb, dict(zip(SELECTOR_Q_GUARD_STATS, stats.tolist()))

    def test_valid_row_bit_identical_to_plain_scatter(self):
        ids = torch.tensor([[[3, 9, 12, 40]]])
        q = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
        dp, fb, st = self._run(q, ids, torch.tensor([[12]]))
        ref = torch.zeros(1, 1, V).scatter_(-1, ids, q)
        self.assertTrue(torch.equal(dp, ref))
        self.assertFalse(bool(fb.any()))
        self.assertEqual(st["fallback"], 0)

    def test_invalid_rows_become_point_mass_at_drafted_token(self):
        ids = torch.tensor([[[3, 9, 12, 40], [3, 9, 12, 40], [3, 3, 12, 40], [3, 9, 12, 40], [1, 2, 5, 6]]])
        q = torch.tensor(
            [[[float("nan")] * 4, [0.1, 0.2, 0.3, 0.3], [0.1, 0.2, 0.3, 0.4], [0.0, 0.5, 0.5, 0.0], [0.25] * 4]]
        )
        x = torch.tensor([[9, 9, 12, 3, 7]])  # nan / off-sum / dup / q(x)=0 / not among ids
        dp, fb, st = self._run(q, ids, x)
        self.assertTrue(bool(fb.all()))
        for e in range(5):
            row = dp[0, e]
            self.assertEqual(float(row.sum()), 1.0)
            self.assertEqual(float(row[int(x[0, e])]), 1.0)
        self.assertEqual(
            (st["fallback"], st["nonfinite"], st["badsum"], st["notin"], st["dup"]),
            (5, 1, 1, 2, 1),
        )


class TestSelectorWalkNanGreedyStaysInSlot(unittest.TestCase):
    def test_greedy_nan_row_token_from_own_slot(self):
        from flliper.kernels.ops.speculative.dflash import selector_walk_triton

        bs, slots, k = 2, 3, 4
        ids = torch.arange(bs * slots * k, dtype=torch.int64).view(bs, slots, k) + 100
        scores = torch.full((bs, slots, k, k), float("nan"))
        tokens, q = selector_walk_triton(
            candidate_ids=ids,
            scores=scores,
            uniforms=torch.rand(bs, slots),
            temperatures=torch.ones(bs),
            greedy_mask=torch.ones(bs, dtype=torch.bool),
        )
        for b in range(bs):
            for s in range(slots):
                self.assertIn(int(tokens[b, s]), ids[b, s].tolist())


if __name__ == "__main__":
    unittest.main()
