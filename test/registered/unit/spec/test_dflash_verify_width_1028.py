"""FLLIPER_ADAPTIVE_K worker glue (dflash_verify_width), CPU with a dummy target.

Self-test of the task: at the forced/fallback width k = B the verify-side
tensors and the token decisions are bit-identical to the pre-adaptive round;
below B the committed tokens are a prefix of what the full verify commits.
"""

import contextlib
import unittest
from types import SimpleNamespace

import torch

from flliper.srt.environ import envs
from flliper.srt.speculative.dflash_utils import compute_dflash_correct_drafts_and_bonus
from flliper.srt.speculative.dflash_verify_width import (
    DFlashVerifyWidth,
    VerifyWidthState,
    pad_out_tokens,
    verify_views,
)
from flliper.srt.speculative.dflash_worker_v2 import _commit_accept
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

B = 8
BS = 5


def _block(seed: int):
    g = torch.Generator().manual_seed(seed)
    draft = torch.randint(0, 50, (BS, B), generator=g)
    # Dummy causal target: agrees with the draft for a random prefix per row.
    target = torch.randint(50, 100, (BS, B), generator=g)
    agree = torch.randint(0, B, (BS,), generator=g)
    for r in range(BS):
        target[r, : agree[r]] = draft[r, 1 : agree[r] + 1]
    positions = torch.arange(BS * B).view(BS, B) + 1000
    cache_loc = torch.arange(BS * B).view(BS, B) * 3
    return draft, target, positions, cache_loc


def _round(draft, target, width):
    """One verify round as the worker runs it; returns the result stride-B ids."""
    vv = verify_views(draft_tokens=draft, positions_2d=torch.zeros_like(draft),
                      cache_loc_2d=torch.zeros_like(draft), width=width)
    predict = target[:, :width].contiguous()  # causal: logits of slot i see slots <= i
    accept, bonus = compute_dflash_correct_drafts_and_bonus(candidates=vv.candidates, target_predict=predict)
    out, commit = _commit_accept(vv.candidates, accept, bonus)
    return pad_out_tokens(out_tokens=out, draft_tokens=draft), commit


class TestBitIdentityAtBlock(unittest.TestCase):
    def test_width_b_tensors_are_todays(self):
        """k = B: the very tensors the pre-adaptive code built (same storage,
        same values), so verify forward and accept see identical inputs."""
        draft, _, positions, cache_loc = _block(1)
        vv = verify_views(draft_tokens=draft, positions_2d=positions, cache_loc_2d=cache_loc, width=B)
        self.assertIs(vv.candidates, draft)
        self.assertIs(vv.cache_loc_2d, cache_loc)
        for got, old in ((vv.input_ids, draft.reshape(-1)), (vv.positions, positions.reshape(-1)),
                         (vv.cache_loc, cache_loc.reshape(-1))):
            self.assertEqual(got.data_ptr(), old.data_ptr())
            self.assertTrue(torch.equal(got, old))

    def test_width_b_token_decisions_bit_identical(self):
        """Old path (candidates = draft_tokens, out_tokens.reshape(-1)) vs the
        adaptive path forced to B: identical committed ids and lengths."""
        for seed in range(20):
            draft, target, _, _ = _block(seed)
            accept, bonus = compute_dflash_correct_drafts_and_bonus(candidates=draft, target_predict=target)
            old_out, old_commit = _commit_accept(draft, accept, bonus)
            new_out, new_commit = _round(draft, target, B)
            self.assertTrue(torch.equal(new_out.reshape(-1), old_out.reshape(-1)))
            self.assertTrue(torch.equal(new_commit, old_commit))


class TestPrefixBelowBlock(unittest.TestCase):
    def test_narrow_verify_commits_a_prefix_of_the_full_verify(self):
        """Derived property that makes k < B safe: with a causal target, the
        width-k round commits min(commit_B, k) tokens and they equal the first
        ones of the full round; the result keeps stride B."""
        for seed in range(20):
            draft, target, _, _ = _block(seed)
            full_out, full_commit = _round(draft, target, B)
            for k in (2, 5, 7):
                out, commit = _round(draft, target, k)
                self.assertEqual(tuple(out.shape), (BS, B))
                self.assertTrue(torch.equal(commit, torch.clamp(full_commit, max=k)))
                for r in range(BS):
                    n = int(commit[r])
                    # even the k-round's bonus (target slot k-1) is the token
                    # the full round commits at that slot
                    self.assertTrue(torch.equal(out[r, :n], full_out[r, :n]))


@contextlib.contextmanager
def _env(**kv):
    with contextlib.ExitStack() as st:
        for name, value in kv.items():
            st.enter_context(getattr(envs, name).override(value))
        yield


class TestRuntime(unittest.TestCase):
    def _rt(self):
        with _env(FLLIPER_ADAPTIVE_K=True, FLLIPER_ADAPTIVE_K_CURVE="", FLLIPER_ADAPTIVE_K_WARMUP=0,
                  FLLIPER_ADAPTIVE_K_PROBE_EVERY=0, FLLIPER_ADAPTIVE_K_MIN_DWELL=0,
                  FLLIPER_ADAPTIVE_K_MARGIN_PCT=0.0, FLLIPER_ADAPTIVE_K_CENSUS_EVERY=0):
            rt = DFlashVerifyWidth(block_size=B, tp_rank=0)
        boot = object()
        rt.boot_runner = boot
        rt.states = {5: VerifyWidthState(width=5, attn_backend="be5", graph_runner="gr5")}
        mr = SimpleNamespace(attn_backend="be8", decode_cuda_graph_runner=boot)
        return rt, mr, boot

    def test_swap_and_exact_restore(self):
        """Bundled A1 curve: bs3 -> width 5 with its state swapped in; restore
        puts back exactly the replaced objects (prefill/flip see boot state)."""
        rt, mr, boot = self._rt()
        self.assertEqual(rt.begin_round(target_model_runner=mr, batch_size=3, eligible=True), 5)
        self.assertEqual((mr.attn_backend, mr.decode_cuda_graph_runner), ("be5", "gr5"))
        rt.restore(target_model_runner=mr)
        self.assertEqual(mr.attn_backend, "be8")
        self.assertIs(mr.decode_cuda_graph_runner, boot)

    def test_fallbacks_stay_block(self):
        """Ineligible batch (grammar/logprobs) and a replaced boot runner
        (flip/sleep recapture -> our graphs may be stale) both run B and
        touch nothing."""
        rt, mr, _ = self._rt()
        self.assertEqual(rt.begin_round(target_model_runner=mr, batch_size=3, eligible=False), B)
        mr.decode_cuda_graph_runner = object()
        self.assertEqual(rt.begin_round(target_model_runner=mr, batch_size=3, eligible=True), B)
        self.assertEqual(mr.attn_backend, "be8")
        self.assertEqual((rt.ineligible_ct, rt.stale_ct), (1, 1))

    def test_learning_lags_exactly_two_rounds(self):
        """Round N folds in rounds <= N-2 regardless of timing -- the property
        that keeps every rank's state identical without a broadcast."""
        rt, mr, _ = self._rt()
        seen = []
        for _ in range(4):
            w = rt.begin_round(target_model_runner=mr, batch_size=3, eligible=True)
            seen.append(rt.controller.state.ema.observed_ct[0])
            rt.record(commit_lens=torch.tensor([2, 2, 2], dtype=torch.int32), width=w)
            rt.restore(target_model_runner=mr)
        self.assertEqual(seen, [0, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()
