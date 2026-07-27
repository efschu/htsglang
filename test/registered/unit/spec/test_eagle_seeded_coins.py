"""EAGLE verify coins: deterministic seeded draw (`_seeded_verify_coins`) and
the dispatch in front of it (`_verify_coins`).

Port of upstream sgl-project/sglang#30822 ("[6/6][kimi-deterministic] Use
deterministic seeded coins for EAGLE rejection sampling"). Upstream's own suite
for this is CUDA-only because the coins hash through the murmur_hash32 Triton
kernel; here the hash is mocked with a deterministic pure-torch stand-in so the
contract runs on CPU. What is pinned:

* run-to-run reproducibility -- identical (seed, seq_lens) produce bitwise
  identical coins, and therefore identical ACCEPT DECISIONS across two runs;
* distinct seeds / sequence positions diverge;
* the column split (columns [0, draft_token_num) -> per-draft rejection coins,
  column draft_token_num -> final-sampling coin);
* unseeded requests keep the plain torch.rand draw (default behaviour, so the
  port is opt-in and the old path is byte-for-byte unchanged);
* coins stay in the half-open [0, 1) the sampling kernels require;
* the fork's #50 rank-0 broadcast is still downstream of the coin draw --
  seeding fixes the draw per run, the broadcast fixes the outcome per rank, and
  the port must not displace the latter.

The two determinism axes are orthogonal, which is why both live in eagle_sample
at once: #50 makes heterogeneous TP ranks agree inside ONE run; the seeded
coins make separate runs agree.
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.eagle_utils import (
    _seeded_verify_coins,
    _verify_coins,
    eagle_sample,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

DRAFT_TOKEN_NUM = 4
UMAX = torch.iinfo(torch.uint32).max
HASH_PATH = "sglang.srt.layers.utils.hash.murmur_hash32"


def _reference_hash(seed, positions, col_indices):
    """Deterministic pure-torch stand-in for the murmur_hash32 Triton kernel.

    Not murmur -- it only has to be a pure function of (seed, position, column)
    with enough mixing that different inputs land on different coins, which is
    the property every test below actually depends on.
    """
    s = seed.to(torch.int64).unsqueeze(1)
    p = positions.to(torch.int64).unsqueeze(1)
    c = col_indices.to(torch.int64).unsqueeze(0)
    h = (s * 6364136223846793005 + p * 1442695040888963407 + c * 1013904223) % (
        2**32
    )
    h = (h ^ (h >> 13)) * 2654435761 % (2**32)
    return h.to(torch.uint32)


def _coins(seeds, seq_lens, hash_fn=_reference_hash):
    with patch(HASH_PATH, side_effect=hash_fn):
        return _seeded_verify_coins(
            sampling_seed=torch.tensor(seeds, dtype=torch.int64),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
            draft_token_num=DRAFT_TOKEN_NUM,
            device="cpu",
        )


def _accept_lengths(coins, target_probs, draft_probs, candidates):
    """Reference Leviathan chain-rejection accept rule, run on the host.

    Stands in for chain_speculative_sampling_triton: accept draft token j while
    coin[i, j] <= target_p / draft_p, stop at the first rejection. Only used to
    turn coins into a decision so that "same seed -> same accepts" is asserted
    on the DECISION, not merely on the random numbers.
    """
    bs, k = candidates.shape
    out = []
    for i in range(bs):
        n = 0
        for j in range(k):
            tok = candidates[i, j]
            ratio = (target_probs[i, j, tok] / draft_probs[i, j, tok]).item()
            if coins[i, j].item() <= ratio:
                n += 1
            else:
                break
        out.append(n)
    return out


def _accept_fixture(bs=3, vocab=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.rand((bs, DRAFT_TOKEN_NUM, vocab), generator=g) + 0.05
    target = target / target.sum(-1, keepdim=True)
    draft = torch.rand((bs, DRAFT_TOKEN_NUM, vocab), generator=g) + 0.05
    draft = draft / draft.sum(-1, keepdim=True)
    candidates = torch.randint(
        0, vocab, (bs, DRAFT_TOKEN_NUM), generator=g, dtype=torch.int64
    )
    return target, draft, candidates


class TestSeededVerifyCoins(CustomTestCase):
    def test_seeded_coins_are_reproducible(self):
        coins_a, final_a = _coins([12345, 67890, 12345], [7, 9, 7])
        coins_b, final_b = _coins([12345, 67890, 12345], [7, 9, 7])

        self.assertEqual(tuple(coins_a.shape), (3, DRAFT_TOKEN_NUM))
        self.assertEqual(tuple(final_a.shape), (3,))
        self.assertTrue(torch.equal(coins_a, coins_b))
        self.assertTrue(torch.equal(final_a, final_b))
        # Same (seed, seq_len) pair hashes to the same coins regardless of row.
        self.assertTrue(torch.equal(coins_a[0], coins_a[2]))
        self.assertEqual(final_a[0].item(), final_a[2].item())
        # Coins live in [0, 1).
        self.assertTrue(bool((coins_a >= 0).all() and (coins_a < 1).all()))
        self.assertTrue(bool((final_a >= 0).all() and (final_a < 1).all()))

    def test_distinct_seeds_or_positions_diverge(self):
        coins, _ = _coins([12345, 67890, 12345], [7, 9, 11])
        self.assertFalse(torch.equal(coins[0], coins[1]))  # different seed
        self.assertFalse(torch.equal(coins[0], coins[2]))  # different seq_len

    def test_column_split_maps_rejection_then_final(self):
        # Structured hash: hashed[i, j] = i * 1000 + j, so each coin names its
        # (row, column) origin. Locks the column-space contract.
        def _structured_hash(seed, positions, col_indices):
            rows = torch.arange(seed.shape[0]).unsqueeze(1)
            return (rows * 1000 + col_indices.to(torch.int64).unsqueeze(0)).to(
                torch.uint32
            )

        coins, final = _coins([1, 2], [3, 4], hash_fn=_structured_hash)

        def _expected(row, col):
            return (
                torch.tensor(row * 1000 + col, dtype=torch.float64)
                .div(UMAX)
                .to(torch.float32)
                .item()
            )

        for row in range(2):
            for col in range(DRAFT_TOKEN_NUM):
                self.assertEqual(coins[row, col].item(), _expected(row, col))
            self.assertEqual(final[row].item(), _expected(row, DRAFT_TOKEN_NUM))

    def test_max_hash_clamps_coins_below_one(self):
        # The top 129 uint32 hashes round to exactly 1.0 under the float32
        # cast; force the worst case and assert the clamp holds the contract.
        def _all_max_hash(seed, positions, col_indices):
            return torch.full(
                (seed.shape[0], col_indices.shape[0]), UMAX, dtype=torch.uint32
            )

        coins, final = _coins([1, 2], [3, 4], hash_fn=_all_max_hash)
        self.assertTrue(bool((coins < 1).all()))
        self.assertTrue(bool((final < 1).all()))
        self.assertEqual(coins.max().item(), 1.0 - 2**-24)


class TestVerifyCoinDispatch(CustomTestCase):
    def _kwargs(self, sampling_seed):
        return dict(
            sampling_info=SimpleNamespace(sampling_seed=sampling_seed),
            seq_lens=torch.tensor([3, 4, 5], dtype=torch.int64),
            draft_token_num=DRAFT_TOKEN_NUM,
            candidates=torch.zeros((3, DRAFT_TOKEN_NUM), dtype=torch.int64),
            device="cpu",
        )

    def test_unseeded_requests_keep_torch_rand(self):
        """Default path: no seed -> unchanged torch.rand draw, hash untouched."""
        kwargs = self._kwargs(None)
        with patch(HASH_PATH) as mock_hash:
            coins_a, final_a = _verify_coins(**kwargs)
            coins_b, final_b = _verify_coins(**kwargs)

        mock_hash.assert_not_called()
        self.assertEqual(tuple(coins_a.shape), (3, DRAFT_TOKEN_NUM))
        self.assertEqual(tuple(final_a.shape), (3,))
        self.assertEqual(coins_a.dtype, torch.float32)
        # torch.rand draws: two calls must not repeat.
        self.assertFalse(torch.equal(coins_a, coins_b))
        self.assertFalse(torch.equal(final_a, final_b))

    def test_seeded_requests_dispatch_to_seeded_coins(self):
        seeds = torch.tensor([12345, 67890, 24680], dtype=torch.int64)
        kwargs = self._kwargs(seeds)
        with patch(HASH_PATH, side_effect=_reference_hash):
            coins, final = _verify_coins(**kwargs)
            expected_coins, expected_final = _seeded_verify_coins(
                sampling_seed=seeds,
                seq_lens=kwargs["seq_lens"],
                draft_token_num=DRAFT_TOKEN_NUM,
                device="cpu",
            )
        self.assertTrue(torch.equal(coins, expected_coins))
        self.assertTrue(torch.equal(final, expected_final))


class TestAcceptDecisionsAreRunToRunStable(CustomTestCase):
    """The point of the port, asserted on the DECISION rather than the coins."""

    def test_same_seed_same_accepts_across_two_runs(self):
        target, draft, candidates = _accept_fixture()
        seeds, seq_lens = [11, 22, 33], [5, 6, 7]

        run_a, _ = _coins(seeds, seq_lens)
        run_b, _ = _coins(seeds, seq_lens)
        accepts_a = _accept_lengths(run_a, target, draft, candidates)
        accepts_b = _accept_lengths(run_b, target, draft, candidates)

        self.assertEqual(accepts_a, accepts_b)
        # The fixture must actually exercise the rejection rule, otherwise the
        # equality above would be vacuous (all-accept or all-reject).
        self.assertTrue(
            0 < sum(accepts_a) < len(accepts_a) * DRAFT_TOKEN_NUM,
            f"degenerate fixture: {accepts_a}",
        )

    def test_unseeded_accepts_are_not_run_to_run_stable(self):
        """Falsifier for the test above: without seeding, two runs disagree."""
        target, draft, candidates = _accept_fixture()
        kwargs = dict(
            sampling_info=SimpleNamespace(sampling_seed=None),
            seq_lens=torch.tensor([5, 6, 7], dtype=torch.int64),
            draft_token_num=DRAFT_TOKEN_NUM,
            candidates=candidates,
            device="cpu",
        )
        torch.manual_seed(0)
        differed = False
        for _ in range(20):
            coins_a, _ = _verify_coins(**kwargs)
            coins_b, _ = _verify_coins(**kwargs)
            if _accept_lengths(coins_a, target, draft, candidates) != _accept_lengths(
                coins_b, target, draft, candidates
            ):
                differed = True
                break
        self.assertTrue(differed, "unseeded coins produced identical accepts 20x")

    def test_different_seed_changes_accepts(self):
        target, draft, candidates = _accept_fixture()
        seq_lens = [5, 6, 7]
        a, _ = _coins([11, 22, 33], seq_lens)
        b, _ = _coins([44, 55, 66], seq_lens)
        self.assertNotEqual(
            _accept_lengths(a, target, draft, candidates),
            _accept_lengths(b, target, draft, candidates),
        )


class TestCoinsSitBeforeTheRank0Broadcast(CustomTestCase):
    """#50 must survive the port: the coins are an INPUT to the verify kernel,
    the broadcast synchronizes its OUTPUT. If a future edit ever moved the coin
    draw below the broadcast (or dropped the broadcast), the two determinism
    axes would stop composing."""

    def test_verify_coins_precede_capture_safe_tp_broadcast(self):
        src = inspect.getsource(eagle_sample)
        coin_at = src.find("_verify_coins(")
        # rfind, not find: #143 inserted a weightless-lane receive branch at the
        # TOP of eagle_sample which carries its own capture_safe_tp_broadcast(.
        # The pin is about the HEAD-side outcome broadcast, which is the last one.
        broadcast_at = src.rfind("capture_safe_tp_broadcast(")
        self.assertNotEqual(coin_at, -1, "_verify_coins call site vanished")
        self.assertNotEqual(
            broadcast_at, -1, "#50 rank-0 verify broadcast vanished from eagle_sample"
        )
        self.assertLess(coin_at, broadcast_at)


if __name__ == "__main__":
    unittest.main()
