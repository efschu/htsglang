"""--d-token-placement (27B, 26.09.): new D KV tokens pick the card by fill.

Pinned without a GPU (design: /spinning/gpu-arb/docs/DYN_D_RESHARD.md sec. 13):
  * INTERLEAVE -- every prefix of the reordered free list draws from class r
    in proportion w_r (+-1 slot) while the class lasts, then the others go on
    (capacity fallback); stable within a class; a permutation (nothing lost);
  * POLICY -- bandwidth shares at low fill, deficit repayment, capacity once
    the bandwidth target passes the fill switch, 'full' when nothing is free;
  * RANKS AGREE -- two replicated allocators with the same history hand out
    the same ids (no collective needed);
  * ALLOCATOR -- the token and the paged (page_size 1) allocator hand out ids
    in the placement's proportion, re-interleave after frees and after clear,
    never fail an allocation the plain list would serve;
  * DCP EXACT -- attention over tokens split across ranks by ANY slot
    ownership, merged by log-sum-exp, equals full attention (fp32 1e-5): new
    tokens choosing the card needs no second ledger in attention/merge;
  * OFF IDENTICAL -- no env: no placement, free list byte-identical; the
    launcher default ships no env; NF refused; corridor steering refused.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from sglang.srt.weg2 import d_reshard as D
from sglang.srt.weg2 import d_token_placement as P
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PREFIX = (0, 21, 43, 64)          # installed vector (21, 22, 21), S = 64
SPEC = P.PlacementSpec("bandwidth", P.RC9_EFF_BW_GBS)


def _static(prefix=PREFIX):
    return lambda: tuple(prefix)


class TestInterleave(unittest.TestCase):
    def test_prefix_proportions_and_permutation(self):
        pages = torch.arange(1, 64 * 200 + 1)
        w = (0.44, 0.28, 0.28)
        out = P.weighted_interleave(pages, 64, PREFIX, w)
        self.assertTrue(torch.equal(torch.sort(out).values, pages))
        cls = P.class_of(out, 64, PREFIX)
        for k in (10, 100, 1000, 3000):
            counts = torch.bincount(cls[:k], minlength=3).tolist()
            for r in range(3):
                self.assertLessEqual(abs(counts[r] - w[r] * k), 1.5, (k, counts))

    def test_stable_within_class_and_fallback(self):
        pages = torch.tensor([5, 30, 50, 6, 31, 51, 7, 8, 9, 10])  # class 0 has 5..10 (6 ids)
        out = P.weighted_interleave(pages, 64, PREFIX, (0.1, 0.45, 0.45))
        for r in range(3):
            got = [int(x) for x in out if int(P.class_of(torch.tensor([int(x)]), 64, PREFIX)) == r]
            want = [int(x) for x in pages if int(P.class_of(torch.tensor([int(x)]), 64, PREFIX)) == r]
            self.assertEqual(got, want)
        # classes 1/2 run out after 2 each; the tail is class 0 (capacity fallback)
        self.assertTrue(all(int(x) < 21 for x in out[-4:]))

    def test_zero_weight_goes_last(self):
        pages = torch.arange(1, 129)
        out = P.weighted_interleave(pages, 64, PREFIX, (1.0, 0.0, 0.0))
        n0 = int((P.class_of(pages, 64, PREFIX) == 0).sum())
        self.assertTrue(bool((P.class_of(out[:n0], 64, PREFIX) == 0).all()))

    def test_slot_class_totals_exact(self):
        for size in (1, 63, 64, 65, 1000, 754048):
            ids = torch.arange(1, size + 1)
            self.assertEqual(P.slot_class_totals(size, 64, PREFIX), P.class_counts(ids, 64, PREFIX))


class TestPolicy(unittest.TestCase):
    def test_low_fill_bandwidth_high_fill_capacity(self):
        tot = [250000, 260000, 250000]
        w, reg = P.placement_weights(tot, tot, SPEC)
        self.assertEqual(reg, "bandwidth")
        self.assertAlmostEqual(w[0], 937 / (937 + 2 * 604), places=3)
        # the 5090 class already holds its bandwidth share of a 90 % fill -> capacity
        free = [25000, 26000, 25000]
        w, reg = P.placement_weights(tot, free, SPEC)
        self.assertEqual(reg, "capacity")
        self.assertAlmostEqual(w[0], 25000 / 76000, places=6)
        self.assertEqual(P.placement_weights(tot, [0, 0, 0], SPEC)[1], "full")

    def test_deficit_repayment(self):
        tot = [250000, 260000, 250000]
        used = [40000, 60000, 60000]      # 5090 below its bandwidth share
        w, reg = P.placement_weights(tot, [t - u for t, u in zip(tot, used)], SPEC)
        self.assertEqual(reg, "bandwidth")
        self.assertGreater(w[0], 0.437)

    def test_capacity_policy_is_capacity(self):
        spec = P.PlacementSpec("capacity", P.RC9_EFF_BW_GBS)
        self.assertEqual(P.placement_weights([10, 10, 10], [10, 10, 10], spec)[1], "capacity")

    def test_spec_roundtrip_and_validation(self):
        self.assertEqual(P.PlacementSpec.from_json(SPEC.to_json()), SPEC)
        with self.assertRaises(P.PlacementError):
            P.PlacementSpec("bandwidth", (1.0,)).validate()
        with self.assertRaises(P.PlacementError):
            P.PlacementSpec("bandwidth", (1.0, 1.0), fill_switch=1.5).validate()
        self.assertIsNone(P.spec_from_env({}))
        self.assertIsNone(P.spec_from_env({P.ENV: P.PlacementSpec("capacity", (1, 1, 1)).to_json()}))


def _paged(size):
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    return PagedTokenToKVPoolAllocator(size, page_size=1, dtype=torch.float16, device="cpu",
                                       kvcache=None, need_sort=False)


def _token(size):
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

    return TokenToKVPoolAllocator(size, dtype=torch.float16, device="cpu", kvcache=None, need_sort=False)


class TestAllocator(unittest.TestCase):
    def _arm(self, alloc, every=4):
        spec = P.PlacementSpec("bandwidth", P.RC9_EFF_BW_GBS, reapply_every=every)
        alloc.set_owner_placement(P.OwnerPlacement(spec, prefix_fn=_static()))

    def test_default_untouched(self):
        for mk in (_paged, _token):
            a = mk(6400)
            self.assertIsNone(a._owner_placement)
            self.assertTrue(torch.equal(a.free_pages, torch.arange(1, 6401)))
            self.assertTrue(torch.equal(a.alloc(10), torch.arange(1, 11)))

    def test_alloc_follows_bandwidth_share(self):
        for mk in (_paged, _token):
            a = mk(64 * 500)
            self._arm(a)
            got = torch.cat([a.alloc(64) for _ in range(20)])
            share0 = float((P.class_of(got, 64, PREFIX) == 0).float().mean())
            self.assertAlmostEqual(share0, 0.437, delta=0.01)
            self.assertEqual(len(torch.unique(got)), got.numel())

    def test_frees_rewash_and_clear_reapplies(self):
        a = _paged(64 * 200)
        self._arm(a, every=2)
        first = a.alloc(640)
        a.free(first[:320])                      # freed ids go to the HEAD (paged, no sort)
        self.assertTrue(a._owner_placement.touched)
        a.alloc(1)
        a.alloc(1)                               # due -> re-interleaved
        self.assertFalse(a._owner_placement.touched)
        nxt = a.alloc(640)
        self.assertAlmostEqual(float((P.class_of(nxt, 64, PREFIX) == 0).float().mean()), 0.437, delta=0.02)
        a.clear()
        self.assertEqual(a._owner_placement.last_regime, "bandwidth")
        head = a.free_pages[:1000]
        self.assertAlmostEqual(float((P.class_of(head, 64, PREFIX) == 0).float().mean()), 0.437, delta=0.01)

    def test_never_fails_what_plain_serves(self):
        a = _token(64 * 10)
        self._arm(a, every=1)
        got = a.alloc(64 * 10)
        self.assertIsNotNone(got)
        self.assertEqual(len(torch.unique(got)), 640)
        self.assertIsNone(a.alloc(1))

    def test_replicated_allocators_agree(self):
        a, b = _paged(64 * 300), _paged(64 * 300)
        self._arm(a)
        self._arm(b)
        g = torch.Generator().manual_seed(3)
        held_a, held_b = [], []
        for step in range(60):
            n = int(torch.randint(1, 200, (1,), generator=g))
            xa, xb = a.alloc(n), b.alloc(n)
            self.assertTrue(torch.equal(xa, xb), step)
            held_a.append(xa)
            held_b.append(xb)
            if step % 7 == 6:
                a.free(held_a.pop(0))
                b.free(held_b.pop(0))
        self.assertTrue(torch.equal(a.free_pages, b.free_pages))

    def test_refuses_next_to_bias_and_page_size(self):
        a = _paged(640)
        a.set_owner_bias((64, 0, 21))
        with self.assertRaises(ValueError):
            self._arm(a)
        with mock.patch.dict(os.environ, {"SGLANG_CORRIDOR_STEERING": "1"}):
            with self.assertRaises(P.PlacementError):
                P.arm_on_allocator(_paged(640), SPEC)
        self.assertFalse(P.arm_on_allocator(_paged(640), None))

    def test_inactive_without_matching_vector(self):
        a = _paged(640)
        spec = P.PlacementSpec("bandwidth", P.RC9_EFF_BW_GBS)
        a.set_owner_placement(P.OwnerPlacement(spec, prefix_fn=lambda: None))
        self.assertTrue(torch.equal(a.free_pages, torch.arange(1, 641)))
        self.assertEqual(a._owner_placement.last_regime, "inactive")


def _attn(q, k, v):
    s = (q @ k.T) / (q.shape[-1] ** 0.5)
    m = s.max(-1, keepdim=True).values
    p = torch.exp(s - m)
    l = p.sum(-1, keepdim=True)
    return (p @ v) / l, (m + torch.log(l)).squeeze(-1)


class TestDcpExactUnderAnyOwnership(unittest.TestCase):
    def test_lse_merge_equals_full(self):
        g = torch.Generator().manual_seed(11)
        T, H = 777, 32
        q = torch.randn(4, H, generator=g, dtype=torch.float64)
        k = torch.randn(T, H, generator=g, dtype=torch.float64)
        v = torch.randn(T, H, generator=g, dtype=torch.float64)
        full, _ = _attn(q, k, v)
        slots = torch.randperm(4096, generator=g)[:T] + 1          # ANY ids, e.g. steered
        for prefix in (PREFIX, (0, 26, 45, 64)):
            owner = P.class_of(slots, 64, prefix)
            outs, lses = [], []
            for r in range(3):
                m = owner == r
                o, l = _attn(q, k[m], v[m])
                outs.append(o)
                lses.append(l)
            L = torch.stack(lses)                                   # [3, q]
            w = torch.softmax(L, 0).unsqueeze(-1)
            merged = (w * torch.stack(outs)).sum(0)
            torch.testing.assert_close(merged, full, atol=1e-10, rtol=1e-10)


class TestLauncher(unittest.TestCase):
    def setUp(self):
        from sglang.srt.weg2 import launcher as L
        self.L = L

    def _model(self, cfg):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        return d

    def test_default_capacity_no_env(self):
        ns = self.L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertEqual(ns.d_token_placement, "capacity")
        self.L.apply_d_token_placement(ns)
        self.assertEqual(self.L.d_token_placement_env(), {})

    def test_bandwidth_env_and_nf_refused(self):
        m = self._model({"text_config": dict(D._QWEN38_27B)})
        ns = self.L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--model", m,
                                               "--d-token-placement", "bandwidth"])
        self.L.apply_d_token_placement(ns)
        env = self.L.d_token_placement_env()
        self.assertEqual(P.PlacementSpec.from_json(env[P.ENV]).shares, P.RC9_EFF_BW_GBS)
        nf = self._model({"model_type": "qwen3_next", "num_experts": 512})
        ns = self.L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--model", nf,
                                               "--d-token-placement", "bandwidth"])
        with self.assertRaises(SystemExit) as cm:
            self.L.apply_d_token_placement(ns)
        self.assertIn("--d-token-placement bandwidth: REFUSED", str(cm.exception.code))
        self.assertEqual(self.L.d_token_placement_env(), {})


if __name__ == "__main__":
    unittest.main()
