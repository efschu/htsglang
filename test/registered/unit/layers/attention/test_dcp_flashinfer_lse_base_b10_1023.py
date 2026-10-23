# SPDX-License-Identifier: Apache-2.0
"""B10 (27B duo, 23.10.): the uneven-DCP merges in the FlashInfer backend read
FlashInfer's LSE as a NATURAL log, but FlashInfer returns it in BASE 2.

FlashInfer (0.6.12 include/flashinfer/attention/prefill.cuh:1855, 0.7.0
attention/state.cuh ``get_lse() { return m + math::ptx_log2(d); }``, scores
pre-scaled by ``sm_scale * log2e``) writes ``lse = log2(sum_j exp(s_j))``; its
own ``merge_state`` (cascade.cuh) combines with ``exp2``/``log2``. The fork's
uneven-DCP port (cc4ff87d41) routes the FlashInfer partials through the
natural-log torch merges inherited from upstream's TRITON path
(``cp_lse_ag_out_ar_mha_uneven``/``_a2a_``: ``torch.logsumexp`` +
``torch.exp``; ``_dcp_final_merge_rows``: ``torch.logaddexp`` + ``torch.exp``).
Fed base-2 values, every partition (rank-0 owned prefix, rank-1 owned prefix,
current chunk) is weighted by ``Z_i ** (1/ln 2) = Z_i ** 1.4427`` instead of
``Z_i``: a softmax over partitions at the wrong temperature. Exact only when
there is a single partition -- i.e. a prompt that fits one chunk (no prefix)
-- which is exactly where the 27B duo's prompt logprobs were healthy; the
deviation grows as the current chunk's share of the context shrinks.

Pinned on CPU with FlashInfer-convention partials (base-2 LSE) against a
reference full softmax attention: the final current/prefix merge and the
two-rank uneven-head cross-rank merge (threads stand in for the DCP group).
Red on the base tree, green with the conversion (``lse * ln 2`` at the merge
boundary, FLLIPER_DCP_FI_LSE_LN2=1 default, 0 = the old reading for an A/B).
"""

import math
import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from flliper.srt.layers.attention import flashinfer_backend as fb  # noqa: E402

D = 8


def _partial(q, k, v):
    """FlashInfer-convention partial attention: (o normalised within the
    partition, lse in BASE 2). q [T,H,D], k/v [N,H,D] (kv already per q head)."""
    s = torch.einsum("thd,nhd->thn", q, k) / math.sqrt(D)
    lse_e = torch.logsumexp(s, dim=-1)                     # [T,H]
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("thn,nhd->thd", p, v)
    return o, lse_e / math.log(2.0)                        # base 2, as FlashInfer


def _reference(q, k, v):
    s = torch.einsum("thd,nhd->thn", q, k) / math.sqrt(D)
    return torch.einsum("thn,nhd->thd", torch.softmax(s, dim=-1), v)


def _setup(T=5, H=4, n_pre=40, seed=0, sink=6.0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, H, D, generator=g, dtype=torch.float64)
    k_pre = torch.randn(n_pre, H, D, generator=g, dtype=torch.float64)
    v_pre = torch.randn(n_pre, H, D, generator=g, dtype=torch.float64)
    k_cur = torch.randn(T, H, D, generator=g, dtype=torch.float64)
    v_cur = torch.randn(T, H, D, generator=g, dtype=torch.float64)
    # an attention sink in the prefix (token 0), as real models have: it makes
    # the partitions' masses unequal, which is where the base matters
    k_pre[0] = q.mean(0) * sink
    return q, k_pre, v_pre, k_cur, v_cur


class _Layer(SimpleNamespace):
    pass


class TestFinalMergeBase(CustomTestCase):
    """_dcp_extend_final_merge: current chunk (ragged) + merged prefix. One
    rank, so the prefix partial goes straight in (its LSE in FlashInfer's base,
    converted by the boundary under test)."""

    def _merge(self, o_cur, lse_cur, o_pre, lse_pre, H):
        me = SimpleNamespace(_dcp_merge_block_tokens=0)
        layer = _Layer(tp_q_head_num=H, head_dim=D)
        q = torch.zeros(o_cur.shape[0], H * D, dtype=torch.float64)
        # the prefix LSE leaves _dcp_uneven_merge in the merge's own (natural)
        # domain; with one rank that is the converted raw value
        lse_pre_nat = fb._fi_lse_to_merge_domain(lse_pre)
        out = fb.FlashInferAttnBackend._dcp_extend_final_merge(
            me, q, layer, o_cur, lse_cur, o_pre, lse_pre_nat
        )
        return out.view(o_cur.shape[0], H, D)

    def test_final_merge_equals_full_attention(self):
        for seed in range(3):
            q, kp, vp, kc, vc = _setup(seed=seed)
            T, H = q.shape[0], q.shape[1]
            # causal current chunk: row t sees cur tokens 0..t
            o_cur = torch.empty_like(q)
            lse_cur = torch.empty(T, H, dtype=torch.float64)
            for t in range(T):
                o, l = _partial(q[t : t + 1], kc[: t + 1], vc[: t + 1])
                o_cur[t], lse_cur[t] = o[0], l[0]
            o_pre, lse_pre = _partial(q, kp, vp)
            got = self._merge(o_cur, lse_cur, o_pre, lse_pre, H)
            want = torch.stack([
                _reference(q[t : t + 1], torch.cat([kp, kc[: t + 1]]),
                           torch.cat([vp, vc[: t + 1]]))[0]
                for t in range(T)
            ])
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)

    def test_old_reading_is_off_by_the_temperature(self):
        """FLLIPER_DCP_FI_LSE_LN2=0 restores the old reading -- and it is
        measurably wrong on the same data (the A/B arm for the metal)."""
        q, kp, vp, kc, vc = _setup(seed=1)
        o_pre, lse_pre = _partial(q, kp, vp)
        o_cur, lse_cur = _partial(q, kc, vc)  # non-causal is fine for this check
        want = _reference(q, torch.cat([kp, kc]), torch.cat([vp, vc]))
        with mock.patch.dict(os.environ, {"FLLIPER_DCP_FI_LSE_LN2": "0"}):
            fb._FI_LSE_LN2_CACHE.clear()
            bad = self._merge(o_cur, lse_cur, o_pre, lse_pre, q.shape[1])
        fb._FI_LSE_LN2_CACHE.clear()
        good = self._merge(o_cur, lse_cur, o_pre, lse_pre, q.shape[1])
        torch.testing.assert_close(good, want, rtol=1e-6, atol=1e-6)
        self.assertGreater((bad - want).abs().max().item(), 1e-2)


class _ThreadGroup:
    """Two in-process 'ranks' (threads) behind the GroupCoordinator surface the
    uneven merge uses: world_size, rank_in_group, all_gather(dim=0), all_reduce."""

    def __init__(self, world, shared, rank):
        self.world_size = world
        self.rank_in_group = rank
        self._s = shared

    def _exchange(self, t):
        s = self._s
        s["slots"][self.rank_in_group] = t.clone()
        s["barrier"].wait()
        got = [x.clone() for x in s["slots"]]
        s["barrier"].wait()
        return got

    def all_gather(self, t, dim=0):
        return torch.cat(self._exchange(t), dim=dim)

    def all_reduce(self, t):
        return torch.stack(self._exchange(t)).sum(0)


class TestCrossRankUnevenMerge(CustomTestCase):
    """Head (rank 0: 3 q heads) + 3080-like (rank 1: 1 q head); the prefix KV is
    token-sharded between the two ranks; each rank attends its owned prefix with
    ALL heads, merges cross-rank, then merges its own current-chunk heads."""

    def test_two_rank_extend_equals_full_attention(self):
        counts = [3, 1]
        q, kp, vp, kc, vc = _setup(H=sum(counts), n_pre=48, seed=4)
        T = q.shape[0]
        owned = [torch.arange(48) % 64 < 37, torch.arange(48) % 64 >= 37]
        # weighted ownership 37:27 per 64 slots -> slots 37..47 on rank 1
        shared = {"slots": [None, None], "barrier": threading.Barrier(2)}
        results = [None, None]
        errors = []

        def rank_body(r):
            try:
                grp = _ThreadGroup(2, shared, r)
                o_raw, lse_raw = _partial(q, kp[owned[r]], vp[owned[r]])
                o_pre, lse_pre = fb._dcp_uneven_merge(
                    o_raw, lse_raw, grp, counts, return_lse=True, block_tokens=0
                )
                lo = sum(counts[:r])
                hs = slice(lo, lo + counts[r])
                o_cur = torch.empty(T, counts[r], D, dtype=torch.float64)
                lse_cur = torch.empty(T, counts[r], dtype=torch.float64)
                for t in range(T):
                    o, l = _partial(q[t : t + 1, hs], kc[: t + 1, hs], vc[: t + 1, hs])
                    o_cur[t], lse_cur[t] = o[0], l[0]
                me = SimpleNamespace(_dcp_merge_block_tokens=0)
                layer = _Layer(tp_q_head_num=counts[r], head_dim=D)
                qz = torch.zeros(T, counts[r] * D, dtype=torch.float64)
                out = fb.FlashInferAttnBackend._dcp_extend_final_merge(
                    me, qz, layer, o_cur, lse_cur, o_pre, lse_pre
                )
                results[r] = out.view(T, counts[r], D)
            except Exception as e:  # surfaced below
                errors.append(e)
                shared["barrier"].abort()

        th = [threading.Thread(target=rank_body, args=(r,)) for r in range(2)]
        for t in th:
            t.start()
        for t in th:
            t.join(60)
        self.assertEqual(errors, [])
        want = torch.stack([
            _reference(q[t : t + 1], torch.cat([kp, kc[: t + 1]]),
                       torch.cat([vp, vc[: t + 1]]))[0]
            for t in range(T)
        ])
        got = torch.cat(results, dim=1)
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def _torch_natural_merge(v_a, s_a, v_b, s_b):
    """merge_state_triton's math (tl.exp / tl.log), in torch."""
    m = torch.maximum(s_a, s_b)
    ea, eb = torch.exp(s_a - m), torch.exp(s_b - m)
    se = ea + eb
    v = v_a * (ea / se).unsqueeze(-1) + v_b * (eb / se).unsqueeze(-1)
    return v, torch.log(se) + m


class TestSafeMergeStateFallback(CustomTestCase):
    """_safe_merge_state's >max-heads fallback (merge_state_triton, natural
    log) must give what flashinfer's base-2 merge_state gives."""

    def test_fallback_on_base2_inputs_equals_full_attention(self):
        q, kp, vp, kc, vc = _setup(seed=9)
        o_a, l_a = _partial(q, kp, vp)
        o_b, l_b = _partial(q, kc, vc)
        v, s = fb._natural_merge_on_base2(_torch_natural_merge, o_a, l_a, o_b, l_b)
        want = _reference(q, torch.cat([kp, kc]), torch.cat([vp, vc]))
        torch.testing.assert_close(v, want, rtol=1e-6, atol=1e-6)
        # merged LSE stays base 2 (log2 of the total partition function)
        s_all = torch.logsumexp(
            torch.einsum("thd,nhd->thn", q, torch.cat([kp, kc])) / math.sqrt(D), dim=-1
        ) / math.log(2.0)
        torch.testing.assert_close(s, s_all, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
