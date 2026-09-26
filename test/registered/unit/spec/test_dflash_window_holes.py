"""SGLANG_DFLASH_WINDOW_HOLE_MASK (27b-draftwin 26.09.): the DFLASH draft window's
hole rows (no draft KV -> hole slot 0) leave the draft softmax.

CPU equivalence tests against a reference of FlashInfer's paged prefill as the
draft runs it (non-causal block, sliding window ``kv_idx + qo_len +
window_left >= kv_len + qo_idx``, GQA ``h -> h // group``, ``sm_scale *=
k_scale``, ``out *= v_scale``, base-2 LSE -- flashinfer 0.6.14
include/flashinfer/attention/{prefill.cuh,variants.cuh}):

* switch off: the rebuild returns None and writes the identical table, the env
  defaults to off; a zero count returns the kernel output BIT-EXACTLY, even
  with a non-finite hole slot;
* switch on: the corrected attention over a partly filled window equals the
  attention over the mapped rows only. Tolerance: fp32 output rtol/atol 2e-5
  (exp2/log2 rounding of the LSE identity); bf16 output 2 bf16 ulp relative
  (1.6e-2) + 2e-3 absolute (the kernel's bf16 rounding of ``o`` carries
  through the division by ``1 - a``).
"""

import math
import os
import types
import unittest

import torch

from sglang.srt.speculative.dflash_solo_pool import (
    DraftKVSlotMapper,
    rebuild_window_rows_sync_free,
)
from sglang.srt.speculative.dflash_window_holes import (
    correct_window_hole_attention,
    draft_hole_buffer_rows,
    window_hole_counts_per_token,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

LOG2E = 1.0 / math.log(2.0)


def _visible(L, block, i, c, wl):
    """FlashInfer's sliding-window rule for kv row c of query i."""
    kv_len, qo_len = L + block, block
    w = kv_len if wl < 0 else wl
    return c + qo_len + w >= kv_len + i


def _flashinfer_ref(q, K, V, slots, L, block, wl, sm_scale, ks=1.0, vs=1.0,
                    keep=None):
    """Reference of the draft's paged prefill for ONE request.

    q [block, Hq, D]; K/V pool [S, Hkv, D]; slots [L + block] draft slots of
    the kv rows. ``keep`` (bool [L + block]) restricts the kv set (the
    'mapped rows only' reference). Returns (o [block,Hq,D] fp32, lse2)."""
    block_, Hq, D = q.shape
    Hkv = K.shape[1]
    g = Hq // Hkv
    kv_len = L + block
    o = torch.zeros(block, Hq, D, dtype=torch.float64)
    lse2 = torch.zeros(block, Hq, dtype=torch.float64)
    Kf = K.double() * ks
    Vf = V.double() * vs
    for i in range(block):
        cols = [c for c in range(kv_len) if _visible(L, block, i, c, wl)
                and (keep is None or bool(keep[c]))]
        idx = torch.tensor([int(slots[c]) for c in cols], dtype=torch.long)
        for h in range(Hq):
            s = (Kf[idx, h // g] @ q[i, h].double()) * sm_scale
            m = s.max()
            p = torch.exp(s - m)
            z = p.sum()
            o[i, h] = (p.unsqueeze(1) * Vf[idx, h // g]).sum(0) / z
            lse2[i, h] = (m + torch.log(z)) * LOG2E
    return o, lse2


class TestHoleCountsPerToken(CustomTestCase):
    def test_matches_brute_force(self):
        gen = torch.Generator().manual_seed(3)
        for trial in range(40):
            bs = int(torch.randint(1, 5, (1,), generator=gen))
            block = int(torch.randint(1, 9, (1,), generator=gen))
            lengths = torch.randint(0, 40, (bs,), generator=gen)
            max_len = int(lengths.max()) + int(torch.randint(0, 5, (1,), generator=gen))
            if max_len == 0:
                max_len = 1
            holes = torch.rand(bs, max_len, generator=gen) < 0.5
            offs = torch.arange(max_len).unsqueeze(0)
            holes &= offs < lengths.unsqueeze(1)
            wl = int(torch.randint(-1, 45, (1,), generator=gen))
            got = window_hole_counts_per_token(holes, lengths, block, wl, bs)
            want = []
            for b in range(bs):
                L = int(lengths[b])
                for i in range(block):
                    want.append(sum(
                        1 for c in range(L)
                        if bool(holes[b, c]) and _visible(L, block, i, c, wl)
                    ))
            self.assertEqual(got.tolist(), [float(x) for x in want], f"trial {trial}")

    def test_none_is_zero(self):
        got = window_hole_counts_per_token(None, torch.tensor([5, 7]), 4, 2047, 2)
        self.assertEqual(got.tolist(), [0.0] * 8)


class TestCorrection(CustomTestCase):
    def _case(self, *, Hq=8, Hkv=2, D=16, block=8, L=40, wl=-1, hole_frac=0.6,
              slot0="zero", ks=1.0, vs=1.0, out_dtype=torch.float32, seed=0,
              contiguous=True):
        gen = torch.Generator().manual_seed(seed)
        S = L + block + 8
        K = torch.randn(S, Hkv, D, generator=gen)
        V = torch.randn(S, Hkv, D, generator=gen)
        if slot0 == "zero":
            K[0] = 0
            V[0] = 0
        # kv rows: window rows [0, L) then the block rows (always mapped).
        slots = torch.randperm(S - 1, generator=gen)[: L + block] + 1
        if contiguous:
            n_holes = int(L * hole_frac)
            holes = torch.zeros(L, dtype=torch.bool)
            holes[:n_holes] = True
        else:
            holes = torch.rand(L, generator=gen) < hole_frac
        slots[:L][holes] = 0
        q = torch.randn(block, Hq, D, generator=gen)
        sm = 1.0 / math.sqrt(D)
        o_full, lse2 = _flashinfer_ref(q, K, V, slots, L, block, wl, sm, ks, vs)
        keep = torch.ones(L + block, dtype=torch.bool)
        keep[:L] = ~holes
        o_masked, _ = _flashinfer_ref(q, K, V, slots, L, block, wl, sm, ks, vs, keep=keep)
        counts = window_hole_counts_per_token(
            holes.unsqueeze(0), torch.tensor([L]), block, wl, 1
        )
        got = correct_window_hole_attention(
            o_full.to(out_dtype), lse2.float(), q, K[0], V[0], counts,
            sm_scale=sm, k_scale=ks, v_scale=vs,
        )
        return got, o_masked, o_full, counts

    def test_fp32_equals_mapped_rows_only(self):
        for seed in range(6):
            for wl in (-1, 2047, 30, 7):
                for slot0 in ("zero", "garbage"):
                    for contiguous in (True, False):
                        got, want, full, counts = self._case(
                            wl=wl, slot0=slot0, seed=seed, contiguous=contiguous
                        )
                        torch.testing.assert_close(
                            got.double(), want, rtol=2e-5, atol=2e-5,
                            msg=f"seed={seed} wl={wl} slot0={slot0} contig={contiguous}",
                        )

    def test_fp32_the_holes_do_dilute(self):
        # Sanity of the premise: without the correction the output differs.
        got, want, full, counts = self._case(hole_frac=0.95, L=200)
        self.assertGreater(float(counts.min()), 0.0)
        self.assertGreater(float((full - want).abs().max()), 1e-2)

    def test_scales_and_gqa(self):
        for Hq, Hkv in ((8, 8), (8, 2), (32, 8)):
            got, want, _, _ = self._case(Hq=Hq, Hkv=Hkv, ks=0.37, vs=1.9,
                                         slot0="garbage", seed=Hq + Hkv)
            torch.testing.assert_close(got.double(), want, rtol=2e-5, atol=2e-5)

    def test_bf16_output_within_bf16_rounding(self):
        for seed in range(4):
            got, want, _, _ = self._case(out_dtype=torch.bfloat16, seed=seed,
                                         hole_frac=0.9, L=120, wl=100)
            self.assertEqual(got.dtype, torch.bfloat16)
            torch.testing.assert_close(got.double(), want, rtol=1.6e-2, atol=2e-3)

    def test_zero_count_is_bit_exact(self):
        gen = torch.Generator().manual_seed(11)
        T, Hq, Hkv, D = 16, 8, 2, 16
        o = torch.randn(T, Hq, D, generator=gen).to(torch.bfloat16)
        lse2 = torch.randn(T, Hq, generator=gen)
        q = torch.randn(T, Hq, D, generator=gen)
        k0 = torch.full((Hkv, D), float("nan"))
        v0 = torch.full((Hkv, D), float("inf"))
        got = correct_window_hole_attention(
            o, lse2, q, k0, v0, torch.zeros(64), sm_scale=0.1
        )
        self.assertTrue(torch.equal(got, o))
        # Mixed: rows with zero count keep their exact bytes.
        cnt = torch.zeros(64)
        cnt[3] = 5.0
        k0 = torch.zeros(Hkv, D)
        v0 = torch.zeros(Hkv, D)
        lse2 = torch.full((T, Hq), 10.0)
        got = correct_window_hole_attention(o, lse2, q, k0, v0, cnt, sm_scale=0.1)
        rows = [t for t in range(T) if t != 3]
        self.assertTrue(torch.equal(got[rows], o[rows]))
        self.assertFalse(torch.equal(got[3], o[3]))


class TestRebuildReturnsHoles(CustomTestCase):
    def _setup(self, seed):
        gen = torch.Generator().manual_seed(seed)
        num_global, width = 3000, 300
        m = DraftKVSlotMapper(num_global, 2000, 64, device="cpu", sync_free=True)
        mapped = torch.randperm(num_global, generator=gen)[:900] + 1
        m.translate_write(mapped.clone())
        m.begin_round()
        target = torch.randint(1, num_global + 1, (8, width), generator=gen).to(torch.int32)
        bs = 3
        rpi = torch.tensor([5, 1, 6])
        lengths = torch.tensor([120, 0, 200], dtype=torch.int32)
        start = torch.tensor([10, 0, 90])
        return m, target, rpi, lengths, start, bs

    def test_default_returns_none_and_same_table(self):
        a = self._setup(1)
        b = self._setup(1)
        ta = torch.full((8, 300), -7, dtype=torch.int32)
        tb = ta.clone()
        r0 = rebuild_window_rows_sync_free(
            mapper=a[0], target_req_to_token=a[1], draft_req_to_token=ta,
            req_pool_indices=a[2], start=a[4], lengths=a[3], max_len=210,
        )
        r1 = rebuild_window_rows_sync_free(
            mapper=b[0], target_req_to_token=b[1], draft_req_to_token=tb,
            req_pool_indices=b[2], start=b[4], lengths=b[3], max_len=210,
            return_holes=True,
        )
        self.assertIsNone(r0)
        self.assertEqual(ta.tolist(), tb.tolist())
        self.assertEqual(a[0].map.tolist(), b[0].map.tolist())
        self.assertEqual(a[0]._slot_epoch.tolist(), b[0]._slot_epoch.tolist())
        self.assertEqual(tuple(r1.shape), (3, 210))

    def test_holes_are_the_unmapped_real_rows(self):
        m, target, rpi, lengths, start, bs = self._setup(2)
        table = torch.zeros(8, 300, dtype=torch.int32)
        holes = rebuild_window_rows_sync_free(
            mapper=m, target_req_to_token=target, draft_req_to_token=table,
            req_pool_indices=rpi, start=start, lengths=lengths, max_len=210,
            return_holes=True,
        )
        for b in range(bs):
            L = int(lengths[b])
            for c in range(210):
                if c >= L:
                    self.assertFalse(bool(holes[b, c]))
                    continue
                g = int(target[int(rpi[b]), int(start[b]) + c])
                self.assertEqual(bool(holes[b, c]), int(m.map[g]) < 0, (b, c))

    def test_empty_round_returns_none(self):
        m = DraftKVSlotMapper(100, 16, 64, device="cpu", sync_free=True)
        r = rebuild_window_rows_sync_free(
            mapper=m, target_req_to_token=torch.zeros(2, 8, dtype=torch.int32),
            draft_req_to_token=torch.zeros(2, 8, dtype=torch.int32),
            req_pool_indices=torch.tensor([0]), start=torch.tensor([0]),
            lengths=torch.tensor([0]), max_len=0, return_holes=True,
        )
        self.assertIsNone(r)


class TestEndToEnd(CustomTestCase):
    """Mapper -> rebuild (return_holes) -> per-token counts -> correction ==
    attention over the mapped rows only, per request, for a batch whose
    windows are partly filled after a 'flip' (prefix unmapped, tail mapped)
    plus a scattered hole (an LRU/dedup-shaped gap)."""

    def test_partly_filled_windows(self):
        gen = torch.Generator().manual_seed(5)
        num_global, S, Hq, Hkv, D, block, wl = 4000, 600, 8, 2, 16, 8, 63
        m = DraftKVSlotMapper(num_global, S, 64, device="cpu", sync_free=True)
        bs, width = 2, 200
        lengths = torch.tensor([64, 50], dtype=torch.int32)
        target = torch.zeros(4, width, dtype=torch.int32)
        rpi = torch.tensor([2, 0])
        perm = torch.randperm(num_global, generator=gen) + 1
        cursor = 0
        mapped_tail = (40, 10)  # rows written by D since the flip
        for b in range(bs):
            n = int(lengths[b]) + block
            g = perm[cursor: cursor + n]
            cursor += n
            target[int(rpi[b]), :n] = g.to(torch.int32)
            tail = g[int(lengths[b]) - mapped_tail[b]:]
            m.translate_write(tail.clone())
        # A scattered hole inside request 0's mapped tail.
        m._apply_free(target[2, 60:61].to(torch.int64))
        m.begin_round()
        K = torch.randn(S + 1, Hkv, D, generator=gen)
        V = torch.randn(S + 1, Hkv, D, generator=gen)
        table = torch.zeros(4, width, dtype=torch.int32)
        holes = rebuild_window_rows_sync_free(
            mapper=m, target_req_to_token=target, draft_req_to_token=table,
            req_pool_indices=rpi, start=torch.zeros(bs, dtype=torch.int64),
            lengths=lengths, max_len=64, return_holes=True,
        )
        counts = window_hole_counts_per_token(holes, lengths, block, wl, bs)
        self.assertEqual(float(counts[0]), 64 - 40 + 1 - 1)  # q0 sees c >= 1
        sm = 1.0 / math.sqrt(D)
        for b in range(bs):
            L = int(lengths[b])
            blk = m.translate_write(target[int(rpi[b]), L: L + block].to(torch.int64))
            slots = torch.cat([table[int(rpi[b]), :L].to(torch.int64), blk])
            q = torch.randn(block, Hq, D, generator=gen)
            o_full, lse2 = _flashinfer_ref(q, K, V, slots, L, block, wl, sm)
            keep = torch.ones(L + block, dtype=torch.bool)
            keep[:L] = ~holes[b, :L]
            want, _ = _flashinfer_ref(q, K, V, slots, L, block, wl, sm, keep=keep)
            got = correct_window_hole_attention(
                o_full.float(), lse2.float(), q, K[0], V[0],
                counts[b * block: (b + 1) * block], sm_scale=sm,
            )
            torch.testing.assert_close(got.double(), want, rtol=2e-5, atol=2e-5)


class TestSwitchAndWorkerStaging(CustomTestCase):
    def test_env_default_off(self):
        from sglang.srt.environ import envs

        saved = os.environ.pop("SGLANG_DFLASH_WINDOW_HOLE_MASK", None)
        try:
            self.assertFalse(envs.SGLANG_DFLASH_WINDOW_HOLE_MASK.get())
        finally:
            if saved is not None:
                os.environ["SGLANG_DFLASH_WINDOW_HOLE_MASK"] = saved

    def test_buffer_rows(self):
        sa = types.SimpleNamespace(speculative_num_draft_tokens=8)
        self.assertEqual(draft_hole_buffer_rows(6, sa), 6 * 16)
        sa = types.SimpleNamespace(speculative_num_draft_tokens=32)
        self.assertEqual(draft_hole_buffer_rows(3, sa), 96)

    def _fake(self, backend, wls):
        from sglang.srt.layers.radix_attention import AttentionType, RadixAttention

        layers = torch.nn.ModuleList(
            [
                RadixAttention(4, 16, 0.25, 2, i, sliding_window_size=w,
                               attn_type=AttentionType.ENCODER_ONLY)
                for i, w in enumerate(wls)
            ]
        )
        return types.SimpleNamespace(
            draft_model_runner=types.SimpleNamespace(attn_backend=backend),
            draft_model=layers,
            _window_hole_wl=None,
            _window_hole_state=None,
        )

    def _bind(self, fake):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        fake._window_hole_resolve = types.MethodType(
            DFlashWorkerV2._window_hole_resolve, fake
        )
        return types.MethodType(DFlashWorkerV2._stage_window_hole_counts, fake)

    def test_stage_writes_counts_and_zero_tail(self):
        buf = torch.full((64,), 9.0)
        backend = types.SimpleNamespace(_dflash_window_hole_tok=buf, uneven_dcp=False)
        fake = self._fake(backend, [2047] * 5)
        stage = self._bind(fake)
        holes = torch.zeros(2, 10, dtype=torch.bool)
        holes[0, :6] = True
        lengths = torch.tensor([10, 4])
        out = stage(holes, lengths, 4, 2)
        self.assertIs(out, buf)
        self.assertEqual(fake._window_hole_wl, 2047)
        self.assertEqual(buf[:8].tolist(), [6.0] * 4 + [0.0] * 4)
        self.assertEqual(buf[8:].tolist(), [0.0] * 56)

    def test_not_armed_leaves_buffer_alone(self):
        for backend, wls in (
            (types.SimpleNamespace(_dflash_window_hole_tok=None, uneven_dcp=False), [5]),
            (types.SimpleNamespace(_dflash_window_hole_tok=torch.full((8,), 9.0),
                                   uneven_dcp=True), [5]),
            (types.SimpleNamespace(_dflash_window_hole_tok=torch.full((8,), 9.0),
                                   uneven_dcp=False), [5, -1]),
        ):
            fake = self._fake(backend, wls)
            stage = self._bind(fake)
            out = stage(torch.ones(1, 4, dtype=torch.bool), torch.tensor([4]), 2, 1)
            self.assertIsNone(out)
            buf = backend._dflash_window_hole_tok
            if buf is not None:
                self.assertEqual(buf.tolist(), [9.0] * 8)

    def test_oversized_round_is_refused(self):
        backend = types.SimpleNamespace(_dflash_window_hole_tok=torch.zeros(4),
                                        uneven_dcp=False)
        stage = self._bind(self._fake(backend, [7]))
        with self.assertRaises(RuntimeError):
            stage(torch.ones(2, 4, dtype=torch.bool), torch.tensor([4, 4]), 4, 2)


class TestBackendHook(CustomTestCase):
    """FlashInferAttnBackend._forward_paged_window_hole_corrected: the plain
    call's arguments to forward_return_lse, slot 0 of this layer's pool as
    the hole K/V, and the corrected output (fake wrapper = the reference)."""

    def test_hook_calls_and_corrects(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )
        from sglang.srt.layers.radix_attention import AttentionType, RadixAttention

        gen = torch.Generator().manual_seed(9)
        Hq, Hkv, D, block, L, wl = 8, 2, 16, 8, 30, 20
        S = 64
        K = torch.randn(S, Hkv, D, generator=gen)
        V = torch.randn(S, Hkv, D, generator=gen)
        slots = torch.randperm(S - 1, generator=gen)[: L + block] + 1
        holes = torch.zeros(L, dtype=torch.bool)
        holes[:18] = True
        slots[:L][holes] = 0
        layer = RadixAttention(Hq, D, 1.0 / math.sqrt(D), Hkv, 3,
                               sliding_window_size=wl,
                               attn_type=AttentionType.ENCODER_ONLY)
        layer.k_scale_float = 0.5
        layer.v_scale_float = 2.0
        seen = {}

        class _Wrapper:
            def forward_return_lse(self, q, kv, **kw):
                seen.update(kw)
                seen["kv"] = kv
                o, lse2 = _flashinfer_ref(q, kv[0], kv[1], slots, L, block, wl,
                                          kw["sm_scale"], kw["k_scale"], kw["v_scale"])
                return o.float(), lse2.float()

        pool = types.SimpleNamespace(get_kv_buffer=lambda lid: (K, V))
        fake = types.SimpleNamespace(token_to_kv_pool=pool)
        counts = torch.zeros(64)
        counts[:block] = window_hole_counts_per_token(
            holes.unsqueeze(0), torch.tensor([L]), block, wl, 1
        )
        q = torch.randn(block, Hq * D, generator=gen)
        got = FlashInferAttnBackend._forward_paged_window_hole_corrected(
            fake, q, layer, _Wrapper(), False, counts
        )
        self.assertEqual(seen["window_left"], wl)
        self.assertEqual(seen["causal"], False)
        self.assertEqual(seen["logits_soft_cap"], layer.logit_cap)
        self.assertEqual((seen["k_scale"], seen["v_scale"]), (0.5, 2.0))
        self.assertIs(seen["kv"][0], K)
        keep = torch.ones(L + block, dtype=torch.bool)
        keep[:L] = ~holes
        want, _ = _flashinfer_ref(q.view(block, Hq, D), K, V, slots, L, block, wl,
                                  layer.scaling, 0.5, 2.0, keep=keep)
        torch.testing.assert_close(got.double(), want, rtol=2e-5, atol=2e-5)
        with self.assertRaises(RuntimeError):
            FlashInferAttnBackend._forward_paged_window_hole_corrected(
                fake, q, layer, _Wrapper(), False, torch.zeros(4)
            )


if __name__ == "__main__":
    unittest.main()
