# SPDX-License-Identifier: Apache-2.0
"""Input (prompt) logprobs: B8 (one request OOMs the D rank) and B10 (implausible
values on the 27B NVFP4 duo), 23.10.

B8. ``/generate`` with ``return_logprob`` and ``logprob_start_len=0`` over ~3080
tokens: the D rank ran a 943-row extend chunk through the NON-chunked branch of
``LogitsProcessor.forward`` -- ``[943, 248320]`` logits as bf16, then ``.float()``
(894 MiB) and ``logits[input_logprob_indices]`` (another 894 MiB) -- and died with
torch.OutOfMemoryError on both ranks (3080 in ``_copy_logits_to_buffer``, 5090 at
the index copy). Upstream sglang already carries the cure:
``process_input_logprobs_by_chunk`` behind ``*_ENABLE_LOGITS_PROCESSER_CHUNK``
(default False there, chunk 2048 rows -- which would not even have split 943
rows). The fork takes it over unchanged and only moves the two defaults: ON, and
64 rows, so the transient logits footprint per forward is bounded by
``64 x vocab`` no matter how long the prompt is. The chunk count derives only
from ``pruned_states.shape[0]`` and the env value, both rank-uniform, so every TP
rank issues the same number of logits all_gathers (no rank-local free-memory
input -- that would desynchronise the collectives).

B10. Pinned here on CPU: the position <-> token alignment of input logprobs is
exact for the very shapes of the 05:49Z probe (fresh prefill in 256-token chunks
with the logprob start inside the 2816..3072 chunk, and the restart from the
2816-token radix hit), chunked or not, with and without the ratio-weighted
(uneven) vocab all_gather; and ``_compute_max_prefix_len`` never lets a cached
prefix pass ``logprob_start_len`` (upstream semantics: prefix <= start, the fork's
bigram claim is one token stricter). These hold on the base tree too -- they are
the evidence that the logprob layer is NOT where the implausible numbers came from
(see the desk report 27b-logprob-1023.md for the transport finding).
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from flliper.srt import environ as env_mod  # noqa: E402
from flliper.srt.layers import logits_processor as lp_mod  # noqa: E402
from flliper.srt.managers import schedule_batch as sb_mod  # noqa: E402
from flliper.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

V = 50  # vocab
H = 16  # hidden


class _FakeHead:
    def __init__(self, weight):
        self.weight = weight


def _bare_lp(chunk_on: bool, chunk_size: int, tp_gather: bool = False):
    lp = lp_mod.LogitsProcessor.__new__(lp_mod.LogitsProcessor)
    torch.nn.Module.__init__(lp)
    lp.vocab_size = V
    lp.logit_scale = None
    lp.use_attn_tp_group = False
    lp.use_fp32_lm_head = False
    lp.rl_on_policy_target = None
    lp.do_tensor_parallel_all_gather = tp_gather
    lp.do_tensor_parallel_all_gather_dp_attn = False
    lp.final_logit_softcapping = None
    lp.verify_local_vocab = False
    lp._logits_gatherer = mock.Mock(side_effect=AssertionError("even gather called"))
    lp.enable_logprobs_chunk = chunk_on
    lp.logprobs_chunk_size = chunk_size
    return lp


def _meta(ext_lens, start_lens, token_ids, top_k=None):
    pruned = [e - s for e, s in zip(ext_lens, start_lens)]
    return lp_mod.LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        extend_return_logprob=any(p > 0 for p in pruned),
        extend_return_top_logprob=top_k is not None and any(k > 0 for k in top_k),
        extend_token_ids_logprob=False,
        extend_seq_lens=torch.tensor(ext_lens),
        extend_seq_lens_cpu=list(ext_lens),
        extend_logprob_start_lens_cpu=list(start_lens),
        extend_logprob_pruned_lens_cpu=pruned,
        top_logprobs_nums=list(top_k) if top_k is not None else [0] * len(ext_lens),
        token_ids_logprobs=[None] * len(ext_lens),
        extend_input_logprob_token_ids_gpu=torch.tensor(token_ids, dtype=torch.int64),
    )


def _logprob_token_ids(origin, prefix_len, end, logprob_start_len, extend_len,
                       extend_logprob_start_len):
    """The token-id rule of ScheduleBatch.prepare_for_extend, verbatim
    (global index slid by one, padded with 0 to the pruned length)."""
    g0 = max(prefix_len, logprob_start_len)
    ids = origin[g0 + 1 : end + 1]
    return ids + [0] * (extend_len - extend_logprob_start_len - len(ids))


class _Model:
    """A deterministic stand-in for the network: hidden(p) depends on the
    prompt up to p only, so a fresh prefill and a restart from any prefix see
    the same rows -- which is what a correct KV/state restore guarantees."""

    def __init__(self, n_tokens: int, seed: int = 7):
        g = torch.Generator().manual_seed(seed)
        self.origin = torch.randint(0, V, (n_tokens,), generator=g).tolist()
        self.hidden = torch.randn(n_tokens, H, generator=g)
        self.weight = torch.randn(V, H, generator=g)
        ref = torch.log_softmax(self.hidden @ self.weight.T, dim=-1)
        # ref_lp[p] = logprob of token p+1 given tokens <= p
        self.ref_lp = ref[torch.arange(n_tokens - 1), torch.tensor(self.origin[1:])]
        self.ref_full = ref

    def run(self, lp, chunks, logprob_start_len, lm_head=None, top_k=None):
        """One forward per (prefix_len, extend_len) chunk of ONE request;
        returns the concatenated input_token_logprobs (positions >= start)."""
        out, tops = [], []
        for prefix_len, extend_len in chunks:
            end = prefix_len + extend_len
            s = sb_mod.compute_extend_logprob_start_len(
                logprob_start_len=logprob_start_len, prefix_len=prefix_len,
                extend_len=extend_len, full_untruncated_fill_len=len(self.origin))
            ids = _logprob_token_ids(self.origin, prefix_len, end,
                                     logprob_start_len, extend_len, s)
            meta = _meta([extend_len], [s], ids,
                         top_k=[top_k] if top_k is not None else None)
            res = lp.forward(None, self.hidden[prefix_len:end],
                             lm_head or _FakeHead(self.weight), meta)
            if res.input_token_logprobs is not None:
                out.append(res.input_token_logprobs)
            if top_k is not None and res.input_top_logprobs_val is not None:
                tops.extend(res.input_top_logprobs_val)
        return torch.cat(out), tops


class TestB10AlignmentOnTheProbeShapes(CustomTestCase):
    """05:49Z probe: 3080 tokens, logprob_start_len = 3080 - 256 = 2824."""

    N, START = 3080, 2824

    def setUp(self):
        self.m = _Model(self.N)
        self.want = self.m.ref_lp[self.START:]  # positions 2824..3078 -> 255 values

    def _fresh(self):  # run 0: d_prefix=0, 256-token chunks (D.log #969 EXTENT)
        return [(p, min(256, self.N - p)) for p in range(0, self.N, 256)]

    def _restart(self):  # runs 1/2: radix hit 2816, then 256 + 8
        return [(2816, 256), (3072, 8)]

    def _restart_one_chunk(self):  # pdflip-6-9: 2816 + 264
        return [(2816, 264)]

    def _check(self, lp, lm_head=None):
        for chunks in (self._fresh(), self._restart(), self._restart_one_chunk()):
            got, _ = self.m.run(lp, chunks, self.START, lm_head=lm_head)
            # the last row of the prompt has no next token -> padded id 0, dropped
            self.assertEqual(got.shape[0], self.N - self.START)
            torch.testing.assert_close(got[:-1], self.want, rtol=1e-5, atol=1e-5)

    def test_unchunked(self):
        self._check(_bare_lp(chunk_on=False, chunk_size=2048))

    def test_chunked_various_sizes(self):
        for c in (1, 3, 7, 64, 255, 256):
            self._check(_bare_lp(chunk_on=True, chunk_size=c))

    def test_uneven_vocab_gather_two_ranks(self):
        """Rank 0's view of the ratio-weighted vocab all_gather (58:25-like
        split), with rank 1's shard computed from the same rows -- the multi-row
        gather must hand back the rows in order."""
        s0, s1 = 34, 16
        m = self.m
        head = _FakeHead(m.weight[:s0])
        head.vocab_partition_sizes = [s0, s1]
        seen = {}
        real_lm_head = lp_mod.LogitsProcessor._compute_lm_head

        def lm_head_spy(self_, hs, lm_head, bias=None):
            seen["hs"] = hs
            return real_lm_head(self_, hs, lm_head, bias)

        def fake_all_gather(padded, dim=-1):
            other = seen["hs"] @ m.weight[s0:].T
            pad1 = padded.new_zeros(padded.shape)
            pad1[..., :s1] = other
            return torch.cat([padded, pad1], dim=-1)

        grp = SimpleNamespace(rank_in_group=0)
        with mock.patch.object(lp_mod.LogitsProcessor, "_compute_lm_head", lm_head_spy), \
             mock.patch.object(lp_mod, "tensor_model_parallel_all_gather", fake_all_gather), \
             mock.patch("flliper.srt.distributed.get_tp_group", return_value=grp):
            for on, c in ((False, 2048), (True, 64), (True, 5)):
                self._check(_bare_lp(chunk_on=on, chunk_size=c, tp_gather=True), head)

    def test_top_logprobs_identical_chunked_vs_unchunked(self):
        a, ta = self.m.run(_bare_lp(False, 2048), self._restart(), self.START, top_k=3)
        b, tb = self.m.run(_bare_lp(True, 7), self._restart(), self.START, top_k=3)
        # same values up to matmul blocking (different row counts per GEMM)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        self.assertEqual(len(ta), len(tb))
        for x, y in zip(ta, tb):
            torch.testing.assert_close(torch.as_tensor(x), torch.as_tensor(y),
                                       rtol=1e-5, atol=1e-5)

    def test_start_zero_whole_prompt(self):
        """B8's request shape: start=0 over the whole prompt, fresh, 943-row
        chunks as in the 05:43Z boot (#969 EXTENT 0..943)."""
        m = _Model(2900, seed=11)
        chunks = [(p, min(943, 2900 - p)) for p in range(0, 2900, 943)]
        for on, c in ((False, 2048), (True, 64)):
            got, _ = m.run(_bare_lp(on, c), chunks, 0)
            torch.testing.assert_close(got[:-1], m.ref_lp, rtol=1e-5, atol=1e-5)


class TestB10MixedBatch(CustomTestCase):
    def test_three_requests_mixed_starts(self):
        """Batch of three: logprobs from 0, a chunk BEFORE its logprob start
        (extend_len == start: one sample row, no logprob row), and a start
        inside the chunk -- chunked equals unchunked equals reference."""
        g = torch.Generator().manual_seed(5)
        w = torch.randn(V, H, generator=g)
        ext = [40, 30, 50]
        starts = [0, 30, 17]
        hs = torch.randn(sum(ext), H, generator=g)
        ids = torch.randint(0, V, (sum(e - s for e, s in zip(ext, starts)),),
                            generator=g).tolist()
        ref_rows = []
        pt = 0
        for e, s in zip(ext, starts):
            ref_rows.append(hs[pt + s : pt + e])
            pt += e
        ref = torch.log_softmax(torch.cat(ref_rows) @ w.T, dim=-1)
        want = ref[torch.arange(ref.shape[0]), torch.tensor(ids)]
        outs = []
        for on, c in ((False, 2048), (True, 1), (True, 9), (True, 64)):
            r = _bare_lp(on, c).forward(None, hs, _FakeHead(w), _meta(ext, starts, ids))
            torch.testing.assert_close(r.input_token_logprobs, want, rtol=1e-5, atol=1e-5)
            # the sampled rows (one per request) are the last row of each request
            last = torch.tensor([39, 69, 119])
            torch.testing.assert_close(r.next_token_logits, (hs[last] @ w.T).float(),
                                       rtol=1e-5, atol=1e-5)
            outs.append(r.input_token_logprobs)
        for o in outs[1:]:
            torch.testing.assert_close(o, outs[0], rtol=1e-5, atol=1e-5)


class TestB8ChunkDefaults(CustomTestCase):
    def test_env_defaults_are_on_and_bounded(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLLIPER_ENABLE_LOGITS_PROCESSER_CHUNK", None)
            os.environ.pop("FLLIPER_LOGITS_PROCESSER_CHUNK_SIZE", None)
            self.assertTrue(env_mod.envs.FLLIPER_ENABLE_LOGITS_PROCESSER_CHUNK.get())
            self.assertEqual(env_mod.envs.FLLIPER_LOGITS_PROCESSER_CHUNK_SIZE.get(), 64)

    def _construct(self):
        sa = SimpleNamespace(enable_dp_lm_head=False, enable_fp32_lm_head=False,
                             enable_mis=False, rl_on_policy_target=None,
                             speculative_algorithm="DFLASH",
                             speculative_cross_algorithm=False)
        par = SimpleNamespace(tp_size=2, attn_dp_size=1, attn_tp_size=2)
        with mock.patch.object(lp_mod, "get_server_args", return_value=sa), \
             mock.patch.object(lp_mod, "get_parallel", return_value=par), \
             mock.patch.object(lp_mod, "tp_vocab_ratios", return_value=None), \
             mock.patch.object(lp_mod.triton_symm_mem_ag, "recommended_max_tokens",
                               return_value=128), \
             mock.patch.object(lp_mod.triton_symm_mem_ag, "MultimemAllGatherer"), \
             mock.patch.dict(os.environ, {}, clear=False):
            for k in ("FLLIPER_ENABLE_LOGITS_PROCESSER_CHUNK",
                      "FLLIPER_LOGITS_PROCESSER_CHUNK_SIZE",
                      lp_mod.VERIFY_LOCAL_VOCAB_ENV):
                os.environ.pop(k, None)
            return lp_mod.LogitsProcessor(SimpleNamespace(vocab_size=V))

    def test_constructed_processor_chunks_by_default(self):
        lp = self._construct()
        self.assertTrue(lp.enable_logprobs_chunk)
        self.assertEqual(lp.logprobs_chunk_size, 64)

    def test_943_row_extend_never_materialises_full_logits(self):
        """The 05:43Z crash shape: one request, 943 rows, start 0. With the
        defaults no single _get_logits call sees more than 64 rows."""
        lp = self._construct()
        lp.do_tensor_parallel_all_gather = False  # single-process stand-in
        g = torch.Generator().manual_seed(1)
        w = torch.randn(V, H, generator=g)
        hs = torch.randn(943, H, generator=g)
        ids = torch.randint(0, V, (943,), generator=g).tolist()
        rows = []
        real = lp_mod.LogitsProcessor._get_logits

        def spy(self_, h, *a, **k):
            rows.append(h.shape[0])
            return real(self_, h, *a, **k)

        with mock.patch.object(lp_mod.LogitsProcessor, "_get_logits", spy):
            r = lp.forward(None, hs, _FakeHead(w), _meta([943], [0], ids))
        self.assertEqual(max(rows), 64)
        self.assertEqual(len(rows), -(-943 // 64))
        ref = torch.log_softmax(hs @ w.T, dim=-1)[torch.arange(943), torch.tensor(ids)]
        torch.testing.assert_close(r.input_token_logprobs, ref, rtol=1e-5, atol=1e-5)


class TestB10PrefixCapSemantics(CustomTestCase):
    """Upstream: max_prefix_len = min(input_len - 1, logprob_start_len) raw
    tokens, so the cached prefix never passes the logprob start. The fork's
    bigram claim raises the raw limit to input_len (N raw = N-1 units) and then
    applies the same cap -- min(N, start) raw = start-1 units of KV."""

    def _cap(self, claim, n, start, want_logprob=True):
        req = SimpleNamespace(return_logprob=want_logprob, logprob_start_len=start)
        with mock.patch("flliper.srt.pdflip.handback_claim.handback_bigram_claim",
                        return_value=claim):
            return sb_mod.Req._compute_max_prefix_len(req, n)

    def test_cap_never_passes_the_start(self):
        for claim in (False, True):
            self.assertEqual(self._cap(claim, 3080, 2824), 2824)
            self.assertEqual(self._cap(claim, 3080, 0), 0)
        self.assertEqual(self._cap(False, 3080, -1), 3079)
        self.assertEqual(self._cap(True, 3080, -1), 3080)
        self.assertEqual(self._cap(False, 3080, 2824, want_logprob=False), 3079)

    def test_extend_logprob_start_relative_to_any_prefix(self):
        f = sb_mod.compute_extend_logprob_start_len
        self.assertEqual(f(logprob_start_len=2824, prefix_len=2816, extend_len=256,
                           full_untruncated_fill_len=3080), 8)
        self.assertEqual(f(logprob_start_len=2824, prefix_len=2560, extend_len=256,
                           full_untruncated_fill_len=3080), 256)  # chunk before start
        self.assertEqual(f(logprob_start_len=2824, prefix_len=3072, extend_len=8,
                           full_untruncated_fill_len=3080), 0)
        self.assertEqual(f(logprob_start_len=-1, prefix_len=3072, extend_len=8,
                           full_untruncated_fill_len=3080), 8)


class TestB10RowHashInstrument(CustomTestCase):
    def test_digest_is_bitwise_and_blockwise(self):
        g = torch.Generator().manual_seed(2)
        a = torch.randn(130, 16, generator=g).to(torch.bfloat16)
        b = a.clone()
        self.assertEqual(lp_mod._rowhash_blocks(a), lp_mod._rowhash_blocks(b))
        self.assertEqual(len(lp_mod._rowhash_blocks(a).split(",")), 3)  # 64+64+2
        b[70, 3] = b[70, 3] + 1
        da, db = lp_mod._rowhash_blocks(a).split(","), lp_mod._rowhash_blocks(b).split(",")
        self.assertEqual(da[0], db[0])
        self.assertNotEqual(da[1], db[1])  # only the block holding row 70
        self.assertEqual(da[2], db[2])

    def test_forward_logs_one_line_per_logprob_forward(self):
        m = _Model(300, seed=3)
        with self.assertLogs(lp_mod.logger.name, level="INFO") as cm:
            m.run(_bare_lp(True, 64), [(0, 300)], 10)
        lines = [r for r in cm.output if "LOGPROB-ROWHASH" in r]
        self.assertEqual(len(lines), 1)
        self.assertIn("rows=290", lines[0])
        with mock.patch.dict(os.environ, {"FLLIPER_LOGPROB_ROWHASH": "0"}):
            with self.assertNoLogs(lp_mod.logger.name, level="INFO"):
                m.run(_bare_lp(False, 2048), [(0, 300)], 10)


if __name__ == "__main__":
    unittest.main()
