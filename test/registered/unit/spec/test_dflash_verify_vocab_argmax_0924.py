# SPDX-License-Identifier: Apache-2.0
"""DFLASH verify: vocab-parallel argmax instead of the [rows, vocab] logits
all_gather (SGLANG_DFLASH_VERIFY_VOCAB_ARGMAX, default off).

Pinned on CPU:

* ``vocab_parallel_argmax`` over simulated TP shards equals ``torch.argmax`` on
  the concatenated logits -- even and uneven shards, padding columns on the last
  shard, bf16 and fp32, ties inside a shard and ACROSS shards (first maximal
  index wins, like torch.argmax), all-negative rows (a padding 0.0 must not
  win), empty shards;
* the pack/unpack of the fp32 value bits is lossless (negative, -inf, -0.0);
* LogitsProcessor: with the switch the target-verify ``_get_logits`` returns the
  raw local shard and never touches the gatherer; any other mode, or the switch
  off, is the stock path; ``finish_local_verify_logits`` reproduces the stock
  tail (gather, vocab slice, fp32, softcap) byte for byte; the switch is decided
  once at construction and only for a non-cross-algorithm DFLASH server with a
  plain TP gather;
* the worker: which rounds may take the argmax (plain greedy, no selector
  sample, grammar, logprobs, added vocab, softcap), and the wiring order in
  ``forward_batch_generation`` (after the verify forward, before the logits
  adjustments, consumed by the greedy accept only).
"""

import ast
import inspect
import os
import textwrap
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

from sglang.srt.layers import logits_processor as lp_mod  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative import dflash_worker_v2 as w  # noqa: E402


def _split(full: torch.Tensor, widths, pad_last: int = 0):
    """Shards of ``full`` (rank order) with ``pad_last`` zero padding columns
    appended to the last shard; returns (shards, num_org, starts)."""
    shards, num_org, starts = [], [], []
    start = 0
    for r, wdt in enumerate(widths):
        part = full[:, start : start + wdt]
        if r == len(widths) - 1 and pad_last:
            part = torch.cat(
                [part, torch.zeros(full.shape[0], pad_last, dtype=full.dtype)], dim=1
            )
        shards.append(part.contiguous())
        num_org.append(wdt)
        starts.append(start)
        start += wdt
    assert start == full.shape[1]
    return shards, num_org, starts


def _vp_argmax_all_ranks(shards, num_org, starts):
    packed = [w.vocab_shard_pack(s, n, st) for s, n, st in zip(shards, num_org, starts)]
    gathered = torch.stack(packed, dim=0)
    outs = []
    for r, (s, n, st) in enumerate(zip(shards, num_org, starts)):
        outs.append(w.vocab_parallel_argmax(s, n, st, lambda t, g=gathered: g))
    return outs


class TestVocabParallelArgmax(CustomTestCase):
    def _check(self, full, widths, pad_last=0):
        want = torch.argmax(full.float(), dim=-1)
        outs = _vp_argmax_all_ranks(*_split(full, widths, pad_last))
        for o in outs:  # every rank reaches the same answer
            torch.testing.assert_close(o, want, rtol=0, atol=0)

    def test_random_even_and_uneven(self):
        g = torch.Generator().manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            for widths in ((300, 300, 300), (480, 210, 210), (1, 5, 894), (7, 890, 3)):
                full = torch.randn(16, sum(widths), generator=g).to(dtype)
                self._check(full, widths)
                self._check(full, widths, pad_last=37)

    def test_ties_inside_and_across_shards(self):
        full = torch.full((4, 30), -5.0)
        full[0, 3] = full[0, 7] = 2.0            # inside shard 0 -> 3
        full[1, 12] = full[1, 25] = 2.0          # shard 1 and shard 2 -> 12
        full[2, 9] = full[2, 10] = 1.0           # last col of shard 0 vs first of 1
        full[3, :] = 0.5                         # all equal -> 0
        for dtype in (torch.float32, torch.bfloat16):
            self._check(full.to(dtype), (10, 10, 10))
        # bf16 rounding makes distinct fp32 values tie; the stock path argmaxes
        # the SAME rounded values, so ties must follow the rounded ones
        f = torch.full((1, 6), -1.0)
        f[0, 1] = 1.0001
        f[0, 4] = 1.0002
        self._check(f.to(torch.bfloat16), (3, 3))

    def test_padding_zero_never_wins_over_negative_logits(self):
        full = torch.full((2, 12), -3.0)
        full[0, 11] = -1.0
        full[1, 0] = -2.0
        self._check(full, (4, 4, 4), pad_last=20)

    def test_empty_shard_and_neg_inf(self):
        full = torch.randn(3, 20)
        full[1, :] = float("-inf")
        full[1, 13] = -1e30
        shards, num_org, starts = _split(full, (10, 10))
        shards.insert(1, torch.zeros(3, 4))      # a rank holding no real vocab
        num_org.insert(1, 0)
        starts.insert(1, 10)
        want = torch.argmax(full, dim=-1)
        for o in _vp_argmax_all_ranks(shards, num_org, starts):
            torch.testing.assert_close(o, want, rtol=0, atol=0)

    def test_pack_roundtrip_bits(self):
        vals = torch.tensor([-0.0, 0.0, -1.5, float("-inf"), 3.4e38, -3.4e38, 1e-38])
        logits = torch.stack([torch.full((5,), -9.0)] * len(vals))
        logits[:, 2] = vals
        packed = w.vocab_shard_pack(logits, 5, 1000)
        self.assertEqual(packed.dtype, torch.int64)
        self.assertEqual(tuple(packed.shape), (len(vals), 2))
        back = packed[:, 1].to(torch.int32).view(torch.float32)
        self.assertTrue(torch.equal(back.view(torch.int32), torch.maximum(vals, torch.tensor(-9.0)).view(torch.int32)))
        self.assertTrue(bool((packed[:, 0] >= 1000).all()))


class _FakeHead:
    def __init__(self, weight):
        self.weight = weight


def _bare_lp(**over):
    lp = lp_mod.LogitsProcessor.__new__(lp_mod.LogitsProcessor)
    torch.nn.Module.__init__(lp)
    lp.vocab_size = 12
    lp.logit_scale = None
    lp.use_attn_tp_group = False
    lp.use_fp32_lm_head = False
    lp.rl_on_policy_target = None
    lp.do_tensor_parallel_all_gather = True
    lp.do_tensor_parallel_all_gather_dp_attn = False
    lp.final_logit_softcapping = None
    lp.verify_local_vocab = True
    lp._logits_gatherer = mock.Mock(side_effect=AssertionError("gather called"))
    for k, v in over.items():
        setattr(lp, k, v)
    return lp


class TestLogitsProcessorLocalVocab(CustomTestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(3)
        self.hs = torch.randn(8, 16, generator=g)
        self.w_local = torch.randn(4, 16, generator=g)
        self.head = _FakeHead(self.w_local)

    def _meta(self, mode):
        return SimpleNamespace(forward_mode=mode, next_token_logits_buffer=None)

    def test_verify_returns_raw_local_shard_without_gather(self):
        lp = _bare_lp()
        out = lp._get_logits(self.hs, self.head, self._meta(ForwardMode.TARGET_VERIFY))
        torch.testing.assert_close(out, self.hs @ self.w_local.T, rtol=0, atol=0)
        lp._logits_gatherer.assert_not_called()

    def test_other_modes_and_switch_off_gather(self):
        full = torch.randn(8, 12)
        for mode, local in ((ForwardMode.DECODE, True), (ForwardMode.TARGET_VERIFY, False)):
            lp = _bare_lp(verify_local_vocab=local,
                          _logits_gatherer=mock.Mock(return_value=full.clone()))
            out = lp._get_logits(self.hs, self.head, self._meta(mode))
            lp._logits_gatherer.assert_called_once()
            self.assertEqual(out.dtype, torch.float32)
            torch.testing.assert_close(out, full, rtol=0, atol=0)

    @mock.patch.object(lp_mod, "_is_cpu", True)   # torch softcap, no Triton on CPU
    def test_finish_equals_stock_tail(self):
        full_padded = torch.randn(8, 14).to(torch.bfloat16)   # 2 padding columns
        for cap in (None, 3.0):
            lp = _bare_lp(final_logit_softcapping=cap,
                          _logits_gatherer=mock.Mock(return_value=full_padded.clone()))
            lp_stock = _bare_lp(final_logit_softcapping=cap, verify_local_vocab=False,
                                _logits_gatherer=mock.Mock(return_value=full_padded.clone()))
            got = lp.finish_local_verify_logits(torch.zeros(8, 5, dtype=torch.bfloat16), self.head)
            want = lp_stock._get_logits(self.hs, self.head, self._meta(ForwardMode.TARGET_VERIFY))
            self.assertEqual(got.dtype, torch.float32)
            self.assertEqual(tuple(got.shape), (8, 12))
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_finish_uses_the_uneven_gather_for_ratio_heads(self):
        lp = _bare_lp()
        head = _FakeHead(self.w_local)
        head.vocab_partition_sizes = [4, 4, 6]
        with mock.patch.object(lp_mod.LogitsProcessor, "_gather_uneven_vocab_logits",
                               autospec=True, return_value=torch.ones(8, 14)) as g:
            out = lp.finish_local_verify_logits(torch.zeros(8, 4), head)
        g.assert_called_once()
        self.assertEqual(tuple(out.shape), (8, 12))

    def _construct(self, env, algo="DFLASH", cross=False, tp=3):
        sa = SimpleNamespace(enable_dp_lm_head=False, enable_fp32_lm_head=False,
                             enable_mis=False, rl_on_policy_target=None,
                             speculative_algorithm=algo,
                             speculative_cross_algorithm=cross)
        par = SimpleNamespace(tp_size=tp, attn_dp_size=1, attn_tp_size=tp)
        env_patch = {lp_mod.VERIFY_LOCAL_VOCAB_ENV: env} if env is not None else {}
        with mock.patch.object(lp_mod, "get_server_args", return_value=sa), \
             mock.patch.object(lp_mod, "get_parallel", return_value=par), \
             mock.patch.object(lp_mod, "tp_vocab_ratios", return_value=None), \
             mock.patch.object(lp_mod.triton_symm_mem_ag, "recommended_max_tokens",
                               return_value=128), \
             mock.patch.object(lp_mod.triton_symm_mem_ag, "MultimemAllGatherer"), \
             mock.patch.dict(os.environ, env_patch, clear=False):
            if env is None:
                os.environ.pop(lp_mod.VERIFY_LOCAL_VOCAB_ENV, None)
            return lp_mod.LogitsProcessor(SimpleNamespace(vocab_size=12))

    def test_decided_at_construction(self):
        self.assertFalse(self._construct(None).verify_local_vocab)
        self.assertTrue(self._construct("1").verify_local_vocab)
        self.assertFalse(self._construct("1", algo="EAGLE").verify_local_vocab)
        self.assertFalse(self._construct("1", cross=True).verify_local_vocab)
        self.assertFalse(self._construct("1", tp=1).verify_local_vocab)


class TestWorkerSide(CustomTestCase):
    def test_plain_greedy_truth_table(self):
        base = dict(is_all_greedy=True, has_custom_logit_processor=False,
                    acc_linear_penalties=None, penalizer_orchestrator=None,
                    vocab_mask=None, logit_bias=None)
        self.assertTrue(w._verify_plain_greedy(None))
        self.assertTrue(w._verify_plain_greedy(SimpleNamespace(**base)))
        for k, v in (("is_all_greedy", False), ("has_custom_logit_processor", True),
                     ("acc_linear_penalties", torch.zeros(1)),
                     ("penalizer_orchestrator", SimpleNamespace(is_required=True)),
                     ("vocab_mask", torch.zeros(1)), ("logit_bias", torch.zeros(1))):
            si = SimpleNamespace(**{**base, k: v})
            self.assertFalse(w._verify_plain_greedy(si), k)
        self.assertTrue(w._verify_plain_greedy(SimpleNamespace(
            **{**base, "penalizer_orchestrator": SimpleNamespace(is_required=False)})))

    def test_eligibility(self):
        fake = SimpleNamespace(_selector_sample=None)
        head = SimpleNamespace(shard_indices=SimpleNamespace(num_added_elements=0))
        lp = SimpleNamespace(final_logit_softcapping=None)
        batch = SimpleNamespace(has_grammar=False, return_logprob=False)
        ok = lambda **kw: w.DFlashWorkerV2._verify_vocab_argmax_eligible(
            kw.get("self", fake), kw.get("batch", batch), None,
            kw.get("head", head), kw.get("lp", lp))
        self.assertTrue(ok())
        self.assertFalse(ok(self=SimpleNamespace(_selector_sample=(1, 2))))
        self.assertFalse(ok(batch=SimpleNamespace(has_grammar=True, return_logprob=False)))
        self.assertFalse(ok(batch=SimpleNamespace(has_grammar=False, return_logprob=True)))
        self.assertFalse(ok(lp=SimpleNamespace(final_logit_softcapping=30.0)))
        self.assertFalse(ok(head=SimpleNamespace()))
        self.assertFalse(ok(head=SimpleNamespace(
            shard_indices=SimpleNamespace(num_added_elements=2))))

    def test_processor_lookup(self):
        lp_on = SimpleNamespace(verify_local_vocab=True)
        mk = lambda lp: SimpleNamespace(target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=SimpleNamespace(logits_processor=lp))))
        f = w.DFlashWorkerV2._verify_local_vocab_processor
        self.assertIs(f(mk(lp_on)), lp_on)
        self.assertIsNone(f(mk(SimpleNamespace(verify_local_vocab=False))))
        self.assertIsNone(f(mk(None)))

    def test_wiring_order(self):
        src = textwrap.dedent(inspect.getsource(w.DFlashWorkerV2.forward_batch_generation))
        keys = [
            "self.target_worker.forward_batch_generation(",
            "_vlp = self._verify_local_vocab_processor()",
            "vocab_target_predict = vocab_parallel_argmax(",
            "_vlp.finish_local_verify_logits(",
            "if sampling_info is not None and vocab_target_predict is None:",
            "target_predict = vocab_target_predict",
            "self._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY, target_predict)",
        ]
        pos = [src.find(k) for k in keys]
        self.assertTrue(all(p >= 0 for p in pos), list(zip(keys, pos)))
        self.assertEqual(pos, sorted(pos), list(zip(keys, pos)))
        # the selector / sampling branches still read the (finished) full logits
        tree = ast.parse(src)
        reads = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                 and n.attr == "next_token_logits"]
        self.assertGreater(len(reads), 3)


if __name__ == "__main__":
    unittest.main()
