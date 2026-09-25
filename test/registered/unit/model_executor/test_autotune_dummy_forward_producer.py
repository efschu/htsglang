# SPDX-License-Identifier: Apache-2.0
"""The warmup flashinfer-autotune dummy forward respects the group's forward kind.

rc8b cu130 acceptance (RC8 d342caa832, 27b-nvfp4 native-mixed, 25.09. 17:29Z):
group P (``--speculative-draft-kv-only``, DFLASH flag set carried for the
drafter identity, PP3) died in Scheduler.__init__ -> EagerRunner.warmup ->
_flashinfer_autotune -> _dummy_run -> gdn_backend.forward_extend
``assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)``.
_dummy_run chose TARGET_VERIFY from ``spec_algorithm.is_speculative()`` alone,
but the draft-KV producer never verifies and has no speculative mamba state
(#1233 FIX 4). The autotune only ran because the 5090 rank resolved an
autotunable FP4 backend (native-mixed -> flashinfer_cutlass).

Desk, no GPU: the dummy forward is stopped at ForwardBatch construction and
its forward mode / token count / spec_info are read there.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import base_runner as BR
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_DRAFT = 8


class _Stop(Exception):
    pass


def _mr(*, speculative=True, producer=False, draft=False):
    spec = types.SimpleNamespace(
        is_speculative=lambda: speculative,
        supports_target_verify_for_draft=lambda: True,
        get_num_tokens_per_bs_for_target_verify=lambda n, d: n,
        is_eagle=lambda: False,
        is_standalone=lambda: False,
    )
    return types.SimpleNamespace(
        spec_algorithm=spec,
        is_draft_kv_only_producer=producer,
        is_draft_model_runner=draft,
        is_generation=True,
        server_args=types.SimpleNamespace(
            speculative_num_draft_tokens=NUM_DRAFT,
            enable_return_hidden_states=False,
            pp_size=1,
            enable_lora=False,
            dp_size=1,
        ),
        attn_backend=types.SimpleNamespace(get_cuda_graph_seq_len_fill_value=lambda: 1),
        device="cpu",
        model=None,
        canary_manager=None,
    )


def _buffers(n=64, bs=8):
    return types.SimpleNamespace(
        input_ids=torch.zeros(n, dtype=torch.int64),
        positions=torch.zeros(n, dtype=torch.int64),
        out_cache_loc=torch.zeros(n, dtype=torch.int64),
        next_token_logits_buffer=None,
        mrope_positions=torch.zeros(3, n, dtype=torch.int64),
        req_pool_indices=torch.zeros(bs, dtype=torch.int64),
        seq_lens=torch.ones(bs, dtype=torch.int32),
        seq_lens_cpu=torch.ones(bs, dtype=torch.int32),
        encoder_lens=None,
        num_token_non_padded=torch.zeros((), dtype=torch.int32),
        global_num_tokens_gpu=torch.zeros(1, dtype=torch.int32),
        global_num_tokens_for_logprob_gpu=torch.zeros(1, dtype=torch.int32),
        custom_mask=None,
        pp_proxy_tensors={},
        ngram_embedding_info=None,
    )


def _dummy(mr, bs=4, override=None):
    """Run BaseRunner._dummy_run until ForwardBatch; return its kwargs and
    whether a verify input was built."""
    seen = {}
    verify_calls = []

    def fb(**kw):
        seen.update(kw)
        raise _Stop

    runner = types.SimpleNamespace(model_runner=mr)
    with mock.patch.object(BR, "ForwardBatch", side_effect=fb), mock.patch.object(
        BR, "require_mlp_tp_gather", return_value=False
    ), mock.patch.object(BR, "require_gathered_buffer", return_value=False), mock.patch.object(
        BR, "require_attn_tp_gather", return_value=False
    ), mock.patch.object(
        BR, "create_dummy_verify_input", side_effect=lambda *a, **k: verify_calls.append(a) or None
    ):
        with self_raises(_Stop):
            BR.BaseRunner._dummy_run(runner, batch_size=bs, buffers=_buffers(), forward_mode_override=override)
    return seen, verify_calls


class self_raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError("ForwardBatch was never reached")
        return issubclass(t, self.exc)


class TestPredicate(unittest.TestCase):
    def test_runs_target_verify(self):
        self.assertTrue(BR.runs_target_verify(_mr(speculative=True)))
        self.assertFalse(BR.runs_target_verify(_mr(speculative=True, producer=True)))
        self.assertFalse(BR.runs_target_verify(_mr(speculative=False)))
        # the producer's own DRAFT runner reports is_draft_kv_only_producer=False
        self.assertTrue(BR.runs_target_verify(_mr(speculative=True, draft=True)))
        # stand-ins without the property (non-ModelRunner callers) keep the old answer
        mr = _mr(speculative=True)
        del mr.is_draft_kv_only_producer
        self.assertTrue(BR.runs_target_verify(mr))


class TestDummyRun(unittest.TestCase):
    def test_producer_never_builds_a_target_verify_forward(self):
        """The rc8b P-group killer: spec flag set, but prefill-only group."""
        seen, verify = _dummy(_mr(producer=True), bs=4)
        self.assertNotEqual(seen["forward_mode"], ForwardMode.TARGET_VERIFY)
        self.assertEqual(seen["input_ids"].shape[0], 4)  # 1 token per request
        self.assertIsNone(seen["spec_info"])
        self.assertEqual(verify, [])

    def test_producer_extend_override_is_honoured(self):
        seen, _ = _dummy(_mr(producer=True), bs=4, override=ForwardMode.EXTEND)
        self.assertEqual(seen["forward_mode"], ForwardMode.EXTEND)
        self.assertEqual(seen["extend_num_tokens"], 4)

    def test_real_spec_target_runner_still_verifies(self):
        seen, verify = _dummy(_mr(producer=False), bs=2)
        self.assertEqual(seen["forward_mode"], ForwardMode.TARGET_VERIFY)
        self.assertEqual(seen["input_ids"].shape[0], 2 * NUM_DRAFT)
        self.assertEqual(len(verify), 1)

    def test_non_spec_runner_decodes(self):
        seen, verify = _dummy(_mr(speculative=False), bs=3)
        self.assertEqual(seen["forward_mode"], ForwardMode.DECODE)
        self.assertEqual(verify, [])


class TestAutotuneEntry(unittest.TestCase):
    def _autotune(self, mr):
        calls = []
        runner = types.SimpleNamespace(
            model_runner=mr, _dummy_run=lambda **kw: calls.append(kw)
        )
        with mock.patch.object(BR, "run_flashinfer_autotune_forward", side_effect=lambda m, fn, **k: fn()):
            BR.BaseRunner._flashinfer_autotune(runner, buffers=object(), batch_size=4)
        return calls[0]

    def test_producer_autotunes_on_extend(self):
        self.assertEqual(self._autotune(_mr(producer=True))["forward_mode_override"], ForwardMode.EXTEND)

    def test_other_runners_keep_their_default_mode(self):
        self.assertIsNone(self._autotune(_mr(producer=False))["forward_mode_override"])
        self.assertIsNone(self._autotune(_mr(speculative=False))["forward_mode_override"])

    def test_eager_autotune_buffers_are_one_token_per_request_on_the_producer(self):
        got = {}
        runner = types.SimpleNamespace(
            model_runner=_mr(producer=True),
            _eager_max_bs=16,
            _alloc_dummy_decode_buffers=lambda bs, num_tokens_per_bs: got.update(bs=bs, n=num_tokens_per_bs),
        )
        EagerRunner._autotune_buffers(runner)
        self.assertEqual(got, {"bs": 16, "n": 1})
        runner.model_runner = _mr(producer=False)
        EagerRunner._autotune_buffers(runner)
        self.assertEqual(got["n"], NUM_DRAFT)


if __name__ == "__main__":
    unittest.main()
