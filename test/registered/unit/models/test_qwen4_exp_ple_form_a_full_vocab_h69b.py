"""fnFL2 H69b (SGLANG_WEG2_FORM_A_PLE_FULL_VOCAB): under Form A the host's PLE
n-gram table must cover the WHOLE n-gram id space, as F13 already does for
``embed_tokens``.

The defect (x168, D TP0): the n-gram ``VocabParallelEmbedding`` keeps the even
TP=3 vocab split (``tp_vocab_ratios`` is "vocab always even"), so the host's
shard is ``[0, V/3)``; ``Qwen4ExpPinnedHostEmbedding.reduce`` skips the
all-reduce under Form A (F12), and no worker builds a PLE -- two thirds of the
n-gram rows (bigram heads 5-7, all trigram heads) are zero on D. The proof
line counted it: kernel_rows 685 of 2048 rows per 32 rounds (33.4 %).

Desk only (CPU, the table on ``meta``): the real ``Qwen4ExpNGramEmbedding``
under a TP=3 group and an installed Form A plan --

* switch off: the host's shard is a third of the id space, and a third of the
  hashed verify rows fall into it (the x168 ratio, reproduced);
* switch on: tp_size 1, the full range, every hashed row inside;
* switch on without Form A, or with a copying offload backend: unchanged
  (the latter refused by name);
* the stage takes its range from the same shard indices.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import contextlib
import types
import unittest
from unittest import mock

import torch

from sglang.srt import rank_role
from sglang.srt.environ import envs
from sglang.srt.layers import vocab_parallel_embedding as vpe
from sglang.srt.models import qwen4_exp as q
from sglang.srt.models import qwen4_exp_ple_decode_pread as dp
from sglang.srt.models import qwen4_exp_ple_prefetch as pf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _config(backend="checkpoint"):
    return types.SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        vocab_size=248320,
        ngram_vocab_size_base=4001,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=248044,
        seed=1234,
        ple_offload_embedding=True,
        ple_offload_backend=backend,
        ple_embedding_dtype=None,
    )


@contextlib.contextmanager
def _d_host(form_a=True, tp_size=3):
    """Rank 0 of D's TP group, optionally with a Form A plan installed."""
    par = types.SimpleNamespace(tp_rank=0, tp_size=tp_size, attn_tp_rank=0, attn_tp_size=tp_size)
    with mock.patch.object(vpe, "get_parallel", lambda: par), \
            mock.patch.object(q, "is_dp_attention_enabled", lambda: False), \
            mock.patch.object(rank_role, "_INSTALLED_PLAN", object() if form_a else None):
        yield


def _emb(config):
    return q.Qwen4ExpNGramEmbedding(config, 2560, ple_layer_index=0, prefix="model.layers.1.ple.ple_embedding")


def _covered(emb, n_ctx=64, seed=3):
    """Fraction of hashed verify rows ([history 2 | bonus + 3 drafts]) inside
    the table's shard -- what the staged kernel counts as kernel_rows."""
    p = dp.PleHashParamsPy.of(pf.PleHashParams.of(emb))
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randint(0, 248320, (n_ctx, 6), generator=g)
    ids = torch.tensor(dp.ple_verify_row_ids(ctx.tolist(), p))
    si = emb.ngram_embedding.shard_indices
    inside = (ids >= si.org_vocab_start_index) & (ids < si.org_vocab_end_index)
    return float(inside.float().mean()), int(ids.max())


class TestFormAFullVocab(CustomTestCase):
    def test_off_the_host_holds_a_third_and_reads_the_rest_as_zero_rows(self):
        with envs.SGLANG_WEG2_FORM_A_PLE_FULL_VOCAB.override(False), _d_host():
            emb = _emb(_config())
        e = emb.ngram_embedding
        self.assertEqual(e.tp_size, 3)
        total = int(emb.ngram_heads_vocab_sizes.sum())
        self.assertEqual(e.shard_indices.org_vocab_start_index, 0)
        self.assertLess(e.shard_indices.org_vocab_end_index, total // 2)
        frac, _ = _covered(emb)
        self.assertAlmostEqual(frac, 1 / 3, delta=0.06)  # x168: 685/2048 = 0.334
        # and the Form A reduce adds nothing back
        with _d_host():
            self.assertTrue(rank_role.form_a_dense_is_unsharded())

    def test_on_the_host_holds_every_row(self):
        with envs.SGLANG_WEG2_FORM_A_PLE_FULL_VOCAB.override(True), _d_host():
            emb = _emb(_config())
        e = emb.ngram_embedding
        self.assertEqual(e.tp_size, 1)
        self.assertFalse(e.enable_tp)
        self.assertEqual(e.shard_indices.org_vocab_start_index, 0)
        frac, top = _covered(emb)
        self.assertEqual(frac, 1.0)
        self.assertGreater(e.shard_indices.org_vocab_end_index, top)
        self.assertGreaterEqual(e.shard_indices.org_vocab_end_index, int(emb.ngram_heads_vocab_sizes.sum()))
        self.assertEqual(e.weight.device.type, "meta")  # nothing materialized

    def test_on_without_form_a_or_with_a_copying_backend_is_unchanged(self):
        with envs.SGLANG_WEG2_FORM_A_PLE_FULL_VOCAB.override(True), _d_host(form_a=False):
            self.assertEqual(_emb(_config()).ngram_embedding.tp_size, 3)
        with envs.SGLANG_WEG2_FORM_A_PLE_FULL_VOCAB.override(True), _d_host():
            with self.assertLogs(q.logger, "WARNING") as cm:
                emb = _emb(_config(backend="pinned"))
        self.assertEqual(emb.ngram_embedding.tp_size, 3)
        self.assertTrue(any("SGLANG_WEG2_FORM_A_PLE_FULL_VOCAB refused" in r for r in cm.output))

    def test_kwargs_and_default(self):
        self.assertFalse(envs.SGLANG_WEG2_FORM_A_PLE_FULL_VOCAB.get())
        with _d_host():
            self.assertEqual(q.ple_ngram_vocab_tp_kwargs(_config(), False), {})
            with envs.SGLANG_WEG2_FORM_A_PLE_FULL_VOCAB.override(True):
                self.assertEqual(q.ple_ngram_vocab_tp_kwargs(_config(), False), {"enable_tp": False})
                # attention-TP n-gram lookups (DP attention) keep their own layout
                self.assertEqual(q.ple_ngram_vocab_tp_kwargs(_config(), True), {})

    def test_the_stage_and_the_gather_read_the_same_range(self):
        src = (q.__file__)
        text = open(src).read()
        self.assertIn("**ple_ngram_vocab_tp_kwargs(config, self.use_attn_tp_ngram),", text)
        attach = text[text.index("    def attach_checkpoint_table(self, table) -> None:"):]
        attach = attach[: attach.index("\n    def ", 10)]
        self.assertIn("vocab_start=self.shard_indices.org_vocab_start_index", attach)
        self.assertIn("vocab_end=self.shard_indices.org_vocab_end_index", attach)


if __name__ == "__main__":
    unittest.main()
