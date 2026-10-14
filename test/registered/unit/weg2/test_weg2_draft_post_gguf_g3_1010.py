"""NF-GGUF AP G3 (2026-10-10): the draft price of a draft that is a GGUF.

``weg2/draft_post.py`` priced a draft from ``*.safetensors`` headers only
(``checkpoint_tensor_mib``), so ``--speculative-draft-model-path <mtp>.gguf``
was "unpriced" and the launcher refused it by name (W128 ``d_draft_park_term``,
``raise_for_draft_post``). Now a GGUF is priced from its own tensor directory:
exact per-tensor bytes, the vocabulary tensors removed by the same fragments
(``lm_head`` / ``embed_tokens``) the safetensors path uses, a combined export
priced by its NEXTN block only, a directory with several unrelated GGUFs not
priced at all (ambiguous).
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gguf  # noqa: E402

from sglang.srt.weg2 import draft_post as dp  # noqa: E402

MIB = float(1 << 20)
REAL_DIR = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-GGUF-unsloth/MTP"
)
REAL_SHARED = os.path.join(REAL_DIR, "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
REAL_FULL = os.path.join(REAL_DIR, "mtp-Qwen3.8-Flash-Next-Q8_0.gguf")
CARDS = [SimpleNamespace(nvml_index=i) for i in (1, 2, 0)]


def _write(path, tensors, *, block_count=None, nextn=None):
    """``tensors``: name -> float32 ndarray (stored F32, so n_bytes = 4 * size)."""
    w = gguf.GGUFWriter(path, "qwen4exp")
    if block_count is not None:
        w.add_block_count(block_count)
    if nextn is not None:
        w.add_uint32("qwen4exp.nextn_predict_layers", nextn)
    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {n: int(a.size) * 4 for n, a in tensors.items()}


def _arr(rows, cols=64):
    return np.zeros((rows, cols), dtype=np.float32)


def _draft(tmp, name="mtp.gguf", *, vocab, block=4):
    tensors = {
        f"blk.{block}.nextn.eh_proj.weight": _arr(64, 128),
        f"blk.{block}.attn_q.weight": _arr(128),
        f"blk.{block}.ffn_up_exps.weight": _arr(256),
    }
    if vocab:
        tensors["token_embd.weight"] = _arr(96)
        tensors["output.weight"] = _arr(96)
    path = os.path.join(tmp, name)
    sizes = _write(path, tensors, block_count=block + 1, nextn=1)
    return path, sizes


class TestGgufDraftPrice(unittest.TestCase):
    def test_self_contained_head_is_priced_by_its_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, sizes = _draft(tmp, vocab=True)
            self.assertAlmostEqual(dp.checkpoint_tensor_mib(path), sum(sizes.values()) / MIB)

    def test_vocabulary_tensors_leave_with_the_same_fragments_as_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, s = _draft(tmp, vocab=True)
            rest = sum(s.values()) - s["token_embd.weight"] - s["output.weight"]
            # P: the last stage shares lm_head with the target (output.weight)
            weights, transient = dp.p_draft_post_mib(path)
            self.assertAlmostEqual(
                weights, (sum(s.values()) - s["output.weight"]) / MIB + dp.DRAFT_RUNNER_BUFFER_MIB
            )
            self.assertEqual(transient, dp.P_DRAFT_PRODUCER_TRANSIENT_MIB)
            # D with H1b: embedding and lm_head both shared
            self.assertAlmostEqual(
                dp.d_draft_host_mib(path, share_embed=True), rest / MIB + dp.DRAFT_RUNNER_BUFFER_MIB
            )
            # D without it: only lm_head leaves
            self.assertAlmostEqual(
                dp.d_draft_host_mib(path, share_embed=False),
                (sum(s.values()) - s["output.weight"]) / MIB + dp.DRAFT_RUNNER_BUFFER_MIB,
            )

    def test_shared_head_has_nothing_to_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, s = _draft(tmp, vocab=False)
            total = sum(s.values()) / MIB
            self.assertAlmostEqual(dp.checkpoint_tensor_mib(path), total)
            self.assertAlmostEqual(dp.d_draft_host_mib(path, share_embed=True), total + dp.DRAFT_RUNNER_BUFFER_MIB)
            self.assertAlmostEqual(dp.d_draft_host_mib(path, share_embed=False), total + dp.DRAFT_RUNNER_BUFFER_MIB)

    def test_directory_with_exactly_one_gguf_is_priced_like_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, s = _draft(tmp, vocab=False)
            self.assertAlmostEqual(dp.checkpoint_tensor_mib(tmp), sum(s.values()) / MIB)

    def test_directory_with_several_unrelated_ggufs_is_not_priced(self):
        """The unsloth MTP/ folder holds four variants of the same head: a sum would
        price the draft at several times its size, so the path must name the file."""
        with tempfile.TemporaryDirectory() as tmp:
            _draft(tmp, "a.gguf", vocab=False)
            _draft(tmp, "b.gguf", vocab=True)
            self.assertIsNone(dp.checkpoint_tensor_mib(tmp))
            self.assertIsNone(dp.d_draft_host_mib(tmp, share_embed=True))

    def test_combined_export_prices_only_the_nextn_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            tensors = {
                "token_embd.weight": _arr(96),
                "output.weight": _arr(96),
                "blk.0.attn_qkv.weight": _arr(512),
                "blk.3.attn_q.weight": _arr(512),
                "blk.4.nextn.eh_proj.weight": _arr(64, 128),
                "blk.4.attn_q.weight": _arr(128),
            }
            path = os.path.join(tmp, "combined.gguf")
            s = _write(path, tensors, block_count=5, nextn=1)
            self.assertAlmostEqual(
                dp.checkpoint_tensor_mib(path),
                (s["blk.4.nextn.eh_proj.weight"] + s["blk.4.attn_q.weight"]) / MIB,
            )

    def test_absent_is_none_never_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(dp.checkpoint_tensor_mib(tmp))
            self.assertIsNone(dp.checkpoint_tensor_mib(os.path.join(tmp, "nothing.gguf")))
            junk = os.path.join(tmp, "junk.bin")
            with open(junk, "wb") as f:
                f.write(b"not a gguf at all")
            self.assertIsNone(dp.checkpoint_tensor_mib(junk))

    def test_safetensors_directory_is_untouched(self):
        """A directory with *.safetensors still goes down the old path (a stray
        .gguf beside them is not added)."""
        import json
        import struct

        with tempfile.TemporaryDirectory() as tmp:
            header = {"mtp.fc.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 1 << 20]}}
            raw = json.dumps(header).encode()
            with open(os.path.join(tmp, "m.safetensors"), "wb") as f:
                f.write(struct.pack("<Q", len(raw)) + raw)
            _draft(tmp, vocab=False)
            self.assertAlmostEqual(dp.checkpoint_tensor_mib(tmp), 1.0)

    def test_raise_for_draft_post_prices_a_gguf_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, s = _draft(tmp, vocab=False)
            new, post, why = dp.raise_for_draft_post(
                fracs=[0.26, 0.45, 0.39], stage_layers=[29, 11, 8], row_mib=3.0,
                num_experts=512, draft_path=path, cards=CARDS,
            )
            self.assertEqual(why, "")
            self.assertIsNotNone(post)
            self.assertAlmostEqual(post.weights_mib, sum(s.values()) / MIB + dp.DRAFT_RUNNER_BUFFER_MIB)

    def test_unpriceable_draft_keeps_the_old_wording(self):
        new, post, why = dp.raise_for_draft_post(
            fracs=[0.26, 0.45, 0.39], stage_layers=[29, 11, 8], row_mib=3.0,
            num_experts=512, draft_path=tempfile.mkdtemp(), cards=CARDS,
        )
        self.assertIsNone(post)
        self.assertIn("no *.safetensors", why)
        self.assertIn("*.gguf", why)


@unittest.skipUnless(
    os.path.isfile(REAL_SHARED) and os.path.isfile(REAL_FULL),
    "unsloth qwen4exp MTP export not on this machine",
)
class TestRealMtpFiles(unittest.TestCase):
    def _header_bytes(self, path):
        r = gguf.GGUFReader(path, "r")
        return {str(t.name): int(t.n_bytes) for t in r.tensors}

    def test_shared_q8_0_is_the_sum_of_its_32_tensors(self):
        rows = self._header_bytes(REAL_SHARED)
        self.assertEqual(len(rows), 32)
        mib = dp.checkpoint_tensor_mib(REAL_SHARED)
        self.assertAlmostEqual(mib, sum(rows.values()) / MIB)
        # sanity against the file itself: tensors are the file minus a small header
        file_mib = os.path.getsize(REAL_SHARED) / MIB
        self.assertLess(mib, file_mib)
        self.assertGreater(mib, file_mib - 16.0)

    def test_full_q8_0_minus_its_vocabulary_is_the_shared_head(self):
        full, shared = self._header_bytes(REAL_FULL), self._header_bytes(REAL_SHARED)
        self.assertEqual(len(full), 34)
        vocab = full["token_embd.weight"] + full["output.weight"]
        self.assertEqual(sum(full.values()) - vocab, sum(shared.values()))
        # D with the H1b share: the self-contained file prices as the shared one
        a = dp.d_draft_host_mib(REAL_FULL, share_embed=True)
        b = dp.d_draft_host_mib(REAL_SHARED, share_embed=True)
        self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(b, sum(shared.values()) / MIB + dp.DRAFT_RUNNER_BUFFER_MIB)
        # P: only lm_head (output.weight) leaves
        wp, _ = dp.p_draft_post_mib(REAL_FULL)
        self.assertAlmostEqual(
            wp, (sum(full.values()) - full["output.weight"]) / MIB + dp.DRAFT_RUNNER_BUFFER_MIB
        )

    def test_the_download_folder_with_both_variants_is_ambiguous(self):
        self.assertIsNone(dp.checkpoint_tensor_mib(REAL_DIR))


if __name__ == "__main__":
    unittest.main()
