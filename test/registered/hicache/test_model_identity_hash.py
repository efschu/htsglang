"""Model identity hash: storage keys must separate KV byte formats.

Storage page hashes cover token ids only, and backend key suffixes cover
served_model_name plus parallel geometry. Entries in a persistent storage
tier outlive the server process, so two runs sharing a served_model_name and
storage location but differing in e.g. --kv-cache-dtype must not hit each
other's pages. compute_model_identity_hash() closes that gap.
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-only key computation, runs in milliseconds on any runner.
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import tempfile
import unittest

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    compute_model_identity_hash,
)


class FakeServerArgs:
    """Minimal stand-in for ServerArgs identity fields."""

    def __init__(
        self,
        model_path="meta-llama/Llama-3-8B",
        revision=None,
        dtype="bfloat16",
        quantization=None,
        kv_cache_dtype=None,
    ):
        self.model_path = model_path
        self.revision = revision
        self.dtype = dtype
        self.quantization = quantization
        self.kv_cache_dtype = kv_cache_dtype


def make_config(identity_hash=None, **kwargs):
    defaults = dict(
        tp_rank=0,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="test-model",
        model_identity_hash=identity_hash,
    )
    defaults.update(kwargs)
    return HiCacheStorageConfig(**defaults)


class TestComputeModelIdentityHash(unittest.TestCase):
    def test_determinism_and_length(self):
        args = FakeServerArgs()
        h1 = compute_model_identity_hash(args)
        h2 = compute_model_identity_hash(args)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_kv_cache_dtype_separates(self):
        args_auto = FakeServerArgs(kv_cache_dtype=None)
        args_fp8 = FakeServerArgs(kv_cache_dtype="fp8_e4m3")
        self.assertNotEqual(
            compute_model_identity_hash(args_auto),
            compute_model_identity_hash(args_fp8),
        )

    def test_quantization_separates(self):
        args_none = FakeServerArgs(quantization=None)
        args_awq = FakeServerArgs(quantization="awq")
        self.assertNotEqual(
            compute_model_identity_hash(args_none),
            compute_model_identity_hash(args_awq),
        )

    def test_model_dtype_separates(self):
        args_bf16 = FakeServerArgs(dtype="bfloat16")
        args_fp16 = FakeServerArgs(dtype="float16")
        self.assertNotEqual(
            compute_model_identity_hash(args_bf16),
            compute_model_identity_hash(args_fp16),
        )

    def test_model_path_separates(self):
        args_a = FakeServerArgs(model_path="meta-llama/Llama-3-8B")
        args_b = FakeServerArgs(model_path="Qwen/Qwen2-7B")
        self.assertNotEqual(
            compute_model_identity_hash(args_a),
            compute_model_identity_hash(args_b),
        )

    def test_path_normalization(self):
        args_trailing = FakeServerArgs(model_path="/models/llama/")
        args_clean = FakeServerArgs(model_path="/models/llama")
        self.assertEqual(
            compute_model_identity_hash(args_trailing),
            compute_model_identity_hash(args_clean),
        )

    def test_none_equals_auto(self):
        args_none = FakeServerArgs(dtype=None, kv_cache_dtype=None)
        args_auto = FakeServerArgs(dtype="auto", kv_cache_dtype="auto")
        self.assertEqual(
            compute_model_identity_hash(args_none),
            compute_model_identity_hash(args_auto),
        )

    def test_case_normalization(self):
        args_upper = FakeServerArgs(dtype="BFloat16")
        args_lower = FakeServerArgs(dtype="bfloat16")
        self.assertEqual(
            compute_model_identity_hash(args_upper),
            compute_model_identity_hash(args_lower),
        )


class TestHiCacheFileKeyIsolation(unittest.TestCase):
    """End-to-end key invariant on the file backend."""

    def _backend(self, identity_hash, **kwargs):
        return HiCacheFile(
            make_config(identity_hash=identity_hash, **kwargs),
            file_path=self._dir,
        )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_same_model_same_geometry_different_kv_dtype_keys_differ(self):
        # Identical model and TP/PP/CP geometry; the only difference is the
        # KV cache dtype of the producing server run.
        hash_auto = compute_model_identity_hash(FakeServerArgs(kv_cache_dtype=None))
        hash_fp8 = compute_model_identity_hash(FakeServerArgs(kv_cache_dtype="fp8_e5m2"))
        key_auto = self._backend(hash_auto)._get_suffixed_key("deadbeef")
        key_fp8 = self._backend(hash_fp8)._get_suffixed_key("deadbeef")
        self.assertNotEqual(key_auto, key_fp8)

    def test_old_layout_key_misses_instead_of_hitting(self):
        # A key written by the pre-identity-hash layout must not match the new
        # layout: the transition is a clean miss, not a silent wrong hit.
        legacy_key = self._backend(None)._get_suffixed_key("deadbeef")
        new_key = self._backend(compute_model_identity_hash(FakeServerArgs()))
        self.assertNotEqual(legacy_key, new_key._get_suffixed_key("deadbeef"))

    def test_suffix_without_hash_unchanged(self):
        backend = self._backend(None)
        self.assertEqual(backend.config_suffix, "_test-model_0_2")
        self.assertNotIn("None", backend.config_suffix)

    def test_dcp_owner_mode_kv_suffix_carries_hash(self):
        identity = compute_model_identity_hash(FakeServerArgs(kv_cache_dtype="fp8_e4m3"))
        backend = self._backend(identity, dcp_owner_mode=True)
        # Rank-shared KV keys (no rank suffix) must still carry the identity.
        self.assertIn(identity, backend._get_suffixed_key("deadbeef"))
        # Component keys keep rank suffix and the identity.
        component = backend._get_suffixed_key("deadbeef.mamba")
        self.assertIn(identity, component)
        self.assertIn("_0_2", component)

    def test_mla_model_suffix_carries_hash(self):
        identity = compute_model_identity_hash(FakeServerArgs())
        backend = self._backend(identity, is_mla_model=True, tp_rank=3, tp_size=8)
        self.assertIn(identity, backend.config_suffix)
        self.assertNotIn("3_8", backend.config_suffix)


if __name__ == "__main__":
    unittest.main()
