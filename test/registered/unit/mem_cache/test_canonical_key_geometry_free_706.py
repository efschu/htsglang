"""#706 remainder: the canonical key must be geometry-FREE, hash included.

Under ``--phase-flip-canonical-kv-page`` the stored page holds EVERY attention
layer at full width, so no parallel split can change its bytes. The key drops
the tp and pp suffixes for exactly that reason
(``hicache_storage.py``: "the pp suffix is honest ONLY while a stage's page
holds just that stage's layers").

The identity hash was not given the same treatment. ``cache_controller.py``
builds it with ``compute_model_identity_hash(server_args)``, whose
``include_parallel_vectors`` defaults to True, so ``rank_tp_ratio`` and
``rank_kv_ratio`` enter the key. Those are geometry: they say how the KV is
split across ranks, not what a canonical page contains.

It is not hypothetical on this rig. The harvest boot ran
``rank_tp_ratio=None`` (falsy, skipped) but ``rank_kv_ratio='coupled'``
(truthy, appended), so the live key carries a geometry term today. Two boots of
the same model with the same kv-dtype and different kv-ratio write
byte-identical canonical pages and miss each other.

The flag to drop the tail already exists and its docstring says why it exists
(#631a guard 1). This pins that the canonical path uses it.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import compute_model_identity_hash
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _args(**over):
    base = dict(
        model_path="/models/Qwen3.8-27B",
        served_model_name="Qwen3.8-27B",
        revision=None,
        dtype="bfloat16",
        quantization=None,
        kv_cache_dtype="fp8_e4m3",
        rank_tp_ratio=None,
        rank_kv_ratio="coupled",
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestTheHashDropsGeometryWhenAsked(CustomTestCase):
    def test_the_parallel_tail_is_a_geometry_term(self):
        """Establish the premise: the vectors DO move the hash."""
        a = compute_model_identity_hash(_args(rank_kv_ratio="coupled"))
        b = compute_model_identity_hash(_args(rank_kv_ratio="split"))
        self.assertNotEqual(a, b, "the fixture must actually exercise the tail")

    def test_dropping_the_tail_makes_the_hash_geometry_free(self):
        a = compute_model_identity_hash(
            _args(rank_kv_ratio="coupled"), include_parallel_vectors=False
        )
        b = compute_model_identity_hash(
            _args(rank_kv_ratio="split"), include_parallel_vectors=False
        )
        self.assertEqual(a, b, "geometry must not survive into a canonical key")

    def test_the_BYTE_FORMAT_terms_still_separate(self):
        """CAN-FAIL: dropping geometry must not drop the byte format.

        kv_cache_dtype is exactly what the hash exists to catch -- a silent
        wrong hit across two byte formats. It must still separate.
        """
        a = compute_model_identity_hash(
            _args(kv_cache_dtype="fp8_e4m3"), include_parallel_vectors=False
        )
        b = compute_model_identity_hash(
            _args(kv_cache_dtype="auto"), include_parallel_vectors=False
        )
        self.assertNotEqual(a, b, "the byte format must always separate")

    def test_the_model_identity_still_separates(self):
        a = compute_model_identity_hash(
            _args(revision="r1"), include_parallel_vectors=False
        )
        b = compute_model_identity_hash(
            _args(revision="r2"), include_parallel_vectors=False
        )
        self.assertNotEqual(a, b)


class TestTheCanonicalPathAsksForIt(CustomTestCase):
    """The wiring: cache_controller must drop the tail under the canonical
    format, and must NOT drop it otherwise."""

    def _hash_for(self, canonical: bool):
        from sglang.srt.managers.cache_controller import (
            canonical_identity_hash_for,
        )

        return canonical_identity_hash_for(_args(), canonical)

    def test_canonical_on_gives_a_geometry_free_hash(self):
        from sglang.srt.managers.cache_controller import (
            canonical_identity_hash_for,
        )

        a = canonical_identity_hash_for(_args(rank_kv_ratio="coupled"), True)
        b = canonical_identity_hash_for(_args(rank_kv_ratio="split"), True)
        self.assertEqual(a, b)

    def test_canonical_off_keeps_the_legacy_hash(self):
        """CAN-FAIL: the default path must be byte-identical to before.

        Without the canonical page a stage's file really does hold only that
        stage's layers, so geometry belongs in the key.
        """
        from sglang.srt.managers.cache_controller import (
            canonical_identity_hash_for,
        )

        a = canonical_identity_hash_for(_args(rank_kv_ratio="coupled"), False)
        b = canonical_identity_hash_for(_args(rank_kv_ratio="split"), False)
        self.assertNotEqual(a, b)
        self.assertEqual(a, compute_model_identity_hash(_args(rank_kv_ratio="coupled")))


if __name__ == "__main__":
    unittest.main()
