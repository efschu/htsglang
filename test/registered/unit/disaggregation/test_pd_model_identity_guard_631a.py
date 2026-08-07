# SPDX-License-Identifier: Apache-2.0
"""The PD handshake refuses two arms holding different weights (#631a guard 1).

``try_ensure_parallel_info`` compared ``page_size`` and ``kv_cache_dtype`` and
nothing else, so a decode arm pointed at a prefill arm holding a DIFFERENT
CHECKPOINT paired cleanly and produced fluent nonsense (#212). Plausible text,
no error, nothing downstream to notice -- the same silent-wrongness shape as
the spec auto-disable in this task's other half.

The guard compares ``model_identity_hash``, reusing
``compute_model_identity_hash`` rather than adding a second hash to keep in
step (the recipe the NCCL transport and the HiCache keys already use).

**The subtle half, and the reason this file exists rather than a one-line
assert:** that hash normally appends ``rank_tp_ratio`` / ``rank_kv_ratio``,
because a stored HiCache PAGE's bytes depend on the writing rank's kv-head
count. A PD handshake asks a different question -- "same weights?" -- and the
two arms are EXPECTED to differ in parallelism: ``TransportIdentity.COMPARED``
omits tp_size/pp_size deliberately, and a token-axis difference is handled by
``owned_ordinals``. Comparing the vectors would therefore refuse Route A (a PP
prefill group feeding a TP+DCP decode group), which the engine transfers
correctly. So the handshake takes the hash with
``include_parallel_vectors=False``, and the first test below is the one that
would have caught that mistake.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    PrefillServerInfo,
)
from sglang.srt.mem_cache.hicache_storage import compute_model_identity_hash
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MODEL = dict(
    model_path="/models/qwen3.6-27b",
    revision=None,
    dtype="auto",
    quantization=None,
    kv_cache_dtype="fp8_e4m3",
)


def _args(**over):
    base = dict(_MODEL, rank_tp_ratio=None, rank_kv_ratio=None)
    base.update(over)
    return SimpleNamespace(**base)


class IdentityHashScopeTest(CustomTestCase):
    """What the hash must and must not depend on."""

    def test_parallel_vectors_excluded_so_route_a_can_pair(self):
        """A PP prefill and a TP+DCP decode on the SAME weights must match.

        This is the regression that matters: with the vectors included they
        differ, and the guard would refuse a supported configuration -- worse
        than having no guard at all.
        """
        decode_arm = _args(rank_tp_ratio="10,1,1", rank_kv_ratio="7,3,3")
        prefill_arm = _args()  # tp_size=1 pipeline: no vectors

        self.assertNotEqual(
            compute_model_identity_hash(decode_arm),
            compute_model_identity_hash(prefill_arm),
            "precondition: with parallel vectors the two arms DO differ",
        )
        self.assertEqual(
            compute_model_identity_hash(decode_arm, include_parallel_vectors=False),
            compute_model_identity_hash(prefill_arm, include_parallel_vectors=False),
            "same weights must hash equal once the parallel vectors are out, "
            "or the guard refuses a Route A pair the engine handles",
        )

    def test_weight_identity_still_separates(self):
        """Dropping the vectors must not drop the point of the hash."""
        a = compute_model_identity_hash(_args(), include_parallel_vectors=False)
        for field, value in (
            ("model_path", "/models/other"),
            ("revision", "abc123"),
            ("quantization", "awq"),
            ("kv_cache_dtype", "auto"),
        ):
            with self.subTest(field=field):
                b = compute_model_identity_hash(
                    _args(**{field: value}), include_parallel_vectors=False
                )
                self.assertNotEqual(a, b, f"{field} must change the identity")

    def test_hash_is_total_over_unusual_server_args(self):
        """The function must not raise on the PD registration path.

        ``revision`` and ``quantization`` were the only parts not
        str()-coerced, so any server_args whose fields are not already
        strings raised TypeError inside "|".join. Latent while the callers
        all ran late with a resolved ServerArgs; #631a added a call during PD
        registration and made it reachable, which is how 11 pre-existing
        tests in test_register_to_bootstrap.py went red.
        """
        self.assertIsInstance(compute_model_identity_hash(mock.MagicMock()), str)

    def test_coercion_did_not_move_any_real_key(self):
        """Persisted HiCache pages outlive the process -- keys must not shift.

        Recomputes the pre-coercion recipe inline and requires equality, so
        this fails if anyone changes the identity string for real inputs.
        """
        import hashlib
        import os

        def pre_coercion_recipe(sa):
            parts = [
                os.path.normpath(sa.model_path) if sa.model_path else "",
                sa.revision or "",
                str(sa.dtype or "auto").lower(),
                sa.quantization or "",
                str(sa.kv_cache_dtype or "auto").lower(),
            ]
            for name in ("rank_tp_ratio", "rank_kv_ratio"):
                value = getattr(sa, name, None)
                if value:
                    parts.append(f"{name}={value}")
            return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

        for over in (
            {},
            {"revision": "r1", "quantization": "awq", "dtype": "bfloat16"},
            {"rank_tp_ratio": "10,1,1", "rank_kv_ratio": "7,3,3"},
            {"model_path": "/models/trailing/"},
        ):
            with self.subTest(over=over):
                args = _args(**over)
                self.assertEqual(
                    compute_model_identity_hash(args), pre_coercion_recipe(args)
                )

    def test_default_is_unchanged_for_existing_callers(self):
        """HiCache keys and the NCCL transport must not be re-keyed by this.

        Persisted pages outlive the process; a changed default would silently
        invalidate every stored page on the rig.
        """
        args = _args(rank_tp_ratio="10,1,1")
        self.assertEqual(
            compute_model_identity_hash(args),
            compute_model_identity_hash(args, include_parallel_vectors=True),
        )


class HandshakeIdentityGuardTest(CustomTestCase):
    """The comparison inside try_ensure_parallel_info."""

    def _manager(self, server_args):
        """A stub carrying only what try_ensure_parallel_info touches."""
        mgr = SimpleNamespace(
            prefill_info_table={},
            kv_args=SimpleNamespace(page_size=1),
            server_args=server_args,
            _resolve_rank_mapping=lambda info: None,
        )
        return mgr

    def _info(self, **over):
        base = dict(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=1,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=False,
        )
        base.update(over)
        return PrefillServerInfo(**base)

    def _run(self, server_args, info):
        response = mock.Mock(status_code=200)
        response.json.return_value = {
            k: v
            for k, v in vars(info).items()
            if k
            in {
                "attn_tp_size",
                "attn_cp_size",
                "dp_size",
                "pp_size",
                "page_size",
                "kv_cache_dtype",
                "follow_bootstrap_room",
                "model_identity_hash",
            }
        }
        with mock.patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            return CommonKVManager.try_ensure_parallel_info(
                self._manager(server_args), "127.0.0.1:8998"
            )

    def test_different_weights_are_refused(self):
        args = _args()
        with self.assertRaises(RuntimeError) as ctx:
            self._run(args, self._info(model_identity_hash="deadbeefdeadbeef"))
        msg = str(ctx.exception)
        self.assertIn("Model identity mismatch", msg)
        self.assertIn("deadbeefdeadbeef", msg, "refusal must name the remote hash")
        self.assertIn(
            "not what this refusal is about",
            msg,
            "refusal must say parallelism is not the cause, or it sends the "
            "operator hunting the wrong difference",
        )

    def test_same_weights_pair(self):
        args = _args()
        good = compute_model_identity_hash(args, include_parallel_vectors=False)
        self.assertTrue(self._run(args, self._info(model_identity_hash=good)))

    def test_same_weights_pair_across_different_parallelism(self):
        """The Route A case, end to end through the handshake."""
        decode_args = _args(rank_tp_ratio="10,1,1", rank_kv_ratio="7,3,3")
        prefill_hash = compute_model_identity_hash(
            _args(), include_parallel_vectors=False
        )
        self.assertTrue(
            self._run(decode_args, self._info(model_identity_hash=prefill_hash)),
            "a PP prefill arm and a TP+DCP decode arm on the same weights "
            "must still pair",
        )

    def test_absent_hash_does_not_refuse(self):
        """A pre-#631a arm sends no hash; refusing on absence breaks rollout."""
        args = _args()
        self.assertTrue(self._run(args, self._info(model_identity_hash=None)))


if __name__ == "__main__":
    unittest.main()
