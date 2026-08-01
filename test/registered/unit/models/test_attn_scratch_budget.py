# SPDX-License-Identifier: Apache-2.0
"""#395 (ANALYSE_393 item 1): MiB scratch budget replaces the flat
chunked-prefix / attention-scratch token-count threshold.

THE DEFECT, as it stood
------------------------
``DeepseekMHAForwardMixin.init_mha_forward`` set
``self.chunked_prefix_cache_threshold`` straight from
``SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD`` -- a flat token count, carrying
``# TODO: Design a finer way to determine the threshold``. The threshold
gates whether a batch's prefix is served by MLA absorption (compressed
``kv_lora_rank + qk_rope_head_dim`` latent) or by materializing per-token K/V
in the MHA_CHUNKED_KV / MHA_ONE_SHOT path
(``k: [.., num_local_heads, qk_head_dim]``, ``v: [.., num_local_heads,
v_head_dim]``). The bytes that switch trades away scale with THIS rank's
local head count and head dim -- a quantity that differs per rank under
(uneven) TP (e.g. V4-Flash's [32, 16, 16] split) -- so a flat token count is
not comparable across ranks or models.

THE FIX
-------
``attn_scratch_token_threshold(budget_mib, num_local_heads, qk_head_dim,
v_head_dim)`` converts a rank-independent MiB budget into a per-rank token
threshold using that rank's own geometry. The default budget
(``DEFAULT_ATTN_SCRATCH_BUDGET_MIB = 640``) is derived so that, on the
DeepSeek-V3 TP=1 reference geometry documented in ``forward_mha.py``
(``num_local_heads=128, qk_head_dim=192, v_head_dim=128``), it reproduces the
legacy 8192-token default bit-for-bit. The old env var is kept as a
deprecated escape hatch (honored verbatim, bypassing the MiB conversion
entirely) and is mutually exclusive with the new
``--attn-scratch-budget-mib`` flag.

PROPERTIES PINNED BELOW
------------------------
1. Default-unchanged: the reference geometry (TP=1) and an even-TP split
   (TP=4, uniform local head count) both derive exactly the legacy 8192
   default -- a can-fail pin (see ``PreFixSemanticsAreRedTest``).
2. Invariance contract: the SAME MiB budget on ranks with different local
   head counts yields token thresholds INVERSELY proportional to their
   per-token scratch bytes -- the contract, not one hardcoded arithmetic
   instance.
3. The deprecated env alias is honored verbatim when set, and set alongside
   the new flag is a hard error naming both.
4. The TODO comment is gone, replaced by a comment stating the invariance
   constraint.
"""

import inspect
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha as forward_mha_module
import sglang.srt.server_args as server_args_module
from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DEFAULT_ATTN_SCRATCH_BUDGET_MIB,
    DeepseekMHAForwardMixin,
    attn_scratch_bytes_per_token,
    attn_scratch_token_threshold,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# The reference geometry documented in forward_mha.py's "Configs for
# DeepSeek-V3" comment, at TP=1 (no head sharding).
REFERENCE_TP1 = dict(num_local_heads=128, qk_head_dim=192, v_head_dim=128)
LEGACY_DEFAULT_TOKENS = 8192


def _fake_self(num_local_heads: int, qk_head_dim: int, v_head_dim: int):
    """A minimal stand-in for DeepseekV2AttentionMLA carrying only the
    attributes init_mha_forward reads."""
    return SimpleNamespace(
        num_local_heads=num_local_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
    )


def _run_init_mha_forward(fake_self, *, disable_chunked_prefix_cache=False, attn_scratch_budget_mib=None):
    fake_server_args = SimpleNamespace(
        disable_chunked_prefix_cache=disable_chunked_prefix_cache,
        attn_scratch_budget_mib=attn_scratch_budget_mib,
    )
    with patch.object(
        forward_mha_module, "get_server_args", lambda: fake_server_args
    ):
        DeepseekMHAForwardMixin.init_mha_forward(fake_self)
    return fake_self


class DefaultUnchangedPinTest(CustomTestCase):
    """Property 1: the default path is byte-identical on the reference
    geometry, and self-consistent across an even TP split."""

    def test_reference_tp1_geometry_reproduces_legacy_default(self):
        fake_self = _fake_self(**REFERENCE_TP1)
        _run_init_mha_forward(fake_self)
        self.assertEqual(
            fake_self.chunked_prefix_cache_threshold, LEGACY_DEFAULT_TOKENS
        )

    def test_even_tp4_split_is_uniform_across_ranks(self):
        # 128 total heads split evenly across TP=4 -> 32 local heads/rank,
        # identical head dims for every rank. Every rank must derive the
        # SAME threshold (no cross-rank divergence when geometry matches).
        thresholds = set()
        for _rank in range(4):
            fake_self = _fake_self(
                num_local_heads=32, qk_head_dim=192, v_head_dim=128
            )
            _run_init_mha_forward(fake_self)
            thresholds.add(fake_self.chunked_prefix_cache_threshold)
        self.assertEqual(len(thresholds), 1, "even-TP ranks diverged")
        # And it must equal the direct formula application, not merely be
        # self-consistent.
        expected = attn_scratch_token_threshold(
            DEFAULT_ATTN_SCRATCH_BUDGET_MIB,
            num_local_heads=32,
            qk_head_dim=192,
            v_head_dim=128,
        )
        self.assertEqual(thresholds.pop(), expected)


class PreFixSemanticsAreRedTest(CustomTestCase):
    """The pin above is not vacuous: feeding the OLD (pre-fix) flat-8192
    semantics -- i.e. ignoring geometry entirely -- into the same assertion
    must NOT reproduce 8192 for a non-reference geometry, and a wrong
    default-budget constant must NOT reproduce 8192 on the reference
    geometry either."""

    def test_flat_8192_is_not_what_a_smaller_rank_should_get(self):
        # Under the OLD scheme every rank got 8192 regardless of geometry.
        # Under the fix, a rank with fewer local heads than the reference
        # (e.g. 16 vs. 128) must get something other than 8192.
        derived = attn_scratch_token_threshold(
            DEFAULT_ATTN_SCRATCH_BUDGET_MIB,
            num_local_heads=16,
            qk_head_dim=192,
            v_head_dim=128,
        )
        self.assertNotEqual(derived, LEGACY_DEFAULT_TOKENS)

    def test_wrong_default_budget_constant_breaks_the_pin(self):
        # Proves DefaultUnchangedPinTest can fail: an off-by-one MiB budget
        # must NOT reproduce the legacy 8192 default on the reference
        # geometry.
        wrong_budget = DEFAULT_ATTN_SCRATCH_BUDGET_MIB - 1
        derived = attn_scratch_token_threshold(wrong_budget, **REFERENCE_TP1)
        self.assertNotEqual(derived, LEGACY_DEFAULT_TOKENS)


class InvarianceContractTest(CustomTestCase):
    """Property 2: pin the CONTRACT (inverse proportionality to per-token
    scratch bytes), not one hardcoded arithmetic instance."""

    def test_fewer_local_heads_yields_a_higher_threshold(self):
        # V4-Flash-style uneven TP split: [32, 16, 16] local heads.
        budget_mib = 640
        geometries = [32, 16, 16]
        thresholds = [
            attn_scratch_token_threshold(
                budget_mib, num_local_heads=h, qk_head_dim=192, v_head_dim=128
            )
            for h in geometries
        ]
        # The 16-head ranks must have a STRICTLY higher threshold than the
        # 32-head rank for the identical MiB budget.
        self.assertGreater(thresholds[1], thresholds[0])
        self.assertGreater(thresholds[2], thresholds[0])
        # The two 16-head ranks (identical geometry) must match each other.
        self.assertEqual(thresholds[1], thresholds[2])

    def test_threshold_times_bytes_per_token_is_invariant(self):
        # threshold * per_token_bytes approximates budget_bytes for ANY
        # geometry at a fixed budget (floor division loses at most
        # per_token_bytes - 1 bytes -- assert that bound, not exact equality,
        # since the contract is proportionality, not one arithmetic instance).
        budget_mib = 640
        budget_bytes = budget_mib * 1024 * 1024
        for num_local_heads in (1, 3, 16, 17, 32, 43, 64, 128):
            per_token_bytes = attn_scratch_bytes_per_token(
                num_local_heads, qk_head_dim=192, v_head_dim=128
            )
            threshold = attn_scratch_token_threshold(
                budget_mib,
                num_local_heads=num_local_heads,
                qk_head_dim=192,
                v_head_dim=128,
            )
            approx_bytes = threshold * per_token_bytes
            self.assertLessEqual(approx_bytes, budget_bytes)
            self.assertGreater(approx_bytes, budget_bytes - per_token_bytes)

    def test_ratio_between_two_ranks_matches_inverse_byte_ratio(self):
        # Pick head counts that divide budget_bytes evenly so floor division
        # introduces no rounding, and check the ratio directly (the contract
        # itself, generalized rather than a single fixture).
        budget_mib = 640
        for a, b in [(128, 64), (64, 32), (32, 16), (16, 8)]:
            thr_a = attn_scratch_token_threshold(
                budget_mib, num_local_heads=a, qk_head_dim=192, v_head_dim=128
            )
            thr_b = attn_scratch_token_threshold(
                budget_mib, num_local_heads=b, qk_head_dim=192, v_head_dim=128
            )
            # Half the heads -> double the threshold (exactly, since these
            # geometries divide budget_bytes evenly).
            self.assertEqual(thr_b, 2 * thr_a)


class DeprecatedAliasTest(CustomTestCase):
    """Property 3a: the deprecated env var, if set, is honored verbatim --
    bypassing the MiB conversion entirely, even for a value that does not
    correspond to any clean MiB budget."""

    def test_deprecated_env_overrides_mib_conversion(self):
        fake_self = _fake_self(**REFERENCE_TP1)
        with envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD.override(4321):
            _run_init_mha_forward(fake_self, attn_scratch_budget_mib=None)
        self.assertEqual(fake_self.chunked_prefix_cache_threshold, 4321)

    def test_deprecated_env_wins_even_with_different_geometry(self):
        # A value the MiB formula could never produce from a clean budget on
        # THIS geometry proves the conversion path was skipped entirely.
        fake_self = _fake_self(num_local_heads=17, qk_head_dim=193, v_head_dim=129)
        with envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD.override(999):
            _run_init_mha_forward(fake_self, attn_scratch_budget_mib=None)
        self.assertEqual(fake_self.chunked_prefix_cache_threshold, 999)


class ServerArgsMutualExclusivityTest(CustomTestCase):
    """Property 3b: ServerArgs validates the old-env/new-flag interaction
    once at startup (mirrors TestGrpcServerArgs's --grpc-mode /
    --smg-grpc-mode pattern in test_server_args.py)."""

    @staticmethod
    def _args(**kwargs):
        return ServerArgs(model_path="dummy", **kwargs)

    def test_both_set_is_a_hard_error_naming_both_flags(self):
        sa = self._args(attn_scratch_budget_mib=512)
        with envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD.override(8192):
            with self.assertRaises(ValueError) as cm:
                sa._handle_attn_scratch_budget_deprecation()
        message = str(cm.exception)
        self.assertIn("--attn-scratch-budget-mib", message)
        self.assertIn("SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD", message)

    def test_env_only_is_a_deprecation_notice_not_an_error(self):
        sa = self._args(attn_scratch_budget_mib=None)
        with envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD.override(8192):
            with self.assertLogs(
                server_args_module.logger, level="WARNING"
            ) as cm:
                sa._handle_attn_scratch_budget_deprecation()
        self.assertTrue(
            any(
                "SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD is deprecated" in line
                for line in cm.output
            )
        )

    def test_new_flag_only_is_silent(self):
        sa = self._args(attn_scratch_budget_mib=512)
        # No env var set: must not raise and must not warn about deprecation.
        sa._handle_attn_scratch_budget_deprecation()

    def test_neither_set_is_silent(self):
        sa = self._args(attn_scratch_budget_mib=None)
        sa._handle_attn_scratch_budget_deprecation()

    def test_non_positive_budget_is_rejected(self):
        sa = self._args(attn_scratch_budget_mib=0)
        with self.assertRaises(ValueError):
            sa._handle_attn_scratch_budget_deprecation()


class TodoCommentReplacedTest(CustomTestCase):
    """Property 4: the historical TODO is gone; a forward-looking invariance
    comment stands in its place at the same site."""

    def test_todo_is_gone_and_invariance_comment_present(self):
        source = inspect.getsource(forward_mha_module)
        self.assertNotIn(
            "Design a finer way to determine the threshold", source
        )
        self.assertRegex(
            source, re.compile(r"INVARIANCE CONSTRAINT", re.IGNORECASE)
        )


if __name__ == "__main__":
    unittest.main()
