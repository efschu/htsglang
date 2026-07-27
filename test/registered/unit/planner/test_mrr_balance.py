"""The GDN-state / KV balance point for ``--max-running-requests`` (#253).

What is pinned here:

1. The slot arithmetic reproduces the runtime's own pool sizing, including
   the reference boot documented for this rig (target 16 -> 100 slots ->
   ~20 admitted requests).
2. The per-slot state bytes and the KV cell the planner derives for the
   Qwen3.6-27B hybrid match the geometry figures recorded for it: 146.8 MiB
   per slot in fp32 / 74.8 MiB in bf16, and 32 KiB per KV token at fp8.
3. The break-even context length -- where one session's recurrent state
   weighs as much as that session's KV -- comes out at the recorded 4698
   tokens (fp32) / 2394 (bf16) PER SLOT, and 6.25x that per admitted
   session, because one admitted session costs ``ratio * safety`` slots.
4. The balance point itself, for fully documented inputs (the geometry below
   is the real Qwen3.6-27B-FP8 config; the rig is the reference 5090 + 2x
   3080). This is the cross-check the ticket asked for.

CROSS-CHECK RESULT, stated plainly
----------------------------------
The ticket cited "balance point mrr=75 carrying 93 sessions instead of 20
(+365%)" from the #119/#123 KV-regain thread. That measurement does not exist
in this project: nothing in the branch, its commits, the design notes or the
runbook records an mrr sweep, a session count of 93, or a +365% figure, and
the #119 branch has no GPU boot recorded at all. What IS recorded is the
geometry above, and it does not support 75 for this model on this rig:

  * one admitted session costs ratio*safety = 6.25 state slots, i.e. 918 MiB
    (fp32) or 468 MiB (bf16) of pool, context-independent;
  * mrr=75 would size the pool to 469 slots = 67 GiB (fp32) / 34 GiB (bf16),
    against a 3-card rig with ~72 GiB total of which ~27 GiB is weights.

So mrr=75 is not merely off-balance for the 27B, it is not allocatable. The
computed balance point lands at mrr 20-25 for short sessions (fp32 state) and
30-41 with ``SGLANG_MAMBA_SSM_DTYPE=bfloat16``, falling to 12-15 at 32k
context. The "20 sessions" half of the cited pair does reproduce exactly: it
is the reference boot's ``max_num_reqs`` at the default target 16.

The tests below assert the computed values with the reasoning above; they do
NOT assert 75.
"""

import dataclasses
import json
import math
import os
import tempfile
import unittest

from sglang.srt.planner.feasibility import plan
from sglang.srt.planner.hardware import hardware_from_manual
from sglang.srt.planner.mrr_balance import (
    MAMBA_RATIO,
    MAMBA_SAFETY_MARGIN,
    PREDICTOR_TARGET_CLAMP,
    admitted_sessions,
    balance_report,
    state_slot_count,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

#: The reference rig, hand-declared: 5090 32 GiB + 2x 3080 20 GiB.
_RIG3 = ("RTX 5090:32760", "RTX 3080:20480", "RTX 3080:20480")

#: Local model-zoo root; the real-checkpoint variant skips without it.
_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")

#: Qwen3.6-27B geometry, verbatim from the checkpoint's ``text_config``:
#: 64 layers on a full_attention_interval of 4 -> 48 linear-attention (GDN)
#: layers + 16 full-attention layers; 4 KV heads at head_dim 256; GDN with
#: 16 key heads / 48 value heads at 128 dims and conv kernel 4.
_QWEN36_27B_TEXT = dict(
    model_type="qwen3_5",
    hidden_size=5120,
    num_hidden_layers=64,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    intermediate_size=17408,
    vocab_size=248064,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    max_position_embeddings=262144,
    layer_types=[
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(64)
    ],
)


def _write_27b_config(tmpdir: str) -> str:
    cfg = dict(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        text_config=dict(_QWEN36_27B_TEXT),
        quantization_config=dict(
            quant_method="fp8", fmt="e4m3", activation_scheme="dynamic"
        ),
    )
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(cfg, f)
    return tmpdir


def _plan_27b(model_path: str, **kw):
    hw = hardware_from_manual(list(_RIG3))
    return plan(
        model_path,
        hw,
        tp_size=3,
        kv_cache_dtype="fp8_e4m3",
        max_running_requests=16,
        with_advantage=False,
        **kw,
    )


class TestSlotArithmetic(CustomTestCase):
    """The pool sizing the recommendation is built on."""

    def test_reference_boot_target16(self):
        # slots = ceil(16 * 5 * 1.25) = 100; max_num_reqs = 100 // 5 = 20.
        # This is the reference boot recorded for this rig.
        self.assertEqual(state_slot_count(16), 100)
        self.assertEqual(admitted_sessions(16), 20)

    def test_admitted_is_target_times_safety(self):
        for target in (1, 2, 4, 8, 16, 32, 48, 75, 128):
            self.assertEqual(
                admitted_sessions(target),
                math.ceil(target * MAMBA_RATIO * MAMBA_SAFETY_MARGIN)
                // MAMBA_RATIO,
            )
        # The pair quoted in the ticket is internally consistent even though
        # the measurement behind it does not exist: 16 -> 20 and 75 -> 93.
        self.assertEqual(admitted_sessions(75), 93)

    def test_spec_pad_and_clamp(self):
        # Speculative decode adds an intermediate state per admitted request.
        self.assertEqual(state_slot_count(16, draft_tokens=0), 100)
        self.assertEqual(state_slot_count(16, draft_tokens=3), 100 + 16 * 3)
        # The clamp reproduces PerfCostModel._mamba_pool_bytes.
        self.assertEqual(
            state_slot_count(200, clamp=PREDICTOR_TARGET_CLAMP),
            state_slot_count(PREDICTOR_TARGET_CLAMP),
        )
        self.assertGreater(
            state_slot_count(200), state_slot_count(200, clamp=PREDICTOR_TARGET_CLAMP)
        )

    def test_placement_helper_still_byte_exact(self):
        """placement._mamba_sessions now delegates here; it must stay the
        clamped variant, or the per-rank VRAM breakdown stops matching the
        cost model's pool bytes."""
        from sglang.srt.planner.placement import _mamba_sessions

        class _M:
            spec_active = True
            spec_draft_tokens = 3

        class _F:
            max_running_requests = 200

        self.assertEqual(
            _mamba_sessions(_M(), _F()),
            state_slot_count(48, draft_tokens=3),
        )


class TestBalanceGeometry(CustomTestCase):
    """Per-slot state, KV cell and break-even for the Qwen3.6-27B hybrid.

    Weight-independent: these come from the model geometry alone, so they are
    asserted against the recorded figures exactly.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = _write_27b_config(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _report(self):
        r = _plan_27b(self.model)
        self.assertIsNotNone(r.mrr_balance, "hybrid model must carry a balance report")
        return r

    def test_state_and_kv_cell_match_recorded_geometry(self):
        bal = self._report().mrr_balance
        # 48 GDN layers x 16 units x [3 v-heads/unit x 128 x 128 x 4B state
        # + (2x128 + 3x128) x 3 x 2B conv] = 146.8 MiB per slot, fp32 SSM.
        self.assertAlmostEqual(bal.state_mib_per_slot, 146.8, delta=0.1)
        self.assertAlmostEqual(
            sum(bal.state_mib_per_slot_per_rank), bal.state_mib_per_slot, places=6
        )
        # 16 full-attention layers x 2 (K+V) x 4 KV heads x 256 dims x 1 B
        # (fp8) = 32 KiB per token.
        self.assertEqual(bal.kv_cell_bytes, 32 * 1024)

    def test_break_even_context(self):
        bal = self._report().mrr_balance
        # Per SLOT the recorded break-even is 4698 tokens in fp32.
        per_slot_tokens = bal.state_mib_per_slot * 2**20 / bal.kv_cell_bytes
        self.assertAlmostEqual(per_slot_tokens, 4698, delta=2)
        # Per admitted SESSION it is ratio*safety = 6.25x that, because the
        # pool charges 6.25 slots per admitted request.
        self.assertAlmostEqual(
            bal.break_even_context_tokens,
            per_slot_tokens * MAMBA_RATIO * MAMBA_SAFETY_MARGIN,
            delta=1,
        )
        self.assertAlmostEqual(bal.break_even_context_tokens, 29362, delta=5)

    def test_bf16_ssm_halves_the_state(self):
        """SGLANG_MAMBA_SSM_DTYPE=bfloat16 (what the rig runbook exports)
        halves the SSM term -> 74.8 MiB/slot, break-even 2394 tokens/slot."""
        prev = os.environ.get("SGLANG_MAMBA_SSM_DTYPE")
        os.environ["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
        try:
            bal = self._report().mrr_balance
        finally:
            if prev is None:
                os.environ.pop("SGLANG_MAMBA_SSM_DTYPE", None)
            else:
                os.environ["SGLANG_MAMBA_SSM_DTYPE"] = prev
        self.assertAlmostEqual(bal.state_mib_per_slot, 74.8, delta=0.1)
        self.assertAlmostEqual(
            bal.state_mib_per_slot * 2**20 / bal.kv_cell_bytes, 2394, delta=2
        )


class TestBalancePoint(CustomTestCase):
    """The recommendation itself, on documented inputs."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = _write_27b_config(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_points_for_4k_10k_32k(self):
        bal = _plan_27b(self.model).mrr_balance
        by_ctx = {p.target_context_tokens: p for p in bal.points}
        self.assertEqual(sorted(by_ctx), [4096, 10240, 32768])

        # Order of magnitude: tens, not 75, and not single digits. The 27B's
        # state is heavy (918 MiB per admitted session in fp32) against a
        # light 32 KiB/token KV cell, so concurrency is expensive here.
        for p in bal.points:
            self.assertGreaterEqual(p.recommended_max_running_requests, 8)
            self.assertLessEqual(p.recommended_max_running_requests, 40)

        # A longer target context always wants LESS concurrency: the KV each
        # session needs grows while its state cost does not.
        mrrs = [by_ctx[c].recommended_max_running_requests for c in (4096, 10240, 32768)]
        self.assertGreater(mrrs[0], mrrs[2])
        self.assertGreaterEqual(mrrs[0], mrrs[1])
        self.assertGreaterEqual(mrrs[1], mrrs[2])

        # mrr=75 is not merely off-balance for this model, it is not
        # allocatable: 75 -> 469 slots x 146.8 MiB = 67 GiB of state pool.
        pool_gib_at_75 = state_slot_count(75) * bal.state_mib_per_slot / 1024
        self.assertGreater(pool_gib_at_75, 60)

    def test_recommendation_is_the_argmax(self):
        """No neighbouring concurrency carries more sessions at that context.

        ``sessions_at_current`` evaluates the SAME session function at
        ``plan_inputs.max_running_requests``, so re-planning with a candidate
        mrr and reading that field is a direct probe of the curve.
        """
        r = _plan_27b(self.model)
        base = r.inputs.rank_tp_ratio or [1] * r.inputs.tp_size
        budgets = r.inputs.effective_vram_mib
        for p in r.mrr_balance.points:
            best = p.recommended_max_running_requests
            self.assertEqual(
                p.sessions, min(p.state_bound_sessions, int(p.kv_bound_sessions))
            )
            for m in range(max(1, best - 5), best + 6):
                probe = balance_report(
                    dataclasses.replace(r.inputs, max_running_requests=m),
                    base,
                    budgets,
                    [p.target_context_tokens],
                )
                self.assertIsNotNone(probe)
                self.assertLessEqual(
                    probe.points[0].sessions_at_current,
                    p.sessions,
                    f"ctx {p.target_context_tokens}: mrr {m} beats the "
                    f"recommended {best}",
                )

    def test_suggestion_changes_nothing(self):
        """The plan keeps the concurrency it was given; the report is
        advisory only."""
        r = _plan_27b(self.model)
        self.assertEqual(r.inputs.max_running_requests, 16)
        self.assertEqual(r.mrr_balance.current_max_running_requests, 16)
        # And the capacity figure is the one for mrr=16, untouched by the
        # suggestion.
        r2 = _plan_27b(self.model)
        self.assertEqual(
            r.capacity.max_context_tokens, r2.capacity.max_context_tokens
        )

    def test_non_hybrid_has_no_balance(self):
        """A model with no GDN layers has no state/KV trade to balance."""
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(
                architectures=["LlamaForCausalLM"],
                model_type="llama",
                hidden_size=4096,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                intermediate_size=14336,
                vocab_size=128256,
                max_position_embeddings=131072,
            )
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f)
            hw = hardware_from_manual(list(_RIG3))
            r = plan(d, hw, tp_size=3, with_advantage=False)
            self.assertIsNone(r.mrr_balance)

    def test_clamp_note_when_recommendation_exceeds_predictor_clamp(self):
        """A recommendation above the predictor's target clamp must say that
        the plan's own max-context figure under-charges the pool there."""
        r = _plan_27b(self.model)
        base = r.inputs.rank_tp_ratio or [1] * r.inputs.tp_size
        # A very short target context pushes the balance towards concurrency.
        bal = balance_report(r.inputs, base, r.inputs.effective_vram_mib, [256])
        self.assertIsNotNone(bal)
        p = bal.points[0]
        if p.recommended_max_running_requests > PREDICTOR_TARGET_CLAMP:
            self.assertTrue(bal.predictor_clamp_note)
        else:
            self.assertFalse(bal.predictor_clamp_note)


@unittest.skipUnless(
    _CACHE and os.path.isdir(os.path.join(_CACHE, "Qwen3.6-27B-FP8")),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-FP8 not present",
)
class TestBalanceOnRealCheckpoint(CustomTestCase):
    """Same cross-check against the checkpoint on disk (weights anchored to
    the real shard sizes rather than estimated from the config)."""

    def test_matches_the_config_only_estimate(self):
        real = _plan_27b(
            os.path.join(_CACHE, "Qwen3.6-27B-FP8"), include_vision=False
        ).mrr_balance
        self.assertIsNotNone(real)
        self.assertAlmostEqual(real.state_mib_per_slot, 146.8, delta=0.1)
        self.assertEqual(real.kv_cell_bytes, 32 * 1024)
        by_ctx = {p.target_context_tokens: p for p in real.points}
        for ctx, lo, hi in ((4096, 15, 35), (10240, 12, 30), (32768, 6, 20)):
            self.assertTrue(
                lo <= by_ctx[ctx].recommended_max_running_requests <= hi,
                f"{ctx}: {by_ctx[ctx].recommended_max_running_requests} "
                f"outside [{lo}, {hi}]",
            )


if __name__ == "__main__":
    unittest.main()
