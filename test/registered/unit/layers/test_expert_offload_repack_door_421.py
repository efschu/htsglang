"""#421 F8: the marlin-repack door refuses a #394 cold shard, by name.

``presplit_expert_offload_after_repack(layer, cold_shard=...)`` had 14 call
sites and exactly one -- a unit test -- passing a real value. The docstring
conceded it ("With no cold_shard (every caller today: fp8.py, gptq_moe.py,
awq_moe.py)") while the #394 merge message claimed both load-time halves
"take their layout from ONE plan object". The audit called that partial
wiring and left the fix open between "thread the context in" and "refuse
explicitly".

It is a refusal, and the reason is measured rather than stylistic: the
parameter's own documented precondition -- a layer that shards experts on
dim 0 -- is exactly the configuration in which a delegated expert becomes
UNREACHABLE rather than relocated, which is what killed the GGUF door's first
boot. So the door is shut in both directions, and passing a cold shard now
raises instead of staging a plan that dies on the first token.

Hermetic: no GPU, no torch import beyond what the module already does.
"""

import unittest

from sglang.srt.layers.moe.expert_offload import (
    ColdShardContext,
    HostShardRatio,
    presplit_expert_offload_after_repack,
    refuse_cold_shard_at_repack_door,
    repack_door_shards_experts_on_dim0,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _Layer:
    def __init__(self, *, moe_ep_size=1, gguf_expert_shard=False, num_local=8):
        self.moe_ep_size = moe_ep_size
        self._gguf_expert_shard = gguf_expert_shard
        self.num_local_experts = num_local


def _cold_shard():
    return ColdShardContext(0, 2, HostShardRatio((0.7, 0.3), "test", "unit test"))


class TestEligibilityIsClassifiedCorrectly(CustomTestCase):
    def test_intermediate_dim_tp_moe_is_not_expert_sharded(self):
        self.assertFalse(repack_door_shards_experts_on_dim0(_Layer()))

    def test_expert_parallel_layer_is_expert_sharded(self):
        self.assertTrue(repack_door_shards_experts_on_dim0(_Layer(moe_ep_size=2)))

    def test_gguf_expert_shard_is_expert_sharded(self):
        self.assertTrue(
            repack_door_shards_experts_on_dim0(_Layer(gguf_expert_shard=True))
        )


class TestTheDoorRefusesByName(CustomTestCase):
    def test_intermediate_dim_tp_refusal_names_the_structural_reason(self):
        with self.assertRaises(ValueError) as ctx:
            refuse_cold_shard_at_repack_door(_Layer())
        msg = str(ctx.exception)
        self.assertIn("intermediate-dim TP MoE", msg)
        self.assertIn("#421 F8", msg)

    def test_expert_sharded_refusal_names_the_measurement(self):
        with self.assertRaises(ValueError) as ctx:
            refuse_cold_shard_at_repack_door(_Layer(moe_ep_size=2))
        msg = str(ctx.exception)
        self.assertIn("UNREACHABLE", msg)
        self.assertIn("SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE", msg)

    def test_the_unsafe_escape_hatch_opens_only_for_an_eligible_layer(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE.override(True):
            # Eligible: the developer escape hatch lets it through.
            refuse_cold_shard_at_repack_door(_Layer(moe_ep_size=2))
            # Ineligible: still structurally impossible, hatch or not.
            with self.assertRaises(ValueError):
                refuse_cold_shard_at_repack_door(_Layer())

    def test_presplit_entry_point_refuses_before_doing_any_work(self):
        """The falsifier at the real entry point: a caller that supplies a
        cold shard gets the error, not a silently wrong staging plan."""
        with self.assertRaises(ValueError) as ctx:
            presplit_expert_offload_after_repack(_Layer(), cold_shard=_cold_shard())
        self.assertIn("refused", str(ctx.exception))

    def test_no_cold_shard_is_unchanged(self):
        """The default path: no cold shard, no refusal. (The staging itself
        is a no-op here because the resident fraction is 1.0.)"""
        self.assertIsNone(presplit_expert_offload_after_repack(_Layer()))


if __name__ == "__main__":
    unittest.main()
