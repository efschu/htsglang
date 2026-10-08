# SPDX-License-Identifier: Apache-2.0
"""PP-POSTEN (27B): group P's pool model books what the P runtime books -- and reproduces the metal.

Boots (P.log 'KV pool sizing', per stage):
  dkr27browauthorityyarn2bar1fs09292006 (YaRN x2, 34,17,13 / attn 8,4,4): 703196 / 1084380 / 273712
  dkr27browauthoritybar1fs09291750      (x1,     43,11,10 / attn 10,3,3): 403128 / 1926954 / 423050
Every P rank: 'KV budget holdback ... holdback=0.0'; 'mamba state pool + speculative intermediate state + prefill
activation reserve' = 37.41 MiB x linear layers (the 24-slot mamba pool alone); PP2 cell = attn x 2048 + 10240.
"""
import json
import os
import types
import unittest

try:
    from flliper.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover

    def register_cpu_ci(*args, **kwargs):
        return None


from flliper.srt.planner import pp_cut as P
from flliper.srt.pdflip import form as F
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

TARGET = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
MEAN = 363.3980255126953          # the launcher's mean_layer_mib of this checkpoint (safetensors headers)
GRAPH = 0.156 * 1024              # 'prefill graph pool=0.156' on every P rank
ACK = ("mamba pre-capture reserve", "speculative intermediate state", "GGUF dequant scratch", "prefill activation reserve")
BOOTS = {
    "yarn2": ([26128, 15704, 15432], True, (34, 17, 13), (703196, 1084380, 273712)),
    "x1": ([26312, 15896, 15624], False, (43, 11, 10), (403128, 1926954, 423050)),
}


def _model(free, yarn):
    fixed = [float(x) for x in str(F.profile_constant("P_PP_STAGE_FIXED_MIB", "qwen27b")).split(",")]
    delta = list(F.profile_constant("P_PP_STAGE_FIXED_YARN2_DELTA_MIB", "qwen27b"))
    fx = tuple(a + (b if yarn else 0.0) for a, b in zip(fixed, delta))
    return P.PhasePoolModel(
        free_mib=tuple(free), weight_mib_per_layer=MEAN, kv_mib_per_token_per_attn_layer=2048 / P.MIB,
        arming_floor_mib=(1229.0,) * 3, stage_fixed_mib=fx, activation_reserve_mib=0.0, corridor_holdback_mib=0.0,
        mamba_mib_per_linear_layer_per_slot=float(F.profile_constant("P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT", "qwen27b")),
        mamba_slots=24, prefill_graph_pool_mib=(GRAPH,) * 3, extra_cell_bytes_by_stage=(0, 0, 10240),
        zero_posts_acknowledged=ACK)


def _kinds():
    cfg = json.load(open(os.path.join(TARGET, "config.json")))
    return (cfg.get("text_config") or cfg)["layer_types"]


@unittest.skipUnless(os.path.isdir(TARGET), "rig checkpoint not mounted")
class PostsBookedTest(CustomTestCase):
    def test_reproduces_both_boots_per_stage(self):
        kinds = _kinds()
        for name, (free, yarn, cut, boot) in BOOTS.items():
            got = P.stage_pp_capacities(cut, P.attention_counts(kinds, list(cut)), _model(free, yarn))
            for g, b in zip(got, boot):
                self.assertLess(abs(g / b - 1.0), 0.05, (name, got, boot))

    def test_yarn2_has_a_cut_that_holds_506000(self):
        kinds, m, best = _kinds(), _model(BOOTS["yarn2"][0], True), 0
        for a in range(1, 63):
            for b in range(1, 64 - a):
                cut = (a, b, 64 - a - b)
                attn = P.attention_counts(kinds, list(cut))
                if min(attn) <= 0:
                    continue
                try:
                    best = max(best, min(P.stage_pp_capacities(cut, attn, m)))
                except ValueError:
                    continue
        self.assertGreaterEqual(best, 506000)
        # and the shipped cut of the failed boot is not priced above what P held
        cut = (34, 17, 13)
        self.assertLessEqual(P.stage_pp_capacities(cut, P.attention_counts(kinds, list(cut)), m)[2], 273712)


class GateTest(CustomTestCase):
    def test_registry_gate_is_per_model(self):
        self.assertTrue(F.PROFILES["qwen27b"].p_pool_posts_as_booked)
        self.assertTrue(F.PROFILES["qwen27b"].p_mamba_slots_from_argv)
        self.assertFalse(F.PROFILES["nextflash"].p_pool_posts_as_booked)

    def test_launcher_reads_the_gate(self):
        from flliper.srt.pdflip import launcher as L

        self.assertTrue(L.p_pool_posts_as_booked(types.SimpleNamespace(profile="qwen27b")))
        self.assertFalse(L.p_pool_posts_as_booked(types.SimpleNamespace(profile="nextflash")))

    def test_zero_activation_only_when_acknowledged(self):
        base = dict(free_mib=(1.0,), weight_mib_per_layer=1.0, kv_mib_per_token_per_attn_layer=1.0,
                    arming_floor_mib=(0.0,), stage_fixed_mib=(1.0,), activation_reserve_mib=0.0,
                    corridor_holdback_mib=0.0, mamba_mib_per_linear_layer_per_slot=1.0, mamba_slots=1)
        self.assertIn("activation_reserve_mib",
                      P.PhasePoolModel(**base, zero_posts_acknowledged=ACK[:3]).unfunded_posts)
        self.assertNotIn("activation_reserve_mib",
                         P.PhasePoolModel(**base, zero_posts_acknowledged=ACK).unfunded_posts)

    def test_yarn_delta_only_at_the_yarn_context(self):
        from flliper.srt.pdflip import launcher as L

        ns = types.SimpleNamespace(profile="qwen27b", extra_p="--context-length 524288", extra_d="")
        vals, line = L.p_stage_fixed_ctx_delta_record(ns, 3)
        self.assertEqual(vals, tuple(F.profile_constant("P_PP_STAGE_FIXED_YARN2_DELTA_MIB", "qwen27b")))
        self.assertIn("gemessen", line)
        self.assertEqual(L.p_stage_fixed_ctx_delta_record(types.SimpleNamespace(profile="qwen27b", extra_p="", extra_d=""), 3),
                         (None, ""))
        ns_nf = types.SimpleNamespace(profile="nextflash", extra_p="--context-length 524288", extra_d="")
        self.assertEqual(L.p_stage_fixed_ctx_delta_record(ns_nf, 3), (None, ""))


if __name__ == "__main__":
    unittest.main()
