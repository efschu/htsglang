"""The profile generator states the MLP-split knee points (task #230).

``crossover.py`` holds where a concentration starts to pay on a given rig;
the profile generator (``flags.profiles``) used to emit the uneven-max-perf
preset without ever consulting it. Now the preset carries the knee points --
and NOTHING here is a built-in number: every figure comes from the rig-local
store, a finding from another rig or a stale one never selects a vector, and
without a finding the preset says UNMEASURED and names the study.

Hermetic: findings are constructed in-memory and passed explicitly;
``profiles()`` itself performs no filesystem reads for them (the webui caller
loads the store).
"""

import time
import unittest

from sglang.srt.planner import flags
from sglang.srt.planner.crossover import (
    MEASURED_HERE,
    REFERENCE_FINDING,
    ConcentrationPoint,
    CrossoverFinding,
    RigDescriptor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# A rig that forces the fork profiles (heterogeneous VRAM), same shape as
# test_flags._HETERO_GPUS.
_GPUS = [
    {"name": "RTX 3080", "total_mib": 20480},
    {"name": "RTX 5090", "total_mib": 32607},
    {"name": "RTX 3080", "total_mib": 20480},
]
_CFG = {
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "head_dim": 128,
}

_RIG = RigDescriptor(
    cards=("RTX 5090", "RTX 3080", "RTX 3080"),
    model="Qwen3.6-27B",
    quant="fp8",
    tp_size=3,
    base_vector=(63, 37, 36),
)

#: The reference campaign's slopes: 3,1,1 turns at 13.7:1, 6,1,1 at 18.5:1,
#: 4,1,1 is dominated (never the best choice anywhere).
_POINTS = [
    ConcentrationPoint((3, 1, 1), 0.0473, 0.648, 7.2, 6.0),
    ConcentrationPoint((4, 1, 1), 0.0649, 1.444, 10.1, 13.4),
    ConcentrationPoint((6, 1, 1), 0.0906, 1.673, 14.7, 15.5),
]


def _local_finding(**kw):
    args = dict(
        rig=_RIG,
        points=list(_POINTS),
        provenance=MEASURED_HERE,
        measured_at=time.time(),
        cache_bypass_proven=True,
    )
    args.update(kw)
    return CrossoverFinding(**args)


def _max_perf(profs):
    return {p.kind: p for p in profs}["uneven-max-perf"]


class TestKneesInTheProfileGenerator(CustomTestCase):
    def test_no_finding_says_unmeasured_and_offers_the_study(self):
        p = _max_perf(flags.profiles(_CFG, _GPUS))
        knee = [i for i in p.info if "knee" in i.lower()]
        self.assertTrue(knee, "max-perf preset carries no knee statement")
        self.assertTrue(any("NOT measured" in i for i in knee))
        self.assertTrue(any("mlp_split_crossover" in i for i in knee))
        self.assertIsNone(p.settings.get("rank_mlp_ratio"))

    def test_a_usable_finding_states_the_knee_points(self):
        p = _max_perf(
            flags.profiles(_CFG, _GPUS, crossover_finding=_local_finding())
        )
        knee = " ".join(p.info)
        # Envelope candidates with their measured break-evens; the dominated
        # 4,1,1 is not offered as a knee.
        self.assertIn("3,1,1 turns at 13.7:1", knee)
        self.assertIn("6,1,1 turns at 18.5:1", knee)
        self.assertNotIn("4,1,1 turns at", knee)
        # Rig label rides with the numbers.
        self.assertIn("RTX 5090", knee)
        # No workload ratio -> stated, never applied.
        self.assertIsNone(p.settings.get("rank_mlp_ratio"))

    def test_a_workload_above_the_knee_applies_the_measured_vector(self):
        p = _max_perf(
            flags.profiles(
                _CFG,
                _GPUS,
                crossover_finding=_local_finding(),
                prompt_to_output_ratio=25.0,
            )
        )
        # At 25:1 the 6,1,1 net (25 x 0.0906 - 1.673) beats 3,1,1's.
        self.assertEqual(p.settings.get("rank_mlp_ratio"), [6, 1, 1])
        # Applied via the pin path, not through the optimizer whose knee
        # guard rejects exactly this measured concentration.
        self.assertEqual(p.settings.get("rank_tp_ratio"), "auto")
        self.assertTrue(any("Applied --rank-mlp-ratio 6,1,1" in i for i in p.info))
        ok, errs = flags.validate_profile(p, _CFG)
        self.assertTrue(ok, errs)

    def test_a_workload_below_every_knee_keeps_the_base_split(self):
        p = _max_perf(
            flags.profiles(
                _CFG,
                _GPUS,
                crossover_finding=_local_finding(),
                prompt_to_output_ratio=5.0,
            )
        )
        self.assertIsNone(p.settings.get("rank_mlp_ratio"))
        self.assertEqual(p.settings.get("rank_tp_ratio"), "auto-performance")
        self.assertTrue(any("net loss" in i for i in p.info))

    def test_another_rigs_finding_never_selects_a_vector(self):
        # REFERENCE_FINDING is provenance measured_elsewhere by construction.
        p = _max_perf(
            flags.profiles(
                _CFG,
                _GPUS,
                crossover_finding=REFERENCE_FINDING,
                prompt_to_output_ratio=25.0,
            )
        )
        self.assertIsNone(p.settings.get("rank_mlp_ratio"))
        self.assertTrue(
            any("does not select a vector" in i for i in p.info)
        )

    def test_a_stale_finding_never_selects_a_vector(self):
        old = _local_finding(measured_at=time.time() - 90 * 86400)
        p = _max_perf(
            flags.profiles(
                _CFG, _GPUS, crossover_finding=old, prompt_to_output_ratio=25.0
            )
        )
        self.assertIsNone(p.settings.get("rank_mlp_ratio"))

    def test_a_tp_mismatched_vector_is_not_applied(self):
        rig2 = RigDescriptor(
            cards=("A", "B"), model="m", quant="fp8", tp_size=2
        )
        f = _local_finding(
            rig=rig2, points=[ConcentrationPoint((3, 1), 0.05, 0.5)]
        )
        p = _max_perf(
            flags.profiles(
                _CFG, _GPUS, crossover_finding=f, prompt_to_output_ratio=25.0
            )
        )
        self.assertIsNone(p.settings.get("rank_mlp_ratio"))
        self.assertTrue(any("not applicable" in i for i in p.info))

    def test_other_presets_stay_untouched(self):
        profs = flags.profiles(
            _CFG,
            _GPUS,
            crossover_finding=_local_finding(),
            prompt_to_output_ratio=25.0,
        )
        for p in profs:
            if p.kind == "uneven-max-perf":
                continue
            self.assertIsNone(p.settings.get("rank_mlp_ratio"), p.kind)
            self.assertFalse(
                any("knee" in i.lower() for i in p.info), p.kind
            )


if __name__ == "__main__":
    unittest.main()
