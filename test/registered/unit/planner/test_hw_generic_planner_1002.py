"""HW-GENERIC 1002, planner side: the planner decides arch, calibration
membership and catalogue misses from card PROPERTIES, never a name substring.

User order 02.10.: the weg2 release must run on any sm_86 / sm_120 card set,
not only the reference rig (1x RTX 5090 + 2x RTX 3080 20 GB).

FALSIFIERS (each fails on 262abf7f75):

* ``flags.rig_has_sm86`` matched the fragment ``"a10"`` -- an A100 (sm80) was
  "sm86"; a name in no fragment list was silently "not sm86"; a ``cc`` field
  on the descriptor was ignored and ``GpuDescriptor`` had no ``cc``.
* ``flags._match_calibration`` counted ``"5090" in name`` -- an
  "RTX 5090 D" was the calibrated 5090.
* ``CardLibrary.get``/``resolve`` raised a bare ``KeyError`` with no card key
  and no measurement command.
* ``power_limit.rate_cut_line`` on foreign cards said only "keine Raten".

Synthetic cards only; nothing touches NVML, CUDA or the local machine.
"""

import json
import os
import tempfile
import types
import unittest
from unittest import mock

from sglang.srt.planner import card_library as cl
from sglang.srt.planner import flags
from sglang.srt.planner import hardware as hw
from sglang.srt.planner import power_limit as P
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _gpu(name, total_mib, cc=None, index=0, uuid=None):
    d = {"name": name, "total_mib": total_mib, "index": index}
    if cc is not None:
        d["cc"] = cc
    if uuid is not None:
        d["uuid"] = uuid
    return d


SM86, SM120, SM80 = (8, 6), (12, 0), (8, 0)

RIG_3X3090 = [_gpu("NVIDIA GeForce RTX 3090", 24576, SM86, i) for i in range(3)]
RIG_3X5090 = [_gpu("NVIDIA GeForce RTX 5090", 32607, SM120, i) for i in range(3)]
RIG_2X5090_1X3080 = [
    _gpu("NVIDIA GeForce RTX 5090", 32607, SM120, 0),
    _gpu("NVIDIA GeForce RTX 5090", 32607, SM120, 1),
    _gpu("NVIDIA GeForce RTX 3080", 20480, SM86, 2),
]
RIG_PRO6000_2XA6000 = [
    _gpu("NVIDIA RTX PRO 6000 Blackwell", 97887, SM120, 0),
    _gpu("NVIDIA RTX A6000", 49140, SM86, 1),
    _gpu("NVIDIA RTX A6000", 49140, SM86, 2),
]
A100 = _gpu("NVIDIA A100-PCIE-40GB", 40960, SM80)
REFERENCE = [
    _gpu("NVIDIA GeForce RTX 3080", 20480, SM86, 0),
    _gpu("NVIDIA GeForce RTX 5090", 32607, SM120, 1),
    _gpu("NVIDIA GeForce RTX 3080", 20480, SM86, 2),
]
REFERENCE_10GB = [
    _gpu("NVIDIA GeForce RTX 3080", 10240, SM86, 0),
    _gpu("NVIDIA GeForce RTX 5090", 32607, SM120, 1),
    _gpu("NVIDIA GeForce RTX 3080", 10240, SM86, 2),
]


class TestArchFromComputeCapability(CustomTestCase):
    def test_the_reference_rig_is_unchanged(self):
        self.assertTrue(flags.rig_has_sm86(REFERENCE))
        # names only (the offline/manual shape the existing suites use)
        self.assertTrue(
            flags.rig_has_sm86([{"name": "RTX 3080", "total_mib": 20480}])
        )
        self.assertFalse(
            flags.rig_has_sm86([{"name": "NVIDIA GeForce RTX 5090", "total_mib": 32607}])
        )

    def test_synthetic_inventories(self):
        self.assertTrue(flags.rig_has_sm86(RIG_3X3090))
        self.assertFalse(flags.rig_has_sm86(RIG_3X5090))
        self.assertTrue(flags.rig_has_sm86(RIG_2X5090_1X3080))
        self.assertTrue(flags.rig_has_sm86(RIG_PRO6000_2XA6000))
        self.assertFalse(flags.rig_has_sm86(RIG_PRO6000_2XA6000[:1]))

    def test_an_a100_is_not_sm86_although_its_name_contains_a10(self):
        self.assertFalse(flags.rig_has_sm86([A100]))
        no_cc = {"name": A100["name"], "total_mib": A100["total_mib"]}
        # without a cc and without an exact catalogue entry the arch is
        # UNKNOWN by name -- never True from the "a10" fragment
        with self.assertRaises(flags.UnknownGpuArch) as cm:
            flags.rig_has_sm86([no_cc])
        self.assertIn("NVIDIA A100-PCIE-40GB", str(cm.exception))
        self.assertIn("HW-ARCH-UNKNOWN", str(cm.exception))
        self.assertEqual(flags.rig_sm86_verdict([no_cc]), (None, ("NVIDIA A100-PCIE-40GB",)))

    def test_the_declared_cc_wins_over_the_name(self):
        # every spelling a descriptor may carry
        for cc in ((8, 6), [8, 6], "8.6", "sm86", "sm_86", 86):
            self.assertTrue(
                flags.rig_has_sm86([{"name": "Anything", "total_mib": 1, "cc": cc}]), cc
            )
        self.assertTrue(
            flags.rig_has_sm86([{"name": "X", "total_mib": 1, "cc_major": 8, "cc_minor": 6}])
        )
        self.assertFalse(
            flags.rig_has_sm86([{"name": "RTX 3080", "total_mib": 1, "cc": "sm_120a"}])
        )

    def test_a_known_sm86_card_decides_even_next_to_an_unknown_one(self):
        rig = [{"name": "SYNTH Accel", "total_mib": 1}, RIG_3X3090[0]]
        self.assertTrue(flags.rig_has_sm86(rig))

    def test_the_catalogue_fallback_is_an_exact_name(self):
        self.assertEqual(flags.gpu_cc({"name": "NVIDIA RTX A6000"}), SM86)
        self.assertEqual(flags.gpu_cc({"name": "NVIDIA GeForce RTX 5090"}), SM120)
        self.assertIsNone(flags.gpu_cc({"name": "NVIDIA A10"}))
        self.assertIsNone(flags.gpu_cc({"name": "NVIDIA GeForce RTX 5090 D"}))

    def test_the_gpu_descriptor_carries_the_cc(self):
        g = hw.GpuDescriptor(index=0, name="NVIDIA RTX PRO 6000 Blackwell",
                             total_mib=97887, cc=SM120)
        self.assertEqual(g.cc, SM120)
        self.assertFalse(flags.rig_has_sm86([g]))
        a100 = hw.GpuDescriptor(index=0, name="NVIDIA A100-PCIE-40GB",
                                total_mib=40960, cc=SM80)
        self.assertFalse(flags.rig_has_sm86([a100]))

    def test_unknown_arch_presets_take_e5m2_and_say_why(self):
        rig = [{"name": "NVIDIA A100-PCIE-40GB", "total_mib": 40960}] * 2
        profs = {p.kind: p for p in flags.profiles({"num_key_value_heads": 8}, rig)}
        self.assertTrue(profs)
        for kind, p in profs.items():
            self.assertEqual(p.settings["kv_cache_dtype"], "fp8_e5m2", kind)
            self.assertTrue(any("UNKNOWN" in i for i in p.info), (kind, p.info))


class TestHardwareSourcesCarryTheCc(CustomTestCase):
    def test_manual_spec_with_declared_cc(self):
        g = hw.parse_manual_gpu("RTX 3090:24576:sm86", 0)
        self.assertEqual((g.name, g.total_mib, g.cc), ("RTX 3090", 24576, SM86))
        g = hw.parse_manual_gpu("NVIDIA RTX PRO 6000 Blackwell:97887:12.0", 1)
        self.assertEqual(g.cc, SM120)
        self.assertIsNone(hw.parse_manual_gpu("RTX 5090:32607", 0).cc)

    def test_json_spec_with_declared_cc(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rig.json")
            with open(path, "w") as f:
                json.dump({"gpus": [
                    {"name": "NVIDIA RTX A6000", "total_mib": 49140, "cc": "8.6"},
                    {"name": "NVIDIA RTX PRO 6000 Blackwell", "total_mib": 97887,
                     "cc_major": 12, "cc_minor": 0},
                    {"name": "NVIDIA RTX A6000", "total_mib": 49140},
                ]}, f)
            spec = hw.hardware_from_json(path)
        self.assertEqual([g.cc for g in spec.gpus], [SM86, SM120, None])

    def test_live_spec_takes_the_nvml_cc_by_uuid(self):
        cards = [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090", "uuid": "GPU-a",
             "mem_total_mib": 24576, "mem_used_mib": 0},
            {"index": 1, "name": "NVIDIA GeForce RTX 3090", "uuid": "GPU-b",
             "mem_total_mib": 24576, "mem_used_mib": 0},
        ]
        devs = [types.SimpleNamespace(uuid="GPU-b", compute_capability=SM86),
                types.SimpleNamespace(uuid="GPU-zz", compute_capability=SM120)]
        with mock.patch.object(hw, "_load_rig_dashboard_sampler",
                               return_value=lambda: (cards, "pynvml")), \
                mock.patch.object(hw, "_annotate_cuda_indices",
                                  side_effect=lambda g: (g, None)), \
                mock.patch("sglang.srt.registry.nvml.list_devices", return_value=devs):
            spec = hw.hardware_from_nvml()
        # GPU-a has no NVML answer: None, never borrowed from its twin
        self.assertEqual([g.cc for g in spec.gpus], [None, SM86])

    def test_composed_library_cards_carry_the_seed_arch(self):
        spec = cl.compose_rig(["RTX 3090", "RTX 5090", "A100 40GB"])
        self.assertEqual([g.cc for g in spec.gpus], [SM86, SM120, SM80])


class TestCalibrationGateByClass(CustomTestCase):
    def test_reference_rig_matches_in_any_order(self):
        self.assertIsNotNone(flags._match_calibration(REFERENCE, "fp8"))
        self.assertIsNotNone(flags._match_calibration(list(reversed(REFERENCE)), "awq"))

    def test_foreign_inventories_get_nothing(self):
        for rig in (RIG_3X3090, RIG_3X5090, RIG_2X5090_1X3080,
                    RIG_PRO6000_2XA6000, REFERENCE_10GB):
            self.assertIsNone(flags._match_calibration(rig, "fp8"), rig)

    def test_a_different_model_whose_name_contains_5090_is_not_the_class(self):
        rig = [
            _gpu("NVIDIA GeForce RTX 5090 D", 32607, SM120, 0),
            _gpu("NVIDIA GeForce RTX 3080", 20480, SM86, 1),
            _gpu("NVIDIA GeForce RTX 3080", 20480, SM86, 2),
        ]
        self.assertIsNone(flags._match_calibration(rig, "fp8"))

    def test_a_declared_cc_must_agree_with_the_class(self):
        rig = [dict(g) for g in REFERENCE]
        rig[1]["cc"] = SM86  # a "5090" that reports sm86 is not the class
        self.assertIsNone(flags._match_calibration(rig, "fp8"))


class TestUncalibratedCatalogueMiss(CustomTestCase):
    def test_get_names_the_card_key_and_the_measurement(self):
        lib = cl.CardLibrary()
        with self.assertRaises(cl.UncalibratedCard) as cm:
            lib.get("NVIDIA RTX PRO 6000 Blackwell")
        msg = str(cm.exception)
        self.assertIsInstance(cm.exception, KeyError)
        self.assertIn("UNCALIBRATED", msg)
        self.assertIn("RTX PRO 6000 Blackwell/?MiB/sm?", msg)
        self.assertIn("python -m sglang.srt.planner.card_rate_pass --run", msg)
        self.assertEqual(cm.exception.card_key, "RTX PRO 6000 Blackwell/?MiB/sm?")

    def test_resolve_names_the_measured_total(self):
        lib = cl.CardLibrary()
        with self.assertRaises(cl.UncalibratedCard) as cm:
            lib.resolve("NVIDIA A100-PCIE-40GB", total_mib=40960)
        self.assertIn("A100-PCIE-40GB/40960MiB/sm?", str(cm.exception))
        # still a KeyError for every existing caller
        with self.assertRaises(KeyError):
            lib.resolve("NVIDIA A100-PCIE-40GB", total_mib=40960)


class TestPowerLimitRatesOnForeignCards(CustomTestCase):
    def _cards(self, rig, uuids):
        return [
            types.SimpleNamespace(nvml_index=g["index"], uuid=u, name=g["name"],
                                  total_mib=g["total_mib"], cc=g.get("cc"))
            for g, u in zip(rig, uuids)
        ]

    def test_foreign_cards_get_a_named_uncalibrated_miss(self):
        cards = self._cards(RIG_3X3090, ["GPU-x0", "GPU-x1", "GPU-x2"])
        line = P.rate_cut_line(
            None, stage_cards=cards, model_name=P.NF_MODEL,
            is_full_attention=[i % 4 == 3 for i in range(48)],
            pinned=[29, 11, 8], chunk_tokens=16384,
        )
        self.assertIn("keine Raten fuer", line)
        self.assertIn("HW-UNCALIBRATED", line)
        self.assertIn("RTX 3090/24576MiB/sm86", line)
        self.assertNotIn("empfohlen", line)

    def test_the_reference_cards_get_no_uncalibrated_note(self):
        cards = self._cards(
            [REFERENCE[1], REFERENCE[0], REFERENCE[2]],
            [P.RIG_5090[0], P.RIG_3080_NVML0[0], P.RIG_3080_NVML2[0]],
        )
        line = P.rate_cut_line(
            None, stage_cards=cards, model_name=P.NF_MODEL,
            is_full_attention=[i % 4 == 3 for i in range(48)],
            pinned=[29, 11, 8], chunk_tokens=16384,
        )
        self.assertNotIn("HW-UNCALIBRATED", line)


if __name__ == "__main__":
    unittest.main()
