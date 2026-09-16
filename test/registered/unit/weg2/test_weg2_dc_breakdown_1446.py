# SPDX-License-Identifier: Apache-2.0
"""#1446: the sleeping group's VRAM residue, attributed on the rank.

The front measures ONE number per card (WEG2-DC); this splits it into what
torch's allocator holds, what the saver holds RESIDENT (mapped tags), and the
remainder no allocator owns (context, driver, communicator, non-torch), and
prints the saver's PAUSED bytes beside it as the part that is unmapped and
therefore NOT in the NVML figure.

Danger directions: a paused tag must not be counted as resident (else 'other'
goes negative and the floor reads too small); a reading that could not be
taken must print n/a, never 0; the release handler must actually call it.
"""
import inspect
import os
import re
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_memory_saver as ms
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1024 * 1024


class Arithmetic(CustomTestCase):
    def test_split_resident_paused_other(self):
        rec = ms.dc_breakdown(
            nvml_proc_bytes=1686 * MIB, torch_reserved=300 * MIB, torch_allocated=250 * MIB,
            tag_bytes={"kv_cache": 6000 * MIB, "cuda_graph": 458 * MIB, "weights": 200 * MIB,
                       "weights_1": 0},
            offload_tags=["kv_cache", "cuda_graph"],
        )
        self.assertEqual(rec["tms_resident_mib"], 200)
        self.assertEqual(rec["resident_tags_mib"], {"weights": 200})
        self.assertEqual(rec["tms_paused_mib"], 6458)
        self.assertEqual(rec["paused_tags_mib"], {"cuda_graph": 458, "kv_cache": 6000})
        # torch keeps resident AND paused regions reserved: untagged = 300 - 200 - 6458 -> 0
        self.assertEqual(rec["torch_untagged_mib"], 0)
        self.assertEqual(rec["other_mib"], 1686 - 200 - 0)
        self.assertEqual(rec["nvml_proc_mib"], 1686)

    def test_paused_tag_is_not_resident(self):
        a = ms.dc_breakdown(nvml_proc_bytes=1000 * MIB, torch_reserved=0, torch_allocated=0,
                            tag_bytes={"kv_cache": 5000 * MIB}, offload_tags=[])
        b = ms.dc_breakdown(nvml_proc_bytes=1000 * MIB, torch_reserved=0, torch_allocated=0,
                            tag_bytes={"kv_cache": 5000 * MIB}, offload_tags=["kv_cache"])
        self.assertEqual(a["other_mib"], -4000)  # an impossible reading shows as such
        self.assertEqual(b["other_mib"], 1000)

    def test_xsn206_tp0_after_every_leg(self):
        # MEASURED boot weg2xsn206 TP0: nvml 1640, reserved 28940, everything paused 28146
        rec = ms.dc_breakdown(nvml_proc_bytes=1640 * MIB, torch_reserved=28940 * MIB,
                              torch_allocated=27218 * MIB,
                              tag_bytes={"kv_cache": 9642 * MIB, "cuda_graph": 458 * MIB,
                                         "weights": 18046 * MIB},
                              offload_tags=["kv_cache", "cuda_graph", "weights"])
        self.assertEqual(rec["torch_untagged_mib"], 28940 - 28146)   # 794
        self.assertEqual(rec["other_mib"], 1640 - 794)                # 846

    def test_unreadable_is_none_not_zero(self):
        rec = ms.dc_breakdown(nvml_proc_bytes=None, torch_reserved=None, torch_allocated=None,
                              tag_bytes={}, offload_tags=None)
        self.assertIsNone(rec["nvml_proc_mib"])
        self.assertIsNone(rec["other_mib"])
        line = ms.format_dc_breakdown(rec, stage="release tags=['kv_cache']")
        self.assertIn("nvml_proc=n/a", line)
        self.assertIn("other n/a", line)
        self.assertIn("torch_untagged n/a", line)
        self.assertTrue(line.startswith("WEG2-DC-BREAKDOWN stage=release tags=['kv_cache']"))

    def test_line_carries_every_term(self):
        rec = ms.dc_breakdown(nvml_proc_bytes=1600 * MIB, torch_reserved=100 * MIB,
                              torch_allocated=90 * MIB, tag_bytes={"weights": 500 * MIB},
                              offload_tags=[])
        line = ms.format_dc_breakdown(rec, stage="s")
        for needle in ("nvml_proc=1600 MiB", "torch_reserved 100", "allocated 90",
                       "tms_resident 500", "torch_untagged 0", "other 1100", "tms_paused 0"):
            self.assertIn(needle, line)


class Wiring(CustomTestCase):
    def test_release_handler_prints_it(self):
        from sglang.srt.managers.scheduler_components import weight_updater as wu
        src = inspect.getsource(wu)
        self.assertIn('self._weg2_log_dc_breakdown("release tags=%s" % (list(tags),))', src)
        self.assertIn("def _weg2_log_dc_breakdown(self, stage: str)", src)
        self.assertIn("def _weg2_nvml_self_bytes(self)", src)
        # the instrument is fail-soft and prints n/a, never a fabricated 0
        body = inspect.getsource(wu.WeightUpdater._weg2_log_dc_breakdown) if hasattr(wu, "WeightUpdater") else src
        self.assertIn("n/a (instrument failed)", body)
        # #1452b: every attribute the instrument writes on the slots=True
        # manager is a declared field (the #1437b lesson, paid again at xsn208)
        fields = wu.SchedulerWeightUpdaterManager.__dataclass_fields__
        self.assertIn("_1452_snapshots", fields)
        snap = inspect.getsource(wu.SchedulerWeightUpdaterManager._weg2_dump_dc_snapshot)
        for attr in re.findall(r"self\.(_[0-9a-z_]+) =", snap):
            self.assertIn(attr, fields, attr)


if __name__ == "__main__":
    unittest.main()
