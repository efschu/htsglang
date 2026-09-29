"""27B review 29.09.: the PP-cut calibration's per-rank residual must not charge
another process's CUDA context to this rank.

``nvml_used_mib`` in the residency census is card-wide (``mem_get_info``). A
census taken while group D's contexts already sat on the card (early D start),
or beside a probe or co-tenant, derived ``residual = used - params - pools``
with those foreign bytes inside, and the cut gate then booked P too full --
VRAM that belongs to P's resident experts.

The census now writes NVML's per-process split (``nvml_process_used_mib``,
``nvml_foreign_mib``); the calibration subtracts the FOREIGN part and names the
source per rank. The driver carve is not foreign and stays in the residual.
Hermetic: NVML is faked.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.planner import pp_cut_calibration as cal
from sglang.srt.planner import residency_census as rc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MIB = 1 << 20


def _census(pp_rank, n_attn, n_linear, **kw):
    blob = {
        "pp_rank": pp_rank,
        "n_attn_layers": n_attn,
        "n_linear_layers": n_linear,
        "params_mib": {"layers_attention": 300.0 * n_attn,
                       "layers_linear": 300.0 * n_linear, "visual": 0.0},
        "pools_mib": 1000.0,
        "nvml_used_mib": 16000.0,
        "nvml_total_mib": 20480.0,
    }
    if pp_rank == 0:
        blob["params_mib"]["embed_tokens"] = 2000.0
    if pp_rank == 2:
        blob["params_mib"]["lm_head"] = 2000.0
    blob.update(kw)
    return blob


def _load(blobs):
    with tempfile.TemporaryDirectory() as d:
        for b in blobs:
            with open(os.path.join(d, f"census_pp{b['pp_rank']}.json"), "w") as fh:
                json.dump(b, fh)
        return cal.load_census_calibration(d)


class ForeignContextIsNotResidual(CustomTestCase):
    def test_foreign_bytes_are_subtracted_per_rank(self):
        # rank 1's card also held a D context of 480 MiB (early D start)
        c = _load([_census(0, 4, 4, nvml_foreign_mib=0.0),
                   _census(1, 4, 4, nvml_foreign_mib=480.0),
                   _census(2, 4, 4, nvml_foreign_mib=35.0)])
        params = 300.0 * 8
        self.assertAlmostEqual(c.residual_mib[0], 16000 - 2000 - params - 1000)
        self.assertAlmostEqual(c.residual_mib[1], 16000 - 480 - params - 1000)
        self.assertAlmostEqual(c.residual_mib[2], 16000 - 35 - 2000 - params - 1000)
        self.assertIn("per-process (card used 16000 - foreign 480)", c.residual_source[1])
        self.assertIn("foreign 480", c.describe())

    def test_old_census_is_card_wide_and_says_so(self):
        c = _load([_census(0, 4, 4), _census(1, 4, 4), _census(2, 4, 4)])
        self.assertAlmostEqual(c.residual_mib[1], 16000 - 300.0 * 8 - 1000)
        self.assertTrue(all(s.startswith("card-wide") for s in c.residual_source))

    def test_unmeasured_split_is_not_zero(self):
        c = _load([_census(0, 4, 4, nvml_foreign_mib=None),
                   _census(1, 4, 4), _census(2, 4, 4)])
        self.assertTrue(c.residual_source[0].startswith("card-wide"))


class _Proc(SimpleNamespace):
    pass


class _FakeNvml:
    def __init__(self, procs):
        self._procs = procs
        self.asked = None

    def nvmlDeviceGetHandleByUUID(self, uuid):
        self.asked = uuid
        return "h"

    def nvmlDeviceGetComputeRunningProcesses_v3(self, handle):
        return self._procs


class CensusSplit(CustomTestCase):
    def _split(self, procs, uuid="5c648f96-be1d", pid=100):
        fake = _FakeNvml(procs)

        class _Sess:
            def __enter__(self_inner):
                return fake

            def __exit__(self_inner, *a):
                return False

        with mock.patch("sglang.srt.registry.nvml.nvml_session", lambda: _Sess()):
            out = rc.card_process_split_mib(uuid, pid=pid)
        return out, fake

    def test_own_and_foreign(self):
        (own, foreign), fake = self._split([
            _Proc(pid=100, usedGpuMemory=12000 * MIB),
            _Proc(pid=200, usedGpuMemory=400 * MIB),
            _Proc(pid=201, usedGpuMemory=80 * MIB)])
        self.assertEqual((own, foreign), (12000.0, 480.0))
        self.assertEqual(fake.asked, "GPU-5c648f96-be1d")

    def test_unknown_usage_is_none_not_zero(self):
        (own, foreign), _ = self._split([_Proc(pid=100, usedGpuMemory=None)])
        self.assertEqual((own, foreign), (None, None))

    def test_no_uuid_is_none(self):
        self.assertEqual(rc.card_process_split_mib(None), (None, None))


if __name__ == "__main__":
    unittest.main()
