"""UNIFY S7: the front's ``--record-line`` (27B 76e87ac3b2) in the S3 form -- the
spec parses into the form's CalibrationIdentity, the sleep-leg gate's SidecarView
reads the sidecar through its ``accepts_sample``; empty = every sample (NF form).
The launcher emits it only where the profile's records row carries the ``line``
term (qwen27b). Hermetic: no GPU, no boot.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import launcher as L
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _form(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile != "nextflash"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return F.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                      flip="family", vision="off", profile=profile, model="m")


class TestRecordLine(CustomTestCase):
    def setUp(self):
        self._saved = os.environ.get(F.FORM_ENV)

    def tearDown(self):
        os.environ.pop(F.FORM_ENV, None)
        if self._saved is not None:
            os.environ[F.FORM_ENV] = self._saved

    def test_empty_spec_is_no_filter(self):
        self.assertIsNone(front_mod.record_identity_from_spec(""))
        self.assertIsNone(front_mod.SidecarView().accept)

    def test_spec_round_trip_and_line_term(self):
        os.environ[F.FORM_ENV] = _form("qwen27b").env_value()
        ns = argparse.Namespace(model="/m/Qwen3.8-27B-INT8", tree="/t")
        spec = L.record_line_spec(ns)
        ident = front_mod.record_identity_from_spec(spec)
        self.assertIsNotNone(ident)
        self.assertTrue(ident.uses_line)
        self.assertEqual(ident.repo, os.path.abspath("/t"))

    def test_sidecar_view_filters_through_accept(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rec.json")
            samples = [{"group": "D", "boot_tag": "a", "at": "1", "rss_shmem_gib": 1.0, "shmem_delta_gib": 1.0},
                       {"group": "D", "boot_tag": "b", "at": "2", "rss_shmem_gib": 2.0, "shmem_delta_gib": 2.0}]
            with open(path, "w") as f:
                json.dump({"samples": samples}, f)
            v = front_mod.SidecarView(accept=lambda s: s.get("boot_tag") == "a")
            rec = v.read(path)
            self.assertEqual(((rec or {}).get("D") or {}).get("boot_tag"), "a")
            plain = front_mod.SidecarView().read(path)
            self.assertEqual(((plain or {}).get("D") or {}).get("boot_tag"), "b")


if __name__ == "__main__":
    unittest.main()
