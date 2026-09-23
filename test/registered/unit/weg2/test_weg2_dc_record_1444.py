# SPDX-License-Identifier: Apache-2.0
"""#1444: group D's dormant VRAM residue is priced from the PREVIOUS boot's
MEASURED record on the same weight form, not from the xsn14 constant.

Why: `DC_MEASURED_D_XCHG_MIB = (2588, 3084, 2588)` was read at boot weg2xsn14
while the weights_draft tag was still RESIDENT on the sleeping cards.  The tag
has travelled with the exchange since; the front measured 1616-1686 MiB (5090)
and 1256-1356 MiB (3080) at boot weg2xsn203.  P's per-card budget paid the
constant anyway -- 1300-1400 MiB per card for bytes that are not there.

Now the front stamps the device residue and the weight form into the
dormant-image record it already appends (`vram_residue_mib`,
`vram_residue_form`), and the launcher reads the newest group-D record back
through ONE selector, `dc_residue_from_record`, before pricing the constant.

DANGER DIRECTIONS this file guards:
* a record from ANOTHER form (serving vs exchange) must NOT be priced;
* a record missing one of this boot's cards must NOT be priced (no partial
  dictionaries, no card silently on the constant while the others are not);
* the measured value is never priced BARE -- the margin rides on it;
* the kill switch prices the constant again;
* the launcher's `dc_expect_d` goes through the selector, the front argv
  carries `--weight-form`, and the front passes the measurement into the
  record (AST pins, so the wiring cannot silently drop out).

Hermetic: no NVML, no boot, no GPU.
"""
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger, launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
S0 = "GPU-0e8cdf1e-9a25-40a1-1ba4-0be6bfa6cdd2"
S2 = "GPU-9b83e1c8-b2c8-4d18-7b0a-4a6b8a7c6f11"
CARDS = [
    SimpleNamespace(uuid=S0, name="NVIDIA GeForce RTX 3080", nvml_index=0),
    SimpleNamespace(uuid=BIG, name="NVIDIA GeForce RTX 5090", nvml_index=1),
    SimpleNamespace(uuid=S2, name="NVIDIA GeForce RTX 3080", nvml_index=2),
]
MEASURED = {S0: 1286, BIG: 1686, S2: 1256}
XCHG = launcher.WEIGHT_SOURCE_EXCHANGE


def _rec(form=XCHG, vals=None, **extra):
    r = {"group": "D", "boot_tag": "weg2xsn203", "at": "2026-09-16T20:36:53Z",
         "vram_residue_form": form,
         "vram_residue_mib": dict(MEASURED if vals is None else vals)}
    r.update(extra)
    return r


class DcResidueFromRecord(CustomTestCase):
    def setUp(self):
        os.environ.pop(launcher.DC_RECORD_ENV, None)

    def test_same_form_prices_measured_plus_margin_per_card(self):
        out, prov = launcher.dc_residue_from_record(_rec(), CARDS, XCHG)
        self.assertIsNotNone(out)
        for c in CARDS:
            self.assertEqual(out[c.uuid], MEASURED[c.uuid] + launcher.DC_RECORD_MARGIN_MIB)
        self.assertIn("weg2xsn203", prov)
        # the record must undercut the stale constant on every card, or the
        # whole point (P's budget) is moot -- pinned against the real triple
        for c in CARDS:
            self.assertLess(out[c.uuid], launcher.dc_measured_d_mib(c, XCHG))

    def test_other_form_is_refused(self):
        out, prov = launcher.dc_residue_from_record(_rec(form="serving"), CARDS, XCHG)
        self.assertIsNone(out)
        self.assertIn("form", prov)

    def test_missing_card_refuses_whole_record(self):
        vals = dict(MEASURED)
        del vals[S2]
        out, prov = launcher.dc_residue_from_record(_rec(vals=vals), CARDS, XCHG)
        self.assertIsNone(out)
        self.assertIn(S2, prov)

    def test_zero_or_bogus_value_refuses(self):
        for bad in (0, -5, "1600", True, None):
            vals = dict(MEASURED)
            vals[BIG] = bad
            out, _ = launcher.dc_residue_from_record(_rec(vals=vals), CARDS, XCHG)
            self.assertIsNone(out, f"value {bad!r} must not be priced")

    def test_no_record_and_kill_switch(self):
        self.assertIsNone(launcher.dc_residue_from_record(None, CARDS, XCHG)[0])
        self.assertIsNone(launcher.dc_residue_from_record({}, CARDS, XCHG)[0])
        os.environ[launcher.DC_RECORD_ENV] = "0"
        try:
            out, prov = launcher.dc_residue_from_record(_rec(), CARDS, XCHG)
        finally:
            os.environ.pop(launcher.DC_RECORD_ENV, None)
        self.assertIsNone(out)
        self.assertIn(launcher.DC_RECORD_ENV, prov)

    def test_margin_is_real(self):
        self.assertGreaterEqual(launcher.DC_RECORD_MARGIN_MIB, 2 * 100)


class DormantImageRecordCarriesResidue(CustomTestCase):
    def _sample(self, **kw):
        return host_ledger.dormant_image_sample(
            group="D", shmem_before_bytes=None, shmem_after_bytes=None, pids=[],
            weight_tags_gib=27.15, interleaved=True, boot_tag="t", commit="c", **kw)

    def test_fields_present_and_typed(self):
        rec = self._sample(vram_residue_mib={BIG: 1686.0, S0: 1286}, vram_residue_form=XCHG)
        self.assertEqual(rec["vram_residue_mib"], {BIG: 1686, S0: 1286})
        self.assertEqual(rec["vram_residue_form"], XCHG)
        # and the record round-trips through the selector
        out, _ = launcher.dc_residue_from_record(rec, CARDS[:2], XCHG)
        self.assertEqual(out[BIG], 1686 + launcher.DC_RECORD_MARGIN_MIB)

    def test_absent_measurement_is_empty_not_missing(self):
        rec = self._sample()
        self.assertEqual(rec["vram_residue_mib"], {})
        self.assertEqual(rec["vram_residue_form"], "")
        self.assertIsNone(launcher.dc_residue_from_record(rec, CARDS, XCHG)[0])


class Wiring(CustomTestCase):
    def test_launcher_and_front_wiring(self):
        import inspect
        from sglang.srt.weg2 import front
        lsrc = inspect.getsource(launcher)
        fsrc = inspect.getsource(front)
        self.assertIn("dc_residue_from_record(\n        _dc_rec_d, cards, ns.weg2_weight_source)", lsrc)
        self.assertIn('"--weight-form", str(ns.weg2_weight_source)', lsrc)
        self.assertIn("self.sample_dormant_image(src, shmem_before, vram_residue_mib=dc)", fsrc)
        self.assertIn("vram_residue_form=self.weight_form", fsrc)
        self.assertIn('ap.add_argument("--weight-form"', fsrc)


if __name__ == "__main__":
    unittest.main()
