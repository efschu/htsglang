"""27B port 20.10. (merge of the NF line rc28 + rc29 parts): the DUAL-TP3PP3 layout stays as it was.

User 20.10. ~20:35Z: "naja beim dual nvfp4 ist die scheduler änderung ja überflüssig. da es keinen flip gibt, sondern
beide phasen parallel". Pins:
  * the dual pump drains with the 27B P drain pass BYTE FOR BYTE (sha256 of the 27B ``_p_drain_pass`` body @0798c91fcd);
  * the dual SHORT seat path keeps the 27B DECODE-COLLECT (window 15 s unless the env names one) -- byte-equal methods
    under a ``_dual27_`` name; the SOLL v2 set scheduler and its SET-VERDICT line run only in the flip form;
  * no lane reconcile in the dual controller;
  * the new cache/PP paths are off in the dual layout, by the rank-uniform FLLIPER_PDFLIP_DUAL_LAYOUT env:
    S108 turn-end L1->L2 copy, the NF W3-HOST-LEAF spill (the 27B Q-697c spill stays), the W2C chain announcement.
"""
import hashlib
import os
import re
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "python", "flliper", "srt"))
FRONT = os.path.join(_PY, "pdflip", "front.py")
URC = os.path.join(_PY, "mem_cache", "unified_radix_cache.py")

#: sha256 of the body of ``_p_drain_pass`` in python/flliper/srt/pdflip/front.py @0798c91fcd (27B line before the port)
DRAIN_27B_SHA = "72864914a671e306644e8fd4857301fe9a1a99ed565cbebc5f964fcdffa6db51"
_PORT_NOTE = ("            # 27B DUAL-TP3PP3: the dual pump keeps", "            # NF drain below", "            # überflüssig")


def _src(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class DualUnchanged(unittest.TestCase):
    def test_dual_pump_runs_the_27b_drain_pass_byte_for_byte(self):
        lines = _src(FRONT).splitlines(keepends=True)
        d = [k for k, l in enumerate(lines) if l.startswith("        async def _p_drain_pass_dual() -> int:")]
        self.assertEqual(len(d), 1)
        r = d[0]
        while lines[r].rstrip("\n") != "            return passes":
            r += 1
        body = "".join(l for l in lines[d[0] + 1:r + 1] if not l.startswith(_PORT_NOTE))
        self.assertEqual(hashlib.sha256(body.encode()).hexdigest(), DRAIN_27B_SHA)
        s = _src(FRONT)
        self.assertEqual(s.count("self._dual_pump(_p_drain_pass_dual)"), 1)
        self.assertNotIn("self._dual_pump(_p_drain_pass)", s)

    def test_dual_short_path_keeps_the_27b_decode_collect(self):
        s = _src(FRONT)
        for name in ("_dual27_dc_decoding", "_dual27_decode_collect", "_dual27_decode_collect_short"):
            self.assertEqual(len(re.findall(r"def %s\(" % name, s)), 1, name)
        i = s.index("async def _acquire_short_seat_core(")
        core = s[i:s.index("\n    async def ", i + 10)]
        self.assertIn('if getattr(self, "dual_layout", False):', core)
        self.assertIn("self._dual27_decode_collect_short(", core)
        # the SET-VERDICT line of a kurz arrival is the flip form's only
        j = s.index("async def _acquire_short_seat(self")
        self.assertIn('not getattr(self, "dual_layout", False)', s[j:i])

    def test_dual_collect_window_is_the_27b_default(self):
        from unittest import mock

        from flliper.srt.pdflip import front as F

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLLIPER_PDFLIP_DECODE_COLLECT_WINDOW_S", None)
            self.assertEqual(F._dual_decode_collect_window_s(), 15.0)
            self.assertEqual(F._decode_collect_window_s(), 0.0)   # the flip form: SOLL R3/R7, no window
        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_DECODE_COLLECT_WINDOW_S": "0"}):
            self.assertEqual(F._dual_decode_collect_window_s(), 0.0)

    def test_no_lane_reconcile_in_the_dual_controller(self):
        s = _src(FRONT)
        i = s.index('await Front._lane_reconcile(self, "controller")')
        self.assertIn("if not self.dual_layout:", s[i - 200:i])

    def test_s108_turn_end_copy_is_off_in_the_dual_layout(self):
        from flliper.srt.pdflip import d_turn_end_l2 as te

        self.assertTrue(te.turn_end_on({"FLLIPER_PDFLIP_GROUP": "D"}))
        self.assertFalse(te.turn_end_on({"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_PDFLIP_DUAL_LAYOUT": "1"}))
        self.assertFalse(te.turn_end_on({"FLLIPER_PDFLIP_GROUP": "P"}))

    def test_w2c_announcement_is_off_in_the_dual_layout(self):
        from unittest import mock

        from flliper.srt.pdflip import pp_chain_announce as w2c

        sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, pp_size=3), _pdflip_told_paced_on=True,
                                      pdflip_dormant=False)
        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1"}):
            self.assertFalse(w2c._armed_pp0(sched))

    def test_nf_host_leaf_spill_is_off_in_the_dual_layout(self):
        s = _src(URC)
        self.assertIn("if _hos.armed(self) and not _das_hl.armed():", s)
        self.assertIn("released += _das.spill_host_only(", s)   # the 27B Q-697c dual spill stays


if __name__ == "__main__":
    unittest.main()


class DualDefaultsKept(unittest.TestCase):
    def test_arrival_seat_rule_default_off_under_the_dual_layout(self):
        from flliper.srt.pdflip import arrival_seat_rule as asr

        self.assertTrue(asr.enabled({}))                                            # flip form: NF default ON
        self.assertFalse(asr.enabled({"FLLIPER_PDFLIP_DUAL_LAYOUT": "1"}))          # dual: the 27B default OFF
        self.assertTrue(asr.enabled({"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}))
        self.assertFalse(asr.enabled({"FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "0"}))

    def test_front_sets_the_dual_env_for_itself(self):
        s = _src(FRONT)
        i = s.index('front.dual_layout = bool(getattr(args, "dual_layout", False))')
        self.assertIn('os.environ.setdefault("FLLIPER_PDFLIP_DUAL_LAYOUT", "1")', s[i:i + 400])

    def test_no_d_probe_and_no_tok_ahead_in_the_dual_layout(self):
        s = _src(FRONT)
        i = s.index("    def _d_probe_start(")
        self.assertIn('if getattr(self, "dual_layout", False):\n            return None', s[i:i + 1200])
        j = s.index("    def _p_tok_ahead_maybe(")
        self.assertIn('if getattr(self, "dual_layout", False):\n                return False', s[j:j + 1500])

    def test_pgap_instrument_off_in_the_dual_layout(self):
        s = _src(os.path.join(_PY, "pdflip", "launcher.py"))
        self.assertIn('_pgap = bool(getattr(ns, "p_hostgap", False)) and not getattr(ns, "dual_layout", False)', s)

    def test_dual_d_resume_reading_is_the_27b_one(self):
        s = _src(os.path.join(_PY, "pdflip", "d_park_runtime.py"))
        self.assertIn('elif str(os.environ.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "")).strip() == "1":   # rank-uniform env\n'
                      '        avail = sched.uniform_min_avail()', s)
