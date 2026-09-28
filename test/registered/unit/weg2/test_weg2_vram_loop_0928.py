"""VRAM loop 28.09. (user: "VRAM liegt brach ODER OOM -- messen statt raten").

P1b: the D row planner's overhead calibration names an expert-offload D
(NF Form A) as NOT applicable instead of refusing it as a "cell mismatch"
-- that planner books every expert resident, and NF's D budget is the
MEASURED-D budget line + FRACTION-SOLVE D, which read NF's own records.
"""

import os
import tempfile
import unittest

from sglang.srt.weg2 import launcher as L


class _Line:
    def accepts_log(self, path):
        return True


def _write_boot(ev, argv_tail, kv_lines):
    stem = os.path.join(ev, "boot_weg2_nft_abcdef0123_0928_175259")
    with open(stem + ".front.log", "w") as f:
        f.write("[t] WEG2-LAUNCH group D argv: /v/python -m sglang.launch_server "
                "--model-path /m " + argv_tail + "\n")
    with open(stem + ".D.log", "w") as f:
        for line in kv_lines:
            f.write(line + "\n")


# NF rc12z26 17:52 D.log: TP0 cell 14143, the two Form A workers cell 0
_NF_KV = [
    "[2026-09-28 17:56:51 TP1] KV pool sizing: available_bytes=1587544064 (1.479 GiB), cell_size=0, page_size=64 -> max_total_num_tokens=1048576",
    "[2026-09-28 17:56:51 TP2] KV pool sizing: available_bytes=595591168 (0.555 GiB), cell_size=0, page_size=64 -> max_total_num_tokens=1048576",
    "[2026-09-28 17:56:51 TP0] KV pool sizing: available_bytes=8013045760 (7.463 GiB), cell_size=14143, page_size=64 -> max_total_num_tokens=566528",
]


class TestExpertOffloadCalibration(unittest.TestCase):
    def test_expert_offload_d_is_named_not_applicable(self):
        with tempfile.TemporaryDirectory() as ev:
            _write_boot(ev, "--rank-gpu-memory-mib 26400,17728,17936 --rank-tp-ratio 1,0,0 "
                            "--max-running-requests 6 --rank-moe-resident-fraction 0.06,0.51,0.48",
                        _NF_KV)
            ovh, why = L.d_overhead_calibration("/m", _Line(), evidence_dir=ev)
        self.assertIsNone(ovh)
        self.assertIn("expert-offload D", why)
        self.assertIn("FRACTION-SOLVE D", why)
        self.assertNotIn("would not transfer", why)



import types as _types


def _plan(ceil, solved=(), owner=()):
    fits = tuple(_types.SimpleNamespace(ceiling_fraction=c) for c in ceil)
    return _types.SimpleNamespace(fits=fits, solved_fractions=tuple(solved),
                                  solved_owner_ratio=tuple(owner))


_GIVEN = [0.06, 0.51, 0.48]
_CEIL = [0.062, 0.586, 0.491]   # NF rc12z26 17:52 front.log FRACTION-SOLVE D
_P0 = {L.TORCH_CACHE_CAP_ENV: "1"}


class TestFrCeilingAdopt(unittest.TestCase):
    def test_edge_wins_with_p0(self):
        new, lines = L.d_fr_ceiling_adopt(_plan(_CEIL), _GIVEN, _P0, owned=False)
        self.assertEqual(new, _CEIL)
        self.assertEqual(len([ln for ln in lines if "rang" in ln]), 3)

    def test_without_p0_the_stated_fr_stays_and_is_named(self):
        new, lines = L.d_fr_ceiling_adopt(_plan(_CEIL), _GIVEN, {}, owned=False)
        self.assertIsNone(new)
        self.assertTrue(any("P0 aus" in ln for ln in lines))

    def test_owned_cut_is_untouched(self):
        new, lines = L.d_fr_ceiling_adopt(_plan(_CEIL, owner=(215, 113, 160)), _GIVEN, _P0,
                                          owned=True)
        self.assertIsNone(new)
        self.assertEqual(lines, [])

    def test_s2b_solved_fr_is_untouched(self):
        new, lines = L.d_fr_ceiling_adopt(_plan(_CEIL, solved=(0.1, 0.5, 0.5)), _GIVEN, _P0,
                                          owned=False)
        self.assertIsNone(new)
        self.assertEqual(lines, [])

    def test_named_cap_binds(self):
        env = dict(_P0, **{L.FR_D_CAP_ENV: "0.07,0.55,0.60"})
        new, _ = L.d_fr_ceiling_adopt(_plan(_CEIL), _GIVEN, env, owned=False)
        self.assertEqual(new, [0.062, 0.55, 0.491])

    def test_raise_only_and_switch_off(self):
        new, lines = L.d_fr_ceiling_adopt(_plan([0.05, 0.586, 0.491]), _GIVEN, _P0, owned=False)
        self.assertEqual(new, [0.06, 0.586, 0.491])
        self.assertTrue(any("Decke unter gegeben" in ln for ln in lines))
        off = dict(_P0, **{L.FR_D_CEILING_ENV: "0"})
        self.assertIsNone(L.d_fr_ceiling_adopt(_plan(_CEIL), _GIVEN, off, owned=False)[0])

    def test_no_edge_on_a_rank_keeps_the_stated_fr(self):
        new, _ = L.d_fr_ceiling_adopt(_plan([0.062, None, 0.491]), _GIVEN, _P0, owned=False)
        self.assertIsNone(new)


if __name__ == "__main__":
    unittest.main()
