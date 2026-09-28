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


if __name__ == "__main__":
    unittest.main()
