"""NF-Bootzeit: the O_DIRECT stream's anon instrument must not bound the load.

MEASURED (rc12z15 dkrnfh91dprsavisodirectbar1dauer09281011, P.log line 1283):
PP0 ``WEG2 LOAD-PROFILE 1192 samples over 73.4 s -- weight_utils.py:1586
_rss_anon_bytes 46.8%; weight_utils.py:1587 _rss_anon_bytes 22.4%`` -- one
/proc/self/status read per tensor over 135 032 NF tensors (median 25.6 KB),
the stream ran at 0.59 GB/s while the same reader does 2.36 GB/s without it.

RED on 4a08bcdb22 (H2): 2000 tensors -> 2000+ reads. GREEN: time-sampled
(<= every 50 ms) plus one forced sample at the end, so the peak line stays.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from sglang.srt.model_loader import weight_utils as W


class TestAnonSampling(unittest.TestCase):
    def test_anon_is_not_read_per_tensor(self):
        d = tempfile.mkdtemp(prefix="bootzeit_anon_")
        p = os.path.join(d, "m.safetensors")
        save_file({f"t{i:05d}": torch.full((16,), i, dtype=torch.int32) for i in range(2000)}, p)
        calls = {"n": 0}
        real = W._rss_anon_bytes

        def counting():
            calls["n"] += 1
            return real()

        with mock.patch.object(W, "_rss_anon_bytes", counting):
            st = W.StreamStats(1 << 20, 2)
            got = sum(1 for _ in W.pread_safetensors_stream([p], None, workers=2, stats=st, log=False))
        self.assertEqual(got, 2000)
        self.assertLess(calls["n"], 200, f"{calls['n']} /proc reads for 2000 tensors")
        self.assertGreaterEqual(calls["n"], 2)  # start + the forced final sample
        self.assertGreaterEqual(st.anon_peak, st.anon_start)


if __name__ == "__main__":
    unittest.main()
